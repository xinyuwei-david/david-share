#!/usr/bin/env python3
r"""Measure the explicit session lifecycle of a Foundry hosted agent (Path B).

One run: create a session, write a marker file into it, ask a first question and a follow-up,
stop the session, send the next turn with the same agent_session_id (the platform provisions
new compute and restores $HOME), read the marker back, ask once more, then delete the session.
With --resume-after idle the explicit stop is replaced by --idle-wait seconds without any request to
the session (the platform's idle timeout, 15 min by default), one status read, then the same resume turn.

    python tools/measure_sessions.py \
        --project-endpoint https://<account>.services.ai.azure.com/api/projects/<project> \
        --agent <hosted-agent> --version <n> --runs 3 \
        --out <private-folder>/sessions-raw.json \
        --public-out evidence/runs/<folder>/sessions.json

Authentication: an Entra token for https://ai.azure.com from `az account get-access-token`, or the
header in SESSIONS_AUTH_HEADER ("Header-Name: value", for example an API Management key in front
of the project). The header value is never printed or written.

Every HTTP attempt is recorded, including retries after 408, 429, 5xx or a network error, so a
service-side incident stays visible instead of disappearing into a median. The public output
replaces session IDs with neutral labels, drops response IDs and the endpoint, and records the
SHA-256 of the private output. API: https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions

Exit codes: 0 every run completed, the file written before stop came back unchanged after resume,
the question count kept growing and every session this run created was deleted; 1 otherwise.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

API_VERSION = "v1"
FIRST = "Where is Bangkok? Answer in one short sentence."
COUNT = "How many questions have I asked before this one? Number only."
MARKER_PATH = "state-marker.txt"
RETRY_STATUS = {0, 408, 429, 500, 502, 503, 504}
TURNS = ("first", "warm", "resume", "warm_after_resume")
CALLS = ("create", "upload_marker", "first", "warm", "stop", "resume", "marker_after_resume",
         "warm_after_resume", "delete")
CLOCKS = {
    "create_to_active_s": "POST .../endpoint/sessions until the session reads active (a successful create returns "
                          "active directly); includes failed attempts and 2 s pauses",
    "<turn>_total_s": "first attempt of POST .../protocols/openai/responses until its answer, including failed "
                      "attempts and 2 s pauses",
    "<turn>_success_s": "the successful HTTP attempt alone",
    "<turn>_server_s": "response completed_at minus created_at as recorded by the service (whole seconds)",
    "stop_to_idle_s": "POST .../sessions/{id}:stop until GET .../sessions/{id} reads idle",
    "idle_wait": "--resume-after idle: no request to the session for the wait, then one GET .../sessions/{id}",
    "resume": "the next /responses call on the stopped or idle session; the platform provisions compute and restores "
              "$HOME",
    "marker": "file written with PUT .../files/content before stop and read with GET after resume",
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auth_header() -> tuple[str, str]:
    raw = os.environ.get("SESSIONS_AUTH_HEADER", "").strip()
    if raw:
        name, sep, value = raw.partition(":")
        if not sep or not name.strip() or not value.strip():
            raise SystemExit("SESSIONS_AUTH_HEADER must look like 'Header-Name: value'")
        return name.strip(), value.strip()
    az = "az.cmd" if os.name == "nt" else "az"
    tok = subprocess.run([az, "account", "get-access-token", "--resource", "https://ai.azure.com",
                          "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True)
    if tok.returncode != 0 or not tok.stdout.strip():
        raise SystemExit("no Entra token for https://ai.azure.com: run az login, or set SESSIONS_AUTH_HEADER")
    return "Authorization", "Bearer " + tok.stdout.strip()


def as_json(payload: bytes):
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def output_text(body) -> str | None:
    if not isinstance(body, dict):
        return None
    parts = [c["text"] for item in body.get("output") or [] if isinstance(item, dict)
             for c in item.get("content") or [] if isinstance(c, dict) and isinstance(c.get("text"), str)]
    return "".join(parts).strip() or None


def dig(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


class Client:
    def __init__(self, base: str, header: tuple[str, str], attempts: int, timeout: float):
        self.base = base.rstrip("/")
        self.header = header
        self.attempts = max(1, attempts)
        self.timeout = timeout
        self.log: list[dict] = []

    def call(self, op: str, method: str, path: str, *, body=None, raw: bytes | None = None):
        url = f"{self.base}{path}{'&' if '?' in path else '?'}api-version={API_VERSION}"
        tries: list[dict] = []
        status, payload, start = 0, b"", time.perf_counter()
        for n in range(1, self.attempts + 1):
            headers = {self.header[0]: self.header[1]}
            data = None
            if raw is not None:
                data, headers["Content-Type"] = raw, "application/octet-stream"
            elif body is not None:
                data, headers["Content-Type"] = json.dumps(body).encode("utf-8"), "application/json"
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status, payload = resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                status, payload = exc.code, exc.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                status, payload = 0, str(exc).encode("utf-8")
            attempt = {"status": status, "seconds": round(time.perf_counter() - t0, 3)}
            tries.append(attempt)
            self.log.append({"op": op, **attempt})
            if status not in RETRY_STATUS or n == self.attempts:
                break
            time.sleep(2)
        ok = 200 <= status < 300
        rec = {"status": status, "attempts": tries, "total_s": round(time.perf_counter() - start, 3),
               "success_s": tries[-1]["seconds"] if ok else None}
        if not ok:
            rec["error"] = payload[:400].decode("utf-8", "replace")
        return rec, payload


def wait_status(client: Client, agent: str, sid: str, wanted: set[str], limit_s: float = 180.0) -> dict:
    start, polls, last = time.perf_counter(), 0, None
    while True:
        rec, payload = client.call("get_session", "GET", f"/agents/{agent}/endpoint/sessions/{sid}")
        polls += 1
        if rec["status"] == 200:
            last = (as_json(payload) or {}).get("status")
        if last in wanted or time.perf_counter() - start >= limit_s:
            break
        time.sleep(0.5)
    return {"state": last, "polls": polls, "seconds": round(time.perf_counter() - start, 3)}


def turn(client: Client, agent: str, sid: str, text: str, previous: str | None) -> dict:
    body = {"input": text, "stream": False, "agent_session_id": sid}
    if previous:
        body["previous_response_id"] = previous
    rec, payload = client.call("responses", "POST", f"/agents/{agent}/endpoint/protocols/openai/responses", body=body)
    out = {**rec, "input": text}
    data = as_json(payload)
    if 200 <= rec["status"] < 300 and isinstance(data, dict):
        out["response"] = output_text(data)
        out["response_id"] = data.get("id")
        out["same_session"] = data.get("agent_session_id") == sid
        created, completed = data.get("created_at"), data.get("completed_at")
        if isinstance(created, (int, float)) and isinstance(completed, (int, float)):
            out["server_s"] = round(completed - created, 3)
    return out


def say(run: int, step: str, rec: dict, extra: str = "") -> None:
    codes = "/".join(str(t["status"]) for t in rec.get("attempts", []))
    print(f"run {run} {step:<19} http={rec.get('status')} total={rec.get('total_s')}s tries={codes} {extra}".rstrip(),
          flush=True)


def run_calls(r: dict) -> tuple[str, ...]:
    return tuple(k for k in CALLS if k != "stop") if "idle" in r else CALLS


def one_run(client: Client, agent: str, version: str, index: int, resume_after: str = "stop",
            idle_wait: float = 0.0) -> dict:
    r: dict = {"run": index, "started_at": utc()}
    sid = None
    try:
        rec, payload = client.call("create_session", "POST", f"/agents/{agent}/endpoint/sessions",
                                   body={"version_indicator": {"type": "version_ref", "agent_version": version}})
        body = as_json(payload) or {}
        sid = body.get("agent_session_id") if 200 <= rec["status"] < 300 else None
        r["create"] = {**rec, "state": body.get("status")}
        if sid and body.get("status") != "active":
            r["create"]["wait_active"] = wait_status(client, agent, sid, {"active"})
        say(index, "create", rec, f"state={body.get('status')}")
        if not sid:
            raise RuntimeError(f"create_session returned HTTP {rec['status']}")
        r["session"] = sid
        marker = f"MARKER-{index}-{int(time.time())}"
        rec, _ = client.call("upload_file", "PUT",
                             f"/agents/{agent}/endpoint/sessions/{sid}/files/content?path={MARKER_PATH}",
                             raw=marker.encode("utf-8"))
        r["upload_marker"] = rec
        say(index, "upload_marker", rec)
        r["first"] = turn(client, agent, sid, FIRST, None)
        say(index, "first", r["first"], f"server={r['first'].get('server_s')}s answer={r['first'].get('response')!r}")
        r["warm"] = turn(client, agent, sid, COUNT, r["first"].get("response_id"))
        say(index, "warm", r["warm"], f"server={r['warm'].get('server_s')}s answer={r['warm'].get('response')!r}")
        if resume_after == "idle":
            t_wait = time.perf_counter()
            time.sleep(idle_wait)
            state = wait_status(client, agent, sid, {"idle", "stopped"}, limit_s=0)["state"]
            r["idle"] = {"waited_s": round(time.perf_counter() - t_wait, 3), "state_before_resume": state}
            print(f"run {index} idle               waited={r['idle']['waited_s']}s state={state}", flush=True)
        else:
            rec, _ = client.call("stop_session", "POST", f"/agents/{agent}/endpoint/sessions/{sid}:stop", body={})
            r["stop"] = {**rec, "wait_stopped": wait_status(client, agent, sid, {"idle", "stopped"})}
            say(index, "stop", rec, f"state={r['stop']['wait_stopped']['state']} after "
                                    f"{r['stop']['wait_stopped']['seconds']}s")
        r["resume"] = turn(client, agent, sid, COUNT, r["warm"].get("response_id"))
        say(index, "resume", r["resume"],
            f"server={r['resume'].get('server_s')}s answer={r['resume'].get('response')!r}")
        r["state_after_resume"] = wait_status(client, agent, sid, {"active"}, limit_s=0)["state"]
        rec, payload = client.call("download_file", "GET",
                                   f"/agents/{agent}/endpoint/sessions/{sid}/files/content?path={MARKER_PATH}")
        matches = payload.decode("utf-8", "replace").strip() == marker if 200 <= rec["status"] < 300 else None
        r["marker_after_resume"] = {**rec, "matches": matches}
        say(index, "marker_after_resume", rec, f"matches={matches} state={r['state_after_resume']}")
        r["warm_after_resume"] = turn(client, agent, sid, COUNT, r["resume"].get("response_id"))
        say(index, "warm_after_resume", r["warm_after_resume"],
            f"server={r['warm_after_resume'].get('server_s')}s answer={r['warm_after_resume'].get('response')!r}")
    except Exception as exc:  # noqa: BLE001 - recorded; cleanup still runs
        r["error"] = f"{type(exc).__name__}: {exc}"
        print(f"run {index} error {r['error']}", flush=True)
    finally:
        if sid:
            rec, _ = client.call("delete_session", "DELETE", f"/agents/{agent}/endpoint/sessions/{sid}")
            r["delete"] = rec
            r["deleted"] = 200 <= rec["status"] < 300
            say(index, "delete", rec)
        r["finished_at"] = utc()
    return r


def sweep(client: Client, agent: str, version: str, since_epoch: float, deleted: set) -> dict:
    """Delete sessions this run created but never got an ID for (a create that timed out on the client)."""
    rec, payload = client.call("list_sessions", "GET", f"/agents/{agent}/endpoint/sessions")
    extra, removed = [], 0
    for s in (as_json(payload) or {}).get("data") or []:
        sid = s.get("agent_session_id")
        created = s.get("created_at")
        if sid in deleted or not isinstance(created, (int, float)):
            continue
        if created >= since_epoch - 5 and dig(s, "version_indicator", "agent_version") == version:
            d, _ = client.call("delete_session", "DELETE", f"/agents/{agent}/endpoint/sessions/{sid}")
            extra.append(sid)
            removed += 200 <= d["status"] < 300
    return {"list_status": rec["status"], "extra_sessions": len(extra), "extra_deleted": removed, "ids": extra}


def stat(values) -> dict:
    v = [round(x, 3) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
    out: dict = {"n": len(v), "values": v}
    if v:
        out.update(median=round(statistics.median(v), 3), min=min(v), max=max(v))
    return out


def ok(r: dict, key: str) -> bool:
    status = dig(r, key, "status")
    return isinstance(status, int) and 200 <= status < 300


def count_grew(r: dict) -> bool:
    counts = [dig(r, t, "response") for t in ("warm", "resume", "warm_after_resume")]
    return (all(isinstance(x, str) and x.strip().isdigit() for x in counts)
            and int(counts[0]) < int(counts[1]) < int(counts[2]))


def passed(record: dict) -> bool:
    runs, c = record["runs"], record["cleanup"]
    return (bool(runs) and record["summary"]["runs_complete"]["count"] == len(runs)
            and all(r.get("deleted") is True and dig(r, "marker_after_resume", "matches") is True and count_grew(r)
                    for r in runs)
            and all(dig(r, "idle", "state_before_resume") in ("idle", "stopped") for r in runs if "idle" in r)
            and c["list_status"] == 200 and c["extra_deleted"] == c["extra_sessions"])


def summarize(runs: list[dict], log: list[dict]) -> dict:
    s: dict = {
        "runs_complete": {"count": sum("error" not in r and all(ok(r, k) for k in run_calls(r)) for r in runs),
                          "n": len(runs)},
        "idle_waited_s": stat([dig(r, "idle", "waited_s") for r in runs]),
        "idle_state_before_resume": [dig(r, "idle", "state_before_resume") for r in runs if "idle" in r],
        "create_to_active_s": stat([dig(r, "create", "total_s") + (dig(r, "create", "wait_active", "seconds") or 0)
                                    for r in runs if ok(r, "create")]),
        "create_success_s": stat([dig(r, "create", "success_s") for r in runs]),
        "stop_to_idle_s": stat([dig(r, "stop", "total_s") + (dig(r, "stop", "wait_stopped", "seconds") or 0)
                                for r in runs if ok(r, "stop")]),
        "delete_success_s": stat([dig(r, "delete", "success_s") for r in runs]),
        "marker_restored": {"count": sum(dig(r, "marker_after_resume", "matches") is True for r in runs),
                            "n": len(runs)},
        "state_after_resume": [r.get("state_after_resume") for r in runs],
        "answers": [[dig(r, t, "response") for t in TURNS] for r in runs],
    }
    for t in TURNS:
        s[f"{t}_total_s"] = stat([dig(r, t, "total_s") for r in runs if ok(r, t)])
        s[f"{t}_success_s"] = stat([dig(r, t, "success_s") for r in runs])
        s[f"{t}_server_s"] = stat([dig(r, t, "server_s") for r in runs])
    by_status = Counter(str(e["status"]) for e in log)
    s["http_attempts"] = {"total": len(log), "by_status": dict(sorted(by_status.items())),
                          "calls_retried": sum(len(dig(r, k, "attempts") or []) > 1 for r in runs for k in run_calls(r))}
    return s


def error_code(text: str) -> str:
    body = as_json(text.encode("utf-8"))
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("code") or "error")
    return "non-JSON error body"


def public_record(record: dict, raw_path: Path) -> dict:
    pub = copy.deepcopy(record)
    pub.pop("project_endpoint", None)
    # build_measurements.py recomputes every published number from "runs"; the summary's *_s keys hold
    # statistics objects, which its duration check would reject, so the summary stays in the private record.
    pub.pop("summary", None)
    for r in pub["runs"]:
        if r.pop("session", None):
            r["session"] = f"session-{r['run']}"
        for key in CALLS:
            step = r.get(key)
            if isinstance(step, dict):
                step.pop("response_id", None)
                if isinstance(step.get("error"), str):
                    step["error"] = error_code(step["error"])
    pub["cleanup"] = {k: v for k, v in pub["cleanup"].items() if k != "ids"}
    pub["_sources"] = [{"file": raw_path.name, "sha256": sha256(raw_path), "bytes": raw_path.stat().st_size,
                        "role": "private raw record from this tool (session IDs, response IDs, endpoint)"}]
    return pub


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-endpoint", required=True, help="https://<account>.services.ai.azure.com/api/projects/<p>")
    p.add_argument("--agent", required=True, help="hosted agent name")
    p.add_argument("--version", required=True, help="concrete agent version to pin each session to")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--resume-after", choices=("stop", "idle"), default="stop",
                   help="stop: explicit :stop before the resume turn; idle: no request for --idle-wait seconds")
    p.add_argument("--idle-wait", type=float, default=17 * 60.0,
                   help="seconds to leave the session untouched with --resume-after idle (default 17 min)")
    p.add_argument("--attempts", type=int, default=4, help="maximum HTTP attempts per call")
    p.add_argument("--timeout", type=float, default=180.0, help="seconds per HTTP attempt")
    p.add_argument("--run-id", default="b-sessions")
    p.add_argument("--path", default="B", choices=("B", "D"),
                   help="B (Foundry-managed compute) or D (the same sessions API on your AKS, BYO-AKS)")
    p.add_argument("--region", default="")
    p.add_argument("--client-note", default="", help="where the client ran and which hops it crossed")
    p.add_argument("--out", required=True, help="private raw JSON (session IDs, endpoint)")
    p.add_argument("--public-out", help="redacted JSON for evidence/runs/")
    a = p.parse_args(argv)
    if a.runs < 1:
        p.error("--runs must be >= 1")

    client = Client(a.project_endpoint, auth_header(), a.attempts, a.timeout)
    started_epoch, started = time.time(), utc()
    rec, payload = client.call("read_version", "GET", f"/agents/{a.agent}/versions/{a.version}")
    definition = (as_json(payload) or {}).get("definition") or {}
    env = definition.get("environment_variables") or {}
    agent = {"name": a.agent, "version": a.version, "read_status": rec["status"], "kind": definition.get("kind"),
             "cpu": definition.get("cpu"), "memory": definition.get("memory"),
             "protocols": [f"{v.get('protocol')} {v.get('version')}" for v in definition.get("protocol_versions") or []],
             "model_deployment": env.get("AI_MODEL_DEPLOYMENT_NAME"), "model_routing": env.get("MODEL_ROUTING")}
    print(f"agent {a.agent} v{a.version}: http={rec['status']} kind={agent['kind']} cpu={agent['cpu']} "
          f"memory={agent['memory']} routing={agent['model_routing']}", flush=True)

    runs = [one_run(client, a.agent, a.version, i, a.resume_after, a.idle_wait) for i in range(1, a.runs + 1)]
    cleanup = sweep(client, a.agent, a.version, started_epoch, {r.get("session") for r in runs if r.get("deleted")})
    record = {
        "run_id": a.run_id, "path": a.path,
        "what": ("Foundry hosted agent" if a.path == "B" else "BYO-AKS hosted agent on your AKS")
                + (": explicit session lifecycle (create, first answer, follow-up, stop, resume on the next request, "
           "file kept, delete)" if a.resume_after == "stop" else
           f": session lifecycle with resume after the idle timeout (create, first answer, follow-up, "
           f"{a.idle_wait:.0f} s without requests, resume on the next request, file kept, delete)"),
        "resume_after": a.resume_after,
        "date_utc": started[:10], "started_at": started, "finished_at": utc(), "region": a.region,
        "client": a.client_note, "project_endpoint": a.project_endpoint,
        "docs": "https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions",
        "agent": agent,
        "inputs": {"first": FIRST, "later_turns": COUNT, "marker_file": MARKER_PATH,
                   "request": "POST .../agents/<agent>/endpoint/protocols/openai/responses with agent_session_id; "
                              "later turns add previous_response_id"},
        "clocks": CLOCKS,
        "retry_policy": f"up to {a.attempts} attempts per call after HTTP 408, 429, 5xx or a network error, "
                        "2 s apart; every attempt is recorded",
        "summary": summarize(runs, client.log),
        "runs": runs, "cleanup": cleanup, "http_log": client.log,
    }
    raw_path = Path(a.out)
    write_json(raw_path, record)
    if a.public_out:
        write_json(Path(a.public_out), public_record(record, raw_path))
    s = record["summary"]
    print(f"SUMMARY runs_complete={s['runs_complete']['count']}/{s['runs_complete']['n']} "
          f"create_to_active={s['create_to_active_s'].get('median')} stop_to_idle={s['stop_to_idle_s'].get('median')} "
          f"resume_total={s['resume_total_s'].get('median')} resume_server={s['resume_server_s'].get('median')} "
          f"marker={s['marker_restored']['count']}/{s['marker_restored']['n']} "
          f"attempts={s['http_attempts']['by_status']} extra_sessions={cleanup['extra_sessions']}", flush=True)
    return 0 if passed(record) else 1


if __name__ == "__main__":
    sys.exit(main())
