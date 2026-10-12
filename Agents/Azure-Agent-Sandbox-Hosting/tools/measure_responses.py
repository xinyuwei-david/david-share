#!/usr/bin/env python3
"""Paths B and D: time new sessions, follow-up turns, concurrency and pod loss through the Responses endpoint.

    python tools/measure_responses.py --cold 3 --warm-turns 3                         # B or D
    python tools/measure_responses.py --cold 3 --warm-turns 3 --scale 4,8 --observe   # D: also watch session pods
    python tools/measure_responses.py --recovery --observe                            # D: delete one session pod

The endpoint is AGENT_RESPONSES_URL; the token comes from `az account get-access-token --resource
https://ai.azure.com`. --observe reads the session pods in SESSION_NAMESPACE with kubectl (D only) and never
starts, stops or scales the cluster. The inputs are the ones used on 2026-09-29:

    cold:   "Reply with exactly COLD-<n>"                       (each a new session)
    warm:   "Remember the code word LANTERN-7. Reply only OK."  then
            "What is the code word? Reply with the word only." then "Reply with exactly WARM-<n>"
    scale:  "Remember SCALE-<n>. Reply only OK." then "What did I ask you to remember? Word only."

Token acquisition is outside the request timer; the token is refreshed before each concurrency level and recovery.
Writes evidence-local/responses-<UTC time>.json. Exit codes: 0 every requested phase answered and retained its
state, 1 any request, observation, concurrency or recovery check failed, 2 a setting is missing or arguments are invalid.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from aca_sandbox import SettingMissing

ROOT = Path(__file__).resolve().parents[1]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def setting(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or "YOUR-" in value:
        raise SettingMissing(name)
    return value


def token() -> str:
    out = subprocess.run(["az", "account", "get-access-token", "--resource", "https://ai.azure.com",
                          "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True,
                         shell=(os.name == "nt"))
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"TOKEN FAILED: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def answer_text(doc: dict) -> str:
    return " ".join(c.get("text", "") for o in doc.get("output", []) if o.get("type") == "message"
                    for c in o.get("content", []))


def invoke(endpoint: str, tok: str, text: str, prev: str | None = None, timeout: int = 420) -> dict:
    body = {"input": text}
    if prev:
        body["previous_response_id"] = prev
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    t0, started = time.perf_counter(), utc()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            first_byte = round(time.perf_counter() - t0, 3)
            doc = json.loads(r.read())
            return {"started_utc": started, "http": r.status, "wall_s": round(time.perf_counter() - t0, 3),
                    "first_byte_s": first_byte, "response_id": doc.get("id"), "status": doc.get("status"),
                    "answer": answer_text(doc)[:80], "error": None}
    except urllib.error.HTTPError as e:
        return {"started_utc": started, "http": e.code, "wall_s": round(time.perf_counter() - t0, 3),
                "first_byte_s": None, "response_id": None, "status": None, "answer": None,
                "error": e.read().decode(errors="replace")[:300]}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"started_utc": started, "http": None, "wall_s": round(time.perf_counter() - t0, 3),
                "first_byte_s": None, "response_id": None, "status": None, "answer": None,
                "error": f"{type(e).__name__}: {e}"[:300]}


def answered(turn: dict | None) -> bool:
    return bool(turn and turn.get("http") == 200 and turn.get("status") == "completed"
                and turn.get("response_id") and isinstance(turn.get("answer"), str)
                and turn["answer"].strip() and not turn.get("error"))


class Pods:
    """Session pods in one namespace, read with kubectl; disabled unless --observe."""

    def __init__(self, namespace: str | None):
        self.namespace = namespace

    def list(self) -> list[dict]:
        if not self.namespace:
            return []
        out = subprocess.run(["kubectl", "get", "pods", "-n", self.namespace, "-o", "json", "--request-timeout=60s"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"kubectl get pods failed: {out.stderr.strip()[:300]}")
        rows = []
        for p in json.loads(out.stdout)["items"]:
            ready = next((c for c in p.get("status", {}).get("conditions", []) if c["type"] == "Ready"), None)
            rows.append({"name": p["metadata"]["name"], "node": p["spec"].get("nodeName"),
                         "created": p["metadata"]["creationTimestamp"],
                         "ready_at": ready["lastTransitionTime"] if ready and ready["status"] == "True" else None,
                         "runtime": p["spec"].get("runtimeClassName")})
        return rows

    def new_since(self, before: set[str]) -> list[dict]:
        rows = []
        for p in self.list():
            if p["name"] in before or not p["ready_at"]:
                continue
            c = datetime.fromisoformat(p["created"].replace("Z", "+00:00"))
            r = datetime.fromisoformat(p["ready_at"].replace("Z", "+00:00"))
            rows.append({**p, "create_to_ready_s": round((r - c).total_seconds(), 1)})
        return rows


def run_cold(endpoint, tok, pods: Pods, n: int) -> list[dict]:
    turns = []
    for i in range(n):
        before = {p["name"] for p in pods.list()}
        r = invoke(endpoint, tok, f"Reply with exactly COLD-{i + 1}")
        time.sleep(3)
        new = pods.new_since(before)
        r["pod"] = new[0] if len(new) == 1 else None
        if pods.namespace and answered(r) and (len(new) != 1 or new[0]["runtime"] != "kata-vm-isolation"):
            r["observation_error"] = f"expected one new Ready Kata pod, observed {len(new)} candidates"
        print(f"{utc()}  cold {i + 1}/{n}: http={r['http']} wall={r['wall_s']}s -> {r['answer']}", flush=True)
        turns.append(r)
    return turns


def run_warm(endpoint, tok, n_turns: int) -> dict:
    first = invoke(endpoint, tok, "Remember the code word LANTERN-7. Reply only OK.")
    turns, prev = [first], first["response_id"]
    if not answered(first):
        return {"turns": turns, "remembered": False}
    for i in range(n_turns):
        text = "What is the code word? Reply with the word only." if i == 0 else f"Reply with exactly WARM-{i}"
        r = invoke(endpoint, tok, text, prev)
        print(f"{utc()}  warm turn {i + 2}: http={r['http']} wall={r['wall_s']}s -> {r['answer']}", flush=True)
        turns.append(r)
        if not answered(r):
            break
        prev = r["response_id"]
    remembered = len(turns) > 1 and answered(turns[1]) and turns[1]["answer"].strip() == "LANTERN-7"
    return {"turns": turns, "remembered": remembered}


def run_scale(endpoint, tok, pods: Pods, n: int) -> dict:
    before = {p["name"] for p in pods.list()}

    def one(i: int) -> dict:
        a = invoke(endpoint, tok, f"Remember SCALE-{i}. Reply only OK.")
        b = invoke(endpoint, tok, "What did I ask you to remember? Word only.", a["response_id"]) if answered(a) else None
        return {"i": i, "t1": a, "t2": b,
                "remembered": b is not None and answered(b) and b["answer"].strip() == f"SCALE-{i}"}

    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        sessions = list(ex.map(one, range(n)))
    time.sleep(5)
    new = pods.new_since(before)
    count = sum(1 for s in sessions if answered(s["t1"]))
    print(f"{utc()}  concurrency {n}: answered {count}/{n}, new pods {len(new)}", flush=True)
    return {"concurrency": n, "sessions": sessions, "answered": count, "new_pods": len(new)}


def run_recovery(endpoint, tok, pods: Pods) -> dict:
    before = {p["name"] for p in pods.list()}
    a = invoke(endpoint, tok, "Remember RECOVER-1. Reply only OK.")
    time.sleep(3)
    new = [p for p in pods.list() if p["name"] not in before]
    if not answered(a) or len(new) != 1 or new[0]["runtime"] != "kata-vm-isolation":
        reason = f"no unique Kata session pod to delete (http={a['http']}, new pods={len(new)})"
        print(f"RECOVERY BLOCKED {reason}", file=sys.stderr)
        return {"skipped": reason, "state_preserved": False}
    victim = new[0]["name"]
    deleted = subprocess.run(["kubectl", "delete", "pod", victim, "-n", pods.namespace, "--wait=true",
                              "--timeout=120s", "--request-timeout=60s"], capture_output=True, text=True, timeout=150)
    if deleted.returncode != 0:
        raise RuntimeError(f"kubectl delete pod failed: {deleted.stderr.strip()[:300]}")
    b = invoke(endpoint, tok, "What did I ask you to remember? Word only.", a["response_id"])
    return {"deleted_pod": victim, "turn_after_delete": b,
            "state_preserved": answered(b) and b["answer"].strip() == "RECOVER-1"}


def report_passes(report: dict) -> bool:
    if report.get("error"):
        return False
    cold, warm = report["cold"], report["warm"]
    if not cold or not warm["remembered"] or not all(answered(t) for t in cold + warm["turns"]):
        return False
    if any(t["answer"].strip() != f"COLD-{i + 1}" or t.get("observation_error") for i, t in enumerate(cold)):
        return False
    for level in report.get("scale", []):
        if len(level["sessions"]) != level["concurrency"] or not all(
            answered(s["t1"]) and answered(s["t2"]) and s["remembered"] for s in level["sessions"]
        ):
            return False
    if "recovery" in report:
        recovery = report["recovery"]
        if recovery.get("skipped") or not recovery.get("state_preserved") or not answered(recovery.get("turn_after_delete")):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cold", type=int, default=3)
    ap.add_argument("--warm-turns", type=int, default=3)
    ap.add_argument("--scale", default="", help="comma list of concurrency levels, for example 4,8")
    ap.add_argument("--recovery", action="store_true", help="D: delete one new session pod and ask again")
    ap.add_argument("--observe", action="store_true", help="D: read session pods in SESSION_NAMESPACE")
    ap.add_argument("--pause", type=int, default=20, help="seconds between phases")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evidence-local")
    args = ap.parse_args(argv)
    if args.cold < 1 or args.warm_turns < 1 or args.pause < 0:
        ap.error("--cold and --warm-turns must be positive; --pause must be non-negative")
    try:
        levels = [int(x) for x in args.scale.split(",") if x.strip()]
    except ValueError:
        ap.error("--scale must be a comma list of positive integers")
    if any(level < 1 for level in levels):
        ap.error("--scale must be a comma list of positive integers")
    report: dict = {"started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        endpoint = setting("AGENT_RESPONSES_URL")
        pods = Pods(setting("SESSION_NAMESPACE") if args.observe or args.recovery else None)
        report["observe"] = bool(pods.namespace)
        tok = token()
        report["cold"] = run_cold(endpoint, tok, pods, args.cold)
        time.sleep(args.pause)
        report["warm"] = run_warm(endpoint, tok, args.warm_turns)
        for level in levels:
            time.sleep(args.pause)
            report.setdefault("scale", []).append(run_scale(endpoint, token(), pods, level))
        if args.recovery:
            time.sleep(args.pause)
            report["recovery"] = run_recovery(endpoint, token(), pods)
    except SettingMissing as exc:
        print(f"SETTING MISSING {exc}: set it in .env (see config/.env.example)", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"MEASURE FAILED {report['error']}", file=sys.stderr)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"responses-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = report_passes(report)
    warm = report.get("warm", {"turns": [], "remembered": False})
    print(f"{'MEASURE PASS' if ok else 'MEASURE FAIL'} cold {len(report.get('cold', []))}, warm "
          f"{len(warm['turns'])} turns, remembered={warm['remembered']}, "
          f"scale levels={len(report.get('scale', []))}, recovery={report.get('recovery', {}).get('state_preserved')} -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
