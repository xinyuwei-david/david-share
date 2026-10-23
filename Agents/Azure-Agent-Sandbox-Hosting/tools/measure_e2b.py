#!/usr/bin/env python3
"""Path E measurement: a Foundry Prompt Agent that calls E2B Firecracker sandboxes on one or more Azure VMs.

    python tools/measure_e2b.py agent --runs 3
    python tools/measure_e2b.py perf --cold 5 --scale 4,8 --recovery
    python tools/measure_e2b.py agent --runs 3 --hosts hosts.json --host b
    python tools/measure_e2b.py scale --hosts hosts.json --per-host 8
    python tools/measure_e2b.py hostloss --hosts hosts.json --lose a --stop-cmd "<command that stops host a>" \
        --start-cmd "<command that starts it again>"
    python tools/measure_e2b.py host --bench 20 --reserve 4      # on the E2B host itself, no model

Each run creates its own agent version (prompt agent, model ACA_MODEL, the one function tool below), runs the
protocol, then deletes the version, every conversation and every sandbox it created. The instructions, tool schema
and prompts are the ones that produced evidence/runs/e-e2b-2026-10-14 and are printed in every output file.

agent    per run: Sandbox.create(template) -> FIRST_Q -> COUNT_Q -> beta_pause -> wait 5 s -> Sandbox.connect ->
         COUNT_Q -> COUNT_Q -> kill. The tool writes the question into /home/user/history.json inside the sandbox
         before running the model's code, so the history length must grow 1, 2, 3, 4 across pause and resume.
perf     cold: n new conversations, each with a new sandbox, one COLD prompt; warm: one session, LANTERN-7 written
         to a file and read back, then two WARM turns; at once: 4 and 8 sessions in parallel, each writes SCALE-<k>
         to a file and reads it back; recovery: kill one sandbox, create a replacement, then a live session reads
         back LANTERN-7.
scale    the same at-once protocol, `--per-host` sessions on every host in --hosts at the same time (round robin).
hostloss on every host: one live session (file written) and one paused sandbox. Then --stop-cmd stops the host
         named by --lose, and the tool records what each host can still do, including whether the lost host's
         paused sandbox can be resumed from another host. --start-cmd brings it back and the lost host's own paused
         sandbox is resumed again.
host     no model or agent; run it on the E2B host. One sandbox reads its kernel, vCPUs and memory with run_code
         while the host's Firecracker processes are listed. --bench n: n sandboxes in turn, each create -> cell
         (x = 41) -> cell (x += 1) -> beta_pause -> wait 5 s -> Sandbox.connect -> cell (print(x) must give 42) ->
         kill. --reserve k: sandboxes created one by one, each checked with a cell, /proc/meminfo read after each;
         it stops at the first refused create, then kills them all.

Hosts: without --hosts, one host from E2B_API_URL, E2B_SANDBOX_URL and E2B_API_KEY (each Embed install has its own
key). --hosts names a JSON file {"a": {"api_url": ..., "sandbox_url": ..., "api_key_env": "ENV_NAME"}, ...}; keys
are read from the named environment variables and never written to the output. Every SDK call passes that host's
api_url, sandbox_url and api_key explicitly, so one process can drive several independent Embed installs.

The tool runs the model's code with E2B's Code Interpreter SDK (e2b-code-interpreter): run_code on its
code-interpreter-v1 template, one cell in the sandbox's Jupyter kernel, so variables persist between calls and across
pause and resume. Each call first records the question in /home/user/history.json; sandbox_tool_s times the whole
call. --executor accepts only code-interpreter; it stays so that the recorded 2026-10-17 commands still run. The
2026-10-14 and 2026-10-15 runs used the revision of commit 7c9a13b, which ran python3 with commands.run in the base
template for every call.

Clocks: time.perf_counter() in this process. create_total_s / sandbox_create_to_ready_s: Sandbox.create until it
returns (Running). A turn's wall_s / agent_turn_s: first POST /openai/v1/responses until the final answer, including
every tool round trip; sandbox_tool_s: the sandbox command alone. agent mode cold_total_s and recover_total_s are
sums (create or connect + turn). A perf cold record's wall_s is one stopwatch around conversation create + sandbox
create + turn. An at-once session's turn1.wall_s is sandbox create + turn 1 (the 2026-10-14 driver's clock), and
turn1.agent_turn_s is the turn alone. Pause, the 5 s wait and idle time are outside every resume clock.

Reads PROJECT_ENDPOINT and ACA_MODEL (not in host mode). Foundry token: with FOUNDRY_TOKEN_SOURCE=imds, the VM's managed identity from
the instance metadata service (the identity needs the Foundry User role on the project); else FOUNDRY_BEARER_TOKEN if
set; else `az account get-access-token --resource https://ai.azure.com`. The token is never printed. Writes
evidence-local/e2b-<mode>-<UTC time>.json. Exit codes: 0 every requested phase passed, 1 a phase failed, lost state
or left a resource behind, 2 a required setting is missing or arguments are invalid.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AGENT_NAME = "e2b-sandbox-benchmark-agent"
TOOL_NAME = "run_in_e2b_sandbox"
INSTRUCTIONS = (
    "You are a concise assistant with access to `run_in_e2b_sandbox`. "
    "For EVERY user message, first call `run_in_e2b_sandbox` once to inspect or update files in /home/user, "
    "then answer concisely as instructed. When asked for a number only or word only, reply with that exact token only."
)
TOOLS_SPEC = [{
    "type": "function",
    "name": TOOL_NAME,
    "description": "Execute Python 3 code inside the session's isolated E2B Firecracker microVM sandbox on Azure.",
    "parameters": {
        "type": "object",
        "properties": {"python_code": {"type": "string",
                                       "description": "Python 3 code to run inside the E2B sandbox. Print output to stdout."}},
        "required": ["python_code"],
        "additionalProperties": False,
    },
    "strict": True,
}]
FIRST_Q = "Where is Bangkok? Answer in one short sentence."
COUNT_Q = "How many questions have I asked before this one? Number only."
COLD_Q = "Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}"
WARM_QS = [
    "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK.",
    "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only.",
    "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1",
    "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2",
]
SCALE_WRITE = "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK."
SCALE_READ = "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."
LIVE_WRITE = "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."
LIVE_READ = "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only."
PAUSE_WRITE = "Use run_in_e2b_sandbox to write PAUSED-{h} to /home/user/paused.txt. Reply only OK."
PAUSE_READ = "Use run_in_e2b_sandbox to read /home/user/paused.txt. What does it say? Word only."
RESUME_WAIT_S = 5
EXECUTOR, TEMPLATE = "code-interpreter", "code-interpreter-v1"
GUEST_PROBE = ("import os, platform\n"
               "mem_kb = next(int(ln.split()[1]) for ln in open('/proc/meminfo') if ln.startswith('MemTotal:'))\n"
               "print(platform.release(), os.cpu_count(), mem_kb // 1024)")


class SettingMissing(RuntimeError):
    pass


def need_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val or val.startswith("YOUR-"):
        raise SettingMissing(f"missing or placeholder environment variable: {name}")
    return val


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def utc_hms() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


IMDS_TOKEN_URL = ("http://169.254.169.254/metadata/identity/oauth2/token"
                  "?api-version=2018-02-01&resource=https%3A%2F%2Fai.azure.com")


def token() -> str:
    if os.environ.get("FOUNDRY_TOKEN_SOURCE", "").strip().lower() == "imds":
        req = urllib.request.Request(IMDS_TOKEN_URL, headers={"Metadata": "true"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tok = json.loads(resp.read().decode("utf-8")).get("access_token", "")
        if not tok:
            raise RuntimeError("the instance metadata service returned no access_token for https://ai.azure.com")
        return tok
    tok = os.environ.get("FOUNDRY_BEARER_TOKEN", "").strip()
    if tok:
        return tok
    az = "az.cmd" if os.name == "nt" else "az"
    out = subprocess.run([az, "account", "get-access-token", "--resource", "https://ai.azure.com",
                          "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True, timeout=90,
                         check=False)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"az account get-access-token failed: {out.stderr.strip()[:300]}")
    return out.stdout.strip()


def load_sdk():
    try:
        from e2b import SandboxQuery, SandboxState
        from e2b_code_interpreter import Sandbox
        return Sandbox, SandboxQuery, SandboxState
    except ImportError as exc:
        raise RuntimeError("Install the SDKs first: pip install e2b==2.56.0 e2b-code-interpreter==2.11.0") from exc


def sdk_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    found = {}
    for name in ("e2b", "e2b-code-interpreter"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


class Foundry:
    """Foundry project data plane. Retries 408, 429 and 5xx twice after 2 s and 4 s, as the 2026-10-14 driver did,
    and records every attempt's HTTP status for the turn that made it, so throttling stays visible."""

    def __init__(self, endpoint: str, tok: str):
        self.endpoint, self.tok = endpoint.rstrip("/"), tok
        self._local = threading.local()

    def begin(self) -> None:
        self._local.statuses = []

    def statuses(self) -> list[int]:
        return list(getattr(self._local, "statuses", []))

    def _note(self, status: int) -> None:
        seen = getattr(self._local, "statuses", None)
        if seen is not None:
            seen.append(status)

    def req(self, method: str, path: str, body: dict | None = None, timeout: int = 180) -> dict:
        url = f"{self.endpoint}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        for attempt in range(3):
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {self.tok}", "Content-Type": "application/json",
                "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    self._note(resp.status)
                    return json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as exc:
                self._note(exc.code)
                text = exc.read().decode("utf-8", errors="replace")
                if exc.code in (408, 429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {text[:300]}") from exc
        raise RuntimeError(f"{method} {path}: no response")

    def create_agent(self, model: str) -> str:
        doc = self.req("POST", f"/agents/{AGENT_NAME}/versions?api-version=v1", {
            "description": "Foundry Prompt Agent benchmarked with E2B Firecracker sandboxes on Azure",
            "definition": {"kind": "prompt", "model": model, "instructions": INSTRUCTIONS, "tools": TOOLS_SPEC}})
        return str(doc.get("version") or "1")

    def delete_agent(self, version: str) -> None:
        self.req("DELETE", f"/agents/{AGENT_NAME}/versions/{version}?api-version=v1")

    def conversation(self) -> str:
        return self.req("POST", "/openai/v1/conversations", {})["id"]

    def drop(self, conv_id: str | None) -> bool:
        if not conv_id:
            return True
        try:
            self.req("DELETE", f"/openai/v1/conversations/{conv_id}")
            return True
        except Exception:  # noqa: BLE001 - reported by the caller through the returned flag
            return False


class Host:
    """One independent E2B Embed install; every SDK call names it explicitly."""

    def __init__(self, name: str, api_url: str, sandbox_url: str, api_key: str, sdk):
        self.name, self.api_url, self.sandbox_url, self._key = name, api_url, sandbox_url, api_key
        self.Sandbox, self.Query, self.State = sdk

    def opts(self) -> dict:
        return {"api_url": self.api_url, "sandbox_url": self.sandbox_url, "api_key": self._key}

    def create(self):
        return self.Sandbox.create(TEMPLATE, timeout=300, **self.opts())

    def connect(self, sandbox_id: str):
        return self.Sandbox.connect(sandbox_id, **self.opts())

    def live(self) -> list[str]:
        pager = self.Sandbox.list(query=self.Query(state=[self.State.RUNNING, self.State.PAUSED]), **self.opts())
        ids = []
        while pager.has_next:
            ids += [s.sandbox_id for s in pager.next_items(**self.opts())]
        return ids

    def health(self) -> int:
        try:
            with urllib.request.urlopen(f"{self.api_url.rstrip('/')}/health", timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except Exception:  # noqa: BLE001 - an unreachable host is status 0, not an exception
            return 0


def load_hosts(path: Path | None, sdk) -> dict[str, Host]:
    if path is None:
        return {"a": Host("a", need_env("E2B_API_URL"), need_env("E2B_SANDBOX_URL"), need_env("E2B_API_KEY"), sdk)}
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingMissing(f"{path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(spec, dict) or not spec:
        raise SettingMissing(f"{path}: expected a JSON object of hosts")
    hosts = {}
    for name, h in spec.items():
        if not isinstance(h, dict):
            raise SettingMissing(f"{path}: host {name} must be a JSON object")
        for k in ("api_url", "sandbox_url", "api_key_env"):
            if not isinstance(h.get(k), str) or not h[k]:
                raise SettingMissing(f"{path}: host {name} has no {k}")
        hosts[name] = Host(name, h["api_url"], h["sandbox_url"], need_env(h["api_key_env"]), sdk)
    return hosts


def sandbox_run_code(sbx, code: str, question: str) -> dict:
    """Code Interpreter: the same history step, then the model's code, as one cell in the sandbox's kernel."""
    cell = (
        "import json as _json, os as _os, platform as _platform\n"
        "_p = '/home/user/history.json'\n"
        "_h = _json.load(open(_p)) if _os.path.exists(_p) else []\n"
        "prior_questions = len(_h)\n"
        f"_h.append({{'turn': prior_questions + 1, 'question': {question!r}}})\n"
        "_json.dump(_h, open(_p, 'w'))\n"
        "history_length = len(_h)\n"
        "print('__STATE__=' + _json.dumps({'prior_questions': prior_questions, 'history_length': history_length, "
        "'kernel': _platform.release()}))\n"
        f"{code}\n"
    )
    t0 = time.perf_counter()
    ex = sbx.run_code(cell, timeout=60)
    elapsed = round(time.perf_counter() - t0, 3)
    if ex.error is not None:
        raise RuntimeError(f"sandbox kernel {ex.error.name}: {ex.error.value}")
    lines = [ln for ln in "".join(ex.logs.stdout).splitlines() if ln.strip()]
    state_line = next((ln[len("__STATE__="):] for ln in lines if ln.startswith("__STATE__=")), "{}")
    out = [ln for ln in lines if not ln.startswith("__STATE__=")] + ([ex.text] if ex.text else [])
    return {"exec_s": elapsed, "stdout": "\n".join(out).strip(), "state": json.loads(state_line)}


def extract_text(doc: dict) -> str:
    return " ".join(c["text"] for item in doc.get("output", []) or [] if item.get("type") == "message"
                    for c in item.get("content", []) or [] if c.get("type") in ("output_text", "text")
                    and c.get("text")).strip()


def ask(fd: Foundry, version: str, conv_id: str, sbx, question: str) -> dict:
    """One user turn; every tool call the model makes runs inside `sbx`."""
    fd.begin()
    t_start, started = time.perf_counter(), utc_hms()
    ref = {"type": "agent_reference", "name": AGENT_NAME, "version": version}
    resp = fd.req("POST", "/openai/v1/responses", {
        "conversation": conv_id, "input": [{"type": "message", "role": "user", "content": question}],
        "agent_reference": ref})
    tool_s, calls, state = 0.0, 0, {}
    while True:
        fn_calls = [i for i in resp.get("output", []) if i.get("type") == "function_call"]
        if not fn_calls:
            break
        outputs = []
        for call in fn_calls:
            calls += 1
            args = json.loads(call.get("arguments") or "{}")
            info = sandbox_run_code(sbx, args.get("python_code", "print('ok')"), question)
            tool_s += info["exec_s"]
            state = info["state"]
            outputs.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps({
                "stdout": info["stdout"], "prior_questions_in_sandbox": state.get("prior_questions"),
                "history_length_in_sandbox": state.get("history_length"), "guest_kernel": state.get("kernel")},
                ensure_ascii=False)})
        resp = fd.req("POST", "/openai/v1/responses", {"conversation": conv_id, "input": outputs,
                                                        "agent_reference": ref})
    if calls == 0:
        # The 2026-10-14 driver kept the sandbox history in step when the model answered without the tool.
        info = sandbox_run_code(sbx, "print('fallback_sync')", question)
        tool_s += info["exec_s"]
        state = info["state"]
    answer = extract_text(resp)
    if not answer:
        raise RuntimeError(f"non-empty answer missing for {question!r}")
    return {"started_utc": started, "http": 200, "wall_s": round(time.perf_counter() - t_start, 2),
            "sandbox_tool_s": round(tool_s, 3), "tool_calls": calls, "response": answer,
            "http_statuses": fd.statuses(),
            "history_length": state.get("history_length", 0), "guest_kernel": state.get("kernel", "")}


def kill(sbx) -> bool:
    if sbx is None:
        return True
    try:
        sbx.kill()
        return True
    except Exception:  # noqa: BLE001 - reported through the returned flag
        return False


def agent_run(fd: Foundry, version: str, host: Host, index: int) -> dict:
    rec: dict = {"run": index, "host": host.name, "started_at": now_iso()}
    sbx, conv = None, None
    try:
        conv = fd.conversation()
        t0 = time.perf_counter()
        sbx = host.create()
        rec["create_total_s"] = round(time.perf_counter() - t0, 2)
        sbx_id = sbx.sandbox_id
        rec["first"] = ask(fd, version, conv, sbx, FIRST_Q)
        rec["cold_total_s"] = round(rec["create_total_s"] + rec["first"]["wall_s"], 2)
        rec["warm"] = ask(fd, version, conv, sbx, COUNT_Q)
        t1 = time.perf_counter()
        sbx.beta_pause()
        rec["stop_total_s"], rec["state_after_stop"] = round(time.perf_counter() - t1, 2), "Paused"
        time.sleep(RESUME_WAIT_S)
        t2 = time.perf_counter()
        sbx = host.connect(sbx_id)
        rec["resume_total_s"], rec["state_after_resume"] = round(time.perf_counter() - t2, 2), "Running"
        rec["first_after_resume"] = ask(fd, version, conv, sbx, COUNT_Q)
        rec["recover_total_s"] = round(rec["resume_total_s"] + rec["first_after_resume"]["wall_s"], 2)
        rec["warm_after_resume"] = ask(fd, version, conv, sbx, COUNT_Q)
        lengths = [rec[k]["history_length"] for k in ("first", "warm", "first_after_resume", "warm_after_resume")]
        rec["history_lengths"] = lengths
        rec["state_kept"] = lengths == [1, 2, 3, 4]
        if not rec["state_kept"]:
            raise RuntimeError(f"sandbox history did not grow 1..4 across pause and resume: {lengths}")
    except Exception as exc:  # noqa: BLE001 - recorded; cleanup still runs
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["sandbox_deleted"] = kill(sbx)
        rec["conversation_deleted"] = fd.drop(conv)
        rec["finished_at"] = now_iso()
    return rec


def cold_once(fd: Foundry, version: str, host: Host, i: int) -> dict:
    rec: dict = {"i": i, "host": host.name, "started_utc": utc_hms()}
    sbx, conv = None, None
    t_all = time.perf_counter()
    try:
        conv = fd.conversation()
        t0 = time.perf_counter()
        sbx = host.create()
        rec["sandbox_create_to_ready_s"] = round(time.perf_counter() - t0, 3)
        turn = ask(fd, version, conv, sbx, COLD_Q.format(i=i))
        rec.update(wall_s=round(time.perf_counter() - t_all, 3), agent_turn_s=turn["wall_s"],
                   sandbox_tool_s=turn["sandbox_tool_s"], http=200, answer=turn["response"],
                   correct=turn["response"].strip() == f"COLD-{i}")
    except Exception as exc:  # noqa: BLE001 - recorded
        rec.update(http=0, error=f"{type(exc).__name__}: {exc}", wall_s=round(time.perf_counter() - t_all, 3))
    finally:
        rec["sandbox_deleted"], rec["conversation_deleted"] = kill(sbx), fd.drop(conv)
    return rec


def warm_session(fd: Foundry, version: str, host: Host) -> dict:
    out: dict = {"host": host.name, "turns": []}
    sbx, conv = None, None
    try:
        conv = fd.conversation()
        t0 = time.perf_counter()
        sbx = host.create()
        out["sandbox_create_to_ready_s"] = round(time.perf_counter() - t0, 3)
        for q in WARM_QS:
            t = ask(fd, version, conv, sbx, q)
            out["turns"].append({"started_utc": t["started_utc"], "http": 200, "wall_s": t["wall_s"],
                                 "sandbox_tool_s": t["sandbox_tool_s"], "answer": t["response"]})
        out["remembered"] = "LANTERN-7" in out["turns"][1]["answer"]
    except Exception as exc:  # noqa: BLE001 - recorded
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["remembered"] = False
    finally:
        out["sandbox_deleted"], out["conversation_deleted"] = kill(sbx), fd.drop(conv)
    return out


def scale_session(fd: Foundry, version: str, host: Host, k: int) -> dict:
    rec: dict = {"slot": k, "host": host.name, "started_utc": utc_hms()}
    sbx, conv = None, None
    t_all = time.perf_counter()
    try:
        conv = fd.conversation()
        t0 = time.perf_counter()
        sbx = host.create()
        rec["sandbox_create_to_ready_s"] = round(time.perf_counter() - t0, 3)
        t1 = ask(fd, version, conv, sbx, SCALE_WRITE.format(k=k))
        # Same clock as the 2026-10-14 driver: turn 1 wall time includes the sandbox create before it.
        rec["turn1"] = {"http": 200, "agent_turn_s": t1["wall_s"],
                        "wall_s": round(t1["wall_s"] + rec["sandbox_create_to_ready_s"], 3), "answer": t1["response"],
                        "http_statuses": t1["http_statuses"]}
        t2 = ask(fd, version, conv, sbx, SCALE_READ)
        rec["turn2"] = {"http": 200, "wall_s": t2["wall_s"], "answer": t2["response"],
                        "http_statuses": t2["http_statuses"]}
        rec["remembered"] = f"SCALE-{k}" in t2["response"]
    except Exception as exc:  # noqa: BLE001 - recorded
        rec.update(error=f"{type(exc).__name__}: {exc}", remembered=False, failed_turn_http_statuses=fd.statuses())
    finally:
        rec["session_s"] = round(time.perf_counter() - t_all, 3)
        rec["sandbox_deleted"], rec["conversation_deleted"] = kill(sbx), fd.drop(conv)
    return rec


def at_once(fd: Foundry, version: str, slots: list[tuple[Host, int]]) -> dict:
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        sessions = list(pool.map(lambda s: scale_session(fd, version, s[0], s[1]), slots))
    per_host: dict[str, dict] = {}
    for s in sessions:
        h = per_host.setdefault(s["host"], {"sessions": 0, "answered": 0})
        h["sessions"] += 1
        h["answered"] += bool(s.get("remembered"))
    seen = [c for s in sessions for key in ("turn1", "turn2") for c in (s.get(key) or {}).get("http_statuses", [])]
    seen += [c for s in sessions for c in s.get("failed_turn_http_statuses", [])]
    return {"sessions_at_once": len(slots), "answered": sum(bool(s.get("remembered")) for s in sessions),
            "per_host": per_host, "wall_s": round(time.perf_counter() - t0, 3),
            "model_http_statuses": {str(c): seen.count(c) for c in sorted(set(seen))}, "sessions": sessions}


def recovery(fd: Foundry, version: str, host: Host) -> dict:
    out: dict = {"host": host.name}
    live = victim = repl = None
    conv = None
    try:
        conv = fd.conversation()
        live = host.create()
        victim = host.create()
        ask(fd, version, conv, live, LIVE_WRITE)
        out["sandbox_killed_utc"] = utc_hms()
        t0 = time.perf_counter()
        victim.kill()
        victim = None
        repl = host.create()
        out["replacement_ready_s"] = round(time.perf_counter() - t0, 3)
        f = ask(fd, version, conv, live, LIVE_READ)
        out["followup_on_live_session"] = {"http": 200, "wall_s": f["wall_s"], "answer": f["response"],
                                           "state_preserved": "LANTERN-7" in f["response"]}
    except Exception as exc:  # noqa: BLE001 - recorded
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out["sandboxes_deleted"] = all([kill(live), kill(victim), kill(repl)])
        out["conversation_deleted"] = fd.drop(conv)
    return out


def run_shell(cmd: str) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=900, check=False)
    return {"exit_code": p.returncode, "seconds": round(time.perf_counter() - t0, 2),
            "tail": (p.stdout + p.stderr)[-600:]}


def wait_health(host: Host, want_up: bool, limit_s: float) -> dict:
    t0 = time.perf_counter()
    while True:
        code = host.health()
        if (code == 200) == want_up or time.perf_counter() - t0 >= limit_s:
            return {"status": code, "seconds": round(time.perf_counter() - t0, 2)}
        time.sleep(2)


def attempt(fn) -> dict:
    t0 = time.perf_counter()
    try:
        value = fn()
        return {"ok": True, "seconds": round(time.perf_counter() - t0, 2), "value": value}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "seconds": round(time.perf_counter() - t0, 2),
                "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def session_attempt(fn) -> dict:
    """attempt() for cold_once, which records its own failure instead of raising."""
    out = attempt(fn)
    if out["ok"] and "error" in out["value"]:
        out.update(ok=False, error=out["value"]["error"])
    return out


def hostloss(fd: Foundry, version: str, hosts: dict[str, Host], lose: str, stop_cmd: str, start_cmd: str) -> dict:
    lost, others = hosts[lose], [h for n, h in hosts.items() if n != lose]
    out: dict = {"lost_host": lose, "other_hosts": [h.name for h in others], "before": {}, "after_loss": {},
                 "after_restart": {}}
    live, paused, convs = {}, {}, []
    try:
        for h in hosts.values():
            c_live = fd.conversation()
            convs.append(c_live)
            c_pause = fd.conversation()
            convs.append(c_pause)
            live[h.name] = (h.create(), c_live)
            ask(fd, version, c_live, live[h.name][0], LIVE_WRITE)
            p = h.create()
            ask(fd, version, c_pause, p, PAUSE_WRITE.format(h=h.name))
            p.beta_pause()
            paused[h.name] = (p.sandbox_id, c_pause)
            out["before"][h.name] = {"health": h.health(), "live_and_paused": len(h.live())}
        out["stop"] = run_shell(stop_cmd)
        out["stop"]["health_down"] = wait_health(lost, want_up=False, limit_s=120)
        for h in others:
            sbx, conv = live[h.name]
            out["after_loss"][h.name] = {
                "live_session_read": attempt(lambda: ask(fd, version, conv, sbx, LIVE_READ)["response"]),
                "new_session": session_attempt(lambda: cold_once(fd, version, h, 1)),
                "connect_lost_hosts_paused_sandbox": attempt(lambda: h.connect(paused[lose][0]).sandbox_id)}
        out["after_loss"][lose] = {"create": attempt(lambda: kill(lost.create())),
                                   "connect_own_paused_sandbox": attempt(lambda: lost.connect(paused[lose][0]).sandbox_id)}
        out["start"] = run_shell(start_cmd)
        out["start"]["health_up"] = wait_health(lost, want_up=True, limit_s=600)

        def resume_own():
            sbx = lost.connect(paused[lose][0])
            paused[lose] = (sbx.sandbox_id, paused[lose][1])
            return ask(fd, version, paused[lose][1], sbx, PAUSE_READ)["response"]

        out["after_restart"][lose] = {"resume_own_paused_sandbox": attempt(resume_own),
                                      "live_sandbox_before_loss": attempt(
                                          lambda: lost.connect(live[lose][0].sandbox_id).sandbox_id),
                                      "new_session": session_attempt(lambda: cold_once(fd, version, lost, 2))}
    except Exception as exc:  # noqa: BLE001 - recorded; cleanup still runs
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        left = {}
        for h in hosts.values():
            if h.health() != 200:
                left[h.name] = "unreachable"
                continue
            try:
                for sid in h.live():
                    h.Sandbox.kill(sid, **h.opts())
                left[h.name] = len(h.live())
            except Exception as exc:  # noqa: BLE001 - recorded
                left[h.name] = f"{type(exc).__name__}: {exc}"
        out["left_after_cleanup"] = left
        out["conversations_deleted"] = all([fd.drop(c) for c in convs])
    return out


def meminfo(path: Path = Path("/proc/meminfo")) -> dict:
    """This machine's hugepage pool (2 MiB pages) and available memory."""
    vals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        vals[key] = int(rest.split()[0]) if rest.split() else 0
    return {"hugepages_total": vals["HugePages_Total"], "hugepages_free": vals["HugePages_Free"],
            "hugepages_rsvd": vals["HugePages_Rsvd"], "mem_available_mb": vals["MemAvailable"] // 1024}


def vmm_processes(proc: Path = Path("/proc")) -> list[str]:
    """Executable path of every Firecracker process visible here; empty unless run on the E2B host."""
    found = []
    for f in proc.glob("[0-9]*/cmdline"):
        try:
            argv0 = f.read_bytes().split(b"\0", 1)[0].decode("utf-8", errors="replace")
        except OSError:
            continue
        if argv0.rsplit("/", 1)[-1] == "firecracker":
            found.append(argv0)
    return sorted(found)


def run_cell(sbx, code: str) -> str:
    ex = sbx.run_code(code, timeout=60)
    if ex.error is not None:
        raise RuntimeError(f"sandbox kernel {ex.error.name}: {ex.error.value}")
    return ("".join(ex.logs.stdout) + (ex.text or "")).strip()


def isolation(host: Host) -> dict:
    out: dict = {"host_kernel": os.uname().release}
    sbx = None
    try:
        sbx = host.create()
        kernel, cpus, mem_mb = run_cell(sbx, GUEST_PROBE).split()
        out.update(guest_kernel=kernel, guest_cpus=int(cpus), guest_mem_total_mb=int(mem_mb))
        info = sbx.get_info()
        out["sandbox_info"] = {"template_name": info.name, "cpu_count": info.cpu_count, "memory_mb": info.memory_mb}
        out["vmm_processes"] = vmm_processes()
        if not out["vmm_processes"]:
            out["vmm_note"] = "no Firecracker process visible: run host mode on the E2B host"
    except Exception as exc:  # noqa: BLE001 - recorded
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out["sandbox_deleted"] = kill(sbx)
    return out


def bench_once(host: Host, i: int) -> dict:
    rec: dict = {"i": i}
    sbx = None
    try:
        t0 = time.perf_counter()
        sbx = host.create()
        rec["create_s"] = round(time.perf_counter() - t0, 4)
        t1 = time.perf_counter()
        run_cell(sbx, "x = 41")
        rec["first_cell_s"] = round(time.perf_counter() - t1, 4)
        t2 = time.perf_counter()
        run_cell(sbx, "x += 1")
        rec["warm_cell_s"] = round(time.perf_counter() - t2, 4)
        t3 = time.perf_counter()
        sbx.beta_pause()
        rec["pause_s"] = round(time.perf_counter() - t3, 4)
        time.sleep(RESUME_WAIT_S)
        t4 = time.perf_counter()
        sbx = host.connect(sbx.sandbox_id)
        rec["resume_s"] = round(time.perf_counter() - t4, 4)
        t5 = time.perf_counter()
        after = run_cell(sbx, "print(x)")
        rec["cell_after_resume_s"] = round(time.perf_counter() - t5, 4)
        rec["state_kept"] = after == "42"
        if not rec["state_kept"]:
            raise RuntimeError(f"x after resume is {after!r}, not '42'")
    except Exception as exc:  # noqa: BLE001 - recorded; cleanup still runs
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["sandbox_deleted"] = kill(sbx)
    return rec


def reservation(host: Host, k: int) -> dict:
    out: dict = {"before": meminfo(), "steps": []}
    live = []
    try:
        for i in range(1, k + 1):
            t0 = time.perf_counter()
            try:
                sbx = host.create()
            except Exception as exc:  # noqa: BLE001 - a refused create ends the probe; it is the observation
                out["refused"] = {"at": i, "seconds": round(time.perf_counter() - t0, 3),
                                  "message": f"{type(exc).__name__}: {str(exc)[:300]}"}
                break
            live.append(sbx)
            seconds = round(time.perf_counter() - t0, 3)
            out["steps"].append({"sandboxes": i, "create_s": seconds, "check_ok": run_cell(sbx, "print(1 + 1)") == "2",
                                 **meminfo()})
    except Exception as exc:  # noqa: BLE001 - recorded; cleanup still runs
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out["sandboxes_deleted"] = all([kill(s) for s in live])
        time.sleep(5)  # pages return to the pool once the Firecracker processes have exited
        out["after"] = meminfo()
    return out


def leftovers(hosts: dict[str, Host]) -> dict:
    out = {}
    for h in hosts.values():
        try:
            out[h.name] = len(h.live())
        except Exception as exc:  # noqa: BLE001 - recorded, and fails the run
            out[h.name] = f"{type(exc).__name__}: {exc}"
    return out


def failed(doc: dict) -> bool:
    def bad(x) -> bool:
        if isinstance(x, dict):
            if "error" in x or x.get("sandbox_deleted") is False or x.get("sandboxes_deleted") is False \
                    or x.get("conversation_deleted") is False or x.get("conversations_deleted") is False:
                return True
            return any(bad(v) for k, v in x.items() if k not in ("after_loss", "after_restart"))
        if isinstance(x, list):
            return any(bad(v) for v in x)
        return False
    if doc["mode"] == "agent":
        return bad(doc["runs"]) or not all(r.get("state_kept") for r in doc["runs"])
    if doc["mode"] == "perf":
        rec = doc.get("recovery")
        return bad(doc) or not doc["warm"]["remembered"] or any(lv["answered"] != lv["sessions_at_once"]
                                                                 for lv in doc["at_once"]) \
            or (rec is not None and not (rec.get("followup_on_live_session") or {}).get("state_preserved"))
    if doc["mode"] == "scale":
        return bad(doc) or doc["at_once"]["answered"] != doc["at_once"]["sessions_at_once"]
    if doc["mode"] == "host":
        res = doc.get("reservation")
        return bad(doc) or "guest_kernel" not in doc["isolation"] \
            or not all(r.get("state_kept") for r in doc["bench"]["runs"]) \
            or (res is not None and not (res["steps"] and all(s["check_ok"] for s in res["steps"])))
    # after_loss and after_restart also hold the failures the stop is expected to cause (the stopped host's create and
    # connects, its running sandbox, another host connecting its paused sandbox); bad() skips them, and only what
    # must survive is required below.
    hl = doc["hostloss"]
    lose = hl.get("lost_host")
    others = [v for n, v in hl.get("after_loss", {}).items() if n != lose]
    back = hl.get("after_restart", {}).get(lose, {})

    def ok_with(rec, word: str | None = None) -> bool:
        return isinstance(rec, dict) and rec.get("ok") is True and (word is None or word in str(rec.get("value")))

    kept = [ok_with(v.get("live_session_read"), "LANTERN-7") and ok_with(v.get("new_session")) for v in others]
    kept += [ok_with(back.get("resume_own_paused_sandbox"), f"PAUSED-{lose}"), ok_with(back.get("new_session"))]
    sessions = [v["new_session"].get("value") or {} for part in ("after_loss", "after_restart")
                for v in hl.get(part, {}).values() if isinstance(v.get("new_session"), dict)]
    return bad(hl) or not all(kept) or hl["stop"]["exit_code"] != 0 or hl["start"]["exit_code"] != 0 \
        or hl["start"]["health_up"]["status"] != 200 or any(v != 0 for v in hl["left_after_cleanup"].values()) \
        or any(s.get("sandbox_deleted") is False or s.get("conversation_deleted") is False for s in sessions)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, text in (("agent", "the 3-run create/answer/pause/resume protocol"),
                       ("perf", "cold, warm, at-once and sandbox replacement on one host"),
                       ("scale", "sessions at the same time across every host in --hosts"),
                       ("hostloss", "stop one host and record what the others and the paused sandboxes can do"),
                       ("host", "on the E2B host, no model: VMM and guest read-back, microbenchmark, hugepages")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--hosts", type=Path, help="JSON file naming several Embed installs (see the module help)")
        p.add_argument("--out-dir", type=Path, default=ROOT / "evidence-local")
        p.add_argument("--executor", choices=[EXECUTOR], default=EXECUTOR,
                       help="how the tool runs the model's code; only code-interpreter (see the module help)")
        if name in ("agent", "perf", "host"):
            p.add_argument("--host", help="host name from --hosts to run on (default: the first)")
        if name == "host":
            p.add_argument("--bench", type=int, default=20, help="sandboxes in the microbenchmark (default 20)")
            p.add_argument("--reserve", type=int, default=0,
                           help="create up to this many sandboxes, reading /proc/meminfo after each (default 0: skip)")
        if name == "agent":
            p.add_argument("--runs", type=int, default=3)
        if name == "perf":
            p.add_argument("--cold", type=int, default=5)
            p.add_argument("--scale", default="4,8", help="comma-separated levels, sessions at once (default 4,8)")
            p.add_argument("--recovery", action="store_true", help="kill a sandbox and time its replacement")
        if name == "scale":
            p.add_argument("--per-host", type=int, default=8, help="sessions at once on each host (default 8)")
        if name == "hostloss":
            p.add_argument("--lose", required=True, help="host name to stop")
            p.add_argument("--stop-cmd", required=True, help="command that stops the E2B stack on that host")
            p.add_argument("--start-cmd", required=True, help="command that starts it again")
    args = ap.parse_args(argv)
    for key in ("runs", "cold", "per_host"):
        if getattr(args, key, 1) < 1:
            ap.error(f"--{key.replace('_', '-')} must be >= 1")
    for key in ("bench", "reserve"):
        if getattr(args, key, 0) < 0:
            ap.error(f"--{key} must be >= 0")
    if args.mode == "perf":
        try:
            args.levels = [int(x) for x in args.scale.split(",") if x.strip()]
        except ValueError:
            ap.error("--scale must be comma-separated integers")
        if any(n < 1 for n in args.levels):
            ap.error("--scale levels must be >= 1")
    if args.mode in ("scale", "hostloss") and args.hosts is None:
        ap.error(f"{args.mode} needs --hosts")
    return args


def host_main(args: argparse.Namespace, doc: dict, hosts: dict[str, Host], host: Host, out_path: Path) -> int:
    try:
        doc["isolation"] = isolation(host)
        doc["bench"] = {"resume_wait_s": RESUME_WAIT_S, "runs": [bench_once(host, i + 1) for i in range(args.bench)]}
        if args.reserve:
            doc["reservation"] = reservation(host, args.reserve)
    except Exception as exc:  # noqa: BLE001 - recorded in the output file
        doc["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        doc["leftover_sandboxes"] = leftovers({n: h for n, h in hosts.items() if h.health() == 200})
        doc["finished_at"] = now_iso()
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"E2B {args.mode} -> {out_path}")
    left_ok = all(v == 0 for v in doc["leftover_sandboxes"].values())
    return 1 if "error" in doc or not left_ok or failed(doc) else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        endpoint = model = None
        if args.mode != "host":
            endpoint, model = need_env("PROJECT_ENDPOINT"), need_env("ACA_MODEL")
        sdk = load_sdk()
        hosts = load_hosts(args.hosts, sdk)
        pick = getattr(args, "host", None) or next(iter(hosts))
        if pick not in hosts or (args.mode == "hostloss" and args.lose not in hosts):
            raise SettingMissing(f"unknown host; known: {sorted(hosts)}")
    except SettingMissing as exc:
        print(f"SETTING MISSING {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"E2B MEASURE FAIL {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"e2b-{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    doc: dict = {"mode": args.mode, "started_at": now_iso(), "executor": EXECUTOR, "template": TEMPLATE,
                 "sdk": sdk_versions(), "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "hosts": {n: {"api_url": h.api_url, "sandbox_url": h.sandbox_url} for n, h in hosts.items()}}
    if args.mode == "host":
        return host_main(args, doc, hosts, hosts[pick], out_path)
    doc.update(model=model, agent={"name": AGENT_NAME, "instructions": INSTRUCTIONS, "tools": TOOLS_SPEC},
               inputs={"first": FIRST_Q, "count": COUNT_Q, "cold": COLD_Q, "warm": WARM_QS,
                       "scale": [SCALE_WRITE, SCALE_READ], "live": [LIVE_WRITE, LIVE_READ],
                       "paused": [PAUSE_WRITE, PAUSE_READ], "resume_wait_s": RESUME_WAIT_S})
    fd, version = None, None
    try:
        fd = Foundry(endpoint, token())
        version = fd.create_agent(model)
        doc["agent"]["version"] = version
        host = hosts[pick]
        if args.mode == "agent":
            doc["runs"] = [agent_run(fd, version, host, i + 1) for i in range(args.runs)]
        elif args.mode == "perf":
            doc["cold"] = [cold_once(fd, version, host, i + 1) for i in range(args.cold)]
            doc["warm"] = warm_session(fd, version, host)
            doc["at_once"] = [at_once(fd, version, [(host, k) for k in range(1, n + 1)]) for n in args.levels]
            if args.recovery:
                doc["recovery"] = recovery(fd, version, host)
        elif args.mode == "scale":
            names = list(hosts)
            slots = [(hosts[names[k % len(names)]], k + 1) for k in range(args.per_host * len(names))]
            doc["at_once"] = at_once(fd, version, slots)
        else:
            doc["hostloss"] = hostloss(fd, version, hosts, args.lose, args.stop_cmd, args.start_cmd)
    except Exception as exc:  # noqa: BLE001 - recorded in the output file
        doc["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if fd is not None and version is not None:
            try:
                fd.delete_agent(version)
                doc["agent"]["deleted"] = True
            except Exception as exc:  # noqa: BLE001 - recorded, and fails the run
                doc["agent"]["deleted"] = False
                doc["agent"]["delete_error"] = str(exc)[:200]
        reachable = {n: h for n, h in hosts.items() if h.health() == 200}
        doc["leftover_sandboxes"] = leftovers(reachable)
        doc["finished_at"] = now_iso()
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"E2B {args.mode} -> {out_path}")
    left_ok = all(v == 0 for v in doc["leftover_sandboxes"].values())
    if "error" in doc or not doc["agent"].get("deleted") or not left_ok or failed(doc):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
