#!/usr/bin/env python3
"""Path E check: E2B's official code-interpreter template on your E2B Embed install.

    python tools/measure_code_interpreter.py

Build the template first with E2B's unchanged script (e2b-dev/code-interpreter, template/build_prod.py) against
the same install; on E2B Embed it needs more hugepages, build disk and host disk than the defaults (see
evidence/runs/e-e2b-2026-10-15/latest.json). Then this tool, with e2b-code-interpreter 2.11.0, runs once:

  1. Sandbox.create() with no template argument (the SDK default, code-interpreter-v1); get_info() reads back
     the template the API used and its vCPU and memory
  2. run_code("x = 1"), then run_code("x+=1; x") -> "2" (state kept between calls)
  3. pandas: int(DataFrame sum) -> "10"
  4. matplotlib: a plot; a PNG result must come back
  5. JavaScript kernel: run_code("[1, 2, 3].reduce((a, b) => a + b)", language="javascript") -> "6"
  6. run_code("1/0") -> error ZeroDivisionError
  7. pause(), wait 5 s, Sandbox.connect, then run_code("x") -> "2" (kernel memory kept across the snapshot)
  8. kill, wait 3 s, then list running and paused sandboxes: none may be left

Every step is timed with time.perf_counter() around its SDK call and records what it observed; every run_code step
also records stdout, stderr and the full error (name, value, traceback) under "execution". These are the steps,
checks and record of the 2026-10-15 check runs 2-4 (run 1 did not record "execution").

Reads E2B_API_URL, E2B_SANDBOX_URL and E2B_API_KEY; the key is never written. Writes
evidence-local/ci-test-<UTC time>.json. Exit codes: 0 every step passed and no sandbox was left, 1 a step failed or
a sandbox was left, 2 a setting is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RESUME_WAIT_S = 5
LEFTOVER_WAIT_S = 3
JS_CODE = "[1, 2, 3].reduce((a, b) => a + b)"


class SettingMissing(RuntimeError):
    pass


def need_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val or val.startswith("YOUR-"):
        raise SettingMissing(f"missing or placeholder environment variable: {name}")
    return val


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def load_sdk():
    try:
        from e2b import SandboxQuery, SandboxState
        from e2b_code_interpreter import Sandbox
        return Sandbox, SandboxQuery, SandboxState
    except ImportError as exc:
        raise RuntimeError("Install the measured SDKs first: pip install e2b==2.56.0 e2b-code-interpreter==2.11.0") \
            from exc


def text(execution) -> str:
    return (execution.text or "").strip()


def execution_detail(value) -> dict | None:
    """stdout, stderr and the full error of a run_code result; None for any other value."""
    if not (hasattr(value, "logs") and hasattr(value, "results")):
        return None
    detail = {"stdout": [s[:300] for s in value.logs.stdout][:10], "stderr": [s[:300] for s in value.logs.stderr][:10]}
    if value.error is not None:
        detail["error"] = {"name": value.error.name, "value": ANSI.sub("", value.error.value or "")[:500],
                           "traceback": ANSI.sub("", value.error.traceback or "")[:2000]}
    return detail


class Run:
    def __init__(self, sdk, opts: dict):
        self.Sandbox, self.Query, self.State = sdk
        self.opts = opts
        self.steps: list[dict] = []

    def step(self, name: str, fn, check):
        t0 = time.perf_counter()
        rec: dict = {"step": name, "started_utc": now()}
        value = None
        try:
            value = fn()
            rec["seconds"] = round(time.perf_counter() - t0, 3)
            ok, observed = check(value)
            rec.update(ok=bool(ok), observed=observed)
            detail = execution_detail(value)
            if detail is not None:
                rec["execution"] = detail
        except Exception as exc:  # noqa: BLE001 - the failure is the observation
            rec.update(seconds=round(time.perf_counter() - t0, 3), ok=False,
                       error=f"{type(exc).__name__}: {str(exc)[:400]}")
        self.steps.append(rec)
        print(json.dumps(rec, ensure_ascii=False)[:300], flush=True)
        return value if rec.get("ok") else None

    def live(self) -> int:
        pager = self.Sandbox.list(query=self.Query(state=[self.State.RUNNING, self.State.PAUSED]), **self.opts)
        n = 0
        while pager.has_next:
            n += len(pager.next_items(**self.opts))
        return n

    def created(self, s) -> tuple[bool, dict]:
        info = s.get_info()
        return bool(s.sandbox_id), {"sdk_default_template": self.Sandbox.default_template,
                                    "template_id": info.template_id, "template_name": info.name,
                                    "cpu_count": info.cpu_count, "memory_mb": info.memory_mb}

    def run(self) -> dict:
        doc: dict = {}
        made: dict = {}

        def create():
            made["sbx"] = self.Sandbox.create(timeout=600, **self.opts)
            return made["sbx"]

        # A sandbox whose create check failed is still killed: the check, not the create, may be what failed.
        sbx = self.step("create (default template)", create, self.created)
        try:
            if sbx is not None:
                self.step("README: x = 1", lambda: sbx.run_code("x = 1"),
                          lambda e: (e.error is None, {"text": text(e)}))
                self.step("README: x+=1; x", lambda: sbx.run_code("x+=1; x"),
                          lambda e: (text(e) == "2", {"text": text(e)}))
                self.step("pandas sum", lambda: sbx.run_code(
                    "import pandas as pd\nint(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).values.sum())"),
                    lambda e: (text(e) == "10", {"text": text(e), "error": e.error.name if e.error else None}))
                self.step("matplotlib png", lambda: sbx.run_code(
                    "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.show()"),
                    lambda e: (any(r.png for r in e.results), {"results": [r.formats() for r in e.results],
                                                                "png_bytes_base64": [len(r.png or "") for r in e.results]}))
                self.step("javascript kernel", lambda: sbx.run_code(JS_CODE, language="javascript"),
                          lambda e: (text(e) == "6", {"text": text(e), "error": e.error.name if e.error else None}))
                self.step("exception", lambda: sbx.run_code("1/0"),
                          lambda e: (e.error is not None and e.error.name == "ZeroDivisionError",
                                     {"error": e.error.name if e.error else None}))
                sid = sbx.sandbox_id
                self.step("pause", lambda: sbx.pause(), lambda r: (r is True, {"returned": r}))
                time.sleep(RESUME_WAIT_S)
                resumed = self.step("connect (resume)", lambda: self.Sandbox.connect(sid, **self.opts),
                                    lambda s: (s.sandbox_id == sid, {}))
                if resumed is not None:
                    sbx = resumed
                    self.step("x after resume", lambda: sbx.run_code("x"),
                              lambda e: (text(e) == "2", {"text": text(e)}))
        finally:
            target = sbx if sbx is not None else made.get("sbx")
            if target is not None:
                self.step("kill", lambda: target.kill(), lambda r: (r is True, {"returned": r}))
            time.sleep(LEFTOVER_WAIT_S)
            try:
                doc["leftover_sandboxes"] = self.live()
            except Exception as exc:  # noqa: BLE001 - recorded, and the run then fails the leftover check
                doc["leftover_sandboxes"], doc["leftover_error"] = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        doc["steps"] = self.steps
        return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evidence-local")
    args = ap.parse_args(argv)
    try:
        opts = {"api_url": need_env("E2B_API_URL"), "sandbox_url": need_env("E2B_SANDBOX_URL"),
                "api_key": need_env("E2B_API_KEY")}
        sdk = load_sdk()
    except SettingMissing as exc:
        print(f"SETTING MISSING {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"CODE INTERPRETER CHECK FAIL {exc}", file=sys.stderr)
        return 1
    doc: dict = {"started_at": now(), "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "python": sys.version.split()[0], "api_url": opts["api_url"]}
    try:
        from importlib.metadata import version
        doc["packages"] = {p: version(p) for p in ("e2b", "e2b-code-interpreter")}
    except Exception:  # noqa: BLE001 - packaging metadata is informative only
        doc["packages"] = {}
    doc.update(Run(sdk, opts).run())
    doc["finished_at"] = now()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"ci-test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    passed = all(s.get("ok") for s in doc["steps"]) and doc["leftover_sandboxes"] == 0
    print(f"CODE INTERPRETER CHECK {'PASS' if passed else 'FAIL'} "
          f"{sum(bool(s.get('ok')) for s in doc['steps'])}/{len(doc['steps'])} steps -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
