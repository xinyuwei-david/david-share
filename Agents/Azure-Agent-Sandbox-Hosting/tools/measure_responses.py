#!/usr/bin/env python3
"""Measure Path B through its public Responses endpoint.

    python tools/measure_responses.py --cold 3 --warm-turns 3
    python tools/measure_responses.py --cold 3 --warm-turns 3 --scale 4,8

AGENT_RESPONSES_URL comes from .env. Azure CLI acquires a token outside every request timer,
refreshing it before each concurrency level. Follow-ups reuse previous_response_id.
These inputs are a new reproducible protocol; the historical B request text was not retained.
Writes evidence-local/responses-<UTC time>.json. Exit 0 requires every requested turn to complete
and each code word to be recalled exactly; exit 1 records failures; exit 2 means missing settings
or invalid arguments. No cluster, pod or private-preview operation is performed.
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


def setting(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or "YOUR-" in value:
        raise SettingMissing(name)
    return value


def token() -> str:
    out = subprocess.run(["az", "account", "get-access-token", "--resource", "https://ai.azure.com",
                          "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True,
                         shell=(os.name == "nt"), timeout=120)
    if out.returncode or not out.stdout.strip():
        raise RuntimeError(f"TOKEN FAILED: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def answered(turn: dict | None) -> bool:
    return bool(turn and turn.get("http") == 200 and turn.get("status") == "completed"
                and turn.get("response_id") and isinstance(turn.get("answer"), str)
                and turn["answer"].strip() and not turn.get("error"))


def invoke(endpoint: str, tok: str, text: str, previous: str | None = None) -> dict:
    body = {"input": text}
    if previous:
        body["previous_response_id"] = previous
    request = urllib.request.Request(
        endpoint, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    start = time.perf_counter()
    record = {"started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(request, timeout=420) as result:
            first_byte = time.perf_counter() - start
            doc = json.loads(result.read())
            answer = " ".join(content.get("text", "") for item in doc.get("output", [])
                              if item.get("type") == "message" for content in item.get("content", []))
            record.update(http=result.status, status=doc.get("status"), response_id=doc.get("id"),
                          answer=answer[:80], first_byte_s=round(first_byte, 3), error=None)
    except urllib.error.HTTPError as exc:
        record.update(http=exc.code, status=None, response_id=None, answer=None, first_byte_s=None,
                      error=exc.read().decode(errors="replace")[:300])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        record.update(http=None, status=None, response_id=None, answer=None, first_byte_s=None,
                      error=f"{type(exc).__name__}: {exc}"[:300])
    record["wall_s"] = round(time.perf_counter() - start, 3)
    return record


def run_cold(endpoint: str, tok: str, count: int) -> list[dict]:
    turns = []
    for i in range(1, count + 1):
        turn = invoke(endpoint, tok, f"Reply with exactly COLD-{i}")
        turns.append(turn)
        print(f"cold {i}/{count}: http={turn['http']} wall={turn['wall_s']}s -> {turn['answer']}", flush=True)
    return turns


def run_warm(endpoint: str, tok: str, count: int) -> dict:
    first = invoke(endpoint, tok, "Remember the code word LANTERN-7. Reply only OK.")
    turns = [first]
    if not answered(first):
        return {"turns": turns, "remembered": False}
    previous = first["response_id"]
    for i in range(count):
        text = "What is the code word? Reply with the word only." if i == 0 else f"Reply with exactly WARM-{i}"
        turn = invoke(endpoint, tok, text, previous)
        turns.append(turn)
        print(f"warm {i + 1}/{count}: http={turn['http']} wall={turn['wall_s']}s -> {turn['answer']}", flush=True)
        if not answered(turn):
            break
        previous = turn["response_id"]
    return {"turns": turns, "remembered": len(turns) > 1 and answered(turns[1])
            and turns[1]["answer"].strip() == "LANTERN-7"}


def run_scale(endpoint: str, tok: str, count: int) -> dict:
    def one(i: int) -> dict:
        first = invoke(endpoint, tok, f"Remember SCALE-{i}. Reply only OK.")
        followup = invoke(endpoint, tok, "What did I ask you to remember? Word only.",
                          first["response_id"]) if answered(first) else None
        return {"i": i, "t1": first, "t2": followup, "remembered": followup is not None
                and answered(followup) and followup["answer"].strip() == f"SCALE-{i}"}

    with cf.ThreadPoolExecutor(max_workers=count) as executor:
        sessions = list(executor.map(one, range(count)))
    passed = sum(session["remembered"] for session in sessions)
    print(f"concurrency {count}: recalled {passed}/{count}", flush=True)
    return {"concurrency": count, "sessions": sessions}


def report_passes(report: dict) -> bool:
    if report.get("error"):
        return False
    cold, warm = report["cold"], report["warm"]
    if not cold or not warm["remembered"] or not all(answered(turn) for turn in cold + warm["turns"]):
        return False
    requested = report["requested"]
    if len(cold) != requested["cold"] or len(warm["turns"]) != requested["warm_turns"] + 1:
        return False
    if any(turn["answer"].strip() != f"COLD-{i + 1}" for i, turn in enumerate(cold)):
        return False
    expected_warm = ["OK", "LANTERN-7"] + [f"WARM-{i}" for i in range(1, requested["warm_turns"])]
    if [turn["answer"].strip() for turn in warm["turns"]] != expected_warm:
        return False
    if [level["concurrency"] for level in report.get("scale", [])] != requested["scale"]:
        return False
    return all(len(level["sessions"]) == level["concurrency"]
               and all(answered(session["t1"]) and answered(session["t2"]) and session["remembered"]
                       for session in level["sessions"]) for level in report.get("scale", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--cold", type=int, default=3)
    parser.add_argument("--warm-turns", type=int, default=3)
    parser.add_argument("--scale", default="", help="comma-separated positive concurrency levels")
    parser.add_argument("--pause", type=int, default=20, help="seconds between phases")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evidence-local")
    args = parser.parse_args(argv)
    if args.cold < 1 or args.warm_turns < 1 or args.pause < 0:
        parser.error("--cold and --warm-turns must be positive; --pause must be non-negative")
    try:
        levels = [int(value) for value in args.scale.split(",") if value.strip()]
    except ValueError:
        parser.error("--scale must be a comma list of positive integers")
    if any(level < 1 for level in levels):
        parser.error("--scale must be a comma list of positive integers")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "requested": {"cold": args.cold, "warm_turns": args.warm_turns, "scale": levels}}
    try:
        endpoint = setting("AGENT_RESPONSES_URL")
        auth = token()
        report["cold"] = run_cold(endpoint, auth, args.cold)
        time.sleep(args.pause)
        report["warm"] = run_warm(endpoint, auth, args.warm_turns)
        for level in levels:
            time.sleep(args.pause)
            report.setdefault("scale", []).append(run_scale(endpoint, token(), level))
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
    print(f"{'MEASURE PASS' if ok else 'MEASURE FAIL'} -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
