#!/usr/bin/env python3
"""Ask the path E application one question: a Foundry Prompt Agent whose one function tool runs the model's Python
with E2B's Code Interpreter (run_code) in a code-interpreter-v1 sandbox on your Azure VM.

    python tools/e2b_agent.py "Where is Bangkok? Answer in one short sentence."

Foundry returns a Prompt Agent's function call to the client instead of running it, so this script is that client:
it creates a temporary agent version (model ACA_MODEL, the run_in_e2b_sandbox tool of tools/measure_e2b.py), one
conversation and one sandbox, sends the question, runs every tool call in that sandbox, prints the answer, then
deletes the sandbox, the conversation and the agent version. Reads PROJECT_ENDPOINT, ACA_MODEL, E2B_API_URL,
E2B_SANDBOX_URL and E2B_API_KEY; the Foundry token comes from the same sources as in tools/measure_e2b.py.
Exit codes: 0 answered and cleaned up, 1 failed or left a resource, 2 a required setting is missing.
"""
from __future__ import annotations

import argparse
import sys

import measure_e2b as m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("question")
    args = ap.parse_args(argv)
    try:
        endpoint, model = m.need_env("PROJECT_ENDPOINT"), m.need_env("ACA_MODEL")
        host = m.load_hosts(None, m.load_sdk())["a"]
    except m.SettingMissing as exc:
        print(f"SETTING MISSING {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"E2B AGENT FAIL {exc}", file=sys.stderr)
        return 1
    fd = version = conv = sbx = None
    answered = False
    try:
        fd = m.Foundry(endpoint, m.token())
        version = fd.create_agent(model)
        conv = fd.conversation()
        sbx = host.create()
        turn = m.ask(fd, version, conv, sbx, args.question)
        print(turn["response"])
        print(f"sandbox {sbx.sandbox_id} ({m.TEMPLATE}): {turn['tool_calls']} tool call(s), "
              f"{turn['sandbox_tool_s']} s in run_code, answer after {turn['wall_s']} s", file=sys.stderr)
        answered = True
    except Exception as exc:  # noqa: BLE001 - reported, then everything created is removed
        print(f"E2B AGENT FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        cleaned = m.kill(sbx)
        if fd is not None:
            cleaned = fd.drop(conv) and cleaned
            if version is not None:
                try:
                    fd.delete_agent(version)
                except Exception as exc:  # noqa: BLE001 - a version left behind fails the run
                    print(f"E2B AGENT FAIL agent version {version} not deleted: {exc}", file=sys.stderr)
                    cleaned = False
    return 0 if answered and cleaned else 1


if __name__ == "__main__":
    raise SystemExit(main())
