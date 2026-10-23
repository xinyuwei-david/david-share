#!/usr/bin/env python3
"""Path A measurement: repeat the 2026-10-09 protocol against your own sandbox group.

    python tools/measure_aca.py agent  --runs 3                      # full agent path, as in the README
    python tools/measure_aca.py agent  --runs 3 --config novolume    # the same without the Blob volume
    python tools/measure_aca.py create --runs 2 --config image,full  # create time by configuration
    python tools/measure_aca.py scale  --scale 4,8                   # N agent sandboxes at once

agent mode, per run: create (full spec; with --config novolume, the same spec without the Blob volume) -> poll
to Running -> first answer -> follow-up -> stop -> poll to Stopped -> wait 5 s -> resume -> poll to Running ->
answer -> answer -> delete. Answers are timed inside the sandbox (POST /reflect from sandbox exec); create, stop
and resume are SDK wall time on this machine.
create mode, per run: create -> poll to Running -> stop -> resume -> delete, recording create time and whether
the guest uptime after resume exceeds its earlier value. Uptime alone is not proof of process-memory continuity.
scale mode, per level N: N sandboxes (full spec) are created at the same time; each asks "Remember SCALE-<k>.
Reply only OK." then "What did I ask you to remember? Word only." (D's and E's concurrency inputs) and is counted
as answered when the second reply contains SCALE-<k>; every sandbox is deleted before the next level.
--disk-image-id overrides the disk image recorded in aca-state.json.

Writes evidence-local/aca-<mode>-<UTC time>.json (Git ignores evidence-local/). Exit codes: 0 every run
completed (and agent mode retained its history), 1 a run recorded an error or left a sandbox behind,
2 a setting is missing or arguments are invalid.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aca_sandbox as aca  # noqa: E402

FIRST = "Where is Bangkok? Answer in one short sentence."
COUNT = "How many questions have I asked before this one? Number only."
SCALE_WRITE = "Remember SCALE-{k}. Reply only OK."
SCALE_READ = "What did I ask you to remember? Word only."


AGENT_CONFIGS = ("full", "novolume")


def agent_run(group, cfg: dict, disk_image_id: str, index: int, config: str = "full") -> dict:
    rec: dict = {"run": index, "config": config, "started_at": aca.now()}
    sandbox = None
    try:
        env = aca.agent_environment(cfg, agent_id=f"agent-host-measure-{index}-{int(time.time())}")
        kwargs = aca.config_kwargs(config, cfg, disk_image_id, env)
        t0 = time.perf_counter()
        sandbox, initial = group._do_create_sandbox(**kwargs)
        rec["put_s"] = round(time.perf_counter() - t0, 2)
        _, running, polls = aca.poll_state(lambda: group.get_sandbox(initial.id), ("running",))
        rec["create_total_s"] = round(time.perf_counter() - t0, 2)
        rec["initial_state"], rec["polls"], rec["vmm_type"] = initial.state, polls, running.vmm_type
        rec["first"] = aca.ask(sandbox, FIRST)
        rec["cold_total_s"] = round(rec["create_total_s"] + rec["first"]["ready_wait_s"] + rec["first"]["guest_s"], 2)
        rec["warm"] = aca.ask(sandbox, COUNT)
        t1 = time.perf_counter()
        sandbox.stop()
        rec["stop_post_s"] = round(time.perf_counter() - t1, 2)
        _, stopped, _ = aca.poll_state(sandbox.get, ("stopped", "suspended", "idle"))
        rec["stop_total_s"], rec["state_after_stop"] = round(time.perf_counter() - t1, 2), stopped.state
        time.sleep(5)
        t2 = time.perf_counter()
        sandbox.resume()
        rec["resume_post_s"] = round(time.perf_counter() - t2, 2)
        _, resumed, _ = aca.poll_state(sandbox.get, ("running",))
        rec["resume_total_s"], rec["state_after_resume"] = round(time.perf_counter() - t2, 2), resumed.state
        rec["first_after_resume"] = aca.ask(sandbox, COUNT)
        after = rec["first_after_resume"]
        rec["recover_total_s"] = round(rec["resume_total_s"] + after["ready_wait_s"] + after["guest_s"], 2)
        rec["warm_after_resume"] = aca.ask(sandbox, COUNT)
        lengths = [rec[key]["history_length"] for key in ("first", "warm", "first_after_resume", "warm_after_resume")]
        rec["state_kept"] = all(isinstance(length, int) for length in lengths) and all(
            after > before for before, after in zip(lengths, lengths[1:]))
        if not rec["state_kept"]:
            raise RuntimeError(f"conversation history did not keep growing across resume: {lengths}")
    except Exception as exc:  # noqa: BLE001 - recorded, the run is marked failed
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            try:
                t3 = time.perf_counter()
                sandbox.delete()
                rec["delete_post_s"] = round(time.perf_counter() - t3, 2)
            except Exception as exc:  # noqa: BLE001
                rec["delete_error"] = f"{type(exc).__name__}: {exc}"
        rec["finished_at"] = aca.now()
    print(json.dumps(rec, default=str), flush=True)
    return rec


def uptime(sandbox) -> float:
    out = sandbox.exec("cat /proc/uptime")
    if getattr(out, "exit_code", 0) or not out.stdout:
        raise RuntimeError(f"guest uptime was not read: {getattr(out, 'stderr', '')}")
    return float(out.stdout.split()[0])


def create_run(group, cfg: dict, disk_image_id: str | None, name: str, index: int) -> dict:
    rec: dict = {"config": name, "run": index, "started_at": aca.now()}
    sandbox = None
    try:
        env = None if name == "builtin" else aca.agent_environment(cfg)
        kwargs = aca.config_kwargs(name, cfg, disk_image_id, env)
        t0 = time.perf_counter()
        sandbox, initial = group._do_create_sandbox(**kwargs)
        rec["put_s"] = round(time.perf_counter() - t0, 2)
        _, running, polls = aca.poll_state(lambda: group.get_sandbox(initial.id), ("running",))
        rec["create_total_s"], rec["polls"], rec["vmm_type"] = round(time.perf_counter() - t0, 2), polls, running.vmm_type
        before = uptime(sandbox)
        t1 = time.perf_counter()
        sandbox.stop()
        _, stopped, _ = aca.poll_state(sandbox.get, ("stopped", "suspended", "idle"))
        rec["stop_total_s"], rec["state_after_stop"] = round(time.perf_counter() - t1, 2), stopped.state
        time.sleep(5)
        t2 = time.perf_counter()
        sandbox.resume()
        aca.poll_state(sandbox.get, ("running",))
        rec["resume_total_s"] = round(time.perf_counter() - t2, 2)
        after = uptime(sandbox)
        rec["uptime_before"], rec["uptime_after_resume"] = before, after
        rec["uptime_increased"] = rec["process_survived"] = after > before
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
            except Exception as exc:  # noqa: BLE001
                rec["delete_error"] = f"{type(exc).__name__}: {exc}"
        rec["finished_at"] = aca.now()
    print(json.dumps(rec, default=str), flush=True)
    return rec


def scale_session(group, cfg: dict, disk_image_id: str, level: int, k: int) -> dict:
    rec: dict = {"slot": k, "started_at": aca.now()}
    sandbox = None
    t_all = time.perf_counter()
    try:
        env = aca.agent_environment(cfg, agent_id=f"agent-host-scale-{level}-{k}-{int(time.time())}")
        kwargs = aca.config_kwargs("full", cfg, disk_image_id, env)
        t0 = time.perf_counter()
        sandbox, initial = group._do_create_sandbox(**kwargs)
        _, running, _ = aca.poll_state(lambda: group.get_sandbox(initial.id), ("running",))
        rec["create_to_running_s"], rec["vmm_type"] = round(time.perf_counter() - t0, 2), running.vmm_type
        rec["turn1"] = aca.ask(sandbox, SCALE_WRITE.format(k=k))
        rec["turn2"] = aca.ask(sandbox, SCALE_READ)
        rec["remembered"] = f"SCALE-{k}" in rec["turn2"]["response"]
    except Exception as exc:  # noqa: BLE001 - recorded, the slot counts as not answered
        rec.update(error=f"{type(exc).__name__}: {exc}", remembered=False)
    finally:
        rec["session_s"] = round(time.perf_counter() - t_all, 2)
        if sandbox is not None:
            try:
                sandbox.delete()
                rec["deleted"] = True
            except Exception as exc:  # noqa: BLE001
                rec["delete_error"] = f"{type(exc).__name__}: {exc}"
        rec["finished_at"] = aca.now()
    return rec


def at_once(group, cfg: dict, disk_image_id: str, level: int) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as pool:
        sessions = list(pool.map(lambda k: scale_session(group, cfg, disk_image_id, level, k), range(level)))
    out = {"sessions_at_once": level, "answered": sum(bool(s.get("remembered")) for s in sessions),
           "wall_s": round(time.perf_counter() - t0, 2), "sessions": sessions}
    print(json.dumps({k: v for k, v in out.items() if k != "sessions"}), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("mode", choices=("agent", "create", "scale"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--scale", default="4,8", help="scale mode: comma list of sandboxes at once")
    ap.add_argument("--disk-image-id", help="disk image to use instead of the one in aca-state.json")
    ap.add_argument("--config", default=None,
                    help=f"create mode: comma list from {', '.join(aca.CONFIGS)} (default builtin,image,full); "
                         f"agent mode: one of {', '.join(AGENT_CONFIGS)} (default full)")
    ap.add_argument("--out-dir", type=Path, default=aca.ROOT / "evidence-local")
    args = ap.parse_args(argv)
    default = "builtin,image,full" if args.mode == "create" else "full"
    configs = [c.strip() for c in (args.config or default).split(",") if c.strip()]
    if args.runs < 1:
        ap.error("--runs must be positive")
    if args.mode == "create" and (not configs or any(c not in aca.CONFIGS for c in configs)):
        ap.error(f"--config must be a comma list from {', '.join(aca.CONFIGS)}")
    if args.mode == "agent" and (len(configs) != 1 or configs[0] not in AGENT_CONFIGS):
        ap.error(f"agent mode: --config must be one of {', '.join(AGENT_CONFIGS)}")
    try:
        levels = [int(x) for x in args.scale.split(",") if x.strip()]
    except ValueError:
        ap.error("--scale must be a comma list of positive integers")
    if args.mode == "scale" and (not levels or any(n < 1 for n in levels)):
        ap.error("--scale must be a comma list of positive integers")
    try:
        cfg = aca.settings()
        disk_image_id = args.disk_image_id or aca.load_state().get("disk_image_id")
        if (args.mode in ("agent", "scale") or any(c != "builtin" for c in configs)) and not disk_image_id:
            raise aca.SettingMissing("disk image (run tools/aca_sandbox.py image first)")
        group = aca.group_client(cfg)
        existing = {s.id for s in group.list_sandboxes()}
        out = {"meta": {"date_utc": aca.now(), "mode": args.mode, "region": cfg["region"], "runs": args.runs,
                        "configs": configs if args.mode != "scale" else ["full"],
                        "inputs": ({"first": FIRST, "follow_up": COUNT} if args.mode == "agent" else
                                   {"write": SCALE_WRITE, "read": SCALE_READ} if args.mode == "scale" else None)},
               "runs": []}
        if args.mode == "scale":
            out["meta"]["levels"] = levels
            for n, level in enumerate(levels):
                if n:
                    time.sleep(20)
                out.setdefault("scale", []).append(at_once(group, cfg, disk_image_id, level))
        for i in range(1, args.runs + 1) if args.mode != "scale" else ():
            if args.mode == "agent":
                out["runs"].append(agent_run(group, cfg, disk_image_id, i, configs[0]))
            else:
                for name in configs:
                    out["runs"].append(create_run(group, cfg, disk_image_id, name, i))
            time.sleep(3)
        out["leftover_sandboxes"] = len([s for s in group.list_sandboxes() if s.id not in existing])
    except aca.SettingMissing as exc:
        print(f"SETTING MISSING {exc}: set it in .env (see config/.env.example)", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out_dir / f"aca-{args.mode}-{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    failed = [r for r in out["runs"] if "error" in r or "delete_error" in r]
    failed += [s for level in out.get("scale", []) for s in level["sessions"]
               if not s.get("remembered") or "delete_error" in s]
    total = len(out["runs"]) + sum(len(level["sessions"]) for level in out.get("scale", []))
    print(f"{'MEASURE FAIL' if failed or out['leftover_sandboxes'] else 'MEASURE PASS'} "
          f"{total - len(failed)}/{total} {'sessions' if args.mode == 'scale' else 'runs'} -> {path}")
    return 1 if failed or out["leftover_sandboxes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
