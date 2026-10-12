#!/usr/bin/env python3
"""Recompute every published number from evidence/runs/ into evidence/measurements.json.

    python tools/build_measurements.py            # write evidence/measurements.json
    python tools/build_measurements.py --check    # fail if the committed file differs from a fresh build

It fails closed: a run file whose SHA-256 differs from evidence/runs/INDEX.json, a missing field,
a run that recorded an error, or a cross-check that does not hold stops the build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evidence" / "runs"
OUT = ROOT / "evidence" / "measurements.json"


class BuildError(RuntimeError):
    pass


def load_runs(runs_dir: Path = RUNS) -> tuple[dict[str, dict], dict[str, str]]:
    index = json.loads((runs_dir / "INDEX.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(runs_dir).as_posix() for p in runs_dir.rglob("*.json")} - {"INDEX.json"}
    if on_disk != set(index):
        raise BuildError(f"run files {sorted(on_disk ^ set(index))} are not in INDEX.json (or missing)")
    out = {}
    for rel, digest in index.items():
        data = (runs_dir / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise BuildError(f"{rel}: SHA-256 differs from INDEX.json")
        out[rel] = json.loads(data.decode("utf-8"))
    return out, index


def stat(values: list[float], unit: str = "s") -> dict:
    """Median of an odd sample is a sample value; an even sample keeps its values and has no median."""
    if not values:
        raise BuildError("empty sample")
    vals = sorted(values)
    rec = {"n": len(vals), "unit": unit, "values": list(values), "min": vals[0], "max": vals[-1]}
    if len(vals) % 2:
        rec["median"] = vals[len(vals) // 2]
    return rec


def single(value: float, unit: str = "s") -> dict:
    return {"n": 1, "unit": unit, "values": [value], "value": value}


def utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build(runs: dict[str, dict], index: dict[str, str]) -> dict:
    m: dict[str, dict] = {}

    def put(key: str, run_id: str, label: str, rec: dict, what: str) -> None:
        if key in m:
            raise BuildError(f"duplicate metric {key}")
        m[key] = {"run": run_id, "label": label, "what": what, **rec}

    # A · 2026-10-09, full agent, n = 3
    a = runs["a-aca-2026-10-09/agent-runs.json"]
    rid = a["run_id"]
    ar = a["runs"]
    if len(ar) != 3:
        raise BuildError("A agent run should have 3 runs")
    for r in ar:
        if "error" in r or "delete_error" in r:
            raise BuildError(f"A run {r['run']}: recorded an error")
        if abs(r["create_total_s"] + r["first"]["guest_s"] - r["cold_total_s"]) > 0.011:
            raise BuildError(f"A run {r['run']}: cold_total_s is not create + first answer")
        if abs(r["resume_total_s"] + r["first_after_resume"]["guest_s"] - r["recover_total_s"]) > 0.011:
            raise BuildError(f"A run {r['run']}: recover_total_s is not resume + answer")
    put("a.create", rid, "LOCAL_MEASUREMENT", stat([r["create_total_s"] for r in ar]),
        "create call until Running, full spec")
    put("a.cold", rid, "LOCAL_MEASUREMENT", stat([r["cold_total_s"] for r in ar]),
        "SDK create wall time + first answer timed inside the sandbox; a component sum, not client end-to-end wall time")
    put("a.first_answer", rid, "LOCAL_MEASUREMENT", stat([r["first"]["guest_s"] for r in ar]),
        "first answer only, timed inside the sandbox")
    put("a.warm", rid, "LOCAL_MEASUREMENT", stat([r["warm"]["guest_s"] for r in ar]),
        "follow-up answer in the running sandbox")
    put("a.stop", rid, "LOCAL_MEASUREMENT", stat([r["stop_total_s"] for r in ar]),
        "stop call until Stopped (memory snapshot)")
    put("a.resume", rid, "LOCAL_MEASUREMENT", stat([r["resume_total_s"] for r in ar]),
        "resume call until Running")
    put("a.first_after_resume", rid, "LOCAL_MEASUREMENT",
        stat([r["first_after_resume"]["guest_s"] for r in ar]), "first answer after resume")
    put("a.recover", rid, "LOCAL_MEASUREMENT", stat([r["recover_total_s"] for r in ar]),
        "SDK resume wall time + first answer inside the sandbox; excludes stop and idle waiting")
    put("a.warm_after_resume", rid, "LOCAL_MEASUREMENT", stat([r["warm_after_resume"]["guest_s"] for r in ar]),
        "next answer after that")
    vmm = sorted({r["vmm_type"] for r in ar})
    put("a.vmm", rid, "LOCAL_MEASUREMENT",
        {"n": len(ar), "unit": "runs", "value": vmm[0] if len(vmm) == 1 else vmm,
         "count": sum(r["vmm_type"] == "cloudhypervisor" for r in ar)}, "vmm_type the API reported")
    kept = sum(r["first_after_resume"]["history_length"] > r["warm"]["history_length"] for r in ar)
    put("a.state_kept", rid, "LOCAL_MEASUREMENT", {"n": len(ar), "unit": "runs", "count": kept},
        "conversation history length kept growing across stop and resume")

    att = runs["a-aca-2026-10-09/create-attribution.json"]
    by_cfg: dict[str, list[float]] = {}
    for r in att["runs"]:
        if "error" in r or "delete_error" in r:
            raise BuildError(f"A attribution {r['config']}/{r['run']}: recorded an error")
        if r["process_survived"] != (r["uptime_after_resume"] > r["uptime_at_exec1"]):
            raise BuildError(f"A attribution {r['config']}/{r['run']}: uptime observation differs from its recorded flag")
        by_cfg.setdefault(r["config"], []).append(r["create_total_s"])
    for cfg in ("builtin", "image", "vnet", "port", "vnetvolume", "full"):
        if cfg not in by_cfg:
            raise BuildError(f"attribution config {cfg} missing")
        put(f"a.attr.{cfg}", att["run_id"], "LOCAL_MEASUREMENT", stat(by_cfg[cfg]), att["configs"][cfg])
    small = by_cfg["image"] + by_cfg["vnet"] + by_cfg["port"]
    put("a.attr.small", att["run_id"], "LOCAL_MEASUREMENT", stat(small),
        "image alone, image + VNet, image + port: every create time")
    survived = [r["process_survived"] for r in att["runs"]]
    put("a.attr.process_survived", att["run_id"], "LOCAL_MEASUREMENT",
        {"n": len(survived), "unit": "runs", "count": sum(bool(x) for x in survived)},
        "guest uptime after resume exceeded its pre-stop value; this alone does not prove process-memory continuity")

    late = runs["a-aca-2026-10-09/late-mount.json"]
    put("a.late.create", late["run_id"], "LOCAL_MEASUREMENT", stat([r["create_total_s"] for r in late["runs"]]),
        "create image + VNet without the volume")
    put("a.late.mount", late["run_id"], "LOCAL_MEASUREMENT", stat([r["add_volume_mount_s"] for r in late["runs"]]),
        "attach the Blob volume to the running sandbox")

    setup = runs["a-aca-2026-09-27/setup-and-wake.json"]
    ev = {e["step"]: e for e in setup["events"]}
    parts = [setup["sandbox_group_deploy_seconds"]] + [ev[s]["seconds"] for s in
                                                       ("vnet_connection", "disk_image", "volume", "sandbox_create")]
    put("a.setup", setup["run_id"], "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "s", "value": round(sum(parts), 1), "parts": parts},
        "sandbox group + VNet connection + disk image + Blob volume + sandbox")
    put("a.wake_http", setup["run_id"], "LOCAL_MEASUREMENT", single(ev["wake_on_http_request"]["seconds"]),
        "HTTP request to a stopped sandbox's on-demand port until the answer")
    iso = setup["isolation_check"]
    put("a.isolation", setup["run_id"], "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "read-back", "vmm_type": iso["vmm_type"], "guest_kernel": iso["guest_kernel"]},
        "read inside the sandbox")

    # Shared foundation, B and C · 2026-09-27, one run each
    f = runs["abc-2026-09-27/foundation.json"]
    put("f.deploy", f["run_id"], "LOCAL_MEASUREMENT", single(f["deploy_minutes"], "min"), "module-01 deployment")
    b = runs["b-hosted-agent-2026-09-27/run.json"]
    rid = b["run_id"]
    put("b.init", rid, "LOCAL_MEASUREMENT", single(b["azd_ai_agent_init_seconds"]), "azd ai agent init")
    put("b.deploy", rid, "LOCAL_MEASUREMENT", single(b["deploy_v1_gateway"]["seconds"]), "azd deploy, version 1")
    put("b.deploy_v2", rid, "LOCAL_MEASUREMENT", single(b["deploy_v2_direct"]["seconds"]), "azd deploy, version 2")
    put("b.cold_gateway", rid, "LOCAL_MEASUREMENT", single(b["invoke_v1_gateway"]["new_session_seconds"]),
        "new session until the answer, gateway route")
    put("b.cold_direct", rid, "LOCAL_MEASUREMENT", single(b["invoke_v2_direct"]["new_session_seconds"]),
        "new session until the answer, direct route")
    put("b.warm_gateway", rid, "LOCAL_MEASUREMENT", stat(b["invoke_v1_gateway"]["warm_seconds"]),
        "follow-up turns, gateway route")
    put("b.warm_direct", rid, "LOCAL_MEASUREMENT", stat(b["invoke_v2_direct"]["warm_seconds"]),
        "follow-up turns, direct route")

    c = runs["c-aks-agent-sandbox-2026-09-27/run.json"]
    rid = c["run_id"]
    put("c.prepare", rid, "LOCAL_MEASUREMENT", single(c["prepare_script_seconds"]),
        "image, AKS, Kata pool, controller, storage")
    put("c.private_link", rid, "LOCAL_MEASUREMENT", single(c["storage_private_link_seconds"]),
        "Blob private endpoint")
    put("c.apply_ready", rid, "LOCAL_MEASUREMENT", single(c["sandbox_apply_to_ready_seconds"]),
        "Sandbox apply until Ready")
    put("c.first", rid, "LOCAL_MEASUREMENT", single(c["reflect"]["first_seconds"]), "first answer")
    put("c.warm", rid, "LOCAL_MEASUREMENT", single(c["reflect"]["second_seconds"]), "second answer")
    put("c.cold", rid, "DERIVED",
        {"n": 1, "unit": "s", "value": round(c["sandbox_apply_to_ready_seconds"] + c["reflect"]["first_seconds"], 2)},
        "apply until Ready + first answer")
    put("c.setup", rid, "DERIVED",
        {"n": 1, "unit": "s", "value": round(c["prepare_script_seconds"] + c["storage_private_link_seconds"]
                                             + c["sandbox_apply_to_ready_seconds"], 1)},
        "prepare + private link + apply until Ready")
    put("c.pod_lost", rid, "LOCAL_MEASUREMENT", single(c["pod_delete_to_ready_seconds"]),
        "pod deleted until Ready again")
    put("c.suspend", rid, "LOCAL_MEASUREMENT", single(c["suspend"]["pod_gone_seconds"]),
        "operatingMode Suspended until the pod is gone")
    put("c.resume_ready", rid, "LOCAL_MEASUREMENT", single(c["resume"]["sandbox_ready_seconds"]),
        "operatingMode Running until the Sandbox is Ready")
    put("c.resume_lb", rid, "LOCAL_MEASUREMENT", single(c["resume"]["lb_health_200_seconds"]),
        "until the LoadBalancer health check returns 200")
    put("c.resume_answer", rid, "LOCAL_MEASUREMENT", single(c["resume"]["first_answer_seconds"]),
        "until the first answer")
    put("c.isolation", rid, "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "read-back", "guest_kernel": c["kata_isolation"]["sandbox_kernel"],
         "system_node_kernel": c["kata_isolation"]["runc_pod_kernel"],
         "runtime_class": c["kata_isolation"]["runtime_class"]},
        "uname -r in the Sandbox pod and a runc pod on a different system pool; system_node_kernel is a legacy field name")

    return {"schema": 1, "built_from": dict(sorted(index.items())), "metrics": m}


def render(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    args = ap.parse_args(argv)
    try:
        text = render(build(*load_runs()))
    except (BuildError, KeyError, IndexError, ValueError) as exc:
        print(f"MEASUREMENTS FAIL {exc}", file=sys.stderr)
        return 1
    if args.check:
        committed = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if committed != text:
            print("MEASUREMENTS FAIL evidence/measurements.json differs from a fresh build", file=sys.stderr)
            return 1
        print(f"MEASUREMENTS PASS {len(json.loads(text)['metrics'])} metrics")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"MEASUREMENTS WRITTEN {len(json.loads(text)['metrics'])} metrics -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
