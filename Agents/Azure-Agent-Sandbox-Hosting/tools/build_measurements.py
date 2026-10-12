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

    # D · BYO-AKS · 2026-09-29
    bench = runs["d-byoaks-2026-09-29/bench.json"]
    early = runs["d-byoaks-2026-09-29/cold-early.json"]
    rid = bench["run_id"]
    for t in bench["cold_turns"] + bench["warm_turns"]:
        if t["http"] != 200:
            raise BuildError(f"D bench turn {t['started_utc']} was not answered")
    fresh = set(early["first_pod_on_fresh_node_s"])
    cached_early = [r["wall_s"] for r in early["runs"] if r["wall_s"] not in fresh and r["http"] == 200]
    if len(cached_early) != 1 or len(fresh) != 2:
        raise BuildError("D early cold runs: expected two fresh-node runs and one cached run")
    cold = cached_early + [t["wall_s"] for t in bench["cold_turns"]] + [bench["warm_turns"][0]["wall_s"]]
    put("d.cold", rid, "LOCAL_MEASUREMENT", stat(cold),
        "new session until the answer, node already holds the image")
    put("d.cold_fresh", early["run_id"], "LOCAL_MEASUREMENT", stat(sorted(fresh, reverse=True)),
        "new session until the answer, first pod on a node without the image")
    ready = bench["pod_create_to_ready_cached_image_s"]
    bench_pods = [t["pod"]["create_to_ready_s"] for t in bench["cold_turns"] if t.get("pod")]
    for v in bench_pods:
        if v not in ready:
            raise BuildError(f"bench pod create-to-Ready {v} s is not in the run manifest's list {ready}")
    for t in bench["cold_turns"]:
        pod = t["pod"]
        if pod["runtime"] != "kata-vm-isolation":
            raise BuildError("D cold pod without kata-vm-isolation")
        if round((utc(pod["ready_at"]) - utc(pod["created"])).total_seconds(), 1) != pod["create_to_ready_s"]:
            raise BuildError("D cold pod create-to-Ready does not match its timestamps")
    put("d.pod_ready", rid, "LOCAL_MEASUREMENT", stat(ready), "session pod created until Ready, image cached")
    put("d.pod_ready_fresh", rid, "LOCAL_MEASUREMENT", stat(bench["pod_create_to_ready_fresh_node_s"]),
        "session pod created until Ready, fresh node")
    follow = bench["warm_turns"][1:]
    put("d.warm", rid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in follow]), "follow-up turns, same session")
    if "LANTERN-7" not in (follow[0]["answer"] or ""):
        raise BuildError("D warm session did not remember LANTERN-7")
    levels = {lv["concurrency"]: lv for lv in bench["scalability"]}
    for n in (4, 8):
        lv = levels[n]
        t1 = [s["turn1"] for s in lv["sessions"]]
        ok = [t["wall_s"] for t in t1 if t["http"] == 200]
        refused = [t for t in t1 if t["http"] != 200]
        codes = sorted({f"{t['http']} {t['error_code']}" for t in refused})
        put(f"d.c{n}", rid, "LOCAL_MEASUREMENT",
            {"n": n, "unit": "sessions", "answered": len(ok), "answered_s": ok, "refused": len(refused),
             "refused_codes": codes, "refused_after_s": [min(t["wall_s"] for t in refused),
                                                          max(t["wall_s"] for t in refused)] if refused else []},
            f"{n} new sessions at once")
    per_node = max(int(x) for x in re.findall(r"(\d+) sbx", bench["node_state_at_failure"]))
    put("d.capacity", rid, "LOCAL_MEASUREMENT",
        {"n": bench["agent_pool_nodes"], "unit": "sessions", "per_node": per_node,
         "max_sessions": bench["max_concurrent_sessions_observed"], "node_sku": bench["node_sku"]},
        "session pods per node when the controller started refusing")
    rec = runs["d-byoaks-2026-09-29/recovery.json"]
    if not rec["followup_after_kill"]["state_preserved"] or rec["followup_after_kill"]["answer"] != "LANTERN-7":
        raise BuildError("the different live session used after D pod deletion did not keep LANTERN-7")
    put("d.pod_lost", rec["run_id"], "LOCAL_MEASUREMENT", single(rec["replacement_ready_s"]),
        "session pod deleted until its replacement is Ready")
    put("d.other_session_followup", rec["run_id"], "LOCAL_MEASUREMENT",
        single(rec["followup_after_kill"]["wall_s"]),
        "follow-up on a different live session after the deletion; that other session kept LANTERN-7")
    cap = runs["d-byoaks-2026-09-29/capacity.json"]
    created = sorted(utc(p["created"]) for p in cap["session_pods_after_cleanup"])
    oldest, newest = created[0], created[-1]
    idle_min = int((newest - oldest).total_seconds() // 60)
    put("d.idle", cap["run_id"], "LOCAL_MEASUREMENT",
        {"n": len(created), "unit": "min", "value": idle_min,
         "oldest_created": oldest.strftime("%H:%MZ"), "listed_after": newest.strftime("%H:%MZ")},
        "pod age lower bound while still Running; no last-request timestamp was retained, so this is not idle duration")
    kernels = sorted({n["kernel"] for n in cap["nodes_after_scale"]})
    put("d.node_kernel", cap["run_id"], "LOCAL_MEASUREMENT",
        {"n": len(cap["nodes_after_scale"]), "unit": "nodes", "value": kernels[0] if len(kernels) == 1 else kernels},
        "kernelVersion of the Kata pool nodes")
    direct = runs["d-byoaks-2026-09-29/model-direct.json"]
    put("d.model_direct", direct["run_id"], "LOCAL_MEASUREMENT", stat([r["wall_s"] for r in direct["runs"]]),
        "same model called directly from the same client")
    put("d.node_scale", rid, "LOCAL_MEASUREMENT", single(bench["scale_one_to_three_seconds"]),
        "Kata pool scale operation from one to three nodes")
    latency = bench["latency_observation"]
    put("d.client_rtt", rid, "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "ms", "min": latency["client_rtt_ms"][0], "max": latency["client_rtt_ms"][1]},
        "client-to-Azure round-trip range recorded in the run manifest")
    put("d.agent_turn", rid, "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "s", "min": latency["agent_created_to_completed_s"][0],
         "max": latency["agent_created_to_completed_s"][1]},
        "response created_at to completed_at range; not an internal gateway clock")
    pl = runs["d-byoaks-2026-09-29/placement.json"]
    pod = pl["session_pod"]
    put("d.placement", pl["run_id"], "LOCAL_MEASUREMENT",
        {"n": 1 + len(pl["session_pods_at_recheck"]), "unit": "pod observations",
         "runtime_class": pod["runtimeClassName"],
         "kata_pods_at_recheck": sum(p["runtimeClassName"] == "kata-vm-isolation" for p in pl["session_pods_at_recheck"]),
         "scheduled_to_ready_s": int((utc(pod["ready_at"]) - utc(pod["scheduled_at"])).total_seconds()),
         "first_session_wall_s": pl["first_session_after_fix"]["client_wall_seconds"],
         "nsg_rule": pl["nsg_rule_added"]["name"], "nsg_priority": pl["nsg_rule_added"]["priority"]},
        "one initial pod observation plus the recheck observations; not a count of distinct pods")
    fld = runs["d-byoaks-2026-09-22/field.json"]
    put("d.aks_start", fld["run_id"], "LOCAL_MEASUREMENT", stat(fld["aks_start_seconds"]), "az aks start until Running")
    onboarding = fld["onboarding_recorded"]
    put("d.role_assignments", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["role_assignments_seconds_approx"]), "BYO-AKS role assignments; approximate")
    put("d.azd_provision", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["azd_provision_seconds"]), "BYO-AKS azd provision")
    put("d.azd_deploy", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["azd_deploy_seconds"]), "first BYO-AKS agent azd deploy")
    put("d.onboarding", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["total_seconds_approx"]), "roles + azd provision + first azd deploy; console not retained")
    put("d.initial_runtime_pods", fld["run_id"], "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "pods", "value": fld["foundry_system_pods"]},
        "foundry-system pods in the initial one-Kata-node read-back")
    cost = fld["cost_floor_usd_per_hour"]
    put("d.cost_floor", fld["run_id"], "SOURCE_FACT",
        {"n": 1, "unit": "USD/h", "value": round(cost["system_pool"] + cost["kata_pool"] + cost["standard_tier"], 3)},
        cost["shape"] + ", " + cost["source"])
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
