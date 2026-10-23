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
import math
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evidence" / "runs"
OUT = ROOT / "evidence" / "measurements.json"


class BuildError(RuntimeError):
    pass


def check_number(value: object, field: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value < 0 or value > sys.float_info.max or not math.isfinite(value)):
        raise BuildError(f"{field}: expected a finite non-negative number")


def check_durations(record: dict | list, path: str) -> None:
    items = record.items() if isinstance(record, dict) else enumerate(record)
    for key, value in items:
        if key == "clocks":
            continue  # This schema field describes the clocks, not their measured values.
        field = f"{path}.{key}"
        if isinstance(value, (dict, list)):
            check_durations(value, field)
        if (isinstance(key, str)
                and (re.search(r"(?:^|_)(?:s|seconds(?:_approx)?|minutes|ms)$", key)
                     or key.startswith("uptime_"))
                and value is not None):
            samples = value if isinstance(value, list) else [value]
            for i, sample in enumerate(samples):
                check_number(sample, f"{field}[{i}]" if isinstance(value, list) else field)


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
    for i, value in enumerate(values):
        check_number(value, f"sample[{i}] ({unit})")
    vals = sorted(values)
    rec = {"n": len(vals), "unit": unit, "values": list(values), "min": vals[0], "max": vals[-1]}
    if len(vals) % 2:
        rec["median"] = vals[len(vals) // 2]
    return rec


def single(value: float, unit: str = "s") -> dict:
    check_number(value, f"single measurement ({unit})")
    return {"n": 1, "unit": unit, "values": [value], "value": value}


def utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build(runs: dict[str, dict], index: dict[str, str]) -> dict:
    for rel, record in runs.items():
        check_durations(record, rel)
    m: dict[str, dict] = {}

    def put(key: str, run_id: str, label: str, rec: dict, what: str) -> None:
        if key in m:
            raise BuildError(f"duplicate metric {key}")
        m[key] = {"run": run_id, "label": label, "what": what, **rec}

    # A · full agent, n = 3: 2026-10-09 with the Blob volume ("a"), 2026-10-18 without it ("a.nv")
    def agent(prefix: str, rel: str, spec: str) -> list[dict]:
        a = runs[rel]
        rid, ar = a["run_id"], a["runs"]
        if len(ar) != 3:
            raise BuildError(f"{rel}: the agent run should have 3 runs")
        for r in ar:
            if "error" in r or "delete_error" in r:
                raise BuildError(f"{rel} run {r['run']}: recorded an error")
            first, after = r["first"], r["first_after_resume"]
            if abs(r["create_total_s"] + first.get("ready_wait_s", 0) + first["guest_s"] - r["cold_total_s"]) > 0.011:
                raise BuildError(f"{rel} run {r['run']}: cold_total_s is not create + wait + first answer")
            if abs(r["resume_total_s"] + after.get("ready_wait_s", 0) + after["guest_s"] - r["recover_total_s"]) > 0.011:
                raise BuildError(f"{rel} run {r['run']}: recover_total_s is not resume + wait + answer")
        wait = "ready_wait_s" in ar[0]["first"]
        put(f"{prefix}.create", rid, "LOCAL_MEASUREMENT", stat([r["create_total_s"] for r in ar]),
            f"create call until Running, {spec}")
        put(f"{prefix}.cold", rid, "LOCAL_MEASUREMENT", stat([r["cold_total_s"] for r in ar]),
            "SDK create wall time + " + ("wait until the agent accepted a connection + " if wait else "") +
            "first answer timed inside the sandbox; a component sum, not client end-to-end wall time")
        put(f"{prefix}.first_answer", rid, "LOCAL_MEASUREMENT", stat([r["first"]["guest_s"] for r in ar]),
            "first answer only, timed inside the sandbox")
        put(f"{prefix}.warm", rid, "LOCAL_MEASUREMENT", stat([r["warm"]["guest_s"] for r in ar]),
            "follow-up answer in the running sandbox")
        put(f"{prefix}.stop", rid, "LOCAL_MEASUREMENT", stat([r["stop_total_s"] for r in ar]),
            "stop call until Stopped (memory snapshot)")
        put(f"{prefix}.resume", rid, "LOCAL_MEASUREMENT", stat([r["resume_total_s"] for r in ar]),
            "resume call until Running")
        put(f"{prefix}.first_after_resume", rid, "LOCAL_MEASUREMENT",
            stat([r["first_after_resume"]["guest_s"] for r in ar]), "first answer after resume")
        put(f"{prefix}.recover", rid, "LOCAL_MEASUREMENT", stat([r["recover_total_s"] for r in ar]),
            "SDK resume wall time + " + ("wait + " if wait else "") +
            "first answer inside the sandbox; excludes stop and idle waiting")
        if wait:
            put(f"{prefix}.ready_wait", rid, "LOCAL_MEASUREMENT", stat([r["first"]["ready_wait_s"] for r in ar]),
                "inside the sandbox, until the agent accepted a connection, before the first answer")
            put(f"{prefix}.ready_wait_resume", rid, "LOCAL_MEASUREMENT",
                stat([r["first_after_resume"]["ready_wait_s"] for r in ar]), "the same before the answer after resume")
        kept = sum(r["first_after_resume"]["history_length"] > r["warm"]["history_length"] for r in ar)
        put(f"{prefix}.state_kept", rid, "LOCAL_MEASUREMENT", {"n": len(ar), "unit": "runs", "count": kept},
            "conversation history length kept growing across stop and resume")
        # Column medians can come from different runs; these parts all belong to the run with the median total.
        for key, total, call, turn in (("cold", "cold_total_s", "create", "first"),
                                       ("recover", "recover_total_s", "resume", "first_after_resume")):
            mid = [r for r in ar if r[total] == m[f"{prefix}.{key}"]["median"]]
            if len(mid) != 1:
                raise BuildError(f"{rel}: expected exactly one run with the median {total}")
            r = mid[0]
            put(f"{prefix}.{key}_mid", rid, "LOCAL_MEASUREMENT",
                {**single(r[total]), "run_no": r["run"], call: r[f"{call}_total_s"],
                 "wait": r[turn].get("ready_wait_s", 0), "answer": r[turn]["guest_s"]},
                f"the run with the median {key}, and its own {call} + wait + answer")
        return a.get("earlier_attempts", [])

    agent("a", "a-aca-2026-10-09/agent-runs.json", "full spec")
    a09 = runs["a-aca-2026-10-09/agent-runs.json"]
    vmm = sorted({r["vmm_type"] for r in a09["runs"]})
    put("a.vmm", a09["run_id"], "LOCAL_MEASUREMENT",
        {"n": len(a09["runs"]), "unit": "runs", "value": vmm[0] if len(vmm) == 1 else vmm,
         "count": sum(r["vmm_type"] == "cloudhypervisor" for r in a09["runs"])}, "vmm_type the API reported")
    earlier = agent("a.nv", "a-aca-2026-10-18/agent-novolume.json", "full spec without the Blob volume")
    nv_rid = runs["a-aca-2026-10-18/agent-novolume.json"]["run_id"]
    failed = [f for e in earlier for f in e["failures"]]
    if not earlier or len(failed) != 1 or sum(e["runs"] for e in earlier) != sum(e["completed"] for e in earlier) + 1:
        raise BuildError("a-2026-10-18-novolume: expected the earlier attempts to hold exactly one failed run")
    put("a.nv.earlier", nv_rid, "LOCAL_MEASUREMENT",
        {"n": sum(e["runs"] for e in earlier), "unit": "runs", "count": sum(e["completed"] for e in earlier)},
        "earlier attempts with the tool revision that did not wait for the agent port: runs completed")
    put("a.nv.earlier_cold", nv_rid, "LOCAL_MEASUREMENT", stat([v for e in earlier for v in e["cold_total_s"]]),
        "cold of the completed earlier attempts")
    put("a.nv.earlier_recover", nv_rid, "LOCAL_MEASUREMENT", stat([v for e in earlier for v in e["recover_total_s"]]),
        "resume → answer of the completed earlier attempts")
    put("a.nv.refused_after", nv_rid, "LOCAL_MEASUREMENT", single(failed[0]["create_total_s"]),
        "create → Running of the earlier run whose first request was refused before the agent listened")

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
    iso = setup["isolation_check"]
    put("a.isolation", setup["run_id"], "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "read-back", "vmm_type": iso["vmm_type"], "guest_kernel": iso["guest_kernel"]},
        "read inside the sandbox")

    # Shared foundation · 2026-09-27, one run
    f = runs["abc-2026-09-27/foundation.json"]
    put("f.deploy", f["run_id"], "LOCAL_MEASUREMENT", single(f["deploy_minutes"], "min"), "module-01 deployment")

    # B · 2026-10-15, explicit session lifecycle through the session API, n = 3
    bs = runs["b-sessions-2026-10-15/sessions.json"]
    rid = bs["run_id"]
    br = bs["runs"]
    if len(br) != 3:
        raise BuildError("B session run should have 3 runs")
    for r in br:
        if "error" in r or r.get("deleted") is not True:
            raise BuildError(f"B session run {r['run']}: recorded an error or did not delete its session")

    def count_grew(r: dict) -> bool:
        counts = [r[k].get("response") for k in ("warm", "resume", "warm_after_resume")]
        return (all(isinstance(x, str) and x.isdigit() for x in counts)
                and int(counts[0]) < int(counts[1]) < int(counts[2]))

    create = [round(r["create"]["total_s"] + ((r["create"].get("wait_active") or {}).get("seconds") or 0), 3)
              for r in br]
    first = [r["first"]["total_s"] for r in br]
    put("b.sess.create", rid, "LOCAL_MEASUREMENT", stat(create),
        "POST .../endpoint/sessions until the session reads active, client wall time")
    put("b.sess.first", rid, "LOCAL_MEASUREMENT", stat(first), "first answer on the active session, client wall time")
    put("b.sess.warm", rid, "LOCAL_MEASUREMENT", stat([r["warm"]["total_s"] for r in br]),
        "follow-up answer on the same session")
    put("b.sess.stop", rid, "LOCAL_MEASUREMENT",
        stat([round(r["stop"]["total_s"] + r["stop"]["wait_stopped"]["seconds"], 3) for r in br]),
        "POST .../sessions/{id}:stop until the session reads idle")
    put("b.sess.resume", rid, "LOCAL_MEASUREMENT", stat([r["resume"]["total_s"] for r in br]),
        "next answer on the stopped session; the platform provisions compute and restores $HOME")
    put("b.sess.warm_after_resume", rid, "LOCAL_MEASUREMENT", stat([r["warm_after_resume"]["total_s"] for r in br]),
        "the answer after that")
    put("b.sess.server", rid, "LOCAL_MEASUREMENT",
        stat([r[k]["server_s"] for r in br for k in ("first", "warm", "resume", "warm_after_resume")]),
        "every turn: response completed_at minus created_at as recorded by the service (whole seconds)")
    put("b.sess.state_kept", rid, "LOCAL_MEASUREMENT",
        {"n": len(br), "unit": "runs", "files": sum(r["marker_after_resume"]["matches"] is True for r in br),
         "conversation": sum(count_grew(r) for r in br)},
        "files: the file written before stop read back unchanged after resume; conversation: the question count "
        "kept growing across stop and resume")

    # C · 2026-09-27, one run
    c = runs["c-aks-agent-sandbox-2026-09-27/run.json"]
    rid = c["run_id"]
    resume = c["resume"]
    if not (resume["sandbox_ready_seconds"] <= resume["lb_health_200_seconds"]
            <= resume["first_answer_seconds"]):
        raise BuildError("C resume: expected Ready <= LoadBalancer healthy <= first answer")
    put("c.apply_ready", rid, "LOCAL_MEASUREMENT", single(c["sandbox_apply_to_ready_seconds"]),
        "Sandbox apply until Ready")
    put("c.pod_lost", rid, "LOCAL_MEASUREMENT", single(c["pod_delete_to_ready_seconds"]),
        "pod deleted until Ready again")
    put("c.resume_ready", rid, "LOCAL_MEASUREMENT", single(c["resume"]["sandbox_ready_seconds"]),
        "operatingMode Running until the Sandbox is Ready")
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
        if pod["create_to_ready_s"] > t["wall_s"] + 0.1:
            raise BuildError("D cold pod create-to-Ready exceeds the paired request wall time")
    put("d.pod_ready", rid, "LOCAL_MEASUREMENT", stat(ready), "session pod created until Ready, image cached")
    put("d.pod_ready_fresh", rid, "LOCAL_MEASUREMENT", stat(bench["pod_create_to_ready_fresh_node_s"]),
        "session pod created until Ready, fresh node")
    put("d.first_after_ready", rid, "DERIVED",
        stat([round(t["wall_s"] - t["pod"]["create_to_ready_s"], 2) for t in bench["cold_turns"]]),
        "paired cold wall time minus session pod create-to-Ready duration")
    follow = bench["warm_turns"][1:]
    if "LANTERN-7" not in (follow[0]["answer"] or ""):
        raise BuildError("D warm session did not remember LANTERN-7")
    levels = {lv["concurrency"]: lv for lv in bench["scalability"]}
    for n in (4, 8):
        lv = levels[n]
        t1 = [s["turn1"] for s in lv["sessions"]]
        if len(t1) != n:
            raise BuildError(f"D concurrency {n}: expected {n} session records, got {len(t1)}")
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
    onboarding = fld["onboarding_recorded"]
    put("d.role_assignments", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["role_assignments_seconds_approx"]), "BYO-AKS role assignments; approximate")
    put("d.azd_provision", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["azd_provision_seconds"]), "BYO-AKS azd provision")
    put("d.azd_deploy", fld["run_id"], "LOCAL_MEASUREMENT",
        single(onboarding["azd_deploy_seconds"]), "first BYO-AKS agent azd deploy")
    put("d.initial_runtime_pods", fld["run_id"], "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "pods", "value": fld["foundry_system_pods"]},
        "foundry-system pods in the initial one-Kata-node read-back")
    cost = fld["cost_floor_usd_per_hour"]
    put("d.cost_floor", fld["run_id"], "SOURCE_FACT",
        {"n": 1, "unit": "USD/h", "value": round(cost["system_pool"] + cost["kata_pool"] + cost["standard_tier"], 3)},
        cost["shape"] + ", " + cost["source"])

    # E · E2B on Azure VM · 2026-10-14
    ea = runs["e-e2b-2026-10-14/agent-runs.json"]
    rid = ea["run_id"]
    er = ea["runs"]
    if len(er) != 3:
        raise BuildError("E agent run should have 3 runs")
    for r in er:
        if "error" in r:
            raise BuildError(f"E run {r['run']}: recorded an error")
        if abs(r["create_total_s"] + r["first"]["guest_s"] - r["cold_total_s"]) > 0.011:
            raise BuildError(f"E run {r['run']}: cold_total_s is not create + first answer")
        if abs(r["resume_total_s"] + r["first_after_resume"]["guest_s"] - r["recover_total_s"]) > 0.011:
            raise BuildError(f"E run {r['run']}: recover_total_s is not resume + answer")
    put("e.create", rid, "LOCAL_MEASUREMENT", stat([r["create_total_s"] for r in er]),
        "E2B Sandbox.create call until Running")
    put("e.first_answer", rid, "LOCAL_MEASUREMENT", stat([r["first"]["guest_s"] for r in er]),
        "first Prompt Agent answer calling E2B sandbox tool")
    put("e.cold", rid, "LOCAL_MEASUREMENT", stat([r["cold_total_s"] for r in er]),
        "E2B create wall time + first Prompt Agent answer")
    put("e.warm", rid, "LOCAL_MEASUREMENT", stat([r["warm"]["guest_s"] for r in er]),
        "follow-up Prompt Agent answer in the running E2B sandbox")
    put("e.stop", rid, "LOCAL_MEASUREMENT", stat([r["stop_total_s"] for r in er]),
        "beta_pause call until Paused (Firecracker memory + disk snapshot)")
    put("e.resume", rid, "LOCAL_MEASUREMENT", stat([r["resume_total_s"] for r in er]),
        "Sandbox.connect call until Running")
    put("e.first_after_resume", rid, "LOCAL_MEASUREMENT",
        stat([r["first_after_resume"]["guest_s"] for r in er]), "first Prompt Agent answer after resume")
    put("e.recover", rid, "LOCAL_MEASUREMENT", stat([r["recover_total_s"] for r in er]),
        "Sandbox.connect wall time + first answer after resume; excludes pause and idle waiting")
    eiso = ea["isolation_check"]
    if "firecracker" not in eiso["vmm_source"]:
        raise BuildError("E VMM read-back does not name the Firecracker process")
    put("e.vmm", rid, "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "read-back", "value": eiso["vmm_type"], "source": eiso["vmm_source"]},
        "VMM process behind the E2B sandboxes in the host's process list")
    ekept = sum(r["first_after_resume"]["history_length"] > r["warm"]["history_length"] for r in er)
    put("e.state_kept", rid, "LOCAL_MEASUREMENT", {"n": len(er), "unit": "runs", "count": ekept},
        "conversation history length inside the E2B sandbox kept growing across pause and resume")
    esetup = ea["setup"]
    put("e.setup", rid, "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "s", "value": round(esetup["apt_and_docker_s"] + esetup["compose_and_template_s"], 1),
         "parts": [esetup["apt_and_docker_s"], esetup["compose_and_template_s"]]},
        "Docker Engine install + E2B Embed Compose up + base Firecracker template build")
    ecost = ea["cost_floor_usd_per_hour"]
    put("e.cost_floor", rid, "SOURCE_FACT",
        {"n": 1, "unit": "USD/h",
         "value": round(ecost["vm_d4ds_v7"] + ecost["public_ip"] + ecost["premium_ssd_p6"], 3)},
        ecost["shape"] + ", " + ecost["source"])
    mb = ea["on_host_microbench_20x"]
    put("e.bench20.create", rid, "LOCAL_MEASUREMENT", stat(mb["create_only_s"]),
        "on-host E2B Sandbox.create only, 20 sequential sandboxes")
    put("e.bench20.warm_cmd", rid, "LOCAL_MEASUREMENT", stat(mb["warm_cmd_s"]),
        "on-host warm command inside a running E2B sandbox, 20 runs")
    put("e.bench20.pause", rid, "LOCAL_MEASUREMENT", stat(mb["pause_s"]),
        "on-host beta_pause memory+disk snapshot, 20 runs")
    put("e.bench20.resume", rid, "LOCAL_MEASUREMENT", stat(mb["resume_connect_s"]),
        "on-host Sandbox.connect resume, 20 runs")

    ep = runs["e-e2b-2026-10-14/perf.json"]
    erid = ep["run_id"]
    for t in ep["cold_turns"] + ep["warm_turns"]:
        if t["http"] != 200:
            raise BuildError(f"E perf turn {t['started_utc']} was not answered")
    for t in ep["cold_turns"]:
        if t["sandbox_create_to_ready_s"] > t["wall_s"] + 0.1:
            raise BuildError("E cold sandbox create-to-Ready exceeds the paired request wall time")
    put("e.d_cold", erid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in ep["cold_turns"]]),
        "new session until the answer, D-matched 5-run cold protocol")
    put("e.d_sandbox_ready", erid, "LOCAL_MEASUREMENT",
        stat([t["sandbox_create_to_ready_s"] for t in ep["cold_turns"]]),
        "E2B sandbox created until Ready during the 5 cold turns")
    put("e.d_agent_turn", erid, "LOCAL_MEASUREMENT",
        stat([t["agent_turn_s"] for t in ep["cold_turns"]]),
        "Prompt Agent turn duration after sandbox Ready during the 5 cold turns")
    efollow = ep["warm_turns"][1:]
    if "LANTERN-7" not in (efollow[0]["answer"] or ""):
        raise BuildError("E warm session did not remember LANTERN-7")
    elevels = {lv["concurrency"]: lv for lv in ep["scalability"]}
    for n in (4, 8):
        lv = elevels[n]
        t1 = [s["turn1"] for s in lv["sessions"]]
        if len(t1) != n:
            raise BuildError(f"E concurrency {n}: expected {n} session records, got {len(t1)}")
        ok = [t["wall_s"] for t in t1 if t["http"] == 200]
        refused = [t for t in t1 if t["http"] != 200]
        put(f"e.c{n}", erid, "LOCAL_MEASUREMENT",
            {"n": n, "unit": "sessions", "answered": len(ok), "answered_s": ok, "refused": len(refused)},
            f"{n} new sessions at once on 1 {ep['node_sku']} VM")
    erec = ep["recovery"]
    if not erec["followup_on_live_session"]["state_preserved"] or erec["followup_on_live_session"]["answer"] != "LANTERN-7":
        raise BuildError("the live session used after E sandbox kill did not keep LANTERN-7")
    put("e.pod_lost", erid, "LOCAL_MEASUREMENT", single(erec["replacement_ready_s"]),
        "sandbox killed until a replacement E2B sandbox is Ready")

    def e_agent_runs(rid_: str, prefix: str, rr: list[dict], where: str) -> None:
        if len(rr) != 3:
            raise BuildError(f"{rid_}: expected 3 agent runs")
        for r in rr:
            if "error" in r or r.get("history_lengths") != [1, 2, 3, 4] or r.get("state_kept") is not True:
                raise BuildError(f"{rid_} run {r['run']}: recorded an error or lost the sandbox history")
            if r.get("sandbox_deleted") is not True or r.get("conversation_deleted") is not True:
                raise BuildError(f"{rid_} run {r['run']}: left its sandbox or conversation")
            if abs(r["create_total_s"] + r["first"]["wall_s"] - r["cold_total_s"]) > 0.011:
                raise BuildError(f"{rid_} run {r['run']}: cold_total_s is not create + first answer")
            if abs(r["resume_total_s"] + r["first_after_resume"]["wall_s"] - r["recover_total_s"]) > 0.011:
                raise BuildError(f"{rid_} run {r['run']}: recover_total_s is not resume + answer")
        for key, pick, what in (
                ("create", lambda r: r["create_total_s"], "Sandbox.create until it returns"),
                ("first", lambda r: r["first"]["wall_s"], "first Prompt Agent turn"),
                ("cold", lambda r: r["cold_total_s"], "create + first Prompt Agent turn, added per run"),
                ("warm", lambda r: r["warm"]["wall_s"], "follow-up turn"),
                ("pause", lambda r: r["stop_total_s"], "beta_pause until it returns"),
                ("resume", lambda r: r["resume_total_s"], "Sandbox.connect until it returns"),
                ("first_after_resume", lambda r: r["first_after_resume"]["wall_s"], "first turn after resume"),
                ("recover", lambda r: r["recover_total_s"], "connect + first turn after resume, added per run")):
            put(f"{prefix}.{key}", rid_, "LOCAL_MEASUREMENT", stat([pick(r) for r in rr]), f"{what}, {where}")
        put(f"{prefix}.state_kept", rid_, "LOCAL_MEASUREMENT", {"n": len(rr), "unit": "runs", "count": len(rr)},
            f"sandbox history grew 1, 2, 3, 4 across pause and resume, {where}")

    # E · 2026-10-15, two independent E2B Embed installs
    mu = runs["e-e2b-2026-10-15/multi.json"]
    mrid = mu["run_id"]
    e_agent_runs(mrid, "e.multi.b", mu["agent_runs_host_b"], "host b")
    refusal_cause = {(c["phase"]): c["cause"] for side in ("a", "b") for c in mu["refused_creates"][side]
                     if c["phase"] in ("perf_host_a", "two_hosts_at_once")}

    def at_once_summary(level: dict, phase: str) -> dict:
        sessions = level["sessions"]
        if len(sessions) != level["sessions_at_once"]:
            raise BuildError(f"E {phase}: expected {level['sessions_at_once']} session records")
        per_host: dict[str, dict] = {}
        for s in sessions:
            h = per_host.setdefault(s["host"], {"sessions": 0, "answered": 0})
            h["sessions"] += 1
            h["answered"] += bool(s.get("remembered"))
        answered = sum(h["answered"] for h in per_host.values())
        if answered != level["answered"] or per_host != level["per_host"]:
            raise BuildError(f"E {phase}: answered counts differ from the session records")
        refused = [s for s in sessions if not s.get("remembered")]
        if not all("Failed to place sandbox" in s.get("error", "") for s in refused):
            raise BuildError(f"E {phase}: a session failed for another reason than a refused sandbox")
        if any(s.get("sandbox_deleted") is not True or s.get("conversation_deleted") is not True for s in sessions):
            raise BuildError(f"E {phase}: a session left its sandbox or conversation")
        return {"n": len(sessions), "unit": "sessions", "answered": answered, "refused": len(refused),
                "per_host": per_host, "model_http_statuses": level["model_http_statuses"],
                "refusal_cause": refusal_cause[phase]}

    put("e.multi.one_host", mrid, "LOCAL_MEASUREMENT", at_once_summary(mu["perf_host_a"]["at_once"], "perf_host_a"),
        "8 sessions at once on host a alone")
    put("e.multi.two_hosts", mrid, "LOCAL_MEASUREMENT", at_once_summary(mu["two_hosts_at_once"], "two_hosts_at_once"),
        "8 sessions at once on each of hosts a and b, spread by the client")
    fits, per_sandbox = [], set()
    for r in mu["idle_capacity_host_a"]:
        ok = [c for c in r["creates"] if c["ok"]]
        if r["refused_at"] != len(ok) + 1 or r["creates"][len(ok)]["ok"] or r["after_kill"]["sandboxes_left"] != 0:
            raise BuildError(f"E idle capacity run {r['run']}: creates do not end in one refusal, or left sandboxes")
        taken = [r["before"]["hugepages_taken"]] + [c["hugepages_taken"] for c in ok]
        per_sandbox |= {b - a for a, b in zip(taken, taken[1:])}
        fits.append(len(ok))
    pool, pg = mu["hosts"]["a"]["hugepages"], mu["hosts"]["a"]["postgres_hugepages"]
    if len(per_sandbox) != 1:
        raise BuildError(f"E idle capacity: per-sandbox hugepage reservation varies {sorted(per_sandbox)}")
    page = per_sandbox.pop()
    if (pool - pg) // page != max(fits):
        raise BuildError("E idle capacity: the largest count differs from the free hugepages / per-sandbox pages")
    put("e.multi.hugepages", mrid, "LOCAL_MEASUREMENT",
        {"n": 1, "unit": "pages", "pool": pool, "postgres": pg, "per_sandbox": page, "fit": max(fits)},
        "2 MiB hugepages: Compose default pool, held by postgres, reserved by each 512 MiB base sandbox at create")
    hl = mu["host_loss"]
    if hl["stop"]["exit_code"] != 0 or hl["start"]["exit_code"] != 0 or hl["start"]["health_up"]["status"] != 200:
        raise BuildError("E host loss: stopping or starting host a's stack failed")
    if hl["left_after_cleanup"] != {"a": 0, "b": 0}:
        raise BuildError("E host loss: sandboxes left after cleanup")

    # E · 2026-10-15, newest SDK and E2B's official code-interpreter template
    la = runs["e-e2b-2026-10-15/latest.json"]
    lrid = la["run_id"]
    e_agent_runs(lrid, "e.latest.sdk", la["sdk_check"]["runs"], "e2b 2.56.0 on host a")
    ci = la["code_interpreter"]
    cr = ci["check_runs"]
    names = [s["step"] for s in cr[0]["steps"]]
    if len(cr) != 4 or any([s["step"] for s in r["steps"]] != names for r in cr):
        raise BuildError("E code-interpreter: expected 4 runs of the same steps")
    if any(r["leftover_sandboxes"] != 0 for r in cr):
        raise BuildError("E code-interpreter: a run left a sandbox")
    steps = {}
    for i, name in enumerate(names):
        rows = [r["steps"][i] for r in cr]
        steps[name] = {"passed": sum(bool(s["ok"]) for s in rows), "min": min(s["seconds"] for s in rows),
                       "max": max(s["seconds"] for s in rows)}
    put("e.latest.ci", lrid, "LOCAL_MEASUREMENT",
        {"n": len(cr), "unit": "runs", "steps": steps,
         "failed": [k for k, v in steps.items() if v["passed"] < len(cr)]},
        "code-interpreter-v1 checks with e2b-code-interpreter 2.11.0; seconds per step, min and max")
    builds = ci["builds"]
    built = [b for b in builds if b["result"] == "built"]
    if len(built) != 1 or built[0]["step"] != built[0]["steps"]:
        raise BuildError("E code-interpreter: expected exactly one complete build")

    # E · 2026-10-17, the agent's code run by E2B's Code Interpreter (run_code) in code-interpreter-v1, host c
    ca, cp = runs["e-e2b-2026-10-17/ci-agent.json"], runs["e-e2b-2026-10-17/ci-perf.json"]
    for doc in (ca, cp):
        if doc["executor"] != "code-interpreter" or doc["template"]["name"] != "code-interpreter-v1":
            raise BuildError(f"{doc['run_id']}: not produced with the code-interpreter executor")
        if doc["leftover_sandboxes"] != 0 or doc["agent_version_deleted"] is not True:
            raise BuildError(f"{doc['run_id']}: left a sandbox or the agent version")
    if ca["tool"] != cp["tool"] or ca["agent"] != cp["agent"]:
        raise BuildError("E code-interpreter: the two runs used different tool revisions or agents")
    crid, prid = ca["run_id"], cp["run_id"]
    e_agent_runs(crid, "e.ci", ca["runs"], "code-interpreter-v1 on host c")
    if {r[t]["tool_calls"] for r in ca["runs"] for t in ("first", "warm", "first_after_resume", "warm_after_resume")} != {1}:
        raise BuildError("E code-interpreter agent runs: expected one sandbox tool call per turn")
    cold = cp["cold"]
    if len(cold) != 5 or not all(t["http"] == 200 and t["correct"] for t in cold):
        raise BuildError("E code-interpreter cold: expected 5 answered, correct new sessions")
    if any(t["sandbox_create_to_ready_s"] > t["wall_s"] for t in cold):
        raise BuildError("E code-interpreter cold: sandbox create-to-Ready exceeds the session wall time")
    put("e.ci.d_cold", prid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in cold]),
        "new session until the answer, D-matched 5-run cold protocol, code-interpreter-v1")
    put("e.ci.d_sandbox_ready", prid, "LOCAL_MEASUREMENT", stat([t["sandbox_create_to_ready_s"] for t in cold]),
        "code-interpreter-v1 sandbox created until Ready during the 5 cold sessions")
    put("e.ci.d_agent_turn", prid, "LOCAL_MEASUREMENT", stat([t["agent_turn_s"] for t in cold]),
        "Prompt Agent turn after sandbox Ready during the 5 cold sessions")
    warm = cp["warm"]["turns"]
    if (not cp["warm"]["remembered"] or warm[1]["answer"] != "LANTERN-7" or any(t["http"] != 200 for t in warm)
            or len(warm) != 4):
        raise BuildError("E code-interpreter warm session: expected 4 answered turns that remember LANTERN-7")
    put("e.ci.d_warm", prid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in warm[1:]]),
        "follow-up turns in the same session, D-matched warm protocol, code-interpreter-v1")
    for lv in cp["at_once"]:
        n, ss = lv["sessions_at_once"], lv["sessions"]
        ok = [s for s in ss if s.get("remembered")]
        refused = [s for s in ss if not s.get("remembered")]
        if len(ss) != n or len(ok) != lv["answered"] or lv["per_host"] != {"c": {"sessions": n, "answered": len(ok)}}:
            raise BuildError(f"E code-interpreter {n} at once: counts differ from the session records")
        if not all("Failed to place sandbox" in s.get("error", "") for s in refused):
            raise BuildError(f"E code-interpreter {n} at once: a session failed for another reason than a refused sandbox")
        if any(s["sandbox_deleted"] is not True or s["conversation_deleted"] is not True for s in ss):
            raise BuildError(f"E code-interpreter {n} at once: a session left its sandbox or conversation")
        put(f"e.ci.c{n}", prid, "LOCAL_MEASUREMENT",
            {"n": n, "unit": "sessions", "answered": len(ok), "refused": len(refused),
             "session_s": [s["session_s"] for s in ok]},
            f"{n} new sessions at once on host c, code-interpreter-v1; a refused session never got a sandbox")
    rec = cp["recovery"]
    live = rec["followup_on_live_session"]
    if not live["state_preserved"] or live["answer"] != "LANTERN-7" or live["http"] != 200:
        raise BuildError("E code-interpreter: the live session used after the sandbox kill did not keep LANTERN-7")
    put("e.ci.pod_lost", prid, "LOCAL_MEASUREMENT", single(rec["replacement_ready_s"]),
        "sandbox killed until a replacement code-interpreter-v1 sandbox is Ready")
    cap = cp["idle_capacity"]
    if cap["restore_rc"] != 0 or cap["restored_meminfo"]["hp_total"] != cap["hugepages_env_found"]:
        raise BuildError("E idle capacity: the hugepage pool was not restored")
    memory_mb = {"code-interpreter-v1": ca["template"]["memory_mb"], "base": 512}
    idle = {"n": 1, "unit": "sandboxes"}
    for ph in cap["phases"]:
        pool = ph["hugepages_env"]
        if ph["compose_up_rc"] != 0 or ph["meminfo"]["hp_total"] != pool:
            raise BuildError(f"E idle capacity: the {pool}-page pool was not applied")
        for r in ph["results"]:
            short = "ci" if r["template"] == "code-interpreter-v1" else "base"
            if r["leftover"] != 0 or r["killed"] != r["created"] or not all(row[2] for row in r["rows"]):
                raise BuildError(f"E idle capacity {pool}/{short}: a check failed or a sandbox was left")
            if "Failed to place sandbox" not in (r["stopped_by"] or {}).get("message", ""):
                raise BuildError(f"E idle capacity {pool}/{short}: creates did not end in a refused placement")
            pages = memory_mb[r["template"]] * 1024 // cap["hugepage_size_kb"]
            if r["before"]["hp_free"] // pages != r["running_at_once"]:
                raise BuildError(f"E idle capacity {pool}/{short}: the count differs from free pages / template pages")
            idle[f"{short}_{pool}"] = r["running_at_once"]
            idle[f"{short}_pages"] = pages
            idle[f"free_{pool}"] = r["before"]["hp_free"]
    put("e.ci.idle", prid, "LOCAL_MEASUREMENT", idle,
        "idle sandboxes created one by one on host c until the first refused create, by template and hugepage pool")
    attempts = ca["build"]["attempts"]
    built = [b for b in attempts if b["result"] == "built"]
    if len(built) != 1 or built[0]["step"] != built[0]["steps"] or built[0] is not attempts[-1]:
        raise BuildError("E code-interpreter on host c: expected exactly one complete build, the last attempt")
    put("e.ci.build", crid, "LOCAL_MEASUREMENT",
        {"n": len(attempts), "unit": "attempts", "built_attempt": built[0]["attempt"], "built_s": built[0]["seconds"],
         "cached_through": built[0]["cached_steps"][1], "steps": built[0]["steps"]},
        "unchanged build_prod.py against host c with the three raised settings of host a; no change between attempts")

    # E · 2026-10-18, code-interpreter-v1 only: host mode on host c, two hosts (a and c), and a D8ds_v7 (host d)
    def readback(doc: dict) -> dict:
        iso = doc["isolation"]
        vmm, info = iso["vmm_processes"], iso["sandbox_info"]
        if "error" in iso or iso["sandbox_deleted"] is not True or not vmm \
                or not all(p.rsplit("/", 1)[-1] == "firecracker" and "/fc-versions/" in p for p in vmm):
            raise BuildError(f"{doc['run_id']}: no Firecracker process behind the sandbox, or its sandbox was left")
        versions = {p.split("/fc-versions/")[1].split("/")[0] for p in vmm}
        if len(versions) != 1 or info["template_name"] != doc["template"]["name"] or info["cpu_count"] != iso["guest_cpus"]:
            raise BuildError(f"{doc['run_id']}: the read-back does not match one Firecracker version and the template")
        return {"n": 1, "unit": "read-back", "vmm_type": "firecracker", "firecracker_version": versions.pop(),
                "guest_kernel": iso["guest_kernel"], "host_kernel": iso["host_kernel"], "guest_cpus": iso["guest_cpus"],
                "guest_mem_total_mb": iso["guest_mem_total_mb"], "template_memory_mb": info["memory_mb"]}

    def reserved(doc: dict) -> dict:
        r = doc["reservation"]
        steps = r["steps"]
        taken = [s["hugepages_total"] - s["hugepages_free"] + s["hugepages_rsvd"] for s in steps]
        per = {b - a for a, b in zip(taken, taken[1:])}
        if "error" in r or not steps or not all(s["check_ok"] for s in steps) or len(per) != 1:
            raise BuildError(f"{doc['run_id']}: a reservation step failed its check or the pages per sandbox vary")
        page = per.pop()
        if page * 2 != doc["template"]["memory_mb"]:
            raise BuildError(f"{doc['run_id']}: {page} 2 MiB pages per sandbox is not the template's memory")
        held, pool, refused = taken[0] - page, steps[0]["hugepages_total"], r.get("refused") or {}
        if refused.get("at") != len(steps) + 1 or "Failed to place sandbox" not in refused.get("message", ""):
            raise BuildError(f"{doc['run_id']}: the creates did not end in one refused placement")
        if (pool - held) // page != len(steps):
            raise BuildError(f"{doc['run_id']}: the sandboxes that fit differ from the free pages / pages per sandbox")
        after = r["after"]
        if r["sandboxes_deleted"] is not True or after["hugepages_total"] - after["hugepages_free"] + \
                after["hugepages_rsvd"] != held:
            raise BuildError(f"{doc['run_id']}: the sandboxes were not deleted or their pages did not return")
        before = r["before"]
        return {"n": 1, "unit": "pages", "pool": pool, "held_by_others": held, "per_sandbox": page,
                "fit": len(steps), "refused_at": refused["at"], "refused_s": refused["seconds"], "taken": taken,
                "taken_before": before["hugepages_total"] - before["hugepages_free"] + before["hugepages_rsvd"],
                "create_s": [s["create_s"] for s in steps],
                "mem_available_mb": [before["mem_available_mb"]] + [s["mem_available_mb"] for s in steps]}

    def at_once_ci(level: dict, where: str) -> dict:
        ss = level["sessions"]
        per_host: dict[str, dict] = {}
        for s in ss:
            h = per_host.setdefault(s["host"], {"sessions": 0, "answered": 0})
            h["sessions"] += 1
            h["answered"] += s.get("remembered") is True
        ok = [s for s in ss if s.get("remembered") is True]
        refused = [s for s in ss if s.get("remembered") is not True]
        if len(ss) != level["sessions_at_once"] or len(ok) != level["answered"] or per_host != level["per_host"]:
            raise BuildError(f"E {where}: counts differ from the session records")
        if any(s["turn2"]["answer"] != f"SCALE-{s['slot']}" for s in ok):
            raise BuildError(f"E {where}: a remembered flag differs from its answer")
        if not all("Failed to place sandbox" in s.get("error", "") for s in refused):
            raise BuildError(f"E {where}: a session failed for another reason than a refused sandbox")
        if any(s["sandbox_deleted"] is not True or s["conversation_deleted"] is not True for s in ss):
            raise BuildError(f"E {where}: a session left its sandbox or conversation")
        return {"n": len(ss), "unit": "sessions", "answered": len(ok), "refused": len(refused), "per_host": per_host,
                "create_s": [s["sandbox_create_to_ready_s"] for s in ok], "session_s": [s["session_s"] for s in ok],
                "wall_s": level["wall_s"], "model_http_statuses": level["model_http_statuses"]}

    hc = runs["e-e2b-2026-10-18/ci-host.json"]
    hrid = hc["run_id"]
    if hc["leftover_sandboxes"] != 0:
        raise BuildError(f"{hrid}: a sandbox was left")
    put("e.h.isolation", hrid, "LOCAL_MEASUREMENT", readback(hc),
        "Firecracker process behind a code-interpreter-v1 sandbox in host c's process list; kernel, vCPUs and "
        "MemTotal read inside it with run_code; uname -r on the host")
    bench = hc["bench"]["runs"]
    if len(bench) != 20 or any("error" in r or r["state_kept"] is not True or r["sandbox_deleted"] is not True
                               for r in bench):
        raise BuildError(f"{hrid}: expected 20 cycles that read x back as 42 and deleted their sandbox")
    for key, what in (("create", "Sandbox.create until it returns"), ("first_cell", "first run_code cell, x = 41"),
                      ("warm_cell", "next cell, x += 1"), ("pause", "beta_pause until it returns"),
                      ("resume", "Sandbox.connect, after a 5 s wait, until it returns"),
                      ("cell_after_resume", "print(x) after resume")):
        vals = [r[f"{key}_s"] for r in bench]
        put(f"e.h.{key}", hrid, "LOCAL_MEASUREMENT",
            {**stat(vals), "under_0_1": sum(v < 0.1 for v in vals), "over_1": sum(v > 1 for v in vals),
             "first_3_max": max(vals[:3])},
            f"{what}; 20 code-interpreter-v1 sandboxes one after another on host c, no model")
    put("e.h.state_kept", hrid, "LOCAL_MEASUREMENT",
        {"n": len(bench), "unit": "cycles", "count": sum(r["state_kept"] is True for r in bench)},
        "x set before beta_pause read back as 42 after Sandbox.connect")
    put("e.h.reserve", hrid, "LOCAL_MEASUREMENT", reserved(hc),
        "2 MiB hugepages taken (total - free + reserved) after each code-interpreter-v1 sandbox created on host c, "
        "until a create was refused")

    cm = runs["e-e2b-2026-10-18/ci-multi.json"]
    cmid = cm["run_id"]
    if cm["leftover_sandboxes"] != 0 or cm["agent_versions_deleted"] is not True:
        raise BuildError(f"{cmid}: left a sandbox or an agent version")
    e_agent_runs(cmid, "e.m.a", cm["agent_runs_host_a"], "code-interpreter-v1 on host a")
    put("e.m.one_host", cmid, "LOCAL_MEASUREMENT", at_once_ci(cm["one_host"], "host a alone"),
        "4 sessions at once on host a alone, code-interpreter-v1")
    put("e.m.two_hosts", cmid, "LOCAL_MEASUREMENT", at_once_ci(cm["two_hosts"], "hosts a and c"),
        "4 sessions at once on each of hosts a and c, spread by the client, code-interpreter-v1")
    hl = cm["host_loss"]
    if hl["stop"]["exit_code"] != 0 or hl["start"]["exit_code"] != 0 or hl["start"]["health_up"]["status"] != 200:
        raise BuildError(f"{cmid}: stopping or starting host a's stack failed")
    if hl["left_after_cleanup"] != {"a": 0, "c": 0} or hl["conversations_deleted"] is not True:
        raise BuildError(f"{cmid}: sandboxes or conversations left after the host stop")

    cs = runs["e-e2b-2026-10-18/ci-scale.json"]
    csid = cs["run_id"]
    if cs["leftover_sandboxes"] != 0 or cs["agent_version_deleted"] is not True:
        raise BuildError(f"{csid}: left a sandbox or the agent version")
    put("e.s.isolation", csid, "LOCAL_MEASUREMENT", readback(cs),
        "the same read-back on host d, a Standard_D8ds_v7 with another host kernel")
    scold = cs["cold"]
    if len(scold) != 5 or not all(t["http"] == 200 and t["correct"] for t in scold):
        raise BuildError(f"{csid}: expected 5 answered, correct new sessions")
    if any(t["sandbox_create_to_ready_s"] > t["wall_s"] for t in scold):
        raise BuildError(f"{csid}: sandbox create-to-Ready exceeds the session wall time")
    put("e.s.d_cold", csid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in scold]),
        "new session until the answer, D-matched 5-run cold protocol, code-interpreter-v1 on host d")
    put("e.s.d_sandbox_ready", csid, "LOCAL_MEASUREMENT", stat([t["sandbox_create_to_ready_s"] for t in scold]),
        "code-interpreter-v1 sandbox created until Ready during the 5 cold sessions on host d")
    put("e.s.d_agent_turn", csid, "LOCAL_MEASUREMENT", stat([t["agent_turn_s"] for t in scold]),
        "Prompt Agent turn after sandbox Ready during the 5 cold sessions on host d")
    swarm = cs["warm"]["turns"]
    if (not cs["warm"]["remembered"] or len(swarm) != 4 or swarm[1]["answer"] != "LANTERN-7"
            or any(t["http"] != 200 for t in swarm)):
        raise BuildError(f"{csid}: expected 4 answered warm turns that remember LANTERN-7")
    put("e.s.d_warm", csid, "LOCAL_MEASUREMENT",
        {**stat([t["wall_s"] for t in swarm[1:]]), "tool_s": [t["sandbox_tool_s"] for t in swarm[1:]]},
        "follow-up turns in the same session, D-matched warm protocol, code-interpreter-v1 on host d")
    for lv in cs["at_once"]:
        n = lv["sessions_at_once"]
        put(f"e.s.c{n}", csid, "LOCAL_MEASUREMENT", at_once_ci(lv, f"host d, {n} at once"),
            f"{n} new sessions at once on host d, code-interpreter-v1")
    rec = cs["recovery"]
    live = rec["followup_on_live_session"]
    if not live["state_preserved"] or live["answer"] != "LANTERN-7" or live["http"] != 200:
        raise BuildError(f"{csid}: the live session used after the sandbox kill did not keep LANTERN-7")
    put("e.s.pod_lost", csid, "LOCAL_MEASUREMENT", single(rec["replacement_ready_s"]),
        "sandbox killed until a replacement code-interpreter-v1 sandbox is Ready on host d")
    put("e.s.reserve", csid, "LOCAL_MEASUREMENT", reserved(cs),
        "2 MiB hugepages taken after each code-interpreter-v1 sandbox created on host d, until a create was refused")
    it, built = cs["install"]["install_time"], cs["install"]["build"]
    parts = [it["docker_engine_s"], it["compose_up_s"], it["project_limits_row_and_wait_s"],
             it["clone_and_build_env_s"], it["template_build_s"]]
    if built["exit_code"] != 0 or built["attempts"] != 1 or sum(parts) != it["total_s"]:
        raise BuildError(f"{csid}: the build did not pass on its one attempt, or the install parts do not add up")

    # A · 2026-10-18, 4 and 8 sandboxes at once with the full specification
    asc = runs["a-aca-2026-10-18/scale.json"]
    rid = asc["run_id"]
    if asc["leftover_sandboxes"] != 0:
        raise BuildError("A at once: a sandbox was left")
    for lv in asc["levels"]:
        n, ss = lv["sessions_at_once"], lv["sessions"]
        if len(ss) != n or lv["answered"] != sum(s["remembered"] is True for s in ss):
            raise BuildError(f"A {n} at once: counts differ from the session records")
        if any(s["remembered"] != (f"SCALE-{s['slot']}" in s["turn2"]["response"]) or s["deleted"] is not True
               for s in ss):
            raise BuildError(f"A {n} at once: a remembered flag differs from its answer, or a sandbox was not deleted")
        put(f"a.c{n}", rid, "LOCAL_MEASUREMENT",
            {"n": n, "unit": "sessions", "answered": lv["answered"], "wall_s": lv["wall_s"],
             "create_s": sorted(s["create_to_running_s"] for s in ss),
             "read_s": sorted(s["turn2"]["guest_s"] for s in ss), "vmm": sorted({s["vmm_type"] for s in ss})},
            f"{n} agent sandboxes created at once with the full specification: create until Running, and the read "
            "turn timed inside each sandbox")

    # B · 2026-10-18, D's protocol through an API Management gateway
    bp = runs["b-hosted-agent-2026-10-18/perf.json"]
    rid = bp["run_id"]
    bcold, bwarm = bp["cold_turns"], bp["warm_turns"]
    if len(bcold) != 5 or any(t["http"] != 200 or t["answer"] != f"COLD-{i + 1}" for i, t in enumerate(bcold)):
        raise BuildError("B D-protocol cold: expected 5 answered new sessions, COLD-1 to COLD-5")
    if (not bp["warm_remembered"] or len(bwarm) != 4 or bwarm[1]["answer"] != "LANTERN-7"
            or any(t["http"] != 200 for t in bwarm)):
        raise BuildError("B D-protocol warm: expected 4 answered turns that remember LANTERN-7")
    put("b.d.cold", rid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in bcold]),
        "new session until the answer, one request through the gateway (D protocol)")
    put("b.d.warm", rid, "LOCAL_MEASUREMENT", stat([t["wall_s"] for t in bwarm[1:]]),
        "follow-up turns in the same session (D protocol)")
    for lv in bp["scalability"]:
        n, ss = lv["concurrency"], lv["sessions"]
        if len(ss) != n or lv["answered"] != sum(s["turn1"]["http"] == 200 for s in ss):
            raise BuildError(f"B {n} at once: counts differ from the session records")
        if any(s["remembered"] != (s["turn2"] is not None and s["turn2"]["answer"] == f"SCALE-{s['i']}") for s in ss):
            raise BuildError(f"B {n} at once: a remembered flag differs from its answer")
        put(f"b.d.c{n}", rid, "LOCAL_MEASUREMENT",
            {"n": n, "unit": "sessions", "answered": lv["answered"], "remembered": sum(s["remembered"] for s in ss),
             "turn1_s": sorted(s["turn1"]["wall_s"] for s in ss)},
            f"{n} new sessions at once through the gateway; each answered a write turn and a read turn")

    # B · 2026-10-18, the next turn after 17 min without requests, and the same after 5 min as a control
    for key, rel, label in (("b.idle", "b-sessions-2026-10-18/idle.json", "about 17 min"),
                            ("b.idle5", "b-sessions-2026-10-18/idle-control.json", "about 5 min")):
        doc = runs[rel]
        rid, rr = doc["run_id"], doc["runs"]
        if len(rr) != 3 or doc["resume_after"] != "idle":
            raise BuildError(f"{rel}: expected 3 runs with --resume-after idle")
        for r in rr:
            if "error" in r or r.get("deleted") is not True or "stop" in r:
                raise BuildError(f"{rel} run {r['run']}: recorded an error, sent a stop or kept its session")
            if r["marker_after_resume"]["matches"] is not True or not count_grew(r):
                raise BuildError(f"{rel} run {r['run']}: lost its file or its question count")
        put(f"{key}.waited", rid, "LOCAL_MEASUREMENT", stat([r["idle"]["waited_s"] for r in rr]),
            f"no request to the session for {label}, then one status read")
        put(f"{key}.warm", rid, "LOCAL_MEASUREMENT", stat([r["warm"]["total_s"] for r in rr]),
            "follow-up turn before the wait, client wall time")
        put(f"{key}.next", rid, "LOCAL_MEASUREMENT", stat([r["resume"]["total_s"] for r in rr]),
            f"the next turn after {label} without requests, client wall time")
        put(f"{key}.next_server", rid, "LOCAL_MEASUREMENT", stat([r["resume"]["server_s"] for r in rr]),
            "that turn: completed_at minus created_at as recorded by the service (whole seconds)")
        put(f"{key}.after", rid, "LOCAL_MEASUREMENT", stat([r["warm_after_resume"]["total_s"] for r in rr]),
            "the turn after that")
        if key == "b.idle":  # the 5 min control is published for its turn times only
            states = [r["idle"]["state_before_resume"] for r in rr]
            put(f"{key}.state", rid, "LOCAL_MEASUREMENT",
                {"n": len(rr), "unit": "runs", "states": states,
                 "idle": sum(s in ("idle", "stopped") for s in states)},
                "session status read after the wait, before the next turn")
            put(f"{key}.state_kept", rid, "LOCAL_MEASUREMENT",
                {"n": len(rr), "unit": "runs", "files": sum(r["marker_after_resume"]["matches"] is True for r in rr),
                 "conversation": sum(count_grew(r) for r in rr)},
                "files: the file written before the wait read back unchanged; conversation: the question count "
                "kept growing")

    # C · 2026-10-18, copies of the workshop Sandbox; answers timed inside the pod
    c18 = runs["c-aks-agent-sandbox-2026-10-18/perf.json"]
    rid = c18["run_id"]
    cn, cpv = c18["cold"], c18["pvc"]
    if len(cn) != 3 or len(cpv) != 3 or c18["leftover_sandboxes"] != 0:
        raise BuildError("C 2026-10-18: expected 3 runs without and 3 with the PVC, and no Sandbox left")
    for r in cn + cpv:
        if "error" in r or r.get("deleted") is not True or r.get("history_grew") is not True or r["pvc"] != (r in cpv):
            raise BuildError(f"C {r['sandbox']}: recorded an error, lost its history or was not deleted")
        if abs(r["apply_to_ready_s"] + r["first"]["guest_s"] - r["cold_total_s"]) > 0.011:
            raise BuildError(f"C {r['sandbox']}: cold_total_s is not apply to Ready + first answer")
    for r in cpv:
        if abs(r["resume_to_ready_s"] + r["after_resume"]["guest_s"] - r["resume_to_answer_s"]) > 0.011:
            raise BuildError(f"C {r['sandbox']}: resume_to_answer_s is not resume to Ready + answer")
        if r["suspended_condition"]["status"] != "False":
            raise BuildError(f"C {r['sandbox']}: the Sandbox read Ready while suspended")
    for prefix, rr, what in (("c.n", cn, "no volume"), ("c.p", cpv, "with the Blob PVC")):
        put(f"{prefix}.apply_ready", rid, "LOCAL_MEASUREMENT", stat([r["apply_to_ready_s"] for r in rr]),
            f"kubectl apply until the Sandbox and its pod are Ready, {what}")
        put(f"{prefix}.first", rid, "LOCAL_MEASUREMENT", stat([r["first"]["guest_s"] for r in rr]),
            f"first answer timed inside the pod, {what}")
        put(f"{prefix}.cold", rid, "LOCAL_MEASUREMENT", stat([r["cold_total_s"] for r in rr]),
            f"apply until Ready + first answer inside the pod, added per run, {what}")
    put("c.p.warm", rid, "LOCAL_MEASUREMENT", stat([r["follow_up"]["guest_s"] for r in cpv]),
        "follow-up answer timed inside the pod, with the Blob PVC")
    put("c.p.pod_lost", rid, "LOCAL_MEASUREMENT", stat([r["pod_delete_to_ready_s"] for r in cpv]),
        "kubectl delete pod until a new pod of the same Sandbox is Ready")
    put("c.p.suspend", rid, "LOCAL_MEASUREMENT", stat([r["suspend_to_pod_gone_s"] for r in cpv]),
        "operatingMode Suspended until the pod is gone")
    put("c.p.resume_ready", rid, "LOCAL_MEASUREMENT", stat([r["resume_to_ready_s"] for r in cpv]),
        "operatingMode Running until a new pod is Ready")
    put("c.p.resume_answer", rid, "LOCAL_MEASUREMENT", stat([r["resume_to_answer_s"] for r in cpv]),
        "Running patch until Ready + the next answer inside the pod, added per run")
    put("c.p.state_kept", rid, "LOCAL_MEASUREMENT", {"n": len(cpv), "unit": "runs", "count": len(cpv)},
        "conversation length kept growing across pod loss and suspend/resume (state on the Blob PVC)")
    start_nodes = {x["name"] for x in c18["kata_nodes_at_start"]}
    for lv in c18["scale"]:
        n, ss = lv["sessions_at_once"], lv["sessions"]
        if len(ss) != n or lv["answered"] != sum(s.get("remembered") is True for s in ss) or lv["deleted"] != n:
            raise BuildError(f"C {n} at once: counts differ from the session records, or a Sandbox was not deleted")
        if any(s["remembered"] != (f"SCALE-{s['slot']}" in s["turn2"]["response"]) for s in ss):
            raise BuildError(f"C {n} at once: a remembered flag differs from its answer")
        put(f"c.c{n}", rid, "LOCAL_MEASUREMENT",
            {"n": n, "unit": "sessions", "answered": lv["answered"], "wall_s": lv["wall_s"], "nodes": len(lv["nodes"]),
             "on_new_nodes": sum(s["server"]["node"] not in start_nodes for s in ss),
             "ready_s": sorted(s["apply_to_ready_s"] for s in ss)},
            f"{n} Sandboxes without the volume applied at once on the Kata pool (cluster autoscaler 1 to 10 nodes)")

    # D · 2026-10-18, explicit session lifecycle through the session API on your AKS
    ds = runs["d-byoaks-2026-10-18/sessions.json"]
    rid = ds["run_id"]
    dr = ds["runs"]
    if len(dr) != 3:
        raise BuildError("D session run should have 3 runs")
    for r in dr:
        if "error" in r or r.get("deleted") is not True or r["stop"]["wait_stopped"]["state"] != "idle":
            raise BuildError(f"D session run {r['run']}: recorded an error, did not stop to idle or kept its session")
        if r["marker_after_resume"]["matches"] is not True or not count_grew(r):
            raise BuildError(f"D session run {r['run']}: lost its file or its question count")
    dcreate = [round(r["create"]["total_s"] + ((r["create"].get("wait_active") or {}).get("seconds") or 0), 3)
               for r in dr]
    put("d.sess.create", rid, "LOCAL_MEASUREMENT", stat(dcreate),
        "POST .../endpoint/sessions until the session reads active, client wall time")
    put("d.sess.first", rid, "LOCAL_MEASUREMENT", stat([r["first"]["total_s"] for r in dr]),
        "first answer on the active session")
    put("d.sess.warm", rid, "LOCAL_MEASUREMENT", stat([r["warm"]["total_s"] for r in dr]),
        "follow-up answer on the same session")
    put("d.sess.stop", rid, "LOCAL_MEASUREMENT",
        stat([round(r["stop"]["total_s"] + r["stop"]["wait_stopped"]["seconds"], 3) for r in dr]),
        "POST .../sessions/{id}:stop until the session reads idle")
    put("d.sess.resume", rid, "LOCAL_MEASUREMENT", stat([r["resume"]["total_s"] for r in dr]),
        "next answer on the stopped session")
    put("d.sess.warm_after_resume", rid, "LOCAL_MEASUREMENT", stat([r["warm_after_resume"]["total_s"] for r in dr]),
        "the answer after that")
    put("d.sess.server", rid, "LOCAL_MEASUREMENT",
        stat([r[k]["server_s"] for r in dr for k in ("first", "warm", "resume", "warm_after_resume")]),
        "every turn: response completed_at minus created_at as recorded by the service (whole seconds)")
    put("d.sess.state_kept", rid, "LOCAL_MEASUREMENT",
        {"n": len(dr), "unit": "runs", "files": sum(r["marker_after_resume"]["matches"] is True for r in dr),
         "conversation": sum(count_grew(r) for r in dr)},
        "files: the file written before stop read back unchanged after resume; conversation: the question count "
        "kept growing across stop and resume")
    pods = ds["pods"]
    if pods["pods_per_session"] != [2, 2, 2] or pods["new_pods_ended_completed"] != pods["new_pods"]:
        raise BuildError("D session pods: expected two pods per session, every one ended")
    put("d.sess.pods", rid, "LOCAL_MEASUREMENT",
        {"n": pods["new_pods"], "unit": "pods", "per_session": 2, "running_after_s": pods["pending_to_running_age_s"],
         "nodes": pods["nodes_of_new_pods"]},
        "session pods that appeared during the three runs (kubectl get pods -w; order and ages, no timestamps)")

    return {"built_from": dict(sorted(index.items())), "metrics": m}


def render(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


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
