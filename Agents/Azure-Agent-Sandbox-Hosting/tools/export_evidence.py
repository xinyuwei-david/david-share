#!/usr/bin/env python3
"""Export the redacted run records under evidence/runs/ from the author's private run folders.

The private folders hold raw consoles, IDs and addresses that do not belong in this repository.
This exporter keeps the measured values, the verbatim inputs and the outcomes, replaces every
identifier with a neutral label, and records the SHA-256 of each private source file. It fails
closed: a source file whose hash differs from the hash its own run manifest recorded stops the
export, and so does a missing file or field.

    python tools/export_evidence.py \
        --a-remeasure <agenthost-remeasure-20261009> --abc-repro <agenthost-repro> \
        --d-retest <placement folder> --d-fix <NSG-fix folder> --d-perf <benchmark folder> \
        --d-field <field-record-evidence> [--out evidence/runs]

Only the author can run it (the inputs are private); everything it writes is committed, and
tools/build_measurements.py recomputes every published number from that output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExportError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    if not path.is_file():
        raise ExportError(f"missing source file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def need(obj: dict, *keys):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise ExportError(f"missing field {'.'.join(keys)}")
        cur = cur[k]
    return cur


def verify(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise ExportError(f"{path.name}: SHA-256 {actual} differs from the run manifest ({expected})")
    return actual


class Aliases:
    """Stable neutral labels for node, pod and namespace names, in order of first appearance."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.map: dict[str, str] = {}

    def __call__(self, name: str | None) -> str | None:
        if name is None:
            return None
        if name not in self.map:
            self.map[name] = f"{self.prefix}-{len(self.map) + 1}"
        return self.map[name]


NODE_RE = re.compile(r"aks-[a-z0-9]+-\d+-vmss[0-9a-z]+")
POD_RE = re.compile(r"sbx-[0-9a-z]+-[0-9a-z]+")


def redact_text(text: str, nodes: Aliases, pods: Aliases) -> str:
    text = NODE_RE.sub(lambda m: nodes(m.group(0)), text)
    return POD_RE.sub(lambda m: pods(m.group(0)), text)


def source(path: Path, digest: str, role: str) -> dict:
    return {"file": path.name, "sha256": digest.lower(), "bytes": path.stat().st_size, "role": role}


# --- A · ACA Sandboxes -------------------------------------------------------------------------

A_AGENT_INPUTS = {
    "first": "Where is Bangkok? Answer in one short sentence.",
    "warm": "How many questions have I asked before this one? Number only.",
    "first_after_resume": "How many questions have I asked before this one? Number only.",
    "warm_after_resume": "How many questions have I asked before this one? Number only.",
}
A_TURN_KEYS = ("guest_s", "exec_wall_s", "response", "history_length")
A_RUN_KEYS = ("run", "started_at", "put_s", "create_total_s", "initial_state", "polls", "vmm_type",
              "cold_total_s", "stop_post_s", "stop_total_s", "state_after_stop", "resume_post_s",
              "resume_total_s", "state_after_resume", "recover_total_s", "delete_post_s", "finished_at")
ATTR_KEYS = ("config", "run", "put_s", "create_total_s", "initial_state", "poll_to_running_s", "vmm_type",
             "exec1_s", "uptime_at_exec1", "stop_total_s", "state_after_stop", "resume_total_s",
             "exec_after_resume_s", "uptime_after_resume", "process_survived")


def export_a(remeasure: Path, repro: Path) -> dict[str, dict]:
    summary = load(remeasure / "evidence.json")
    hashes = need(summary, "source_files")
    srcs = {}
    for name in ("remeasure_agent_20261009.json", "remeasure_agent_20261009.py", "remeasure_20261009.json",
                 "remeasure_20261009.py", "remeasure_20261009_attribution.json",
                 "remeasure_20261009_attribution2.json", "latemount_20261009.json", "latemount_20261009.py"):
        srcs[name] = verify(remeasure / name, need(hashes, name, "sha256"))
    script = (remeasure / "remeasure_agent_20261009.py").read_text(encoding="utf-8")
    for text in set(A_AGENT_INPUTS.values()):
        if f'"{text}"' not in script:
            raise ExportError(f"input not found verbatim in remeasure_agent_20261009.py: {text!r}")

    agent = load(remeasure / "remeasure_agent_20261009.json")
    runs = []
    for r in need(agent, "runs"):
        if "error" in r:
            raise ExportError(f"A run {r.get('run')} recorded an error: {r['error']}")
        row = {k: need(r, k) for k in A_RUN_KEYS}
        for turn in A_AGENT_INPUTS:
            row[turn] = {k: need(r, turn, k) for k in A_TURN_KEYS}
        runs.append(row)
    if need(agent, "leftovers"):
        raise ExportError("A agent run left sandboxes behind")
    meta = need(agent, "meta")
    agent_out = {
        "run_id": "a-2026-10-09-agent",
        "path": "A",
        "what": "Full agent path in ACA Sandboxes, three times: create, first answer, follow-up, stop, resume, answer after resume",
        "date_utc": meta["date_utc"][:10],
        "region": need(meta, "region"),
        "service": need(meta, "service"),
        "sdk": need(meta, "sdk"),
        "client": need(meta, "client"),
        "configuration": need(meta, "configuration"),
        "sandbox_spec": {"cpu": "1000m", "memory": "2048Mi", "auto_suspend_seconds": 900,
                         "auto_suspend_mode": "Memory", "port": 8080, "port_activation": "OnDemand",
                         "vnet_connection": True, "blob_volume_mount": "/app/app/data",
                         "entrypoint": "cd /app && exec python -m app.main"},
        "inputs": {**A_AGENT_INPUTS, "request": "POST http://127.0.0.1:8080/reflect {\"query\": <input>}, "
                   "sent with sandbox exec from inside the sandbox"},
        "clocks": {
            "create_total_s": "SDK create call until the sandbox reads Running (polled every 0.25 s), wall time on the client",
            "guest_s": "HTTP /reflect duration measured inside the sandbox",
            "cold_total_s": "create_total_s + first.guest_s",
            "resume_total_s": "SDK resume call until Running",
            "recover_total_s": "resume_total_s + first_after_resume.guest_s",
        },
        "runs": runs,
        "leftover_sandboxes": 0,
        "_sources": [source(remeasure / "remeasure_agent_20261009.json", srcs["remeasure_agent_20261009.json"], "run record"),
                     source(remeasure / "remeasure_agent_20261009.py", srcs["remeasure_agent_20261009.py"], "script, inputs")],
    }

    attr_runs = []
    for name in ("remeasure_20261009.json", "remeasure_20261009_attribution.json", "remeasure_20261009_attribution2.json"):
        for r in need(load(remeasure / name), "runs"):
            if "error" in r:
                raise ExportError(f"attribution run {r.get('config')}/{r.get('run')} recorded an error")
            attr_runs.append({k: r.get(k) for k in ATTR_KEYS})
    attr = need(summary, "attribution")
    attr_out = {
        "run_id": "a-2026-10-09-attribution",
        "path": "A",
        "what": "Create time of one sandbox by configuration, to find which part of the full spec costs time",
        "date_utc": "2026-10-09",
        "region": "eastus2",
        "sdk": "azure-containerapps-sandbox 0.1.0b4",
        "configs": {
            "builtin": "built-in ubuntu disk, SDK defaults, no VNet, volume or port",
            "image": "the agent's disk image from ACR, nothing else",
            "vnet": "image + VNet connection",
            "port": "image + OnDemand port with IP access control",
            "vnetvolume": "image + VNet connection + private Blob volume",
            "full": "image + VNet connection + private Blob volume + OnDemand port",
        },
        "clock": "SDK create call until the sandbox reads Running, polled every 0.25 s, client wall time",
        "runs": attr_runs,
        "not_measured": need(attr, "volume_without_vnet"),
        "_sources": [source(remeasure / n, srcs[n], "run record") for n in
                     ("remeasure_20261009.json", "remeasure_20261009_attribution.json",
                      "remeasure_20261009_attribution2.json")]
                    + [source(remeasure / "remeasure_20261009.py", srcs["remeasure_20261009.py"], "script, configurations")],
    }

    late = load(remeasure / "latemount_20261009.json")
    late_out = {
        "run_id": "a-2026-10-09-late-mount",
        "path": "A",
        "what": "Start the sandbox without the Blob volume (image + VNet), then attach the volume to the running sandbox",
        "date_utc": "2026-10-09",
        "runs": [{k: need(r, k) for k in ("run", "put_s", "create_total_s", "add_volume_mount_s",
                                           "state_after_mount", "mount_check")} for r in need(late, "runs")],
        "_sources": [source(remeasure / "latemount_20261009.json", srcs["latemount_20261009.json"], "run record"),
                     source(remeasure / "latemount_20261009.py", srcs["latemount_20261009.py"], "script")],
    }

    events_path = repro / "logs" / "m04_events.json"
    events = need(load(events_path), "events")
    repro_summary = load(repro / "evidence.json")
    m04 = need(repro_summary, "module04_solution_c_aca_sandboxes")
    keep = {"vnet_connection", "disk_image", "volume", "sandbox_create", "reflect_in_sandbox", "suspend_sdk",
            "resume_sdk", "reflect_after_resume", "reflect_warm_after_resume", "suspend_sdk_2",
            "wake_on_http_request", "reflect_external_warm"}
    ev_out = []
    for ev in events:
        if ev["step"] not in keep:
            continue
        row = {"step": ev["step"], "at": ev["at"], "seconds": ev["seconds"]}
        for k in ("question", "response", "history_length", "exec_wall", "state", "code"):
            if k in ev:
                row[k] = ev[k]
        if ev["step"] in ("wake_on_http_request", "reflect_external_warm"):
            body = json.loads(ev["body"])
            row["query"] = body["query"]
            row["response"] = body["response"]
            row["history_length"] = body["history_length"]
        ev_out.append(row)
    setup_out = {
        "run_id": "a-2026-09-27-setup",
        "path": "A",
        "what": "First deployment of the agent to ACA Sandboxes: sandbox group, VNet connection, disk image, Blob volume, "
                "sandbox; then answer, suspend, resume and wake on an HTTP request",
        "date_utc": "2026-09-27",
        "region": "eastus2",
        "sdk": "azure-containerapps-sandbox 0.1.0b4",
        "sandbox_group_deploy_seconds": need(m04, "sandbox_group_deploy_seconds"),
        "sandbox_spec": need(m04, "sandbox_spec"),
        "disk_image_auth": need(m04, "disk_image_auth"),
        "isolation_check": need(m04, "isolation_check"),
        "process_survived_resume": need(m04, "process_survived_resume"),
        "notes": need(m04, "notes"),
        "events": ev_out,
        "_sources": [source(events_path, sha256(events_path), "SDK event log"),
                     source(repro / "evidence.json", sha256(repro / "evidence.json"), "run summary")],
    }
    return {"a-aca-2026-10-09/agent-runs.json": agent_out,
            "a-aca-2026-10-09/create-attribution.json": attr_out,
            "a-aca-2026-10-09/late-mount.json": late_out,
            "a-aca-2026-09-27/setup-and-wake.json": setup_out}


# --- B, C and the shared foundation (agenthost workshop modules 01-03, 2026-09-27) --------------

def export_abc(repro: Path) -> dict[str, dict]:
    path = repro / "evidence.json"
    s = load(path)
    digest = sha256(path)
    meta = need(s, "meta")
    if need(meta, "workshop_commit") != "905f2c2":
        raise ExportError("B/C run summary does not name workshop commit 905f2c2")
    src = [source(path, digest, "run summary")]
    common = {"date_utc": "2026-09-27", "region": need(meta, "region"),
              "workshop": "https://github.com/oxcp/ainotes/tree/905f2c2/agenthost",
              "operator_shell": need(meta, "operator_shell"), "model": need(meta, "model")}
    m01 = need(s, "module01_core")
    found = {"run_id": "abc-2026-09-27-foundation", "path": "shared",
             "what": "Workshop module-01: the shared foundation A, B and C use",
             **common, "module": "module-01",
             "deploy_minutes": need(m01, "deploy_minutes"),
             "core_resources_arm_seconds": need(m01, "core_resources_arm_seconds"),
             "creates": need(m01, "creates"),
             "apim_gateway_check": need(m01, "step3_apim_gateway"),
             "native_ai_gateway_step": need(m01, "step4_native_ai_gateway"),
             "deviations_from_workshop": need(s, "deviations_from_readme"),
             "not_reproduced": need(s, "not_reproduced"),
             "_sources": src}
    m02 = need(s, "module02_solution_a_hosted_agent")
    b = {"run_id": "b-2026-09-27", "path": "B",
         "what": "Foundry hosted agent: azd init, deploy, then a new session and follow-up turns, through the "
                 "project's gateway route (v1) and direct to the project (v2)",
         **common, "module": "module-02 (the workshop calls it Solution A)",
         "azd_ai_agent_init_seconds": need(m02, "azd_ai_agent_init_seconds"),
         "local_run": need(m02, "local_run"),
         "deploy_v1_gateway": need(m02, "deploy_v1_gateway"),
         "invoke_v1_gateway": need(m02, "invoke_v1_gateway"),
         "deploy_v2_direct": need(m02, "deploy_v2_direct"),
         "invoke_v2_direct": need(m02, "invoke_v2_direct"),
         "inputs": "not retained: the run summary records timings and outcomes, not the request text",
         "notes": need(m02, "notes"),
         "_sources": src}
    m03 = need(s, "module03_solution_b_aks_agent_sandbox")
    notes = [re.sub(r"\([A-Z]+ policy\)", "(a subscription policy)", n) for n in need(m03, "notes")
             if "operator IP" not in n]
    notes.append("The public LoadBalancer was restricted with loadBalancerSourceRanges to one operator address")
    c = {"run_id": "c-2026-09-27", "path": "C",
         "what": "AKS + agent-sandbox v0.5.2 on a Kata node pool: build, apply the Sandbox, answer, delete the pod, "
                 "suspend and resume by patching operatingMode",
         **common, "module": "module-03 (the workshop calls it Solution B)",
         "prepare_script_seconds": need(m03, "prepare_script_seconds"),
         "prepare_breakdown": need(m03, "prepare_breakdown"),
         "cluster": need(m03, "cluster"),
         "storage_private_link_seconds": need(m03, "storage_private_link_seconds"),
         "sandbox_apply_to_ready_seconds": need(m03, "sandbox_apply_to_ready_seconds"),
         "reflect": need(m03, "reflect"),
         "kata_isolation": need(m03, "kata_isolation"),
         "kata_isolation_method": "uname -r inside the Sandbox pod, and inside a pod without runtimeClassName that "
                                  "the scheduler placed on the Ubuntu system pool (workshop module-03, 'Verify Pod "
                                  "Sandboxing Kernel Isolation')",
         "pod_delete_to_ready_seconds": need(m03, "pod_delete_to_ready_seconds"),
         "state_restored_after_pod_delete": need(m03, "state_restored_after_pod_delete"),
         "suspend": need(m03, "suspend"),
         "resume": need(m03, "resume"),
         "inputs": "not retained: the run summary records timings and outcomes, not the request text",
         "notes": notes,
         "_sources": src}
    return {"abc-2026-09-27/foundation.json": found, "b-hosted-agent-2026-09-27/run.json": b,
            "c-aks-agent-sandbox-2026-09-27/run.json": c}


# --- D · BYO-AKS ----------------------------------------------------------------------------------

def manifest_hashes(manifest: dict) -> dict[str, str]:
    return {a["file"]: a["sha256"] for a in need(manifest, "artifacts")}


def export_d(retest: Path, fix: Path, perf: Path, field: Path) -> dict[str, dict]:
    nodes, pods = Aliases("agent-node"), Aliases("session-pod")

    tls = fix
    tls_manifest = load(tls / "manifest.json")
    tls_hashes = manifest_hashes(tls_manifest)
    tls_src = {}
    for name in ("session-pod.json", "nsg-rule-create.json", "invoke-3.json", "invoke-3-podwatch.txt",
                 "ip-flow-verify.txt", "svc-foundry-system.json"):
        tls_src[name] = verify(tls / name, tls_hashes[name])
    rule = load(tls / "nsg-rule-create.json")
    pod_doc = load(tls / "session-pod.json")
    items = need(pod_doc, "items")
    if len(items) != 1:
        raise ExportError("session-pod.json should hold exactly one pod")
    p = items[0]
    spec, status = need(p, "spec"), need(p, "status")
    conds = {c["type"]: c.get("lastTransitionTime") for c in status.get("conditions", [])}
    svc = load(tls / "svc-foundry-system.json")
    router = [i for i in need(svc, "items") if i["metadata"]["name"] == "foundry-router"]
    if not router or router[0]["spec"]["type"] != "LoadBalancer":
        raise ExportError("foundry-router LoadBalancer not found in svc-foundry-system.json")
    invoke = load(tls / "invoke-3.json")
    answer = " ".join(c.get("text", "") for o in invoke["output"] if o.get("type") == "message"
                      for c in o.get("content", []))
    wall = re.search(r"HTTP 200 completed .* in (\d+)s", need(tls_manifest, "result", "invoke"))
    if not wall:
        raise ExportError("NSG-fix manifest: invoke result without a duration")
    flows = []
    for line in (tls / "ip-flow-verify.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(control: )?(\w+) \([\d.]+\) -> node:443\s+access=(\w+)\s+rule=(\S+)", line)
        if m:
            flows.append({"source": m.group(2), "destination": "agent node :443", "access": m.group(3),
                          "rule": m.group(4)})
    if len(flows) != 3:
        raise ExportError("ip-flow-verify.txt: expected three verdicts")
    recheck = load(retest / "recheck-1815" / "session-pods.json")
    multiturn = load(retest / "recheck-1815" / "multiturn.json")
    mt = []
    for key in ("turn1", "turn2"):
        t = need(multiturn, key)
        mt.append({"turn": key, "status": t["status"],
                   "answer": " ".join(c.get("text", "") for o in t["output"] if o.get("type") == "message"
                                      for c in o.get("content", [])),
                   "agent_version": need(t, "agent_reference", "version"),
                   "server_seconds": t["completed_at"] - t["created_at"]})
    if mt[0]["answer"] != "OK" or not re.fullmatch(r"[A-Z]+-\d+", mt[1]["answer"]):
        raise ExportError(f"recheck multiturn answers are not OK + a code word: {[t['answer'] for t in mt]}")
    placement = {
        "run_id": "d-2026-09-29-placement",
        "path": "D",
        "what": "Where a BYO-AKS session runs, after the product-side enablement fix and the subnet NSG fix",
        "date_utc": "2026-09-29",
        "region": "eastus2",
        "agent": "agent-framework-agent-basic-responses-byoaks (private-preview sample), agent version 3",
        "root_cause": "The subnet NSG on the AKS subnet had no inbound rule for the foundry-router load balancer, so "
                      "DenyAllInBound dropped Foundry's calls to <router-ip>:443. The AKS-managed NIC NSG already "
                      "allowed AzureCloud to that address; the load-balancer rule uses floating IP, so both NSGs "
                      "evaluate the frontend address.",
        "ip_flow_verify_before_fix": flows,
        "router_service": {"name": "foundry-router", "namespace": "foundry-system", "type": "LoadBalancer",
                           "ports": [x["port"] for x in router[0]["spec"]["ports"]]},
        "nsg_rule_added": {"name": need(rule, "name"), "priority": need(rule, "priority"),
                           "direction": need(rule, "direction"), "access": need(rule, "access"),
                           "protocol": need(rule, "protocol"),
                           "source": need(rule, "sourceAddressPrefix"),
                           "destination": "<router-ip>", "destination_port": need(rule, "destinationPortRange"),
                           "nsg": "<subnet-nsg>", "provisioning_state": need(rule, "provisioningState")},
        "first_session_after_fix": {
            "http": 200, "answer": answer.splitlines()[0],
            "server_seconds": invoke["completed_at"] - invoke["created_at"],
            "agent_version": need(invoke, "agent_reference", "version"),
            "client_wall_seconds": int(wall.group(1)),
            "request_text": "not retained in the run record",
        },
        "session_pod": {
            "name": pods(p["metadata"]["name"]),
            "namespace": "foundry-<project>-<hash>",
            "runtimeClassName": spec.get("runtimeClassName"),
            "node": nodes(spec.get("nodeName")),
            "owner_kind": p["metadata"]["ownerReferences"][0]["kind"],
            "containers": [{"name": c["name"], "resources": c.get("resources")} for c in spec["containers"]],
            "init_containers": [c["name"] for c in spec.get("initContainers", [])],
            "overhead": spec.get("overhead"),
            "phase": status.get("phase"),
            "scheduled_at": conds.get("PodScheduled"),
            "ready_at": conds.get("Ready"),
        },
        "session_pods_at_recheck": [{"created": i["metadata"]["creationTimestamp"],
                                     "runtimeClassName": i["spec"].get("runtimeClassName"),
                                     "phase": i["status"].get("phase")} for i in need(recheck, "items")],
        "recheck_two_turns": mt,
        "guest_kernel_inside_session_pod": "not read: no uname -r was run inside a BYO-AKS session pod",
        "_sources": [source(tls / n, tls_src[n], "run record") for n in sorted(tls_src)]
                    + [source(retest / "recheck-1815" / "session-pods.json",
                              sha256(retest / "recheck-1815" / "session-pods.json"), "kubectl read-back"),
                       source(retest / "recheck-1815" / "multiturn.json",
                              sha256(retest / "recheck-1815" / "multiturn.json"), "Responses read-back")],
    }

    pm = load(perf / "manifest.json")
    perf_hashes = manifest_hashes(pm)
    perf_src = {}
    for name in ("byoaks-bench.json", "byoaks_bench.py", "cold-start-3x.json", "recovery.json",
                 "model-direct-baseline.json", "nodes-after-scale.txt", "ds-after-scale.txt",
                 "pods-after-cleanup.txt", "bench-console.txt"):
        perf_src[name] = verify(perf / name, perf_hashes[name])
    script = (perf / "byoaks_bench.py").read_text(encoding="utf-8")
    bench_inputs = {"cold": "Reply with exactly COLD-{i}",
                    "warm_first": "Remember the code word LANTERN-7. Reply only OK.",
                    "warm_followup_1": "What is the code word? Reply with the word only.",
                    "warm_followup_n": "Reply with exactly WARM-{i}",
                    "scale_turn1": "Remember SCALE-{i}. Reply only OK.",
                    "scale_turn2": "What did I ask you to remember? Word only."}
    for text in bench_inputs.values():
        if not all(part in script for part in text.split("{i}")):
            raise ExportError(f"bench input not found verbatim in byoaks_bench.py: {text!r}")
    bench = load(perf / "byoaks-bench.json")

    def turn(t: dict) -> dict:
        code = None
        if t.get("error"):
            m = re.search(r'"code":\s*"([a-z_]+)"', t["error"])
            code = m.group(1) if m else t["error"][:60]
        return {"started_utc": t["started_utc"], "http": t["http"], "wall_s": t["wall_s"],
                "first_byte_s": t["first_byte_s"], "status": t["status"], "answer": t["answer"],
                "error_code": code}

    def pod(pd: dict | None) -> dict | None:
        if not pd:
            return None
        return {"name": pods(pd["name"]), "node": nodes(pd["node"]), "phase": pd["phase"],
                "created": pd["created"], "ready_at": pd["ready_at"], "runtime": pd["runtime"],
                "create_to_ready_s": pd["create_to_ready_s"]}

    cold_turns = []
    for t in need(bench, "cold_start", "turns"):
        row = turn(t)
        row["pod"] = pod(t.get("pod"))
        cold_turns.append(row)
    warm_turns = [turn(t) for t in need(bench, "warm", "turns")]
    scale = []
    for level in need(bench, "scalability"):
        sessions = []
        for sess in level["sessions"]:
            sessions.append({"i": sess["i"], "turn1": turn(sess["t1"]),
                             "turn2": turn(sess["t2"]) if sess.get("t2") else None,
                             "remembered": sess["remembered"]})
        scale.append({"concurrency": level["concurrency"], "started_utc": level["started_utc"],
                      "ended_utc": level["ended_utc"], "sessions": sessions, "new_pods": level["new_pods"],
                      "pods_per_node": {nodes(k): v for k, v in level["pods_per_node"].items()}})
    setup_text = need(pm, "setup")
    scale_match = re.search(r"scale op (\d+)m(\d+)s", setup_text)
    latency_text = need(pm, "latency_decomposition")
    agent_match = re.search(r"agent-side (\d+)-(\d+) s", latency_text)
    direct_match = re.search(r"direct model call .*? (\d+) s model / ([\d.]+)-([\d.]+) s wall", latency_text)
    rtt_match = re.search(r"RTT (\d+)-(\d+) ms", latency_text)
    if not all((scale_match, agent_match, direct_match, rtt_match)):
        raise ExportError("D benchmark manifest: setup or latency fields no longer match the export contract")
    scale_seconds = int(scale_match.group(1)) * 60 + int(scale_match.group(2))
    perf_out = {
        "run_id": "d-2026-09-29-perf",
        "path": "D",
        "what": "BYO-AKS cold start, follow-up turns, concurrency and pod loss on a three-node Kata pool",
        "date_utc": "2026-09-29",
        "region": "eastus2",
        "node_sku": need(bench, "node_sku"),
        "agent_pool_nodes": need(bench, "agent_pool_nodes"),
        "session_pods_before": need(bench, "baseline_session_pods"),
        "session_pods_after": len(need(bench, "final_session_pods")),
        "session_pod_shape": "500m CPU / 1Gi request per session, plus 600Mi Kata overhead",
        "client": "Windows laptop in China; direct REST POST to the agent's Responses endpoint with an Azure CLI token",
        "inputs": bench_inputs,
        "request": "POST <agent responses endpoint> {\"input\": <input>} with previous_response_id for follow-ups",
        "clocks": {"wall_s": "client POST until the full response body is read",
                   "create_to_ready_s": "pod creationTimestamp to its Ready condition (Kubernetes timestamps)"},
        "setup_recorded": setup_text,
        "scale_one_to_three_seconds": scale_seconds,
        "cold_turns": cold_turns,
        "warm_turns": warm_turns,
        "warm_note": "warm_turns[0] opens a new session (it is a cold start); warm_turns[1:] are the follow-ups",
        "scalability": scale,
        "node_state_at_failure": need(pm, "scalability", "node_state_at_failure"),
        "max_concurrent_sessions_observed": need(pm, "scalability", "max_concurrent_sessions_observed"),
        "pod_create_to_ready_cached_image_s": need(pm, "cold_start", "pod_create_to_ready_s"),
        "pod_create_to_ready_fresh_node_s": need(pm, "cold_start", "pod_create_to_ready_fresh_node_s"),
        "latency_observation": {
            "agent_created_to_completed_s": [int(agent_match.group(1)), int(agent_match.group(2))],
            "direct_model_internal_s": int(direct_match.group(1)),
            "direct_model_wall_s": [float(direct_match.group(2)), float(direct_match.group(3))],
            "client_rtt_ms": [int(rtt_match.group(1)), int(rtt_match.group(2))],
            "boundary": "Internal endpoint, routing and agent components were not instrumented; subtracting these "
                        "clocks does not produce a measured component time"
        },
        "token": "Azure CLI token acquired before cold and warm, then refreshed before each concurrency level and "
                 "recovery; token acquisition occurs outside each invoke wall_s",
        "_sources": [source(perf / n, perf_src[n], "run record") for n in
                     ("byoaks-bench.json", "bench-console.txt", "byoaks_bench.py")]
                    + [source(perf / "manifest.json", sha256(perf / "manifest.json"), "run manifest")],
    }
    early = load(perf / "cold-start-3x.json")
    cold_early = {
        "run_id": "d-2026-09-29-cold-early",
        "path": "D",
        "what": "Three new sessions right after the pool grew from one to three nodes (15:43-15:44Z)",
        "date_utc": "2026-09-29",
        "runs": [{k: r[k] for k in ("n", "start_utc", "http", "wall_s", "answer")} for r in early],
        "first_pod_on_fresh_node_s": need(pm, "cold_start", "first_pod_on_fresh_node_s"),
        "note": "The first two landed on nodes that had not pulled the image yet; the third on a node that had. "
                "The request text was not saved with these three; the answers COLD-1..3 match the bench's cold input.",
        "_sources": [source(perf / "cold-start-3x.json", perf_src["cold-start-3x.json"], "run record")],
    }
    rec = load(perf / "recovery.json")
    recovery_out = {
        "run_id": "d-2026-09-29-recovery", "path": "D", "date_utc": "2026-09-29",
        "what": "Delete one live session pod; time its replacement and ask a live session again",
        "pod_deleted_utc": need(rec, "pod_deleted_utc"),
        "replacement_ready_s": need(rec, "replacement_ready_s"),
        "followup_after_kill": need(rec, "followup_after_kill"),
        "_sources": [source(perf / "recovery.json", perf_src["recovery.json"], "run record")],
    }
    direct = load(perf / "model-direct-baseline.json")
    direct_out = {
        "run_id": "d-2026-09-29-model-direct", "path": "D", "date_utc": "2026-09-29",
        "what": "The same model called directly from the same client, without an agent, as a baseline",
        "model": "gpt-5.4-mini, low reasoning effort",
        "runs": direct,
        "_sources": [source(perf / "model-direct-baseline.json", perf_src["model-direct-baseline.json"], "run record")],
    }
    node_rows = []
    for line in (perf / "nodes-after-scale.txt").read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split()
        node_rows.append({"node": nodes(cols[0]), "status": cols[1], "kernel": cols[2],
                          "sandbox_node_label": cols[3], "kata_label": cols[4]})
    ds_rows = []
    for line in (perf / "ds-after-scale.txt").read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split()
        ds_rows.append({"daemonset": cols[0], "desired": int(cols[1]), "ready": int(cols[2])})
    left = []
    for line in (perf / "pods-after-cleanup.txt").read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split()
        left.append({"name": pods(cols[0]), "phase": cols[1], "created": cols[2]})
    capacity_out = {
        "run_id": "d-2026-09-29-capacity", "path": "D", "date_utc": "2026-09-29",
        "what": "Nodes, DaemonSets and the session pods still alive after the benchmark and a pod-deletion attempt",
        "nodes_after_scale": node_rows,
        "daemonsets_after_scale": ds_rows,
        "session_pods_after_cleanup": left,
        "cleanup_note": "Session pods deleted at about 16:01Z were recreated by their Deployments within seconds; "
                        "the two pods created at 14:33Z and 14:35Z were never deleted and stayed Running.",
        "_sources": [source(perf / n, perf_src[n], "kubectl read-back") for n in
                     ("nodes-after-scale.txt", "ds-after-scale.txt", "pods-after-cleanup.txt")],
    }

    day2 = load(field / "day2-operations-2026-09-22.json")
    live = load(field / "placement-live-2026-09-24.json")
    steps = {s["step"]: s for s in need(day2, "steps")}
    field_out = {
        "run_id": "d-2026-09-22-field", "path": "D",
        "what": "Day-2 operations on the same cluster before the placement fix (earlier field record)",
        "aks_start_seconds": [need(steps, "az aks start", "elapsed_seconds"),
                              need(live, "cluster", "start_seconds")],
        "aks_start_dates": ["2026-09-22", "2026-09-24"],
        "aks_stop_seconds": need(steps, "az aks stop", "elapsed_seconds"),
        "foundry_system_pods": need(steps, "kubectl inspection", "foundry_system_pods_running"),
        "onboarding_recorded": {
            "role_assignments_seconds_approx": 120,
            "azd_provision_seconds": 522,
            "azd_deploy_seconds": 212,
            "total_seconds_approx": 854,
            "provenance": "author's console, recorded in the earlier field record's README (private repository "
                          "foundry-byo-aks-private-preview at 89fe792); the console itself was not kept"},
        "cost_floor_usd_per_hour": {"system_pool": 0.192, "kata_pool": 0.192, "standard_tier": 0.10,
                                    "region": "eastus2", "source": "Azure retail prices queried 2026-09-21",
                                    "shape": "1 x Standard_D4s_v5 system + 1 x Standard_D4s_v5 Kata + Standard tier"},
        "_sources": [source(field / "day2-operations-2026-09-22.json",
                            sha256(field / "day2-operations-2026-09-22.json"), "run record"),
                     source(field / "placement-live-2026-09-24.json",
                            sha256(field / "placement-live-2026-09-24.json"), "run record")],
    }
    return {"d-byoaks-2026-09-29/placement.json": placement,
            "d-byoaks-2026-09-29/bench.json": perf_out,
            "d-byoaks-2026-09-29/cold-early.json": cold_early,
            "d-byoaks-2026-09-29/recovery.json": recovery_out,
            "d-byoaks-2026-09-29/model-direct.json": direct_out,
            "d-byoaks-2026-09-29/capacity.json": capacity_out,
            "d-byoaks-2026-09-22/field.json": field_out}


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a-remeasure", type=Path, required=True, help="folder of the 2026-10-09 ACA runs")
    ap.add_argument("--abc-repro", type=Path, required=True, help="folder of the 2026-09-27 workshop runs")
    ap.add_argument("--d-retest", type=Path, required=True, help="folder of the 2026-09-29 placement runs")
    ap.add_argument("--d-fix", type=Path, required=True, help="folder of the 2026-09-29 NSG fix and first session")
    ap.add_argument("--d-perf", type=Path, required=True, help="folder of the 2026-09-29 benchmark")
    ap.add_argument("--d-field", type=Path, required=True, help="evidence folder of the earlier field record")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "runs")
    args = ap.parse_args(argv)
    try:
        files = {}
        files.update(export_a(args.a_remeasure, args.abc_repro))
        files.update(export_abc(args.abc_repro))
        files.update(export_d(args.d_retest, args.d_fix, args.d_perf, args.d_field))
    except (ExportError, KeyError, ValueError) as exc:
        print(f"EXPORT FAIL {exc}", file=sys.stderr)
        return 1
    for rel, obj in files.items():
        target = args.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dump(obj), encoding="utf-8", newline="\n")
    index = {rel: hashlib.sha256(dump(obj).encode("utf-8")).hexdigest() for rel, obj in sorted(files.items())}
    (args.out / "INDEX.json").write_text(dump(index), encoding="utf-8", newline="\n")
    print(f"EXPORT PASS {len(files)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
