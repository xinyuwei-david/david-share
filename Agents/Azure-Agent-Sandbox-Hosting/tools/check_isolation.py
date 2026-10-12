#!/usr/bin/env python3
"""Paths C and D: check that agent pods run as Kata micro-VMs on your nodes.

    python tools/check_isolation.py -n "$NAMESPACE" -l app=agent-host --exec-uname          # C
    python tools/check_isolation.py -n "$SESSION_NAMESPACE" --container agent --exec-uname   # D

For every matching pod it prints one line:

    CHECK isolation PASS pod=<pod> runtimeClass=kata-vm-isolation node=<node> nodeKernel=<k> podKernel=<k>

PASS needs runtimeClassName kata-vm-isolation and phase Running. With --exec-uname the pod's own kernel
(uname -r inside the pod) is compared with the kernelVersion of its node: a different string proves the pod does
not share the node's kernel (a runc container always reports the node's); the same string is reported as
inconclusive. If exec is refused, the kernel is reported as not read.
Exit codes: 0 every pod PASS, 1 any pod FAIL, 3 no pod matched.

Offline: --pods-json and --nodes-json evaluate saved `kubectl get pods/nodes -o json` output, and
--uname POD=KERNEL supplies kernels read earlier.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

KATA = "kata-vm-isolation"


def evaluate(pods_doc: dict, nodes_doc: dict, unames: dict[str, str | None]) -> list[dict]:
    kernels = {n["metadata"]["name"]: n.get("status", {}).get("nodeInfo", {}).get("kernelVersion")
               for n in nodes_doc.get("items", [])}
    results = []
    for p in pods_doc.get("items", []):
        name = p["metadata"]["name"]
        node = p.get("spec", {}).get("nodeName")
        runtime = p.get("spec", {}).get("runtimeClassName")
        node_kernel = kernels.get(node)
        pod_kernel = unames.get(name)
        reasons = []
        if runtime != KATA:
            reasons.append(f"runtimeClassName is {runtime or 'unset (runc)'}")
        if p.get("status", {}).get("phase") != "Running":
            reasons.append(f"phase {p.get('status', {}).get('phase')}")
        if not pod_kernel or not node_kernel:
            kernel = "not read"
        elif pod_kernel != node_kernel:
            kernel = "differs from node"
        else:
            kernel = "same version string as node (inconclusive)"
        results.append({"pod": name, "runtime_class": runtime, "node": node, "node_kernel": node_kernel,
                        "pod_kernel": pod_kernel, "kernel": kernel,
                        "verdict": "FAIL" if reasons else "PASS", "reasons": reasons})
    return results


def line(r: dict) -> str:
    text = (f"CHECK isolation {r['verdict']} pod={r['pod']} runtimeClass={r['runtime_class'] or 'runc'} "
            f"node={r['node']} nodeKernel={r['node_kernel'] or 'unknown'} podKernel={r['pod_kernel'] or 'not read'} "
            f"kernel={r['kernel'].replace(' ', '-')}")
    return text + (f" ({'; '.join(r['reasons'])})" if r["reasons"] else "")


def kubectl_json(*args: str) -> dict:
    out = subprocess.run(["kubectl", *args, "-o", "json", "--request-timeout=60s"], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"kubectl {' '.join(args)} failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def read_uname(namespace: str, pod: str, container: str | None) -> str | None:
    cmd = ["kubectl", "exec", "-n", namespace, pod]
    if container:
        cmd += ["-c", container]
    out = subprocess.run(cmd + ["--", "uname", "-r"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-n", "--namespace", help="namespace of the agent or session pods")
    ap.add_argument("-l", "--selector", help="label selector, for example app=agent-host")
    ap.add_argument("--container", help="container to exec into (D: agent)")
    ap.add_argument("--exec-uname", action="store_true", help="read uname -r inside each pod")
    ap.add_argument("--pods-json", type=Path, help="offline: saved kubectl get pods -o json")
    ap.add_argument("--nodes-json", type=Path, help="offline: saved kubectl get nodes -o json")
    ap.add_argument("--uname", action="append", default=[], metavar="POD=KERNEL", help="offline: a kernel read earlier")
    args = ap.parse_args(argv)
    if args.pods_json:
        pods = json.loads(args.pods_json.read_text(encoding="utf-8"))
        nodes = json.loads(args.nodes_json.read_text(encoding="utf-8")) if args.nodes_json else {"items": []}
        unames = dict(u.split("=", 1) for u in args.uname)
    else:
        if not args.namespace:
            ap.error("--namespace is required unless --pods-json is given")
        pods = kubectl_json("get", "pods", "-n", args.namespace, *(["-l", args.selector] if args.selector else []))
        nodes = kubectl_json("get", "nodes")
        unames = {}
        if args.exec_uname:
            for p in pods.get("items", []):
                unames[p["metadata"]["name"]] = read_uname(args.namespace, p["metadata"]["name"], args.container)
    results = evaluate(pods, nodes, unames)
    if not results:
        print("CHECK isolation BLOCKED no pod matched")
        return 3
    for r in results:
        print(line(r))
    return 1 if any(r["verdict"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
