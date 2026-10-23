#!/usr/bin/env python3
"""Path C measurement: agent-sandbox Sandboxes on the AKS Kata pool, timed with kubectl and inside the pod.

    python tools/measure_c.py --cold 3 --pvc 3 --scale 4,8 --out-dir <folder>

Every test Sandbox is a copy of the workshop's `agent-host` Sandbox spec (same image, service account, model
gateway, Kata runtime class and resources) under a new name and the label app=agent-host-measure, so the
workshop's own Sandbox and LoadBalancer are not touched.

  cold   (no external volume), per run: kubectl apply -> Sandbox Ready -> first answer -> follow-up -> delete.
  pvc    (the workshop's private Blob PVC), per run: apply -> Ready -> first answer -> follow-up -> delete the pod
         -> Ready again -> answer -> operatingMode Suspended -> pod gone -> operatingMode Running -> Ready ->
         answer -> delete. The conversation length must keep growing (state on the Blob volume).
  scale  per level N: N Sandboxes without the volume applied at once; each answers "Remember SCALE-<k>. Reply only
         OK." then "What did I ask you to remember? Word only." (A's, D's and E's concurrency inputs) and counts as
         answered when the second reply contains SCALE-<k>; the node of every pod is recorded.

Answers are timed inside the pod (POST 127.0.0.1:8080/reflect run through kubectl exec, the same code A uses
inside its sandbox), so the kubectl round trip is excluded; Ready and suspend times are client wall time
between the kubectl call and the observed state (polled every 0.5 s), with the server's condition timestamps
recorded alongside. Writes <out-dir>/c-<UTC time>.json. Exit codes: 0 every requested run and session passed and
every test Sandbox was deleted, 1 otherwise, 2 arguments are invalid.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aca_sandbox import reflect_code  # noqa: E402 - the same in-guest request A uses

FIRST = "Where is Bangkok? Answer in one short sentence."
COUNT = "How many questions have I asked before this one? Number only."
SCALE_WRITE = "Remember SCALE-{k}. Reply only OK."
SCALE_READ = "What did I ask you to remember? Word only."
LABEL = "agent-host-measure"
POLL_S = 0.5


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def kubectl(*args: str, stdin: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], input=stdin, capture_output=True, text=True, timeout=timeout)


def kjson(*args: str) -> dict | None:
    out = kubectl(*args, "-o", "json", "--request-timeout=60s")
    if out.returncode != 0:
        if "NotFound" in out.stderr or "not found" in out.stderr:
            return None
        raise RuntimeError(f"kubectl {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def condition(obj: dict | None, kind: str = "Ready") -> dict:
    for c in ((obj or {}).get("status") or {}).get("conditions") or []:
        if c.get("type") == kind:
            return c
    return {}


def sandbox_manifest(template: dict, name: str, ns: str, pvc: bool, tag: str) -> dict:
    spec = copy.deepcopy(template["spec"])
    spec["operatingMode"] = "Running"
    pod = spec["podTemplate"]
    pod.setdefault("metadata", {}).setdefault("labels", {})["app"] = LABEL
    if not pvc:
        claims = {v["name"] for v in pod["spec"].get("volumes") or [] if "persistentVolumeClaim" in v}
        pod["spec"]["volumes"] = [v for v in pod["spec"].get("volumes") or [] if v["name"] not in claims] or None
        if pod["spec"]["volumes"] is None:
            del pod["spec"]["volumes"]
        for c in pod["spec"]["containers"]:
            mounts = [m for m in c.get("volumeMounts") or [] if m["name"] not in claims]
            if mounts:
                c["volumeMounts"] = mounts
            else:
                c.pop("volumeMounts", None)
    return {"apiVersion": template["apiVersion"], "kind": "Sandbox",
            "metadata": {"name": name, "namespace": ns, "labels": {"app": LABEL, "measure-run": tag}}, "spec": spec}


class Sandbox:
    def __init__(self, ns: str, name: str, container: str):
        self.ns, self.name, self.container = ns, name, container

    def get(self) -> dict | None:
        return kjson("get", "sandbox", self.name, "-n", self.ns)

    def pod(self) -> dict | None:
        p = kjson("get", "pod", self.name, "-n", self.ns)
        if p is not None:
            return p
        pods = kjson("get", "pods", "-n", self.ns, "-l", f"app={LABEL}") or {"items": []}
        for item in pods["items"]:
            if any(o.get("kind") == "Sandbox" and o.get("name") == self.name
                   for o in item["metadata"].get("ownerReferences") or []):
                return item
        return None

    def wait(self, test, limit: float, what: str) -> float:
        t0 = time.perf_counter()
        while True:
            if test():
                return round(time.perf_counter() - t0, 2)
            if time.perf_counter() - t0 > limit:
                raise TimeoutError(f"{self.name}: {what} not reached after {limit:.0f} s")
            time.sleep(POLL_S)

    def ready(self, old_uid: str | None = None) -> bool:
        if condition(self.get()).get("status") != "True":
            return False
        p = self.pod()
        return bool(p) and p["metadata"]["uid"] != old_uid and condition(p).get("status") == "True"

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        pod = self.pod()
        if not pod:
            raise RuntimeError(f"{self.name}: no pod to ask")
        out = kubectl("exec", "-n", self.ns, pod["metadata"]["name"], "-c", self.container, "--request-timeout=60s",
                      "--", "python3", "-c", reflect_code(question), timeout=180)
        if out.returncode != 0:
            raise RuntimeError(f"{self.name}: reflect exec failed: {out.stderr.strip()[:300]}")
        parsed = json.loads(out.stdout.strip().splitlines()[-1])
        body = parsed["body"]
        response = body.get("response") if isinstance(body, dict) else None
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"{self.name}: /reflect returned no answer: {str(body)[:200]}")
        return {"guest_s": round(parsed["seconds"], 2), "exec_wall_s": round(time.perf_counter() - t0, 2),
                "response": response.strip()[:160], "history_length": body.get("history_length")}

    def server_times(self) -> dict:
        sbx, pod = self.get() or {}, self.pod() or {}
        return {"sandbox_created": sbx.get("metadata", {}).get("creationTimestamp"),
                "sandbox_ready": condition(sbx).get("lastTransitionTime"),
                "pod_created": pod.get("metadata", {}).get("creationTimestamp"),
                "pod_ready": condition(pod).get("lastTransitionTime"), "node": pod.get("spec", {}).get("nodeName")}

    def patch_mode(self, mode: str) -> None:
        out = kubectl("patch", "sandbox", self.name, "-n", self.ns, "--type", "merge", "--request-timeout=60s",
                      "-p", json.dumps({"spec": {"operatingMode": mode}}))
        if out.returncode != 0:
            raise RuntimeError(f"{self.name}: patch {mode} failed: {out.stderr.strip()[:300]}")

    def delete(self) -> bool:
        out = kubectl("delete", "sandbox", self.name, "-n", self.ns, "--wait=false", "--request-timeout=60s")
        return out.returncode == 0


def apply(objs: list[dict]) -> None:
    out = kubectl("apply", "-f", "-", "--request-timeout=60s",
                  stdin=json.dumps({"apiVersion": "v1", "kind": "List", "items": objs}))
    if out.returncode != 0:
        raise RuntimeError(f"kubectl apply failed: {out.stderr.strip()[:300]}")


def history_grew(turns: list[dict]) -> bool:
    lengths = [t.get("history_length") for t in turns]
    return all(isinstance(x, int) for x in lengths) and all(b > a for a, b in zip(lengths, lengths[1:]))


def cold_run(template, ns, container, tag, i, pvc: bool) -> dict:
    name = f"{LABEL}-{'pvc' if pvc else 'cold'}-{i}-{int(time.time())}"
    sbx, rec = Sandbox(ns, name, container), {"run": i, "sandbox": name, "pvc": pvc, "started_at": now()}
    try:
        t0 = time.perf_counter()
        apply([sandbox_manifest(template, name, ns, pvc, tag)])
        sbx.wait(sbx.ready, 600, "Ready")
        rec["apply_to_ready_s"] = round(time.perf_counter() - t0, 2)
        rec["server"] = sbx.server_times()
        rec["first"] = sbx.ask(FIRST)
        rec["cold_total_s"] = round(rec["apply_to_ready_s"] + rec["first"]["guest_s"], 2)
        rec["follow_up"] = sbx.ask(COUNT)
        turns = [rec["first"], rec["follow_up"]]
        if pvc:
            uid = sbx.pod()["metadata"]["uid"]
            t1 = time.perf_counter()
            kubectl("delete", "pod", sbx.pod()["metadata"]["name"], "-n", ns, "--wait=false", "--request-timeout=60s")
            sbx.wait(lambda: sbx.ready(uid), 600, "Ready after pod delete")
            rec["pod_delete_to_ready_s"] = round(time.perf_counter() - t1, 2)
            rec["after_pod_delete"] = sbx.ask(COUNT)
            turns.append(rec["after_pod_delete"])
            uid = sbx.pod()["metadata"]["uid"]
            t2 = time.perf_counter()
            sbx.patch_mode("Suspended")
            sbx.wait(lambda: sbx.pod() is None, 600, "pod gone after Suspended")
            rec["suspend_to_pod_gone_s"] = round(time.perf_counter() - t2, 2)
            rec["suspended_condition"] = {k: condition(sbx.get()).get(k) for k in ("status", "reason")}
            time.sleep(5)
            t3 = time.perf_counter()
            sbx.patch_mode("Running")
            sbx.wait(lambda: sbx.ready(uid), 600, "Ready after Running")
            rec["resume_to_ready_s"] = round(time.perf_counter() - t3, 2)
            rec["after_resume"] = sbx.ask(COUNT)
            rec["resume_to_answer_s"] = round(rec["resume_to_ready_s"] + rec["after_resume"]["guest_s"], 2)
            turns.append(rec["after_resume"])
        rec["history_grew"] = history_grew(turns)
        if not rec["history_grew"]:
            raise RuntimeError(f"conversation length did not keep growing: {[t.get('history_length') for t in turns]}")
    except Exception as exc:  # noqa: BLE001 - recorded, the run is marked failed
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["deleted"] = sbx.delete()
        rec["finished_at"] = now()
    print(json.dumps({k: v for k, v in rec.items() if k not in ("server",)}, default=str)[:600], flush=True)
    return rec


def scale_session(template, ns, container, tag, level: int, k: int, t_apply: float) -> dict:
    name = f"{LABEL}-scale{level}-{k}-{tag[-6:]}"
    sbx, rec = Sandbox(ns, name, container), {"slot": k, "sandbox": name}
    try:
        sbx.wait(sbx.ready, 900, "Ready")
        rec["apply_to_ready_s"] = round(time.perf_counter() - t_apply, 2)
        rec["server"] = sbx.server_times()
        rec["turn1"] = sbx.ask(SCALE_WRITE.format(k=k))
        rec["turn2"] = sbx.ask(SCALE_READ)
        rec["remembered"] = f"SCALE-{k}" in rec["turn2"]["response"]
    except Exception as exc:  # noqa: BLE001 - the slot counts as not answered
        rec.update(error=f"{type(exc).__name__}: {exc}", remembered=False)
        pod = sbx.pod()
        rec["pod_phase"] = (pod or {}).get("status", {}).get("phase")
        rec["pod_reason"] = [c.get("message", "")[:160] for c in (pod or {}).get("status", {}).get("conditions") or []
                             if c.get("status") == "False"][:2]
    return rec


def at_once(template, ns, container, tag, level: int) -> dict:
    names = [f"{LABEL}-scale{level}-{k}-{tag[-6:]}" for k in range(level)]
    t0 = time.perf_counter()
    apply([sandbox_manifest(template, n, ns, False, tag) for n in names])
    with cf.ThreadPoolExecutor(max_workers=level) as pool:
        sessions = list(pool.map(lambda k: scale_session(template, ns, container, tag, level, k, t0), range(level)))
    wall = round(time.perf_counter() - t0, 2)
    deleted = sum(Sandbox(ns, n, container).delete() for n in names)
    nodes = sorted({(s.get("server") or {}).get("node") for s in sessions if (s.get("server") or {}).get("node")})
    out = {"sessions_at_once": level, "answered": sum(bool(s.get("remembered")) for s in sessions), "wall_s": wall,
           "nodes": nodes, "deleted": deleted, "sessions": sessions}
    print(json.dumps({k: v for k, v in out.items() if k != "sessions"}), flush=True)
    return out


def wait_gone(ns: str, tag: str, limit: float = 600) -> int:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        left = kjson("get", "sandbox", "-n", ns, "-l", f"measure-run={tag}") or {"items": []}
        if not left["items"]:
            return 0
        time.sleep(2)
    return len(left["items"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", default="agent", help="the workshop namespace (module-03 default: agent)")
    ap.add_argument("--template", default="agent-host", help="Sandbox whose spec every test Sandbox copies")
    ap.add_argument("--cold", type=int, default=3, help="runs without the external volume")
    ap.add_argument("--pvc", type=int, default=3, help="runs with the workshop's Blob PVC, including pod loss and "
                                                       "suspend/resume")
    ap.add_argument("--scale", default="4,8", help="comma list of Sandboxes at once (without the volume)")
    ap.add_argument("--pause", type=int, default=20, help="seconds between runs and phases")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "evidence-local")
    a = ap.parse_args(argv)
    try:
        levels = [int(x) for x in a.scale.split(",") if x.strip()]
    except ValueError:
        ap.error("--scale must be a comma list of positive integers")
    if a.cold < 0 or a.pvc < 0 or any(n < 1 for n in levels):
        ap.error("--cold and --pvc must be >= 0 and --scale levels >= 1")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
    template = kjson("get", "sandbox", a.template, "-n", a.namespace)
    if template is None:
        print(f"template Sandbox {a.template} not found in namespace {a.namespace}", file=sys.stderr)
        return 2
    container = template["spec"]["podTemplate"]["spec"]["containers"][0]["name"]
    nodes = kjson("get", "nodes", "-l", "kata-containers=true") or {"items": []}
    report: dict = {
        "started_at": now(), "tag": tag, "namespace": a.namespace, "template": a.template,
        "image": template["spec"]["podTemplate"]["spec"]["containers"][0].get("image"),
        "runtime_class": template["spec"]["podTemplate"]["spec"].get("runtimeClassName"),
        "resources": template["spec"]["podTemplate"]["spec"]["containers"][0].get("resources"),
        "kata_nodes_at_start": [{"name": n["metadata"]["name"], "vm_size": n["metadata"]["labels"].get(
            "node.kubernetes.io/instance-type"), "kubelet": n["status"]["nodeInfo"]["kubeletVersion"],
            "kernel": n["status"]["nodeInfo"]["kernelVersion"]} for n in nodes["items"]],
        "inputs": {"first": FIRST, "follow_up": COUNT, "scale_write": SCALE_WRITE, "scale_read": SCALE_READ},
        "cold": [], "pvc": [], "scale": []}
    for i in range(1, a.cold + 1):
        report["cold"].append(cold_run(template, a.namespace, container, tag, i, pvc=False))
        time.sleep(a.pause)
    for i in range(1, a.pvc + 1):
        report["pvc"].append(cold_run(template, a.namespace, container, tag, i, pvc=True))
        time.sleep(a.pause)
    for n, level in enumerate(levels):
        if n:
            wait_gone(a.namespace, tag)
            time.sleep(a.pause)
        report["scale"].append(at_once(template, a.namespace, container, tag, level))
    report["leftover_sandboxes"] = wait_gone(a.namespace, tag)
    nodes = kjson("get", "nodes", "-l", "kata-containers=true") or {"items": []}
    report["kata_nodes_at_end"] = sorted(n["metadata"]["name"] for n in nodes["items"])
    report["finished_at"] = now()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"c-{tag}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    failed = [r for r in report["cold"] + report["pvc"] if "error" in r or not r.get("deleted")]
    failed += [s for lv in report["scale"] for s in lv["sessions"] if not s.get("remembered")]
    total = len(report["cold"]) + len(report["pvc"]) + sum(len(lv["sessions"]) for lv in report["scale"])
    ok = not failed and not report["leftover_sandboxes"]
    print(f"{'MEASURE PASS' if ok else 'MEASURE FAIL'} {total - len(failed)}/{total} -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
