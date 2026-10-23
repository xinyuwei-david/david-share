#!/usr/bin/env python3
"""Path A: deploy and operate the agent in ACA Sandboxes with the Python SDK (azure-containerapps-sandbox).

Run the steps in order after the workshop's module-04 sandbox-deploy.sh has created the sandbox group:

    python tools/aca_sandbox.py vnet       # connect the group to your delegated subnet
    python tools/aca_sandbox.py image      # build a disk image from the agent image in ACR
    python tools/aca_sandbox.py volume     # register the Blob container as a volume
    python tools/aca_sandbox.py create     # create the sandbox (1000m / 2048Mi, Memory suspend, VNet, volume, port)
    python tools/aca_sandbox.py ask "Where is Bangkok? Answer in one short sentence."
    python tools/aca_sandbox.py stop | resume | show | delete

Settings come from the environment (config/.env.example). Each step prints one JSON line with the
seconds it took; state (disk image ID, sandbox ID) is kept in aca-state.json, which Git ignores.
Exit codes: 0 done, 1 Azure or SDK error, 2 a setting is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "aca-state.json"
POLL_S = 0.25
STATE_MOUNT = "/app/app/data"


class SettingMissing(RuntimeError):
    pass


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value or value.startswith("YOUR-"):
        raise SettingMissing(name)
    return value


def settings() -> dict:
    sub, rg = env("SUBSCRIPTION_ID"), env("RESOURCE_GROUP")
    arm = f"/subscriptions/{sub}/resourceGroups/{rg}/providers"
    return {
        "subscription": sub,
        "resource_group": rg,
        "region": env("LOCATION", "eastus2"),
        "group": env("ACA_SANDBOX_GROUP"),
        "acr_name": env("ACA_ACR_NAME"),
        "image": env("ACA_IMAGE"),
        "uami_id": f"{arm}/Microsoft.ManagedIdentity/userAssignedIdentities/{env('ACA_UAMI_NAME')}",
        "uami_name": env("ACA_UAMI_NAME"),
        "container_id": (f"{arm}/Microsoft.Storage/storageAccounts/{env('ACA_STORAGE_ACCOUNT')}"
                         f"/blobServices/default/containers/{env('ACA_STATE_CONTAINER', 'agent-state')}"),
        "subnet_id": os.environ.get("ACA_SUBNET_ID", ""),
        "vnet_connection": env("ACA_VNET_CONNECTION", "aca-vnet"),
        "volume": env("ACA_VOLUME", "agent-state-blob"),
        "apim_endpoint": env("ACA_APIM_ENDPOINT"),
        "model": env("ACA_MODEL", "gpt-5.4-mini"),
        "allow_cidr": os.environ.get("ACA_ALLOW_CIDR", ""),
    }


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def emit(step: str, seconds: float, **extra) -> dict:
    row = {"step": step, "at": now(), "seconds": round(seconds, 2), **extra}
    print(json.dumps(row, default=str), flush=True)
    return row


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def az(*args: str) -> str:
    out = subprocess.run(["az", *args], capture_output=True, text=True, shell=(os.name == "nt"))
    if out.returncode != 0:
        raise RuntimeError(f"az {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
    return out.stdout.strip()


def credential(cfg: dict):
    from azure.identity import AzureCliCredential  # imported late so --help works without the SDK

    return AzureCliCredential(subscription=cfg["subscription"], process_timeout=300)


def group_client(cfg: dict):
    from azure.containerapps.sandbox import SandboxGroupClient, endpoint_for_region

    return SandboxGroupClient(endpoint_for_region(cfg["region"]), credential(cfg), subscription_id=cfg["subscription"],
                              resource_group=cfg["resource_group"], sandbox_group=cfg["group"])


def agent_environment(cfg: dict, agent_id: str = "agent-host-on-aca") -> dict:
    client_id = os.environ.get("ACA_UAMI_CLIENT_ID") or az(
        "identity", "show", "-g", cfg["resource_group"], "-n", cfg["uami_name"], "--query", "clientId", "-o", "tsv")
    return {"AGENT_ID": agent_id, "AGENT_PORT": "8080", "STATE_MOUNT_PATH": STATE_MOUNT,
            "AGENT_APIM_ENDPOINT": cfg["apim_endpoint"], "LLM_MODEL": cfg["model"],
            "LLM_TOKEN_SCOPE": "https://ai.azure.com/.default", "AZURE_CLIENT_ID": client_id,
            "PYTHONPATH": "/app/deps", "PYTHONUNBUFFERED": "1"}


def port_request(cfg: dict):
    from azure.containerapps.sandbox import (AddPortRequest, PortAuthConfig, PortIpAccessControl,
                                             PortIpAccessControlRule)

    rules = []
    if cfg["allow_cidr"]:
        rules.append(PortIpAccessControlRule(name="operator", action="Allow", priority=100,
                                             source_cidrs=[cfg["allow_cidr"]]))
    return AddPortRequest(port=8080, auth=PortAuthConfig(anonymous=True), activation_mode="OnDemand",
                          ip_access_control=PortIpAccessControl(default_action="Deny", rules=rules))


CONFIGS = ("builtin", "image", "vnet", "port", "vnetvolume", "full", "novolume")


def config_kwargs(name: str, cfg: dict, disk_image_id: str | None, environment: dict | None) -> dict:
    """Create arguments for one configuration; 'full' is the spec measured on 2026-09-27 and 2026-10-09,
    'novolume' is 'full' without the Blob volume."""
    if name not in CONFIGS:
        raise ValueError(f"unknown configuration {name!r}; choose from {', '.join(CONFIGS)}")
    base = dict(disk=None, disk_id=None, snapshot_id=None, preset=None, cpu="1000m", memory="2048Mi",
                disk_size=None, auto_suspend_seconds=900, auto_suspend_mode="Memory",
                labels={"name": f"agent-host-{name}"}, environment=None, connections=None, egress_policy=None,
                volumes=None, ports=None, entrypoint=None, cmd=None, skip_egress_proxy=None,
                customer_vnet_connection_name=None, vmm_type=None)
    if name == "builtin":
        base["disk"] = "ubuntu"
        return base
    if not disk_image_id:
        raise SettingMissing("disk image (run the image step first)")
    base.update(disk_id=disk_image_id, environment=environment,
                entrypoint=["/bin/sh", "-c"], cmd=["cd /app && exec python -m app.main"])
    if name in ("vnet", "vnetvolume", "full", "novolume"):
        base["customer_vnet_connection_name"] = cfg["vnet_connection"]
    if name in ("vnetvolume", "full"):
        from azure.containerapps.sandbox import SandboxVolume

        base["volumes"] = [SandboxVolume(volume_name=cfg["volume"], mountpoint=STATE_MOUNT)]
    if name in ("port", "full", "novolume"):
        base["ports"] = [port_request(cfg)]
    return base


def poll_state(getter, targets: tuple[str, ...], timeout: float = 300):
    """Poll until the sandbox state is one of targets; return (seconds, sandbox, polls)."""
    t0, polls = time.perf_counter(), 0
    while True:
        sbx = getter()
        polls += 1
        state = (sbx.state or "").lower()
        if state in targets:
            return time.perf_counter() - t0, sbx, polls
        if state == "failed":
            raise RuntimeError(f"sandbox state Failed: {sbx.state_details}")
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"still {sbx.state} after {timeout} s")
        time.sleep(POLL_S)


def reflect_code(question: str) -> str:
    """Python one-liner run inside the sandbox: wait (up to 60 s) until the agent accepts a connection on 8080,
    then POST /reflect and time it there. Running can come before the agent listens when no volume delays it."""
    payload = json.dumps({"query": question})
    return ("import json,socket,time,urllib.request as u;"
            f"body={payload!r}.encode();"
            "w=time.time();"
            "up=lambda:(lambda s:(s.connect_ex(('127.0.0.1',8080))==0,s.close())[0])(socket.socket());"
            "any(up() or time.sleep(0.1) for _ in range(600));"
            "w=time.time()-w;"
            "t=time.time();"
            "r=u.Request('http://127.0.0.1:8080/reflect',data=body,headers={'Content-Type':'application/json'});"
            "b=u.urlopen(r,timeout=120).read().decode();"
            "print(json.dumps({'seconds':round(time.time()-t,3),'wait':round(w,3),'body':json.loads(b)}))")


def ask(sandbox, question: str) -> dict:
    t0 = time.perf_counter()
    result = sandbox.exec(f"python3 -c {json.dumps(reflect_code(question))}")
    wall = time.perf_counter() - t0
    if getattr(result, "exit_code", 0):
        raise RuntimeError(f"reflect exit {result.exit_code}: {getattr(result, 'stderr', '') or result.stdout}")
    parsed = json.loads((result.stdout or "").strip().splitlines()[-1])
    body = parsed["body"]
    response = body.get("response") if isinstance(body, dict) else None
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"reflect did not return a non-empty answer: {body}")
    return {"guest_s": round(parsed["seconds"], 2), "ready_wait_s": round(parsed.get("wait", 0), 2),
            "exec_wall_s": round(wall, 2), "response": response, "history_length": body.get("history_length")}


def step_vnet(cfg: dict, state: dict, _args) -> None:
    from azure.containerapps.sandbox import SandboxGroupManagementClient

    if not cfg["subnet_id"] or cfg["subnet_id"].startswith("YOUR-"):
        raise SettingMissing("ACA_SUBNET_ID")
    t = time.perf_counter()
    mgmt = SandboxGroupManagementClient(credential(cfg), subscription_id=cfg["subscription"],
                                        resource_group=cfg["resource_group"])
    mgmt.create_or_update_vnet_connection(cfg["group"], cfg["vnet_connection"], cfg["subnet_id"],
                                          location=cfg["region"])
    emit("vnet_connection", time.perf_counter() - t, name=cfg["vnet_connection"])


def step_image(cfg: dict, state: dict, _args) -> None:
    from azure.containerapps.sandbox import RegistryCredentials

    # A managed-identity pull returned RegistryAuthFailed (401) with SDK 0.1.0b4; a short-lived ACR token works.
    token = json.loads(az("acr", "login", "--name", cfg["acr_name"], "--expose-token", "-o", "json"))["accessToken"]
    creds = RegistryCredentials(username="00000000-0000-0000-0000-000000000000", token=token)
    t = time.perf_counter()
    img = group_client(cfg).begin_create_disk_image(cfg["image"], name="agent-host-latest",
                                                    registry_credentials=creds, polling_timeout=900).result()
    state["disk_image_id"] = img.id
    save_state(state)
    emit("disk_image", time.perf_counter() - t, status=getattr(img.status, "state", None))


def step_volume(cfg: dict, state: dict, _args) -> None:
    from azure.containerapps.sandbox import AzureBlobByoManagedIdentityAuth, SandboxGroupIdentitySelector

    t = time.perf_counter()
    vol = group_client(cfg).create_volume(
        cfg["volume"], type="AzureBlobByo", storage_container_resource_id=cfg["container_id"],
        auth=AzureBlobByoManagedIdentityAuth(
            identity=SandboxGroupIdentitySelector(kind="UserAssigned", resource_id=cfg["uami_id"])))
    emit("volume", time.perf_counter() - t, name=vol.name, type=vol.type)


def step_create(cfg: dict, state: dict, args) -> None:
    kwargs = config_kwargs(args.config, cfg, state.get("disk_image_id"),
                           None if args.config == "builtin" else agent_environment(cfg))
    group = group_client(cfg)
    t = time.perf_counter()
    sandbox, initial = group._do_create_sandbox(**kwargs)  # the call the measured runs timed
    seconds, running, polls = poll_state(lambda: group.get_sandbox(initial.id), ("running",))
    state["sandbox_id"] = initial.id
    save_state(state)
    emit("sandbox_create", time.perf_counter() - t, config=args.config, state=running.state,
         vmm_type=running.vmm_type, polls=polls)


def current_sandbox(cfg: dict, state: dict):
    if not state.get("sandbox_id"):
        raise SettingMissing("sandbox (run the create step first)")
    return group_client(cfg).get_sandbox_client(state["sandbox_id"])


def step_ask(cfg: dict, state: dict, args) -> None:
    sandbox = current_sandbox(cfg, state)
    answer = ask(sandbox, " ".join(args.question) or "Where is Bangkok? Answer in one short sentence.")
    emit("ask", answer["exec_wall_s"], **answer)


def step_stop(cfg: dict, state: dict, _args) -> None:
    sandbox = current_sandbox(cfg, state)
    t = time.perf_counter()
    sandbox.stop()
    _, stopped, _ = poll_state(sandbox.get, ("stopped", "suspended", "idle"))
    emit("stop", time.perf_counter() - t, state=stopped.state)


def step_resume(cfg: dict, state: dict, _args) -> None:
    sandbox = current_sandbox(cfg, state)
    t = time.perf_counter()
    sandbox.resume()
    _, running, _ = poll_state(sandbox.get, ("running",))
    emit("resume", time.perf_counter() - t, state=running.state)


def step_show(cfg: dict, state: dict, _args) -> None:
    group = group_client(cfg)
    t = time.perf_counter()
    sandboxes = [{"state": s.state, "ours": s.id == state.get("sandbox_id")} for s in group.list_sandboxes()]
    info = {}
    if state.get("sandbox_id"):
        got = group.get_sandbox(state["sandbox_id"])
        info = {"state": got.state, "vmm_type": got.vmm_type, "ports": len(got.ports or []),
                "vnet": got.customer_vnet_connection_name}
    emit("show", time.perf_counter() - t, sandboxes=sandboxes, sandbox=info)


def step_delete(cfg: dict, state: dict, _args) -> None:
    sandbox_id = state.get("sandbox_id")
    if not sandbox_id:
        emit("delete", 0.0, deleted=False)
        return
    t = time.perf_counter()
    group_client(cfg).delete_sandbox(sandbox_id)
    state.pop("sandbox_id")
    save_state(state)
    emit("delete", time.perf_counter() - t, deleted=True)


STEPS = {"vnet": step_vnet, "image": step_image, "volume": step_volume, "create": step_create, "ask": step_ask,
         "stop": step_stop, "resume": step_resume, "show": step_show, "delete": step_delete}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("question", nargs="*", help="for ask: the question sent to the agent's /reflect")
    ap.add_argument("--config", default="full", choices=CONFIGS, help="for create: which configuration")
    args = ap.parse_args(argv)
    try:
        cfg = settings()
        STEPS[args.step](cfg, load_state(), args)
    except SettingMissing as exc:
        print(f"SETTING MISSING {exc}: set it in .env (see config/.env.example)", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - every Azure/SDK failure ends the step with exit 1
        print(f"STEP FAILED {args.step}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
