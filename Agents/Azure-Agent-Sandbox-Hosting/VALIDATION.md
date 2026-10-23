# Validation Details for Reviewers

[中文](VALIDATION_CN.md) · Customer steps are in [README.md](README.md)

This page is for readers who want to check the numbers: how each path was measured, the input and result of every run, what was not tested, and how to recompute everything offline. A customer following the steps does not need it.

## Architecture and Metrics

The narrative is organized around two questions: who runs the session lifecycle, and whose node the session's micro-VM runs on. The workload contract is comparable, but the container is not identical across all five paths: A and C use the workshop's [module-03 reflection agent](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-03/agent-src/app/main.py); B uses its distinct [module-02 Agent Framework agent with ResponsesHostServer](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-02/agent-src/main.py); D uses the private-preview sample `agent-framework-agent-basic-responses-byoaks`; E uses a Foundry Prompt Agent whose function tool runs its code in an E2B sandbox. All call `gpt-5.4-mini` and exercise a new stateful session plus a follow-up.

- **A · ACA Sandboxes.** The agent's container image becomes a sandbox disk image; one sandbox per measured session, `cpu 1000m`, `memory 2048Mi` (the M tier), auto-suspend after 900 s in Memory mode, a VNet connection, a private Blob volume at `/app/app/data` and port 8080 in on-demand mode. The control plane is ARM (`Microsoft.App/sandboxGroups`), the data plane `management.azuredevcompute.io`; the caller needs *Container Apps SandboxGroup Data Owner*.
- **B · Foundry hosted agent.** `azd ai agent init` binds the workshop's `azure.yaml` to the existing project and `azd deploy` builds and publishes a version. Learn describes per-session VM-isolated sandboxes, `$HOME` and `/files` persisted across idle, an idle timeout of 2–60 minutes (default 15) and deletion after 30 days of inactivity.
- **C · AKS + agent-sandbox.** AKS with Workload Identity and the Blob CSI driver, an Azure Linux node pool with `KataVmIsolation` (`Standard_D4s_v3` in our run), agent-sandbox `v0.5.2` (`agents.x-k8s.io/v1beta1`, kind `Sandbox`), a Blob PVC for state and a LoadBalancer restricted to one source address. Suspend and resume are `spec.operatingMode: Suspended | Running`; the pod is deleted and file state is restored from Blob, not from a memory snapshot. Learn names the stack: Azure Linux host, Microsoft Hyper-V, Cloud Hypervisor VMM, Kata Containers.
- **D · BYO-AKS.** The initial one-Kata-node read-back had 13 runtime pods in `foundry-system`: 9 fixed Deployment replicas (router ×3 behind a LoadBalancer on `:443`, controller ×2, identity proxy ×2, telemetry controller ×2) plus 4 DaemonSets. Each DaemonSet adds one pod per matching Kata node, so the total grows with the node pool. Each session is its own Deployment, ReplicaSet and pod with `runtimeClassName: kata-vm-isolation` in a per-project namespace.
- **E · E2B on Azure VM.** An Ubuntu 24.04 LTS Gen2 VM (`Standard_D4ds_v7`, or the `Standard_D8ds_v7` of `e-2026-10-18-ci-scale`, with nested virtualization `/dev/kvm`) runs the open-source E2B Embed Docker Compose stack (`e2b-dev/runtime` at `588e5b9`). In the comparison, E2B's Code Interpreter `Sandbox.create()` boots a `code-interpreter-v1` Firecracker `v1.14-0.2.0` micro-VM (2 vCPU, 2048 MB, guest kernel `6.1.177+`) and the tool runs the model's Python with `run_code` in its Jupyter kernel; the runs before 2026-10-17 used `Sandbox.create("base")` (2 vCPU, 512 MB) and `commands.run("python3")`, and the runs of 2026-10-17 and 2026-10-18 supersede them. `beta_pause` snapshots memory and disk (`vm_state` + `mem_file`) to the VM's OS disk, where E2B also keeps its templates, and `Sandbox.connect` restores it. A Foundry Prompt Agent (`gpt-5.4-mini`) calls the sandbox via a function tool over ports `3000` and `3002`. Embed is [one machine by design](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/README.md#when-one-node-is-not-enough), answers over plain HTTP, and has no persistent data volume; `e-2026-10-18-ci-multi` runs two independent installs side by side, as `e-2026-10-15-multi` did with `base`.

Why these clocks:

| Metric | Start | End | Measured where |
|---|---|---|---|
| Cold, A (component sum) | SDK create call → Running | add first `/reflect` answer duration | create on the client; answer inside the sandbox |
| Cold, B and D | client POST of a new session | full response body read | client, through the Foundry endpoint |
| Cold, C (component sum) | `kubectl apply` → Sandbox Ready | add the first answer's duration, at the LoadBalancer (2026-09-27) or inside the pod (2026-10-18) | client; the answer through its own LoadBalancer or with `kubectl exec` inside the pod; model inference still uses APIM |
| Warm | request of the next turn | its answer | as for cold |
| Resume, A (component sum) | SDK resume call → Running | add first answer duration after resume | resume on the client; answer inside |
| Resume, C | `operatingMode: Running` patch | first answer at the LoadBalancer (2026-09-27); Ready plus the answer timed inside the pod (2026-10-18, component sum) | client |
| Pod lost | pod deleted | replacement pod Ready (Kubernetes timestamps; C on 2026-10-18: polled from the client) | cluster |
| At once, A and C (2026-10-18) | N creates or applies at the same time | each sandbox's second answer, timed inside it; the level ends when the last one is deleted | create, apply and delete on the client; answers inside |
| B and D session API: create, stop | `POST .../endpoint/sessions`, `POST .../sessions/{id}:stop` | the session reads `active`, `idle` | client; failed attempts and their 2 s pauses included |
| Cold, E A protocol (component sum) | `Sandbox.create` (`code-interpreter-v1`; `base` before 2026-10-17) → it returns | add the first Prompt Agent turn: first `POST /openai/v1/responses` until the final answer, every tool round trip included | the tool's process; on the E2B host VM in E's runs |
| Cold, E D protocol | conversation create | the turn's final answer, with `Sandbox.create` in between (one stopwatch) | as above |
| Resume, E (component sum) | `Sandbox.connect` → it returns | add the first turn after it | as above |
| Pause, E | `beta_pause` | it returns | as above |
| Host mode, E (2026-10-18) | `Sandbox.create`, a `run_code` cell, `beta_pause` or `Sandbox.connect` | it returns | the tool on the E2B host itself, no model |

E's clocks and those of `b-2026-10-15-sessions` and of every 2026-10-18 run are `time.perf_counter()` in the measuring process; a range is the minimum to the maximum. A, E and C's 2026-10-18 runs wait 5 s between the end of stop, pause or suspend and the start of resume, outside every clock. The E concurrency tables give sandbox create plus turn 1, the clock of the 2026-10-14 driver.

A's answers are timed inside the sandbox because the 2026-09-27 run could not reach the port from the operator address (the port's IP rule saw rotating Azure egress addresses). The paths therefore do not share a network path, and a row compares implementations, not networks.

A's cold is `create_total_s + first.guest_s` and resume-to-answer is `resume_total_s + first_after_resume.guest_s`: component sums, not a single end-to-end client stopwatch. C's cold is also a component sum. For both A and C, stop and idle waiting are excluded from resume-to-answer.

The README's comparison uses only the ready clocks: create → ready, resume → ready, and pod or sandbox lost → ready. The answer clocks stay on this page: an answer time mostly measures the model, the agent and the network path, and a question that needs no code, such as `"Where is Bangkok?"`, does not show that code ran in the sandbox.

## What the Scripts Run

[`tools/aca_sandbox.py`](tools/aca_sandbox.py) (A), with `azure-containerapps-sandbox` 0.1.0b4:

| Step | SDK or CLI call |
|---|---|
| `vnet` | `SandboxGroupManagementClient.create_or_update_vnet_connection(group, ACA_VNET_CONNECTION, ACA_SUBNET_ID)` |
| `image` | `az acr login --expose-token`, then `SandboxGroupClient.begin_create_disk_image(ACA_IMAGE, registry_credentials=…)` |
| `volume` | `create_volume(ACA_VOLUME, type="AzureBlobByo", storage_container_resource_id=…, auth=UserAssigned identity)` |
| `create` | `_do_create_sandbox(...)` with the configuration above, then poll `get_sandbox` every 0.25 s until Running |
| `ask` | `sandbox.exec("python3 -c …")`, which POSTs `{"query": …}` to `http://127.0.0.1:8080/reflect` and times it inside |
| `stop`, `resume`, `delete` | `sandbox.stop()` / `resume()` and poll; `delete_sandbox(id)` |

[`tools/measure_aca.py`](tools/measure_aca.py) repeats the 2026-10-09 protocol: `agent` mode runs create → answer → answer → stop → resume → answer → answer → delete per run and fails if conversation-history lengths do not keep growing (`--config novolume` runs the same without the Blob volume, `a-2026-10-18-novolume`); every request first waits inside the sandbox until the agent accepts a connection on port 8080, and that wait is counted in the cold time; `create` mode times each configuration (`builtin`, `image`, `vnet`, `port`, `vnetvolume`, `full`, `novolume`) and records whether guest uptime after resume exceeds its pre-stop value. Uptime alone does not prove process-memory continuity; the retained `process_survived` field is a legacy name for that observation. `scale` mode creates the sandboxes of each `--scale` level at the same time with the full specification; each asks D's and E's two concurrency questions and counts as answered when its second answer names its own word, and every sandbox is deleted before the next level (`a-2026-10-18-scale`).

[`tools/measure_responses.py`](tools/measure_responses.py) (B and D) POSTs `{"input": …}` to `AGENT_RESPONSES_URL` with an Azure CLI token for `https://ai.azure.com` and adds `previous_response_id` for follow-ups. Token acquisition is outside the request timer; it refreshes before each concurrency level and recovery. With `--observe`, it accepts only one new Ready Kata pod as a candidate; ambiguity or a failed pod read is a failure, not a placement result. Run this observation in a quiet project namespace. `--recovery` deletes the one newly observed Kata pod, waits for deletion to finish, then follows up using that session's response ID. Exit 0 requires every requested cold, warm, concurrency and recovery check to pass; skipped recovery, deletion failure and forgotten state return exit 1. With `RESPONSES_AUTH_HEADER` (`Header-Name: value`, for example an API Management key) it sends that header instead of the token, as in `b-2026-10-18-perf`; the value is never printed or written.

[`tools/check_isolation.py`](tools/check_isolation.py) (C and D) reads `kubectl get pods` and `kubectl get nodes`, requires `runtimeClassName: kata-vm-isolation` and phase Running, and with `--exec-uname` compares `uname -r` inside the pod with the node's `kernelVersion`.

[`tools/measure_e2b.py`](tools/measure_e2b.py) (E) creates its own Prompt Agent version (the instructions and tool schema printed in every output), runs the A-matched 3-run `agent` protocol (`Sandbox.create` → first answer → follow-up → `beta_pause` → wait 5 s → `Sandbox.connect` → follow-ups → `kill`), the D-matched `perf` protocol (cold n=5, warm follow-ups with `LANTERN-7`, concurrency 4 and 8, and sandbox replacement), `scale` (sessions spread over several Embed installs named in `--hosts`) and `hostloss` (stop one install's stack and record what the others and the paused sandboxes can do), then deletes the agent version, every conversation and every sandbox. Exit 1 if any phase failed, lost state or left a resource. The 2026-10-15 runs used its revision `036dc81…`, except host b's agent runs, which used the earlier `89255d0b…`. The repository revision changes only failure handling and keeps the prompts, protocol and clocks: exit 1 when the recovery follow-up loses the code word, or when `hostloss` loses what should survive (the other host's live session and new sessions, the stopped host's paused sandbox and new sessions after restart) or leaves a sandbox or conversation; cleanup when a create fails midway; exit 2 for an unreadable `--hosts` file. The 2026-10-14 numbers came from the author's private driver, whose agent, prompts and clocks it reproduces. The tool runs the model's Python only with E2B's Code Interpreter: it creates the sandbox with `e2b-code-interpreter` in `code-interpreter-v1` and runs one `run_code` cell per tool call (revision `cbc96f2…` on 2026-10-17, `aca40d1…` on 2026-10-18); `--executor` accepts only `code-interpreter`, so the recorded 2026-10-17 commands still run. Every E run before 2026-10-17 ran the model's Python as `python3` with `commands.run` in a `base` sandbox (the tool's method up to commit `7c9a13b`); those runs are superseded. `host` mode runs on the E2B host without a model: one sandbox reads its kernel, vCPUs and memory while the tool lists the host's Firecracker processes; `--bench n` times n create → cell → pause → resume → cell cycles; `--reserve k` creates up to k sandboxes one by one and reads `/proc/meminfo` after each until a create is refused (`e-2026-10-18-ci-host`, `e-2026-10-18-ci-scale`). Each output records the executor, the template and the SDK versions.

[`tools/e2b_agent.py`](tools/e2b_agent.py) (E) is README step 7's application call: it creates a temporary Prompt Agent version with the same tool, one conversation and one `code-interpreter-v1` sandbox, sends one question, runs every tool call with `run_code`, prints the answer, and deletes the sandbox, the conversation and the agent version. Exit 0 answered and cleaned up, 1 a failure or a resource left, 2 a missing setting; on 2026-10-17 it answered on host c and left nothing.

[`tools/measure_code_interpreter.py`](tools/measure_code_interpreter.py) (E) runs the 11 steps of `e-2026-10-15-latest` against E2B's official `code-interpreter-v1` template with `e2b-code-interpreter` 2.11.0 and fails if any step fails or a sandbox is left.

[`tools/measure_sessions.py`](tools/measure_sessions.py) (B and D) runs the explicit session lifecycle from [Manage hosted agent sessions](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions): create a session pinned to one agent version, write a file into it, answer twice, `:stop` until `idle`, answer on the same `agent_session_id`, read the file back, answer once more, delete. It authenticates with an Azure CLI token for `https://ai.azure.com` or with the header in `SESSIONS_AUTH_HEADER`, records every HTTP attempt including retries, and writes a private record plus a public one without session IDs or the endpoint. Exit 0 needs every run complete, the file back unchanged, the question count growing and every session it created deleted; the 2026-10-15 run meets all four. `--resume-after idle` replaces the stop with `--idle-wait` seconds without any request to the session and one status read, and exit 0 then also needs that status to be `idle` or `stopped` (`b-2026-10-18-idle`); `--path D` labels a BYO-AKS run (`d-2026-10-18-sessions`).

[`tools/measure_c.py`](tools/measure_c.py) (C) copies the spec of the workshop's `agent-host` Sandbox under new names with the label `app=agent-host-measure`, so the workshop's own Sandbox and LoadBalancer are not touched. It times `kubectl apply` until the Sandbox and its pod are Ready, asks through `kubectl exec` inside the pod (the in-guest request A uses), and with the Blob PVC deletes the pod and runs a suspend/resume cycle; `--scale` applies N Sandboxes without the volume at once. Exit 0 needs every run's conversation to keep growing, every at-once Sandbox to name its own word and every test Sandbox deleted (`c-2026-10-18-perf`).

The workshop scripts (`module-01/setup.sh`, `module-03/prepare-agent-sandbox.sh`, `module-04/sandbox-deploy.sh`) are used unchanged at `905f2c2`; this repository does not copy them.

## Measured Runs

All runs used Azure resources in East US 2. The client was a Windows workstation, except `b-2026-10-15-sessions` and the 2026-10-18 runs (a Windows VM in Japan East, from WSL Ubuntu) and E, whose tools ran on the E2B host VM (host a on 2026-10-15, host c on 2026-10-17, hosts c, a and d on 2026-10-18). The A/B/C record names Azure CLI 2.88.0 and azd 1.34.1; A's SDK record also names Python 3.13. Raw consoles and IDs stay in the author's private folders; [`tools/export_evidence.py`](tools/export_evidence.py) projected them into [`evidence/runs/`](evidence/runs/), recording the SHA-256 of every private source, and replaced node, pod and namespace names with neutral labels. The 2026-10-18 records were projected the same way by a private script whose SHA-256 each record lists.

Runs are grouped A–E, then by date from oldest to newest within each path, matching the README tables.

### a-2026-09-27-setup: first deployment, and wake on an HTTP request

**Question.** How long does the A path take to set up on the shared foundation, does a stopped sandbox keep its process, and does an HTTP request wake it?

**Input.** The steps of `tools/aca_sandbox.py` in order, then questions sent from inside the sandbox, a stop, a resume, a second stop and an HTTP request to the on-demand port ([`evidence/runs/a-aca-2026-09-27/setup-and-wake.json`](evidence/runs/a-aca-2026-09-27/setup-and-wake.json)). The questions are in the table.

**Varied and fixed.** A single sequence on one sandbox; the full A specification in [Architecture and Metrics](#architecture-and-metrics).

**Result.**

<!-- BEGIN GENERATED: a-setup -->
| Step | Input | s | Reply or state |
|---|---|---:|---|
| sandbox group (Bicep) |  | 90 |  |
| VNet connection |  | 9.6 |  |
| disk image from ACR |  | 21.9 |  |
| Blob volume |  | 3.6 |  |
| sandbox create |  | 21.5 | Running |
| answer, timed inside | "where is Bangkok? one sentence" | 3.6 | Bangkok is the capital city of Thailand, in Southeast Asia. |
| answer, timed inside | "What did I just ask you about? Reply in five words." | 1.1 | Bangkok’s location in the world. |
| stop (memory snapshot) |  | 9.9 | Stopped |
| resume |  | 1.5 | Running |
| answer after resume |  | 12.5 | 2 |
| next answer | "Name the city we discussed. One word." | 0.9 | Bangkok |
| next answer | "Reply with OK." | 1.1 | OK |
| stop again |  | 6.3 | Stopped |
| HTTP request to the stopped sandbox | "How many questions have I asked before this one? Number only." | 23.7 | 5 |
| next HTTP request | "Reply with OK." | 2.5 | OK |

Read inside the sandbox: vmm_type `cloudhypervisor`, kernel `6.12.8+`, 1 vCPU.
<!-- END GENERATED: a-setup -->

**Boundary.**
- One run. The 24 s wake time was a single observation and is not used in the README's comparison; its resume row uses the three explicit resumes without the Blob volume of 2026-10-18.
- The first outbound call after a memory resume took 12.5 s, then about 1 s; we read this as stale keep-alive connections, which we did not prove.
- The wake test used a temporary anonymous port window because the IP rule could not admit the operator; the port was removed afterwards.

### a-2026-10-09-agent: the full agent path in ACA Sandboxes, 3 runs

**Question.** How long does a new sandbox take to answer, how fast is the next answer, and how long does a resume request → first answer take after suspension has finished, on the GA service with the full specification the workshop uses?

**Input.** Each run created a new sandbox from the agent's disk image and asked, in order, these exact questions ([`evidence/runs/a-aca-2026-10-09/agent-runs.json`](evidence/runs/a-aca-2026-10-09/agent-runs.json)):

<!-- BEGIN GENERATED: a-agent-input -->
> Configuration: custom agent disk image + VNet + Blob volume + OnDemand port; cpu 1000m, memory 2048Mi, auto-suspend 900 s in Memory mode.
> Request: POST http://127.0.0.1:8080/reflect {"query": <input>}, sent with sandbox exec from inside the sandbox.
> First answer: "Where is Bangkok? Answer in one short sentence."
> Every later turn: "How many questions have I asked before this one? Number only."
<!-- END GENERATED: a-agent-input -->

**Varied and fixed.** Only the run number varied. The service ([GA since 2026-09-23](https://techcommunity.microsoft.com/blog/appsonazureblog/azure-container-apps-sandboxes-now-generally-available/4559125)), SDK 0.1.0b4 (still beta), the sandbox group, the image and the specification were the same in all three.

**Result.**

<!-- BEGIN GENERATED: a-agent -->
Seconds; answers are the agent's replies in order. Totals are added per run before taking the median; the independently reported column medians must not be added.

| Run | Create → Running | First answer | Cold | Follow-up | Stop | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 44.86 | 5.55 | 50.41 | 1.54 | 7.56 | 2.83 | 14.39 | 17.22 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 24.12 | 3.88 | 28.0 | 1.04 | 5.97 | 1.61 | 12.84 | 14.45 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 24.25 | 2.82 | 27.07 | 1.05 | 9.31 | 3.02 | 9.42 | 12.44 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| **median** | **24.25** | **3.88** | **28.0** | **1.05** | **7.56** | **2.83** | **12.84** | **14.45** |  |
<!-- END GENERATED: a-agent -->

All three runs reported `vmm_type` <!--v:a.vmm.value-->cloudhypervisor<!--/v-->, and the conversation history kept growing across stop and resume in <!--v:a.state_kept.count-->3<!--/v--> of 3. No sandbox was left behind.

**Boundary.**
- Three runs. The first run's create took <!--v:a.create.max-->44.86<!--/v--> s against about 24 s for the other two, so its cold time is the maximum of the range.
- Create time is SDK wall time from the laptop and includes its round trip; answer time is measured inside the sandbox and excludes it.
- Cold and resume-to-answer add those separate clocks; stop and idle waiting are excluded, as is the client-side exec overhead.
- The documented sub-second start applies to prewarmed pools; this configuration includes a custom image, a VNet connection and a private Blob volume.

### a-2026-10-09-attribution: which part of the specification costs create time

**Question.** The full specification took over 20 s to create. Is it the custom image, the VNet, the port or the Blob volume?

**Input.** One sandbox at a time, each configuration created, stopped, resumed and deleted. The six configurations, verbatim from [`evidence/runs/a-aca-2026-10-09/create-attribution.json`](evidence/runs/a-aca-2026-10-09/create-attribution.json), are in the first two columns below; the same `cpu`, `memory` and suspend mode as above.

**Varied and fixed.** One configuration item at a time was added to the agent's image. The sandbox group, region, SDK and client stayed the same; runs per configuration were 3 for `builtin`, `image` and `full` and 2 for the others.

**Result.**

<!-- BEGIN GENERATED: a-attr -->
| Configuration | What is added | n | Create → Running, s | Median |
|---|---|---:|---|---:|
| `builtin` | built-in ubuntu disk, SDK defaults, no VNet, volume or port | 3 | 5.74, 2.2, 4.68 | 4.68 |
| `image` | the agent's disk image from ACR, nothing else | 3 | 2.31, 1.6, 2.7 | 2.31 |
| `vnet` | image + VNet connection | 2 | 1.97, 2.52 | — (n=2) |
| `port` | image + OnDemand port with IP access control | 2 | 2.33, 2.3 | — (n=2) |
| `vnetvolume` | image + VNet connection + private Blob volume | 2 | 25.12, 15.79 | — (n=2) |
| `full` | image + VNet connection + private Blob volume + OnDemand port | 3 | 25.9, 22.2, 26.32 | 25.9 |

Guest uptime after resume exceeded its pre-stop value in 15 of 15 sandboxes. This observation alone does not prove process-memory continuity.
<!-- END GENERATED: a-attr -->

**Boundary.**
- Two or three runs per configuration.
- The uptime comparison is only an observation, not proof that a particular process or its memory survived.
- The volume was not tried without the VNet connection to a working end: against Storage with public access disabled the request did not return and no sandbox appeared; it was abandoned after about 8 minutes and is not in the table.
- This shows where the time appears, not why: the Blob mount is on the create path, the reason inside the service is not visible.

### a-2026-10-09-late-mount: start without the volume, mount it afterwards

**Question.** If the volume is the slow part, can the sandbox start without it and attach it once running?

**Input.** Create with the image and the VNet connection but no volume, then `add_volume_mount` of the same Blob volume to the running sandbox, then list the mount ([`evidence/runs/a-aca-2026-10-09/late-mount.json`](evidence/runs/a-aca-2026-10-09/late-mount.json)).

**Varied and fixed.** Only the order changed (volume after start instead of at create); everything else as in `vnetvolume`.

**Result.**

<!-- BEGIN GENERATED: a-late -->
| Run | Create without volume, s | Attach Blob volume, s | State after mount |
|---|---:|---:|---|
| 1 | 7.2 | 20.29 | Running |
| 2 | 3.99 | 12.4 | Running |
<!-- END GENERATED: a-late -->

Both mounts listed the existing state files, and the sandbox stayed Running.

**Boundary.** Two runs. Useful only when the first answer does not need the saved state: the agent starts with an empty state directory until the mount completes.

### a-2026-10-18-scale: 4 and 8 agent sandboxes at once

**Question.** Do 4, and then 8, agent sandboxes created at the same time with the full specification all start and answer, and how long does creation take under that load?

**Input.** [`tools/measure_aca.py`](tools/measure_aca.py) `scale --scale 4,8` against the sandbox group and disk image of `a-2026-10-09-agent` ([`evidence/runs/a-aca-2026-10-18/scale.json`](evidence/runs/a-aca-2026-10-18/scale.json)). Each sandbox was asked "Remember SCALE-<k>. Reply only OK." and then "What did I ask you to remember? Word only.", the concurrency questions of D and E, and was then deleted.

**Varied and fixed.** Only the number of sandboxes created at once: 4, then 8, 20 s apart. The specification (custom image, VNet, private Blob volume, port; 1 vCPU, 2 GiB) was that of `a-2026-10-09-agent`; the client was a Windows VM in Japan East (WSL Ubuntu), and the answers were timed inside each sandbox.

**Result.**

<!-- BEGIN GENERATED: a-scale -->
| At once | Answered | Create → Running, s | Read turn inside the sandbox, s | All sessions, s |
|---:|---|---:|---:|---:|
| 4 | 4/4 | 15.66–24.55 | 0.93–1.11 | 32.9 |
| 8 | 8/8 | 15.49–24.25 | 1.01–1.74 | 33.91 |

Every sandbox's second answer was its own `SCALE-<k>`; `vmm_type` `cloudhypervisor` for all; every sandbox was deleted and none was left. All sessions: from the first create call until the last sandbox was deleted.
<!-- END GENERATED: a-scale -->

**Boundary.**
- One run per level from one client. 8 sandboxes are 8 active cores, far below the default quota of 500, so this does not probe the limit.
- Create times are SDK wall time on a client in Japan East; the 2026-10-09 runs came from a workstation in China.
- Each sandbox wrote its own small state file (named after the sandbox) to the shared Blob container, as earlier A runs did; those files were left there.

### a-2026-10-18-novolume: the full agent path without the Blob volume, 3 runs

**Question.** Without the private Blob volume, how long does a new sandbox take to its first answer, and does the conversation survive stop and resume on the Memory snapshot alone?

**Input.** [`tools/measure_aca.py`](tools/measure_aca.py) `agent --runs 3 --config novolume` against the sandbox group and disk image of `a-2026-10-09-agent` ([`evidence/runs/a-aca-2026-10-18/agent-novolume.json`](evidence/runs/a-aca-2026-10-18/agent-novolume.json)), with these exact inputs:

<!-- BEGIN GENERATED: a-nv-input -->
> Configuration: custom agent disk image + VNet + OnDemand port, no Blob volume; cpu 1000m, memory 2048Mi, auto-suspend 900 s in Memory mode.
> Request: POST http://127.0.0.1:8080/reflect {"query": <input>}, sent with sandbox exec from inside the sandbox, after 127.0.0.1:8080 accepts a connection.
> First answer: "Where is Bangkok? Answer in one short sentence."
> Every later turn: "How many questions have I asked before this one? Number only."
<!-- END GENERATED: a-nv-input -->

**Varied and fixed.** Only the Blob volume was removed. The image, VNet connection, port, `cpu`, `memory`, Memory-mode suspend, questions and the agent's state path (`STATE_MOUNT_PATH`, now a folder on the sandbox's own disk) were those of `a-2026-10-09-agent`. The client was a Windows VM in Japan East (WSL Ubuntu); the 2026-10-09 runs came from a workstation in China.

**Result.**

<!-- BEGIN GENERATED: a-nv -->
Seconds; answers are the agent's replies in order. Totals are added per run before taking the median; the independently reported column medians must not be added. Wait for the agent: inside the sandbox, until it accepted a connection on port 8080, before the first request; part of Cold. The same wait before the answer after resume was 0 s in every run and is part of Resume → answer.

| Run | Create → Running | Wait for the agent | First answer | Cold | Follow-up | Stop | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.15 | 0.0 | 5.96 | 8.11 | 1.02 | 20.23 | 1.22 | 1.74 | 2.96 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 1.35 | 0.0 | 3.51 | 4.86 | 0.84 | 10.67 | 4.46 | 2.32 | 6.78 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 1.22 | 0.0 | 3.13 | 4.35 | 0.67 | 9.88 | 0.82 | 1.88 | 2.7 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **median** | **1.35** | **0.0** | **3.51** | **4.86** | **0.84** | **10.67** | **1.22** | **1.88** | **2.96** |  |
<!-- END GENERATED: a-nv -->

Without the volume the conversation kept growing across stop and resume in <!--v:a.nv.state_kept.count-->3<!--/v--> of <!--v:a.nv.state_kept.n-->3<!--/v--> runs: the agent's state file stayed on the sandbox disk, and the Memory snapshot kept it. Compared with `a-2026-10-09-agent`, the cold-start median fell from <!--v:a.cold.median-->28.0<!--/v--> s to <!--v:a.nv.cold.median-->4.86<!--/v--> s and the resume → answer median from <!--v:a.recover.median-->14.45<!--/v--> s to <!--v:a.nv.recover.median-->2.96<!--/v--> s. To compare parts that add up, take on each day the run whose total is the median: cold start was create <!--v:a.cold_mid.create-->24.12<!--/v--> s + answer <!--v:a.cold_mid.answer-->3.88<!--/v--> s with the volume and <!--v:a.nv.cold_mid.create-->1.35<!--/v--> s + <!--v:a.nv.cold_mid.answer-->3.51<!--/v--> s without, so the difference is in create; resume → answer was resume <!--v:a.recover_mid.resume-->1.61<!--/v--> s + answer <!--v:a.recover_mid.answer-->12.84<!--/v--> s with the volume and <!--v:a.nv.recover_mid.resume-->1.22<!--/v--> s + <!--v:a.nv.recover_mid.answer-->1.74<!--/v--> s without, so the difference is in the first answer after resume, timed inside the sandbox. Medians of single columns in the tables come from different runs and do not add up.

**Boundary.**
- Three runs with explicit stop and resume; auto-suspend after 900 s idle was not waited for.
- The record shows where the extra time after resume is (inside the sandbox, in the first answer), not why; that it comes from the Blob mount is an inference from this difference.
- Stop took longer without the volume (<!--v:a.nv.stop.median-->10.67<!--/v--> s against <!--v:a.stop.median-->7.56<!--/v--> s median, one stop <!--v:a.nv.stop.max-->20.23<!--/v--> s); the record does not show why.
- Without the volume, Running can come before the agent listens. Before these three runs, a smoke run and a first set of three ran with a tool revision that sent the first request as soon as the sandbox read Running: <!--v:a.nv.earlier.count-->3<!--/v--> of <!--v:a.nv.earlier.n-->4<!--/v--> completed (cold <!--v:a.nv.earlier_cold.range-->4.33–5.03<!--/v--> s, resume → answer <!--v:a.nv.earlier_recover.range-->2.57–10.67<!--/v--> s), and in the other one the first request was refused <!--v:a.nv.refused_after.value-->2.18<!--/v--> s after create because the agent was not listening yet. The tool now waits inside the sandbox until the agent accepts a connection and counts that wait in the cold time. All of them are in the record (`earlier_attempts`).
- The state then lives only in that sandbox. Deleting the sandbox, or reading the state from another sandbox, was not tested.
- Create, stop and resume are SDK wall time on the client, which was in Japan East here and in China on 2026-10-09; the answers are timed inside the sandbox on both days.

### b-2026-09-27: Foundry hosted agent, deploy and two routes to the model

**Question.** How long does B take from `azd` to an answering agent, and how fast are a new session and its follow-ups?

**Input.** The workshop's module-02 commands (README step 5): `azd ai agent init`, `azd deploy` with `MODEL_ROUTING=gateway` (version 1), then a second `azd deploy` with `direct` (version 2), each followed by `azd ai agent invoke` on a new session and two follow-up turns. The request text was not kept in the run record ([`evidence/runs/b-hosted-agent-2026-09-27/run.json`](evidence/runs/b-hosted-agent-2026-09-27/run.json)).

**Varied and fixed.** Only the model route differed between the two versions; the project, model and client were the same.

**Result.**

<!-- BEGIN GENERATED: b-run -->
| Step | Route to the model | s |
|---|---|---:|
| `azd ai agent init` |  | 38.4 |
| `azd deploy`, version 1 | gateway | 161 |
| new session → answer | gateway | 9.43 |
| follow-up turns | gateway | 4.03, 5.54 |
| `azd deploy`, version 2 | direct | 100 |
| new session → answer | direct | 5.29 |
| follow-up turns | direct | 4.94, 4.38 |
<!-- END GENERATED: b-run -->

**Boundary.**
- One run, two follow-up samples per route: not enough to call gateway and direct different.
- The direct route ran without the Foundry native AI gateway, because that portal step could not be automated.
- Resume after idle and session loss were not exercised here; resume after an explicit stop is in `b-2026-10-15-sessions`, and the turn after an idle wait in `b-2026-10-18-idle`.

### b-2026-10-15-sessions: Foundry hosted agent, explicit session lifecycle, 3 runs

**Question.** How long does B take to create a session, answer, stop, and answer again after the stop? Does the session keep its file and conversation across the stop?

**Input.** [`tools/measure_sessions.py`](tools/measure_sessions.py) against the module-02 agent, version `2` (0.5 vCPU / 1 GiB, `gpt-5.4-mini`, `MODEL_ROUTING=direct`), 3 runs ([`evidence/runs/b-sessions-2026-10-15/sessions.json`](evidence/runs/b-sessions-2026-10-15/sessions.json)). Each run: `POST .../endpoint/sessions` pinned to version `2`; `PUT .../files/content` writes a marker file; first turn "Where is Bangkok? Answer in one short sentence."; every later turn "How many questions have I asked before this one? Number only." with `previous_response_id`; `POST .../sessions/{id}:stop`, then poll until `idle`; the next turn on the same `agent_session_id`; read the marker back; one more turn; `DELETE` the session.

**Varied and fixed.** Nothing varied between the three runs. The agent version, prompts and client were fixed: a Windows VM in Japan East (WSL Ubuntu) calling the project in East US 2 through an Azure API Management gateway. Client times include that path; the service's own `created_at` → `completed_at` is reported separately.

**Result.**

<!-- BEGIN GENERATED: b-sessions -->
Seconds, client wall time; the last column is the agent's replies in order.

| Run | Create → active | First answer | Follow-up | Stop → idle | Answer after stop | Next answer | File kept | Answers |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 6.69 | 5.62 | 4.59 | 2.9 | 12.35 | 4.6 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 6.87 | 4.79 | 4.81 | 3.05 | 12.02 | 4.11 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 6.51 | 5.85 | 4.42 | 7.71 | 11.66 | 3.88 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **median** | **6.69** | **5.62** | **4.59** | **3.05** | **12.02** | **4.11** |  |  |

35 HTTP attempts: 200 ×23, 201 ×6, 204 ×6; none retried.
<!-- END GENERATED: b-sessions -->

The service recorded <!--v:b.sess.server.range-->1–4<!--/v--> s of generation per turn (whole seconds); the rest of each client time is the network path, the gateway and the platform.

**Boundary.**
- Three runs from one client: the client times include a Japan East → East US 2 round trip and the gateway, so they are not comparable with the 2026-09-27 run from China.
- Resume was measured after an explicit `:stop`, not after the idle timeout (that case is `b-2026-10-18-idle`). There is no memory snapshot: the agent process starts again, and only `$HOME`, `/files` and the conversation stored by the platform carry over.
- One session at a time; concurrency is in `b-2026-10-18-perf`; no session loss.

### b-2026-10-18-perf: Foundry hosted agent, D's protocol: new sessions, follow-ups, 4 and 8 at once

**Question.** On B, how fast is a new session started by a single request, how fast are follow-ups, and do 4 and 8 new sessions at once all answer?

**Input.** [`tools/measure_responses.py`](tools/measure_responses.py) `--cold 5 --warm-turns 3 --scale 4,8`, the protocol of `d-2026-09-29-perf`, against the module-02 agent's Responses endpoint through the gateway of `b-2026-10-15-sessions` ([`evidence/runs/b-hosted-agent-2026-10-18/perf.json`](evidence/runs/b-hosted-agent-2026-10-18/perf.json)). The exact inputs:

<!-- BEGIN GENERATED: b-perf-input -->
> Request: POST <gateway>/agents/<agent>/endpoint/protocols/openai/responses {"input": <input>} with previous_response_id for follow-ups; an API Management subscription key header (RESPONSES_AUTH_HEADER) instead of an Entra token; nothing to acquire inside or outside the timed requests.
> Agent: `maf-agent` version 2 (0.5 vCPU, 1Gi, `MODEL_ROUTING=direct`, `gpt-5.4-mini`); no version pinned in the requests; the endpoint's only rule sends 100% to @latest, which read back as version 2 on 2026-10-18.
> New sessions: "Reply with exactly COLD-{i}" (n = 1–5)
> One session: "Remember the code word LANTERN-7. Reply only OK.", then "What is the code word? Reply with the word only.", then "Reply with exactly WARM-{i}" (n = 1, 2)
> At once, per session: "Remember SCALE-{i}. Reply only OK.", then "What did I ask you to remember? Word only." (4, then 8 sessions)
> Client: Windows VM in Japan East (WSL Ubuntu) calling the Foundry project in East US 2 through an Azure API Management gateway (the path of b-2026-10-15-sessions).
<!-- END GENERATED: b-perf-input -->

**Varied and fixed.** New sessions, then one session with follow-ups, then 4 and 8 sessions at once, 20 s apart. The agent version, gateway and client were fixed.

**Result.**

<!-- BEGIN GENERATED: b-perf -->
New sessions:

| Start (UTC) | Answer | New session → answer, s | First byte, s |
|---|---|---:|---:|
| 03:49:28 | COLD-1 | 11.48 | 11.48 |
| 03:49:42 | COLD-2 | 11.03 | 11.03 |
| 03:49:57 | COLD-3 | 11.57 | 11.57 |
| 03:50:11 | COLD-4 | 10.98 | 10.98 |
| 03:50:25 | COLD-5 | 11.0 | 11.0 |

11.03 s median (10.98–11.57; n=5).

Follow-ups in one session:

| Turn | Answer | Client wall, s |
|---|---|---:|
| 1 (new session) | OK | 11.05 |
| 2 | LANTERN-7 | 3.94 |
| 3 | WARM-1 | 4.08 |
| 4 | WARM-2 | 3.88 |

Follow-ups 2–4: 3.94 s median (3.88–4.08; n=3).

At once:

| At once | Answered | Read back its word | Write turn, s |
|---:|---|---|---:|
| 4 | 4/4 | 4/4 | 10.95–12.32 |
| 8 | 8/8 | 8/8 | 10.02–12.63 |

Every request returned HTTP 200; the tool sends each request once.
<!-- END GENERATED: b-perf -->

**Boundary.**
- One campaign from one client; client times include the Japan East → East US 2 round trip and the gateway, so they compare with `b-2026-10-15-sessions`, not with D's laptop runs from China.
- B shows no pods; where the platform ran the 8 sessions is not observable.
- 8 sessions at once show that B answered them all without a refusal; they do not probe a limit.

### b-2026-10-18-idle: Foundry hosted agent, the next turn after 17 minutes without requests

**Question.** The default idle timeout is 15 minutes. What does the next turn cost when a session got no request for longer than that, and does the session keep its file and conversation?

**Input.** [`tools/measure_sessions.py`](tools/measure_sessions.py) `--resume-after idle --idle-wait 1020` against version `2` through the gateway ([`evidence/runs/b-sessions-2026-10-18/idle.json`](evidence/runs/b-sessions-2026-10-18/idle.json)), 3 runs: create a session, write a marker file, ask "Where is Bangkok? Answer in one short sentence." and then "How many questions have I asked before this one? Number only.", send nothing to the session for 1020 s, read its status once, ask the second question twice more, read the marker back, delete the session. The control ([`idle-control.json`](evidence/runs/b-sessions-2026-10-18/idle-control.json)) is the same with `--idle-wait 300` and started after the first set had finished.

**Varied and fixed.** Only the wait: 1020 s, past the 900 s default, or 300 s, below it. The agent version (no `idle_timeout_seconds` of its own, so the default applies), questions, gateway and client were fixed.

**Result.**

<!-- BEGIN GENERATED: b-idle -->
Seconds, client wall time; the service time is the response's `completed_at` minus `created_at` (whole seconds); the last column is the agent's replies in order.

| Wait, s | Run | Status after the wait | Follow-up before the wait | Next turn | Service time of that turn | Turn after that | File kept | Answers |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1021.51 | 1 | `active` | 4.72 | 12.67 | 2 | 3.88 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 1021.48 | 2 | `idle` | 3.81 | 12.71 | 2 | 5.42 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 1021.22 | 3 | `idle` | 3.63 | 12.12 | 2 | 3.99 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **1021.48** | **median** |  | **3.81** | **12.67** | **2** | **3.99** |  |  |
| 301.75 | 1 | `active` | 3.7 | 5.76 | 3 | 4.16 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 301.45 | 2 | `active` | 3.85 | 5.16 | 2 | 4.4 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 301.47 | 3 | `active` | 4.21 | 4.43 | 2 | 3.89 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **301.47** | **median** |  | **3.85** | **5.16** | **2** | **4.16** |  |  |
<!-- END GENERATED: b-idle -->

The status after the wait is in the third column. It was read once, so it does not show when or whether the platform released the compute. After about 5 minutes the next turn took <!--v:b.idle5.next.median-->5.16<!--/v--> s median, somewhat longer than the follow-up before the wait (<!--v:b.idle5.warm.median-->3.85<!--/v--> s). After about 17 minutes it took <!--v:b.idle.next.median-->12.67<!--/v--> s, about as long as a resume after an explicit `:stop` (<!--v:b.sess.resume.median-->12.02<!--/v--> s in `b-2026-10-15-sessions`), and the service recorded at most <!--v:b.idle.next_server.max-->2<!--/v--> s of it as generation. So the extra time appears between 5 and 17 minutes. That is consistent with the platform releasing the session's compute after the 15-minute default, also in the 17-minute run whose status still read `active`. It is an inference from times; B does not expose its compute.

**Boundary.**
- Three runs per wait from one client; one status read per run, after the wait.
- B does not expose its compute, so whether a session's compute was released can only be inferred from the time of the next turn.
- `measure_sessions.py` exits 1 for a set in which any status still read `active` before the next turn: by design, exit 0 needs `idle` or `stopped` there.

### c-2026-09-27: AKS + agent-sandbox, isolation, pod loss, suspend and resume

**Question.** Does the agent pod run in its own micro-VM, and what do pod loss and a suspend/resume cycle cost?

**Input.** The workshop's module-03 script and manifest (README step 3), questions over the LoadBalancer, then `kubectl delete pod`, then `operatingMode: Suspended` and `Running` patches ([`evidence/runs/c-aks-agent-sandbox-2026-09-27/run.json`](evidence/runs/c-aks-agent-sandbox-2026-09-27/run.json)). The kernel check followed the workshop's own verification: `uname -r` in the Sandbox pod and in a pod without `runtimeClassName`. The request text was not kept.

**Varied and fixed.** One Sandbox, one sequence; Kubernetes 1.35.8, one `Standard_D4s_v3` Kata node, agent-sandbox `v0.5.2`.

**Result.**

<!-- BEGIN GENERATED: c-run -->
| Step | Observed |
|---|---|
| `prepare-agent-sandbox.sh` | 600 s |
| Blob private link | 168 s |
| `kubectl apply` → Sandbox Ready | 44.9 s |
| first answer, second answer | 12.75 s, 1.96 s; context remembered: true |
| `uname -r` in the Sandbox pod / in a runc pod | `6.6.137.mshv1-1.azl3` / `6.8.0-1067-azure` |
| delete the pod → Ready again | 23.4 s, state restored: true |
| `operatingMode: Suspended` → pod gone | 9.7 s; health while suspended HTTP 503 |
| `operatingMode: Running` → Ready / LB 200 / answer | 17.1 / 26.1 / 34.2 s |
<!-- END GENERATED: c-run -->

**Boundary.**
- One run; three runs of each step, with the answers timed inside the pod, are in `c-2026-10-18-perf`.
- The runc comparison pod ran on the Ubuntu system pool, so the two kernels differ by node as well as by runtime. The Kata kernel's `mshv` suffix, the runtime class and Learn's documented stack carry the conclusion.
- Wake on request is not part of agent-sandbox; the resume was a manual patch.

### c-2026-10-18-perf: AKS + agent-sandbox, cold start without and with the Blob PVC, pod loss, suspend/resume, 4 and 8 at once

**Question.** How fast does a Sandbox start and answer without an external volume, and with the workshop's Blob PVC? What do pod loss and a suspend/resume cycle cost over three runs? How many Sandboxes fit at once?

**Input.** [`tools/measure_c.py`](tools/measure_c.py) `--cold 3 --pvc 3 --scale 4,8` against the workshop's cluster ([`evidence/runs/c-aks-agent-sandbox-2026-10-18/perf.json`](evidence/runs/c-aks-agent-sandbox-2026-10-18/perf.json)). Every test Sandbox copies the spec of the workshop's `agent-host` Sandbox, and the answers are POSTed to `/reflect` inside the pod with `kubectl exec`:

<!-- BEGIN GENERATED: c18-input -->
> Cluster: the cluster of c-2026-09-27, started for this run; the workshop's own agent-host Sandbox kept running on the Kata node throughout; Kubernetes 1.35.8, Free tier; system pool 1 x standard_d4ads_v7 Ubuntu; Kata pool Standard_D4s_v3 AzureLinux KataVmIsolation, autoscale 1-10, 1 node at start; agent-sandbox v0.5.2 (controller image read back 2026-10-18).
> Each test Sandbox copies the spec of the workshop's `agent-host` Sandbox in namespace `agent` under a new name and the label `app=agent-host-measure`: image `<registry>.azurecr.io/agent-host:latest`, runtime class `kata-vm-isolation`, requests 500m CPU and 512Mi, limits 2000m CPU and 2Gi. `AGENT_ID` comes from the pod name, so each copy keeps its conversation in its own state file.
> Request: POST http://127.0.0.1:8080/reflect {"query": <input>}, sent with kubectl exec python3 inside the agent container.
> First answer: "Where is Bangkok? Answer in one short sentence."
> Every later turn: "How many questions have I asked before this one? Number only."; with the PVC once more after the pod was deleted and once after suspend/resume
> At once, per Sandbox without the volume: "Remember SCALE-{k}. Reply only OK.", then "What did I ask you to remember? Word only." (4, then 8 Sandboxes)
> Client: Windows VM in Japan East (WSL Ubuntu); kubectl against the cluster in East US 2; answers timed inside the pod with kubectl exec.
<!-- END GENERATED: c18-input -->

**Varied and fixed.** With or without the Blob PVC (for the runs without it, the `persistentVolumeClaim` volume and its mount are removed from the copy), and 4 or 8 Sandboxes at once, 20 s between runs. The cluster, image, agent and client were fixed.

**Result.** Cold start and follow-up:

<!-- BEGIN GENERATED: c18-runs -->
| Sandbox | Volume | `kubectl apply` → Ready, s | First answer, s | Cold total, s | Follow-up, s | Conversation length |
|---|---|---:|---:|---:|---:|---|
| cold-1 | none | 18.9 | 8.44 | 27.34 | 1.08 | 1, 2 |
| cold-2 | none | 15.32 | 8.5 | 23.82 | 1.0 | 1, 2 |
| cold-3 | none | 16.45 | 7.82 | 24.27 | 1.03 | 1, 2 |
| pvc-1 | Blob PVC | 16.81 | 7.38 | 24.19 | 36.73 | 1, 2, 3, 4 |
| pvc-2 | Blob PVC | 16.26 | 10.42 | 26.68 | 1.78 | 1, 2, 3, 4 |
| pvc-3 | Blob PVC | 16.46 | 7.58 | 24.04 | 1.22 | 1, 2, 3, 4 |

Without the volume: apply → Ready 16.45 s median (15.32–18.9; n=3), first answer 8.44 s median (7.82–8.5; n=3), cold total 24.27 s median (23.82–27.34; n=3). With the Blob PVC: apply → Ready 16.46 s median (16.26–16.81; n=3), first answer 7.58 s median (7.38–10.42; n=3), cold total 24.19 s median (24.04–26.68; n=3). Follow-ups with the PVC: 36.73, 1.78, 1.22 s (n=3); the record does not show where the slow one spent its time. Answers are timed inside the pod; the `kubectl exec` round trip around each added 2.53–3.21 s on the client and is excluded.
<!-- END GENERATED: c18-runs -->

Pod loss and suspend/resume, with the Blob PVC:

<!-- BEGIN GENERATED: c18-pvc -->
| Sandbox | Pod deleted → Ready, s | Answer, s | Suspended → pod gone, s | Running → Ready, s | Answer, s | Resume → answer, s |
|---|---:|---:|---:|---:|---:|---:|
| pvc-1 | 25.24 | 7.72 | 9.24 | 15.87 | 7.47 | 23.34 |
| pvc-2 | 24.79 | 7.66 | 9.1 | 15.93 | 6.52 | 22.45 |
| pvc-3 | 23.25 | 7.11 | 9.45 | 17.28 | 6.95 | 24.23 |
| **median** | **24.79** |  | **9.24** | **15.93** |  | **23.34** |

While suspended, the Sandbox's Ready condition read `False` (`SandboxSuspended`) in every run. The conversation length kept growing after the pod was deleted and after suspend/resume (last column of the table above), so the state came back from the Blob PVC each time. Resume → answer is the Running patch until Ready plus the answer inside the pod.
<!-- END GENERATED: c18-pvc -->

At once, without the volume:

<!-- BEGIN GENERATED: c18-scale -->
| At once | Answered | Apply → Ready, s | Sandboxes per Kata node | All Sandboxes, s |
|---:|---|---:|---|---:|
| 4 | 4/4 | 18.32–19.96 | node-1: 4 | 42.15 |
| 8 | 8/8 | 18.52–141.79 | node-1: 6, node-2: 2 | 159.28 |

The run started with 1 Kata node, which also ran the workshop's own Sandbox. With 8 at once, 6 test Sandboxes fit there (each requests 500m CPU); the other 2 ran on a node the cluster autoscaler added during the run and were Ready after 141.55–141.79 s. Kata nodes at the end: 2. Every Sandbox answered with its own `SCALE-<k>` and was deleted.
<!-- END GENERATED: c18-scale -->

**Boundary.**
- Three runs per variant and one per concurrency level, from one client.
- `c-2026-09-27` timed the answers at the LoadBalancer, and its apply was the first on that cluster; these runs time the answers inside the pod with the image already on the node, so the two sets differ in clock and in cache.
- The Blob PVC was already bound; mounting it added nothing measurable at apply. Binding a new PVC was not measured.
- Each copy wrote its own state file (named after its pod) to the shared Blob container; the files of the three PVC runs were left there, as were those of earlier runs.

### d-2026-09-29-placement: where a BYO-AKS session runs

**Question.** After binding the cluster, does a session actually run on your Kata pool?

**Input.** The private-preview sample agent (version 3) deployed with `azd deploy`; one new session, then two turns at a later recheck ([`evidence/runs/d-byoaks-2026-09-29/placement.json`](evidence/runs/d-byoaks-2026-09-29/placement.json)). The request text of the first session was not kept; the recheck answers were "OK" and then a code word.

**Varied and fixed.** The one change before this run was the subnet NSG rule; the product group had fixed the subscription's enablement earlier the same day.

**Result.**

<!-- BEGIN GENERATED: d-placement -->
| Check | Observed |
|---|---|
| IP flow verify before the fix | AzureCloud → :443 Deny; AzureCloud → :443 Deny; VNet → :443 Allow |
| rule added to the subnet NSG | `Allow-AzureCloud-FoundryRouter-443`, priority 500, Inbound Allow Tcp AzureCloud → <router-ip>:443 |
| first session after the fix | HTTP 200, "BYO-AKS-OK", 51 s on the client |
| its session pod | runtimeClassName `kata-vm-isolation`, owner ReplicaSet, requests 500m / 1Gi, overhead 600Mi, init containers foundry-sandbox-agent, foundry-sandbox-init |
| scheduled → Ready | 35 s |
| session pods at the 18:15 recheck | 3 of 3 Running with `kata-vm-isolation` |
| two turns at the recheck | "OK" → "HERON-1815" |
| kernel of the Kata pool nodes | `6.6.137.mshv2-2.azl3` |
<!-- END GENERATED: d-placement -->

**Boundary.**
- Before 2026-09-29 our earlier field record (the private repository `foundry-byo-aks-private-preview`) found sessions answered with the cluster stopped and no session pod with the cluster running. Two causes, in order: the product-side enablement of the subscription, then our subnet NSG. That record's placement result is superseded by this run.
- The guest kernel was not read inside a session pod; the node kernel and `runtimeClassName` are the evidence.
- Placement can change in a private preview.

### d-2026-09-29-perf: cold start, follow-ups and concurrency on three nodes

**Question.** On a three-node Kata pool, how fast is a new session and a follow-up, and how many sessions fit at once?

**Input.** [`tools/measure_responses.py`](tools/measure_responses.py) (then named `byoaks_bench.py`) with these exact inputs ([`evidence/runs/d-byoaks-2026-09-29/bench.json`](evidence/runs/d-byoaks-2026-09-29/bench.json), [`cold-early.json`](evidence/runs/d-byoaks-2026-09-29/cold-early.json)):

<!-- BEGIN GENERATED: d-input -->
> Request: POST <agent responses endpoint> {"input": <input>} with previous_response_id for follow-ups.
> New sessions: "Reply with exactly COLD-{i}" (n = 1, 2, 3)
> One session: "Remember the code word LANTERN-7. Reply only OK.", then "What is the code word? Reply with the word only.", then "Reply with exactly WARM-{i}" (n = 1, 2)
> At once, per session: "Remember SCALE-{i}. Reply only OK.", then "What did I ask you to remember? Word only."
> Client: Windows laptop in China; direct REST POST to the agent's Responses endpoint with an Azure CLI token; recorded round trip 320–800 ms.
<!-- END GENERATED: d-input -->

**Varied and fixed.** Cold runs varied whether the node already held the image; concurrency varied 4 and 8. The pool (3 × `Standard_D4s_v5`), the agent version, the session pod shape and the client stayed the same.

**Result.** New sessions:

<!-- BEGIN GENERATED: d-cold -->
| Start (UTC) | Node | Answer | New session → answer, s | Pod created → Ready, s |
|---|---|---|---:|---:|
| 15:43:00 | no image yet | COLD-1 | 41.9 | not paired |
| 15:43:47 | no image yet | COLD-2 | 40.6 | not paired |
| 15:44:32 | image cached | COLD-3 | 26.7 | not paired |
| 15:47:43 | image cached | COLD-1 | 24.01 | 10.0 |
| 15:48:15 | image cached | COLD-2 | 25.49 | 10.0 |
| 15:48:50 | image cached | COLD-3 | 26.16 | 10.0 |
| 15:49:37 | image cached | OK | 23.53 | not paired |

With the image cached: 25.49 s median (23.53–26.7; n=5). First pod on a node without the image: 41.9, 40.6 s (n=2). Session pods created → Ready across the run: 8, 9, 10, 10, 10, 10 s with the image cached, 25, 25 s on a fresh node.
<!-- END GENERATED: d-cold -->

Follow-ups in one session:

<!-- BEGIN GENERATED: d-warm -->
| Turn | Answer | Client wall, s |
|---|---|---:|
| 1 (new session) | OK | 23.53 |
| 2 | LANTERN-7 | 16.42 |
| 3 | WARM-1 | 11.17 |
| 4 | WARM-2 | 12.9 |

The same model called directly from the same client, without an agent: 2.69 s median (1.82–5.07; n=5). This baseline used low reasoning effort; it is not a matched agent-configuration benchmark.
<!-- END GENERATED: d-warm -->

At once:

<!-- BEGIN GENERATED: d-scale -->
| At once | Answered | Refused | Refusal | Refused after, s |
|---:|---|---|---|---:|
| 4 | 0/4 | 4/4 | `500 session_creation_failed` | 18.4–18.9 |
| 8 | 2/8 (24.14, 24.96 s) | 6/8 | `429 aks_quota_exceeded` | 68.03–80.51 |

When the refusals began, 12 session pods were alive on 3 `Standard_D4s_v5` nodes, at most 5 per node, and no node had 500m CPU left.
<!-- END GENERATED: d-scale -->

**Boundary.**
- The concurrency levels ran after earlier sessions had filled the pool, so they measure capacity exhaustion, not latency under load. Latency under true concurrency with free capacity was not measured.
- The harness acquired one token before cold + warm and refreshed it before each concurrency level; token acquisition happened outside every timed request. The responses' own `created_at` to `completed_at` was <!--v:d.agent_turn.range-->2–5<!--/v--> s per turn (run manifest). The remaining client wall time is uninstrumented; the endpoint, session routing, agent execution and client network cannot be separately attributed from these records.
- The cold set mixes the three early runs and the benchmark; the two early runs on nodes without the image are reported separately.

### d-2026-09-29-recovery: delete a session pod

**Question.** After a session pod is deleted, how quickly is a replacement Ready, and can another live session still answer? This run did not test the deleted session's continuity.

**Input.** `kubectl delete pod` of one live session pod, then the next turn on a live session, "What did I ask you to remember? Word only." ([`evidence/runs/d-byoaks-2026-09-29/recovery.json`](evidence/runs/d-byoaks-2026-09-29/recovery.json)).

**Varied and fixed.** One pod deletion on the same pool as the benchmark.

**Result.**

<!-- BEGIN GENERATED: d-recovery -->
| Step | Observed |
|---|---|
| session pod deleted (UTC) | 16:04:16 |
| replacement pod Ready after | 23.9 s |
| follow-up on a different live session | HTTP 200, 14.8 s, "LANTERN-7", that session kept its state: true |
<!-- END GENERATED: d-recovery -->

**Boundary.** One deletion. The follow-up went to a different live session than the deleted one. It shows that the pool kept serving and that other session retained `LANTERN-7`; it does **not** establish continuity of the deleted session.

### d-2026-10-18-sessions: BYO-AKS, explicit session lifecycle, 3 runs

**Question.** Do the session API's stop and resume work on BYO-AKS, what do they cost, and does a session keep its file and conversation across them?

**Input.** [`tools/measure_sessions.py`](tools/measure_sessions.py) `--path D` against version `3` of the private-preview sample agent on its project endpoint, authenticated with the client VM's managed identity ([`evidence/runs/d-byoaks-2026-10-18/sessions.json`](evidence/runs/d-byoaks-2026-10-18/sessions.json)). The steps and questions are those of `b-2026-10-15-sessions`: "Where is Bangkok? Answer in one short sentence.", then "How many questions have I asked before this one? Number only." for every later turn, with a marker file written before `:stop` and read back after the resume turn.

**Varied and fixed.** Nothing varied between the three runs. The Kata pool had two `Standard_D4s_v5` nodes during the runs (see the boundary).

**Result.**

<!-- BEGIN GENERATED: d-sessions -->
Seconds, client wall time; the last column is the agent's replies in order.

| Run | Create → active | First answer | Follow-up | Stop → idle | Answer after stop | Next answer | File kept | Answers |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 11.51 | 11.01 | 10.05 | 23.55 | 19.78 | 10.05 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 9.18 | 10.71 | 11.05 | 19.54 | 22.4 | 10.95 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 9.32 | 11.39 | 10.66 | 22.72 | 19.14 | 10.71 | yes | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **median** | **9.32** | **11.01** | **10.66** | **22.72** | **19.78** | **10.71** |  |  |

35 HTTP attempts: 200 ×23, 201 ×6, 204 ×6; none retried.

Session pods during the three runs: 6 new pods, 2 per session, each Running 3–4 s after it appeared (pod age at the change). Both pods of a session ended (Terminating, then Completed); their ages fit `:stop` ending the first and the delete ending the second. `kubectl get pods -w` records order and ages, not timestamps.
<!-- END GENERATED: d-sessions -->

The service recorded <!--v:d.sess.server.range-->1–3<!--/v--> s of generation per turn; the rest of each client time is the network path and the endpoint, and for the turn after a stop also starting a new session pod.

**Boundary.**
- Three runs from one client, which called the project endpoint in East US 2 directly from Japan East.
- With the pool at one node, every session create was refused with `429 aks_quota_exceeded`: session deployments from 2026-09-29, back when the cluster started, held the node's CPU requests. We added a second node for these runs and removed it afterwards; the old sessions were left in place.
- Private preview: the session API's behaviour on BYO-AKS may change.

### e-2026-10-14-agent: Foundry Prompt Agent + E2B Firecracker sandbox on Azure VM, 3 runs

**Question.** When E2B Embed runs on an Azure Gen2 VM with `/dev/kvm`, how fast are Firecracker sandbox creation, pause and resume, and how fast does a Foundry Prompt Agent answer across pause and resume under the same 3-run protocol as A?

**Input.** Each run created a new E2B `base` sandbox and a new Foundry conversation (`gpt-5.4-mini` in East US 2) and asked, in order, these exact questions ([`evidence/runs/e-e2b-2026-10-14/agent-runs.json`](evidence/runs/e-e2b-2026-10-14/agent-runs.json)):

<!-- BEGIN GENERATED: e-agent-input -->
> Host VM: Standard_D4ds_v7; E2B `base` template (2 vCPU, 512 MB, Firecracker memory + disk snapshot via `beta_pause`).
> Versions: E2B Python SDK `e2b 2.46.0`; E2B Embed runtime `588e5b9`.
> Request: POST <Foundry project>/openai/v1/responses with Prompt Agent calling function tool run_in_e2b_sandbox on E2B API :3000/:3002.
> Agent instructions: "You are a concise assistant with access to `run_in_e2b_sandbox`. For EVERY user message, first call `run_in_e2b_sandbox` once to inspect or update files in /home/user, then answer concisely as instructed. When asked for a number only or word only, reply with that exact token only."
> Its one function tool `run_in_e2b_sandbox(python_code)`: "Execute Python 3 code inside the session's isolated E2B Firecracker microVM sandbox on Azure."
> First answer: "Where is Bangkok? Answer in one short sentence."
> Every later turn: "How many questions have I asked before this one? Number only."
<!-- END GENERATED: e-agent-input -->

**Varied and fixed.** Only the run number varied across the 3 agent runs; the 20-run microbenchmark ran on the host VM against `localhost:3000` and `localhost:3002`.

**Result.**

<!-- BEGIN GENERATED: e-agent -->
| Run | Create → Running | First answer | Cold | Follow-up | Pause | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.14 | 4.12 | 4.26 | 2.95 | 0.13 | 0.08 | 6.4 | 6.48 | Bangkok is the capital of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 0.07 | 2.82 | 2.89 | 2.83 | 0.11 | 0.11 | 12.71 | 12.82 | Bangkok is the capital of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 3 | 0.06 | 8.38 | 8.44 | 3.92 | 0.15 | 0.09 | 4.64 | 4.73 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| **median** | **0.07** | **4.12** | **4.26** | **2.95** | **0.13** | **0.09** | **6.4** | **6.48** |  |

On-host 20-sandbox microbenchmark (zero WAN RTT): create 0.051–0.089 s (n=20), warm command 0.007–0.01 s (n=20), pause 0.051–1.81 s (n=20), resume 0.086–0.62 s (n=20). VMM `firecracker` `v1.14-0.2.0` from the host's process list; guest kernel `6.1.177+` from `uname -r` inside the sandbox; host kernel `6.17.0-1022-azure`.
<!-- END GENERATED: e-agent -->

The VMM process behind the sandboxes in the host's process list was <!--v:e.vmm.value-->firecracker<!--/v-->, and the conversation history inside the sandbox grew across pause and resume in <!--v:e.state_kept.count-->3<!--/v--> of 3. Initial VM setup took <!--v:e.setup.value-->177<!--/v--> s.

**Boundary.**
- Superseded: this run used the earlier executor (`base` template, `python3` through `commands.run`); `e-2026-10-17-ci-agent` and `e-2026-10-18-ci-multi` repeat the agent runs, and `e-2026-10-18-ci-host` the on-host cycles, with `code-interpreter-v1`.
- Three agent runs plus 20 on-host microbenchmark iterations on one `Standard_D4ds_v7` VM in East US 2.
- The Prompt Agent runs in Foundry and calls the E2B sandbox as a function tool over the network, whereas A's reflection agent ran inside the ACA sandbox itself.

### e-2026-10-14-perf: cold start, follow-ups, concurrency and sandbox replacement on E2B

**Question.** Under D's benchmark protocol (`COLD-{i}`, `LANTERN-7` warm follow-ups, 4 and 8 concurrent sessions, and sandbox replacement), how does a single `Standard_D4ds_v7` VM running E2B perform?

**Input.** The D-matched benchmark driver with these exact inputs ([`evidence/runs/e-e2b-2026-10-14/perf.json`](evidence/runs/e-e2b-2026-10-14/perf.json)):

<!-- BEGIN GENERATED: e-perf-input -->
> Request: POST <Foundry project>/openai/v1/responses {"input": <input>} with Prompt Agent calling E2B sandbox tool; agent instructions and tool as in `e-2026-10-14-agent`.
> New sessions: "Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}" (n = 1, 2, 3, 4, 5)
> One session: "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only.", then "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1", then "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2"
> At once, per session: "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."
> Replacement: a live session ran "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."; after another sandbox was killed and replaced, it ran "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only."
> Driver: the author's private script `run_abcd_matched_benchmarks.py` (SHA-256 `8e8f6fc5b7b6…` in the record); `tools/measure_e2b.py perf` sends the same prompts with the same agent and clocks.
<!-- END GENERATED: e-perf-input -->

**Varied and fixed.** Cold runs repeated 5 new sessions; concurrency varied 4 and 8 sessions at once on 1 `Standard_D4ds_v7` VM.

**Result.** New sessions:

<!-- BEGIN GENERATED: e-cold -->
| Start (UTC) | Answer | Sandbox created → Ready, s | Agent turn, s | New session → answer, s |
|---|---|---:|---:|---:|
| 06:14:58 | COLD-1 | 0.056 | 7.75 | 7.94 |
| 06:15:06 | COLD-2 | 0.068 | 2.28 | 2.54 |
| 06:15:08 | COLD-3 | 0.068 | 3.44 | 3.6 |
| 06:15:12 | COLD-4 | 0.057 | 3.52 | 3.65 |
| 06:15:16 | COLD-5 | 0.066 | 3.19 | 3.34 |

Across the 5 cold runs: sandbox create → Ready 0.066 s median (0.056–0.068; n=5), Prompt Agent turn 3.44 s median (2.28–7.75; n=5), end-to-end new session → answer 3.6 s median (2.54–7.94; n=5).
<!-- END GENERATED: e-cold -->

Follow-ups in one session:

<!-- BEGIN GENERATED: e-warm -->
| Turn | Answer | Sandbox tool, s | Turn, s |
|---|---|---:|---:|
| 1 (new session; includes the sandbox create) | OK | 0.14 | 2.76 |
| 2 | LANTERN-7 | 0.033 | 5.03 |
| 3 | WARM-1 | 0.029 | 3.19 |
| 4 | WARM-2 | 0.028 | 3.39 |
<!-- END GENERATED: e-warm -->

At once and sandbox replacement:

<!-- BEGIN GENERATED: e-scale -->
| At once | Answered | Refused | Sandbox create + turn 1, s |
|---:|---|---|---|
| 4 | 4/4 | 0/4 | 3.9, 3.12, 3.9, 3.17 |
| 8 | 8/8 | 0/8 | 4.59, 2.48, 4.63, 6.94, 5.4, 3.7, 7.34, 3.82 |

Sandbox replacement after `kill`: new Firecracker sandbox Ready in 0.1 s; follow-up on a live session HTTP 200, 2.61 s, "LANTERN-7", state preserved: true. No count of live micro-VMs is published for this run: the driver counted them with `pgrep -fc`, which also counted its own shell.
<!-- END GENERATED: e-scale -->

**Boundary.**
- Superseded: this run used the earlier executor (`base` template, `python3` through `commands.run`); `e-2026-10-17-ci-perf` and `e-2026-10-18-ci-scale` repeat the protocol with `code-interpreter-v1`.
- Single-VM deployment (`Standard_D4ds_v7`, 4 vCPU / 16 GiB RAM). At concurrency 8, one follow-up turn (`SCALE-8`) took 78.99 s while all 8/8 sessions answered and remembered their code words; the record does not say where that time went.
- Like D's recovery run, the follow-up after `kill` went to a different live session (`LANTERN-7`): the host kept serving and the replacement sandbox was ready in <!--v:e.pod_lost.value-->0.1<!--/v--> s. It does not show continuity of the killed sandbox.
- The numbers came from the author's private driver; `tools/measure_e2b.py perf` reproduces its agent, prompts and clocks but was not the program that produced this record.

### e-2026-10-15-latest: the newest E2B SDK, and E2B's official code-interpreter template

**Question.** Does the newest E2B Python SDK (`2.56.0`, released 2026-10-13) still run the measured protocol against this Embed install, and does E2B's own code-interpreter template (`e2b-code-interpreter 2.11.0`) build and work on it?

**Input.** On host a of `e-2026-10-15-multi` ([`evidence/runs/e-e2b-2026-10-15/latest.json`](evidence/runs/e-e2b-2026-10-15/latest.json)):

<!-- BEGIN GENERATED: e-latest-input -->
> SDK check: `tools/measure_e2b.py agent --runs 3` with `e2b 2.56.0` (Python 3.12.3), tool SHA-256 `036dc81db607…`; "Where is Bangkok? Answer in one short sentence.", then "How many questions have I asked before this one? Number only." 3 times, pausing for 5 s after the second turn.
> code-interpreter: template `code-interpreter-v1` (2 vCPU, 2048 MB) from [`e2b-dev/code-interpreter@162a2a2`](https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template), built with its unchanged `template/build_prod.py` (build environment `e2b 2.32.0`); checked with `e2b-code-interpreter 2.11.0` and `tools/measure_code_interpreter.py`.
> Cells, in order, with no model: "x = 1", "x+=1; x", a pandas sum, a matplotlib plot, JavaScript "[1, 2, 3].reduce((a, b) => a + b)", "1/0", then after pause and resume "x".
<!-- END GENERATED: e-latest-input -->

**Varied and fixed.** The SDK check kept the agent, prompts, protocol and clocks of `e-2026-10-14-agent` and ran on host a with the tool revision of `e-2026-10-15-multi`. Against `e-2026-10-14-agent` it changed the SDK (2.46.0 → 2.56.0) and the program (the private driver → `tools/measure_e2b.py`); host a had since booted kernel `7.0.0-1017-azure`, and its OS disk was Standard HDD. The code-interpreter template was built with E2B's unchanged script; between build attempts we raised three host settings, listed under the build table. The 4 check runs used the same template and steps.

**Result.** The SDK check:

<!-- BEGIN GENERATED: e-latest-sdk -->
| Run | Create → Running | First answer | Cold | Follow-up | Pause | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1 | 5.24 | 5.34 | 3.7 | 0.15 | 0.08 | 6.72 | 6.8 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.06 | 4.93 | 4.99 | 4.24 | 0.14 | 0.08 | 6.2 | 6.28 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| 3 | 0.06 | 4.0 | 4.06 | 35.02 | 0.17 | 0.11 | 8.14 | 8.25 | Bangkok is the capital city of Thailand. / 2 / 3 / 4 |
| **median** | **0.06** | **4.93** | **4.99** | **4.24** | **0.15** | **0.08** | **6.72** | **6.8** |  |

Model HTTP attempts: 200 ×24. History grew 1, 2, 3, 4 in 3 of 3 runs; no sandbox was left.
<!-- END GENERATED: e-latest-sdk -->

Building `code-interpreter-v1` (30 steps):

<!-- BEGIN GENERATED: e-latest-build -->
| Attempt | HUGEPAGES | Build workspace, MiB | OS disk, GB | Result | Seconds |
|---:|---:|---:|---:|---|---:|
| 1 | 2048 | 512 | 50 | failed at step 6/30: `mmap memfd: cannot allocate memory` while resuming the build VM | 104.5 |
| 2 | 4096 | 512 | 50 | failed at step 8/30: `pip install` in the build VM: `[Errno 28] No space left on device` | 56.5 |
| 3 | 4096 | 8192 | 50 | stopped by us at step 12/30: below 8 GiB free on the OS disk, which also holds the E2B database | 501 |
| 4 | 4096 | 8192 | 128 | built, 30/30 steps; steps 1–11 from attempt 3's cache | 426 |

By step 12 attempt 3 had written 12839 MiB of templates and 10140 MiB under the orchestrator's directory, with 5084 MiB left on the OS disk; attempt 4 then used 29 GiB (free 93 → 64 GiB). What we changed: `HUGEPAGES` in the Compose `.env` 2048 → 4096, then `docker compose up -d --wait`; the test team's build workspace (`disk_mb`) 512 → 8192 MiB, by adding one row to Embed's internal `project_limits` table; the OS disk 50 → 128 GB (deallocate, snapshot, `az disk update --size-gb`, start).
<!-- END GENERATED: e-latest-build -->

The checks, 4 runs:

<!-- BEGIN GENERATED: e-latest-ci -->
| Step | Passed | Seconds, min–max | Observed in run 4 |
|---|---:|---:|---|
| `create (default template)` | 4/4 | 0.116–0.194 |  |
| `README: x = 1` | 4/4 | 0.082–0.11 |  |
| `README: x+=1; x` | 4/4 | 0.011–0.012 | `2` |
| `pandas sum` | 4/4 | 0.565–1.303 | `10` |
| `matplotlib png` | 4/4 | 0.977–3.317 | `png` |
| `javascript kernel` | 0/4 | 0.03–0.064 | `"<template id>"[(1 , 2 , 3)].reduce is not a function` |
| `exception` | 4/4 | 0.038–0.042 | `ZeroDivisionError` |
| `pause` | 4/4 | 0.157–0.4 |  |
| `connect (resume)` | 4/4 | 0.233–0.902 |  |
| `x after resume` | 4/4 | 0.088–0.652 | `2` |
| `kill` | 4/4 | 0.057–0.746 |  |

Why JavaScript failed, reproduced in new sandboxes: JavaScript as the first cell of a new sandbox, then the same cell again: `TypeError` → `6`; a Python cell first, then the JavaScript cell (the order of the check): ok → `TypeError`; the JavaScript cell with a leading semicolon: `6`. In `template/server/messaging.py` at `162a2a2`, line 160 writes `process.env['KEY'] = 'VALUE'` with no semicolon, and lines 351–361 put those lines before the user code on a context's first execution, so code that starts with `[` continues the last of them. Cells that start with `console.log` or a number ran.
<!-- END GENERATED: e-latest-ci -->

**Boundary.**
- Superseded for the SDK check: it used the earlier executor (`base` template, `python3` through `commands.run`); `e-2026-10-17-ci-agent` and `e-2026-10-18-ci-multi` run the same protocol with `code-interpreter-v1`. The template build and the code-interpreter checks stand.
- With Embed's defaults the official template does not build: the hugepage pool, the build workspace and the 50 GB OS disk each stopped an attempt. The build workspace was raised by adding one row to Embed's internal `project_limits` table, which E2B does not document; a run with that change shows the template working, not a supported configuration.
- The build disk need is from this one build: attempt 4 reused attempt 3's cache for steps 1–11, so a clean build would need about the sum of both, which was not measured.
- Each `code-interpreter-v1` sandbox is 2048 MB, four times `base`; how many fit on one host was not measured here (`e-2026-10-17-ci-perf` measured it on host c).
- The JavaScript failure is in E2B's template server, not in Azure or the host; whether E2B Cloud, which runs the same server code, fails the same way was not checked.
- Host a was deallocated by a subscription automation at 2026-10-15T16:48Z before cleanup, so the raised settings and the built template remain on its disk.

### e-2026-10-15-multi: E2B on two independent Azure VMs

**Question.** E2B Embed is one machine by design. What does a second, independent install add: does the protocol give the same result on another VM, does capacity add up when a client spreads sessions over two hosts, and what survives when one host's E2B stack stops?

**Input.** Two VMs, one tool run on host a for every phase ([`evidence/runs/e-e2b-2026-10-15/multi.json`](evidence/runs/e-e2b-2026-10-15/multi.json)):

<!-- BEGIN GENERATED: e-multi-input -->
> Tool: `tools/measure_e2b.py` (SHA-256 `036dc81db607…`); the agent runs on host b used the revision `89255d0b00a3…`, which did not yet record HTTP statuses. Agent instructions, tool schema and prompts as in `e-2026-10-14-agent` and `e-2026-10-14-perf`.
> Host a: `Standard_D4ds_v7`, zone `1`, host kernel `7.0.0-1017-azure`, Docker `29.9.0`, OS disk `Standard HDD (Standard_LRS) · 50 GB`, `HUGEPAGES=2048`; installed on 2026-10-14 for `e-2026-10-14-agent`, deallocated overnight, then started and brought up once with `docker compose up -d --wait`, as E2B's guide asks after a reboot.
> Host b: `Standard_D4ds_v7`, zone `1`, host kernel `6.17.0-1022-azure`, Docker `29.9.1`, OS disk `Premium SSD P6 (Premium_LRS) · 50 GB`, `HUGEPAGES=2048`; installed on 2026-10-15 with README step 7 (`compose.yaml` and `.env` at `588e5b9`, both SHA-256 OK) in 175 s.
> Network: both hosts in one subnet and zone; the tool ran on host a and reached host b's ports `3000` and `3002` over the private network; host b's NSG admitted only host a's private address on those two ports and no other inbound traffic.
> Agent runs on host b: "Where is Bangkok? Answer in one short sentence.", then "How many questions have I asked before this one? Number only." 3 times, pausing for 5 s after the second turn.
> One host at once, on host a: 8 sessions; new session "Use run_in_e2b_sandbox to print COLD-1, then reply with exactly COLD-1" and one session with "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK." before it.
> At once, per session: "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only.".
> Host stop: on each host one live session ("Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK.") and one paused sandbox ("Use run_in_e2b_sandbox to write PAUSED-{h} to /home/user/paused.txt. Reply only OK.", `{h}` = the host's name); then `docker compose stop` on host a with the VM running, later `docker compose up -d --wait`; reads back "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only." and "Use run_in_e2b_sandbox to read /home/user/paused.txt. What does it say? Word only.".
> Idle capacity: `Sandbox.create("base")` on host a, one after another with no model and no command, until a create was refused, 3 times.
<!-- END GENERATED: e-multi-input -->

**Varied and fixed.** Fixed: VM size, image, zone, E2B commit, compose files, hugepage pool, SDK `e2b 2.46.0`, model, agent and prompts. Varied: the host, and how many hosts the sessions were spread over. Not fixed: host a's kernel and OS disk differ from host b's (below).

**Result.** The 3-run agent protocol on host b:

<!-- BEGIN GENERATED: e-multi-agent -->
| Run | Create → Running | First answer | Cold | Follow-up | Pause | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.09 | 5.58 | 5.67 | 6.9 | 0.15 | 0.08 | 6.87 | 6.95 | Bangkok is the capital city of Thailand, in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.06 | 3.85 | 3.91 | 5.92 | 0.11 | 0.1 | 7.05 | 7.15 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| 3 | 0.06 | 45.56 | 45.62 | 3.6 | 0.12 | 0.08 | 45.53 | 45.61 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| **median** | **0.06** | **5.58** | **5.67** | **5.92** | **0.12** | **0.08** | **7.05** | **7.15** |  |

History inside the sandbox grew 1, 2, 3, 4 in 3 of 3 runs. Run 3's slow turns were model turns: its first answer took 45.56 s, of which the sandbox command took 0.11 s.
<!-- END GENERATED: e-multi-agent -->

Sessions at once:

<!-- BEGIN GENERATED: e-multi-scale -->
| Phase | Sessions at once | Answered | Refused | Refusal cause | Sandbox create, s | Whole session, s | Model HTTP attempts |
|---|---:|---|---|---|---:|---:|---|
| host a alone | 8 | 7/8 (a 7/8) | 1/8 | `cannot allocate memory` | 0.164–0.53 | 8.03–25.65 | 200 ×28 |
| hosts a and b | 16 | 14/16 (a 7/8, b 7/8) | 2/16 | `cannot allocate memory` | 0.147–0.432 | 8.57–42.09 | 200 ×56, 500 ×2 |

The client spread the 16 sessions over the two hosts in turn. Each refusal was E2B's `500: Failed to place sandbox`, with `cannot allocate memory` from the orchestrator behind it; the model's 2 HTTP 500 answers were retried within their turns and those sessions answered.
<!-- END GENERATED: e-multi-scale -->

Idle `base` sandboxes on host a:

<!-- BEGIN GENERATED: e-multi-idle -->
| Run | Created before the refusal | Refusal | Hugepages taken: before → at the refusal | After killing them |
|---:|---:|---|---|---|
| 1 | 5 | `error setting fc metrics (connection refused)` | 75 → 1355 | 75 pages, 0 sandboxes |
| 2 | 7 | `cannot allocate memory` | 75 → 1867 | 75 pages, 0 sandboxes |
| 3 | 7 | `cannot allocate memory` | 75 → 1867 | 75 pages, 0 sandboxes |

Pages are 2 MiB. Each `base` sandbox (512 MiB) reserved 256 pages when it was created, idle or not. The pool is 2048 pages and postgres in the Compose stack holds 75, which leaves 1973 pages: room for 7 sandboxes. Run 1 stopped earlier on a different error.
<!-- END GENERATED: e-multi-idle -->

Stopping host a's E2B stack with `docker compose stop` (the VM kept running):

<!-- BEGIN GENERATED: e-multi-hostloss -->
| Check | Observed |
|---|---|
| host b, live session: read its file | ok, 4.72 s, "LANTERN-7" |
| host b: new session | ok, 4.38 s, "COLD-1" |
| host b: resume host a's paused sandbox | failed: `SandboxNotFoundException: Paused sandbox … not found` |
| host a, stopped: create | failed: `connection refused` |
| host a, stopped: resume its own paused sandbox | failed: `connection refused` |
| host a, after restart: resume its own paused sandbox | ok, 4.72 s, "PAUSED-a" |
| host a, after restart: the sandbox that was running when it stopped | failed: `SandboxNotFoundException: Paused sandbox … not found` |
| host a, after restart: new session | ok, 4.09 s, "COLD-2" |
| `docker compose stop` / `docker compose up -d --wait` | 73.69 s / 59.55 s |
<!-- END GENERATED: e-multi-hostloss -->

**Boundary.**
- Superseded: this run used the earlier executor (`base` template, `python3` through `commands.run`); `e-2026-10-18-ci-multi` repeats the agent runs, the at-once phases and the host stop with `code-interpreter-v1`, and `e-2026-10-18-ci-host` measures the hugepages each `code-interpreter-v1` sandbox takes.
- Two independent installs and a client that spreads sessions over them, not an E2B cluster: there is no shared scheduler, no failover and no shared storage. The sessions per host add up; a paused sandbox can only resume on the host that paused it.
- The host stop was `docker compose stop` on a running VM. A lost VM or disk, and the official Terraform module's scale-set replacement from a fresh disk, were not measured.
- The two hosts were not identical: host a booted kernel `7.0.0-1017-azure` after its overnight deallocation and its OS disk had been changed from Premium SSD to Standard HDD by a subscription policy after the 2026-10-14 runs; host b ran `6.17.0-1022-azure` on Premium SSD.
- One run of each phase; capacity is for the 512 MB `base` template and the Compose default pool on a `Standard_D4ds_v7` (`e-2026-10-17-ci-perf` measures `code-interpreter-v1` and a 4096-page pool).

### e-2026-10-17-ci-agent: E2B's Code Interpreter as the Prompt Agent's sandbox, 3 runs

**Question.** E2B's README runs code in a sandbox with its Code Interpreter: `Sandbox.create()` from `e2b-code-interpreter`, then `run_code`, in the official `code-interpreter-v1` template. When the Prompt Agent's tool runs its Python that way instead of `commands.run("python3")` in `base`, how fast are create, the first answer, pause, resume and resume → answer under A's 3-run protocol, and what does building the template take on a fresh install?

**Input.** On host c, a third Embed install ([`evidence/runs/e-e2b-2026-10-17/ci-agent.json`](evidence/runs/e-e2b-2026-10-17/ci-agent.json)):

<!-- BEGIN GENERATED: e-ci-input -->
> Host c: `Standard_D4ds_v7` (4 vCPU, 16 GiB), zone `2`, 128 GB OS disk; E2B Embed `588e5b9` installed on 2026-10-16 from the two files README step 7 downloads (`compose.yaml` and `.env`, SHA-256 OK), then `HUGEPAGES=4096` appended to `.env`.
> Template: `code-interpreter-v1` (2 vCPU, 2048 MB) from [`e2b-dev/code-interpreter@162a2a2`](https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template), built on host c with its unchanged `template/build_prod.py` (build table below).
> Executor: E2B Code Interpreter, `e2b 2.56.0` and `e2b-code-interpreter 2.11.0`; each call of the `run_in_e2b_sandbox` tool is one `run_code` cell in the sandbox's Jupyter kernel: the tool's history step, then the model's Python.
> Tool: `tools/measure_e2b.py` (SHA-256 `cbc96f273c07…`) with `--executor code-interpreter`; model `gpt-5.4-mini`; agent instructions and tool schema as in `e-2026-10-14-agent`.
> Agent runs: "Where is Bangkok? Answer in one short sentence.", then "How many questions have I asked before this one? Number only." 3 times, pausing for 5 s after the second turn.
> Perf: new sessions "Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}" (n = 1–5); one session "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only.", then "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1", then "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2"; 4 and 8 sessions at once, each "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."; replacement: a live session ran "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK." and, after another sandbox was killed and replaced, "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only.".
> Idle capacity: `Sandbox.create` of each template one after another, checked with `run_code("1+1")` or `echo ok` and no model, until a create was refused; at `HUGEPAGES=4096`, then at the Compose default 2048, then 4096 restored. The probe is the author's script, SHA-256 in the record.
<!-- END GENERATED: e-ci-input -->

**Varied and fixed.** Against the SDK check of `e-2026-10-15-latest`, the executor changed (`commands.run("python3")` in `base` → `run_code` in `code-interpreter-v1`), and so did the host (host c, zone `2`) and the tool revision, which adds the executor and keeps the agent, prompts and clocks. The build used the three host settings that let host a build on 2026-10-15.

**Result.** The 3-run agent protocol:

<!-- BEGIN GENERATED: e-ci-agent -->
| Run | Create → Running | First answer | Cold | Follow-up | Pause | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.22 | 3.14 | 3.36 | 4.09 | 0.29 | 0.63 | 4.33 | 4.96 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.08 | 3.85 | 3.93 | 4.17 | 0.09 | 0.64 | 4.64 | 5.28 | Bangkok is the capital of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 3 | 0.07 | 4.55 | 4.62 | 3.61 | 0.08 | 1.43 | 4.77 | 6.2 | Bangkok is the capital of Thailand. / 1 / 2 / 3 |
| **median** | **0.08** | **3.85** | **3.93** | **4.09** | **0.09** | **0.64** | **4.64** | **5.28** |  |

History inside the sandbox grew 1, 2, 3, 4 in 3 of 3 runs. Every turn made one `run_code` call, which took 0.011–0.123 s of the turn; guest kernel `6.1.177+`. Model HTTP attempts: 200 ×24. No sandbox, conversation or agent version was left.
<!-- END GENERATED: e-ci-agent -->

Building `code-interpreter-v1` on host c (30 steps):

<!-- BEGIN GENERATED: e-ci-build -->
| Attempt | Result | Seconds |
|---:|---|---:|
| 1 | failed at step 8/30: `pip install --no-cache-dir -r requirements.txt: [Errno 28] No space left on device` | 181 |
| 2 | failed at step 20/30: `error building step 20: resume sandbox: failed to start FC: error setting fc metrics` | 840 |
| 3 | built, 30/30 steps; steps 1–19 from attempt 2's cache (builder: 1m25s) | 92 |

All three attempts ran with the settings that let host a build on 2026-10-15: `HUGEPAGES` 4096, the test team's build workspace (`disk_mb`) 8192 MiB in one `project_limits` row, and a 128 GB OS disk; nothing changed between attempts. Attempt 1 started 16 s after the row was added and ran out of space at step 8; attempt 2, started about 5 minutes after the row, passed that step. That fits the new limit taking effect only after a delay, which we did not verify further. Before attempt 3 a subscription automation had deallocated the VM and we started it again.
<!-- END GENERATED: e-ci-build -->

**Boundary.**
- 3 runs on one host. The tool ran on the host itself (`localhost`), so no network between client and Embed is in these numbers. Compare with the SDK check of `e-2026-10-15-latest` as a field report: the host, its disk and the template differ, not only the executor.
- Resume took <!--v:e.ci.resume.median-->0.64<!--/v--> s median here against <!--v:e.latest.sdk.resume.median-->0.08<!--/v--> s for `base` on host a. The 2 GiB sandbox has four times `base`'s memory and the hosts differ; this run does not separate the two. `e-2026-10-18-ci-multi` ran this protocol with `code-interpreter-v1` on host a: resume <!--v:e.m.a.resume.median-->0.39<!--/v--> s median (<!--v:e.m.a.resume.range-->0.38–1.46<!--/v--> s), between the two: on host a the 2 GiB template resumed more slowly than `base`, and on host c more slowly than on host a; one run each, on different days.
- The successful build reused steps 1–19 of the failed attempt 2, so its 92 s is not a clean build time; host a's 426 s in `e-2026-10-15-latest` also reused steps 1–11.
- The `project_limits` row is an undocumented change to Embed's internal table, as in `e-2026-10-15-latest`; that it takes effect only after a delay is inferred from two attempts, not measured.
- The step 20 failure (`error setting fc metrics`) did not recur after the VM restart; its cause was not found.

### e-2026-10-17-ci-perf: cold start, follow-ups, concurrency, replacement and idle capacity with code-interpreter-v1

**Question.** Under the D-matched protocol of `e-2026-10-14-perf`, how does `code-interpreter-v1` (2 GiB per sandbox) behave on one `Standard_D4ds_v7`, and how many such sandboxes fit on it?

**Input.** The host, tool and agent of `e-2026-10-17-ci-agent`, running `perf --cold 5 --scale 4,8 --recovery --executor code-interpreter`, then the idle-capacity probe ([`evidence/runs/e-e2b-2026-10-17/ci-perf.json`](evidence/runs/e-e2b-2026-10-17/ci-perf.json)); every prompt is listed under `e-2026-10-17-ci-agent`.

**Varied and fixed.** Cold runs repeated 5 new sessions; concurrency varied 4 and 8 sessions at once; the idle probe varied the template and the hugepage pool (`HUGEPAGES=4096`, then the Compose default 2048).

**Result.** New sessions:

<!-- BEGIN GENERATED: e-ci-cold -->
| Start (UTC) | Answer | Sandbox created → Ready, s | Agent turn, s | New session → answer, s |
|---|---|---:|---:|---:|
| 16:56:18 | COLD-1 | 0.13 | 3.17 | 3.35 |
| 16:56:21 | COLD-2 | 0.073 | 5.43 | 5.66 |
| 16:56:27 | COLD-3 | 0.072 | 3.48 | 3.64 |
| 16:56:31 | COLD-4 | 0.069 | 3.12 | 3.31 |
| 16:56:34 | COLD-5 | 0.1 | 2.5 | 2.7 |

Across the 5 new sessions: sandbox create → Ready 0.073 s median (0.069–0.13; n=5), Prompt Agent turn 3.17 s median (2.5–5.43; n=5), end-to-end new session → answer 3.35 s median (2.7–5.66; n=5).
<!-- END GENERATED: e-ci-cold -->

Follow-ups in one session:

<!-- BEGIN GENERATED: e-ci-warm -->
| Turn | Answer | Sandbox tool, s | Turn, s |
|---|---|---:|---:|
| 1 (new session; the sandbox was created before it) | OK | 0.05 | 3.68 |
| 2 | LANTERN-7 | 0.014 | 3.36 |
| 3 | WARM-1 | 0.012 | 3.93 |
| 4 | WARM-2 | 0.012 | 4.59 |

The sandbox was created in 0.07 s before turn 1. Follow-ups 2–4: 3.93 s median (3.36–4.59; n=3).
<!-- END GENERATED: e-ci-warm -->

At once and sandbox replacement:

<!-- BEGIN GENERATED: e-ci-scale -->
| At once | Answered | Refused | Sandbox create, s | Whole session, s | Model HTTP attempts |
|---:|---|---|---:|---:|---|
| 4 | 3/4 | 1/4 | 0.23–0.288 | 6.43–15.07 | 200 ×12 |
| 8 | 3/8 | 5/8 | 0.124–0.231 | 5.94–8.59 | 200 ×12 |

Each refused session got E2B's `500: Failed to place sandbox` at create and never reached the model. Sandbox replacement after `kill`: a new `code-interpreter-v1` sandbox was Ready in 0.12 s; follow-up on a live session HTTP 200, 6.86 s, "LANTERN-7", state preserved: true. No sandbox or conversation was left.
<!-- END GENERATED: e-ci-scale -->

Idle sandboxes by template and hugepage pool:

<!-- BEGIN GENERATED: e-ci-idle -->
| `HUGEPAGES` (2 MiB pages) | Free at start | `code-interpreter-v1` (2048 MB = 1024 pages) | `base` (512 MB = 256 pages) |
|---:|---:|---:|---:|
| 4096 | 4083 | 3 | 15 |
| 2048 | 2035 | 1 | 7 |

Each count is the free pages divided by the template's pages, rounded down: with 3 `code-interpreter-v1` sandboxes running, the next one needs 1024 pages when 4083 − 3 × 1024 = 1011 are left. That matches each sandbox reserving its whole memory from the pool at create, as `e-2026-10-15-multi` measured for `base`; this run recorded only `HugePages_Free`, which counts pages already touched (261–285 per idle `code-interpreter-v1` sandbox), not reservations. Every create ended in `500: Failed to place sandbox`, every sandbox was killed, and the pool went back to 4096 pages.
<!-- END GENERATED: e-ci-idle -->

**Boundary.**
- One run of each phase on one host; the tool ran on the host itself.
- With `HUGEPAGES=4096` the host held 3 sandboxes, so 4 and 8 sessions at once measure refusals at a full pool, not throughput. MemAvailable was 5700 MiB with the 4096-page pool and 9782 MiB with 2048; a larger pool was not tried.
- The pages each sandbox takes are inferred from the counts and the template memory; `HugePages_Rsvd` was not recorded.
- As in `e-2026-10-14-perf`, the follow-up after `kill` went to a different live session; it does not show continuity of the killed sandbox.

### e-2026-10-18-ci-host: code-interpreter-v1 on host c with no model, 20 pause and resume cycles and the hugepages per sandbox

**Question.** Without a model in the way, how long do create, a cell, pause and resume take for a 2 GiB `code-interpreter-v1` sandbox when 20 run one after another, does the kernel's state come back after each pause, and how many hugepages does one sandbox take, which sets how many fit on a host?

**Input.** The tool's `host` mode on host c ([`evidence/runs/e-e2b-2026-10-18/ci-host.json`](evidence/runs/e-e2b-2026-10-18/ci-host.json)):

<!-- BEGIN GENERATED: e-host-input -->
> Host c: `Standard_D4ds_v7` (4 vCPU, 16 GiB), zone `2`, OS disk `Standard HDD (Standard_LRS) · 128 GB`, `HUGEPAGES=4096`; the install of `e-2026-10-17-ci-agent`, where `code-interpreter-v1` was built.
> Tool: `tools/measure_e2b.py` (SHA-256 `aca40d17e315…`), `measure_e2b.py host --bench 20 --reserve 4` on host c itself against `localhost`; no model and no agent; E2B Code Interpreter only (`e2b 2.56.0`, `e2b-code-interpreter 2.11.0`, template `code-interpreter-v1`, 2 vCPU, 2048 MB).
> Read-back: one sandbox runs the cell `import os, platform; mem_kb = next(int(ln.split()[1]) for ln in open('/proc/meminfo') if ln.startswith('MemTotal:')); print(platform.release(), os.cpu_count(), mem_kb // 1024)` while the tool lists the host's Firecracker processes from `/proc/*/cmdline`.
> Cycles: 20 sandboxes one after another, each `Sandbox.create` → cell `x = 41` → cell `x += 1` → `beta_pause` → wait 5 s → `Sandbox.connect` → cell `print(x)`, which must print `42` → `kill`.
> Reservation: sandboxes created one by one and kept, each checked with `print(1 + 1)`; `/proc/meminfo` read before the first and after each, until a create was refused; then every sandbox killed and `/proc/meminfo` read again 5 s later.
<!-- END GENERATED: e-host-input -->

**Varied and fixed.** Fixed: host c, its install, pool and template, the SDKs and the tool; no model, no agent and no network between the tool and E2B. Varied: the cycle, and in the reservation the number of sandboxes kept at once.

**Result.** The 20 cycles:

<!-- BEGIN GENERATED: e-host-bench -->
| Cycle | Create | First cell | Next cell | Pause | Resume | Cell after resume | `x` after resume |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.088 | 0.091 | 0.009 | 0.14 | 0.55 | 0.08 | 42 |
| 2 | 0.099 | 0.1 | 0.009 | 2.99 | 0.38 | 0.074 | 42 |
| 3 | 0.091 | 0.1 | 0.011 | 0.53 | 0.4 | 0.082 | 42 |
| 4 | 0.098 | 0.17 | 0.043 | 3.07 | 5.31 | 0.13 | 42 |
| 5 | 0.12 | 0.095 | 0.011 | 0.87 | 6.3 | 0.05 | 42 |
| 6 | 0.26 | 0.17 | 0.048 | 0.62 | 6.4 | 0.044 | 42 |
| 7 | 0.12 | 0.11 | 0.018 | 3.15 | 4.84 | 0.071 | 42 |
| 8 | 0.56 | 0.08 | 0.008 | 1.07 | 5.54 | 0.062 | 42 |
| 9 | 0.48 | 0.1 | 0.008 | 0.66 | 7.33 | 0.11 | 42 |
| 10 | 0.17 | 0.14 | 0.011 | 1.48 | 5.38 | 0.067 | 42 |
| 11 | 0.13 | 0.095 | 0.009 | 0.67 | 10.24 | 0.058 | 42 |
| 12 | 0.3 | 0.11 | 0.008 | 1.8 | 4.68 | 0.066 | 42 |
| 13 | 0.18 | 0.12 | 0.009 | 1.36 | 5.91 | 0.17 | 42 |
| 14 | 0.24 | 0.094 | 0.024 | 1.85 | 4.99 | 0.13 | 42 |
| 15 | 0.14 | 0.12 | 0.018 | 3.27 | 3.84 | 0.15 | 42 |
| 16 | 0.27 | 0.19 | 0.01 | 2.03 | 4.7 | 0.05 | 42 |
| 17 | 0.29 | 0.17 | 0.008 | 0.56 | 8.08 | 0.065 | 42 |
| 18 | 0.38 | 0.1 | 0.009 | 1.47 | 4.91 | 0.12 | 42 |
| 19 | 0.11 | 0.13 | 0.031 | 1.74 | 6.16 | 0.21 | 42 |
| 20 | 0.14 | 0.18 | 0.025 | 3.06 | 4.61 | 0.071 | 42 |

Seconds. Over the 20 cycles: create 0.088–0.56 s (n=20), 4 of 20 under 0.1 s; first cell 0.08–0.19 s (n=20); next cell 0.008–0.048 s (n=20); pause 0.14–3.27 s (n=20), 13 of 20 over 1 s; resume 0.38–10.24 s (n=20): the first 3 cycles at most 0.55 s, every later one 3.84–10.24 s; cell after resume 0.044–0.21 s (n=20). `print(x)` printed 42 in 20 of 20 cycles: the kernel's variable came back with the memory snapshot. No sandbox was left.
<!-- END GENERATED: e-host-bench -->

Hugepages taken as sandboxes were added:

<!-- BEGIN GENERATED: e-host-reserve -->
| Sandboxes | Create, s | `HugePages_Free` | `HugePages_Rsvd` | Pages taken | MemAvailable, MiB |
|---|---:|---:|---:|---:|---:|
| 0 (before) |  | 4083 | 62 | 75 | 5634 |
| 1 | 0.22 | 3815 | 818 | 1099 | 5628 |
| 2 | 0.14 | 3522 | 1549 | 2123 | 5623 |
| 3 | 0.08 | 3251 | 2302 | 3147 | 5631 |
| 4: refused | 0.217 |  |  |  |  |
| 0 (all killed) |  | 4083 | 62 | 75 | 5680 |

Pages taken = `HugePages_Total` (4096) − free + reserved; pages are 2 MiB. Each sandbox took 1024 pages, its whole 2048 MiB, as soon as it was created; 75 pages were taken by other processes on host c. With 3 sandboxes, 4096 − 75 − 3 × 1024 = 949 pages were left, fewer than one sandbox needs, and create 4 got `500: Failed to place sandbox` in 0.217 s. MemAvailable stayed at 5623–5634 MiB: the pool is set aside at boot, so sandbox memory does not show there.

Read-back: Firecracker `v1.14-0.2.0` behind the sandbox in the host's process list; inside it kernel `6.1.177+`, 2 vCPU and MemTotal 1982 MiB for the template's 2048 MB; host kernel `7.0.0-1017-azure`.
<!-- END GENERATED: e-host-reserve -->

**Boundary.**
- One host and one run of each probe; the tool ran on host c itself, so these are the E2B calls alone, not agent turns.
- The jump in resume time after the first cycles is recorded, not explained. Disk activity was not recorded; E2B writes each pause snapshot (`vm_state` + `mem_file`) to the host's OS disk, a Standard HDD on host c, so a busy disk is a candidate cause that this run does not test.
- The cycles ran back to back with only the 5 s wait before each resume; in the agent runs, model turns of several seconds separate one pause from the next, and resume took <!--v:e.ci.resume.range-->0.63–1.43<!--/v--> s on host c (`e-2026-10-17-ci-agent`) and <!--v:e.m.a.resume.range-->0.38–1.46<!--/v--> s on host a (`e-2026-10-18-ci-multi`).

### e-2026-10-18-ci-multi: code-interpreter-v1 on two independent Azure VMs

**Question.** With E2B's Code Interpreter in `code-interpreter-v1`, does the agent protocol give the same result on host a, does capacity add up when a client spreads sessions over two installs, and what survives when one host's E2B stack stops?

**Input.** Hosts a and c, the tool on host a for every phase ([`evidence/runs/e-e2b-2026-10-18/ci-multi.json`](evidence/runs/e-e2b-2026-10-18/ci-multi.json)):

<!-- BEGIN GENERATED: e-cm-input -->
> Host a: `Standard_D4ds_v7` (4 vCPU, 16 GiB), zone `1`, OS disk `Standard HDD (Standard_LRS) · 128 GB`, `HUGEPAGES=4096`; installed for `e-2026-10-14-agent`; the pool raised to 4096 pages, the build workspace row added, the OS disk grown to 128 GB and `code-interpreter-v1` built in `e-2026-10-15-latest`.
> Host c: `Standard_D4ds_v7` (4 vCPU, 16 GiB), zone `2`, OS disk `Standard HDD (Standard_LRS) · 128 GB`, `HUGEPAGES=4096`; the install of `e-2026-10-17-ci-agent`.
> Tool: `tools/measure_e2b.py` (SHA-256 `aca40d17e315…`), every phase run on host a; E2B Code Interpreter only (`e2b 2.56.0`, `e2b-code-interpreter 2.11.0`, template `code-interpreter-v1`, 2 vCPU, 2048 MB).
> Network: the tool reached host c's ports `3000` and `3002` over the private network; host c's NSG admitted only host a's private address on those two ports and denied every other inbound connection. Each host's API key came from an environment variable named in the hosts file.
> Agent runs on host a: "Where is Bangkok? Answer in one short sentence.", then "How many questions have I asked before this one? Number only." 3 times, pausing for 5 s after the second turn; agent instructions and tool schema as in `e-2026-10-14-agent`.
> At once, 4 sessions on host a alone, then 4 on each host: "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only.".
> Host stop: on each host one live session ("Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK.") and one paused sandbox ("Use run_in_e2b_sandbox to write PAUSED-{h} to /home/user/paused.txt. Reply only OK.", `{h}` = the host's name); then `docker compose stop` on host a with the VM running, later `docker compose up -d --wait`; reads back "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only." and "Use run_in_e2b_sandbox to read /home/user/paused.txt. What does it say? Word only.".
<!-- END GENERATED: e-cm-input -->

**Varied and fixed.** Fixed: the template, the pool (`HUGEPAGES=4096` on both), the SDKs, the tool, the model, the agent and the prompts. Varied: the host, and how many hosts the sessions were spread over. Not fixed: the hosts are in different zones (host a in `1`, host c in `2`), and host a has run E2B since 2026-10-14.

**Result.** The 3-run agent protocol on host a:

<!-- BEGIN GENERATED: e-cm-agent -->
| Run | Create → Running | First answer | Cold | Follow-up | Pause | Resume | Answer after resume | Resume → answer | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 10.53 | 2.95 | 13.48 | 3.03 | 2.08 | 0.39 | 3.91 | 4.3 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.09 | 2.65 | 2.74 | 4.02 | 0.16 | 1.46 | 2.68 | 4.14 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 3 | 0.22 | 2.5 | 2.72 | 2.59 | 0.15 | 0.38 | 3.36 | 3.74 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| **median** | **0.22** | **2.65** | **2.74** | **3.03** | **0.16** | **0.39** | **3.36** | **4.14** |  |

History inside the sandbox grew 1, 2, 3, 4 in 3 of 3 runs. Run 1's create took 10.53 s, the first `code-interpreter-v1` create of this run on host a; the others took 0.09, 0.22 s. The record does not show where that time went. Every turn made one `run_code` call, 0.013–0.574 s of the turn; guest kernel `6.1.177+`. Model HTTP attempts: 200 ×24. No sandbox, conversation or agent version was left.
<!-- END GENERATED: e-cm-agent -->

Sessions at once:

<!-- BEGIN GENERATED: e-cm-scale -->
| Phase | Sessions at once | Answered | Refused | Sandbox create, s | Whole session, s | Model HTTP attempts |
|---|---:|---|---|---:|---:|---|
| host a alone | 4 | 3/4 (a 3/4) | 1/4 | 0.279–0.31 | 6.13–8.16 | 200 ×12 |
| hosts a and c | 8 | 6/8 (a 3/4, c 3/4) | 2/8 | 0.242–0.265 | 5.25–7.13 | 200 ×24 |

Every refused session got `500: Failed to place sandbox` at create and never reached the model. At `HUGEPAGES=4096` one host fits 3 `code-interpreter-v1` sandboxes (`e-2026-10-18-ci-host`), so 4 on a host leaves one refused there; the second host added its own room. The client gave the sessions to the hosts in turn. No sandbox or conversation was left.
<!-- END GENERATED: e-cm-scale -->

Stopping host a's E2B stack with `docker compose stop` (the VM kept running):

<!-- BEGIN GENERATED: e-cm-hostloss -->
| Check | Observed |
|---|---|
| host c, live session: read its file | ok, 2.77 s, "LANTERN-7" |
| host c: new session | ok, 2.52 s, "COLD-1" |
| host c: resume host a's paused sandbox | failed: `SandboxNotFoundException: Paused sandbox … not found` |
| host a, stopped: create | failed: `connection refused` |
| host a, stopped: resume its own paused sandbox | failed: `connection refused` |
| host a, after restart: resume its own paused sandbox | ok, 4.78 s, "PAUSED-a" |
| host a, after restart: the sandbox that was running when it stopped | failed: `SandboxNotFoundException: Paused sandbox … not found` |
| host a, after restart: new session | ok, 4.47 s, "COLD-2" |
| `docker compose stop` / `docker compose up -d --wait` | 77.86 s / 63.14 s |
<!-- END GENERATED: e-cm-hostloss -->

**Boundary.**
- Two independent installs and a client that spreads sessions over them, not an E2B cluster: no shared scheduler, no failover and no shared storage. A paused sandbox could only resume on the host that paused it, as with `base` in `e-2026-10-15-multi`.
- The host stop was `docker compose stop` on a running VM; a lost VM or disk was not measured.
- One run of each phase. The tool reached host c over the private network from host a, so host c's sessions include that hop and host a's do not.

### e-2026-10-18-ci-scale: a new Standard_D8ds_v7 sized for 8 code-interpreter-v1 sandboxes

**Question.** If the hugepage pool, not the CPU, limits a host, does a VM with room for 8 `code-interpreter-v1` sandboxes answer 8 sessions at once, how long does README step 7 take on a new VM, and does the template build on the first attempt?

**Input.** Host d, a new VM ([`evidence/runs/e-e2b-2026-10-18/ci-scale.json`](evidence/runs/e-e2b-2026-10-18/ci-scale.json)):

<!-- BEGIN GENERATED: e-cs-input -->
> Host d: a new `Standard_D8ds_v7` (8 vCPU, 32 GiB), zone `2`, OS disk `Standard HDD (Standard_LRS) · 128 GB`, host kernel `6.17.0-1022-azure`.
> Install: README step 7 as written, with one difference: `HUGEPAGES=8448` instead of 4096, room for 8 sandboxes of 1024 pages plus 256; Docker Engine from Docker's apt repository; `compose.yaml` and `.env` at `588e5b9` (SHA-256 OK); then the `project_limits` row, `sleep 300`, `code-interpreter` at `162a2a2` and its unchanged `build_prod.py`.
> Tool: `tools/measure_e2b.py` (SHA-256 `aca40d17e315…`), `measure_e2b.py perf --cold 5 --scale 4,8 --recovery`, then `measure_e2b.py host --bench 0 --reserve 9`, on host d itself against `localhost`; E2B Code Interpreter only (`e2b 2.56.0`, `e2b-code-interpreter 2.11.0`, template `code-interpreter-v1`, 2 vCPU, 2048 MB).
> Perf prompts as in `e-2026-10-17-ci-agent`: new sessions "Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}" (n = 1–5); one session "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only.", then "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1", then "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2"; 4 and 8 sessions at once, each "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK.", then "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."; replacement: a live session ran "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK." and, after another sandbox was killed and replaced, "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only.".
> Reservation: as in `e-2026-10-18-ci-host`, up to 9 sandboxes, each checked with `print(1 + 1)`.
<!-- END GENERATED: e-cs-input -->

**Varied and fixed.** Against `e-2026-10-17-ci-perf`, the VM size (`Standard_D4ds_v7` → `Standard_D8ds_v7`), the pool (4096 → 8448 pages), the host kernel and the tool revision (`cbc96f2…` → `aca40d1…`) changed; the template, SDKs, model, agent, prompts and clocks stayed.

**Result.** README step 7 on host d:

<!-- BEGIN GENERATED: e-cs-install -->
| Step | Start (UTC) | Seconds |
|---|---|---:|
| Docker Engine from Docker's apt repository | 09:32:14 | 42 |
| download and check the two files, `docker compose up -d --wait` | 09:32:56 | 215 |
| `project_limits` row, then `sleep 300` | 09:36:31 | 301 |
| clone `code-interpreter` at `162a2a2`, build environment | 09:41:32 | 7 |
| `build_prod.py`, 30 steps | 09:41:39 | 929 |
| **total** |  | **1494** |

`psql` printed `INSERT 0 1`; the build exited 0 on its first attempt, its last builder step was 30/30 and it reported `Build finished, took 15m26s`.
<!-- END GENERATED: e-cs-install -->

New sessions:

<!-- BEGIN GENERATED: e-cs-cold -->
| Start (UTC) | Answer | Sandbox created → Ready, s | Agent turn, s | New session → answer, s |
|---|---|---:|---:|---:|
| 10:01:46 | COLD-1 | 0.24 | 2.83 | 3.15 |
| 10:01:50 | COLD-2 | 0.068 | 2.26 | 2.38 |
| 10:01:52 | COLD-3 | 0.062 | 2.56 | 2.67 |
| 10:01:55 | COLD-4 | 0.07 | 2.26 | 2.38 |
| 10:01:57 | COLD-5 | 0.06 | 2.55 | 2.66 |

Across the 5 new sessions: sandbox create → Ready 0.068 s median (0.06–0.24; n=5), Prompt Agent turn 2.55 s median (2.26–2.83; n=5), end-to-end new session → answer 2.66 s median (2.38–3.15; n=5).
<!-- END GENERATED: e-cs-cold -->

Follow-ups in one session:

<!-- BEGIN GENERATED: e-cs-warm -->
| Turn | Answer | Sandbox tool, s | Turn, s |
|---|---|---:|---:|
| 1 (new session; the sandbox was created before it) | OK | 0.047 | 2.76 |
| 2 | LANTERN-7 | 0.013 | 2.28 |
| 3 | WARM-1 | 0.01 | 2.76 |
| 4 | WARM-2 | 0.009 | 33.32 |

The sandbox was created in 0.053 s before turn 1. Follow-ups 2–4: 2.76 s median (2.28–33.32; n=3).

Turn 4 took 33.32 s, of which the sandbox tool took 0.009 s; the rest was the model's side of the turn.
<!-- END GENERATED: e-cs-warm -->

At once and sandbox replacement:

<!-- BEGIN GENERATED: e-cs-scale -->
| At once | Answered | Refused | Sandbox create, s | Whole session, s | Model HTTP attempts |
|---:|---|---|---:|---:|---|
| 4 | 4/4 | 0/4 | 0.092–0.159 | 4.47–7.97 | 200 ×16 |
| 8 | 8/8 | 0/8 | 0.105–0.337 | 4.85–10.18 | 200 ×32 |

No session was refused. Sandbox replacement after `kill`: a new `code-interpreter-v1` sandbox was Ready in 0.063 s; follow-up on a live session HTTP 200, 3.5 s, "LANTERN-7", state preserved: true. No sandbox or conversation was left.
<!-- END GENERATED: e-cs-scale -->

Hugepages taken as sandboxes were added:

<!-- BEGIN GENERATED: e-cs-reserve -->
| Sandboxes | Create, s | `HugePages_Free` | `HugePages_Rsvd` | Pages taken | MemAvailable, MiB |
|---|---:|---:|---:|---:|---:|
| 0 (before) |  | 8175 | 751 | 1024 | 11899 |
| 1 | 0.05 | 8183 | 759 | 1024 | 11895 |
| 2 | 0.05 | 7908 | 1508 | 2048 | 11887 |
| 3 | 0.053 | 7640 | 2264 | 3072 | 11879 |
| 4 | 0.052 | 7372 | 3020 | 4096 | 11891 |
| 5 | 0.057 | 7103 | 3775 | 5120 | 11887 |
| 6 | 0.056 | 6830 | 4526 | 6144 | 11887 |
| 7 | 0.062 | 6565 | 5285 | 7168 | 11885 |
| 8 | 0.06 | 6286 | 6030 | 8192 | 11875 |
| 9: refused | 0.047 |  |  |  |  |
| 0 (all killed) |  | 8448 | 0 | 0 | 11931 |

Pages taken = `HugePages_Total` (8448) − free + reserved; pages are 2 MiB. Each sandbox took 1024 pages, its whole 2048 MiB, as soon as it was created; 0 pages were taken by other processes on host d. With 8 sandboxes, 8448 − 0 − 8 × 1024 = 256 pages were left, fewer than one sandbox needs, and create 9 got `500: Failed to place sandbox` in 0.047 s. MemAvailable stayed at 11875–11899 MiB: the pool is set aside at boot, so sandbox memory does not show there. The reading before the first create showed 1024 pages taken, and after it still 1024: the read-back sandbox, killed just before, had not yet given its pages back.

Read-back on host d: Firecracker `v1.14-0.2.0`, guest kernel `6.1.177+`, 2 vCPU, MemTotal 1982 MiB, as on host c; host kernel `6.17.0-1022-azure`.
<!-- END GENERATED: e-cs-reserve -->

**Boundary.**
- One run of each phase on one host; the tool ran on host d itself.
- 8 at once answered 8/8 here and 3/8 on the 4096-page `Standard_D4ds_v7` of `e-2026-10-17-ci-perf`. The hosts also differ in size, kernel and day, so this shows the larger pool admitting the sandboxes, not a matched speed comparison.
- The build passed on its one attempt after the README's 5-minute wait; one build does not show that it always will.
- As in `e-2026-10-17-ci-perf`, the follow-up after `kill` went to a different live session; it does not show continuity of the killed sandbox.

### Numbers you can recompute

Every published metric above comes from [`evidence/measurements.json`](evidence/measurements.json), which [`tools/build_measurements.py`](tools/build_measurements.py) recomputes from `evidence/runs/` after checking each file's SHA-256 against [`evidence/runs/INDEX.json`](evidence/runs/INDEX.json). It also checks: A's cold equals create plus first answer and recover equals resume plus answer in every run; each D cold pod's create-to-Ready equals its Kubernetes timestamps and runs `kata-vm-isolation`; D's warm session and the different live session used after pod deletion both remembered `LANTERN-7`. For the 2026-10-18 runs: C's cold equals apply → Ready plus the first answer and its resume-to-answer equals Ready plus the answer in every run; B's and D's session runs must keep their file and question count; every at-once session's remembered flag must match its answer, and A and C must leave nothing behind. For E's 2026-10-18 runs: each of the 20 cycles must read its variable back as 42; the pages taken must grow by the same amount per sandbox, equal to the template's memory, and the sandboxes that fit must equal the pool minus the other pages divided by that amount; the install's parts must add up to its total. Medians are reported only for odd samples; even samples show their values.

Duration fields and published samples must be finite, non-negative numbers, not booleans or numeric strings. An optional timing recorded as null remains unmeasured and cannot enter a published sample. C's resume milestones must satisfy Ready → LoadBalancer healthy → first answer; a paired D pod-ready duration cannot exceed its request wall time beyond timestamp rounding, and concurrency totals require every session record. JSON output rejects NaN and infinity.

These are unit, type, sample-count and timeline checks, not a universal maximum-latency threshold. Small samples, different clock locations and unmatched workloads still prevent a product ranking; values that look plausible are not a substitute for the retained run evidence.

## Tests and Offline Checks

**Deployment evidence boundary.** These are optional checks after deployment, not commands that replace creation of the application. The full parameterized README sequence has not been re-run in a fresh subscription. Original historical runs do not prove every modified deployment command. Module 1 foundation took <!--v:f.deploy.value-->3<!--/v--> minutes in the retained run.

**Live measurement.** Return to `HOSTING_REPO`, load `.env`, and select the deployed B or D endpoint in `AGENT_RESPONSES_URL`; for E, export `E2B_API_KEY` as in README step 7. Run only the measurement for that path. The D recovery command deletes a session pod; the A measurement stops and deletes the sandboxes it creates; the E measurements create and delete their own agent version, conversations and sandboxes, and `perf` starts 8 sessions at once; `measure_c.py` creates and deletes its own Sandboxes next to the workshop's, and 8 at once can make the cluster autoscaler add a Kata node. Obtain explicit approval for those lifecycle operations before running them.

```bash
cd "$HOSTING_REPO"
set -a && . ./.env && set +a
python tools/measure_aca.py agent --runs 3
python tools/measure_aca.py agent --runs 3 --config novolume
python tools/measure_aca.py scale --scale 4,8
python tools/measure_responses.py --cold 3 --warm-turns 3
python tools/measure_responses.py --cold 3 --warm-turns 3 --scale 4,8 --recovery --observe
python tools/measure_e2b.py agent --runs 3
python tools/measure_e2b.py perf --cold 5 --scale 4,8 --recovery
python tools/measure_e2b.py host --bench 20 --reserve 4   # run this one on the E2B host itself
python tools/measure_c.py --cold 3 --pvc 3 --scale 4,8
```

Exit 0 requires every requested phase to pass; full-pool refusals are saved but still exit 1. The different historical workloads and clocks do not make this a product ranking.

**Optional isolation observation.** Use the correct namespace for C or D. The base check reads configured runtime and pod state; for D add `--container agent --exec-uname` only when exec is authorized. Requested read failure exits 1.

```bash
python tools/check_isolation.py -n "$NAMESPACE" -l app=agent-host --exec-uname
python tools/check_isolation.py -n "$SESSION_NAMESPACE"
```

**Optional C lifecycle exercise.** These commands intentionally suspend the application. Run only after the owner approves that operation; they are not required to deploy or answer.

```bash
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Suspended"}}'
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Running"}}'
kubectl wait sandbox agent-host -n "$NAMESPACE" --for=condition=Ready --timeout=180s
```


The offline checks need Python 3.10 or newer and no Azure access:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/build_measurements.py --check
python tools/render_docs.py --check
python tools/build_diagrams.py --check
python tools/validate_docs.py
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate` instead of `. .venv/bin/activate`. Done when every command exits 0 and each check prints `PASS`.

- **`build_measurements.py --check`** rebuilds every metric and fails if a run file's hash, a field or a cross-check is wrong, or if the committed `measurements.json` differs.
- **`render_docs.py --check`** fails if a table or inline number in any of the four pages differs from a fresh render.
- **`build_diagrams.py --check`** compares both architecture figures and every icon with [`images/SOURCES.json`](images/SOURCES.json).
- **`validate_docs.py`** checks the three README headings in order, the VALIDATION headings, the core claim contract and canonical run order in [`evidence/claims.json`](evidence/claims.json), that every measured-run section has Question, Input, Varied and fixed, Result and Boundary in that order with no table before the input, that every run is in the README's results table and all four pages use the same run order, links and images, figure widths, English/Chinese number parity by generated block and table cell (not just document-wide totals), Steps commands that use only variables, keep the reader's shell open (no `set -e` or `exit`), install into a virtual environment and keep `az group delete` alone with its confirmation prompt, and private-content guards (subscription and tenant IDs, IPv4 addresses, emails, internal resource names, and retired claims).
- **`pytest`** runs these checks as tests, recomputes the medians from raw values, evaluates `check_isolation.py` on saved pod and node JSON, runs every tool's `--help` without the Azure SDK, and breaks each guard (a tampered run file, invalid duration, reversed timeline, incomplete concurrency sample, edited number, inserted subscription ID, removed heading, table before an input, reordered runs) to confirm it fails.
- Mocked live-tool tests also check that failed concurrency follow-ups, skipped recovery, failed pod reads or deletions, lost history, invalid arguments and missing settings cannot report success. They make no Azure calls.

CI runs the same on Ubuntu with Python 3.12 ([workflow](https://github.com/david-xinyuwei/david-share/blob/master/.github/workflows/agent-sandbox-hosting-ci.yml)).

The live checks need Azure and the resources from the README: `tools/aca_sandbox.py`, `tools/measure_aca.py`, `tools/measure_responses.py`, `tools/measure_sessions.py`, `tools/measure_c.py`, `tools/measure_e2b.py`, `tools/e2b_agent.py` and `tools/check_isolation.py` without `--pods-json`.

Not tested here: the tools against a fresh subscription end to end after this repository was assembled (they are the run scripts parameterised with settings; `--help` and the offline logic are tested), regions other than East US 2, and Azure Cloud Shell as the operator shell.

## Limits, Assets and Sources

**Limits.**

- `LOCAL_MEASUREMENT`: A full-agent latency has n=3 with the Blob volume and n=3 without it, plus one run each of 4 and 8 sandboxes at once. B is one end-to-end run, three session-API runs, a D-protocol campaign (cold n=5, warm n=3, 4 and 8 at once) and three runs after each of two idle waits. C is one end-to-end run plus three runs without and three with the Blob PVC (pod loss and suspend/resume included) and one run each of 4 and 8 at once. D is one benchmark campaign: cached-image cold n=5, warm n=3, and one placement and one recovery observation, plus three session-API runs. Repeat with the live measurement commands in this page before relying on a number.
- `LOCAL_MEASUREMENT`: A and C use the workshop reflection agent; B uses its Agent Framework agent with ResponsesHostServer; D uses the private-preview sample. B's 2026-09-27 run and D include the Foundry endpoint and a client in China (round trip <!--v:d.client_rtt.range-->320–800<!--/v--> ms); B's 2026-10-15 and 2026-10-18 runs add a client in Japan East and an API Management gateway; D's 2026-10-18 runs called the endpoint directly from Japan East; A's answers were timed inside the sandbox; C's at its LoadBalancer on 2026-09-27 and inside the pod on 2026-10-18. The rows compare field implementations, not identical containers or network paths.
- `LOCAL_MEASUREMENT`: in A, the private Blob volume put create at <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s; image, VNet and port alone stayed at <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s.
- `LOCAL_MEASUREMENT`: the oldest D session pod was still Running at least <!--v:d.idle.value-->87<!--/v--> minutes after it was created. This is pod age, not idle duration: no last-request timestamp was retained. It establishes neither a configured idle timeout nor a continuously idle interval.
- `LOCAL_MEASUREMENT`: D's capacity was at most <!--v:d.capacity.per_node-->5<!--/v--> sessions per `Standard_D4s_v5` node with the sample's 500m CPU / 1 Gi request plus 600 Mi Kata overhead; a node that also hosts system pods fits fewer.
- `NOT_MEASURED`: when B's platform reclaims an idle session (`b-2026-10-18-idle` read the status once, after the wait) and B's session loss; A's host loss; D's behaviour once the service ends a session; the Foundry-governed agent calling an ACA sandbox per task; D's latency under concurrency with free capacity; third-party sandboxes other than E2B; E2B Bring Your Own Cloud and private cloud; the official E2B Azure Terraform module, including its scale-set replacement of an unhealthy instance from a fresh disk; the loss of a whole E2B VM or its disk.
- `SOURCE_FACT`: E2B Embed "is one machine, by design"; E2B's ways past one machine are private cloud (in development), Bring Your Own Cloud operated by E2B in your account, and E2B Cloud ([Embed](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/README.md#when-one-node-is-not-enough)). It answers over plain HTTP with no TLS of its own. The Azure Terraform install is a scale set of one with no persistent data volume: a replacement starts from a fresh disk and loses the templates you built, and stopping the stack gets the instance replaced about two minutes after `/health` stops answering ([Azure guide](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/terraform/azure/README.md#limitations)).
- `SOURCE_FACT`: E2B's Compose guide sizes `HUGEPAGES` by "the sandboxes you will run at once" and states 256 pages for a 512 MiB `base` sandbox at full memory and about 90 idle ([requirements](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/compose/README.md#requirements)). In `e-2026-10-18-ci-host` and `e-2026-10-18-ci-scale` each `code-interpreter-v1` sandbox took 1024 pages, its whole 2 GiB, at create, idle or not, so 4096 pages held 3 and 8448 held 8; `e-2026-10-15-multi` measured the same for `base`: 256 pages each, 7 in the default pool, not about 20.
- `SOURCE_FACT`: ACA Sandboxes start "from prewarmed pools for near-instant availability" and resume "with sub-second restore times" ([overview](https://learn.microsoft.com/azure/container-apps/sandboxes-overview), resource tiers rechecked 2026-10-23). Selectable tiers (vCPU / GiB): XS 0.25 / 0.5, S 0.5 / 1, M 1 / 2, L 2 / 4, XL 4 / 8, 2XL 8 / 16, 3XL 16 / 32. Their disks are 20 / 20 / 20 / 40 / 80 / 160 / 320 GiB in that order. These are documented allocations, not seven tested configurations. Idle auto-suspend is configurable; Memory preserves memory and disk, while Disk preserves disk only. The local 900 s setting is not a service-wide default.
- `SOURCE_FACT`: hosted agents run in "per-session VM-isolated sandboxes" with `$HOME` and `/files` persisted; idle timeout 2–60 minutes, default 15; sessions deleted after 30 days of inactivity ([hosted agents](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents)). Stopping a session with `POST .../sessions/{id}:stop` ends its compute and keeps the persistent filesystem ([manage sessions](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions#stop-a-session)).
- `SOURCE_FACT`: hosted agent [sandbox sizes](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents#sandbox-sizes) are 0.5 vCPU / 1 GiB, 1 vCPU / 2 GiB and 2 vCPU / 4 GiB; the values set on an agent version apply to each session, and no VM SKU is chosen. The workshop's module-02 uses 0.5 vCPU / 1 GiB.
- `SOURCE_FACT`: AKS Pod Sandboxing is Azure Linux host + Microsoft Hyper-V + Cloud Hypervisor + Kata Containers; the node VM must be a generation 2 size that supports nested virtualization; each pod VM's memory follows the pod memory limit (512 Mi if unset) and fractional CPU limits round up to a whole vCPU ([pod sandboxing](https://learn.microsoft.com/azure/aks/use-pod-sandboxing)).
- `SOURCE_FACT`: agent-sandbox v0.5.2 supports an absolute `shutdownTime`; underlying pods and services are deleted on expiry, and `shutdownPolicy` controls retention of the Sandbox object ([official API](https://github.com/kubernetes-sigs/agent-sandbox/blob/v0.5.2/docs/api.md)). This is scheduled expiry, not a request-idle timer. `operatingMode` still exposes explicit suspend/resume.
- `NOT_VERIFIED`: no public official BYO-AKS idle policy or exhaustive node-SKU list was verified in this review; explicit `:stop` and resume through the session API worked in `d-2026-10-18-sessions`, an observation of this preview build, not a documented policy. Confirm preview requirements with the product group; the pod-age observation cannot establish those policies.
- `SOURCE_FACT`: [ACA's official quota documentation](https://sandboxes.azure.com/docs/sandboxes/limits) measures concurrent active cores; stopped sandboxes do not consume this quota. This is not a fixed session-count limit.
- `SOURCE_FACT`: [AKS Pod Sandboxing](https://learn.microsoft.com/azure/aks/concepts-pod-sandboxing) isolates compute, not the control plane, storage/data paths or cluster-admin actions. The isolation tool checks runtime configuration; requested kernel reads must succeed, and a version string is not hardware attestation.
- `SOURCE_FACT`: the [fixed module-02 implementation](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-02/agent-src/main.py) makes `MODEL_ROUTING=direct` call the Foundry project endpoint. A project with a registered native AI Gateway may still forward those calls through APIM; the mode name alone does not prove a gateway bypass.
- `SOURCE_FACT`: status sources: A [GA on 2026-09-23](https://techcommunity.microsoft.com/blog/appsonazureblog/azure-container-apps-sandboxes-now-generally-available/4559125); B [GA in 2026-06](https://azure.microsoft.com/updates?id=563596) (Azure Update "Generally Available: Hosted agents in Foundry Agent Service"; the earlier [Build post of 2026-06-02](https://devblogs.microsoft.com/foundry/hosted-agents-build26/) described the public preview before GA); C [Pod Sandboxing GA on 2025-11-18](https://azure.microsoft.com/updates?id=526237), with agent-sandbox v0.5.2 as an open-source project.
- `LOCAL_MEASUREMENT`: A's history length continued increasing in the saved run. This checks history continuity, not exact answer correctness or preservation of arbitrary process memory.
- `SOURCE_FACT`: D's cost floor of <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h is two `Standard_D4s_v5` nodes at 0.192 USD/h each plus the AKS Standard tier at 0.10 USD/h, East US 2 retail prices queried 2026-09-21; E's single-VM cost floor of <!--v:e.cost_floor.value-->0.344<!--/v--> USD/h is one `Standard_D4ds_v7` Linux VM at 0.326 USD/h, one Standard static public IP at 0.005 USD/h and one 50 GB Premium SSD P6 disk at 0.013 USD/h, East US 2 retail prices queried 2026-10-14; a `Standard_D8ds_v7` Linux VM, the size of `e-2026-10-18-ci-scale`, is 0.653 USD/h (queried 2026-10-18); no discounts or tax. This is not the cost of the three-Kata-node benchmark pool.
- D's onboarding durations (role assignments about <!--v:d.role_assignments.value-->120<!--/v--> s, `azd provision` <!--v:d.azd_provision.value-->522<!--/v--> s, `azd deploy` <!--v:d.azd_deploy.value-->212<!--/v--> s) come from the author's console recorded in the earlier field record; the console itself was not kept.

**Assets.**

- [`tools/`](tools/): `aca_sandbox.py`, `measure_aca.py`, `measure_responses.py`, `measure_e2b.py`, `measure_code_interpreter.py`, `measure_sessions.py`, `measure_c.py`, `e2b_agent.py`, `check_isolation.py` (live); `export_evidence.py` (author only), `build_measurements.py`, `render_docs.py`, `build_diagrams.py`, `validate_docs.py` (offline).
- [`evidence/claims.json`](evidence/claims.json): the canonical A–E identities, statuses, ownership, boundaries, letter mapping and run order; [`evidence/runs/`](evidence/runs/): the redacted run records with the SHA-256 of their private sources, and `INDEX.json`; [`evidence/measurements.json`](evidence/measurements.json): every published metric.
- [`images/`](images/): the architecture figures, their source of truth `tools/build_diagrams.py`, the 8 Azure icons and [`SOURCES.json`](images/SOURCES.json).
- [`config/.env.example`](config/.env.example): every setting; [`tests/`](tests/): the offline tests and their fixtures.

**Sources.**

- Microsoft Learn: [Container Apps Sandboxes overview](https://learn.microsoft.com/azure/container-apps/sandboxes-overview) · [Hosted agents in Foundry Agent Service](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents) · [Pod sandboxing with AKS](https://learn.microsoft.com/azure/aks/use-pod-sandboxing) · [Azure Architecture Icons](https://learn.microsoft.com/azure/architecture/icons/)
- [agent-sandbox v0.5.2](https://github.com/kubernetes-sigs/agent-sandbox/releases/tag/v0.5.2) · [agenthost workshop at 905f2c2](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost) (no license file; referenced, not copied)
- [`azure-containerapps-sandbox` on PyPI](https://pypi.org/project/azure-containerapps-sandbox/) (0.1.0b4)
- E2B: [Embed at 588e5b9](https://github.com/e2b-dev/runtime/tree/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed) · [code-interpreter template at 162a2a2](https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template)
- BYO-AKS private-preview sample and onboarding guide: product group, access on request.
