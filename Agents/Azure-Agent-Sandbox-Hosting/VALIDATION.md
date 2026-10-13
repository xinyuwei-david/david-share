# Validation Details for Reviewers

[中文](VALIDATION_CN.md) · Customer steps are in [README.md](README.md)

This page is for readers who want to check the numbers: how each path was measured, the input and result of every run, what was not tested, and how to recompute everything offline. A customer following the steps does not need it.

## Architecture and Metrics

The narrative is organized around two questions: who runs the session lifecycle, and whose node the session's micro-VM runs on. The workload contract is comparable, but the container is not identical across all four paths: A, B and C use the agenthost workshop's reflection agent ([`oxcp/ainotes` at `905f2c2`](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost)); D uses the private-preview sample `agent-framework-agent-basic-responses-byoaks`. All call `gpt-5.4-mini` and exercise a new stateful session plus a follow-up.

- **A · ACA Sandboxes.** The agent's container image becomes a sandbox disk image; one sandbox per measured session, `cpu 1000m`, `memory 2048Mi` (the M tier), auto-suspend after 900 s in Memory mode, a VNet connection, a private Blob volume at `/app/app/data` and port 8080 in on-demand mode. The control plane is ARM (`Microsoft.App/sandboxGroups`), the data plane `management.azuredevcompute.io`; the caller needs *Container Apps SandboxGroup Data Owner*.
- **B · Foundry hosted agent.** `azd ai agent init` binds the workshop's `azure.yaml` to the existing project and `azd deploy` builds and publishes a version. Learn describes per-session VM-isolated sandboxes, `$HOME` and `/files` persisted across idle, an idle timeout of 2–60 minutes (default 15) and deletion after 30 days of inactivity.
- **C · AKS + agent-sandbox.** AKS with Workload Identity and the Blob CSI driver, an Azure Linux node pool with `KataVmIsolation` (`Standard_D4s_v3` in our run), agent-sandbox `v0.5.2` (`agents.x-k8s.io/v1beta1`, kind `Sandbox`), a Blob PVC for state and a LoadBalancer restricted to one source address. Suspend and resume are `spec.operatingMode: Suspended | Running`; the pod is deleted and file state is restored from Blob, not from a memory snapshot. Learn names the stack: Azure Linux host, Microsoft Hyper-V, Cloud Hypervisor VMM, Kata Containers.
- **D · BYO-AKS.** The initial one-Kata-node read-back had 13 runtime pods in `foundry-system`: 9 fixed Deployment replicas (router ×3 behind a LoadBalancer on `:443`, controller ×2, identity proxy ×2, telemetry controller ×2) plus 4 DaemonSets. Each DaemonSet adds one pod per matching Kata node, so the total grows with the node pool. Each session is its own Deployment, ReplicaSet and pod with `runtimeClassName: kata-vm-isolation` in a per-project namespace.

Why these clocks:

| Metric | Start | End | Measured where |
|---|---|---|---|
| Cold, A (component sum) | SDK create call → Running | add first `/reflect` answer duration | create on the client; answer inside the sandbox |
| Cold, B and D | client POST of a new session | full response body read | client, through the Foundry endpoint |
| Cold, C (component sum) | `kubectl apply` → Sandbox Ready | add first answer duration at the LoadBalancer | client, no gateway |
| Warm | request of the next turn | its answer | as for cold |
| Resume, A (component sum) | SDK resume call → Running | add first answer duration after resume | resume on the client; answer inside |
| Resume, C | `operatingMode: Running` patch | first answer at the LoadBalancer | client |
| Pod lost | pod deleted | replacement pod Ready (Kubernetes timestamps) | cluster |

A's answers are timed inside the sandbox because the 2026-09-27 run could not reach the port from the operator address (the port's IP rule saw rotating Azure egress addresses). The paths therefore do not share a network path, and a row compares implementations, not networks.

A's cold is `create_total_s + first.guest_s` and resume-to-answer is `resume_total_s + first_after_resume.guest_s`: component sums, not a single end-to-end client stopwatch. C's cold is also a component sum. For both A and C, stop and idle waiting are excluded from resume-to-answer.

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

[`tools/measure_aca.py`](tools/measure_aca.py) repeats the 2026-10-09 protocol: `agent` mode runs create → answer → answer → stop → resume → answer → answer → delete per run and fails if conversation-history lengths do not keep growing; `create` mode times each configuration (`builtin`, `image`, `vnet`, `port`, `vnetvolume`, `full`) and records whether guest uptime after resume exceeds its pre-stop value. Uptime alone does not prove process-memory continuity; the retained `process_survived` field is a legacy name for that observation.

[`tools/measure_responses.py`](tools/measure_responses.py) (B and D) POSTs `{"input": …}` to `AGENT_RESPONSES_URL` with an Azure CLI token for `https://ai.azure.com` and adds `previous_response_id` for follow-ups. Token acquisition is outside the request timer; it refreshes before each concurrency level and recovery. With `--observe`, it accepts only one new Ready Kata pod as a candidate; ambiguity or a failed pod read is a failure, not a placement result. Run this observation in a quiet project namespace. `--recovery` deletes the one newly observed Kata pod, waits for deletion to finish, then follows up using that session's response ID. Exit 0 requires every requested cold, warm, concurrency and recovery check to pass; skipped recovery, deletion failure and forgotten state return exit 1.

[`tools/check_isolation.py`](tools/check_isolation.py) (C and D) reads `kubectl get pods` and `kubectl get nodes`, requires `runtimeClassName: kata-vm-isolation` and phase Running, and with `--exec-uname` compares `uname -r` inside the pod with the node's `kernelVersion`.

The workshop scripts (`module-01/setup.sh`, `module-03/prepare-agent-sandbox.sh`, `module-04/sandbox-deploy.sh`) are used unchanged at `905f2c2`; this repository does not copy them.

## Measured Runs

All runs were in East US 2 from a Windows workstation. The A/B/C record names Azure CLI 2.88.0 and azd 1.34.1; A's SDK record also names Python 3.13. Raw consoles and IDs stay in the author's private folders; [`tools/export_evidence.py`](tools/export_evidence.py) projected them into [`evidence/runs/`](evidence/runs/), recording the SHA-256 of every private source, and replaced node, pod and namespace names with neutral labels.

Runs are grouped A–D, then by date from oldest to newest within each path, matching the README tables.

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
- One run. The 24 s wake time was a single observation and is not used for the resume row; the 2026-10-09 explicit resume (three runs) is.
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

**Varied and fixed.** Only the run number varied. The service (GA since 2026-09-23), SDK 0.1.0b4 (the latest public release on that day, still beta), the sandbox group, the image and the specification were the same in all three.

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
- Resume after idle and session loss were not exercised.

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
- One run.
- The runc comparison pod ran on the Ubuntu system pool, so the two kernels differ by node as well as by runtime. The Kata kernel's `mshv` suffix, the runtime class and Learn's documented stack carry the conclusion.
- Wake on request is not part of agent-sandbox; the resume was a manual patch.

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

### Numbers you can recompute

Every published metric above comes from [`evidence/measurements.json`](evidence/measurements.json), which [`tools/build_measurements.py`](tools/build_measurements.py) recomputes from `evidence/runs/` after checking each file's SHA-256 against [`evidence/runs/INDEX.json`](evidence/runs/INDEX.json). It also checks: A's cold equals create plus first answer and recover equals resume plus answer in every run; each D cold pod's create-to-Ready equals its Kubernetes timestamps and runs `kata-vm-isolation`; D's warm session and the different live session used after pod deletion both remembered `LANTERN-7`. Medians are reported only for odd samples; even samples show their values.

Duration fields and published samples must be finite, non-negative numbers, not booleans or numeric strings. An optional timing recorded as null remains unmeasured and cannot enter a published sample. C's resume milestones must satisfy Ready → LoadBalancer healthy → first answer; a paired D pod-ready duration cannot exceed its request wall time beyond timestamp rounding, and concurrency totals require every session record. JSON output rejects NaN and infinity.

These are unit, type, sample-count and timeline checks, not a universal maximum-latency threshold. Small samples, different clock locations and unmatched workloads still prevent a product ranking; values that look plausible are not a substitute for the retained run evidence.

## Tests and Offline Checks

The offline checks need Python 3.10 or newer and no Azure access:

```bash
python -m pip install -r requirements-dev.txt
python tools/build_measurements.py --check
python tools/render_docs.py --check
python tools/build_diagrams.py --check
python tools/validate_docs.py
python -m pytest -q
```

Done when every command exits 0 and each check prints `PASS`.

- **`build_measurements.py --check`** rebuilds every metric and fails if a run file's hash, a field or a cross-check is wrong, or if the committed `measurements.json` differs.
- **`render_docs.py --check`** fails if a table or inline number in any of the four pages differs from a fresh render.
- **`build_diagrams.py --check`** compares both architecture figures and every icon with [`images/SOURCES.json`](images/SOURCES.json).
- **`validate_docs.py`** checks the three README headings in order, the VALIDATION headings, the core claim contract and canonical run order in [`evidence/claims.json`](evidence/claims.json), that every measured-run section has Question, Input, Varied and fixed, Result and Boundary in that order with no table before the input, that every run is in the README's results table and all four pages use the same run order, links and images, figure widths, English/Chinese number parity by generated block and table cell (not just document-wide totals), variable-only commands in Steps, and private-content guards (subscription and tenant IDs, IPv4 addresses, emails, internal resource names, and retired claims).
- **`pytest`** runs these checks as tests, recomputes the medians from raw values, evaluates `check_isolation.py` on saved pod and node JSON, runs every tool's `--help` without the Azure SDK, and breaks each guard (a tampered run file, invalid duration, reversed timeline, incomplete concurrency sample, edited number, inserted subscription ID, removed heading, table before an input, reordered runs) to confirm it fails.
- Mocked live-tool tests also check that failed concurrency follow-ups, skipped recovery, failed pod reads or deletions, lost history, invalid arguments and missing settings cannot report success. They make no Azure calls.

CI runs the same on Ubuntu and Windows with Python 3.12 ([workflow](https://github.com/david-xinyuwei/david-share/blob/master/.github/workflows/agent-sandbox-hosting-ci.yml)).

The live checks need Azure and the resources from the README: `tools/aca_sandbox.py`, `tools/measure_aca.py`, `tools/measure_responses.py` and `tools/check_isolation.py` without `--pods-json`.

Not tested here: the tools against a fresh subscription end to end after this repository was assembled (they are the run scripts parameterised with settings; `--help` and the offline logic are tested), regions other than East US 2, and Azure Cloud Shell as the operator shell.

## Limits, Assets and Sources

**Limits.**

- `LOCAL_MEASUREMENT`: A full-agent latency has n=3. B and C are one end-to-end run each. D is one benchmark campaign: cached-image cold n=5, warm n=3, and one placement and one recovery observation. Repeat with README step 7 before relying on a number.
- `LOCAL_MEASUREMENT`: A, B and C use the workshop reflection agent; D uses the private-preview sample. B and D include the Foundry endpoint and a client in China (round trip <!--v:d.client_rtt.range-->320–800<!--/v--> ms); A's answers were timed inside the sandbox; C's at its LoadBalancer. The rows compare field implementations, not identical containers or network paths.
- `LOCAL_MEASUREMENT`: in A, the private Blob volume put create at <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s; image, VNet and port alone stayed at <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s.
- `LOCAL_MEASUREMENT`: the oldest D session pod was still Running at least <!--v:d.idle.value-->87<!--/v--> minutes after it was created. This is pod age, not idle duration: no last-request timestamp was retained. It establishes neither a configured idle timeout nor a continuously idle interval.
- `LOCAL_MEASUREMENT`: D's capacity was at most <!--v:d.capacity.per_node-->5<!--/v--> sessions per `Standard_D4s_v5` node with the sample's 500m CPU / 1 Gi request plus 600 Mi Kata overhead; a node that also hosts system pods fits fewer.
- `NOT_MEASURED`: B's resume and session loss; A's host loss; D's behaviour once the service ends a session; the Foundry-governed agent calling an ACA sandbox per task; latency under concurrency with free capacity; E2B or any other third-party sandbox.
- `SOURCE_FACT`: ACA Sandboxes start "from prewarmed pools for near-instant availability" and resume "with sub-second restore times" ([overview](https://learn.microsoft.com/azure/container-apps/sandboxes-overview)). Selectable tiers (vCPU / GB): XS 0.25 / 0.5, S 0.5 / 1, M 1 / 2, L 2 / 4, XL 4 / 8. Idle auto-suspend is configurable; Memory preserves memory and disk, while Disk preserves disk only. The local 900 s setting is not a service-wide default.
- `SOURCE_FACT`: hosted agents run in "per-session VM-isolated sandboxes" with `$HOME` and `/files` persisted; idle timeout 2–60 minutes, default 15; sessions deleted after 30 days of inactivity ([hosted agents](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents)).
- `SOURCE_FACT`: hosted agents let you configure CPU and memory. The [deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent) shows `cpu="1"` and `memory="2Gi"` as an example, not an exhaustive selectable list or a maximum. The reviewed official pages do not establish the previously listed 2 vCPU / 4 GiB cap, so the selectable row does not publish that cap as verified.
- `SOURCE_FACT`: AKS Pod Sandboxing is Azure Linux host + Microsoft Hyper-V + Cloud Hypervisor + Kata Containers; the node VM must be a generation 2 size that supports nested virtualization; each pod VM's memory follows the pod memory limit (512 Mi if unset) and fractional CPU limits round up to a whole vCPU ([pod sandboxing](https://learn.microsoft.com/azure/aks/use-pod-sandboxing)).
- `SOURCE_FACT`: agent-sandbox v0.5.2 supports an absolute `shutdownTime`; underlying pods and services are deleted on expiry, and `shutdownPolicy` controls retention of the Sandbox object ([official API](https://github.com/kubernetes-sigs/agent-sandbox/blob/v0.5.2/docs/api.md)). This is scheduled expiry, not a request-idle timer. `operatingMode` still exposes explicit suspend/resume.
- `NOT_VERIFIED`: no public official BYO-AKS idle policy, suspend support or exhaustive node-SKU list was verified in this review. Confirm preview requirements with the product group; the pod-age observation cannot establish those policies.
- `SOURCE_FACT`: D's cost floor of <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h is two `Standard_D4s_v5` nodes at 0.192 USD/h each plus the AKS Standard tier at 0.10 USD/h, East US 2 retail prices queried 2026-09-21; no discounts or tax. This is not the cost of the three-Kata-node benchmark pool.
- D's onboarding durations (role assignments about <!--v:d.role_assignments.value-->120<!--/v--> s, `azd provision` <!--v:d.azd_provision.value-->522<!--/v--> s, `azd deploy` <!--v:d.azd_deploy.value-->212<!--/v--> s) come from the author's console recorded in the earlier field record; the console itself was not kept.

**Assets.**

- [`tools/`](tools/): `aca_sandbox.py`, `measure_aca.py`, `measure_responses.py`, `check_isolation.py` (live); `export_evidence.py` (author only), `build_measurements.py`, `render_docs.py`, `build_diagrams.py`, `validate_docs.py` (offline).
- [`evidence/claims.json`](evidence/claims.json): the canonical A–D identities, statuses, ownership, boundaries, selection contract, letter mapping and run order; [`evidence/runs/`](evidence/runs/): the redacted run records with the SHA-256 of their private sources, and `INDEX.json`; [`evidence/measurements.json`](evidence/measurements.json): every published metric.
- [`images/`](images/): the architecture figures, their source of truth `tools/build_diagrams.py`, the 8 Azure icons and [`SOURCES.json`](images/SOURCES.json).
- [`config/.env.example`](config/.env.example): every setting; [`tests/`](tests/): the offline tests and their fixtures.

**Sources.**

- Microsoft Learn: [Container Apps Sandboxes overview](https://learn.microsoft.com/azure/container-apps/sandboxes-overview) · [Hosted agents in Foundry Agent Service](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents) · [Pod sandboxing with AKS](https://learn.microsoft.com/azure/aks/use-pod-sandboxing) · [Azure Architecture Icons](https://learn.microsoft.com/azure/architecture/icons/)
- [agent-sandbox v0.5.2](https://github.com/kubernetes-sigs/agent-sandbox/releases/tag/v0.5.2) · [agenthost workshop at 905f2c2](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost) (no license file; referenced, not copied)
- [`azure-containerapps-sandbox` on PyPI](https://pypi.org/project/azure-containerapps-sandbox/) (0.1.0b4)
- BYO-AKS private-preview sample and onboarding guide: product group, access on request.
