# Four Ways to Host an Agent Sandbox on Azure: ACA Sandboxes, Foundry Hosted Agent, AKS + agent-sandbox, BYO-AKS

[![Paths](https://img.shields.io/badge/paths-A%20%C2%B7%20B%20%C2%B7%20C%20%C2%B7%20D-0078D4)](#purpose)
[![Measured](https://img.shields.io/badge/measured-East%20US%202%2C%202026--09--22%20to%2010--09-0078D4)](#results)
[![Docs CI](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml/badge.svg)](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml)

Author: Xinyu Wei · [中文](README_CN.md)

## Purpose

You want each agent session to run in its own micro-VM on Azure, and you have to pick who runs that micro-VM and its lifecycle: a service, Foundry, or you on your own AKS nodes. The four paths make different trade-offs, and their seconds are hard to compare from product pages alone. This repository applies one comparable stateful interaction contract to all four paths, gives the steps to reproduce each one, and shows what we measured. A and C share the workshop reflection agent; B uses its distinct Agent Framework agent with ResponsesHostServer; D uses the product group's private-preview sample, so this is not an identical-container benchmark:

| Question | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| Status | GA (2026-09-23) | Public preview | Pod Sandboxing GA (2025-11-18); agent-sandbox v0.5.2 is open source | **Private preview** |
| Who runs the session lifecycle | The ACA service | Foundry | You: controller, gateway, idle sweeper | Foundry, on your cluster |
| Where the session's micro-VM runs | Service-managed compute | Microsoft-managed compute | Your Kata node pool | Your Kata node pool |
| Suspend with memory snapshot | Yes | No: idle releases compute, `$HOME` and `/files` are restored | No memory snapshot: you patch `operatingMode`; the pod goes, file state comes back from Blob | Not exercised; confirm preview suspend support with the product group |
| Selectable size / node SKU (docs) | XS: 0.25 vCPU / 0.5 GB<br>S: 0.5 vCPU / 1 GB<br>M (default): 1 vCPU / 2 GB<br>L: 2 vCPU / 4 GB<br>XL: 4 vCPU / 8 GB | 0.5 vCPU / 1 GiB<br>1 vCPU / 2 GiB<br>2 vCPU / 4 GiB<br>No VM SKU to choose | Kata-compatible Gen2 VM SKUs, such as the Dsv3 series; Azure Linux and nested virtualization required | Kata node SKUs within the approved preview requirements; confirm the selectable list with the product group |
| Choose it when | You want per-user micro-VMs, snapshots and no cluster | You want Foundry to run everything and its limits do not bite | You must own the node, the network and the control plane | You need Foundry's agent platform **and** sessions on nodes you own |

Product capabilities come from Microsoft Learn and status from the official announcements linked below; measured claims come from our runs; the last row is our selection guidance. Start with B for a general managed enterprise agent when its limits fit and a public preview is acceptable. For a Qira-like long-running, per-user workload that needs memory snapshots without a cluster, start with A. Use C or D only when owning the execution nodes is non-negotiable; choose D when Foundry must still run the agent platform.

The selectable row separates [A's sandbox resource tiers](https://learn.microsoft.com/azure/container-apps/sandboxes-overview#resource-tiers), [B's sandbox sizes](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents#sandbox-sizes), and [C/D's node VM SKUs](https://learn.microsoft.com/azure/aks/use-pod-sandboxing). A node SKU is not a per-session allocation; Kata pod requests, limits and overhead also constrain each session. Node availability and quota vary by region and subscription, and D has additional preview requirements.

Status sources: [A GA announcement](https://techcommunity.microsoft.com/blog/appsonazureblog/azure-container-apps-sandboxes-now-generally-available/4559125), [B public preview](https://devblogs.microsoft.com/foundry/hosted-agents-build26/), [C Pod Sandboxing GA](https://azure.microsoft.com/updates?id=526237); agent-sandbox itself is an open-source project, not an Azure service. For C/D, [Kata provides compute isolation only](https://learn.microsoft.com/azure/aks/concepts-pod-sandboxing), not isolation of the control plane, storage/data paths or cluster-admin actions.

<img src="images/architecture-en.png" width="900" alt="Four columns A to D. Each shows the client entry, the Microsoft-managed zone and your subscription. A: ACA Sandboxes service and a sandbox micro-VM, with sandbox group, registry and Blob volume in your subscription. B: Foundry Agent Service and a per-session sandbox, with project and Application Insights in your subscription. C: AKS control plane, then controller, Kata pod VM, Kata node pool and Blob PVC in your subscription. D: Foundry Agent Service, then subnet NSG, foundry-system, Kata pod VM per session and registry and storage in your subscription.">

What the figure shows: who owns each component and where the micro-VM boundary sits. A, C and D placement comes from our read-backs; B's VM-isolated boundary comes from Learn. It does not show networking detail, identity flows or the model gateway.

Words used in the results:

| Term | Meaning |
|---|---|
| Cold | A new session or sandbox until its first answer arrives |
| Warm | A follow-up turn in the same running session |
| Resume | An already suspended session: resume request → first answer; stop and idle waiting are excluded |
| Pod lost | Delete the session's pod; Kubernetes creates it again |
| Micro-VM | A lightweight VM with its own guest kernel per sandbox, session or pod (Kata Containers on AKS; Cloud Hypervisor read back inside ACA) |

## Steps

**Deployment order.** Path A, B and C reuse one foundation from the agenthost workshop ([`oxcp/ainotes` at `905f2c2`](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost)); this repository adds the SDK automation for A, the measurement and isolation checks for all four, and the BYO-AKS steps for D. Run only the paths you need:

| Order | Where | What | Script |
|---|---|---|---|
| 1 | Your workstation | Clone both repositories, fill one settings file | `config/.env.example` |
| 2 | Azure subscription | Shared foundation: Foundry project and model, APIM, ACR, Storage, managed identity | workshop `module-01/setup.sh` |
| 3 | C | AKS with a Kata node pool and the agent-sandbox controller | workshop `module-03/prepare-agent-sandbox.sh` |
| 4 | A | Sandbox group, then VNet connection, disk image, Blob volume, sandbox | workshop `module-04/sandbox-deploy.sh`, then [`tools/aca_sandbox.py`](tools/aca_sandbox.py) |
| 5 | B | Hosted agent with `azd` | workshop `module-02` |
| 6 | D | Your AKS bound to Foundry, the agent deployed onto it | private-preview sample + steps D1–D5 below |
| 7 | Each path | Measure, check the micro-VM | [`tools/measure_aca.py`](tools/measure_aca.py), [`tools/measure_responses.py`](tools/measure_responses.py), [`tools/check_isolation.py`](tools/check_isolation.py) |

The letters in this repository are canonical: **A = ACA Sandboxes, B = Foundry hosted agent, C = AKS + agent-sandbox, D = BYO-AKS**. The pinned upstream workshop predates this mapping and calls module-02 Solution A, module-03 Solution B and module-04 Solution C. Use its module numbers, not its solution letters, when following the upstream steps.

The example for A reuses C's image, VNet and Blob private endpoint, so that example requires step 3. ACA itself does not require AKS: for an A-only deployment, supply your own delegated `ACA_SUBNET_ID`, a reachable Blob private endpoint and the agent image instead of the AKS VNet commands below.

**Prerequisites.**
- Bash (Azure Cloud Shell, Linux, macOS or Git Bash) with Azure CLI 2.80 or newer, `kubectl`, `azd` 1.34 or newer with the `microsoft.foundry` extension (`azd ext install microsoft.foundry`), and Python 3.10 or newer;
- Owner on the subscription, or Contributor plus Role Based Access Control Administrator: the workshop creates role assignments;
- `gpt-5.4-mini` GlobalStandard quota in the region (we used East US 2);
- D only: the subscription enabled for the BYO-AKS private preview by the product group, and access to its sample.

**1. Clone and fill in the settings.** Subscription, region and names are typed once, in `.env`; every later command reads them:

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/david-xinyuwei/david-share.git
git -C david-share sparse-checkout set Agents/Azure-Agent-Sandbox-Hosting
cd david-share/Agents/Azure-Agent-Sandbox-Hosting
git clone https://github.com/oxcp/ainotes.git "$HOME/ainotes" && git -C "$HOME/ainotes" checkout 905f2c2
cp config/.env.example .env            # fill SUBSCRIPTION_ID and LOCATION now; SN after step 2
python -m pip install -r requirements.txt
set -a && . ./.env && set +a && az account set --subscription "$SUBSCRIPTION_ID"
```

`az account show --query id -o tsv` prints your subscription ID. Run `set -a && . ./.env && set +a` again in every new shell and after each edit of `.env`.

**2. Deploy the shared foundation.** It creates the Foundry account and project, the `gpt-5.4-mini` deployment, APIM, ACR, Storage and a user-assigned identity, and tags the resource group with a deployment suffix:

```bash
(cd "$WORKSHOP_DIR/agenthost/module-01" && chmod +x setup.sh && ./setup.sh)
az group show -g "$RESOURCE_GROUP" --query tags.deploymentSN -o tsv     # paste the value into SN in .env
```

It took <!--v:f.deploy.value-->3<!--/v--> minutes in our run and prints the deployment outputs. Then reload `.env`.

**3. Path C: AKS + agent-sandbox.** The workshop script builds the agent image, creates AKS with a Kata node pool (`--workload-runtime KataVmIsolation`, Azure Linux, a Gen2 VM size with nested virtualization) and installs agent-sandbox `v0.5.2`. If your Storage account has public network access disabled, run its private-link script as well:

```bash
cd "$WORKSHOP_DIR/agenthost/module-03"
chmod +x prepare-agent-sandbox.sh deploy-storage-private-link.sh && ./prepare-agent-sandbox.sh
./deploy-storage-private-link.sh        # only when public network access is disabled on the Storage account
kubectl apply -f agent-sandbox.yaml
kubectl wait --for=condition=Ready pod -l app=agent-host -n "$NAMESPACE" --timeout=3m
cd - && python tools/check_isolation.py -n "$NAMESPACE" -l app=agent-host --exec-uname
```

Success: `prepare-agent-sandbox.sh` ends with `Solution B infrastructure prepared` (the workshop's letter for this path), the wait prints `condition met`, and the check prints `CHECK isolation PASS … runtimeClass=kata-vm-isolation`. PASS checks the configured runtime and Running phase, not hardware attestation. Kernel versions may differ or be identical; an identical version is inconclusive, not failure. A failed requested `--exec-uname` read exits 1. Suspend and resume are one patch each; your gateway does this in production:

```bash
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Suspended"}}'
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Running"}}'
kubectl wait sandbox agent-host -n "$NAMESPACE" --for=condition=Ready --timeout=180s
```

**4. Path A: ACA Sandboxes.** The workshop script creates the sandbox group; [`tools/aca_sandbox.py`](tools/aca_sandbox.py) does the rest with the Python SDK, replacing the workshop's portal steps. The sandbox reuses the image built in step 3; without step 3, build it first with `az acr build -r "$ACA_ACR_NAME" -t agent-host:latest "$WORKSHOP_DIR/agenthost/module-03/agent-src"`. The subnet for the VNet connection must be delegated to `Microsoft.App/environments`; this example adds it to the AKS-managed VNet from step 3, as the workshop does:

```bash
(cd "$WORKSHOP_DIR/agenthost/module-04" && chmod +x sandbox-deploy.sh && ./sandbox-deploy.sh)
NODE_RG=$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query nodeResourceGroup -o tsv)
VNET=$(az network vnet list -g "$NODE_RG" --query "[0].name" -o tsv)
az network vnet subnet create -g "$NODE_RG" --vnet-name "$VNET" -n aca-subnet \
  --address-prefixes "$ACA_SUBNET_PREFIX" --delegations Microsoft.App/environments --query id -o tsv
# paste the printed ID into ACA_SUBNET_ID in .env, reload it, then:
python tools/aca_sandbox.py vnet && python tools/aca_sandbox.py image && python tools/aca_sandbox.py volume
python tools/aca_sandbox.py create
python tools/aca_sandbox.py ask "Where is Bangkok? Answer in one short sentence."
```

Each step prints one JSON line with its seconds. Success: `create` prints `"state": "Running"` and `"vmm_type": "cloudhypervisor"`, and `ask` prints the agent's answer with `"history_length": 1`. `stop`, `resume` and `delete` work the same way. Exit 2 names a missing setting; exit 1 prints the Azure error.

| If | Do |
|---|---|
| `image` fails with `RegistryAuthFailed` | The script uses a short-lived ACR token because a managed-identity pull returned 401 with SDK 0.1.0b4; check you can run `az acr login -n "$ACA_ACR_NAME" --expose-token` |
| `create` never reaches Running with the Blob volume | Storage with public access disabled needs the VNet connection and the private endpoint from step 3; the volume alone hung in our run |
| `ask` fails with `exit` | The agent is still starting; run `ask` again after 10 s |

**5. Path B: Foundry hosted agent.** Follow the workshop's module-02 from a working folder outside the clone. The commands, with this repository's variables:

```bash
mkdir -p "$HOME/workshop-b" && cd "$HOME/workshop-b" && azd auth login
azd ai agent init -m "$WORKSHOP_DIR/agenthost/module-02/azure.yaml" --project-id "$PROJECT_ID"
cd maf-agent && sed -i "s/<SN>/$SN/g" azure.yaml
azd env set AZURE_SUBSCRIPTION_ID "$SUBSCRIPTION_ID" && azd env set AZURE_LOCATION "$LOCATION"
azd env set AZURE_RESOURCE_GROUP "$RESOURCE_GROUP" && azd env set AZURE_AI_PROJECT_ID "$PROJECT_ID"
azd env set FOUNDRY_PROJECT_ENDPOINT "$PROJECT_ENDPOINT" && azd env set AI_MODEL_DEPLOYMENT_NAME gpt-5.4-mini
azd env set AZURE_TENANT_ID "$(az account show --query tenantId -o tsv)"
azd deploy && azd ai agent invoke "Hi"
```

Success: `azd deploy` reports the agent deployed (<!--v:b.deploy.value-->161<!--/v--> s in our gateway-route run) and `invoke` prints a reply. `MODEL_ROUTING` in `azure.yaml` chooses `gateway` (through APIM) or `direct`; the Results comparison uses the gateway-route observations, not the direct route. Before measuring B in step 7, copy its Responses endpoint from `azd ai agent show` into `AGENT_RESPONSES_URL` and reload this repository's `.env`. If `azd deploy` times out with `last status: creating`, open the agent in the Foundry portal; if it is not Running, delete it there and run `azd deploy` again.

**6. Path D: BYO-AKS (private preview).** Use a non-production subscription and confirm what you may share with the product group first. Three checks fail silently, so do them before anything bills:

| Check | Command | Expected |
|---|---|---|
| Subscription enabled | ask the product group; a `Registered` `FoundryComputePreview` feature flag is not enough | `azd provision` does not fail with `BringOwnFeatureNotEnabled` |
| VM size offered in the region | `az vm list-skus -l "$LOCATION" --size "$BYOAKS_KATA_VM_SIZE" --query "[].restrictions" -o json` | `[[]]`, not `NotAvailableForSubscription` (a quota ticket does not fix this) |
| Model quota in the same region | `az cognitiveservices usage list -l "$LOCATION" --query "[?contains(name.value,'gpt-5.4-mini')]" -o table` | capacity left for GlobalStandard |

**D1. A Kata node pool on your cluster.** The runtime must be exactly `KataVmIsolation`; `az` also accepts `KataMshvVmIsolation`, Foundry rejects it later, and the field cannot be changed:

```bash
az aks nodepool add -g "$BYOAKS_RESOURCE_GROUP" --cluster-name "$BYOAKS_CLUSTER" -n "$BYOAKS_KATA_POOL" \
  --os-sku AzureLinux --workload-runtime KataVmIsolation --node-vm-size "$BYOAKS_KATA_VM_SIZE" --node-count 1
```

**D2. Bind the cluster** with the product group's sample (`azd` only; there is no portal step). `azd` needs two sign-ins: `azd config set auth.useAzCliAuth true` for provision, and `azd auth login` for the `azure.ai.agents` extension. In the sample's folder, set `AKS_RESOURCE_ID`, `HOSTING_IDENTITY_RESOURCE_ID`, `WORKLOAD_IDENTITY_RESOURCE_ID`, `STORAGE_ACCOUNT_RESOURCE_ID`, `ACR_RESOURCE_ID` and `APPLICATIONINSIGHTS_RESOURCE_ID` with `azd env set` as its guide describes, then:

```bash
azd config set auth.useAzCliAuth true && azd auth login
azd provision
az aks get-credentials -g "$BYOAKS_RESOURCE_GROUP" -n "$BYOAKS_CLUSTER" --overwrite-existing
kubectl get pods -n foundry-system                  # all Ready; 13 total in our one-Kata-node onboarding
```

`azd provision` took <!--v:d.azd_provision.value-->522<!--/v--> s in our run. Foundry has now installed its runtime, including a `foundry-router` LoadBalancer on `:443` in your VNet.

**D3. Let Foundry reach the router, then deploy the agent.** If the AKS subnet has its own NSG, its default `DenyAllInBound` drops Foundry's calls to the router even though the AKS-managed NIC NSG allows them; our sessions failed for a day until we added this rule:

```bash
ROUTER_IP=$(kubectl get svc foundry-router -n foundry-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
az network nsg rule create -g "$BYOAKS_SUBNET_NSG_RG" --nsg-name "$BYOAKS_SUBNET_NSG" \
  -n Allow-AzureCloud-FoundryRouter-443 --priority 500 --direction Inbound --access Allow --protocol Tcp \
  --source-address-prefixes AzureCloud --destination-address-prefixes "$ROUTER_IP" --destination-port-ranges 443 \
  --query provisioningState -o tsv
azd deploy
```

Success: the rule prints `Succeeded`, and `azd deploy` finishes (<!--v:d.azd_deploy.value-->212<!--/v--> s in our run). Copy the agent's Responses endpoint from `azd ai agent show` into `AGENT_RESPONSES_URL`, and the session namespace (`kubectl get ns | grep foundry-`) into `SESSION_NAMESPACE`.

**D4. Check that sessions run on your nodes.**

```bash
python tools/measure_responses.py --cold 1 --warm-turns 1 --observe
python tools/check_isolation.py -n "$SESSION_NAMESPACE"
```

Success: `MEASURE PASS … remembered=True`, and one `CHECK isolation PASS … runtimeClass=kata-vm-isolation` line per session pod. Our run did not read the kernel inside a session pod; add `--container agent --exec-uname` only if your role may exec into session pods, and a refused read exits 1. `CHECK isolation BLOCKED no pod matched` means no session pod reached your cluster: check the NSG rule in D3, then ask the product group whether the subscription is enabled for placement.

**D5. Size the pool by sessions opened.** The oldest session pod was still Running at least <!--v:d.idle.value-->87<!--/v--> minutes after creation. This is pod age, not idle duration: its last-request timestamp was not retained, so we did not measure an idle eviction timeout. Until your preview build proves an eviction timeout, conservatively assume every opened session keeps a pod (500m CPU / 1 Gi plus 600 Mi Kata overhead) until the service ends it. Plan at most <!--v:d.capacity.per_node-->5<!--/v--> sessions per `Standard_D4s_v5` node (fewer on a node that also hosts system pods); the capacity-exhaustion runs returned both HTTP 500 `session_creation_failed` and HTTP 429 `aks_quota_exceeded`. Deleting a session pod does not free capacity: its Deployment recreates it.

| Error | Cause | What to do |
|---|---|---|
| `already bound to another Foundry account` | A failed Foundry account keeps the cluster binding | Delete **and purge** that account |
| `foundry-controller` in `CrashLoopBackOff`, Blob `not authorized` | Storage has public access disabled | Add a blob private endpoint reachable from the AKS subnet |
| `capabilityHosts: []`, `InternalServerError` after a failed provision | A rolled-back account cannot be reused | Delete and purge, then use a **new** `azd` environment name |
| `CannotDeleteResource … nested resources exist` | A hidden child project | List projects over ARM REST and delete them first |
| `AzureCLICredential: exit status 1` in `azd deploy` | The agents extension ignores `auth.useAzCliAuth` | Run `azd auth login` as well |
| `HTTP 429 aks_quota_exceeded` | Existing session pods occupy the Kata pool | Add nodes (scaling 1 → 3 took <!--v:d.node_scale.value-->143<!--/v--> s) |
| `HTTP 500 session_creation_failed` | Session creation failed; also observed during the full-pool run | Inspect the Kata pool and controller logs; do not assume every capacity refusal is HTTP 429 |

**7. Measure.** Return to this repository's root from the workshop or private-preview sample folder and reload `.env`; select B's or D's `AGENT_RESPONSES_URL` before each run. Each tool writes its record to `evidence-local/`, which Git ignores:

```bash
python tools/measure_aca.py agent --runs 3                           # A: the full agent path
python tools/measure_responses.py --cold 3 --warm-turns 3            # B (or D)
python tools/measure_responses.py --cold 3 --warm-turns 3 --scale 4,8 --recovery --observe   # D
```

Exit 0 means every requested phase passed: A checks growing conversation-history lengths across resume; B and D check completed responses and exact code-word recall, including each concurrency follow-up and recovery when requested. A skipped recovery, failed pod observation or failed deletion is not a pass. Expected full-pool refusals are saved as measurements but still return exit 1.

**Offboard and remove, only after explicit owner approval.** Stopped sandboxes do not bill CPU or memory. [Stopping AKS saves compute costs](https://learn.microsoft.com/azure/aks/start-stop-cluster), but retained disks, public IPs, storage and other resources can still incur charges; stopping is not zero total cost:

```bash
python tools/aca_sandbox.py delete
az aks stop -g "$RESOURCE_GROUP" -n "$AKS_NAME"                       # C; or az aks delete
az aks stop -g "$BYOAKS_RESOURCE_GROUP" -n "$BYOAKS_CLUSTER"         # D: ends node billing; the Foundry account and agent stay
az group delete -n "$RESOURCE_GROUP" --yes --no-wait                  # removes the foundation, A, B and C
```

## Results

**Where each path's micro-VM runs.** Measured read-backs for A, C and D, plus Learn's documented boundary for B:

| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| Boundary | One micro-VM per sandbox | Per-session VM-isolated sandbox (Learn) | One Kata pod VM per Sandbox | One Kata pod VM per session |
| VMM and hypervisor | `vmm_type` <!--v:a.vmm.value-->cloudhypervisor<!--/v--> from the API; the docs name no hypervisor | Not published; only VM isolation is documented | Cloud Hypervisor on Microsoft Hyper-V (Learn) | Same as C |
| Kernel read | Guest <!--v:a.isolation.guest_kernel-->6.12.8+<!--/v--> | Not exposed; not read back | Pod <!--v:c.isolation.guest_kernel-->6.6.137.mshv1-1.azl3<!--/v-->, runc pod <!--v:c.isolation.system_node_kernel-->6.8.0-1067-azure<!--/v--> | Kata pool nodes <!--v:d.node_kernel.value-->6.6.137.mshv2-2.azl3<!--/v-->; not read inside a session pod |
| Who owns the node | Microsoft | Microsoft | You | You |

B's per-session VM-isolated boundary is a `SOURCE_FACT`, not a local VMM or kernel read-back. C and D use the same Kata runtime class, so isolation does not separate them: who runs the platform does.

**The seconds.** A is the median of three runs on 2026-10-09; B and C are single runs on 2026-09-27; D is one benchmark campaign on 2026-09-29. A and C use the workshop reflection agent; B uses its Agent Framework agent with ResponsesHostServer; D uses the private-preview sample. All use the same model and a stateful two-turn interaction, but not the same container. B and D were called through the Foundry endpoint from a laptop in China (Azure round trip <!--v:d.client_rtt.range-->320–800<!--/v--> ms); A's answers were timed inside the sandbox and C's at its LoadBalancer. Compare within a row as a field report, not as a controlled product benchmark:

A's cold and resume-to-answer numbers are component sums, not a single end-to-end client stopwatch: SDK create or resume wall time plus the answer timed inside the sandbox. C's cold number is likewise apply-to-Ready plus first-answer time. Resume starts after suspension has finished; stop and idle waiting are excluded. B's rows use the gateway (APIM) model route; its direct-route results are separate in VALIDATION.

**Do not rank the four products by these seconds.** Clock locations, prompts, reasoning settings, output token counts, image cache and sample sizes were not uniformly controlled or fully recorded across paths. The same model name does not make these a matched latency benchmark. Numerical sanity checks can reject invalid timings; they cannot establish product performance or fill missing instrumentation.

<!-- BEGIN GENERATED: comparison -->
| Observed or documented | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| **New session → first answer** | 28.0 s median (27.07–50.41; n=3); SDK create + answer inside sandbox | 9.43 s (n=1, via the Foundry endpoint; model route: gateway (APIM)) | 57.65 s (n=1; apply → Ready 44.9 s + answer 12.75 s) | 25.49 s median (23.53–26.7; n=5), via the Foundry endpoint |
| **Follow-up turn, same session** | 1.05 s median (1.04–1.54; n=3) | 4.03, 5.54 s (n=2); model route: gateway (APIM) | 1.96 s (n=1) | 12.9 s median (11.17–16.42; n=3) |
| **Sandbox or pod until ready** | 24.25 s median (24.12–44.86; n=3), create → Running | not visible (managed) | 44.9 s (n=1), apply → Ready | 8–10 s (n=6) with the image cached; 25 s on a fresh node (n=2) |
| **Resume request → first answer (stop excluded)** | 14.45 s median (12.44–17.22; n=3); separately reported resume-only median 2.83 s | not exercised (the platform restores $HOME) | 34.2 s (n=1; Ready 17.1 s, LoadBalancer healthy 26.1 s) | Not exercised; confirm preview suspend support with the product group |
| **Pod lost → ready again** | not exercised (service-managed) | not exercised (managed) | 23.4 s (n=1); same Sandbox state restored | 23.9 s (n=1); deleted session continuity not tested |
| **Idle policy (official docs)** | [Docs](https://learn.microsoft.com/azure/container-apps/sandboxes-overview#lifecycle-states): configurable idle auto-suspend; Memory or Disk snapshot mode | [Docs](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents#sessions): 2–60 min idle timeout (default 15); compute released, $HOME and /files persisted; session deleted after 30 days of inactivity | [Official API](https://github.com/kubernetes-sigs/agent-sandbox/blob/v0.5.2/docs/api.md): shutdownTime is scheduled expiry, not request-idle detection; implement request-idle recycling yourself | No verified public official idle policy; confirm with the product group |
| **Capacity** | serverless; quota is concurrent cores (docs) | managed, per session | your Kata node pool | observed at most 5 per Standard_D4s_v5 node; 12 on 3 nodes; `500 session_creation_failed`, `429 aks_quota_exceeded` |
| **Path setup in our run** | 146.6 s (sandbox group, VNet connection, disk image, volume, sandbox) | 161 s (`azd deploy`; model route: gateway (APIM)) | 812.9 s (prepare script, private link, apply) | 854 s (roles, `azd provision`, `azd deploy`; author's console, not kept) |

**Local configuration and observation, not official idle policy:** A used Memory mode with a 900 s auto-suspend setting. D's oldest pod was still Running at least 87 min after creation; this is pod age, not idle duration or a reclamation timeout.
<!-- END GENERATED: comparison -->

What stands out:
- A's create is dominated by the private Blob volume: the image, VNet or port alone stayed at <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s, the volume raised create to <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s. Starting without the volume and mounting it later is an option only when the first answer does not need saved state.
- D's cold time includes more than pod boot: the session pod was Ready in <!--v:d.pod_ready.range-->8–10<!--/v--> s inside a <!--v:d.cold.range-->23.53–26.7<!--/v--> s end-to-end range, and the same model called directly from the same laptop took <!--v:d.model_direct.median-->2.69<!--/v--> s (median, low reasoning effort). This direct baseline is not a matched agent-configuration benchmark. The uninstrumented remainder is the endpoint, session and agent path collectively; it is not a measured gateway component.
- In the observed D preview build, plan capacity conservatively by sessions opened, not only requests active; the pod-age observation does not establish an idle eviction timeout.

**Measured runs.** Grouped A–D, then by date from oldest to newest within each path, in the same order as [VALIDATION.md](VALIDATION.md), which records each run's input, procedure, result and boundary:

<!-- BEGIN GENERATED: runs -->
| Run | Path | Scenario | Result |
|---|---|---|---|
| `a-2026-09-27-setup` | A | first deployment, wake on request | setup 146.6 s; a request to a stopped sandbox answered in 23.7 s; kernel `6.12.8+` |
| `a-2026-10-09-agent` | A | full agent, 3 runs | cold 28.0 s, warm 1.05 s, resume → answer 14.45 s (medians); vmm `cloudhypervisor` in 3/3; history grew 3/3 |
| `a-2026-10-09-attribution` | A | create time by configuration | image, VNet or port alone 1.6–2.7 s; adding the private Blob volume 15.79–25.12 s |
| `a-2026-10-09-late-mount` | A | start first, mount Blob later | start 3.99–7.2 s, then mount 12.4–20.29 s (n=2) |
| `b-2026-09-27` | B | deploy, new session, follow-ups; gateway model route | deploy 161 s; new session 9.43 s; follow-ups 4.03, 5.54 s |
| `c-2026-09-27` | C | Kata Sandbox, pod loss, suspend/resume | pod kernel `6.6.137.mshv1-1.azl3` vs runc pod on the Ubuntu system pool `6.8.0-1067-azure`; pod lost → Ready 23.4 s; resume → answer 34.2 s |
| `d-2026-09-29-placement` | D | where a session runs | every session pod `kata-vm-isolation` on your node pool after NSG rule `Allow-AzureCloud-FoundryRouter-443` |
| `d-2026-09-29-perf` | D | cold, warm, 4 and 8 at once | cold 25.49 s, warm 12.9 s (medians); 4 at once 0/4, 8 at once 2/8 answered (pool full) |
| `d-2026-09-29-recovery` | D | delete a session pod | replacement Ready in 23.9 s; a different live session answered in 14.8 s; deleted session continuity not tested |
<!-- END GENERATED: runs -->

**Limits.**
- One region (East US 2) and one client location. A full-agent latency has n=3. B and C are one end-to-end run each. D is one benchmark campaign: cached-image cold n=5, warm n=3, and one placement and one recovery observation. Repeat them with step 7 before you rely on a number.
- These are implementation observations, not product benchmarks or SLAs. The docs state sub-second sandbox start for A from prewarmed pools; our full specification with a private Blob volume did not get there.
- D is a private preview. Its behaviour (placement, idle reclaim, limits) may change, and its onboarding needs the product group's sample and enablement.
- Not measured: B's resume, A's and B's pod or host loss, the Foundry-governed agent calling an ACA sandbox per task (the pattern Microsoft describes since GA), and E2B or any other third-party sandbox.
- Cost: only D's floor was priced, <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h for two `Standard_D4s_v5` nodes and the AKS Standard tier at East US 2 retail prices. This is not the cost of the three-Kata-node benchmark pool; check your own prices.

**Repository.** [`tools/`](tools/) setup, measurement and checks · [`evidence/claims.json`](evidence/claims.json) canonical A–D claims · [`evidence/`](evidence/) redacted run records and `measurements.json` · [`images/`](images/) figures and icon ledger · [`config/`](config/) settings template · [`tests/`](tests/) offline tests (`python -m pytest -q`, run by CI on every push).
