# Four Ways to Host an Agent Sandbox on Azure: ACA Sandboxes, Foundry Hosted Agent, AKS + agent-sandbox, BYO-AKS

[![Paths](https://img.shields.io/badge/paths-A%20%C2%B7%20B%20%C2%B7%20C%20%C2%B7%20D-0078D4)](#purpose)
[![Measured](https://img.shields.io/badge/measured-East%20US%202%2C%202026--09--22%20to%2010--09-0078D4)](#results)
[![Docs CI](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml/badge.svg)](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml)

Author: Xinyu Wei · [中文](README_CN.md)

## Purpose

You want each agent session to run in its own micro-VM on Azure, and you have to pick who runs that micro-VM and its lifecycle: a service, Foundry, or you on your own AKS nodes. The four paths make different trade-offs, and their seconds are hard to compare from product pages alone. This repository applies one comparable stateful interaction contract to all four paths, gives the steps to reproduce each one, and shows what we measured. A and C share the workshop reflection agent; B uses its distinct Agent Framework agent with ResponsesHostServer; D uses the product group's private-preview sample, so this is not an identical-container benchmark:

| Question | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| Status | GA (2026-09-23) | GA (2026-06) | Pod Sandboxing GA (2025-11-18); agent-sandbox v0.5.2 is open source | **Private preview** |
| Who runs the session lifecycle | The ACA service | Foundry | You: controller, gateway, idle sweeper | Foundry, on your cluster |
| Where the session's micro-VM runs | Service-managed compute | Microsoft-managed compute | Your Kata node pool | Your Kata node pool |
| Suspend with memory snapshot | Yes | No: idle releases compute, `$HOME` and `/files` are restored | No memory snapshot: you patch `operatingMode`; the pod goes, file state comes back from Blob | Not exercised; confirm preview suspend support with the product group |
| Selectable size / node SKU (docs) | XS: 0.25 vCPU / 0.5 GB<br>S: 0.5 vCPU / 1 GB<br>M (default): 1 vCPU / 2 GB<br>L: 2 vCPU / 4 GB<br>XL: 4 vCPU / 8 GB | 0.5 vCPU / 1 GiB<br>1 vCPU / 2 GiB<br>2 vCPU / 4 GiB<br>No VM SKU to choose | Kata-compatible Gen2 VM SKUs, such as the Dsv3 series; Azure Linux and nested virtualization required | Kata node SKUs within the approved preview requirements; confirm the selectable list with the product group |
| Choose it when | You want per-user micro-VMs, snapshots and no cluster | You want Foundry to run everything and its limits do not bite | You must own the node, the network and the control plane | You need Foundry's agent platform **and** sessions on nodes you own |

Product capabilities come from Microsoft Learn and status from the official announcements linked below; measured claims come from our runs; the last row is our selection guidance. Start with B for a general managed enterprise agent when its limits fit. For a long-running, per-user workload that needs memory snapshots without a cluster, start with A. Use C or D only when owning the execution nodes is non-negotiable; choose D when Foundry must still run the agent platform.

The selectable row separates [A's sandbox resource tiers](https://learn.microsoft.com/azure/container-apps/sandboxes-overview#resource-tiers), [B's sandbox sizes](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents#sandbox-sizes), and [C/D's node VM SKUs](https://learn.microsoft.com/azure/aks/use-pod-sandboxing). A node SKU is not a per-session allocation; Kata pod requests, limits and overhead also constrain each session. Node availability and quota vary by region and subscription, and D has additional preview requirements.

Status sources: [A GA announcement](https://techcommunity.microsoft.com/blog/appsonazureblog/azure-container-apps-sandboxes-now-generally-available/4559125), [B GA announcement](https://azure.microsoft.com/updates?id=563596), [C Pod Sandboxing GA](https://azure.microsoft.com/updates?id=526237); agent-sandbox itself is an open-source project, not an Azure service. For C/D, [Kata provides compute isolation only](https://learn.microsoft.com/azure/aks/concepts-pod-sandboxing), not isolation of the control plane, storage/data paths or cluster-admin actions.

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

**Deploy first, then use the agent.** These steps follow the [pinned colleague workshop](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost): shared foundation, the selected hosting path, and an actual application call. Benchmarking and evidence replay are optional and live in VALIDATION, not on the deployment path.

**Execution boundary:** the original deployments produced the retained historical results, but this parameterized sequence has not been re-run end to end in a fresh subscription. D also requires the product group's private sample and approval. Offline CI does not certify a successful cloud deployment.

| Order | Where | Action | Entry |
|---|---|---|---|
| 1 | Workstation | Clone, configure and sign in | `config/.env.example` |
| 2 | Azure subscription | Create the shared foundation | workshop `module-01/setup.sh` |
| 3 | C | Build the image, create AKS, deploy the Sandbox and open the app | workshop `module-03` |
| 4 | A | Create the sandbox group and deploy the agent sandbox | workshop `module-04` + `tools/aca_sandbox.py` |
| 5 | B | Deploy the hosted agent and invoke it | workshop `module-02` |
| 6 | D | Bind an existing AKS, deploy and invoke the private-preview agent | product-group sample |

**A = ACA Sandboxes, B = Foundry hosted agent, C = AKS + agent-sandbox, D = BYO-AKS**. Upstream calls module-02 Solution A, module-03 Solution B and module-04 Solution C. Use module numbers when reading it. Run only the path you need: B needs steps 1–2; the full A example reuses the image and network created in step 3. ACA itself does not require AKS. D uses an existing cluster and its own approved sample, not an automatically created cluster from this page.

**Prerequisites.** Bash, Git, curl, Python 3.10+ with the venv module (on Ubuntu/Debian install `python3-venv`), Azure CLI 2.80+, kubectl and azd 1.34+; install the Foundry extension with `azd ext install microsoft.foundry`. You need subscription Owner, or Contributor plus Role Based Access Control Administrator, and model quota in your selected region. For A, your identity also needs the `Container Apps SandboxGroup Data Owner` role ([Learn](https://learn.microsoft.com/azure/container-apps/sandboxes-overview#prerequisites)); the workshop's `sandbox-deploy.sh` assigns it when it can resolve your signed-in user and otherwise only prints a warning. Use a non-production subscription. For C, provide your actual public egress CIDR; this HTTP workshop app is not a production HTTPS/login deployment.

**1. Clone and configure.** Clone into new directories, then edit `.env`: set `SUBSCRIPTION_ID`, `LOCATION` and `RESOURCE_GROUP`; set `C_ALLOW_CIDR` if using C. Leave `SN` until step 2.

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/david-xinyuwei/david-share.git
git -C david-share sparse-checkout set Agents/Azure-Agent-Sandbox-Hosting
cd david-share/Agents/Azure-Agent-Sandbox-Hosting
export HOSTING_REPO="$PWD"
git clone https://github.com/oxcp/ainotes.git "$HOME/ainotes"
git -C "$HOME/ainotes" checkout 905f2c2
cp config/.env.example .env
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

After editing `.env`, sign in and run the upstream prerequisite checker. Resolve failed tool, permission or quota prerequisites before creating resources. Keep this shell open: it holds `HOSTING_REPO`, the stable return directory, and the activated `.venv`.

```bash
az login
set -a && . "$HOSTING_REPO/.env" && set +a
az account set --subscription "$SUBSCRIPTION_ID"
(cd "$WORKSHOP_DIR/agenthost" && chmod +x check-prerequisites.sh && ./check-prerequisites.sh)
```

**2. Create the shared foundation.** [Module 1](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-01/README.md) creates the Foundry project/model, APIM, ACR, Storage and identity.

```bash
(cd "$WORKSHOP_DIR/agenthost/module-01" && chmod +x setup.sh && ./setup.sh)
az group show -g "$RESOURCE_GROUP" --query tags.deploymentSN -o tsv
```

Success: `Template deployed successfully.` and a non-empty suffix from the second command. Paste that suffix into `SN` in `.env`, then reload it so the derived resource names change together. Do not continue with `YOUR-DEPLOYMENT-SUFFIX`.

```bash
set -a && . "$HOSTING_REPO/.env" && set +a
```

**3. C — Deploy the AKS application.** Follow [Module 3](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-03/README.md). Preparation creates AKS/Kata, controller, image and PVC; it does not deploy the application. The separate `kubectl apply` creates the Sandbox and its Service.

```bash
cd "$WORKSHOP_DIR/agenthost/module-03"
chmod +x prepare-agent-sandbox.sh deploy-storage-private-link.sh
./prepare-agent-sandbox.sh
if [ "$(az storage account show -g "$RESOURCE_GROUP" -n "$ACA_STORAGE_ACCOUNT" --query publicNetworkAccess -o tsv)" = "Disabled" ]; then
  ./deploy-storage-private-link.sh
fi
kubectl apply -f agent-sandbox.yaml
kubectl wait sandbox agent-host -n "$NAMESPACE" --for=condition=Ready --timeout=5m
```

Success: the Sandbox reaches `Ready`. Now allow only your configured egress CIDR, obtain the actual application URL, and make a real request:

```bash
if [ -z "$C_ALLOW_CIDR" ] || [[ "$C_ALLOW_CIDR" == YOUR-* ]]; then
else
  kubectl patch service agent-host-lb -n "$NAMESPACE" --type merge \
    -p "{\"spec\":{\"loadBalancerSourceRanges\":[\"$C_ALLOW_CIDR\"]}}"
  kubectl wait service agent-host-lb -n "$NAMESPACE" --for=jsonpath='{.status.loadBalancer.ingress[0].ip}' --timeout=5m
  C_AGENT_HOST=$(kubectl get service agent-host-lb -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  C_AGENT_URL="http://${C_AGENT_HOST}"
  printf 'C application: %s\n' "$C_AGENT_URL"
  curl --fail-with-body --show-error --max-time 30 "$C_AGENT_URL/ready"
  curl --fail-with-body --show-error --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"query":"Where is Bangkok? Answer in one short sentence."}' "$C_AGENT_URL/reflect" | \
    python -c 'import json,sys; body=json.load(sys.stdin); answer=body.get("response"); print(json.dumps(body,ensure_ascii=False)); sys.exit(0 if isinstance(answer,str) and answer.strip() else "C invocation failed: non-empty response missing")'
fi
cd "$HOSTING_REPO"
```

If the first line prints the `C_ALLOW_CIDR` message, nothing else ran: set it, reload `.env` and run the block again. 
Open the printed URL in a browser to use the chat page. `/ready` must succeed and `/reflect` must return a non-empty `response`; a Ready pod alone is not an answering application. On failure inspect `kubectl get pods`, `kubectl describe pod` and `kubectl logs` in `NAMESPACE`; for unreachable HTTP also check the egress CIDR, Service address and network rules.
 Following the workshop, `aca-subnet` goes into the VNet of the AKS node resource group; [AKS documents modifying resources there as unsupported](https://learn.microsoft.com/azure/aks/faq) and deletes that group with the cluster. Treat this as a workshop shortcut; outside the workshop, give the sandbox group a delegated subnet in a VNet you own.
**4. A — Deploy the ACA sandbox agent.** [Module 4](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-04/README.md) creates the sandbox group. Its disk-image and sandbox steps are separate; the SDK commands below perform those operations rather than replay measurements. This full example reuses step 3's image/network. Choose a free `ACA_SUBNET_PREFIX` inside that VNet.

```bash
(cd "$WORKSHOP_DIR/agenthost/module-04" && chmod +x sandbox-deploy.sh && ./sandbox-deploy.sh)
NODE_RG=$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query nodeResourceGroup -o tsv)
VNET=$(az network vnet list -g "$NODE_RG" --query "[0].name" -o tsv)
az network vnet subnet create -g "$NODE_RG" --vnet-name "$VNET" -n aca-subnet \
  --address-prefixes "$ACA_SUBNET_PREFIX" --delegations Microsoft.App/environments --query id -o tsv
```

Paste the printed subnet ID into `ACA_SUBNET_ID` in `.env`, then run from this repository:

```bash
cd "$HOSTING_REPO"
set -a && . ./.env && set +a
python tools/aca_sandbox.py vnet
python tools/aca_sandbox.py image
python tools/aca_sandbox.py volume
python tools/aca_sandbox.py create
python tools/aca_sandbox.py ask "Where is Bangkok? Answer in one short sentence."
```

Success: `create` reports Running and `ask` returns an actual answer. The answer is requested through sandbox exec; direct browser ingress is not implied. For the browser-port deployment use Module 4's port/IP configuration. If a setting is missing, fill it before retrying; image access failures require ACR permission, and private Blob access requires the VNet and private endpoint.

**5. B — Deploy and call the hosted agent.** Use [Module 2](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-02/README.md) from a new working directory outside the source clone. Bind the existing project, then deploy; do not run `azd provision` for B.

```bash
mkdir -p "$HOME/workshop-b" && cd "$HOME/workshop-b"
azd auth login
azd ai agent init -m "$WORKSHOP_DIR/agenthost/module-02/azure.yaml" --project-id "$PROJECT_ID"
cd maf-agent
sed -i "s/<SN>/$SN/g" azure.yaml
azd env set AZURE_SUBSCRIPTION_ID "$SUBSCRIPTION_ID"
azd env set AZURE_LOCATION "$LOCATION"
azd env set AZURE_RESOURCE_GROUP "$RESOURCE_GROUP"
azd env set AZURE_AI_PROJECT_ID "$PROJECT_ID"
azd env set FOUNDRY_PROJECT_ENDPOINT "$PROJECT_ENDPOINT"
azd env set AI_MODEL_DEPLOYMENT_NAME gpt-5.4-mini
azd env set AZURE_TENANT_ID "$(az account show --query tenantId -o tsv)"
azd deploy
azd ai agent invoke "Hi"
azd ai agent show
cd "$HOSTING_REPO"
```

Success: deployment finishes and `invoke` returns a real answer. `show` provides the deployed endpoint for API clients; the Foundry agent playground can also invoke it. Keep the default gateway route unless intentionally changing it. If deployment remains creating, inspect `azd ai agent monitor` and the portal status rather than treating timeout as success or automatically deleting the agent.

**6. D — Private-preview deployment.** This path is blocked without the product group's sample and subscription approval. Before running it, fill the BYOAKS values in `.env`, including `BYOAKS_SAMPLE_DIR`, and configure the sample's `AKS_RESOURCE_ID`, `HOSTING_IDENTITY_RESOURCE_ID`, `WORKLOAD_IDENTITY_RESOURCE_ID`, `STORAGE_ACCOUNT_RESOURCE_ID`, `ACR_RESOURCE_ID` and `APPLICATIONINSIGHTS_RESOURCE_ID` using its current guide. Feature registration alone is not approval.

**D1. Prepare the existing cluster's Kata pool.** Confirm this node SKU is available for your subscription/region and meets the approved preview requirements; do not recreate an existing pool of the same name.

```bash
az aks nodepool add -g "$BYOAKS_RESOURCE_GROUP" --cluster-name "$BYOAKS_CLUSTER" -n "$BYOAKS_KATA_POOL" \
  --os-sku AzureLinux --workload-runtime KataVmIsolation --node-vm-size "$BYOAKS_KATA_VM_SIZE" --node-count 1
```

**D2. Provision the binding from the approved sample directory.** Success means provision completes and foundry-system components become Ready; their count is not a success contract.

```bash
cd "$BYOAKS_SAMPLE_DIR"
azd config set auth.useAzCliAuth true
azd auth login
azd provision
az aks get-credentials -g "$BYOAKS_RESOURCE_GROUP" -n "$BYOAKS_CLUSTER" --overwrite-existing
kubectl get pods -n foundry-system
```

**D3. Configure the router network path only when needed.** The following subnet-NSG rule is the retained workshop recipe, not a universal production rule. Run it only if the product group confirms this ingress/source applies to your preview build and your subnet NSG blocks that route; otherwise use the approved network configuration.

```bash
ROUTER_IP=$(kubectl get svc foundry-router -n foundry-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$ROUTER_IP" ]; then
  echo "foundry-router has no LoadBalancer address; resolve provisioning before deploying the agent" >&2
else
  az network nsg rule create -g "$BYOAKS_SUBNET_NSG_RG" --nsg-name "$BYOAKS_SUBNET_NSG" \
    -n Allow-AzureCloud-FoundryRouter-443 --priority 500 --direction Inbound --access Allow --protocol Tcp \
    --source-address-prefixes AzureCloud --destination-address-prefixes "$ROUTER_IP" --destination-port-ranges 443 \
    --query provisioningState -o tsv
fi
```

**D4. Deploy and invoke the application.** These commands remain in the sample directory. Success: deployment completes, invoke answers, and show displays the endpoint. On failures inspect the sample/controller logs and the approved storage, identity and network settings; do not automatically purge resources.

```bash
azd deploy
azd ai agent invoke "Hi"
azd ai agent show
cd "$HOSTING_REPO"
```

**Done.** The selected path has an actual answering application and its usable entry, not merely a completed measurement command. Optional cold/warm tests, isolation checks and historical-number replay are in [VALIDATION.md](VALIDATION.md#tests-and-offline-checks).

**Offboard only after explicit owner approval; not part of deployment.** Each command below is its own block: run only the ones for resources you intend to remove. Stopping AKS saves compute charges, not all retained-resource charges.

Delete the A sandbox:

```bash
python tools/aca_sandbox.py delete
```

Stop the C cluster:

```bash
az aks stop -g "$RESOURCE_GROUP" -n "$AKS_NAME"
```

Stop the D cluster:

```bash
az aks stop -g "$BYOAKS_RESOURCE_GROUP" -n "$BYOAKS_CLUSTER"
```

Delete the resource group. This removes the shared foundation and A/B/C, and `az` asks you to confirm first:

```bash
az group delete -n "$RESOURCE_GROUP" --no-wait
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
- One region (East US 2) and one client location. A full-agent latency has n=3. B and C are one end-to-end run each. D is one benchmark campaign: cached-image cold n=5, warm n=3, and one placement and one recovery observation. Repeat them with the live measurement commands in VALIDATION before you rely on a number.
- These are implementation observations, not product benchmarks or SLAs. The docs state sub-second sandbox start for A from prewarmed pools; our full specification with a private Blob volume did not get there.
- D is a private preview. Its behaviour (placement, idle reclaim, limits) may change, and its onboarding needs the product group's sample and enablement.
- Not measured: B's resume, A's and B's pod or host loss, the Foundry-governed agent calling an ACA sandbox per task (the pattern Microsoft describes since GA), and E2B or any other third-party sandbox.
- Cost: only D's floor was priced, <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h for two `Standard_D4s_v5` nodes and the AKS Standard tier at East US 2 retail prices. This is not the cost of the three-Kata-node benchmark pool; check your own prices.

**Repository.** [`tools/`](tools/) setup, measurement and checks · [`evidence/claims.json`](evidence/claims.json) canonical A–D claims · [`evidence/`](evidence/) redacted run records and `measurements.json` · [`images/`](images/) figures and icon ledger · [`config/`](config/) settings template · [`tests/`](tests/) offline tests (`python -m pytest -q`, run by CI on every push).
