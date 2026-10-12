# 在 Azure 上托管 agent 沙箱的四种方式：ACA Sandboxes、Foundry Hosted Agent、AKS + agent-sandbox、BYO-AKS

[![Paths](https://img.shields.io/badge/paths-A%20%C2%B7%20B%20%C2%B7%20C%20%C2%B7%20D-0078D4)](#目的)
[![Measured](https://img.shields.io/badge/measured-East%20US%202%2C%202026--09--22%20to%2010--09-0078D4)](#实现效果)
[![Docs CI](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml/badge.svg)](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml)

作者：魏新宇 · [English](README.md)

## 目的

你希望 agent 的每个会话都跑在 Azure 上独立的 micro-VM（带独立内核的轻量虚拟机）里，接下来要决定由谁来运行这个 micro-VM 和它的生命周期：交给一个托管服务、交给 Foundry，还是放在你自己的 AKS 节点上。四条路径各有取舍，光看产品页很难横向比较它们的秒数。本仓库用同一套可比的有状态交互口径测试四条路径，给出每条路径的复现步骤，并列出实测结果。A、B、C 使用 workshop 的 reflection agent，D 使用产品组的私有预览 sample，所以这不是相同容器的基准测试：

| 问题 | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| 状态 | GA | GA | Pod Sandboxing GA；agent-sandbox v0.5.2 为开源项目 | **私有预览** |
| 谁负责会话生命周期 | ACA 服务 | Foundry | 你自己：控制器、网关、空闲回收 | Foundry，运行在你的集群上 |
| 会话的 micro-VM 跑在哪里 | 服务托管的算力 | 微软托管的算力 | 你的 Kata 节点池 | 你的 Kata 节点池 |
| 带内存快照的挂起 | 支持 | 不支持：空闲后释放算力，恢复时还原 `$HOME` 和 `/files` | 不保存内存快照：你修改 `operatingMode`；pod 被删除，文件状态从 Blob 恢复 | 预览版不挂起 |
| 单个会话最大规格（文档） | 4 vCPU / 8 GB | 2 vCPU / 4 GiB | 受你选择的节点 VM 限制 | 节点 SKU 由你选择；预览版没有公布单会话上限 |
| 适合的场景 | 要每用户一个 micro-VM、要快照、不想管集群 | 希望 Foundry 托管一切，且它的限制对你没有影响 | 必须自己掌控节点、网络和控制面 | 既要 Foundry 的 agent 平台，**又**要求会话跑在你自己的节点上 |

产品能力和状态来自 Microsoft Learn，实测声明来自我们的运行记录，最后一行是我们的选型建议。一般企业 agent 在 B 的限制可接受时先选 B。对于 Qira 这类需要按用户长期运行、要内存快照又不想管集群的场景，先选 A。只有执行节点必须归你时才选 C 或 D；如果还要求 Foundry 继续运行 agent 平台，选 D。

<img src="images/architecture-zh.png" width="900" alt="A 到 D 四列，每列包含客户端入口、微软托管区和你的订阅。A：ACA Sandboxes 服务和沙箱 micro-VM，你的订阅里有 sandbox group、镜像仓库和 Blob 卷。B：Foundry Agent Service 和每会话沙箱，你的订阅里有项目和 Application Insights。C：AKS 控制面，你的订阅里有控制器、Kata pod VM、Kata 节点池和 Blob PVC。D：Foundry Agent Service，你的订阅里有子网 NSG、foundry-system、每会话 Kata pod VM，以及镜像仓库和存储。">

这张图说明的是：各组件归谁所有、micro-VM 的边界在哪里。A、C、D 的落点来自我们的读回，B 的 VM 级隔离边界来自 Learn。图里没有画网络细节、身份流转和模型网关。

看结果时要用到的几个词：

| 术语 | 含义 |
|---|---|
| 冷启动（cold） | 新会话或新沙箱从创建到返回第一个回答 |
| 后续回答（warm） | 同一个运行中会话里的下一轮对话 |
| 恢复（resume） | 会话已完成挂起：发起恢复 → 第一个回答；不含停止和空闲等待 |
| pod 丢失（pod lost） | 删除会话所在的 pod，由 Kubernetes 重新创建 |
| micro-VM | 每个沙箱、会话或 pod 一个带独立 guest 内核的轻量虚拟机（AKS 上是 Kata Containers；ACA 沙箱内读到的 VMM 是 Cloud Hypervisor） |

## 操作步骤

**部署顺序。** A、B、C 复用 agenthost workshop（[`oxcp/ainotes`，固定在 `905f2c2`](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost)）的同一套基础资源；本仓库补充了 A 的 SDK 自动化、四条路径通用的测量和隔离检查，以及 D 的 BYO-AKS 步骤。只需执行你用得到的路径：

| 顺序 | 在哪里 | 做什么 | 脚本 |
|---|---|---|---|
| 1 | 你的工作站 | 克隆两个仓库，填写一个配置文件 | `config/.env.example` |
| 2 | Azure 订阅 | 共享基础资源：Foundry 项目和模型、APIM、ACR、Storage、托管身份 | workshop `module-01/setup.sh` |
| 3 | C | 带 Kata 节点池的 AKS，以及 agent-sandbox 控制器 | workshop `module-03/prepare-agent-sandbox.sh` |
| 4 | A | Sandbox group，然后是 VNet 连接、磁盘镜像、Blob 卷、沙箱 | workshop `module-04/sandbox-deploy.sh`，然后 [`tools/aca_sandbox.py`](tools/aca_sandbox.py) |
| 5 | B | 用 `azd` 部署 hosted agent | workshop `module-02` |
| 6 | D | 把你的 AKS 绑定到 Foundry，并把 agent 部署上去 | 私有预览示例 + 下文 D1–D5 |
| 7 | 每条路径 | 测量，检查 micro-VM | [`tools/measure_aca.py`](tools/measure_aca.py)、[`tools/measure_responses.py`](tools/measure_responses.py)、[`tools/check_isolation.py`](tools/check_isolation.py) |

本仓库的字母是统一口径：**A = ACA Sandboxes，B = Foundry hosted agent，C = AKS + agent-sandbox，D = BYO-AKS**。固定版本的上游 workshop 早于这次字母重映射，它把 module-02 叫 Solution A、module-03 叫 Solution B、module-04 叫 Solution C。执行上游步骤时请看 module 编号，不要沿用它的 Solution 字母。

下面 A 的示例复用 C 的镜像、VNet 和 Blob 私有终结点，因此这个示例需要第 3 步。ACA 本身不依赖 AKS：只部署 A 时，可自行提供已委派的 `ACA_SUBNET_ID`、可访问的 Blob 私有终结点和 agent 镜像，替代下文读取 AKS VNet 的命令。

**前提条件。**
- Bash 环境（Azure Cloud Shell、Linux、macOS 或 Git Bash），装好 Azure CLI 2.80 及以上、`kubectl`、`azd` 1.34 及以上并安装 `microsoft.foundry` 扩展（`azd ext install microsoft.foundry`），以及 Python 3.10 及以上；
- 订阅的 Owner，或 Contributor 加 Role Based Access Control Administrator：workshop 会创建角色分配；
- 所在区域有 `gpt-5.4-mini` GlobalStandard 配额（我们用的是 East US 2）；
- 仅 D 需要：产品组已为该订阅开通 BYO-AKS 私有预览，并给了示例代码的访问权限。

**1. 克隆并填写配置。** 订阅、区域和资源名只在 `.env` 里填一次，后面所有命令都从这里读取：

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/david-xinyuwei/david-share.git
git -C david-share sparse-checkout set Agents/Azure-Agent-Sandbox-Hosting
cd david-share/Agents/Azure-Agent-Sandbox-Hosting
git clone https://github.com/oxcp/ainotes.git "$HOME/ainotes" && git -C "$HOME/ainotes" checkout 905f2c2
cp config/.env.example .env            # 先填 SUBSCRIPTION_ID 和 LOCATION；SN 在第 2 步之后填
python -m pip install -r requirements.txt
set -a && . ./.env && set +a && az account set --subscription "$SUBSCRIPTION_ID"
```

`az account show --query id -o tsv` 会打印你的订阅 ID。每开一个新 shell、每次修改 `.env` 之后，都要重新执行 `set -a && . ./.env && set +a`。

**2. 部署共享基础资源。** 它会创建 Foundry 账户和项目、`gpt-5.4-mini` 部署、APIM、ACR、Storage 和一个用户分配的托管身份，并给资源组打上部署后缀标签：

```bash
(cd "$WORKSHOP_DIR/agenthost/module-01" && chmod +x setup.sh && ./setup.sh)
az group show -g "$RESOURCE_GROUP" --query tags.deploymentSN -o tsv     # 把输出填到 .env 的 SN
```

我们这次用了 <!--v:f.deploy.value-->3<!--/v--> 分钟，结束时会打印部署输出。然后重新加载 `.env`。

**3. 路径 C：AKS + agent-sandbox。** workshop 脚本会构建 agent 镜像，创建带 Kata 节点池的 AKS（`--workload-runtime KataVmIsolation`、Azure Linux、支持嵌套虚拟化的 Gen2 规格），并安装 agent-sandbox `v0.5.2`。如果 Storage 账户禁用了公网访问，还要运行它的私有链接脚本：

```bash
cd "$WORKSHOP_DIR/agenthost/module-03"
chmod +x prepare-agent-sandbox.sh deploy-storage-private-link.sh && ./prepare-agent-sandbox.sh
./deploy-storage-private-link.sh        # 仅在 Storage 账户禁用公网访问时需要
kubectl apply -f agent-sandbox.yaml
kubectl wait --for=condition=Ready pod -l app=agent-host -n "$NAMESPACE" --timeout=3m
cd - && python tools/check_isolation.py -n "$NAMESPACE" -l app=agent-host --exec-uname
```

成功的标志：`prepare-agent-sandbox.sh` 最后输出 `Solution B infrastructure prepared`（workshop 里这条路径叫 Solution B），wait 输出 `condition met`，检查输出 `CHECK isolation PASS … runtimeClass=kata-vm-isolation … kernel=differs-from-node`。挂起和恢复各是一条 patch 命令；生产环境里由你的网关来执行：

```bash
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Suspended"}}'
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Running"}}'
kubectl wait sandbox agent-host -n "$NAMESPACE" --for=condition=Ready --timeout=180s
```

**4. 路径 A：ACA Sandboxes。** workshop 脚本创建 sandbox group；剩下的工作由 [`tools/aca_sandbox.py`](tools/aca_sandbox.py) 用 Python SDK 完成，替代 workshop 里的门户操作。沙箱复用第 3 步构建的镜像；如果没做第 3 步，先用 `az acr build -r "$ACA_ACR_NAME" -t agent-host:latest "$WORKSHOP_DIR/agenthost/module-03/agent-src"` 构建。VNet 连接用的子网必须委派给 `Microsoft.App/environments`；下面和 workshop 一样，把它加在第 3 步的 AKS 托管 VNet 里：

```bash
(cd "$WORKSHOP_DIR/agenthost/module-04" && chmod +x sandbox-deploy.sh && ./sandbox-deploy.sh)
NODE_RG=$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query nodeResourceGroup -o tsv)
VNET=$(az network vnet list -g "$NODE_RG" --query "[0].name" -o tsv)
az network vnet subnet create -g "$NODE_RG" --vnet-name "$VNET" -n aca-subnet \
  --address-prefixes "$ACA_SUBNET_PREFIX" --delegations Microsoft.App/environments --query id -o tsv
# 把打印出的 ID 填到 .env 的 ACA_SUBNET_ID，重新加载后执行：
python tools/aca_sandbox.py vnet && python tools/aca_sandbox.py image && python tools/aca_sandbox.py volume
python tools/aca_sandbox.py create
python tools/aca_sandbox.py ask "Where is Bangkok? Answer in one short sentence."
```

每一步都会打印一行 JSON，带上耗时秒数。成功的标志：`create` 输出 `"state": "Running"` 和 `"vmm_type": "cloudhypervisor"`，`ask` 输出 agent 的回答并带 `"history_length": 1`。`stop`、`resume`、`delete` 用法相同。退出码 2 表示缺少某个配置项（会打印名字）；退出码 1 会打印 Azure 返回的错误。

| 现象 | 处理 |
|---|---|
| `image` 报 `RegistryAuthFailed` | 用 SDK 0.1.0b4 时托管身份拉取镜像返回过 401，所以脚本改用短期 ACR token；确认你能执行 `az acr login -n "$ACA_ACR_NAME" --expose-token` |
| 带 Blob 卷时 `create` 一直到不了 Running | Storage 禁用公网时，需要 VNet 连接和第 3 步的私有终结点；我们只挂卷不连 VNet 时请求一直没有返回 |
| `ask` 报 `exit` | agent 还在启动，10 s 后再执行一次 `ask` |

**5. 路径 B：Foundry hosted agent。** 按 workshop 的 module-02 在克隆目录之外的工作目录里执行。下面是代入本仓库变量后的命令：

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

成功的标志：`azd deploy` 报告 agent 部署完成（本次 gateway 路由 <!--v:b.deploy.value-->161<!--/v--> s），`invoke` 打印出回答。`azure.yaml` 里的 `MODEL_ROUTING` 用来选 `gateway`（经 APIM）还是 `direct`；结果总表使用 gateway 路由的观察值，不是 direct 路由。第 7 步测 B 之前，把 `azd ai agent show` 中它的 Responses 端点填到 `AGENT_RESPONSES_URL`，并重新加载本仓库的 `.env`。如果 `azd deploy` 以 `last status: creating` 超时，到 Foundry 门户里看这个 agent；如果它不是 Running，在门户里删除后重新 `azd deploy`。

**6. 路径 D：BYO-AKS（私有预览）。** 请使用非生产订阅，并先和产品组确认哪些内容可以对外分享。下面三项检查失败时不会有明确报错，所以要在产生费用之前做：

| 检查项 | 命令 | 期望结果 |
|---|---|---|
| 订阅已开通 | 找产品组确认；`FoundryComputePreview` 功能标志显示 `Registered` 并不代表已开通 | `azd provision` 不报 `BringOwnFeatureNotEnabled` |
| 该区域提供这个 VM 规格 | `az vm list-skus -l "$LOCATION" --size "$BYOAKS_KATA_VM_SIZE" --query "[].restrictions" -o json` | 输出 `[[]]`，而不是 `NotAvailableForSubscription`（这个问题提配额工单解决不了） |
| 同一区域有模型配额 | `az cognitiveservices usage list -l "$LOCATION" --query "[?contains(name.value,'gpt-5.4-mini')]" -o table` | GlobalStandard 还有剩余容量 |

**D1. 在你的集群上加一个 Kata 节点池。** 运行时必须写 `KataVmIsolation`；`az` 也接受 `KataMshvVmIsolation`，但 Foundry 之后会拒绝，而且这个字段创建后不能修改：

```bash
az aks nodepool add -g "$BYOAKS_RESOURCE_GROUP" --cluster-name "$BYOAKS_CLUSTER" -n "$BYOAKS_KATA_POOL" \
  --os-sku AzureLinux --workload-runtime KataVmIsolation --node-vm-size "$BYOAKS_KATA_VM_SIZE" --node-count 1
```

**D2. 绑定集群。** 使用产品组提供的示例（只能用 `azd`，门户里没有这个步骤）。`azd` 需要两次登录：`azd config set auth.useAzCliAuth true` 供 provision 使用，`azd auth login` 供 `azure.ai.agents` 扩展使用。在示例目录里按它的指南用 `azd env set` 设置 `AKS_RESOURCE_ID`、`HOSTING_IDENTITY_RESOURCE_ID`、`WORKLOAD_IDENTITY_RESOURCE_ID`、`STORAGE_ACCOUNT_RESOURCE_ID`、`ACR_RESOURCE_ID` 和 `APPLICATIONINSIGHTS_RESOURCE_ID`，然后：

```bash
azd config set auth.useAzCliAuth true && azd auth login
azd provision
az aks get-credentials -g "$BYOAKS_RESOURCE_GROUP" -n "$BYOAKS_CLUSTER" --overwrite-existing
kubectl get pods -n foundry-system                  # 全部 Ready；我们单 Kata 节点接入时共 13 个
```

我们这次 `azd provision` 用了 <!--v:d.azd_provision.value-->522<!--/v--> s。到这一步，Foundry 已经装好它的运行时，其中包括你 VNet 里一个监听 `:443` 的 `foundry-router` LoadBalancer。

**D3. 让 Foundry 能访问 router，然后部署 agent。** 如果 AKS 子网自己绑了 NSG，它默认的 `DenyAllInBound` 会丢弃 Foundry 发往 router 的请求，即使 AKS 托管的网卡 NSG 已经放行；我们的会话因此失败了一整天，直到加上这条规则：

```bash
ROUTER_IP=$(kubectl get svc foundry-router -n foundry-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
az network nsg rule create -g "$BYOAKS_SUBNET_NSG_RG" --nsg-name "$BYOAKS_SUBNET_NSG" \
  -n Allow-AzureCloud-FoundryRouter-443 --priority 500 --direction Inbound --access Allow --protocol Tcp \
  --source-address-prefixes AzureCloud --destination-address-prefixes "$ROUTER_IP" --destination-port-ranges 443 \
  --query provisioningState -o tsv
azd deploy
```

成功的标志：规则命令输出 `Succeeded`，`azd deploy` 正常结束（我们这次 <!--v:d.azd_deploy.value-->212<!--/v--> s）。把 `azd ai agent show` 里 agent 的 Responses 端点填到 `AGENT_RESPONSES_URL`，把会话所在的命名空间（`kubectl get ns | grep foundry-`）填到 `SESSION_NAMESPACE`。

**D4. 确认会话跑在你的节点上。**

```bash
python tools/measure_responses.py --cold 1 --warm-turns 1 --observe
python tools/check_isolation.py -n "$SESSION_NAMESPACE" --container agent --exec-uname
```

成功的标志：输出 `MEASURE PASS … remembered=True`，并且每个会话 pod 输出一行 `CHECK isolation PASS … runtimeClass=kata-vm-isolation`。如果输出 `CHECK isolation BLOCKED no pod matched`，说明没有会话 pod 到达你的集群：先检查 D3 的 NSG 规则，再问产品组该订阅是否已开通会话落地。

**D5. 按“开过的会话数”规划节点池。** 最早的会话 pod 在创建至少 <!--v:d.idle.value-->87<!--/v--> 分钟后仍是 Running。这是 pod 的年龄，不是空闲时长：没有保留它最后一次请求的时间戳，所以未测出空闲回收超时。在你的预览 build 证实有回收超时之前，应保守地假设每个开过的会话都会一直占着一个 pod（500m CPU / 1 Gi，外加 600 Mi Kata 开销），直到服务端结束它。每个 `Standard_D4s_v5` 节点最多按 <!--v:d.capacity.per_node-->5<!--/v--> 个会话规划（同时跑系统 pod 的节点更少）；容量耗尽实测既返回过 HTTP 500 `session_creation_failed`，也返回过 HTTP 429 `aks_quota_exceeded`。删除会话 pod 不会释放容量：它的 Deployment 会把它重新创建出来。

| 报错 | 原因 | 处理 |
|---|---|---|
| `already bound to another Foundry account` | 失败的 Foundry 账户仍占着集群绑定 | 删除**并清除（purge）**该账户 |
| `foundry-controller` 处于 `CrashLoopBackOff`，Blob 报 `not authorized` | Storage 禁用了公网访问 | 添加一个 AKS 子网能访问到的 blob 私有终结点 |
| provision 失败后出现 `capabilityHosts: []`、`InternalServerError` | 回滚过的账户无法复用 | 删除并清除，然后换一个**新的** `azd` 环境名 |
| `CannotDeleteResource … nested resources exist` | 存在一个看不见的子项目 | 通过 ARM REST 列出并先删除子项目 |
| `azd deploy` 报 `AzureCLICredential: exit status 1` | agents 扩展不认 `auth.useAzCliAuth` | 再执行一次 `azd auth login` |
| `HTTP 429 aks_quota_exceeded` | 已有会话 pod 占用 Kata 节点池 | 加节点（从 1 个扩到 3 个用了 <!--v:d.node_scale.value-->143<!--/v--> s） |
| `HTTP 500 session_creation_failed` | 会话创建失败；节点池满时也观察到过 | 检查 Kata 节点池和控制器日志；不要假设每次容量拒绝都返回 HTTP 429 |

**7. 测量。** 从 workshop 或私有预览示例目录返回本仓库根目录，重新加载 `.env`；每次运行前选好 B 或 D 的 `AGENT_RESPONSES_URL`。每个工具都把记录写到 `evidence-local/`，这个目录被 Git 忽略：

```bash
python tools/measure_aca.py agent --runs 3                           # A：完整 agent 流程
python tools/measure_responses.py --cold 3 --warm-turns 3            # B（或 D）
python tools/measure_responses.py --cold 3 --warm-turns 3 --scale 4,8 --recovery --observe   # D
```

退出码 0 表示每个请求的阶段都通过：A 检查对话历史长度在恢复前后持续增长；B 和 D 检查响应完成和口令精确匹配，包括要求执行的每个并发追问和恢复。跳过恢复、读取 pod 失败或删除失败都不算通过。节点池满时的预期拒绝仍会保存为测量记录，但返回退出码 1。

**下线与删除。** 沙箱停止后不再收取 CPU 和内存费用；AKS 集群在停止或删除之前会一直按节点计费：

```bash
python tools/aca_sandbox.py delete
az aks stop -g "$RESOURCE_GROUP" -n "$AKS_NAME"                       # C；或 az aks delete
az aks stop -g "$BYOAKS_RESOURCE_GROUP" -n "$BYOAKS_CLUSTER"         # D：停止节点计费；Foundry 账户和 agent 保留
az group delete -n "$RESOURCE_GROUP" --yes --no-wait                  # 删除基础资源以及 A、B、C
```

## 实现效果

**每条路径的 micro-VM 跑在哪里。** A、C、D 来自实测读回，B 来自 Learn 写明的隔离边界：

| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| 隔离边界 | 每个沙箱一个 micro-VM | 每会话一个 VM 级隔离的沙箱（Learn） | 每个 Sandbox 一个 Kata pod VM | 每个会话一个 Kata pod VM |
| VMM 和 hypervisor | API 返回 `vmm_type` <!--v:a.vmm.value-->cloudhypervisor<!--/v-->；文档没有写 hypervisor | 未公开；文档只说明 VM 级隔离 | Microsoft Hyper-V 上的 Cloud Hypervisor（Learn） | 与 C 相同 |
| 读到的内核 | guest <!--v:a.isolation.guest_kernel-->6.12.8+<!--/v--> | 未暴露，无法读回 | pod <!--v:c.isolation.guest_kernel-->6.6.137.mshv1-1.azl3<!--/v-->，runc pod <!--v:c.isolation.system_node_kernel-->6.8.0-1067-azure<!--/v--> | Kata 节点池节点 <!--v:d.node_kernel.value-->6.6.137.mshv2-2.azl3<!--/v-->；未在会话 pod 内读取 |
| 节点归谁 | 微软 | 微软 | 你 | 你 |

B 的“每会话一个、VM 级隔离”是 `SOURCE_FACT`，不是本地读回的 VMM 或内核结果。C 和 D 用的是同一个 Kata runtime class，所以隔离方式并不能区分它们，真正的区别在于由谁来运行平台。

**秒数。** A 是 2026-10-09 三次实测的中位数；B 和 C 是 2026-09-27 各一次实测；D 是 2026-09-29 的一轮压测。A、B、C 使用 workshop reflection agent，D 使用私有预览 sample。它们使用同一个模型和同一种有状态两轮交互，但不是相同容器。B 和 D 是从中国的一台笔记本经 Foundry 端点调用的（到 Azure 的往返时延 <!--v:d.client_rtt.range-->320–800<!--/v--> ms）；A 的回答在沙箱内计时，C 在它的 LoadBalancer 处计时。请把同一行作为现场记录来读，不要当作受控的产品基准测试：

A 的冷启动和恢复到回答采用分段相加，不是客户端连续端到端计时：SDK 创建或恢复的耗时，加上沙箱内计时的回答耗时。C 的冷启动同样是 apply 到 Ready 加第一个回答。恢复从挂起完成后开始计时，不含停止和空闲等待。B 的各行使用 gateway（APIM）模型路由；direct 路由的结果在 VALIDATION 单独列出。

<!-- BEGIN GENERATED: comparison -->
| 实测或文档项 | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |
|---|---|---|---|---|
| **新会话 → 第一个回答** | 中位数 28.0 s（27.07–50.41；n=3）；SDK 创建 + 沙箱内回答 | 9.43 s（n=1，经 Foundry 端点；模型路由：gateway（APIM）） | 57.65 s（n=1；apply → Ready 44.9 s + 回答 12.75 s） | 中位数 25.49 s（23.53–26.7；n=5），经 Foundry 端点 |
| **同一会话的后续回答** | 中位数 1.05 s（1.04–1.54；n=3） | 4.03, 5.54 s（n=2）；模型路由：gateway（APIM） | 1.96 s（n=1） | 中位数 12.9 s（11.17–16.42；n=3） |
| **沙箱或 pod 就绪** | 中位数 24.25 s（24.12–44.86；n=3），create → Running | 不可见（平台托管） | 44.9 s（n=1），apply → Ready | 镜像已缓存 8–10 s（n=6）；新节点 25 s（n=2） |
| **发起恢复 → 第一个回答（不含停止）** | 中位数 14.45 s（12.44–17.22；n=3）；恢复本身另行统计的中位数 2.83 s | 未测（平台恢复 $HOME） | 34.2 s（n=1；Ready 17.1 s，LoadBalancer 健康 26.1 s） | 预览版没有挂起 |
| **pod 丢失 → 重新就绪** | 未测（服务托管） | 未测（平台托管） | 23.4 s（n=1）；同一 Sandbox 的状态已恢复 | 23.9 s（n=1）；未测试被删会话的连续性 |
| **空闲时的行为** | 到你设置的空闲时间（本次 900 s）自动挂起，保存内存快照 | 文档：空闲 2–60 min（默认 15）后释放算力，恢复时还原 $HOME 和 /files | 无内置机制；需要你修改 operatingMode | 最早的 pod 创建至少 87 min 后仍是 Running；未测空闲时长 |
| **容量** | 无服务器；配额按并发核数计（文档） | 平台托管，按会话 | 你的 Kata 节点池 | 单个 Standard_D4s_v5 节点观察到最多 5 个会话；3 个节点共 12 个；`500 session_creation_failed`、`429 aks_quota_exceeded` |
| **本次路径部署耗时** | 146.6 s（sandbox group、VNet 连接、磁盘镜像、卷、沙箱） | 161 s（`azd deploy`；模型路由：gateway（APIM）） | 812.9 s（准备脚本、私有链接、apply） | 854 s（角色、`azd provision`、`azd deploy`；作者控制台记录，原始输出未保留） |
<!-- END GENERATED: comparison -->

值得注意的几点：
- A 的创建耗时主要花在私有 Blob 卷上：只加镜像、VNet 或端口时是 <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s，加上卷之后升到 <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s。只有在第一个回答不依赖已保存状态时，才适合先启动、后挂卷。
- D 的冷启动不只是 pod 启动：会话 pod 在 <!--v:d.pod_ready.range-->8–10<!--/v--> s 内就绪，端到端范围却是 <!--v:d.cold.range-->23.53–26.7<!--/v--> s；同一台笔记本直接调用同一个模型的中位数是 <!--v:d.model_direct.median-->2.69<!--/v--> s（low reasoning effort）。该直连基线不是与 agent 配置完全匹配的基准测试。没有打点的剩余时间只能统称为端点、会话和 agent 链路，不能当成已测出的网关内部耗时。
- 在本次观察到的 D 预览 build 里，应保守地按“开过的会话数”而不只是“正在执行的请求数”规划容量；pod 年龄的观察不能证明空闲回收超时。

**每次实测。** 按 A→B→C→D 分组，同一路径内按日期从早到晚排列，与 [VALIDATION_CN.md](VALIDATION_CN.md) 的顺序一致；该页记录每次实测的输入、过程、结果和边界：

<!-- BEGIN GENERATED: runs -->
| 实测 | 路径 | 场景 | 结果 |
|---|---|---|---|
| `a-2026-09-27-setup` | A | 首次部署，请求唤醒 | 部署 146.6 s；向已停止的沙箱发请求，23.7 s 后回答；内核 `6.12.8+` |
| `a-2026-10-09-agent` | A | 完整 agent，跑 3 次 | 冷启动 28.0 s、后续 1.05 s、恢复到回答 14.45 s（中位数）；3/3 次 vmm 为 `cloudhypervisor`；状态保留 3/3 |
| `a-2026-10-09-attribution` | A | 按配置拆分创建耗时 | 只加镜像、VNet 或端口 1.6–2.7 s；加私有 Blob 卷后 15.79–25.12 s |
| `a-2026-10-09-late-mount` | A | 先启动、后挂 Blob | 启动 3.99–7.2 s，再挂载 12.4–20.29 s（n=2） |
| `b-2026-09-27` | B | 部署、新会话、后续回答；gateway 模型路由 | 部署 161 s；新会话 9.43 s；后续 4.03, 5.54 s |
| `c-2026-09-27` | C | Kata Sandbox、pod 丢失、挂起/恢复 | pod 内核 `6.6.137.mshv1-1.azl3`，Ubuntu 系统节点池的 runc pod 内核 `6.8.0-1067-azure`；pod 丢失后 23.4 s 就绪；恢复到回答 34.2 s |
| `d-2026-09-29-placement` | D | 会话跑在哪里 | 加 NSG 规则 `Allow-AzureCloud-FoundryRouter-443` 后，每个会话 pod 都是 `kata-vm-isolation`，跑在你的节点池上 |
| `d-2026-09-29-perf` | D | 冷启动、后续、4 和 8 并发 | 冷启动 25.49 s、后续 12.9 s（中位数）；4 并发 0/4、8 并发 2/8 回答（节点池已满） |
| `d-2026-09-29-recovery` | D | 删除一个会话 pod | 23.9 s 后替代 pod 就绪；另一个存活会话在 14.8 s 内回答；未测试被删会话的连续性 |
<!-- END GENERATED: runs -->

**边界。**
- 只测了一个区域（East US 2）和一个客户端位置。A 的完整 agent 时延 n=3；B 和 C 各是一轮端到端实测；D 是一轮压测，其中镜像已缓存的冷启动 n=5、后续回答 n=3，会话落地和恢复各一次。在依赖某个数字之前，请用第 7 步重新测。
- 这些是实施过程中的观察，不是产品基准测试，也不是 SLA。文档说 A 可以从预热池实现亚秒级启动；我们带私有 Blob 卷的完整配置没有达到。
- D 是私有预览，它的行为（会话落地、空闲回收、各项限制）可能变化，接入需要产品组的示例和开通。
- 没有测的：B 的恢复，A 和 B 的 pod 或宿主机丢失，由 Foundry 治理的 agent 按任务调用 ACA 沙箱（微软在 GA 后介绍的模式），以及 E2B 或其他第三方沙箱。
- 成本：只给 D 的保底费用算了价：两个 `Standard_D4s_v5` 节点加 AKS Standard 层，按 East US 2 零售价为 <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h。这不是三个 Kata 节点压测池的费用；请按你自己的价格核算。

**仓库目录。** [`tools/`](tools/) 部署、测量和检查工具 · [`evidence/claims.json`](evidence/claims.json) A–D canonical 声明 · [`evidence/`](evidence/) 脱敏后的实测记录和 `measurements.json` · [`images/`](images/) 图和图标台账 · [`config/`](config/) 配置模板 · [`tests/`](tests/) 离线测试（`python -m pytest -q`，CI 每次推送都会运行）。
