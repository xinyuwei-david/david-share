# Azure Agent 沙箱托管：ACA Sandboxes、Foundry Hosted Agent、AKS + agent-sandbox

[![Paths](https://img.shields.io/badge/paths-A%20%C2%B7%20B%20%C2%B7%20C-0078D4)](#目的)
[![Measured](https://img.shields.io/badge/measured-East%20US%202%2C%202026--09--27%20to%2010--09-0078D4)](#实现效果)
[![CI](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml/badge.svg)](https://github.com/david-xinyuwei/david-share/actions/workflows/agent-sandbox-hosting-ci.yml)

作者：魏新宇 · [English](README.md)

## 目的

你需要在 Azure 上运行隔离、有状态的 agent 会话，并决定由谁负责生命周期、执行节点归谁。这个公开版使用 agenthost workshop 的 reflection-agent 工作负载，比较三条可复现路径。容器打包方式、计时口径和客户端网络链路不同：这是现场记录，不是相同容器的产品基准测试。

**发布范围。** 这里包含 A/B/C 的文档、脱敏实测和工具。D / BYO-AKS 是私有预览，接入步骤、实现细节和实测不在此公开。[完整配套仓库](https://github.com/david-xinyuwei/azure-agent-sandbox-hosting) 保持 Private，需要访问权限。

| 问题 | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox |
|---|---|---|---|
| 状态 | GA | GA | Pod Sandboxing GA；agent-sandbox v0.5.2 为开源项目 |
| 谁负责会话生命周期 | ACA 服务 | Foundry | 你自己：控制器、网关、空闲回收 |
| 会话的 micro-VM 跑在哪里 | 服务托管的算力 | 微软托管的算力 | 你的 Kata 节点池 |
| 带内存快照的挂起 | 支持 | 不支持：空闲后释放算力，恢复时还原 `$HOME` 和 `/files` | 不保存内存快照：你修改 `operatingMode`；pod 被删除，文件状态从 Blob 恢复 |
| 单个会话最大规格（文档） | 4 vCPU / 8 GB | 2 vCPU / 4 GiB | 受你选择的节点 VM 限制 |

一般企业 agent 在 B 的限制可接受时先选 B。对于 Qira 这类需要按用户长期运行、要内存快照又不想管集群的场景，先选 A。需要客户自有执行节点、自己运行会话控制器、网关和空闲策略时选 C；AKS 的 Kubernetes API 仍由微软托管。

<img src="images/architecture-zh.png" width="900" alt="A 到 C 三列展示客户端、微软托管服务和你的订阅。A 和 B 使用托管算力，C 在你的节点池运行 Kata pod VM。A/C 的落点来自实测，B 的 VM 级隔离边界来自文档。">

这是原创解释图，不是产品截图。A/C 的落点来自读回，B 的每会话 VM 级隔离边界来自 Learn。图中未展开身份和模型网关链路。

| 术语 | 含义 |
|---|---|
| 冷启动（cold） | 新沙箱或会话到第一个回答；各路径采用的计时口径不同 |
| 后续回答（warm） | 同一运行中会话的追问 |
| 恢复（resume） | 会话已完成挂起：发起恢复 → 第一个回答；不含停止和空闲等待 |
| pod 丢失 | 删除 pod，测替代 pod 到 Ready 的时间 |

## 操作步骤

**部署顺序。** A/B/C 复用 [agenthost `905f2c2`](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost) 的基础资源。这里的统一字母为 **A = ACA Sandboxes，B = Foundry hosted agent，C = AKS + agent-sandbox**。上游把 module-02 叫 Solution A、module-03 叫 Solution B、module-04 叫 Solution C；执行时看 module 编号，不要沿用上游字母。

A 的示例复用 C 的镜像、VNet 和 Blob 私有终结点，因此需要第 3 步。ACA 本身不依赖 AKS：只部署 A 时，自行提供已委派的 `ACA_SUBNET_ID`、可访问的 Blob 私有终结点和 agent 镜像，替代下文读取 AKS VNet 的命令。

**前提。** Bash（Linux、macOS 或 Git Bash）、Azure CLI 2.80 及以上、`kubectl`、安装 `microsoft.foundry` 的 `azd` 1.34 及以上、Python 3.10 及以上、`gpt-5.4-mini` GlobalStandard 配额，以及 Owner 或 Contributor 加 Role Based Access Control Administrator。没有测试 Azure Cloud Shell。

**1. 克隆并配置。** 稀疏检出避免下载这个大型仓库的无关项目：

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/david-xinyuwei/david-share.git
git -C david-share sparse-checkout set Agents/Azure-Agent-Sandbox-Hosting
cd david-share/Agents/Azure-Agent-Sandbox-Hosting
git clone https://github.com/oxcp/ainotes.git "$HOME/ainotes" && git -C "$HOME/ainotes" checkout 905f2c2
cp config/.env.example .env
python -m pip install -r requirements.txt
az login && azd ext install microsoft.foundry
set -a && . ./.env && set +a && az account set --subscription "$SUBSCRIPTION_ID"
```

先填 `SUBSCRIPTION_ID` 和 `LOCATION`；第 2 步之后填 `SN`。`az account show --query id -o tsv` 确认订阅。每次修改 `.env`、每开一个新 shell 都要重新加载。

**2. 部署基础资源。** 创建 Foundry 账户/项目和模型、APIM、ACR、Storage 和托管身份：

```bash
(cd "$WORKSHOP_DIR/agenthost/module-01" && chmod +x setup.sh && ./setup.sh)
az group show -g "$RESOURCE_GROUP" --query tags.deploymentSN -o tsv
```

成功时打印部署输出（本次 <!--v:f.deploy.value-->3<!--/v--> 分钟）。把 tag 填到 `.env` 的 `SN`，再加载。失败时先解决 ARM 错误，不要假设基础资源已创建。

**3. C：AKS + agent-sandbox。** 固定版本的 workshop 构建镜像并安装 Kata 节点池、控制器。只在 Storage 禁用公网时执行私有链接步骤：

```bash
cd "$WORKSHOP_DIR/agenthost/module-03"
chmod +x prepare-agent-sandbox.sh deploy-storage-private-link.sh && ./prepare-agent-sandbox.sh
./deploy-storage-private-link.sh        # 仅在 Storage 禁用公网时执行
kubectl apply -f agent-sandbox.yaml
kubectl wait --for=condition=Ready pod -l app=agent-host -n "$NAMESPACE" --timeout=3m
cd - && python tools/check_isolation.py -n "$NAMESPACE" -l app=agent-host --exec-uname
```

成功时依次输出 `Solution B infrastructure prepared`、`condition met`、`CHECK isolation PASS … runtimeClass=kata-vm-isolation`。能读取 guest 内核时，会与实际宿主节点比较。没有匹配 pod 时检查命名空间与 Sandbox 事件；exec 被拒绝时会明确显示内核未读取。Git Bash 把 ARM ID 转为 Windows 路径时，设置 `MSYS_NO_PATHCONV=1`。

```bash
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Suspended"}}'
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Running"}}'
kubectl wait sandbox agent-host -n "$NAMESPACE" --for=condition=Ready --timeout=180s
```

最后的 wait 输出 `condition met`；失败时检查 Sandbox 和 pod 事件。恢复的是 Blob 文件状态，不是进程内存。示例没有内置空闲回收或请求唤醒。

**4. A：ACA Sandboxes。** workshop 创建 group，SDK 工具注册 VNet 连接、磁盘镜像和 Blob 卷。这里使用 C 的 VNet：

```bash
(cd "$WORKSHOP_DIR/agenthost/module-04" && chmod +x sandbox-deploy.sh && ./sandbox-deploy.sh)
NODE_RG=$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query nodeResourceGroup -o tsv)
VNET=$(az network vnet list -g "$NODE_RG" --query "[0].name" -o tsv)
az network vnet subnet create -g "$NODE_RG" --vnet-name "$VNET" -n aca-subnet \
  --address-prefixes "$ACA_SUBNET_PREFIX" --delegations Microsoft.App/environments --query id -o tsv
```

把 ID 填到 `ACA_SUBNET_ID`，重新加载 `.env`，然后：

```bash
python tools/aca_sandbox.py vnet && python tools/aca_sandbox.py image && python tools/aca_sandbox.py volume
python tools/aca_sandbox.py create
python tools/aca_sandbox.py ask "Where is Bangkok? Answer in one short sentence."
```

成功时 create 输出 `"state": "Running"` 和 `"vmm_type": "cloudhypervisor"`；ask 输出回答及历史长度。退出码 2 指出缺失配置，退出码 1 报 SDK 错误。镜像导入报 `RegistryAuthFailed` 时确认 `az acr login -n "$ACA_ACR_NAME" --expose-token` 可用；带私有 Blob 时创建卡住，检查 VNet 连接和私有终结点；agent 尚在启动时，10 s 后重试 ask。

```bash
python tools/aca_sandbox.py stop
python tools/aca_sandbox.py resume
python tools/aca_sandbox.py show
```

成功时 stop 报 Stopped，resume 报 Running；失败状态或超时不能算恢复成功。

**5. B：Foundry hosted agent。** 在克隆目录外的工作目录部署：

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

成功时部署完成（记录的 gateway 路由 <!--v:b.deploy.value-->161<!--/v--> s），invoke 打印回答。`MODEL_ROUTING` 选择 gateway/APIM 或 direct，结果总表使用 gateway 的观察值。测量前把 `azd ai agent show` 中该 agent 的 Responses 端点填到 `AGENT_RESPONSES_URL`，并加载本项目的 `.env`。部署以 `last status: creating` 超时时，检查 Foundry 中的 agent；不是 Running 时，删除失败版本后重试。

**6. 测量。** 返回本项目根目录、重新加载 `.env`。结果保存在被 Git 忽略的 `evidence-local/`：

```bash
python tools/measure_aca.py agent --runs 3
python tools/measure_responses.py --cold 3 --warm-turns 3
python tools/measure_responses.py --cold 3 --warm-turns 3 --scale 4,8
```

退出码 0 表示所有请求的检查通过：A 的历史长度在恢复前后持续增长；B 的响应完成且口令精确匹配，包括并发追问。失败会记录并返回退出码 1；配置缺失或参数无效返回退出码 2。新实测不必等于这些历史耗时。

**删除测试资源。** ACA 停止后不收 CPU/内存费；AKS 在停止或删除前持续计费：

```bash
python tools/aca_sandbox.py delete
az aks stop -g "$RESOURCE_GROUP" -n "$AKS_NAME"
az group delete -n "$RESOURCE_GROUP" --yes --no-wait
```

成功时，存在沙箱则删除输出 `deleted: true`，AKS stop 完成，测试资源组在异步删除后消失。失败时检查 Azure 错误，不要假设费用已停止。

## 实现效果

**隔离。** A/C 是本地读回，B 是文档写明的边界：

| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox |
|---|---|---|---|
| 隔离边界 | 每个沙箱一个 micro-VM | 每会话一个 VM 级隔离的沙箱（Learn） | 每个 Sandbox 一个 Kata pod VM |
| VMM 和 hypervisor | API 返回 `vmm_type` <!--v:a.vmm.value-->cloudhypervisor<!--/v-->；文档未写 hypervisor | 未公开；文档写明 VM 隔离 | Microsoft Hyper-V 上的 Cloud Hypervisor（Learn） |
| 读到的内核 | guest <!--v:a.isolation.guest_kernel-->6.12.8+<!--/v--> | 未读回 | Sandbox pod <!--v:c.isolation.guest_kernel-->6.6.137.mshv1-1.azl3<!--/v-->；Ubuntu 系统节点池的 runc pod <!--v:c.isolation.system_node_kernel-->6.8.0-1067-azure<!--/v--> |
| 节点归谁 | 微软 | 微软 | 你 |

**计时口径。** A 采用分段相加，不是客户端连续端到端计时：SDK 创建/恢复耗时，加上沙箱内回答耗时。C 的冷启动是 apply 到 Ready 加第一个回答。恢复不含停止和空闲等待。每次实测先求总耗时，再取中位数，不能把各项独立统计的中位数相加。B 在客户端经 Foundry 计时，模型路由为 gateway/APIM。

<!-- BEGIN GENERATED: comparison -->
| 实测或文档项 | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox |
|---|---|---|---|
| **新会话 → 第一个回答** | 中位数 28.0 s（27.07–50.41；n=3）；SDK 创建 + 沙箱内回答 | 9.43 s（n=1，经 Foundry 端点；模型路由：gateway（APIM）） | 57.65 s（n=1；apply → Ready 44.9 s + 回答 12.75 s） |
| **同一会话的后续回答** | 中位数 1.05 s（1.04–1.54；n=3） | 4.03, 5.54 s（n=2）；模型路由：gateway（APIM） | 1.96 s（n=1） |
| **沙箱或 pod 就绪** | 中位数 24.25 s（24.12–44.86；n=3），create → Running | 不可见（平台托管） | 44.9 s（n=1），apply → Ready |
| **发起恢复 → 第一个回答（不含停止）** | 中位数 14.45 s（12.44–17.22；n=3）；恢复本身另行统计的中位数 2.83 s | 未测（平台恢复 $HOME） | 34.2 s（n=1；Ready 17.1 s，LoadBalancer 健康 26.1 s） |
| **pod 丢失 → 重新就绪** | 未测（服务托管） | 未测（平台托管） | 23.4 s（n=1）；同一 Sandbox 的状态已恢复 |
| **空闲时的行为** | 到你设置的空闲时间（本次 900 s）自动挂起，保存内存快照 | 文档：空闲 2–60 min（默认 15）后释放算力，恢复时还原 $HOME 和 /files | 无内置机制；需要你修改 operatingMode |
| **容量** | 无服务器；配额按并发核数计（文档） | 平台托管，按会话 | 你的 Kata 节点池 |
| **本次路径部署耗时** | 146.6 s（sandbox group、VNet 连接、磁盘镜像、卷、沙箱） | 161 s（`azd deploy`；模型路由：gateway（APIM）） | 812.9 s（准备脚本、私有链接、apply） |
<!-- END GENERATED: comparison -->

A 的创建耗时由镜像/VNet/端口配置的 <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s，升到私有 Blob 卷配置的 <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s。这说明额外耗时出现在哪一步，并未打点服务内部原因。第一个回答不需要已保存状态时，才适合后挂卷。

**每次实测。** 按 A→B→C 分组，组内日期从早到晚，与 [VALIDATION_CN.md](VALIDATION_CN.md) 完全一致：

<!-- BEGIN GENERATED: runs -->
| 实测 | 路径 | 场景 | 结果 |
|---|---|---|---|
| `a-2026-09-27-setup` | A | 首次部署，请求唤醒 | 部署 146.6 s；向已停止的沙箱发请求，23.7 s 后回答；内核 `6.12.8+` |
| `a-2026-10-09-agent` | A | 完整 agent，跑 3 次 | 冷启动 28.0 s、后续 1.05 s、恢复到回答 14.45 s（中位数）；3/3 次 vmm 为 `cloudhypervisor`；状态保留 3/3 |
| `a-2026-10-09-attribution` | A | 按配置拆分创建耗时 | 只加镜像、VNet 或端口 1.6–2.7 s；加私有 Blob 卷后 15.79–25.12 s |
| `a-2026-10-09-late-mount` | A | 先启动、后挂 Blob | 启动 3.99–7.2 s，再挂载 12.4–20.29 s（n=2） |
| `b-2026-09-27` | B | 部署、新会话、后续回答；gateway 模型路由 | 部署 161 s；新会话 9.43 s；后续 4.03, 5.54 s |
| `c-2026-09-27` | C | Kata Sandbox、pod 丢失、挂起/恢复 | pod 内核 `6.6.137.mshv1-1.azl3`，Ubuntu 系统节点池的 runc pod 内核 `6.8.0-1067-azure`；pod 丢失后 23.4 s 就绪；恢复到回答 34.2 s |
<!-- END GENERATED: runs -->

**边界。**
- 只测 East US 2。A 的完整 agent n=3；B、C 各一轮历史实测。工作负载、时钟和客户端链路不是受控的产品基准测试或 SLA。
- B 的 direct 结果在 VALIDATION 单独列出。B/C 的历史请求原文没有保留，新工具不能重新生成缺失的原始提示词。
- A 的 guest uptime 观察本身不能证明进程内存连续性；恢复前后的对话历史长度另有记录。
- 未测：B 的空闲恢复和宿主机丢失、A 的宿主机丢失、其他区域、第三方沙箱，以及这些整理后工具在全新订阅的端到端运行。
- 公开版不包含 D / BYO-AKS 私有预览细节或实测；有访问权限时再查看受限配套版本。

离线校验见 [VALIDATION_CN.md](VALIDATION_CN.md)。目录：[工具](tools/)、[统一声明](evidence/claims.json)、[可重算指标](evidence/measurements.json)、[图和图标台账](images/SOURCES.json)、[配置](config/.env.example)、[测试](tests/)。
