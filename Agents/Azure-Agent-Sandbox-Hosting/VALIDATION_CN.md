# 评审用的验证细节

[English](VALIDATION.md) · 客户操作步骤见 [README_CN.md](README_CN.md)

这一页写给需要核对数字的读者：每条路径怎么测的、每次实测的输入和结果、哪些没有测，以及如何离线重算全部数字。按 README 操作的客户不需要看这一页。

## 架构与指标口径

全文围绕两个问题组织：谁负责会话生命周期，会话的 micro-VM 跑在谁的节点上。四条路径的工作负载口径可比，但容器并不完全相同：A、B、C 用 agenthost workshop 的 reflection agent（[`oxcp/ainotes`，固定在 `905f2c2`](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost)），D 用私有预览示例 `agent-framework-agent-basic-responses-byoaks`。它们都调用 `gpt-5.4-mini`，并执行新建有状态会话和同会话追问。

- **A · ACA Sandboxes。** agent 的容器镜像被构建成沙箱磁盘镜像；每个实测会话一个沙箱，`cpu 1000m`、`memory 2048Mi`（M 档），Memory 模式下空闲 900 s 自动挂起，带 VNet 连接、挂在 `/app/app/data` 的私有 Blob 卷，以及按需激活的 8080 端口。控制面是 ARM（`Microsoft.App/sandboxGroups`），数据面是 `management.azuredevcompute.io`；调用方需要 *Container Apps SandboxGroup Data Owner* 角色。
- **B · Foundry hosted agent。** `azd ai agent init` 把 workshop 的 `azure.yaml` 绑定到已有项目，`azd deploy` 构建并发布一个版本。Learn 的描述是：每会话一个 VM 级隔离的沙箱，空闲期间保留 `$HOME` 和 `/files`，空闲超时 2–60 分钟（默认 15），30 天不活跃后删除会话。
- **C · AKS + agent-sandbox。** AKS 开启 Workload Identity 和 Blob CSI 驱动，一个 `KataVmIsolation` 的 Azure Linux 节点池（我们用 `Standard_D4s_v3`），agent-sandbox `v0.5.2`（`agents.x-k8s.io/v1beta1`，kind `Sandbox`），一个存状态的 Blob PVC，以及只允许一个来源地址访问的 LoadBalancer。挂起和恢复就是 `spec.operatingMode: Suspended | Running`；pod 被删除，文件状态从 Blob 恢复，不是内存快照恢复。Learn 写明的技术栈：Azure Linux 主机、Microsoft Hyper-V、Cloud Hypervisor VMM、Kata Containers。
- **D · BYO-AKS。** 最初只有一个 Kata 节点时，`foundry-system` 里读到 13 个运行时 pod：9 个固定的 Deployment 副本（router ×3，前面是 `:443` 的 LoadBalancer；controller ×2；identity proxy ×2；telemetry controller ×2）加 4 个 DaemonSet。每个 DaemonSet 会在每个匹配的 Kata 节点上增加一个 pod，所以总数随节点池增长。每个会话是独立的 Deployment、ReplicaSet 和 pod，`runtimeClassName: kata-vm-isolation`，放在按项目划分的命名空间里。

为什么这样计时：

| 指标 | 起点 | 终点 | 在哪里测 |
|---|---|---|---|
| 冷启动，A（分段相加） | SDK create 调用 → Running | 加上第一个 `/reflect` 回答耗时 | create 在客户端计时；回答在沙箱内计时 |
| 冷启动，B 和 D | 客户端 POST 一个新会话 | 读完整个响应体 | 客户端，经 Foundry 端点 |
| 冷启动，C（分段相加） | `kubectl apply` → Sandbox Ready | 加上 LoadBalancer 的第一个回答耗时 | 客户端，不经网关 |
| 后续回答 | 下一轮请求发出 | 收到回答 | 同冷启动 |
| 恢复，A（分段相加） | SDK resume 调用 → Running | 加上恢复后的第一个回答耗时 | resume 在客户端计时；回答在沙箱内计时 |
| 恢复，C | patch `operatingMode: Running` | LoadBalancer 返回第一个回答 | 客户端 |
| pod 丢失 | pod 被删除 | 替代 pod 变为 Ready（Kubernetes 时间戳） | 集群 |

A 的回答在沙箱内计时，是因为 2026-09-27 那次从操作员地址访问不到端口（端口的 IP 规则看到的是不断变化的 Azure 出口地址）。所以四条路径的网络链路并不相同，同一行比较的是实现方式，而不是网络。

A 的冷启动为 `create_total_s + first.guest_s`，恢复到回答为 `resume_total_s + first_after_resume.guest_s`：采用分段相加，不是客户端连续端到端计时。C 的冷启动也采用分段相加。A 和 C 的恢复到回答均不含停止和空闲等待。

## 脚本实际执行了什么

[`tools/aca_sandbox.py`](tools/aca_sandbox.py)（A），使用 `azure-containerapps-sandbox` 0.1.0b4：

| 步骤 | SDK 或 CLI 调用 |
|---|---|
| `vnet` | `SandboxGroupManagementClient.create_or_update_vnet_connection(group, ACA_VNET_CONNECTION, ACA_SUBNET_ID)` |
| `image` | `az acr login --expose-token`，然后 `SandboxGroupClient.begin_create_disk_image(ACA_IMAGE, registry_credentials=…)` |
| `volume` | `create_volume(ACA_VOLUME, type="AzureBlobByo", storage_container_resource_id=…, auth=UserAssigned identity)` |
| `create` | 按上面的配置调用 `_do_create_sandbox(...)`，再每 0.25 s 轮询 `get_sandbox` 直到 Running |
| `ask` | `sandbox.exec("python3 -c …")`，在沙箱内向 `http://127.0.0.1:8080/reflect` POST `{"query": …}` 并计时 |
| `stop`、`resume`、`delete` | `sandbox.stop()` / `resume()` 后轮询；`delete_sandbox(id)` |

[`tools/measure_aca.py`](tools/measure_aca.py) 复现 2026-10-09 的测法：`agent` 模式每次执行 create → 回答 → 回答 → stop → resume → 回答 → 回答 → delete，对话历史长度未持续增长时会失败；`create` 模式对每种配置（`builtin`、`image`、`vnet`、`port`、`vnetvolume`、`full`）测创建耗时，并记录恢复后的 guest uptime 是否大于停止前的值。uptime 本身不能证明进程内存的连续性；保留的 `process_survived` 字段只是这个观察的历史命名。

[`tools/measure_responses.py`](tools/measure_responses.py)（B 和 D）用 `https://ai.azure.com` 的 Azure CLI token 向 `AGENT_RESPONSES_URL` POST `{"input": …}`，后续轮次加上 `previous_response_id`。token 获取发生在请求计时之外，每个并发档位和恢复前都会刷新。带 `--observe` 时，仅接受唯一新增的 Ready Kata pod 作为候选；候选不唯一或读取失败都会失败，不能作为落地证据。观察时请使用没有其他新建会话的项目命名空间。`--recovery` 删除唯一新增的 Kata pod，等待删除完成后，沿用该会话的 response ID 追问。退出码 0 要求 cold、warm、并发和恢复中所有请求的检查都通过；跳过恢复、删除失败或遗忘状态会返回退出码 1。

[`tools/check_isolation.py`](tools/check_isolation.py)（C 和 D）读取 `kubectl get pods` 和 `kubectl get nodes`，要求 `runtimeClassName: kata-vm-isolation` 且 phase 为 Running；带 `--exec-uname` 时比较 pod 内的 `uname -r` 和所在节点的 `kernelVersion`。

workshop 的脚本（`module-01/setup.sh`、`module-03/prepare-agent-sandbox.sh`、`module-04/sandbox-deploy.sh`）按 `905f2c2` 原样使用，本仓库不复制它们。

## 实测记录

所有实测都在 East US 2，客户端是一台 Windows 工作站。A/B/C 记录明确写了 Azure CLI 2.88.0 和 azd 1.34.1；A 的 SDK 记录还写了 Python 3.13。原始控制台输出和 ID 留在作者的私有目录里；[`tools/export_evidence.py`](tools/export_evidence.py) 把它们投影到 [`evidence/runs/`](evidence/runs/)，记录了每个私有源文件的 SHA-256，并把节点、pod 和命名空间名替换成中性标签。

实测按 A→B→C→D 分组，同一路径内按日期从早到晚排列，与 README 表格的顺序一致。

### a-2026-09-27-setup：首次部署，以及用 HTTP 请求唤醒

**问题。** 在共享基础资源上部署 A 要多久？停止后的沙箱能保住进程吗？HTTP 请求能把它唤醒吗？

**输入。** 按顺序执行 `tools/aca_sandbox.py` 的各步，然后从沙箱内提问、停止、恢复、再停止，最后向按需端口发一个 HTTP 请求（[`evidence/runs/a-aca-2026-09-27/setup-and-wake.json`](evidence/runs/a-aca-2026-09-27/setup-and-wake.json)）。问题原文在表里。

**变量与固定项。** 在一个沙箱上执行一遍；使用[架构与指标口径](#架构与指标口径)中的 A 完整配置。

**结果。**

<!-- BEGIN GENERATED: a-setup -->
| 步骤 | 输入 | s | 回答或状态 |
|---|---|---:|---|
| sandbox group（Bicep） |  | 90 |  |
| VNet 连接 |  | 9.6 |  |
| 从 ACR 构建磁盘镜像 |  | 21.9 |  |
| Blob 卷 |  | 3.6 |  |
| 创建沙箱 |  | 21.5 | Running |
| 回答（沙箱内计时） | "where is Bangkok? one sentence" | 3.6 | Bangkok is the capital city of Thailand, in Southeast Asia. |
| 回答（沙箱内计时） | "What did I just ask you about? Reply in five words." | 1.1 | Bangkok’s location in the world. |
| 停止（内存快照） |  | 9.9 | Stopped |
| 恢复 |  | 1.5 | Running |
| 恢复后回答 |  | 12.5 | 2 |
| 下一轮回答 | "Name the city we discussed. One word." | 0.9 | Bangkok |
| 下一轮回答 | "Reply with OK." | 1.1 | OK |
| 再次停止 |  | 6.3 | Stopped |
| 向已停止的沙箱发 HTTP 请求 | "How many questions have I asked before this one? Number only." | 23.7 | 5 |
| 下一个 HTTP 请求 | "Reply with OK." | 2.5 | OK |

在沙箱内读到：vmm_type `cloudhypervisor`，内核 `6.12.8+`，1 个 vCPU。
<!-- END GENERATED: a-setup -->

**边界。**
- 只有一次。24 s 的唤醒时间是单次观察，没有用在“恢复”那一行；那一行用的是 2026-10-09 三次显式恢复的数据。
- 内存恢复后第一次对外调用用了 12.5 s，之后约 1 s；我们推测是 keep-alive 连接失效，但没有证实。
- 唤醒测试临时开了一个匿名端口窗口，因为 IP 规则放不进操作员地址；测完已删除端口。

### a-2026-10-09-agent：在 ACA Sandboxes 里跑完整 agent 流程，共 3 次

**问题。** 在 GA 服务上、用 workshop 的完整配置，新沙箱多久能给出回答，下一轮回答多快？挂起完成后，发起恢复 → 第一个回答要多久？

**输入。** 每次都用 agent 的磁盘镜像新建一个沙箱，并依次问下面这些原样的问题（[`evidence/runs/a-aca-2026-10-09/agent-runs.json`](evidence/runs/a-aca-2026-10-09/agent-runs.json)）：

<!-- BEGIN GENERATED: a-agent-input -->
> 配置：custom agent disk image + VNet + Blob volume + OnDemand port；cpu 1000m，memory 2048Mi，Memory 模式下空闲 900 s 自动挂起。
> 请求：POST http://127.0.0.1:8080/reflect {"query": <input>}, sent with sandbox exec from inside the sandbox。
> 第一个问题："Where is Bangkok? Answer in one short sentence."
> 之后每一轮："How many questions have I asked before this one? Number only."
<!-- END GENERATED: a-agent-input -->

**变量与固定项。** 只有轮次不同。服务（2026-09-23 起 GA）、SDK 0.1.0b4（当天 PyPI 上最新的公开版本，仍是 beta）、sandbox group、镜像和配置三次都相同。

**结果。**

<!-- BEGIN GENERATED: a-agent -->
单位为秒；最后一列是 agent 依次给出的回答。总耗时先在每次实测内相加，再取中位数；不能把各列独立统计的中位数相加。

| 次 | 创建 → Running | 第一个回答 | 冷启动 | 后续 | 停止 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 44.86 | 5.55 | 50.41 | 1.54 | 7.56 | 2.83 | 14.39 | 17.22 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 24.12 | 3.88 | 28.0 | 1.04 | 5.97 | 1.61 | 12.84 | 14.45 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 24.25 | 2.82 | 27.07 | 1.05 | 9.31 | 3.02 | 9.42 | 12.44 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| **中位数** | **24.25** | **3.88** | **28.0** | **1.05** | **7.56** | **2.83** | **12.84** | **14.45** |  |
<!-- END GENERATED: a-agent -->

三次的 `vmm_type` 都是 <!--v:a.vmm.value-->cloudhypervisor<!--/v-->，对话历史在停止和恢复之后继续增长的有 <!--v:a.state_kept.count-->3<!--/v--> 次（共 3 次）。没有遗留沙箱。

**边界。**
- 只有三次。第一次的创建用了 <!--v:a.create.max-->44.86<!--/v--> s，后两次约 24 s，所以冷启动范围的上限来自第一次。
- 创建耗时是笔记本上的 SDK 耗时，包含往返时延；回答耗时在沙箱内测，不含往返时延。
- 冷启动和恢复到回答是这两类独立时钟相加，不含停止和空闲等待，也不含客户端 exec 的额外开销。
- 文档写的亚秒级启动指的是预热池；这里的配置包含自定义镜像、VNet 连接和私有 Blob 卷。

### a-2026-10-09-attribution：配置里哪一项拖慢了创建

**问题。** 完整配置的创建耗时超过 20 s。原因是自定义镜像、VNet、端口，还是 Blob 卷？

**输入。** 一次只建一个沙箱，每种配置都执行创建、停止、恢复、删除。六种配置原样取自 [`evidence/runs/a-aca-2026-10-09/create-attribution.json`](evidence/runs/a-aca-2026-10-09/create-attribution.json)，列在下表前两列；`cpu`、`memory` 和挂起模式与上面相同。

**变量与固定项。** 在 agent 镜像的基础上每次多加一项配置。sandbox group、区域、SDK 和客户端不变；`builtin`、`image`、`full` 各跑 3 次，其余各 2 次。

**结果。**

<!-- BEGIN GENERATED: a-attr -->
| 配置 | 加了什么 | n | 创建 → Running，s | 中位数 |
|---|---|---:|---|---:|
| `builtin` | built-in ubuntu disk, SDK defaults, no VNet, volume or port | 3 | 5.74, 2.2, 4.68 | 4.68 |
| `image` | the agent's disk image from ACR, nothing else | 3 | 2.31, 1.6, 2.7 | 2.31 |
| `vnet` | image + VNet connection | 2 | 1.97, 2.52 | —（n=2） |
| `port` | image + OnDemand port with IP access control | 2 | 2.33, 2.3 | —（n=2） |
| `vnetvolume` | image + VNet connection + private Blob volume | 2 | 25.12, 15.79 | —（n=2） |
| `full` | image + VNet connection + private Blob volume + OnDemand port | 3 | 25.9, 22.2, 26.32 | 25.9 |

15 个沙箱中有 15 个恢复后的 guest uptime 大于停止前的值。这个观察本身不能证明进程内存的连续性。
<!-- END GENERATED: a-attr -->

**边界。**
- 每种配置只有两到三次。
- uptime 的比较只是观察，不能证明某个具体进程或其内存被保留。
- 没有测出“只挂卷、不连 VNet”的结果：对禁用公网的 Storage，请求一直没有返回，也没有出现沙箱；约 8 分钟后放弃，不在表里。
- 这只说明耗时出现在哪一步，不说明原因：Blob 挂载在创建路径上，服务内部的原因看不到。

### a-2026-10-09-late-mount：先启动、不带卷，之后再挂

**问题。** 既然卷是慢的那一步，能不能先不带卷启动，运行起来以后再挂？

**输入。** 用镜像和 VNet 连接创建、不带卷，然后对运行中的沙箱调用 `add_volume_mount` 挂同一个 Blob 卷，再列出挂载目录（[`evidence/runs/a-aca-2026-10-09/late-mount.json`](evidence/runs/a-aca-2026-10-09/late-mount.json)）。

**变量与固定项。** 只改了顺序（启动后挂卷，而不是创建时挂卷）；其他与 `vnetvolume` 相同。

**结果。**

<!-- BEGIN GENERATED: a-late -->
| 次 | 不带卷创建，s | 挂载 Blob 卷，s | 挂载后状态 |
|---|---:|---:|---|
| 1 | 7.2 | 20.29 | Running |
| 2 | 3.99 | 12.4 | Running |
<!-- END GENERATED: a-late -->

两次挂载后都能列出已有的状态文件，沙箱一直保持 Running。

**边界。** 只有两次。只在第一个回答不需要已保存状态时有用：挂载完成之前，agent 的状态目录是空的。

### b-2026-09-27：Foundry hosted agent，部署和两种模型路由

**问题。** B 从 `azd` 到 agent 能回答要多久？新会话和后续回答有多快？

**输入。** workshop module-02 的命令（README 第 5 步）：`azd ai agent init`，`MODEL_ROUTING=gateway` 时 `azd deploy`（版本 1），再用 `direct` 第二次 `azd deploy`（版本 2），每次部署后用 `azd ai agent invoke` 开一个新会话并追问两轮。实测记录里没有保留请求原文（[`evidence/runs/b-hosted-agent-2026-09-27/run.json`](evidence/runs/b-hosted-agent-2026-09-27/run.json)）。

**变量与固定项。** 两个版本只有模型路由不同；项目、模型和客户端相同。

**结果。**

<!-- BEGIN GENERATED: b-run -->
| 步骤 | 到模型的路由 | s |
|---|---|---:|
| `azd ai agent init` |  | 38.4 |
| `azd deploy`，版本 1 | gateway | 161 |
| 新会话 → 回答 | gateway | 9.43 |
| 后续回答 | gateway | 4.03, 5.54 |
| `azd deploy`，版本 2 | direct | 100 |
| 新会话 → 回答 | direct | 5.29 |
| 后续回答 | direct | 4.94, 4.38 |
<!-- END GENERATED: b-run -->

**边界。**
- 只有一次，每种路由两个后续样本：不足以说明 gateway 和 direct 有差异。
- direct 路由没有经过 Foundry 原生 AI gateway，因为那个门户步骤无法自动化。
- 没有测空闲后的恢复和会话丢失。

### c-2026-09-27：AKS + agent-sandbox，隔离、pod 丢失、挂起和恢复

**问题。** agent pod 是否跑在独立的 micro-VM 里？pod 丢失和一次挂起/恢复各要多久？

**输入。** workshop module-03 的脚本和清单（README 第 3 步），经 LoadBalancer 提问，然后 `kubectl delete pod`，再 patch `operatingMode: Suspended` 和 `Running`（[`evidence/runs/c-aks-agent-sandbox-2026-09-27/run.json`](evidence/runs/c-aks-agent-sandbox-2026-09-27/run.json)）。内核检查按 workshop 自己的验证方法：在 Sandbox pod 和一个不带 `runtimeClassName` 的 pod 里各执行 `uname -r`。请求原文没有保留。

**变量与固定项。** 一个 Sandbox，执行一遍；Kubernetes 1.35.8，一个 `Standard_D4s_v3` Kata 节点，agent-sandbox `v0.5.2`。

**结果。**

<!-- BEGIN GENERATED: c-run -->
| 步骤 | 观察结果 |
|---|---|
| `prepare-agent-sandbox.sh` | 600 s |
| Blob 私有链接 | 168 s |
| `kubectl apply` → Sandbox Ready | 44.9 s |
| 第一个、第二个回答 | 12.75 s, 1.96 s；记住上下文：true |
| Sandbox pod / runc pod 中的 `uname -r` | `6.6.137.mshv1-1.azl3` / `6.8.0-1067-azure` |
| 删除 pod → 重新就绪 | 23.4 s，状态恢复：true |
| `operatingMode: Suspended` → pod 消失 | 9.7 s；挂起期间健康检查 HTTP 503 |
| `operatingMode: Running` → Ready / LB 200 / 回答 | 17.1 / 26.1 / 34.2 s |
<!-- END GENERATED: c-run -->

**边界。**
- 只有一次。
- 用来对比的 runc pod 跑在 Ubuntu 系统节点池上，所以两个内核的差别既来自运行时，也来自节点。结论依据的是 Kata 内核的 `mshv` 后缀、runtime class，以及 Learn 写明的技术栈。
- 请求唤醒不是 agent-sandbox 的功能；恢复是手动 patch 的。

### d-2026-09-29-placement：BYO-AKS 的会话跑在哪里

**问题。** 绑定集群之后，会话是否真的跑在你的 Kata 节点池上？

**输入。** 用 `azd deploy` 部署私有预览示例 agent（版本 3）；开一个新会话，之后复查时再对话两轮（[`evidence/runs/d-byoaks-2026-09-29/placement.json`](evidence/runs/d-byoaks-2026-09-29/placement.json)）。第一个会话的请求原文没有保留；复查时的回答依次是“OK”和一个口令词。

**变量与固定项。** 本次之前唯一的改动是子网 NSG 规则；当天早些时候产品组已修复了该订阅的开通问题。

**结果。**

<!-- BEGIN GENERATED: d-placement -->
| 检查项 | 观察结果 |
|---|---|
| 修复前 IP flow verify | AzureCloud → :443 Deny; AzureCloud → :443 Deny; VNet → :443 Allow |
| 在子网 NSG 上新增的规则 | `Allow-AzureCloud-FoundryRouter-443`, 优先级 500, Inbound Allow Tcp AzureCloud → <router-ip>:443 |
| 修复后的第一个会话 | HTTP 200, "BYO-AKS-OK", 51 s 客户端耗时 |
| 它的会话 pod | runtimeClassName `kata-vm-isolation`, 属主 ReplicaSet, requests 500m / 1Gi, overhead 600Mi, init 容器 foundry-sandbox-agent, foundry-sandbox-init |
| 调度 → Ready | 35 s |
| 18:15 复查时的会话 pod | 3 个中 3 个为 Running 且是 `kata-vm-isolation` |
| 复查时的两轮对话 | "OK" → "HERON-1815" |
| Kata 节点池的节点内核 | `6.6.137.mshv2-2.azl3` |
<!-- END GENERATED: d-placement -->

**边界。**
- 2026-09-29 之前，我们更早的现场记录（私有仓库 `foundry-byo-aks-private-preview`）发现：集群停止时会话照样能回答，集群运行时也看不到会话 pod。原因有两个，先后是：产品侧对订阅的开通，然后是我们的子网 NSG。那份记录里关于会话落地的结论已被本次实测取代。
- 没有在会话 pod 内读 guest 内核；证据是节点内核和 `runtimeClassName`。
- 私有预览期间，会话落地方式可能变化。

### d-2026-09-29-perf：三个节点上的冷启动、后续回答和并发

**问题。** 在三节点的 Kata 节点池上，新会话和后续回答有多快？同时能容纳多少个会话？

**输入。** [`tools/measure_responses.py`](tools/measure_responses.py)（当时叫 `byoaks_bench.py`），使用下面这些原样输入（[`evidence/runs/d-byoaks-2026-09-29/bench.json`](evidence/runs/d-byoaks-2026-09-29/bench.json)、[`cold-early.json`](evidence/runs/d-byoaks-2026-09-29/cold-early.json)）：

<!-- BEGIN GENERATED: d-input -->
> 请求：POST <agent responses endpoint> {"input": <input>} with previous_response_id for follow-ups。
> 新会话："Reply with exactly COLD-{i}"（n = 1、2、3）
> 同一会话："Remember the code word LANTERN-7. Reply only OK."，然后 "What is the code word? Reply with the word only."，再 "Reply with exactly WARM-{i}"（n = 1、2）
> 并发时每个会话："Remember SCALE-{i}. Reply only OK."，然后 "What did I ask you to remember? Word only."
> 客户端：Windows laptop in China; direct REST POST to the agent's Responses endpoint with an Azure CLI token；记录到的往返时延 320–800 ms。
<!-- END GENERATED: d-input -->

**变量与固定项。** 冷启动的变量是节点上是否已有镜像；并发的变量是 4 和 8。节点池（3 × `Standard_D4s_v5`）、agent 版本、会话 pod 规格和客户端不变。

**结果。** 新会话：

<!-- BEGIN GENERATED: d-cold -->
| 开始（UTC） | 节点 | 回答 | 新会话 → 回答，s | pod 创建 → Ready，s |
|---|---|---|---:|---:|
| 15:43:00 | 尚无镜像 | COLD-1 | 41.9 | 未配对 |
| 15:43:47 | 尚无镜像 | COLD-2 | 40.6 | 未配对 |
| 15:44:32 | 镜像已缓存 | COLD-3 | 26.7 | 未配对 |
| 15:47:43 | 镜像已缓存 | COLD-1 | 24.01 | 10.0 |
| 15:48:15 | 镜像已缓存 | COLD-2 | 25.49 | 10.0 |
| 15:48:50 | 镜像已缓存 | COLD-3 | 26.16 | 10.0 |
| 15:49:37 | 镜像已缓存 | OK | 23.53 | 未配对 |

镜像已缓存：中位数 25.49 s（23.53–26.7；n=5）。节点上还没有镜像时的第一个 pod：41.9, 40.6 s（n=2）。本次所有会话 pod 从创建到 Ready：镜像已缓存 8, 9, 10, 10, 10, 10 s，新节点 25, 25 s。
<!-- END GENERATED: d-cold -->

同一会话的后续回答：

<!-- BEGIN GENERATED: d-warm -->
| 轮次 | 回答 | 客户端耗时，s |
|---|---|---:|
| 1（新会话） | OK | 23.53 |
| 2 | LANTERN-7 | 16.42 |
| 3 | WARM-1 | 11.17 |
| 4 | WARM-2 | 12.9 |

同一客户端不经 agent 直接调用同一个模型：中位数 2.69 s（1.82–5.07；n=5）。该基线使用 low reasoning effort，不是与 agent 配置完全匹配的基准测试。
<!-- END GENERATED: d-warm -->

并发：

<!-- BEGIN GENERATED: d-scale -->
| 并发 | 回答 | 拒绝 | 拒绝原因 | 拒绝耗时，s |
|---:|---|---|---|---:|
| 4 | 0/4 | 4/4 | `500 session_creation_failed` | 18.4–18.9 |
| 8 | 2/8 (24.14, 24.96 s) | 6/8 | `429 aks_quota_exceeded` | 68.03–80.51 |

开始拒绝时，3 个 `Standard_D4s_v5` 节点上共有 12 个会话 pod，单节点最多 5 个，没有节点还剩 500m CPU。
<!-- END GENERATED: d-scale -->

**边界。**
- 并发测试是在之前的会话已经占满节点池之后跑的，所以测到的是容量耗尽，而不是负载下的时延。在有空闲容量时的真实并发时延没有测。
- 压测在 cold + warm 前取一次 token，并在每个并发档位前刷新；token 获取发生在每个计时请求之外。响应自身从 `created_at` 到 `completed_at` 每轮 <!--v:d.agent_turn.range-->2–5<!--/v--> s（见运行清单）；后续回答剩下的时间花在 Foundry 端点、会话路由和客户端网络上。
- 冷启动样本合并了早先的三次和压测里的几次；节点上还没有镜像的两次单独列出。

### d-2026-09-29-recovery：删除一个会话 pod

**问题。** 删除会话 pod 后，替代 pod 多久就绪？另一个存活会话是否还能回答？本次没有测试被删会话自身的连续性。

**输入。** `kubectl delete pod` 删除一个存活会话的 pod，然后对一个存活会话发下一轮：“What did I ask you to remember? Word only.”（[`evidence/runs/d-byoaks-2026-09-29/recovery.json`](evidence/runs/d-byoaks-2026-09-29/recovery.json)）。

**变量与固定项。** 在压测用的同一个节点池上删除一个 pod。

**结果。**

<!-- BEGIN GENERATED: d-recovery -->
| 步骤 | 观察结果 |
|---|---|
| 删除会话 pod（UTC） | 16:04:16 |
| 替代 pod 就绪耗时 | 23.9 s |
| 另一个存活会话的后续回答 | HTTP 200, 14.8 s, "LANTERN-7"，该会话保留了自身状态：true |
<!-- END GENERATED: d-recovery -->

**边界。** 只删了一次。后续那一轮发给的是另一个存活会话，而不是被删 pod 的会话。它说明节点池继续服务，而且另一个会话保留了 `LANTERN-7`；它**不能**证明被删会话自身的连续性。

### 可以重算的数字

上面每个发布指标都来自 [`evidence/measurements.json`](evidence/measurements.json)。[`tools/build_measurements.py`](tools/build_measurements.py) 会先按 [`evidence/runs/INDEX.json`](evidence/runs/INDEX.json) 校验每个文件的 SHA-256，再从 `evidence/runs/` 重算全部指标。它还检查：A 每次的冷启动等于创建加第一个回答、恢复等于 resume 加回答；D 每个冷启动 pod 的创建到 Ready 与 Kubernetes 时间戳一致，且是 `kata-vm-isolation`；D 的后续会话和删除 pod 后使用的另一个存活会话都记住了 `LANTERN-7`。奇数个样本才给中位数，偶数个样本直接列出数值。

## 测试与离线校验

离线校验需要 Python 3.10 及以上，不需要访问 Azure：

```bash
python -m pip install -r requirements-dev.txt
python tools/build_measurements.py --check
python tools/render_docs.py --check
python tools/build_diagrams.py --check
python tools/validate_docs.py
python -m pytest -q
```

每条命令都退出 0、每个检查都打印 `PASS`，就算通过。

- **`build_measurements.py --check`** 重建全部指标；任何运行文件的哈希、字段或交叉校验出错，或者已提交的 `measurements.json` 与重建结果不同，都会失败。
- **`render_docs.py --check`** 四个页面里任何一张表或一个行内数字与重新渲染的结果不同，就会失败。
- **`build_diagrams.py --check`** 把两张架构图和每个图标与 [`images/SOURCES.json`](images/SOURCES.json) 比对。
- **`validate_docs.py`** 检查：README 的三个标题及顺序、VALIDATION 的标题、[`evidence/claims.json`](evidence/claims.json) 中的核心声明合同和统一实测顺序；每个实测小节按“问题、输入、变量与固定项、结果、边界”的顺序出现，且输入之前没有表格；每次实测都出现在 README 的结果表里，四份文档的实测顺序一致；链接和图片、图片宽度、中英文数字按生成区块和表格单元格对齐（不只比较全文总数）、操作步骤里的命令只用变量；以及私有内容防护（订阅和租户 ID、IPv4 地址、邮箱、内部资源名、已撤回的结论）。
- **`pytest`** 把上面这些检查作为测试运行，从原始数值重算中位数，用保存的 pod 和节点 JSON 测 `check_isolation.py`，在不装 Azure SDK 的情况下运行每个工具的 `--help`，并把每类防护各故意破坏一次（篡改运行文件、改一个数字、插入订阅 ID、删一个标题、把表格放到输入前面、打乱实测顺序），确认它会失败。
- 在线工具的 mock 测试还检查：并发追问失败、跳过恢复、读取或删除 pod 失败、历史丢失、参数无效和配置缺失都不能报成功；这些测试不会调用 Azure。

CI 在 Ubuntu 和 Windows 上用 Python 3.12 运行同样的检查（[工作流](https://github.com/david-xinyuwei/david-share/blob/master/.github/workflows/agent-sandbox-hosting-ci.yml)）。

在线检查需要 Azure 和 README 里创建的资源：`tools/aca_sandbox.py`、`tools/measure_aca.py`、`tools/measure_responses.py`，以及不带 `--pods-json` 的 `tools/check_isolation.py`。

这里没有测的：本仓库整理完成后，这些工具在全新订阅上的端到端运行（它们是实测脚本改为读配置后的版本；`--help` 和离线逻辑已测试），East US 2 以外的区域，以及用 Azure Cloud Shell 作为操作环境。

## 边界、目录与资料

**边界。**

- `LOCAL_MEASUREMENT`：A 的完整 agent 时延 n=3；B 和 C 各是一轮端到端实测；D 是一轮压测，其中镜像已缓存的冷启动 n=5、后续回答 n=3，会话落地和恢复各一次。依赖某个数字之前，请按 README 第 7 步重新测。
- `LOCAL_MEASUREMENT`：A、B、C 使用 workshop reflection agent，D 使用私有预览 sample。B 和 D 包含 Foundry 端点和位于中国的客户端（往返 <!--v:d.client_rtt.range-->320–800<!--/v--> ms）；A 的回答在沙箱内计时；C 在它的 LoadBalancer 处计时。表中每一行比较的是现场实现，不是相同容器或相同网络链路。
- `LOCAL_MEASUREMENT`：A 加上私有 Blob 卷后创建耗时为 <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s；只加镜像、VNet 或端口时为 <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s。
- `LOCAL_MEASUREMENT`：最早的 D 会话 pod 在创建至少 <!--v:d.idle.value-->87<!--/v--> 分钟后仍是 Running。这是 pod 的年龄，不是空闲时长：没有保留最后一次请求的时间戳。它不能证明配置的空闲超时，也不能证明期间始终空闲。
- `LOCAL_MEASUREMENT`：按示例的 500m CPU / 1 Gi 请求加 600 Mi Kata 开销，D 每个 `Standard_D4s_v5` 节点最多容纳 <!--v:d.capacity.per_node-->5<!--/v--> 个会话；同时运行系统 pod 的节点更少。
- `NOT_MEASURED`：B 的恢复和会话丢失；A 的宿主机丢失；服务端结束会话后 D 的行为；由 Foundry 治理的 agent 按任务调用 ACA 沙箱；有空闲容量时的并发时延；E2B 或其他第三方沙箱。
- `SOURCE_FACT`：ACA Sandboxes 从“prewarmed pools for near-instant availability”启动，恢复“with sub-second restore times”（[概览](https://learn.microsoft.com/azure/container-apps/sandboxes-overview)）；规格从 XS 0.25 vCPU / 0.5 GB 到 XL 4 vCPU / 8 GB。
- `SOURCE_FACT`：hosted agent 运行在“per-session VM-isolated sandboxes”中，保留 `$HOME` 和 `/files`；空闲超时 2–60 分钟，默认 15；30 天不活跃后删除会话（[hosted agents](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents)）。
- `SOURCE_FACT`：AKS Pod Sandboxing 由 Azure Linux 主机 + Microsoft Hyper-V + Cloud Hypervisor + Kata Containers 组成；节点 VM 必须是支持嵌套虚拟化的第 2 代规格；每个 pod VM 的内存等于 pod 的内存 limit（未设置时为 512 Mi），CPU limit 的小数部分向上取整到整数 vCPU（[pod sandboxing](https://learn.microsoft.com/azure/aks/use-pod-sandboxing)）。
- `SOURCE_FACT`：D 的保底费用 <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h，是两个 `Standard_D4s_v5` 节点各 0.192 USD/h 加 AKS Standard 层 0.10 USD/h，按 2026-09-21 查询的 East US 2 零售价；不含折扣和税。这不是三个 Kata 节点压测池的费用。
- D 的接入耗时（角色分配约 <!--v:d.role_assignments.value-->120<!--/v--> s、`azd provision` <!--v:d.azd_provision.value-->522<!--/v--> s、`azd deploy` <!--v:d.azd_deploy.value-->212<!--/v--> s）来自更早那份现场记录里的作者控制台；控制台原始输出没有保留。

**目录。**

- [`tools/`](tools/)：`aca_sandbox.py`、`measure_aca.py`、`measure_responses.py`、`check_isolation.py`（在线）；`export_evidence.py`（仅作者可运行）、`build_measurements.py`、`render_docs.py`、`build_diagrams.py`、`validate_docs.py`（离线）。
- [`evidence/claims.json`](evidence/claims.json)：A–D 的 canonical 名称、状态、归属、隔离边界、选型合同、字母映射和实测顺序；[`evidence/runs/`](evidence/runs/)：脱敏后的实测记录，带私有源文件的 SHA-256，以及 `INDEX.json`；[`evidence/measurements.json`](evidence/measurements.json)：所有发布的指标。
- [`images/`](images/)：架构图、它们的生成源 `tools/build_diagrams.py`、8 个 Azure 图标和 [`SOURCES.json`](images/SOURCES.json)。
- [`config/.env.example`](config/.env.example)：全部配置项；[`tests/`](tests/)：离线测试和测试数据。

**资料。**

- Microsoft Learn：[Container Apps Sandboxes 概览](https://learn.microsoft.com/azure/container-apps/sandboxes-overview) · [Foundry Agent Service 的 Hosted agents](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents) · [AKS Pod sandboxing](https://learn.microsoft.com/azure/aks/use-pod-sandboxing) · [Azure 架构图标](https://learn.microsoft.com/azure/architecture/icons/)
- [agent-sandbox v0.5.2](https://github.com/kubernetes-sigs/agent-sandbox/releases/tag/v0.5.2) · [agenthost workshop，固定在 905f2c2](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost)（没有许可证文件；只引用，不复制）
- [PyPI 上的 `azure-containerapps-sandbox`](https://pypi.org/project/azure-containerapps-sandbox/)（0.1.0b4）
- BYO-AKS 私有预览示例和接入指南：由产品组提供，需申请访问。
