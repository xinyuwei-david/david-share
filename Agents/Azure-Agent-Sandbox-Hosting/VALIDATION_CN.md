# 评审用的验证细节

[English](VALIDATION.md) · 客户操作步骤见 [README_CN.md](README_CN.md)

这一页写给需要核对数字的读者：每条路径怎么测的、每次实测的输入和结果、哪些没有测，以及如何离线重算全部数字。按 README 操作的客户不需要看这一页。

## 架构与指标口径

全文围绕两个问题组织：谁负责会话生命周期，会话的 micro-VM 跑在谁的节点上。五条路径的工作负载口径可比，但容器并不完全相同：A 和 C 使用 workshop 的 [module-03 reflection agent](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-03/agent-src/app/main.py)；B 使用另一个 [module-02 Agent Framework + ResponsesHostServer agent](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-02/agent-src/main.py)；D 用私有预览示例 `agent-framework-agent-basic-responses-byoaks`；E 用 Foundry Prompt Agent，它的函数工具在 E2B 沙箱里执行代码。它们都调用 `gpt-5.4-mini`，并执行新建有状态会话和同会话追问。

- **A · ACA Sandboxes。** agent 的容器镜像被构建成沙箱磁盘镜像；每个实测会话一个沙箱，`cpu 1000m`、`memory 2048Mi`（M 档），Memory 模式下空闲 900 s 自动挂起，带 VNet 连接、挂在 `/app/app/data` 的私有 Blob 卷，以及按需激活的 8080 端口。控制面是 ARM（`Microsoft.App/sandboxGroups`），数据面是 `management.azuredevcompute.io`；调用方需要 *Container Apps SandboxGroup Data Owner* 角色。
- **B · Foundry hosted agent。** `azd ai agent init` 把 workshop 的 `azure.yaml` 绑定到已有项目，`azd deploy` 构建并发布一个版本。Learn 的描述是：每会话一个 VM 级隔离的沙箱，空闲期间保留 `$HOME` 和 `/files`，空闲超时 2–60 分钟（默认 15），30 天不活跃后删除会话。
- **C · AKS + agent-sandbox。** AKS 开启 Workload Identity 和 Blob CSI 驱动，一个 `KataVmIsolation` 的 Azure Linux 节点池（我们用 `Standard_D4s_v3`），agent-sandbox `v0.5.2`（`agents.x-k8s.io/v1beta1`，kind `Sandbox`），一个存状态的 Blob PVC，以及只允许一个来源地址访问的 LoadBalancer。挂起和恢复就是 `spec.operatingMode: Suspended | Running`；pod 被删除，文件状态从 Blob 恢复，不是内存快照恢复。Learn 写明的技术栈：Azure Linux 主机、Microsoft Hyper-V、Cloud Hypervisor VMM、Kata Containers。
- **D · BYO-AKS。** 最初只有一个 Kata 节点时，`foundry-system` 里读到 13 个运行时 pod：9 个固定的 Deployment 副本（router ×3，前面是 `:443` 的 LoadBalancer；controller ×2；identity proxy ×2；telemetry controller ×2）加 4 个 DaemonSet。每个 DaemonSet 会在每个匹配的 Kata 节点上增加一个 pod，所以总数随节点池增长。每个会话是独立的 Deployment、ReplicaSet 和 pod，`runtimeClassName: kata-vm-isolation`，放在按项目划分的命名空间里。
- **E · E2B on Azure VM。** 一台开启嵌套虚拟化（`/dev/kvm`）的 Ubuntu 24.04 LTS Gen2 VM（`Standard_D4ds_v7`，`e-2026-10-18-ci-scale` 用的是 `Standard_D8ds_v7`）运行开源 E2B Embed Docker Compose 栈（`e2b-dev/runtime`，固定在 `588e5b9`）。对比表里，E2B Code Interpreter 的 `Sandbox.create()` 启动一个 `code-interpreter-v1` Firecracker `v1.14-0.2.0` micro-VM（2 vCPU、2048 MB、guest 内核 `6.1.177+`），工具用 `run_code` 在它的 Jupyter 内核里执行模型写的 Python；2026-10-17 之前的实测用的是 `Sandbox.create("base")`（2 vCPU、512 MB）和 `commands.run("python3")`，已被 2026-10-17 和 2026-10-18 的实测取代。`beta_pause` 把内存和磁盘快照（`vm_state` + `mem_file`）存到 VM 的系统盘上（E2B 的模板也在这块盘上），`Sandbox.connect` 恢复。Foundry Prompt Agent（`gpt-5.4-mini`）通过 `3000` 和 `3002` 端口上的函数工具调用沙箱。Embed [按设计只有一台机器](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/README.md#when-one-node-is-not-enough)，只提供明文 HTTP，没有持久数据卷；`e-2026-10-18-ci-multi` 把两套独立安装并排运行，`e-2026-10-15-multi` 曾用 `base` 这样做过。

为什么这样计时：

| 指标 | 起点 | 终点 | 在哪里测 |
|---|---|---|---|
| 冷启动，A（分段相加） | SDK create 调用 → Running | 加上第一个 `/reflect` 回答耗时 | create 在客户端计时；回答在沙箱内计时 |
| 冷启动，B 和 D | 客户端 POST 一个新会话 | 读完整个响应体 | 客户端，经 Foundry 端点 |
| 冷启动，C（分段相加） | `kubectl apply` → Sandbox Ready | 加上第一个回答耗时：在 LoadBalancer 处（2026-09-27）或在 pod 内（2026-10-18） | 客户端；回答经自己的 LoadBalancer，或用 `kubectl exec` 在 pod 内；模型推理仍经 APIM |
| 后续回答 | 下一轮请求发出 | 收到回答 | 同冷启动 |
| 恢复，A（分段相加） | SDK resume 调用 → Running | 加上恢复后的第一个回答耗时 | resume 在客户端计时；回答在沙箱内计时 |
| 恢复，C | patch `operatingMode: Running` | LoadBalancer 返回第一个回答（2026-09-27）；Ready 加 pod 内计时的回答（2026-10-18，分段相加） | 客户端 |
| pod 丢失 | pod 被删除 | 替代 pod 变为 Ready（Kubernetes 时间戳；C 的 2026-10-18 实测由客户端轮询） | 集群 |
| 并发，A 和 C（2026-10-18） | 同时发出 N 个创建或 apply | 每个沙箱的第二个回答，在沙箱内计时；最后一个删除后该档结束 | 创建、apply 和删除在客户端；回答在沙箱内 |
| B 和 D 会话接口：创建、停止 | `POST .../endpoint/sessions`、`POST .../sessions/{id}:stop` | 会话读到 `active`、`idle` | 客户端；含失败的尝试及其间隔 2 s |
| 冷启动，E 的 A 口径（分段相加） | `Sandbox.create`（`code-interpreter-v1`；2026-10-17 之前为 `base`）→ 返回 | 加上第一轮 Prompt Agent：第一次 `POST /openai/v1/responses` 到最终回答，含每次工具往返 | 工具进程内；E 的实测在 E2B 宿主 VM 上 |
| 冷启动，E 的 D 口径 | 创建对话 | 这一轮的最终回答，中间含 `Sandbox.create`（一次计时） | 同上 |
| 恢复，E（分段相加） | `Sandbox.connect` → 返回 | 加上恢复后的第一轮 | 同上 |
| 暂停，E | `beta_pause` | 返回 | 同上 |
| host 模式，E（2026-10-18） | `Sandbox.create`、一个 `run_code` 单元、`beta_pause` 或 `Sandbox.connect` | 返回 | E2B 宿主机本机上的工具，不调模型 |

E、`b-2026-10-15-sessions` 和 2026-10-18 各次实测的计时都用测量进程里的 `time.perf_counter()`；范围是最小值到最大值。A、E 和 C 的 2026-10-18 实测在停止、暂停或挂起结束后等 5 s 再恢复，这段等待不计入任何计时。E 的并发表给的是沙箱创建加第 1 轮，这是 2026-10-14 驱动的计时口径。

A 的回答在沙箱内计时，是因为 2026-09-27 那次从操作员地址访问不到端口（端口的 IP 规则看到的是不断变化的 Azure 出口地址）。所以四条路径的网络链路并不相同，同一行比较的是实现方式，而不是网络。

A 的冷启动为 `create_total_s + first.guest_s`，恢复到回答为 `resume_total_s + first_after_resume.guest_s`：采用分段相加，不是客户端连续端到端计时。C 的冷启动也采用分段相加。A 和 C 的恢复到回答均不含停止和空闲等待。

README 的对比表只用就绪类计时：创建到就绪、恢复到就绪、pod 或沙箱丢失后重新就绪。回答类计时只保留在本页：回答耗时主要反映模型、agent 和网络链路；像 `"Where is Bangkok?"` 这样不需要执行代码的问题，也不能说明代码在沙箱里执行。

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

[`tools/measure_aca.py`](tools/measure_aca.py) 复现 2026-10-09 的测法：`agent` 模式每次执行 create → 回答 → 回答 → stop → resume → 回答 → 回答 → delete，对话历史长度未持续增长时会失败（`--config novolume` 不挂 Blob 卷跑同样流程，见 `a-2026-10-18-novolume`）；每个请求先在沙箱内等到 agent 在 8080 端口接受连接，这段等待计入冷启动；`create` 模式对每种配置（`builtin`、`image`、`vnet`、`port`、`vnetvolume`、`full`、`novolume`）测创建耗时，并记录恢复后的 guest uptime 是否大于停止前的值。uptime 本身不能证明进程内存的连续性；保留的 `process_survived` 字段只是这个观察的历史命名。`scale` 模式按 `--scale` 的每一档同时创建这么多完整规格的沙箱；每个沙箱问 D 和 E 的两个并发问题，第二个回答说出自己的词才算回答，进入下一档之前删除全部沙箱（`a-2026-10-18-scale`）。

[`tools/measure_responses.py`](tools/measure_responses.py)（B 和 D）用 `https://ai.azure.com` 的 Azure CLI token 向 `AGENT_RESPONSES_URL` POST `{"input": …}`，后续轮次加上 `previous_response_id`。token 获取发生在请求计时之外，每个并发档位和恢复前都会刷新。带 `--observe` 时，仅接受唯一新增的 Ready Kata pod 作为候选；候选不唯一或读取失败都会失败，不能作为落地证据。观察时请使用没有其他新建会话的项目命名空间。`--recovery` 删除唯一新增的 Kata pod，等待删除完成后，沿用该会话的 response ID 追问。退出码 0 要求 cold、warm、并发和恢复中所有请求的检查都通过；跳过恢复、删除失败或遗忘状态会返回退出码 1。设置 `RESPONSES_AUTH_HEADER`（`Header-Name: value`，例如 API Management 密钥）时改发这个请求头而不是 token，`b-2026-10-18-perf` 就是这样；它的值不会打印或写入任何文件。

[`tools/check_isolation.py`](tools/check_isolation.py)（C 和 D）读取 `kubectl get pods` 和 `kubectl get nodes`，要求 `runtimeClassName: kata-vm-isolation` 且 phase 为 Running；带 `--exec-uname` 时比较 pod 内的 `uname -r` 和所在节点的 `kernelVersion`。

[`tools/measure_e2b.py`](tools/measure_e2b.py)（E）先创建自己的 Prompt Agent 版本（指令和工具定义写进每份输出），然后运行与 A 对齐的 3 次 `agent` 协议（`Sandbox.create` → 首答 → 追问 → `beta_pause` → 等 5 s → `Sandbox.connect` → 追问 → `kill`）、与 D 对齐的 `perf` 协议（冷启动 n=5、带 `LANTERN-7` 的同会话追问、并发 4 和 8，以及沙箱替换）、`scale`（把会话分到 `--hosts` 里列出的多套 Embed 安装上）和 `hostloss`（停掉一套安装的服务栈，记录其他安装和暂停的沙箱还能做什么），最后删除 agent 版本、所有对话和所有沙箱。任何阶段失败、丢失状态或留下资源都返回退出码 1。2026-10-15 的实测用的是它的 `036dc81…` 版，只有 host b 上的 agent 实测用的是更早的 `89255d0b…`。仓库里的版本只改了失败处理，问题、协议和计时不变：恢复后的追问丢了口令，或 `hostloss` 丢了本应保留的东西（另一台的存活会话和新会话、停掉那台重启后的暂停沙箱和新会话），或留下沙箱或对话时，返回退出码 1；创建中途失败时也会清理；`--hosts` 文件读不了时返回退出码 2。2026-10-14 的数字来自作者的私有驱动脚本，它复现了那个脚本的 agent、问题和计时。工具只用 E2B 的 Code Interpreter 执行模型写的 Python：用 `e2b-code-interpreter` 在 `code-interpreter-v1` 里创建沙箱，每次工具调用执行一个 `run_code` 单元（2026-10-17 用的是 `cbc96f2…` 版，2026-10-18 用的是 `aca40d1…` 版）；`--executor` 只接受 `code-interpreter`，这样 2026-10-17 记录下来的命令仍能运行。2026-10-17 之前的所有 E 实测都在 `base` 沙箱里用 `commands.run` 运行 `python3`（工具到提交 `7c9a13b` 为止的做法），这些实测已被取代。`host` 模式在 E2B 宿主机上运行、不调模型：一个沙箱读出自己的内核、vCPU 和内存，同时工具列出宿主机上的 Firecracker 进程；`--bench n` 计时 n 次创建 → 单元 → 暂停 → 恢复 → 单元的循环；`--reserve k` 逐个创建最多 k 个沙箱，每个之后读 `/proc/meminfo`，直到创建被拒（`e-2026-10-18-ci-host`、`e-2026-10-18-ci-scale`）。每份输出都记录执行方式、模板和 SDK 版本。

[`tools/e2b_agent.py`](tools/e2b_agent.py)（E）是 README 第 7 步的应用调用：用同一个工具创建临时 Prompt Agent 版本、一个对话和一个 `code-interpreter-v1` 沙箱，发一个问题，用 `run_code` 执行每次工具调用，打印回答，然后删除沙箱、对话和 agent 版本。退出码 0 表示已回答并清理完毕，1 表示失败或留下资源，2 表示缺少设置；2026-10-17 它在 host c 上回答成功，没有留下任何资源。

[`tools/measure_code_interpreter.py`](tools/measure_code_interpreter.py)（E）用 `e2b-code-interpreter` 2.11.0 对 E2B 官方的 `code-interpreter-v1` 模板跑 `e-2026-10-15-latest` 的 11 个步骤，任何一步失败或留下沙箱都会失败。

[`tools/measure_sessions.py`](tools/measure_sessions.py)（B 和 D）按 [Manage hosted agent sessions](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions) 执行显式会话生命周期：创建固定到某个 agent 版本的会话，向会话写入一个文件，回答两轮，`:stop` 并等到 `idle`，在同一个 `agent_session_id` 上再回答，读回文件，再回答一轮，删除会话。认证用 `https://ai.azure.com` 的 Azure CLI token 或 `SESSIONS_AUTH_HEADER` 里的请求头；每次 HTTP 尝试（含重试）都会记录；输出一份私有记录和一份去掉会话 ID 与端点的公开记录。退出码 0 要求每次都跑完、文件原样读回、问题计数持续增长，并且它创建的会话全部删除；2026-10-15 的实测四项都满足。`--resume-after idle` 把停止换成 `--idle-wait` 秒内不向会话发任何请求、再读一次状态，此时退出码 0 还要求该状态为 `idle` 或 `stopped`（`b-2026-10-18-idle`）；`--path D` 把一次实测标为 BYO-AKS（`d-2026-10-18-sessions`）。

[`tools/measure_c.py`](tools/measure_c.py)（C）复制 workshop 的 `agent-host` Sandbox 的 spec，换新名字并加标签 `app=agent-host-measure`，所以不碰 workshop 自己的 Sandbox 和 LoadBalancer。它计时 `kubectl apply` 到 Sandbox 及其 pod 都 Ready，用 `kubectl exec` 在 pod 内提问（与 A 在沙箱内的请求相同）；挂 Blob PVC 时还会删除 pod 并跑一次挂起/恢复；`--scale` 同时 apply N 个不挂卷的 Sandbox。退出码 0 要求每次的对话都持续增长、每个并发 Sandbox 都说出自己的词，并且测试 Sandbox 全部删除（`c-2026-10-18-perf`）。

workshop 的脚本（`module-01/setup.sh`、`module-03/prepare-agent-sandbox.sh`、`module-04/sandbox-deploy.sh`）按 `905f2c2` 原样使用，本仓库不复制它们。

## 实测记录

所有实测使用的 Azure 资源都在 East US 2。客户端是一台 Windows 工作站，只有 `b-2026-10-15-sessions` 和 2026-10-18 的各次实测（Japan East 的一台 Windows VM，从 WSL Ubuntu 发起）以及 E（测量工具直接在 E2B 宿主 VM 上运行，2026-10-15 是在 host a 上，2026-10-17 是在 host c 上，2026-10-18 是在 host c、a 和 d 上）例外。A/B/C 记录明确写了 Azure CLI 2.88.0 和 azd 1.34.1；A 的 SDK 记录还写了 Python 3.13。原始控制台输出和 ID 留在作者的私有目录里；[`tools/export_evidence.py`](tools/export_evidence.py) 把它们投影到 [`evidence/runs/`](evidence/runs/)，记录了每个私有源文件的 SHA-256，并把节点、pod 和命名空间名替换成中性标签。2026-10-18 的记录由一个私有脚本按同样的方式投影，每份记录都列出了它的 SHA-256。

实测按 A→B→C→D→E 分组，同一路径内按日期从早到晚排列，与 README 表格的顺序一致。

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
- 只有一次。24 s 的唤醒时间是单次观察，没有用在 README 的对比表里；对比表的“恢复到就绪”一行用的是 2026-10-18 不挂 Blob 卷的三次显式恢复。
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

**变量与固定项。** 只有轮次不同。服务（[2026-09-23 起 GA](https://techcommunity.microsoft.com/blog/appsonazureblog/azure-container-apps-sandboxes-now-generally-available/4559125)）、SDK 0.1.0b4（仍是 beta）、sandbox group、镜像和配置三次都相同。

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

### a-2026-10-18-scale：同时创建 4 个和 8 个 agent 沙箱

**问题。** 用完整规格同时创建 4 个、再同时创建 8 个 agent 沙箱，是否都能启动并回答？这种负载下创建要多久？

**输入。** [`tools/measure_aca.py`](tools/measure_aca.py) `scale --scale 4,8`，使用 `a-2026-10-09-agent` 的 sandbox group 和磁盘镜像（[`evidence/runs/a-aca-2026-10-18/scale.json`](evidence/runs/a-aca-2026-10-18/scale.json)）。每个沙箱先问 "Remember SCALE-<k>. Reply only OK."，再问 "What did I ask you to remember? Word only."（D 和 E 的并发问题），然后删除。

**变量与固定项。** 只改同时创建的数量：先 4 个，再 8 个，间隔 20 s。规格（自定义镜像、VNet、私有 Blob 卷、端口；1 vCPU、2 GiB）与 `a-2026-10-09-agent` 相同；客户端是 Japan East 的一台 Windows VM（WSL Ubuntu），回答在每个沙箱内计时。

**结果。**

<!-- BEGIN GENERATED: a-scale -->
| 并发 | 回答 | 创建 → Running，s | 沙箱内的读取轮次，s | 全部会话，s |
|---:|---|---:|---:|---:|
| 4 | 4/4 | 15.66–24.55 | 0.93–1.11 | 32.9 |
| 8 | 8/8 | 15.49–24.25 | 1.01–1.74 | 33.91 |

每个沙箱的第二个回答都是它自己的 `SCALE-<k>`；`vmm_type` 都是 `cloudhypervisor`；所有沙箱都已删除，没有遗留。全部会话：从第一个创建调用到最后一个沙箱删除。
<!-- END GENERATED: a-scale -->

**边界。**
- 每档只跑一次，只有一个客户端。8 个沙箱共 8 个活跃核，远低于默认配额 500，所以没有测到上限。
- 创建耗时是 Japan East 客户端上的 SDK 计时；2026-10-09 的实测是从中国的一台工作站发起的。
- 每个沙箱都在共享的 Blob 容器里写了一个以沙箱命名的小状态文件，与更早的 A 实测一样；这些文件留在了容器里。

### a-2026-10-18-novolume：不挂 Blob 卷跑完整 agent 流程，共 3 次

**问题。** 不挂私有 Blob 卷时，新沙箱多久给出第一个回答？只靠 Memory 快照，停止再恢复后对话还在不在？

**输入。** [`tools/measure_aca.py`](tools/measure_aca.py) `agent --runs 3 --config novolume`，使用 `a-2026-10-09-agent` 的 sandbox group 和磁盘镜像（[`evidence/runs/a-aca-2026-10-18/agent-novolume.json`](evidence/runs/a-aca-2026-10-18/agent-novolume.json)），输入原文如下：

<!-- BEGIN GENERATED: a-nv-input -->
> 配置：custom agent disk image + VNet + OnDemand port, no Blob volume；cpu 1000m，memory 2048Mi，Memory 模式下空闲 900 s 自动挂起。
> 请求：POST http://127.0.0.1:8080/reflect {"query": <input>}, sent with sandbox exec from inside the sandbox, after 127.0.0.1:8080 accepts a connection。
> 第一个问题："Where is Bangkok? Answer in one short sentence."
> 之后每一轮："How many questions have I asked before this one? Number only."
<!-- END GENERATED: a-nv-input -->

**变量与固定项。** 只去掉了 Blob 卷。镜像、VNet 连接、端口、`cpu`、`memory`、Memory 模式挂起、问题，以及 agent 的状态路径（`STATE_MOUNT_PATH`，现在是沙箱自己磁盘上的一个目录）都与 `a-2026-10-09-agent` 相同。客户端是 Japan East 的一台 Windows VM（WSL Ubuntu）；2026-10-09 的实测来自中国的一台工作站。

**结果。**

<!-- BEGIN GENERATED: a-nv -->
单位为秒；最后一列是 agent 依次给出的回答。总耗时先在每次实测内相加，再取中位数；不能把各列独立统计的中位数相加。等 agent 就绪：在沙箱内等到 agent 在 8080 端口接受连接，再发第一个请求，计入冷启动。恢复后回答前的同样等待每次都是 0 s，计入恢复 → 回答。

| 次 | 创建 → Running | 等 agent 就绪 | 第一个回答 | 冷启动 | 后续 | 停止 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.15 | 0.0 | 5.96 | 8.11 | 1.02 | 20.23 | 1.22 | 1.74 | 2.96 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 1.35 | 0.0 | 3.51 | 4.86 | 0.84 | 10.67 | 4.46 | 2.32 | 6.78 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 1.22 | 0.0 | 3.13 | 4.35 | 0.67 | 9.88 | 0.82 | 1.88 | 2.7 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **中位数** | **1.35** | **0.0** | **3.51** | **4.86** | **0.84** | **10.67** | **1.22** | **1.88** | **2.96** |  |
<!-- END GENERATED: a-nv -->

不挂卷时，停止再恢复后对话接着增长 <!--v:a.nv.state_kept.count-->3<!--/v-->/<!--v:a.nv.state_kept.n-->3<!--/v--> 次：agent 的状态文件留在沙箱磁盘上，Memory 快照把它保住了。与 `a-2026-10-09-agent` 相比，冷启动中位数从 <!--v:a.cold.median-->28.0<!--/v--> s 降到 <!--v:a.nv.cold.median-->4.86<!--/v--> s，恢复到回答中位数从 <!--v:a.recover.median-->14.45<!--/v--> s 降到 <!--v:a.nv.recover.median-->2.96<!--/v--> s。要比较能加起来的分项，每天各取总耗时为中位数的那一次：冷启动带卷是创建 <!--v:a.cold_mid.create-->24.12<!--/v--> s + 回答 <!--v:a.cold_mid.answer-->3.88<!--/v--> s，不挂卷是 <!--v:a.nv.cold_mid.create-->1.35<!--/v--> s + <!--v:a.nv.cold_mid.answer-->3.51<!--/v--> s，差别在创建；恢复到回答带卷是恢复 <!--v:a.recover_mid.resume-->1.61<!--/v--> s + 回答 <!--v:a.recover_mid.answer-->12.84<!--/v--> s，不挂卷是 <!--v:a.nv.recover_mid.resume-->1.22<!--/v--> s + <!--v:a.nv.recover_mid.answer-->1.74<!--/v--> s，差别在恢复后沙箱内的第一个回答。表中单独一列的中位数来自不同的实测，不能相加。

**边界。**
- 共三次，都是显式停止和恢复；没有等 900 s 空闲后的自动挂起。
- 记录只说明恢复后多出的时间在哪里（沙箱内的第一个回答），不说明原因；说它来自 Blob 挂载，是从这个差别推断的。
- 不挂卷时停止反而更慢（中位数 <!--v:a.nv.stop.median-->10.67<!--/v--> s 对 <!--v:a.stop.median-->7.56<!--/v--> s，其中一次 <!--v:a.nv.stop.max-->20.23<!--/v--> s）；记录看不出原因。
- 不挂卷时，Running 可能早于 agent 开始监听。在这三次之前，先用旧版工具（沙箱一读到 Running 就发第一个请求）跑了一次冒烟和第一组三次：<!--v:a.nv.earlier.count-->3<!--/v-->/<!--v:a.nv.earlier.n-->4<!--/v--> 次完成（冷启动 <!--v:a.nv.earlier_cold.range-->4.33–5.03<!--/v--> s，恢复到回答 <!--v:a.nv.earlier_recover.range-->2.57–10.67<!--/v--> s），另一次在创建 <!--v:a.nv.refused_after.value-->2.18<!--/v--> s 后发出的第一个请求被拒绝，因为 agent 还没有开始监听。现在工具会先在沙箱内等到 agent 接受连接，并把这段等待计入冷启动。这些都在记录的 `earlier_attempts` 里。
- 这时状态只存在这个沙箱里。删除沙箱后、或从别的沙箱读取状态，都没有测。
- 创建、停止和恢复是客户端上的 SDK 耗时，这次客户端在 Japan East，2026-10-09 在中国；两天的回答都在沙箱内计时。

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
- 这次没有测空闲后的恢复和会话丢失；显式停止后的恢复见 `b-2026-10-15-sessions`，空闲等待后的下一轮见 `b-2026-10-18-idle`。

### b-2026-10-15-sessions：Foundry hosted agent，显式会话生命周期，共 3 次

**问题。** B 创建会话、回答、停止，再在停止后回答，各要多久？停止前后，会话的文件和对话还在不在？

**输入。** [`tools/measure_sessions.py`](tools/measure_sessions.py) 调用 module-02 的 agent 版本 `2`（0.5 vCPU / 1 GiB，`gpt-5.4-mini`，`MODEL_ROUTING=direct`），共 3 次（[`evidence/runs/b-sessions-2026-10-15/sessions.json`](evidence/runs/b-sessions-2026-10-15/sessions.json)）。每次：`POST .../endpoint/sessions` 固定到版本 `2`；`PUT .../files/content` 写入一个标记文件；第一个问题 "Where is Bangkok? Answer in one short sentence."；之后每一轮都问 "How many questions have I asked before this one? Number only."，并带上 `previous_response_id`；`POST .../sessions/{id}:stop` 后轮询到 `idle`；在同一个 `agent_session_id` 上发下一轮；读回标记文件；再问一轮；`DELETE` 会话。

**变量与固定项。** 三次之间没有变量。固定项是 agent 版本、问题原文和客户端：Japan East 的一台 Windows VM（WSL Ubuntu）经 Azure API Management 网关调用 East US 2 的项目。客户端计时包含这段链路；服务端自己记录的 `created_at` → `completed_at` 单独列出。

**结果。**

<!-- BEGIN GENERATED: b-sessions -->
单位为秒，客户端计时；最后一列是 agent 依次给出的回答。

| 次 | 创建 → active | 第一个回答 | 后续 | 停止 → idle | 停止后回答 | 再下一轮 | 文件保留 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 6.69 | 5.62 | 4.59 | 2.9 | 12.35 | 4.6 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 6.87 | 4.79 | 4.81 | 3.05 | 12.02 | 4.11 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 6.51 | 5.85 | 4.42 | 7.71 | 11.66 | 3.88 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **中位数** | **6.69** | **5.62** | **4.59** | **3.05** | **12.02** | **4.11** |  |  |

共 35 次 HTTP 尝试：200 ×23、201 ×6、204 ×6；没有重试。
<!-- END GENERATED: b-sessions -->

服务端记录的每轮生成时间为 <!--v:b.sess.server.range-->1–4<!--/v--> s（整秒精度）；客户端计时的其余部分是网络链路、网关和平台。

**边界。**
- 只从一个客户端跑了三次：客户端计时包含 Japan East → East US 2 的往返和网关，不能与 2026-09-27 从中国发起的实测直接比较。
- 恢复是在显式 `:stop` 之后测的，不是空闲超时之后（那种情况见 `b-2026-10-18-idle`）。没有内存快照：agent 进程会重新启动，只有 `$HOME`、`/files` 和平台保存的对话会延续。
- 一次只有一个会话；并发见 `b-2026-10-18-perf`；没有测会话丢失。

### b-2026-10-18-perf：Foundry hosted agent，按 D 的口径：新会话、后续回答、4 和 8 并发

**问题。** 在 B 上，一次请求开新会话要多快？后续回答多快？同时开 4 个和 8 个新会话是否都能回答？

**输入。** [`tools/measure_responses.py`](tools/measure_responses.py) `--cold 5 --warm-turns 3 --scale 4,8`，即 `d-2026-09-29-perf` 的口径，经 `b-2026-10-15-sessions` 用的网关调用 module-02 agent 的 Responses 端点（[`evidence/runs/b-hosted-agent-2026-10-18/perf.json`](evidence/runs/b-hosted-agent-2026-10-18/perf.json)）。输入原文：

<!-- BEGIN GENERATED: b-perf-input -->
> 请求：POST <gateway>/agents/<agent>/endpoint/protocols/openai/responses {"input": <input>} with previous_response_id for follow-ups；认证：an API Management subscription key header (RESPONSES_AUTH_HEADER) instead of an Entra token; nothing to acquire inside or outside the timed requests。
> Agent：`maf-agent` 版本 2（0.5 vCPU、1Gi、`MODEL_ROUTING=direct`、`gpt-5.4-mini`）；no version pinned in the requests; the endpoint's only rule sends 100% to @latest, which read back as version 2 on 2026-10-18。
> 新会话："Reply with exactly COLD-{i}"（n = 1–5）
> 同一会话："Remember the code word LANTERN-7. Reply only OK."，然后 "What is the code word? Reply with the word only."，再 "Reply with exactly WARM-{i}"（n = 1、2）
> 并发时每个会话："Remember SCALE-{i}. Reply only OK."，然后 "What did I ask you to remember? Word only."（先 4 个，再 8 个会话）
> 客户端：Windows VM in Japan East (WSL Ubuntu) calling the Foundry project in East US 2 through an Azure API Management gateway (the path of b-2026-10-15-sessions)。
<!-- END GENERATED: b-perf-input -->

**变量与固定项。** 先测新会话，再测同一会话的后续回答，然后同时开 4 个和 8 个会话，各阶段间隔 20 s。agent 版本、网关和客户端固定。

**结果。**

<!-- BEGIN GENERATED: b-perf -->
新会话：

| 开始（UTC） | 回答 | 新会话 → 回答，s | 首字节，s |
|---|---|---:|---:|
| 03:49:28 | COLD-1 | 11.48 | 11.48 |
| 03:49:42 | COLD-2 | 11.03 | 11.03 |
| 03:49:57 | COLD-3 | 11.57 | 11.57 |
| 03:50:11 | COLD-4 | 10.98 | 10.98 |
| 03:50:25 | COLD-5 | 11.0 | 11.0 |

中位数 11.03 s（10.98–11.57；n=5）。

同一会话的后续回答：

| 轮次 | 回答 | 客户端耗时，s |
|---|---|---:|
| 1（新会话） | OK | 11.05 |
| 2 | LANTERN-7 | 3.94 |
| 3 | WARM-1 | 4.08 |
| 4 | WARM-2 | 3.88 |

后续第 2–4 轮：中位数 3.94 s（3.88–4.08；n=3）。

并发：

| 并发 | 回答 | 读回自己的词 | 写入轮次，s |
|---:|---|---|---:|
| 4 | 4/4 | 4/4 | 10.95–12.32 |
| 8 | 8/8 | 8/8 | 10.02–12.63 |

每个请求都返回 HTTP 200；工具对每个请求只发一次。
<!-- END GENERATED: b-perf -->

**边界。**
- 只从一个客户端跑了一轮：客户端计时包含 Japan East → East US 2 的往返和网关，所以可以和 `b-2026-10-15-sessions` 比，不能和 D 从中国笔记本发起的实测比。
- B 看不到 pod；平台把这 8 个会话放在哪里跑，观察不到。
- 同时 8 个会话说明 B 都回答了、没有拒绝，并没有测到上限。

### b-2026-10-18-idle：Foundry hosted agent，17 分钟没有请求后的下一轮

**问题。** 默认空闲超时是 15 分钟。一个会话超过这么久没有收到请求，下一轮要多久？会话的文件和对话还在不在？

**输入。** [`tools/measure_sessions.py`](tools/measure_sessions.py) `--resume-after idle --idle-wait 1020`，经网关调用版本 `2`（[`evidence/runs/b-sessions-2026-10-18/idle.json`](evidence/runs/b-sessions-2026-10-18/idle.json)），共 3 次：创建会话，写入标记文件，先问 "Where is Bangkok? Answer in one short sentence."，再问 "How many questions have I asked before this one? Number only."，然后 1020 s 不向会话发任何请求，读一次状态，再问两次第二个问题，读回标记文件，删除会话。对照组（[`idle-control.json`](evidence/runs/b-sessions-2026-10-18/idle-control.json)）相同，只是 `--idle-wait 300`，在第一组结束后才开始。

**变量与固定项。** 只改等待时长：1020 s，超过默认的 900 s；或 300 s，低于它。agent 版本（自身没有设置 `idle_timeout_seconds`，所以用默认值）、问题、网关和客户端固定。

**结果。**

<!-- BEGIN GENERATED: b-idle -->
单位为秒，客户端计时；服务端耗时是响应的 `completed_at` 减 `created_at`（整秒）；最后一列是 agent 依次给出的回答。

| 等待，s | 次 | 等待后的状态 | 等待前的后续 | 下一轮 | 该轮服务端耗时 | 再下一轮 | 文件保留 | 回答 |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1021.51 | 1 | `active` | 4.72 | 12.67 | 2 | 3.88 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 1021.48 | 2 | `idle` | 3.81 | 12.71 | 2 | 5.42 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 1021.22 | 3 | `idle` | 3.63 | 12.12 | 2 | 3.99 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **1021.48** | **中位数** |  | **3.81** | **12.67** | **2** | **3.99** |  |  |
| 301.75 | 1 | `active` | 3.7 | 5.76 | 3 | 4.16 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 301.45 | 2 | `active` | 3.85 | 5.16 | 2 | 4.4 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 301.47 | 3 | `active` | 4.21 | 4.43 | 2 | 3.89 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **301.47** | **中位数** |  | **3.85** | **5.16** | **2** | **4.16** |  |  |
<!-- END GENERATED: b-idle -->

等待后的状态见第三列。它只读了一次，不能说明平台何时、是否释放了算力。约 5 分钟后，下一轮中位数 <!--v:b.idle5.next.median-->5.16<!--/v--> s，比等待前的后续回答（<!--v:b.idle5.warm.median-->3.85<!--/v--> s）略长一些。约 17 分钟后为 <!--v:b.idle.next.median-->12.67<!--/v--> s，和显式 `:stop` 之后的恢复差不多（`b-2026-10-15-sessions` 中为 <!--v:b.sess.resume.median-->12.02<!--/v--> s），而服务端最多只把其中 <!--v:b.idle.next_server.max-->2<!--/v--> s 记为生成时间。所以多出的时间出现在 5 到 17 分钟之间。这与平台在 15 分钟的默认值之后释放会话算力一致，17 分钟那组里状态仍读到 `active` 的那一次也是如此。这是从耗时推断的，B 不暴露它的算力。

**边界。**
- 每种等待只从一个客户端跑三次；每次只在等待后读一次状态。
- B 不暴露算力，会话的算力是否已释放，只能从下一轮的耗时推断。
- 只要有一次在下一轮之前读到的状态仍是 `active`，`measure_sessions.py` 对这一组就返回退出码 1：按设计，退出码 0 要求那里是 `idle` 或 `stopped`。

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
- 只有一次；每一步各三次、在 pod 内计时回答的实测见 `c-2026-10-18-perf`。
- 用来对比的 runc pod 跑在 Ubuntu 系统节点池上，所以两个内核的差别既来自运行时，也来自节点。结论依据的是 Kata 内核的 `mshv` 后缀、runtime class，以及 Learn 写明的技术栈。
- 请求唤醒不是 agent-sandbox 的功能；恢复是手动 patch 的。

### c-2026-10-18-perf：AKS + agent-sandbox，不挂卷和挂 Blob PVC 的冷启动、pod 丢失、挂起/恢复、4 和 8 并发

**问题。** 不挂外部卷时，以及挂 workshop 的 Blob PVC 时，Sandbox 启动并回答各要多快？三次实测中，pod 丢失和一次挂起/恢复各要多久？一次能放下几个 Sandbox？

**输入。** [`tools/measure_c.py`](tools/measure_c.py) `--cold 3 --pvc 3 --scale 4,8`，针对 workshop 的集群（[`evidence/runs/c-aks-agent-sandbox-2026-10-18/perf.json`](evidence/runs/c-aks-agent-sandbox-2026-10-18/perf.json)）。每个测试 Sandbox 都复制 workshop 的 `agent-host` Sandbox 的 spec，回答通过 `kubectl exec` 在 pod 内向 `/reflect` POST：

<!-- BEGIN GENERATED: c18-input -->
> 集群：the cluster of c-2026-09-27, started for this run; the workshop's own agent-host Sandbox kept running on the Kata node throughout；Kubernetes 1.35.8，Free 层级；系统节点池 1 x standard_d4ads_v7 Ubuntu；Kata 节点池 Standard_D4s_v3 AzureLinux KataVmIsolation, autoscale 1-10, 1 node at start；agent-sandbox v0.5.2 (controller image read back 2026-10-18)。
> 每个测试 Sandbox 复制命名空间 `agent` 里 workshop 的 `agent-host` Sandbox 的 spec，换一个新名字并加标签 `app=agent-host-measure`：镜像 `<registry>.azurecr.io/agent-host:latest`，运行时类 `kata-vm-isolation`，requests 500m CPU、512Mi，limits 2000m CPU、2Gi。`AGENT_ID` 取自 pod 名，所以每个副本把对话存在自己的状态文件里。
> 请求：POST http://127.0.0.1:8080/reflect {"query": <input>}, sent with kubectl exec python3 inside the agent container。
> 第一个问题："Where is Bangkok? Answer in one short sentence."
> 之后每一轮："How many questions have I asked before this one? Number only."；挂 PVC 时在删除 pod 后、挂起/恢复后各再问一次
> 并发时每个 Sandbox（不挂卷）："Remember SCALE-{k}. Reply only OK."，然后 "What did I ask you to remember? Word only."（先 4 个，再 8 个 Sandbox）
> 客户端：Windows VM in Japan East (WSL Ubuntu); kubectl against the cluster in East US 2; answers timed inside the pod with kubectl exec。
<!-- END GENERATED: c18-input -->

**变量与固定项。** 挂或不挂 Blob PVC（不挂时，从副本中去掉 `persistentVolumeClaim` 卷和它的挂载），以及同时 4 个或 8 个 Sandbox，每次之间间隔 20 s。集群、镜像、agent 和客户端固定。

**结果。** 冷启动和后续回答：

<!-- BEGIN GENERATED: c18-runs -->
| Sandbox | 卷 | `kubectl apply` → Ready，s | 第一个回答，s | 冷启动合计，s | 后续，s | 对话长度 |
|---|---|---:|---:|---:|---:|---|
| cold-1 | 无 | 18.9 | 8.44 | 27.34 | 1.08 | 1, 2 |
| cold-2 | 无 | 15.32 | 8.5 | 23.82 | 1.0 | 1, 2 |
| cold-3 | 无 | 16.45 | 7.82 | 24.27 | 1.03 | 1, 2 |
| pvc-1 | Blob PVC | 16.81 | 7.38 | 24.19 | 36.73 | 1, 2, 3, 4 |
| pvc-2 | Blob PVC | 16.26 | 10.42 | 26.68 | 1.78 | 1, 2, 3, 4 |
| pvc-3 | Blob PVC | 16.46 | 7.58 | 24.04 | 1.22 | 1, 2, 3, 4 |

不挂卷：apply → Ready 中位数 16.45 s（15.32–18.9；n=3），第一个回答 中位数 8.44 s（7.82–8.5；n=3），冷启动合计 中位数 24.27 s（23.82–27.34；n=3）。挂 Blob PVC：apply → Ready 中位数 16.46 s（16.26–16.81；n=3），第一个回答 中位数 7.58 s（7.38–10.42；n=3），冷启动合计 中位数 24.19 s（24.04–26.68；n=3）。挂 PVC 时的后续回答：36.73, 1.78, 1.22 s（n=3）；记录里看不出慢的那一次时间花在哪里。回答在 pod 内计时；外层的 `kubectl exec` 往返每次在客户端多花 2.53–3.21 s，不计入。
<!-- END GENERATED: c18-runs -->

挂 Blob PVC 时的 pod 丢失和挂起/恢复：

<!-- BEGIN GENERATED: c18-pvc -->
| Sandbox | 删除 pod → Ready，s | 回答，s | Suspended → pod 消失，s | Running → Ready，s | 回答，s | 恢复 → 回答，s |
|---|---:|---:|---:|---:|---:|---:|
| pvc-1 | 25.24 | 7.72 | 9.24 | 15.87 | 7.47 | 23.34 |
| pvc-2 | 24.79 | 7.66 | 9.1 | 15.93 | 6.52 | 22.45 |
| pvc-3 | 23.25 | 7.11 | 9.45 | 17.28 | 6.95 | 24.23 |
| **中位数** | **24.79** |  | **9.24** | **15.93** |  | **23.34** |

挂起期间，每次 Sandbox 的 Ready 条件都是 `False`（`SandboxSuspended`）。删除 pod 后、挂起/恢复后对话长度都接着增长（见上表最后一列），说明状态每次都从 Blob PVC 恢复。恢复 → 回答是 Running patch 到 Ready 再加 pod 内的回答。
<!-- END GENERATED: c18-pvc -->

不挂卷时的并发：

<!-- BEGIN GENERATED: c18-scale -->
| 并发 | 回答 | Apply → Ready，s | 每个 Kata 节点上的 Sandbox | 全部 Sandbox，s |
|---:|---|---:|---|---:|
| 4 | 4/4 | 18.32–19.96 | node-1：4 | 42.15 |
| 8 | 8/8 | 18.52–141.79 | node-1：6、node-2：2 | 159.28 |

开始时只有 1 个 Kata 节点，workshop 自己的 Sandbox 也在上面。8 个同时开时，6 个测试 Sandbox 放进了这个节点（每个 requests 500m CPU）；另外 2 个跑在集群自动扩缩容在测试中新加的节点上，141.55–141.79 s 后才就绪。结束时 Kata 节点数：2。每个 Sandbox 都回答了自己的 `SCALE-<k>`，并已删除。
<!-- END GENERATED: c18-scale -->

**边界。**
- 每种情况三次、每个并发档一次，只有一个客户端。
- `c-2026-09-27` 在 LoadBalancer 处计时回答，而且那次 apply 是该集群上的第一次；这几次在 pod 内计时回答，节点上已有镜像，所以两组的计时位置和缓存都不同。
- Blob PVC 已经绑定好；挂载它在 apply 时测不出额外耗时。新建并绑定 PVC 的耗时没有测。
- 每个副本都在共享的 Blob 容器里写了自己的状态文件（以 pod 命名）；三次挂 PVC 实测的文件留在了容器里，更早实测的文件也一样。

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
- 压测在 cold + warm 前取一次 token，并在每个并发档位前刷新；token 获取发生在每个计时请求之外。响应自身从 `created_at` 到 `completed_at` 每轮 <!--v:d.agent_turn.range-->2–5<!--/v--> s（见运行清单）。客户端总耗时中其余部分没有打点，现有记录不能分别归因到端点、会话路由、agent 执行或客户端网络。
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

### d-2026-10-18-sessions：BYO-AKS，显式会话生命周期，共 3 次

**问题。** 会话接口的停止和恢复在 BYO-AKS 上能不能用？各要多久？会话的文件和对话能否跨过停止和恢复保留下来？

**输入。** [`tools/measure_sessions.py`](tools/measure_sessions.py) `--path D`，在项目端点上调用私有预览示例 agent 的版本 `3`，用客户端 VM 的托管身份认证（[`evidence/runs/d-byoaks-2026-10-18/sessions.json`](evidence/runs/d-byoaks-2026-10-18/sessions.json)）。步骤和问题与 `b-2026-10-15-sessions` 相同："Where is Bangkok? Answer in one short sentence."，之后每一轮都问 "How many questions have I asked before this one? Number only."；`:stop` 之前写入标记文件，恢复那一轮之后读回。

**变量与固定项。** 三次之间没有变量。实测期间 Kata 节点池有两个 `Standard_D4s_v5` 节点（见边界）。

**结果。**

<!-- BEGIN GENERATED: d-sessions -->
单位为秒，客户端计时；最后一列是 agent 依次给出的回答。

| 次 | 创建 → active | 第一个回答 | 后续 | 停止 → idle | 停止后回答 | 再下一轮 | 文件保留 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 11.51 | 11.01 | 10.05 | 23.55 | 19.78 | 10.05 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 9.18 | 10.71 | 11.05 | 19.54 | 22.4 | 10.95 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 3 | 9.32 | 11.39 | 10.66 | 22.72 | 19.14 | 10.71 | 是 | Bangkok is the capital city of Thailand in Southeast Asia. / 2 / 3 / 4 |
| **中位数** | **9.32** | **11.01** | **10.66** | **22.72** | **19.78** | **10.71** |  |  |

共 35 次 HTTP 尝试：200 ×23、201 ×6、204 ×6；没有重试。

三次实测中的会话 pod：新出现 6 个，每个会话 2 个，每个出现后 3–4 s 进入 Running（变化时的 pod 年龄）。同一会话的两个 pod 都结束了（Terminating，然后 Completed）；年龄与 `:stop` 结束第一个、删除会话结束第二个相符。`kubectl get pods -w` 只记录顺序和年龄，没有时间戳。
<!-- END GENERATED: d-sessions -->

服务端记录的每轮生成时间为 <!--v:d.sess.server.range-->1–3<!--/v--> s；客户端计时的其余部分是网络链路和端点，停止后的那一轮还包括启动新的会话 pod。

**边界。**
- 只从一个客户端跑了三次，它从 Japan East 直接调用 East US 2 的项目端点。
- 节点池只有一个节点时，每次创建会话都被拒绝，返回 `429 aks_quota_exceeded`：集群启动后恢复的 2026-09-29 会话 Deployment 占满了这个节点的 CPU requests。我们为这几次实测加了第二个节点，测完后撤掉；旧会话保留未动。
- 私有预览：BYO-AKS 上会话接口的行为可能变化。

### e-2026-10-14-agent：Foundry Prompt Agent + Azure VM 上的 E2B Firecracker 沙箱，共 3 次

**问题。** 当 E2B Embed 运行在开启 `/dev/kvm` 的 Azure Gen2 VM 上时，Firecracker 沙箱的创建、暂停和恢复有多快？在与 A 相同的 3 次完整协议下，Foundry Prompt Agent 跨暂停和恢复的回答耗时是多少？

**输入。** 每次新建一个 E2B `base` 沙箱和一个 Foundry 会话（East US 2 的 `gpt-5.4-mini`），并依次询问下面这些原样问题（[`evidence/runs/e-e2b-2026-10-14/agent-runs.json`](evidence/runs/e-e2b-2026-10-14/agent-runs.json)）：

<!-- BEGIN GENERATED: e-agent-input -->
> 宿主机 VM：Standard_D4ds_v7；E2B `base` 模板（2 vCPU，512 MB，经 `beta_pause` 保存 Firecracker 内存+磁盘快照）。
> 版本：E2B Python SDK `e2b 2.46.0`；E2B Embed 运行时 `588e5b9`。
> 请求：POST <Foundry project>/openai/v1/responses with Prompt Agent calling function tool run_in_e2b_sandbox on E2B API :3000/:3002。
> Agent 指令："You are a concise assistant with access to `run_in_e2b_sandbox`. For EVERY user message, first call `run_in_e2b_sandbox` once to inspect or update files in /home/user, then answer concisely as instructed. When asked for a number only or word only, reply with that exact token only."
> 唯一的函数工具 `run_in_e2b_sandbox(python_code)`："Execute Python 3 code inside the session's isolated E2B Firecracker microVM sandbox on Azure."
> 第一个问题："Where is Bangkok? Answer in one short sentence."
> 之后每一轮："How many questions have I asked before this one? Number only."
<!-- END GENERATED: e-agent-input -->

**变量与固定项。** 3 次 agent 实测之间只有轮次不同；20 次微基准在宿主机 VM 上直接访问 `localhost:3000` 和 `localhost:3002`。

**结果。**

<!-- BEGIN GENERATED: e-agent -->
| 次 | 创建 → Running | 第一个回答 | 冷启动 | 后续 | 暂停 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.14 | 4.12 | 4.26 | 2.95 | 0.13 | 0.08 | 6.4 | 6.48 | Bangkok is the capital of Thailand in Southeast Asia. / 2 / 3 / 4 |
| 2 | 0.07 | 2.82 | 2.89 | 2.83 | 0.11 | 0.11 | 12.71 | 12.82 | Bangkok is the capital of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 3 | 0.06 | 8.38 | 8.44 | 3.92 | 0.15 | 0.09 | 4.64 | 4.73 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| **中位数** | **0.07** | **4.12** | **4.26** | **2.95** | **0.13** | **0.09** | **6.4** | **6.48** |  |

本机 20 次沙箱微基准（零跨网 RTT）：创建 0.051–0.089 s（n=20），热命令 0.007–0.01 s（n=20），暂停 0.051–1.81 s（n=20），恢复 0.086–0.62 s（n=20）。VMM `firecracker` `v1.14-0.2.0` 来自宿主机进程列表；guest 内核 `6.1.177+` 来自沙箱内的 `uname -r`；宿主机内核 `6.17.0-1022-azure`。
<!-- END GENERATED: e-agent -->

宿主机进程列表里沙箱背后的 VMM 进程是 <!--v:e.vmm.value-->firecracker<!--/v-->，沙箱内的对话历史在暂停和恢复后继续增长的有 <!--v:e.state_kept.count-->3<!--/v--> 次（共 3 次）。VM 首次部署耗时 <!--v:e.setup.value-->177<!--/v--> s。

**边界。**
- 已被取代：这次用的是早期的执行方式（`base` 模板，用 `commands.run` 跑 `python3`）；`e-2026-10-17-ci-agent` 和 `e-2026-10-18-ci-multi` 用 `code-interpreter-v1` 重做了 agent 实测，`e-2026-10-18-ci-host` 重做了本机循环。
- 单台 East US 2 `Standard_D4ds_v7` VM 上的三次 agent 实测加 20 次本机微基准。
- Prompt Agent 运行在 Foundry 中并通过网络把 E2B 沙箱作为函数工具调用，而 A 的 reflection agent 直接运行在 ACA 沙箱内部。

### e-2026-10-14-perf：E2B 上的冷启动、后续回答、并发与沙箱替换

**问题。** 在与 D 相同的压测协议（`COLD-{i}`、`LANTERN-7` 多轮追问、4 和 8 并发会话以及沙箱删除替换）下，运行 E2B 的单台 `Standard_D4ds_v7` VM 表现如何？

**输入。** 与 D 对齐的压测脚本及下面这些原样输入（[`evidence/runs/e-e2b-2026-10-14/perf.json`](evidence/runs/e-e2b-2026-10-14/perf.json)）：

<!-- BEGIN GENERATED: e-perf-input -->
> 请求：POST <Foundry project>/openai/v1/responses {"input": <input>} with Prompt Agent calling E2B sandbox tool；agent 指令和工具与 `e-2026-10-14-agent` 相同。
> 新会话："Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}"（n = 1、2、3、4、5）
> 同一会话："Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only."，再 "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1"，再 "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2"
> 并发时每个会话："Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."
> 替换：一个存活会话先发 "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."；另一个沙箱被删除并替换后，再发 "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only."
> 驱动：作者的私有脚本 `run_abcd_matched_benchmarks.py`（SHA-256 `8e8f6fc5b7b6…` 记在记录里）；`tools/measure_e2b.py perf` 用同样的 agent、问题和计时。
<!-- END GENERATED: e-perf-input -->

**变量与固定项。** 冷启动重复 5 个新会话；并发在 1 台 `Standard_D4ds_v7` VM 上分别测试 4 和 8 个并发会话。

**结果。** 新会话：

<!-- BEGIN GENERATED: e-cold -->
| 开始（UTC） | 回答 | 沙箱创建 → Ready，s | Agent 轮次，s | 新会话 → 回答，s |
|---|---|---:|---:|---:|
| 06:14:58 | COLD-1 | 0.056 | 7.75 | 7.94 |
| 06:15:06 | COLD-2 | 0.068 | 2.28 | 2.54 |
| 06:15:08 | COLD-3 | 0.068 | 3.44 | 3.6 |
| 06:15:12 | COLD-4 | 0.057 | 3.52 | 3.65 |
| 06:15:16 | COLD-5 | 0.066 | 3.19 | 3.34 |

5 次冷启动汇总：沙箱创建 → Ready 中位数 0.066 s（0.056–0.068；n=5），Prompt Agent 轮次 中位数 3.44 s（2.28–7.75；n=5），端到端新会话 → 回答 中位数 3.6 s（2.54–7.94；n=5）。
<!-- END GENERATED: e-cold -->

同一会话的后续回答：

<!-- BEGIN GENERATED: e-warm -->
| 轮次 | 回答 | 沙箱工具耗时，s | 轮次耗时，s |
|---|---|---:|---:|
| 1（新会话；含沙箱创建） | OK | 0.14 | 2.76 |
| 2 | LANTERN-7 | 0.033 | 5.03 |
| 3 | WARM-1 | 0.029 | 3.19 |
| 4 | WARM-2 | 0.028 | 3.39 |
<!-- END GENERATED: e-warm -->

并发与沙箱替换：

<!-- BEGIN GENERATED: e-scale -->
| 并发 | 回答 | 拒绝 | 沙箱创建 + 第 1 轮，s |
|---:|---|---|---|
| 4 | 4/4 | 0/4 | 3.9, 3.12, 3.9, 3.17 |
| 8 | 8/8 | 0/8 | 4.59, 2.48, 4.63, 6.94, 5.4, 3.7, 7.34, 3.82 |

`kill` 删除沙箱后的替换：新 Firecracker 沙箱在 0.1 s 内就绪；另一个存活会话的追问 HTTP 200，2.61 s，"LANTERN-7"，保留状态：true。本次不公布同时运行的 micro-VM 数：驱动用 `pgrep -fc` 计数，把它自己的 shell 也算了进去。
<!-- END GENERATED: e-scale -->

**边界。**
- 已被取代：这次用的是早期的执行方式（`base` 模板，用 `commands.run` 跑 `python3`）；`e-2026-10-17-ci-perf` 和 `e-2026-10-18-ci-scale` 用 `code-interpreter-v1` 重做了这个口径。
- 单台 VM 部署（`Standard_D4ds_v7`，4 vCPU / 16 GiB 内存）。8 并发时，其中一个追问轮次（`SCALE-8`）用了 78.99 s，全部 8/8 会话都回答并记住了各自的口令；记录里没有这段时间花在哪里。
- 与 D 的恢复测试一样，`kill` 删除沙箱后的追问发给了另一个存活会话（`LANTERN-7`）：宿主机继续服务，替代沙箱在 <!--v:e.pod_lost.value-->0.1<!--/v--> s 内就绪。它不能证明被删沙箱的连续性。
- 这些数字来自作者的私有驱动脚本；`tools/measure_e2b.py perf` 复现了它的 agent、问题和计时，但产生这份记录的不是它。

### e-2026-10-15-latest：最新 E2B SDK，以及 E2B 官方 code-interpreter 模板

**问题。** 最新的 E2B Python SDK（`2.56.0`，2026-10-13 发布）在这套 Embed 上还能跑通实测协议吗？E2B 自己的 code-interpreter 模板（`e2b-code-interpreter 2.11.0`）能建起来并正常工作吗？

**输入。** 在 `e-2026-10-15-multi` 的 host a 上（[`evidence/runs/e-e2b-2026-10-15/latest.json`](evidence/runs/e-e2b-2026-10-15/latest.json)）：

<!-- BEGIN GENERATED: e-latest-input -->
> SDK 检查：`tools/measure_e2b.py agent --runs 3`，`e2b 2.56.0`（Python 3.12.3），工具 SHA-256 `036dc81db607…`；"Where is Bangkok? Answer in one short sentence."，然后 3 次 "How many questions have I asked before this one? Number only."，第二轮后暂停 5 s。
> code-interpreter：模板 `code-interpreter-v1`（2 vCPU，2048 MB），来自 [`e2b-dev/code-interpreter@162a2a2`](https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template)，用其原样的 `template/build_prod.py` 构建（构建环境 `e2b 2.32.0`）；用 `e2b-code-interpreter 2.11.0` 和 `tools/measure_code_interpreter.py` 检查。
> 按顺序执行的代码单元（不经模型）："x = 1"、"x+=1; x"、pandas 求和、matplotlib 画图、JavaScript "[1, 2, 3].reduce((a, b) => a + b)"、"1/0"，暂停并恢复后再执行 "x"。
<!-- END GENERATED: e-latest-input -->

**变量与固定项。** SDK 检查沿用 `e-2026-10-14-agent` 的 agent、问题、协议和计时，在 host a 上用 `e-2026-10-15-multi` 的同一版工具运行。与 `e-2026-10-14-agent` 相比，换了 SDK（2.46.0 → 2.56.0）和程序（私有驱动脚本 → `tools/measure_e2b.py`）；host a 此后启动的内核是 `7.0.0-1017-azure`，系统盘是 Standard HDD。code-interpreter 模板用 E2B 原样的脚本构建；两次构建之间我们调大了三项宿主机设置，列在构建表下面。4 次检查用的是同一个模板和同样的步骤。

**结果。** SDK 检查：

<!-- BEGIN GENERATED: e-latest-sdk -->
| 次 | 创建 → Running | 第一个回答 | 冷启动 | 后续 | 暂停 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1 | 5.24 | 5.34 | 3.7 | 0.15 | 0.08 | 6.72 | 6.8 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.06 | 4.93 | 4.99 | 4.24 | 0.14 | 0.08 | 6.2 | 6.28 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| 3 | 0.06 | 4.0 | 4.06 | 35.02 | 0.17 | 0.11 | 8.14 | 8.25 | Bangkok is the capital city of Thailand. / 2 / 3 / 4 |
| **中位数** | **0.06** | **4.93** | **4.99** | **4.24** | **0.15** | **0.08** | **6.72** | **6.8** |  |

模型 HTTP 尝试：200 ×24。历史在 3 次中有 3 次按 1、2、3、4 增长；没有遗留沙箱。
<!-- END GENERATED: e-latest-sdk -->

构建 `code-interpreter-v1`（共 30 步）：

<!-- BEGIN GENERATED: e-latest-build -->
| 次 | HUGEPAGES | 构建工作空间，MiB | 系统盘，GB | 结果 | 秒 |
|---:|---:|---:|---:|---|---:|
| 1 | 2048 | 512 | 50 | 第 6/30 步失败：恢复构建 VM 时 `mmap memfd: cannot allocate memory` | 104.5 |
| 2 | 4096 | 512 | 50 | 第 8/30 步失败：构建 VM 里的 `pip install`：`[Errno 28] No space left on device` | 56.5 |
| 3 | 4096 | 8192 | 50 | 我们在第 12/30 步停止：系统盘剩余不到 8 GiB，而 E2B 的数据库也在这块盘上 | 501 |
| 4 | 4096 | 8192 | 128 | 建成，30/30 步；第 1–11 步用第 3 次的缓存 | 426 |

到第 12 步时，第 3 次已写入 12839 MiB 模板和 10140 MiB 编排器目录数据，系统盘只剩 5084 MiB；第 4 次又用掉 29 GiB（剩余 93 → 64 GiB）。我们改了这些：Compose `.env` 里的 `HUGEPAGES` 2048 → 4096，再执行 `docker compose up -d --wait`；测试团队的构建工作空间（`disk_mb`）512 → 8192 MiB，做法是往 Embed 内部的 `project_limits` 表加一行；系统盘 50 → 128 GB（释放、拍快照、`az disk update --size-gb`、开机）。
<!-- END GENERATED: e-latest-build -->

检查，共 4 次：

<!-- BEGIN GENERATED: e-latest-ci -->
| 步骤 | 通过 | 秒，最小–最大 | 第 4 次的观察 |
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

JavaScript 失败的原因（在新沙箱里复现）：新沙箱的第一个单元就是这段 JavaScript，再执行一次同样的单元：`TypeError` → `6`；先执行一个 Python 单元，再执行这段 JavaScript（检查里的顺序）：成功 → `TypeError`；在这段 JavaScript 前加一个分号：`6`。在 `162a2a2` 的 `template/server/messaging.py` 里，第 160 行写出的 `process.env['KEY'] = 'VALUE'` 不带分号，第 351–361 行在一个上下文第一次执行时把这些语句放在用户代码前面，于是以 `[` 开头的代码接到了最后一句上。以 `console.log` 或数字开头的代码可以正常执行。
<!-- END GENERATED: e-latest-ci -->

**边界。**
- SDK 检查已被取代：它用的是早期的执行方式（`base` 模板，用 `commands.run` 跑 `python3`）；`e-2026-10-17-ci-agent` 和 `e-2026-10-18-ci-multi` 用 `code-interpreter-v1` 跑了同样的口径。模板构建和 code-interpreter 检查仍然有效。
- 用 Embed 的默认值建不起官方模板：大页池、构建工作空间和 50 GB 系统盘各卡住过一次。构建工作空间是往 Embed 内部的 `project_limits` 表里加一行调大的，E2B 没有为此写文档；这次结果只说明模板能工作，不代表这是受支持的配置。
- 构建所需磁盘只来自这一次：第 4 次的第 1–11 步复用了第 3 次的缓存，从零构建大约要两者之和，这一点没有实测。
- 每个 `code-interpreter-v1` 沙箱是 2048 MB，是 `base` 的四倍；本次没有实测一台主机能放几个（`e-2026-10-17-ci-perf` 在 host c 上测了）。
- JavaScript 的失败在 E2B 的模板服务端，与 Azure 和宿主机无关；运行同一份服务端代码的 E2B Cloud 是否同样失败，没有检查。
- 2026-10-15T16:48Z 订阅的自动化在清理前把 host a 释放了，所以调大的设置和建好的模板还留在它的盘上。

### e-2026-10-15-multi：两台独立 Azure VM 上的 E2B

**问题。** E2B Embed 按设计只有一台机器。再装一套独立的能带来什么：换一台 VM 结果是否一致？客户端把会话分到两台上，容量能否相加？一台的 E2B 停掉后还剩下什么？

**输入。** 两台 VM，每个阶段都由 host a 上的同一个工具运行（[`evidence/runs/e-e2b-2026-10-15/multi.json`](evidence/runs/e-e2b-2026-10-15/multi.json)）：

<!-- BEGIN GENERATED: e-multi-input -->
> 工具：`tools/measure_e2b.py`（SHA-256 `036dc81db607…`）；host b 上的 agent 实测用的是前一版 `89255d0b00a3…`，那一版还不记录 HTTP 状态。agent 指令、工具定义和问题与 `e-2026-10-14-agent`、`e-2026-10-14-perf` 相同。
> Host a：`Standard_D4ds_v7`，zone `1`，宿主机内核 `7.0.0-1017-azure`，Docker `29.9.0`，系统盘 `Standard HDD (Standard_LRS) · 50 GB`，`HUGEPAGES=2048`；2026-10-14 为 `e-2026-10-14-agent` 安装，夜间释放；开机后按 E2B 指南对重启的要求执行一次 `docker compose up -d --wait`。
> Host b：`Standard_D4ds_v7`，zone `1`，宿主机内核 `6.17.0-1022-azure`，Docker `29.9.1`，系统盘 `Premium SSD P6 (Premium_LRS) · 50 GB`，`HUGEPAGES=2048`；2026-10-15 按 README 第 7 步安装（`588e5b9` 的 `compose.yaml` 和 `.env`，SHA-256 均校验通过），用时 175 s。
> 网络：两台主机在同一子网和可用区；工具在 host a 上运行，经私网访问 host b 的 `3000` 和 `3002` 端口；host b 的 NSG 只允许 host a 的私网地址访问这两个端口，不放行其他入站流量。
> host b 上的 agent 实测："Where is Bangkok? Answer in one short sentence."，然后 3 次 "How many questions have I asked before this one? Number only."，第二轮后暂停 5 s。
> 单台并发（host a）：8 个会话；在此之前先跑一个新会话 "Use run_in_e2b_sandbox to print COLD-1, then reply with exactly COLD-1" 和一个以 "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK." 开头的会话。
> 并发时每个会话："Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."。
> 停一台：每台各有一个存活会话（"Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."）和一个暂停的沙箱（"Use run_in_e2b_sandbox to write PAUSED-{h} to /home/user/paused.txt. Reply only OK."，`{h}` 为主机名）；然后在 host a 上执行 `docker compose stop`（VM 不关），之后再 `docker compose up -d --wait`；读回时发 "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only." 和 "Use run_in_e2b_sandbox to read /home/user/paused.txt. What does it say? Word only."。
> 空闲容量：在 host a 上逐个 `Sandbox.create("base")`，不调模型、不跑命令，直到创建被拒，共 3 次。
<!-- END GENERATED: e-multi-input -->

**变量与固定项。** 固定：VM 规格、镜像、可用区、E2B 提交、compose 文件、大页池、SDK `e2b 2.46.0`、模型、agent 和问题。变化：主机，以及会话分到几台主机上。未固定：host a 的内核和系统盘与 host b 不同（见下）。

**结果。** host b 上的 3 次 agent 协议：

<!-- BEGIN GENERATED: e-multi-agent -->
| 次 | 创建 → Running | 第一个回答 | 冷启动 | 后续 | 暂停 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.09 | 5.58 | 5.67 | 6.9 | 0.15 | 0.08 | 6.87 | 6.95 | Bangkok is the capital city of Thailand, in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.06 | 3.85 | 3.91 | 5.92 | 0.11 | 0.1 | 7.05 | 7.15 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| 3 | 0.06 | 45.56 | 45.62 | 3.6 | 0.12 | 0.08 | 45.53 | 45.61 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| **中位数** | **0.06** | **5.58** | **5.67** | **5.92** | **0.12** | **0.08** | **7.05** | **7.15** |  |

沙箱内的历史在 3 次中有 3 次按 1、2、3、4 增长。第 3 次慢在模型轮次：第一个回答用了 45.56 s，其中沙箱命令只占 0.11 s。
<!-- END GENERATED: e-multi-agent -->

同时开多个会话：

<!-- BEGIN GENERATED: e-multi-scale -->
| 阶段 | 同时会话 | 回答 | 拒绝 | 拒绝原因 | 沙箱创建，s | 整个会话，s | 模型 HTTP 尝试 |
|---|---:|---|---|---|---:|---:|---|
| 仅 host a | 8 | 7/8 (a 7/8) | 1/8 | `cannot allocate memory` | 0.164–0.53 | 8.03–25.65 | 200 ×28 |
| host a 和 b | 16 | 14/16 (a 7/8，b 7/8) | 2/16 | `cannot allocate memory` | 0.147–0.432 | 8.57–42.09 | 200 ×56, 500 ×2 |

客户端把 16 个会话轮流分给两台主机。每次拒绝都是 E2B 的 `500: Failed to place sandbox`，背后是编排器的 `cannot allocate memory`；模型返回的 2 次 HTTP 500 都在同一轮内重试成功，这些会话照常回答。
<!-- END GENERATED: e-multi-scale -->

host a 上的空闲 `base` 沙箱：

<!-- BEGIN GENERATED: e-multi-idle -->
| 次 | 被拒前创建成功 | 拒绝原因 | 已占大页：开始 → 被拒时 | 全部删除后 |
|---:|---:|---|---|---|
| 1 | 5 | `error setting fc metrics (connection refused)` | 75 → 1355 | 75 页，0 个沙箱 |
| 2 | 7 | `cannot allocate memory` | 75 → 1867 | 75 页，0 个沙箱 |
| 3 | 7 | `cannot allocate memory` | 75 → 1867 | 75 页，0 个沙箱 |

每页 2 MiB。每个 `base` 沙箱（512 MiB）在创建时就预留 256 页，不论是否空闲。大页池共 2048 页，Compose 栈里的 postgres 占 75 页，剩下 1973 页，能放下 7 个沙箱。第 1 次因为另一个错误提前停止。
<!-- END GENERATED: e-multi-idle -->

用 `docker compose stop` 停掉 host a 的 E2B（VM 不关）：

<!-- BEGIN GENERATED: e-multi-hostloss -->
| 检查项 | 观察结果 |
|---|---|
| host b，存活会话：读回文件 | 成功，4.72 s, "LANTERN-7" |
| host b：新会话 | 成功，4.38 s, "COLD-1" |
| host b：恢复 host a 上暂停的沙箱 | 失败：`SandboxNotFoundException: Paused sandbox … not found` |
| host a 停止中：创建 | 失败：`connection refused` |
| host a 停止中：恢复自己暂停的沙箱 | 失败：`connection refused` |
| host a 重启后：恢复自己暂停的沙箱 | 成功，4.72 s, "PAUSED-a" |
| host a 重启后：停止时正在运行的沙箱 | 失败：`SandboxNotFoundException: Paused sandbox … not found` |
| host a 重启后：新会话 | 成功，4.09 s, "COLD-2" |
| `docker compose stop` / `docker compose up -d --wait` | 73.69 s / 59.55 s |
<!-- END GENERATED: e-multi-hostloss -->

**边界。**
- 已被取代：这次用的是早期的执行方式（`base` 模板，用 `commands.run` 跑 `python3`）；`e-2026-10-18-ci-multi` 用 `code-interpreter-v1` 重做了 agent 实测、并发和停一台，`e-2026-10-18-ci-host` 测了每个 `code-interpreter-v1` 沙箱占多少大页。
- 这是两套独立安装加一个把会话分给它们的客户端，不是 E2B 集群：没有共享调度、没有故障转移、没有共享存储。每台的会话数可以相加；暂停的沙箱只能在暂停它的那台主机上恢复。
- 停一台用的是在运行中的 VM 上 `docker compose stop`。VM 或磁盘丢失，以及官方 Terraform 模块的规模集用新盘替换实例，都没有实测。
- 两台主机并不完全相同：host a 夜间释放后启动的内核是 `7.0.0-1017-azure`，它的系统盘在 2026-10-14 实测之后被订阅策略从 Premium SSD 改成了 Standard HDD；host b 是 `6.17.0-1022-azure`，Premium SSD。
- 每个阶段只跑了一次；容量针对的是 512 MB 的 `base` 模板、`Standard_D4ds_v7` 上 Compose 默认的大页池（`e-2026-10-17-ci-perf` 测了 `code-interpreter-v1` 和 4096 页的大页池）。

### e-2026-10-17-ci-agent：用 E2B 的 Code Interpreter 作为 Prompt Agent 的沙箱，共 3 次

**问题。** E2B 的 README 用 Code Interpreter 在沙箱里执行代码：先用 `e2b-code-interpreter` 的 `Sandbox.create()`，再调用 `run_code`，模板是官方的 `code-interpreter-v1`。Prompt Agent 的工具改用这种方式执行 Python（不再在 `base` 里 `commands.run("python3")`）后，按 A 的 3 次口径，创建、第一个回答、暂停、恢复和恢复到回答各要多久？在一套新装的 Embed 上构建这个模板要做什么？

**输入。** 在 host c 上，这是第三套 Embed 安装（[`evidence/runs/e-e2b-2026-10-17/ci-agent.json`](evidence/runs/e-e2b-2026-10-17/ci-agent.json)）：

<!-- BEGIN GENERATED: e-ci-input -->
> Host c：`Standard_D4ds_v7`（4 vCPU、16 GiB），zone `2`，系统盘 128 GB；E2B Embed `588e5b9` 于 2026-10-16 用 README 第 7 步下载的那两个文件安装（`compose.yaml` 和 `.env`，SHA-256 均校验通过），之后在 `.env` 末尾加上 `HUGEPAGES=4096`。
> 模板：`code-interpreter-v1`（2 vCPU、2048 MB），来自 [`e2b-dev/code-interpreter@162a2a2`](https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template)，在 host c 上用其未改动的 `template/build_prod.py` 构建（见下方构建表）。
> 执行方式：E2B Code Interpreter，`e2b 2.56.0` 和 `e2b-code-interpreter 2.11.0`；每次调用 `run_in_e2b_sandbox` 工具就是在沙箱的 Jupyter 内核里执行一个 `run_code` 单元：先是工具记录历史的那一步，然后是模型写的 Python。
> 工具：`tools/measure_e2b.py`（SHA-256 `cbc96f273c07…`），加 `--executor code-interpreter`；模型 `gpt-5.4-mini`；agent 指令和工具定义与 `e-2026-10-14-agent` 相同。
> agent 实测："Where is Bangkok? Answer in one short sentence."，然后 3 次 "How many questions have I asked before this one? Number only."，第二轮后暂停 5 s。
> 性能：新会话 "Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}"（n = 1–5）；同一会话 "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only."，再 "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1"，再 "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2"；4 和 8 个会话同时开始，每个发 "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."；替换：一个存活会话先发 "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."，另一个沙箱被删除并替换后再发 "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only."。
> 空闲容量：逐个 `Sandbox.create` 每种模板，用 `run_code("1+1")` 或 `echo ok` 检查、不调模型，直到创建被拒；先在 `HUGEPAGES=4096` 下测，再在 Compose 默认的 2048 下测，最后恢复 4096。探测脚本是作者的私有脚本，SHA-256 记在记录里。
<!-- END GENERATED: e-ci-input -->

**变量与固定项。** 与 `e-2026-10-15-latest` 的 SDK 检查相比，执行方式变了（`base` 里的 `commands.run("python3")` → `code-interpreter-v1` 里的 `run_code`），主机（host c，zone `2`）和工具版本也变了；新版工具只增加了执行方式选项，agent、问题和计时不变。构建用的是 2026-10-15 让 host a 建成的三项宿主机设置。

**结果。** 3 次 agent 实测：

<!-- BEGIN GENERATED: e-ci-agent -->
| 次 | 创建 → Running | 第一个回答 | 冷启动 | 后续 | 暂停 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.22 | 3.14 | 3.36 | 4.09 | 0.29 | 0.63 | 4.33 | 4.96 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.08 | 3.85 | 3.93 | 4.17 | 0.09 | 0.64 | 4.64 | 5.28 | Bangkok is the capital of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 3 | 0.07 | 4.55 | 4.62 | 3.61 | 0.08 | 1.43 | 4.77 | 6.2 | Bangkok is the capital of Thailand. / 1 / 2 / 3 |
| **中位数** | **0.08** | **3.85** | **3.93** | **4.09** | **0.09** | **0.64** | **4.64** | **5.28** |  |

沙箱内的历史在 3 次中有 3 次按 1、2、3、4 增长。每一轮都只调用一次 `run_code`，占该轮 0.011–0.123 s；guest 内核 `6.1.177+`。模型 HTTP 尝试：200 ×24。没有遗留沙箱、会话或 agent 版本。
<!-- END GENERATED: e-ci-agent -->

在 host c 上构建 `code-interpreter-v1`（30 步）：

<!-- BEGIN GENERATED: e-ci-build -->
| 尝试 | 结果 | 秒 |
|---:|---|---:|
| 1 | 在第 8/30 步失败：`pip install --no-cache-dir -r requirements.txt: [Errno 28] No space left on device` | 181 |
| 2 | 在第 20/30 步失败：`error building step 20: resume sandbox: failed to start FC: error setting fc metrics` | 840 |
| 3 | 建成，30/30 步；第 1–19 步来自第 2 次的缓存（构建器显示 1m25s） | 92 |

三次尝试都用了 2026-10-15 让 host a 建成的设置：`HUGEPAGES` 4096、测试团队的构建工作空间（`disk_mb`）8192 MiB（`project_limits` 表里一行）、128 GB 系统盘；各次尝试之间没有改任何设置。第 1 次在加这一行之后 16 s 开始，在第 8 步空间不足；第 2 次在加行约 5 分钟后开始，通过了这一步。这与新上限要过一段时间才生效相符，我们没有进一步核实。第 3 次之前，一个订阅自动化把 VM 释放了，我们重新开机。
<!-- END GENERATED: e-ci-build -->

**边界。**
- 一台主机上的 3 次实测。工具在主机本机运行（`localhost`），这些数字不含客户端到 Embed 的网络。与 `e-2026-10-15-latest` 的 SDK 检查对比时请当作现场记录：不只执行方式不同，主机、磁盘和模板也不同。
- 这里恢复的中位数为 <!--v:e.ci.resume.median-->0.64<!--/v--> s，host a 上的 `base` 为 <!--v:e.latest.sdk.resume.median-->0.08<!--/v--> s。2 GiB 沙箱的内存是 `base` 的四倍，两台主机也不同；本次实测没有把这两个因素分开。`e-2026-10-18-ci-multi` 在 host a 上用 `code-interpreter-v1` 跑了同样的口径：恢复中位数 <!--v:e.m.a.resume.median-->0.39<!--/v--> s（<!--v:e.m.a.resume.range-->0.38–1.46<!--/v--> s），介于两者之间：在 host a 上 2 GiB 模板比 `base` 恢复得慢，在 host c 上又比在 host a 上慢；各只有一次实测，且不在同一天。
- 成功的那次构建复用了失败的第 2 次的第 1–19 步，所以 92 s 不是干净构建的耗时；`e-2026-10-15-latest` 在 host a 上的 426 s 也复用了第 1–11 步。
- `project_limits` 这一行和 `e-2026-10-15-latest` 一样，是对 Embed 内部表的未公开改动；它要过一段时间才生效，这是从两次尝试推断的，没有实测。
- 第 20 步的失败（`error setting fc metrics`）在 VM 重启后没有再出现，原因没有查明。

### e-2026-10-17-ci-perf：code-interpreter-v1 的冷启动、后续回答、并发、沙箱替换与空闲容量

**问题。** 按 `e-2026-10-14-perf` 的 D 口径，`code-interpreter-v1`（每个沙箱 2 GiB）在一台 `Standard_D4ds_v7` 上表现如何？这台机器能放下几个这样的沙箱？

**输入。** 与 `e-2026-10-17-ci-agent` 相同的主机、工具和 agent，运行 `perf --cold 5 --scale 4,8 --recovery --executor code-interpreter`，然后运行空闲容量探测（[`evidence/runs/e-e2b-2026-10-17/ci-perf.json`](evidence/runs/e-e2b-2026-10-17/ci-perf.json)）；全部问题列在 `e-2026-10-17-ci-agent` 下。

**变量与固定项。** 冷启动重复 5 个新会话；并发分别为同时 4 个和 8 个会话；空闲探测改变模板和大页池（先 `HUGEPAGES=4096`，再 Compose 默认的 2048）。

**结果。** 新会话：

<!-- BEGIN GENERATED: e-ci-cold -->
| 开始（UTC） | 回答 | 沙箱创建 → Ready，s | Agent 轮次，s | 新会话 → 回答，s |
|---|---|---:|---:|---:|
| 16:56:18 | COLD-1 | 0.13 | 3.17 | 3.35 |
| 16:56:21 | COLD-2 | 0.073 | 5.43 | 5.66 |
| 16:56:27 | COLD-3 | 0.072 | 3.48 | 3.64 |
| 16:56:31 | COLD-4 | 0.069 | 3.12 | 3.31 |
| 16:56:34 | COLD-5 | 0.1 | 2.5 | 2.7 |

5 个新会话汇总：沙箱创建 → Ready 中位数 0.073 s（0.069–0.13；n=5），Prompt Agent 轮次 中位数 3.17 s（2.5–5.43；n=5），端到端新会话 → 回答 中位数 3.35 s（2.7–5.66；n=5）。
<!-- END GENERATED: e-ci-cold -->

同一会话中的后续回答：

<!-- BEGIN GENERATED: e-ci-warm -->
| 轮次 | 回答 | 沙箱工具耗时，s | 轮次耗时，s |
|---|---|---:|---:|
| 1（新会话；沙箱在此之前创建） | OK | 0.05 | 3.68 |
| 2 | LANTERN-7 | 0.014 | 3.36 |
| 3 | WARM-1 | 0.012 | 3.93 |
| 4 | WARM-2 | 0.012 | 4.59 |

沙箱在第 1 轮之前创建，用时 0.07 s。后续第 2–4 轮：中位数 3.93 s（3.36–4.59；n=3）。
<!-- END GENERATED: e-ci-warm -->

并发与沙箱替换：

<!-- BEGIN GENERATED: e-ci-scale -->
| 并发 | 回答 | 拒绝 | 沙箱创建，s | 整个会话，s | 模型 HTTP 尝试 |
|---:|---|---|---:|---:|---|
| 4 | 3/4 | 1/4 | 0.23–0.288 | 6.43–15.07 | 200 ×12 |
| 8 | 3/8 | 5/8 | 0.124–0.231 | 5.94–8.59 | 200 ×12 |

每个被拒的会话都在创建沙箱时收到 E2B 的 `500: Failed to place sandbox`，没有走到模型。`kill` 删除沙箱后的替换：新的 `code-interpreter-v1` 沙箱在 0.12 s 内就绪；另一个存活会话的追问 HTTP 200，6.86 s，"LANTERN-7"，保留状态：true。没有遗留沙箱或会话。
<!-- END GENERATED: e-ci-scale -->

按模板和大页池统计的空闲沙箱：

<!-- BEGIN GENERATED: e-ci-idle -->
| `HUGEPAGES`（2 MiB 页） | 开始时空闲 | `code-interpreter-v1`（2048 MB = 1024 页） | `base`（512 MB = 256 页） |
|---:|---:|---:|---:|
| 4096 | 4083 | 3 | 15 |
| 2048 | 2035 | 1 | 7 |

每个数都等于空闲页数除以模板所需页数后向下取整：已有 3 个 `code-interpreter-v1` 沙箱在运行时，下一个需要 1024 页，而此时只剩 4083 − 3 × 1024 = 1011 页。这与每个沙箱在创建时就从大页池预留全部内存相符，`e-2026-10-15-multi` 对 `base` 实测过这一点；本次只记录了 `HugePages_Free`，它只计已经用到的页（每个空闲 `code-interpreter-v1` 沙箱 261–285 页），不计预留。每轮创建都以 `500: Failed to place sandbox` 结束，所有沙箱都已删除，大页池恢复为 4096 页。
<!-- END GENERATED: e-ci-idle -->

**边界。**
- 每个阶段在一台主机上跑一次；工具在主机本机运行。
- `HUGEPAGES=4096` 时主机只放得下 3 个沙箱，所以同时 4 个和 8 个会话测到的是大页池满时的拒绝，不是吞吐量。大页池为 4096 页时 MemAvailable 为 5700 MiB，2048 页时为 9782 MiB；更大的大页池没有试。
- 每个沙箱占多少页是根据数量和模板内存推断的；没有记录 `HugePages_Rsvd`。
- 和 `e-2026-10-14-perf` 一样，`kill` 之后的追问发给了另一个存活会话；它不能说明被删沙箱的连续性。

### e-2026-10-18-ci-host：host c 上不调模型的 code-interpreter-v1：20 次暂停与恢复循环，以及每个沙箱占多少大页

**问题。** 没有模型参与时，2 GiB 的 `code-interpreter-v1` 沙箱连续跑 20 次，创建、执行一个单元、暂停和恢复各要多久？每次暂停后内核里的状态能不能恢复？一个沙箱占多少大页，也就是一台主机能放几个？

**输入。** 在 host c 上运行工具的 `host` 模式（[`evidence/runs/e-e2b-2026-10-18/ci-host.json`](evidence/runs/e-e2b-2026-10-18/ci-host.json)）：

<!-- BEGIN GENERATED: e-host-input -->
> Host c：`Standard_D4ds_v7`（4 vCPU、16 GiB），zone `2`，系统盘 `Standard HDD (Standard_LRS) · 128 GB`，`HUGEPAGES=4096`；即 `e-2026-10-17-ci-agent` 的安装，`code-interpreter-v1` 就建在这台上。
> 工具：`tools/measure_e2b.py`（SHA-256 `aca40d17e315…`），在 host c 本机上对 `localhost` 运行 `measure_e2b.py host --bench 20 --reserve 4`；不调模型、不建 agent；只用 E2B Code Interpreter（`e2b 2.56.0`、`e2b-code-interpreter 2.11.0`，模板 `code-interpreter-v1`，2 vCPU、2048 MB）。
> 读回：一个沙箱执行单元 `import os, platform; mem_kb = next(int(ln.split()[1]) for ln in open('/proc/meminfo') if ln.startswith('MemTotal:')); print(platform.release(), os.cpu_count(), mem_kb // 1024)`，同时工具从 `/proc/*/cmdline` 列出宿主机上的 Firecracker 进程。
> 循环：20 个沙箱逐个进行，每个 `Sandbox.create` → 单元 `x = 41` → 单元 `x += 1` → `beta_pause` → 等待 5 s → `Sandbox.connect` → 单元 `print(x)`（必须打印 `42`）→ `kill`。
> 预留：逐个创建沙箱并保留，每个用 `print(1 + 1)` 检查；在第一个之前和每个之后读 `/proc/meminfo`，直到创建被拒；然后删除全部沙箱，5 s 后再读一次 `/proc/meminfo`。
<!-- END GENERATED: e-host-input -->

**变量与固定项。** 固定：host c、它的安装、大页池和模板、SDK 和工具；不调模型、不建 agent，工具和 E2B 之间没有网络。变化：第几次循环，以及预留探测里同时保留的沙箱数。

**结果。** 20 次循环：

<!-- BEGIN GENERATED: e-host-bench -->
| 循环 | 创建 | 第一个单元 | 下一个单元 | 暂停 | 恢复 | 恢复后单元 | 恢复后的 `x` |
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

单位为秒。20 次循环中：创建 0.088–0.56 s（n=20），20 次里 4 次低于 0.1 s；第一个单元 0.08–0.19 s（n=20）；下一个单元 0.008–0.048 s（n=20）；暂停 0.14–3.27 s（n=20），20 次里 13 次超过 1 s；恢复 0.38–10.24 s（n=20）：前 3 次最多 0.55 s，之后每次 3.84–10.24 s；恢复后单元 0.044–0.21 s（n=20）。`print(x)` 在 20 次中有 20 次打印 42：内核里的变量随内存快照一起恢复。没有遗留沙箱。
<!-- END GENERATED: e-host-bench -->

逐个加沙箱时的已占大页：

<!-- BEGIN GENERATED: e-host-reserve -->
| 沙箱数 | 创建，s | `HugePages_Free` | `HugePages_Rsvd` | 已占页数 | MemAvailable，MiB |
|---|---:|---:|---:|---:|---:|
| 0（开始前） |  | 4083 | 62 | 75 | 5634 |
| 1 | 0.22 | 3815 | 818 | 1099 | 5628 |
| 2 | 0.14 | 3522 | 1549 | 2123 | 5623 |
| 3 | 0.08 | 3251 | 2302 | 3147 | 5631 |
| 4：被拒 | 0.217 |  |  |  |  |
| 0（全部删除后） |  | 4083 | 62 | 75 | 5680 |

已占页数 = `HugePages_Total`（4096）− 空闲 + 预留；每页 2 MiB。每个沙箱一创建就占 1024 页，即它的全部 2048 MiB；host c 上其他进程占 75 页。有 3 个沙箱时只剩 4096 − 75 − 3 × 1024 = 949 页，不够再放一个，第 4 次创建在 0.217 s 后收到 `500: Failed to place sandbox`。MemAvailable 一直在 5623–5634 MiB：大页池在开机时就划出去了，沙箱内存不体现在这里。

读回：宿主机进程列表里沙箱背后是 Firecracker `v1.14-0.2.0`；沙箱内内核 `6.1.177+`、2 vCPU、MemTotal 1982 MiB（模板 2048 MB）；宿主机内核 `7.0.0-1017-azure`。
<!-- END GENERATED: e-host-reserve -->

**边界。**
- 一台主机，每种探测只跑一次；工具在 host c 本机运行，所以这些只是 E2B 调用本身，不是 agent 轮次。
- 前几次循环之后恢复时间突然变长，这里只记录了现象，没有解释。没有记录磁盘活动；E2B 把每次暂停的快照（`vm_state` + `mem_file`）写到宿主机的系统盘上，host c 的系统盘是 Standard HDD，所以磁盘繁忙是一个可能的原因，本次没有验证。
- 各次循环一个接一个，只在每次恢复前等 5 s；在 agent 实测里，两次暂停之间隔着数秒的模型轮次，恢复在 host c 上为 <!--v:e.ci.resume.range-->0.63–1.43<!--/v--> s（`e-2026-10-17-ci-agent`），在 host a 上为 <!--v:e.m.a.resume.range-->0.38–1.46<!--/v--> s（`e-2026-10-18-ci-multi`）。

### e-2026-10-18-ci-multi：两台独立 Azure VM 上的 code-interpreter-v1

**问题。** 用 E2B 的 Code Interpreter 在 `code-interpreter-v1` 里执行时，agent 口径在 host a 上结果是否一样？客户端把会话分到两套安装上时，容量能否相加？一台主机的 E2B 服务栈停掉后，什么还在？

**输入。** host a 和 host c，每个阶段的工具都在 host a 上运行（[`evidence/runs/e-e2b-2026-10-18/ci-multi.json`](evidence/runs/e-e2b-2026-10-18/ci-multi.json)）：

<!-- BEGIN GENERATED: e-cm-input -->
> Host a：`Standard_D4ds_v7`（4 vCPU、16 GiB），zone `1`，系统盘 `Standard HDD (Standard_LRS) · 128 GB`，`HUGEPAGES=4096`；为 `e-2026-10-14-agent` 安装；在 `e-2026-10-15-latest` 中大页池调到 4096 页、加了构建工作空间那一行、系统盘扩到 128 GB，并建好 `code-interpreter-v1`。
> Host c：`Standard_D4ds_v7`（4 vCPU、16 GiB），zone `2`，系统盘 `Standard HDD (Standard_LRS) · 128 GB`，`HUGEPAGES=4096`；即 `e-2026-10-17-ci-agent` 的安装。
> 工具：`tools/measure_e2b.py`（SHA-256 `aca40d17e315…`），每个阶段都在 host a 上运行；只用 E2B Code Interpreter（`e2b 2.56.0`、`e2b-code-interpreter 2.11.0`，模板 `code-interpreter-v1`，2 vCPU、2048 MB）。
> 网络：工具经私网访问 host c 的 `3000` 和 `3002` 端口；host c 的 NSG 只允许 host a 的私网地址访问这两个端口，拒绝其他所有入站连接。每台主机的 API key 来自 hosts 文件里指定的环境变量。
> host a 上的 agent 实测："Where is Bangkok? Answer in one short sentence."，然后 3 次 "How many questions have I asked before this one? Number only."，第二轮后暂停 5 s；agent 指令和工具定义与 `e-2026-10-14-agent` 相同。
> 并发：先只在 host a 上同时开 4 个会话，再在两台上各开 4 个："Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."。
> 停一台：每台各有一个存活会话（"Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."）和一个暂停的沙箱（"Use run_in_e2b_sandbox to write PAUSED-{h} to /home/user/paused.txt. Reply only OK."，`{h}` 为主机名）；然后在 host a 上执行 `docker compose stop`（VM 不关），之后再 `docker compose up -d --wait`；读回时发 "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only." 和 "Use run_in_e2b_sandbox to read /home/user/paused.txt. What does it say? Word only."。
<!-- END GENERATED: e-cm-input -->

**变量与固定项。** 固定：模板、大页池（两台都是 `HUGEPAGES=4096`）、SDK、工具、模型、agent 和问题。变化：主机，以及会话分到几台主机上。未固定：两台在不同的 zone（host a 在 `1`，host c 在 `2`），host a 从 2026-10-14 起就在跑 E2B。

**结果。** host a 上的 3 次 agent 实测：

<!-- BEGIN GENERATED: e-cm-agent -->
| 次 | 创建 → Running | 第一个回答 | 冷启动 | 后续 | 暂停 | 恢复 | 恢复后回答 | 恢复 → 回答 | 回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 10.53 | 2.95 | 13.48 | 3.03 | 2.08 | 0.39 | 3.91 | 4.3 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 2 | 0.09 | 2.65 | 2.74 | 4.02 | 0.16 | 1.46 | 2.68 | 4.14 | Bangkok is the capital city of Thailand in Southeast Asia. / 1 / 2 / 3 |
| 3 | 0.22 | 2.5 | 2.72 | 2.59 | 0.15 | 0.38 | 3.36 | 3.74 | Bangkok is the capital city of Thailand. / 1 / 2 / 3 |
| **中位数** | **0.22** | **2.65** | **2.74** | **3.03** | **0.16** | **0.39** | **3.36** | **4.14** |  |

沙箱内的历史在 3 次中有 3 次按 1、2、3、4 增长。第 1 次创建用了 10.53 s，是本次在 host a 上第一次创建 `code-interpreter-v1`；其余两次 0.09, 0.22 s。记录里看不出这段时间花在哪里。每一轮都只调用一次 `run_code`，占该轮 0.013–0.574 s；guest 内核 `6.1.177+`。模型 HTTP 尝试：200 ×24。没有遗留沙箱、会话或 agent 版本。
<!-- END GENERATED: e-cm-agent -->

同时进行的会话：

<!-- BEGIN GENERATED: e-cm-scale -->
| 阶段 | 同时会话 | 回答 | 拒绝 | 沙箱创建，s | 整个会话，s | 模型 HTTP 尝试 |
|---|---:|---|---|---:|---:|---|
| 仅 host a | 4 | 3/4 (a 3/4) | 1/4 | 0.279–0.31 | 6.13–8.16 | 200 ×12 |
| host a 和 c | 8 | 6/8 (a 3/4，c 3/4) | 2/8 | 0.242–0.265 | 5.25–7.13 | 200 ×24 |

每个被拒的会话都在创建沙箱时收到 `500: Failed to place sandbox`，没有走到模型。`HUGEPAGES=4096` 时一台主机能放 3 个 `code-interpreter-v1` 沙箱（`e-2026-10-18-ci-host`），所以每台 4 个就有一个被拒；第二台主机带来它自己的空间。客户端把会话轮流分给两台主机。没有遗留沙箱或会话。
<!-- END GENERATED: e-cm-scale -->

用 `docker compose stop` 停掉 host a 的 E2B 服务栈（VM 不关）：

<!-- BEGIN GENERATED: e-cm-hostloss -->
| 检查项 | 观察结果 |
|---|---|
| host c，存活会话：读回文件 | 成功，2.77 s, "LANTERN-7" |
| host c：新会话 | 成功，2.52 s, "COLD-1" |
| host c：恢复 host a 上暂停的沙箱 | 失败：`SandboxNotFoundException: Paused sandbox … not found` |
| host a 停止中：创建 | 失败：`connection refused` |
| host a 停止中：恢复自己暂停的沙箱 | 失败：`connection refused` |
| host a 重启后：恢复自己暂停的沙箱 | 成功，4.78 s, "PAUSED-a" |
| host a 重启后：停止时正在运行的沙箱 | 失败：`SandboxNotFoundException: Paused sandbox … not found` |
| host a 重启后：新会话 | 成功，4.47 s, "COLD-2" |
| `docker compose stop` / `docker compose up -d --wait` | 77.86 s / 63.14 s |
<!-- END GENERATED: e-cm-hostloss -->

**边界。**
- 这是两套独立安装加一个把会话分给它们的客户端，不是 E2B 集群：没有共享调度、没有故障转移、没有共享存储。暂停的沙箱只能在暂停它的那台主机上恢复，和 `e-2026-10-15-multi` 用 `base` 时一样。
- 停一台用的是在运行中的 VM 上 `docker compose stop`；VM 或磁盘丢失没有实测。
- 每个阶段只跑了一次。工具从 host a 经私网访问 host c，所以 host c 的会话含这一跳网络，host a 的不含。

### e-2026-10-18-ci-scale：按 8 个 code-interpreter-v1 沙箱配置的新 Standard_D8ds_v7

**问题。** 如果限制主机的是大页池而不是 CPU，一台能放 8 个 `code-interpreter-v1` 沙箱的 VM 能不能同时回答 8 个会话？在新 VM 上走完 README 第 7 步要多久？模板能不能第一次就建成？

**输入。** host d，一台新 VM（[`evidence/runs/e-e2b-2026-10-18/ci-scale.json`](evidence/runs/e-e2b-2026-10-18/ci-scale.json)）：

<!-- BEGIN GENERATED: e-cs-input -->
> Host d：新建的 `Standard_D8ds_v7`（8 vCPU、32 GiB），zone `2`，系统盘 `Standard HDD (Standard_LRS) · 128 GB`，宿主机内核 `6.17.0-1022-azure`。
> 安装：按 README 第 7 步原样执行，只有一处不同：`HUGEPAGES=8448` 而不是 4096，可放 8 个各 1024 页的沙箱再加 256 页余量；Docker Engine 来自 Docker 的 apt 源；`588e5b9` 的 `compose.yaml` 和 `.env`（SHA-256 校验通过）；然后加 `project_limits` 那一行、`sleep 300`、`162a2a2` 的 `code-interpreter` 及其未改动的 `build_prod.py`。
> 工具：`tools/measure_e2b.py`（SHA-256 `aca40d17e315…`），在 host d 本机上对 `localhost` 先运行 `measure_e2b.py perf --cold 5 --scale 4,8 --recovery`，再运行 `measure_e2b.py host --bench 0 --reserve 9`；只用 E2B Code Interpreter（`e2b 2.56.0`、`e2b-code-interpreter 2.11.0`，模板 `code-interpreter-v1`，2 vCPU、2048 MB）。
> 性能问题与 `e-2026-10-17-ci-agent` 相同：新会话 "Use run_in_e2b_sandbox to print COLD-{i}, then reply with exactly COLD-{i}"（n = 1–5）；同一会话 "Use run_in_e2b_sandbox to save the code word LANTERN-7 to /home/user/codeword.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/codeword.txt. What is the code word? Reply with the word only."，再 "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-1"，再 "Use run_in_e2b_sandbox to verify the sandbox is alive, then reply with exactly WARM-2"；4 和 8 个会话同时开始，每个发 "Use run_in_e2b_sandbox to write SCALE-{k} to /home/user/scale.txt. Reply only OK."，然后 "Use run_in_e2b_sandbox to read /home/user/scale.txt. What did I ask you to remember? Word only."；替换：一个存活会话先发 "Use run_in_e2b_sandbox to write LANTERN-7 to /home/user/live.txt. Reply only OK."，另一个沙箱被删除并替换后再发 "Use run_in_e2b_sandbox to read /home/user/live.txt. What is the code word? Reply with the word only."。
> 预留：与 `e-2026-10-18-ci-host` 相同，最多 9 个沙箱，每个用 `print(1 + 1)` 检查。
<!-- END GENERATED: e-cs-input -->

**变量与固定项。** 与 `e-2026-10-17-ci-perf` 相比，VM 规格（`Standard_D4ds_v7` → `Standard_D8ds_v7`）、大页池（4096 → 8448 页）、宿主机内核和工具版本（`cbc96f2…` → `aca40d1…`）变了；模板、SDK、模型、agent、问题和计时不变。

**结果。** 在 host d 上走 README 第 7 步：

<!-- BEGIN GENERATED: e-cs-install -->
| 步骤 | 开始（UTC） | 秒 |
|---|---|---:|
| 从 Docker 的 apt 源安装 Docker Engine | 09:32:14 | 42 |
| 下载并校验两个文件，`docker compose up -d --wait` | 09:32:56 | 215 |
| 加 `project_limits` 那一行，然后 `sleep 300` | 09:36:31 | 301 |
| 克隆 `162a2a2` 的 `code-interpreter`，建构建环境 | 09:41:32 | 7 |
| `build_prod.py`，30 步 | 09:41:39 | 929 |
| **合计** |  | **1494** |

`psql` 输出 `INSERT 0 1`；构建第一次尝试就以退出码 0 结束，最后一步是 30/30，并输出 `Build finished, took 15m26s`。
<!-- END GENERATED: e-cs-install -->

新会话：

<!-- BEGIN GENERATED: e-cs-cold -->
| 开始（UTC） | 回答 | 沙箱创建 → Ready，s | Agent 轮次，s | 新会话 → 回答，s |
|---|---|---:|---:|---:|
| 10:01:46 | COLD-1 | 0.24 | 2.83 | 3.15 |
| 10:01:50 | COLD-2 | 0.068 | 2.26 | 2.38 |
| 10:01:52 | COLD-3 | 0.062 | 2.56 | 2.67 |
| 10:01:55 | COLD-4 | 0.07 | 2.26 | 2.38 |
| 10:01:57 | COLD-5 | 0.06 | 2.55 | 2.66 |

5 个新会话汇总：沙箱创建 → Ready 中位数 0.068 s（0.06–0.24；n=5），Prompt Agent 轮次 中位数 2.55 s（2.26–2.83；n=5），端到端新会话 → 回答 中位数 2.66 s（2.38–3.15；n=5）。
<!-- END GENERATED: e-cs-cold -->

同一会话中的后续回答：

<!-- BEGIN GENERATED: e-cs-warm -->
| 轮次 | 回答 | 沙箱工具耗时，s | 轮次耗时，s |
|---|---|---:|---:|
| 1（新会话；沙箱在此之前创建） | OK | 0.047 | 2.76 |
| 2 | LANTERN-7 | 0.013 | 2.28 |
| 3 | WARM-1 | 0.01 | 2.76 |
| 4 | WARM-2 | 0.009 | 33.32 |

沙箱在第 1 轮之前创建，用时 0.053 s。后续第 2–4 轮：中位数 2.76 s（2.28–33.32；n=3）。

第 4 轮用了 33.32 s，其中沙箱工具只占 0.009 s；其余是该轮模型那边的时间。
<!-- END GENERATED: e-cs-warm -->

并发与沙箱替换：

<!-- BEGIN GENERATED: e-cs-scale -->
| 并发 | 回答 | 拒绝 | 沙箱创建，s | 整个会话，s | 模型 HTTP 尝试 |
|---:|---|---|---:|---:|---|
| 4 | 4/4 | 0/4 | 0.092–0.159 | 4.47–7.97 | 200 ×16 |
| 8 | 8/8 | 0/8 | 0.105–0.337 | 4.85–10.18 | 200 ×32 |

没有会话被拒。`kill` 删除沙箱后的替换：新的 `code-interpreter-v1` 沙箱在 0.063 s 内就绪；另一个存活会话的追问 HTTP 200，3.5 s，"LANTERN-7"，保留状态：true。没有遗留沙箱或会话。
<!-- END GENERATED: e-cs-scale -->

逐个加沙箱时的已占大页：

<!-- BEGIN GENERATED: e-cs-reserve -->
| 沙箱数 | 创建，s | `HugePages_Free` | `HugePages_Rsvd` | 已占页数 | MemAvailable，MiB |
|---|---:|---:|---:|---:|---:|
| 0（开始前） |  | 8175 | 751 | 1024 | 11899 |
| 1 | 0.05 | 8183 | 759 | 1024 | 11895 |
| 2 | 0.05 | 7908 | 1508 | 2048 | 11887 |
| 3 | 0.053 | 7640 | 2264 | 3072 | 11879 |
| 4 | 0.052 | 7372 | 3020 | 4096 | 11891 |
| 5 | 0.057 | 7103 | 3775 | 5120 | 11887 |
| 6 | 0.056 | 6830 | 4526 | 6144 | 11887 |
| 7 | 0.062 | 6565 | 5285 | 7168 | 11885 |
| 8 | 0.06 | 6286 | 6030 | 8192 | 11875 |
| 9：被拒 | 0.047 |  |  |  |  |
| 0（全部删除后） |  | 8448 | 0 | 0 | 11931 |

已占页数 = `HugePages_Total`（8448）− 空闲 + 预留；每页 2 MiB。每个沙箱一创建就占 1024 页，即它的全部 2048 MiB；host d 上其他进程占 0 页。有 8 个沙箱时只剩 8448 − 0 − 8 × 1024 = 256 页，不够再放一个，第 9 次创建在 0.047 s 后收到 `500: Failed to place sandbox`。MemAvailable 一直在 11875–11899 MiB：大页池在开机时就划出去了，沙箱内存不体现在这里。第一次创建前读到已占 1024 页，创建后仍是 1024 页：刚删除的读回沙箱当时还没把它的页还回去。

host d 上的读回：Firecracker `v1.14-0.2.0`，guest 内核 `6.1.177+`，2 vCPU，MemTotal 1982 MiB，与 host c 相同；宿主机内核 `6.17.0-1022-azure`。
<!-- END GENERATED: e-cs-reserve -->

**边界。**
- 每个阶段在一台主机上跑一次；工具在 host d 本机运行。
- 这里 8 并发回答 8/8，`e-2026-10-17-ci-perf` 中 4096 页的 `Standard_D4ds_v7` 上是 3/8。两台主机在规格、内核和日期上也不同，所以这只说明更大的大页池能放下这些沙箱，不是对齐的速度对比。
- 按 README 等 5 分钟后，构建在唯一一次尝试中通过；一次构建不能说明每次都会通过。
- 和 `e-2026-10-17-ci-perf` 一样，`kill` 之后的追问发给了另一个存活会话；它不能说明被删沙箱的连续性。

### 可以重算的数字

上面每个发布指标都来自 [`evidence/measurements.json`](evidence/measurements.json)。[`tools/build_measurements.py`](tools/build_measurements.py) 会先按 [`evidence/runs/INDEX.json`](evidence/runs/INDEX.json) 校验每个文件的 SHA-256，再从 `evidence/runs/` 重算全部指标。它还检查：A 每次的冷启动等于创建加第一个回答、恢复等于 resume 加回答；D 每个冷启动 pod 的创建到 Ready 与 Kubernetes 时间戳一致，且是 `kata-vm-isolation`；D 的后续会话和删除 pod 后使用的另一个存活会话都记住了 `LANTERN-7`。对 2026-10-18 的实测：C 每次的冷启动等于 apply → Ready 加第一个回答，恢复到回答等于 Ready 加回答；B 和 D 的会话实测必须保留文件和问题计数；每个并发会话的“记住”标记必须与它的回答一致，A 和 C 不能留下任何资源。对 E 的 2026-10-18 实测：20 次循环每次都必须读回 42；每多一个沙箱，已占页数必须增加同样的量，且等于模板内存，放下的沙箱数必须等于大页池减去其他占用再除以这个量；安装各段之和必须等于总耗时。奇数个样本才给中位数，偶数个样本直接列出数值。

耗时字段和发布样本必须是有限、非负的数值，不能用布尔值或数字字符串代替。可选计时字段中的 null 仍表示未测，不能进入发布样本。C 的恢复里程碑必须满足 Ready → LoadBalancer 健康 → 第一个回答；D 已配对的 pod 就绪耗时不能超过该请求的总耗时（允许时间戳舍入误差），并发统计必须有每个会话的记录。JSON 输出拒绝 NaN 和无穷值。

这些是单位、类型、样本数和时间线校验，不是统一的时延上限。样本少、计时位置不同、工作负载未对齐，仍然不能给产品做性能排名；数字看起来合理，也不能代替保留的实测证据。

## 测试与离线校验

**部署证据边界。** 本节是在应用部署后执行的可选检查，不能代替应用创建。README 的完整参数化流程尚未在全新订阅逐条重跑；历史运行成功不能证明每条改写后的部署命令都成功过。保留的 Module 1 共享底座部署用时 <!--v:f.deploy.value-->3<!--/v--> 分钟。

**在线测量。** 回到 `HOSTING_REPO`、加载 `.env`，把已部署的 B 或 D 端点填到 `AGENT_RESPONSES_URL`；测 E 时按 README 第 7 步导出 `E2B_API_KEY`。只运行对应方案的测量。D 的恢复命令会删除会话 pod；A 的测量会停止并删除它新建的沙箱；E 的测量会创建并删除自己的 agent 版本、对话和沙箱，`perf` 会同时开 8 个会话；`measure_c.py` 会在 workshop 的 Sandbox 旁边创建并删除自己的 Sandbox，同时开 8 个可能让集群自动扩缩容加一个 Kata 节点。这些生命周期操作需事先得到明确授权。

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

退出码 0 要求每个请求阶段都通过；容量耗尽的拒绝会保存，但仍退出 1。历史工作负载和时钟不一致，不能用它给产品排名。

**可选隔离观察。** C 或 D 使用各自命名空间。基础检查读取配置的 runtime 和 pod 状态；D 只有获准 exec 时才加 `--container agent --exec-uname`。要求执行的读取失败时退出 1。

```bash
python tools/check_isolation.py -n "$NAMESPACE" -l app=agent-host --exec-uname
python tools/check_isolation.py -n "$SESSION_NAMESPACE"
```

**可选 C 生命周期演练。** 下方命令会主动挂起应用，只能在资源所有者批准后运行；部署和获得回答不需要执行它们。

```bash
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Suspended"}}'
kubectl patch sandbox agent-host -n "$NAMESPACE" --type merge -p '{"spec":{"operatingMode":"Running"}}'
kubectl wait sandbox agent-host -n "$NAMESPACE" --for=condition=Ready --timeout=180s
```


离线校验需要 Python 3.10 及以上，不需要访问 Azure：

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

在 Windows 上用 `.venv\Scripts\activate` 代替 `. .venv/bin/activate` 激活。每条命令都退出 0、每个检查都打印 `PASS`，就算通过。

- **`build_measurements.py --check`** 重建全部指标；任何运行文件的哈希、字段或交叉校验出错，或者已提交的 `measurements.json` 与重建结果不同，都会失败。
- **`render_docs.py --check`** 四个页面里任何一张表或一个行内数字与重新渲染的结果不同，就会失败。
- **`build_diagrams.py --check`** 把两张架构图和每个图标与 [`images/SOURCES.json`](images/SOURCES.json) 比对。
- **`validate_docs.py`** 检查：README 的三个标题及顺序、VALIDATION 的标题、[`evidence/claims.json`](evidence/claims.json) 中的核心声明合同和统一实测顺序；每个实测小节按“问题、输入、变量与固定项、结果、边界”的顺序出现，且输入之前没有表格；每次实测都出现在 README 的结果表里，四份文档的实测顺序一致；链接和图片、图片宽度、中英文数字按生成区块和表格单元格对齐（不只比较全文总数）、操作步骤里的命令只用变量、不会关闭读者的 shell（不用 `set -e` 或 `exit`）、在虚拟环境里安装依赖，且 `az group delete` 单独成块并保留确认提示；以及私有内容防护（订阅和租户 ID、IPv4 地址、邮箱、内部资源名、已撤回的结论）。
- **`pytest`** 把上面这些检查作为测试运行，从原始数值重算中位数，用保存的 pod 和节点 JSON 测 `check_isolation.py`，在不装 Azure SDK 的情况下运行每个工具的 `--help`，并逐项破坏输入（篡改运行文件、无效耗时、倒序时间线、缺少并发样本、改一个数字、插入订阅 ID、删一个标题、把表格放到输入前面、打乱实测顺序），确认它会失败。
- 在线工具的 mock 测试还检查：并发追问失败、跳过恢复、读取或删除 pod 失败、历史丢失、参数无效和配置缺失都不能报成功；这些测试不会调用 Azure。

CI 在 Ubuntu 上用 Python 3.12 运行同样的检查（[工作流](https://github.com/david-xinyuwei/david-share/blob/master/.github/workflows/agent-sandbox-hosting-ci.yml)）。

在线检查需要 Azure 和 README 里创建的资源：`tools/aca_sandbox.py`、`tools/measure_aca.py`、`tools/measure_responses.py`、`tools/measure_sessions.py`、`tools/measure_c.py`、`tools/measure_e2b.py`、`tools/e2b_agent.py`，以及不带 `--pods-json` 的 `tools/check_isolation.py`。

这里没有测的：本仓库整理完成后，这些工具在全新订阅上的端到端运行（它们是实测脚本改为读配置后的版本；`--help` 和离线逻辑已测试），East US 2 以外的区域，以及用 Azure Cloud Shell 作为操作环境。

## 边界、目录与资料

**边界。**

- `LOCAL_MEASUREMENT`：A 的完整 agent 时延带 Blob 卷 n=3、不挂卷 n=3，另有同时 4 个和 8 个沙箱各一次；B 是一轮端到端实测、三次会话接口实测、一轮 D 口径压测（冷启动 n=5、后续回答 n=3、4 和 8 并发），以及两种空闲等待后各三次；C 是一轮端到端实测，加不挂卷和挂 Blob PVC 各三次（含 pod 丢失和挂起/恢复），以及 4 和 8 并发各一次；D 是一轮压测，其中镜像已缓存的冷启动 n=5、后续回答 n=3，会话落地和恢复各一次，另有三次会话接口实测。依赖某个数字之前，请按本页在线测量命令重新测。
- `LOCAL_MEASUREMENT`：A 和 C 使用 workshop reflection agent；B 使用 Agent Framework 和 ResponsesHostServer；D 使用私有预览 sample。B 的 2026-09-27 实测和 D 包含 Foundry 端点和位于中国的客户端（往返 <!--v:d.client_rtt.range-->320–800<!--/v--> ms）；B 的 2026-10-15 和 2026-10-18 实测客户端在 Japan East，并经过 API Management 网关；D 的 2026-10-18 实测从 Japan East 直接调用端点；A 的回答在沙箱内计时；C 在 2026-09-27 于它的 LoadBalancer 处计时，2026-10-18 在 pod 内计时。表中每一行比较的是现场实现，不是相同容器或相同网络链路。
- `LOCAL_MEASUREMENT`：A 加上私有 Blob 卷后创建耗时为 <!--v:a.attr.vnetvolume.range-->15.79–25.12<!--/v--> s；只加镜像、VNet 或端口时为 <!--v:a.attr.small.range-->1.6–2.7<!--/v--> s。
- `LOCAL_MEASUREMENT`：最早的 D 会话 pod 在创建至少 <!--v:d.idle.value-->87<!--/v--> 分钟后仍是 Running。这是 pod 的年龄，不是空闲时长：没有保留最后一次请求的时间戳。它不能证明配置的空闲超时，也不能证明期间始终空闲。
- `LOCAL_MEASUREMENT`：按示例的 500m CPU / 1 Gi 请求加 600 Mi Kata 开销，D 每个 `Standard_D4s_v5` 节点最多容纳 <!--v:d.capacity.per_node-->5<!--/v--> 个会话；同时运行系统 pod 的节点更少。
- `NOT_MEASURED`：B 的平台何时回收空闲会话（`b-2026-10-18-idle` 只在等待结束后读了一次状态）和 B 的会话丢失；A 的宿主机丢失；服务端结束会话后 D 的行为；由 Foundry 治理的 agent 按任务调用 ACA 沙箱；D 在有空闲容量时的并发时延；E2B 以外的其他第三方沙箱；E2B Bring Your Own Cloud 和 private cloud；E2B 官方的 Azure Terraform 模块，包括它用新盘替换不健康实例的规模集行为；整台 E2B VM 或其磁盘丢失。
- `SOURCE_FACT`：E2B Embed“is one machine, by design”；E2B 超出单机的方式是 private cloud（开发中）、由 E2B 在你的账号里运维的 Bring Your Own Cloud，以及 E2B Cloud（[Embed](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/README.md#when-one-node-is-not-enough)）。它只提供明文 HTTP，自身不带 TLS。Azure Terraform 安装方式是只有一台实例的规模集，没有持久数据卷：替换出的实例从新盘启动，自建的模板会丢失；停掉服务栈后，`/health` 不响应约两分钟，实例就会被替换（[Azure 指南](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/terraform/azure/README.md#limitations)）。
- `SOURCE_FACT`：E2B 的 Compose 指南要求按“同时运行的沙箱数”设定 `HUGEPAGES`，并写明 512 MiB 的 `base` 沙箱满内存时占 256 页、空闲时约 90 页（[requirements](https://github.com/e2b-dev/runtime/blob/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed/compose/README.md#requirements)）。在 `e-2026-10-18-ci-host` 和 `e-2026-10-18-ci-scale` 中，每个 `code-interpreter-v1` 沙箱不论是否空闲，一创建就占 1024 页，即它的全部 2 GiB，所以 4096 页放下 3 个，8448 页放下 8 个；`e-2026-10-15-multi` 对 `base` 测到的也一样：每个 256 页，默认大页池只放下 7 个，不是约 20 个。
- `SOURCE_FACT`：ACA Sandboxes 从“prewarmed pools for near-instant availability”启动，恢复“with sub-second restore times”（[概览](https://learn.microsoft.com/azure/container-apps/sandboxes-overview)，资源档位于 2026-10-23 复核）。可选档位（vCPU / GiB）：XS 0.25 / 0.5、S 0.5 / 1、M 1 / 2、L 2 / 4、XL 4 / 8、2XL 8 / 16、3XL 16 / 32。对应磁盘依次为 20 / 20 / 20 / 40 / 80 / 160 / 320 GiB。这些是文档规格，不是七种配置全部做过实测。可配置空闲自动挂起；Memory 保存内存和磁盘，Disk 只保存磁盘。本次 900 s 是本地设置，不是全服务默认值。
- `SOURCE_FACT`：hosted agent 运行在“per-session VM-isolated sandboxes”中，保留 `$HOME` 和 `/files`；空闲超时 2–60 分钟，默认 15；30 天不活跃后删除会话（[hosted agents](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents)）。用 `POST .../sessions/{id}:stop` 停止会话会结束算力并保留持久文件系统（[管理会话](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions#stop-a-session)）。
- `SOURCE_FACT`：hosted agent 的[沙箱规格](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents#sandbox-sizes)为 0.5 vCPU / 1 GiB、1 vCPU / 2 GiB、2 vCPU / 4 GiB；agent 版本上设置的值作用于每个会话，不选择 VM SKU。workshop 的 module-02 使用 0.5 vCPU / 1 GiB。
- `SOURCE_FACT`：AKS Pod Sandboxing 由 Azure Linux 主机 + Microsoft Hyper-V + Cloud Hypervisor + Kata Containers 组成；节点 VM 必须是支持嵌套虚拟化的第 2 代规格；每个 pod VM 的内存等于 pod 的内存 limit（未设置时为 512 Mi），CPU limit 的小数部分向上取整到整数 vCPU（[pod sandboxing](https://learn.microsoft.com/azure/aks/use-pod-sandboxing)）。
- `SOURCE_FACT`：agent-sandbox v0.5.2 支持绝对到期时间 `shutdownTime`；到期后删除 pod 和 Service，`shutdownPolicy` 决定是否保留 Sandbox 对象（[官方 API](https://github.com/kubernetes-sigs/agent-sandbox/blob/v0.5.2/docs/api.md)）。这是定时到期回收，不是按请求计时的空闲检测；`operatingMode` 仍提供显式挂起和恢复。
- `NOT_VERIFIED`：本次未核实到 BYO-AKS 官方公开的空闲策略或完整节点 SKU 清单；会话接口的显式 `:stop` 和恢复在 `d-2026-10-18-sessions` 中可用，这是对当前预览 build 的观察，不是文档写明的策略。预览要求须向产品组确认，pod 年龄的现场观察不能证明这些策略。
- `SOURCE_FACT`：[ACA 官方配额说明](https://sandboxes.azure.com/docs/sandboxes/limits) 按并发 active cores 计量，停止的沙箱不占该配额；这不是固定会话数上限。
- `SOURCE_FACT`：[AKS Pod Sandboxing](https://learn.microsoft.com/azure/aks/concepts-pod-sandboxing) 只隔离计算，不隔离控制面、存储与数据路径，也不限制 cluster-admin 的权限。隔离工具检查 runtime 配置；要求执行的内核读取必须成功，版本字符串不是硬件证明。
- `SOURCE_FACT`：[固定版本 module-02 的实现](https://github.com/oxcp/ainotes/blob/905f2c2/agenthost/module-02/agent-src/main.py) 中，`MODEL_ROUTING=direct` 调用 Foundry project endpoint。如果项目注册了原生 AI Gateway，该请求仍可能经 APIM；不能只凭模式名判断它绕过了网关。
- `SOURCE_FACT`：状态来源：A [2026-09-23 GA](https://techcommunity.microsoft.com/blog/appsonazureblog/azure-container-apps-sandboxes-now-generally-available/4559125)；B [2026-06 GA](https://azure.microsoft.com/updates?id=563596)（Azure Update 标题 “Generally Available: Hosted agents in Foundry Agent Service”；更早的 [2026-06-02 Build 博客](https://devblogs.microsoft.com/foundry/hosted-agents-build26/)写的是 GA 前的公开预览）；C [Pod Sandboxing 于 2025-11-18 GA](https://azure.microsoft.com/updates?id=526237)，agent-sandbox v0.5.2 为开源项目。
- `LOCAL_MEASUREMENT`：A 的保存记录显示对话历史长度继续增长。这检查历史连续性，不证明答案精确正确，也不证明任意进程内存都被保留。
- `SOURCE_FACT`：D 的保底费用 <!--v:d.cost_floor.value-->0.484<!--/v--> USD/h，是两个 `Standard_D4s_v5` 节点各 0.192 USD/h 加 AKS Standard 层 0.10 USD/h，按 2026-09-21 查询的 East US 2 零售价；E 的单 VM 保底费用 <!--v:e.cost_floor.value-->0.344<!--/v--> USD/h，是一台 `Standard_D4ds_v7` Linux VM 0.326 USD/h、一个 Standard 静态公网 IP 0.005 USD/h 加一块 50 GB Premium SSD P6 磁盘 0.013 USD/h，按 2026-10-14 查询的 East US 2 零售价；`e-2026-10-18-ci-scale` 所用规格的 `Standard_D8ds_v7` Linux VM 为 0.653 USD/h（2026-10-18 查询）；不含折扣和税。这不是三个 Kata 节点压测池的费用。
- D 的接入耗时（角色分配约 <!--v:d.role_assignments.value-->120<!--/v--> s、`azd provision` <!--v:d.azd_provision.value-->522<!--/v--> s、`azd deploy` <!--v:d.azd_deploy.value-->212<!--/v--> s）来自更早那份现场记录里的作者控制台；控制台原始输出没有保留。

**目录。**

- [`tools/`](tools/)：`aca_sandbox.py`、`measure_aca.py`、`measure_responses.py`、`measure_e2b.py`、`measure_code_interpreter.py`、`measure_sessions.py`、`measure_c.py`、`e2b_agent.py`、`check_isolation.py`（在线）；`export_evidence.py`（仅作者可运行）、`build_measurements.py`、`render_docs.py`、`build_diagrams.py`、`validate_docs.py`（离线）。
- [`evidence/claims.json`](evidence/claims.json)：A–E 的 canonical 名称、状态、归属、隔离边界、字母映射和实测顺序；[`evidence/runs/`](evidence/runs/)：脱敏后的实测记录，带私有源文件的 SHA-256，以及 `INDEX.json`；[`evidence/measurements.json`](evidence/measurements.json)：所有发布的指标。
- [`images/`](images/)：架构图、它们的生成源 `tools/build_diagrams.py`、8 个 Azure 图标和 [`SOURCES.json`](images/SOURCES.json)。
- [`config/.env.example`](config/.env.example)：全部配置项；[`tests/`](tests/)：离线测试和测试数据。

**资料。**

- Microsoft Learn：[Container Apps Sandboxes 概览](https://learn.microsoft.com/azure/container-apps/sandboxes-overview) · [Foundry Agent Service 的 Hosted agents](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents) · [AKS Pod sandboxing](https://learn.microsoft.com/azure/aks/use-pod-sandboxing) · [Azure 架构图标](https://learn.microsoft.com/azure/architecture/icons/)
- [agent-sandbox v0.5.2](https://github.com/kubernetes-sigs/agent-sandbox/releases/tag/v0.5.2) · [agenthost workshop，固定在 905f2c2](https://github.com/oxcp/ainotes/tree/905f2c2/agenthost)（没有许可证文件；只引用，不复制）
- [PyPI 上的 `azure-containerapps-sandbox`](https://pypi.org/project/azure-containerapps-sandbox/)（0.1.0b4）
- E2B：[Embed（588e5b9）](https://github.com/e2b-dev/runtime/tree/588e5b9500b0987bc5a700d91483b03f0ae7d2a2/embed) · [code-interpreter 模板（162a2a2）](https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template)
- BYO-AKS 私有预览示例和接入指南：由产品组提供，需申请访问。
