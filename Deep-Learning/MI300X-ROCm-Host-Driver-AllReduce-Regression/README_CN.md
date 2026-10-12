# Azure MI300X：宿主机驱动升级后的 AllReduce 性能回归

在一台 Azure ND MI300X v5 VM 上，**原生升级 VM 操作系统的 AMDGPU 内核驱动以及 ROCm/RCCL** 后，8 卡 AllReduce 带宽从约 315 GB/s 降到约 42 GB/s，数据校验仍全部通过。仅在旧驱动上升级容器用户态，并没有复现这个问题。

本目录用于复现和定位问题，**不是生产升级建议，也不是厂商已确认的代码根因**。“宿主机”指运行容器的 VM 操作系统，不是 Azure 底层物理宿主机或其 GIM 驱动。

作者：Xinyu Wei · [English](README.md) · [统一数据](data/native-summary.json) · [脱敏原始证据](evidence/) · [无需 GPU 的验证](#无需-gpu-的验证)

## 原生升级前后实测

**1 GiB 消息、AllReduce out-of-place `busbw`、三轮独立运行的中位数；带宽单位是十进制 GB/s。**

<!-- BEGIN GENERATED: native-results -->
| 进程布局 | 升级前 GB/s | 升级后 GB/s | 性能下降 |
|---|---:|---:|---:|
| 单进程、8 GPU | 314.60 | 41.78 | 86.72% |
| 8 进程、每进程 1 GPU | 314.81 | 41.74 | 86.74% |
<!-- END GENERATED: native-results -->

| 项目 | 升级前 | 升级后 |
|---|---|---|
| 执行位置 | VM 操作系统原生 | VM 操作系统原生，真实重启 |
| 实际加载的 AMDGPU 模块 | `6.12.12` | `6.16.13` |
| AMD 驱动包 | `1:6.12.12.60404-2202139.24.04` | `1:6.16.13.30300400-2341068.24.04`，AMDGPU 30.30.4 |
| ROCm | `6.4.4` | `7.2.4` |
| RCCL | `2.22.3` | `2.27.7` |
| Linux 内核 | `6.8.0-1067-azure` | 不变 |
| MPI | HPC-X Open MPI `4.1.7rc1` | 不变 |
| GPU 分区 | 8 张 MI300X VF，`SPX / NPS1` | 不变 |
| 数据校验 | 全部 `#wrong = 0` | 全部 `#wrong = 0` |

两边使用[同一份官方 rccl-tests 源码](https://github.com/ROCm/rocm-systems/tree/1284e5da11f7904e8a6efd902b9de52253785bd0/projects/rccl-tests)，提交为 `1284e5da11f7904e8a6efd902b9de52253785bd0`；针对各自原生 HIP/RCCL 重新编译，配置为 `MPI=1`、`GPU_TARGETS=gfx942`。测试覆盖 **1 MiB～1 GiB**、消息大小每次翻倍，FP32 `sum`，20 次预热、100 次迭代，不使用 graph capture，同时保存 out-of-place 与 in-place 结果。

主实验包含 **12 轮原生运行、264 条分阶段测量、44 行前后对照**。[comparison.csv](data/comparison.csv) 保存每个消息大小的对比。不要把混合消息大小的 `Avg bus bandwidth` 与大消息平台值直接比较，也不要把归一化的 AllReduce `busbw` 当作原始 XGMI 链路吞吐。

## 为什么只测容器会漏掉这个问题

三个实验的边界不同：

1. **旧驱动上只换容器用户态**：ROCm 6.4.4/RCCL 2.22.3 和 ROCm 7.2.4/RCCL 2.27.7 都正常；这没有验证新内核驱动。
2. **宿主机原生升级并重启**：两种进程布局都复现了主表中的低带宽。
3. **保持相同不可变用户态镜像，仅在新宿主机驱动上运行**：连旧用户态也变慢。下表的旧驱动数值是三轮中位数，新驱动数值各来自**一轮额外诊断**，不是原生主表结果。

<!-- BEGIN GENERATED: driver-isolation -->
| 保持不变的用户态镜像 | 驱动 6.12.12，GB/s | 驱动 6.16.13，GB/s |
|---|---:|---:|
| ROCm 6.4.4 / RCCL 2.22.3 | 314.76 | 37.44 |
| ROCm 7.2.4 / RCCL 2.27.7 | 321.67 | 41.70 |
<!-- END GENERATED: driver-isolation -->

保留了[不可变镜像 ID 和诊断测量](data/native-summary.json)及[原始输出](evidence/fixed-images-new-driver/)。这些结果支持优先排查**宿主机驱动变化及其固件、虚拟化和互连交互**，不能据此说 RCCL 2.27.7 单独必然慢。

## 新内核异常是线索，还不是已证明的代码根因

新驱动启动时出现：

```text
psp gfx command INVOKE_CMD(0x3) failed and response status is (0xFFFF000F)
index 8 is out of range for type 'xgmi_connected_port_num [8]'
psp_xgmi_get_topology_info
amdgpu_xgmi_add_device
```

<!-- BEGIN GENERATED: kernel-symptoms -->
| 在完整私有启动日志中检索的模式 | 旧驱动启动 | 新驱动启动 |
|---|---:|---:|
| `xgmi_connected_port_num` | 0 | 4 |
| `psp_xgmi_get_topology_info` | 0 | 4 |
| `INVOKE_CMD(0x3) failed` | 0 | 8 |
<!-- END GENERATED: kernel-symptoms -->

参见[旧启动脱敏摘录](evidence/before/kernel-xgmi-excerpt.txt)、[新启动摘录](evidence/after/kernel-xgmi-excerpt.txt)和[检索计数](data/kernel-symptom-counts.json)。计数来自完整原始日志，公开摘录省略了无关行。

**新旧驱动的拓扑输出仍把链路标成 `XGMI`。**链路标签、能枚举出 GPU 或 `#wrong = 0` 都不能证明传输性能正常。异常与低带宽同时出现，但尚未证明某一处数组访问造成了性能下降。没有采集物理宿主机的 GIM/固件详情，**也尚未测试“只回退驱动后带宽恢复”**。

## 按顺序原生复现

### 0. 使用独立、可恢复的测试 VM

实测环境是 Ubuntu 24.04、Azure `Standard_ND96isr_MI300X_v5`。初始状态应为原生 ROCm 6.4.4/RCCL 2.22.3、已加载驱动 6.12.12。先停止自己的 GPU 工作负载，不要修改其他用户正在运行的任务。

该升级组合已复现低性能。只在可丢弃测试 VM 和维护窗口操作，准备经过验证的管理恢复入口、系统盘快照和精确旧包备份。**不要为了重启而 deallocate 稀缺 Spot GPU VM**，之后可能无法重新分配；快照和运行中的 VM 均会产生费用。

在已经登录的**隔离 Azure CLI 环境**创建快照：

```powershell
$env:AZURE_CONFIG_DIR = "C:\Temp\az-mi300x-case"
$subscription = "<YOUR_SUBSCRIPTION_ID>"
$resourceGroup = "<YOUR_RESOURCE_GROUP>"
$vm = "<YOUR_VM_NAME>"
$disk = az vm show --subscription $subscription --resource-group $resourceGroup --name $vm --query storageProfile.osDisk.managedDisk.id -o tsv
az snapshot create --subscription $subscription --resource-group $resourceGroup --name mi300x-pre-driver-upgrade --source $disk --incremental true --sku Standard_LRS
az snapshot show --subscription $subscription --resource-group $resourceGroup --name mi300x-pre-driver-upgrade --query provisioningState -o tsv
```

成功标准：快照状态为 `Succeeded`。失败时**不要安装新驱动**。快照引用仅保存在私有位置，不要提交。

### 1. 采集并运行宿主机原生基线

在 Linux VM 的本目录执行，不是在容器内：

```bash
export WORKDIR="$PWD/local-runs/native-ab"
# 两阶段使用同一个现有 MPI 安装。
export MPI_HOME="/opt/hpcx-v2.23-gcc-doca_ofed-ubuntu24.04-cuda12-x86_64/hpcx-rebuild"
sudo -E bash scripts/run_native.sh before "$WORKDIR"
```

成功标准：六轮输出 `NATIVE_PHASE_BENCHMARK_PASS`，校验全部通过，`before/host-metadata.json` 记录实际加载的驱动 6.12.12。按机器上的 MPI 路径设置 `MPI_HOME`，两阶段不要更换。源码、库来源、原生执行或数据校验检查失败时，查看 `WORKDIR/before` 并停止升级。

### 2. 显式执行原生升级，然后重启

```bash
export SNAPSHOT_REFERENCE="<YOUR_VERIFIED_PRIVATE_OS_SNAPSHOT_REFERENCE>"
sudo -E bash scripts/upgrade_host.sh --apply "$WORKDIR"
```

脚本先备份旧包版本与 APT/DKMS/启动状态，下载官方安装器，使用签名仓库并锁定实测包版本；事务模拟会拒绝删除无关软件或改动内核、SSH、Azure Agent、RDMA 组件。安装位置是 **VM 操作系统**：

```text
amdgpu-dkms=1:6.16.13.30300400-2341068.24.04
rocm=7.2.4.70204-93~24.04
rccl=2.27.7.70204-93~24.04
rccl-dev=2.27.7.70204-93~24.04
```

重启前成功标准：`INSTALLATION_COMPLETE_REBOOT_REQUIRED`。磁盘上的新模块**不等于内核实际加载的新模块**。下载、事务模拟、包安装或 DKMS 编译失败时，保留日志，不要启动未经验证的驱动。

原地重启 VM：

```bash
sudo reboot
```

使用原有 SSH 端口和凭据重新连接，检查：

```bash
cat /sys/module/amdgpu/version
cat /sys/module/amdgpu/srcversion
cat /proc/sys/kernel/random/boot_id
dkms status
/opt/rocm/bin/rocm-smi --showproductname
/opt/rocm/bin/rocm-smi --showcomputepartition --showmemorypartition
```

成功标准：实际加载 6.16.13、新 boot ID、与安装模块一致的 `srcversion`，8 张卡仍为 `SPX / NPS1`。本案例 GPU 初始化需要几分钟。SSH 或 GPU 初始化失败时，使用预先验证的管理恢复通道和快照/旧包；本目录刻意**不把未经实测的自动回退脚本当成已验证修复**。

### 3. 对新宿主机栈重新编译并测试

```bash
sudo -E bash scripts/run_native.sh after "$WORKDIR"
```

成功标准：确认实际原生 RCCL 2.27.7、内核加载驱动 6.16.13及真实重启；内核和 MPI 与基线一致，六轮正确性校验通过。低带宽本身是有效实验结果，**不能为了让结果好看而丢弃正确运行**。分区或拓扑变化时，先记录变化，再判断原因。

`local-runs/` 下的原始 JSON 和文本日志已被 Git 忽略，可能含环境标识。分享前应生成脱敏副本；不要提交快照、凭据、备份或私有运行目录。

## 下一步控制变量实验

以下是**尚未测出的结论**：

1. 保持 ROCm 7.2.4/RCCL 2.27.7，只回退内核驱动栈，重启并验证带宽能否恢复；同时记录依赖和固件包变化。
2. 在有回退保护的新测试 VM 上，对比中间的受支持驱动版本；固定用户态、内核、分区、源码和 MPI，找出首个坏版本。
3. 设置 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM`，比较实际通道与选用算法；不能仅凭某个环境变量值推断物理通信路径发生了变化。
4. 请 AMD/Azure 工程团队将 XGMI/PSP 症状与 VF/GIM/BKC 兼容性对应；不要从客户 VM 中修改或声称控制 Azure 的物理宿主机/GIM。
5. 驱动与 VF 的交互问题清楚后，再隔离内核代码修复，并在相同硬件上证明 **A/B/A** 恢复。

## 无需 GPU 的验证

只需 Python 3.10+ 标准库：

```bash
python3 tools/verify_case.py --check
python3 -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
```

成功标准：原始测量、实际加载驱动与重启证据、固定镜像诊断，以及中英文 README 的数字全部一致。仅重新生成派生表格和 CSV：

```bash
python3 tools/verify_case.py --write
python3 tools/verify_case.py --check
```

不要修改原始证据来使检查通过。

## 证据和公开边界

公开的 JSON/文本是**真实 rccl-tests 输出和内核日志的脱敏副本**，不是模拟结果。掩码覆盖主机/IP/账号/云资源标识、boot UUID、hive/MAC 标识及内存地址；公开文本副本去除行末空白。测量数值、软件版本、测试参数和错误码不变，私有原件单独保留。

本目录不发布密码、SSH 密钥、租户/订阅 ID、快照引用、模型权重、安装包归档或私有主机信息。原实验保留了升级后的低带宽现场，没有宣称未做过的降级恢复。

## 官方参考

- [ROCm 7.2.4 原生 Ubuntu 安装](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/install-methods/package-manager/package-manager-ubuntu.html)
- [AMDGPU 30.30.4 原生 Ubuntu 安装](https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.30.4/install/detailed-install/package-manager/package-manager-ubuntu.html)
- [用户态和内核驱动兼容矩阵](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/reference/user-kernel-space-compat-matrix.html)
- [RCCL 2.27.7 环境变量](https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.4/api-reference/env-variables.html)
- [锁定的官方 rccl-tests 源码](https://github.com/ROCm/rocm-systems/tree/1284e5da11f7904e8a6efd902b9de52253785bd0/projects/rccl-tests)
