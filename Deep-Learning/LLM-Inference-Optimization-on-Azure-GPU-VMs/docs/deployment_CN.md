# 部署细节与运维

[English](deployment.md) · [README](../README_CN.md) · [优化原理](optimizations_CN.md) · **部署细节** · [复测与核对](benchmark_CN.md) · [参考](reference_CN.md) · [实测结果](results_CN.md)

本页补充 [README 操作步骤](../README_CN.md#操作步骤)的细节：每一步做什么、为什么这样做，怎样确认优化已生效，日常运维，哪些步骤实际跑过，以及容易误判的故障。照着执行的命令顺序以 README 为准。

## 部署步骤详解

用本仓库的 Dockerfile 部署优化后的栈：镜像只构建一次，推到你自己的镜像仓库，每个服务都作为一个独立容器从这个镜像启动。两条路线用同一个镜像。**单机**：一个 TP8 服务，用最终 runtime 的配置。**两台 VM 的 1P1D**：实测吞吐背后的 prefill/decode 拓扑。不用 GPU 核对已发布数字是另一条路径，见[测试与离线校验](benchmark_CN.md#测试与离线校验)。

**1. 检查每台 VM。** Azure ND MI300X v5，已装好 ROCm 驱动和 Docker（含 BuildKit）。

```bash
rocm-smi --showproductname   # 8 个条目 GPU[0] ... GPU[7]，型号为 MI300X
ibv_devinfo -l               # 1P1D：8 个 RDMA 设备，mlx5_ib0 ... mlx5_ib7
show_gids                    # 1P1D：确定 MC_GID_INDEX 用的 GID index（实测 VM 上是 3）
df -h /mnt/models            # 本地 NVMe 卷上留出权重的空间（约 1 TB）
docker buildx version
```

权重从 VM 的本地 NVMe 卷加载：每次启动容器都要把权重完整读一遍。这类 VM 的本地 NVMe 是临时存储，VM deallocate 后数据就没了，所以持久副本要放在别处（例如 Azure Blob Storage），每次分配 VM 后再拷到 NVMe 上。

**2. 在每台 VM 上取得部署文件和模型。** 下载固定在版本 `7208274`：这是最后一次改动权重或建模代码的 commit（之后的提交只改了模型卡），这样每台 VM、以后每次重新部署加载的都是同一批文件。启动参数里有 `--trust-remote-code`，因为模型自带建模代码；请审阅你固定的那个版本里的代码。

```bash
git clone --filter=blob:none --sparse https://github.com/david-xinyuwei/david-share.git
cd david-share && git sparse-checkout set Deep-Learning/LLM-Inference-Optimization-on-Azure-GPU-VMs
cd Deep-Learning/LLM-Inference-Optimization-on-Azure-GPU-VMs
export MODELS=/mnt/models
export MODEL_REPO=        # 模型在 Hugging Face 上的仓库 id，见其模型卡
hf download "$MODEL_REPO" --revision 7208274e94787a92d7261675fb1de56ccb0e94b5 --local-dir "$MODELS/MiMo-V2.5-Pro"
cp docker/mimo.env.example docker/mimo.env
```

填好 [`docker/mimo.env`](../docker/mimo.env.example)，把同一份文件复制到每台 VM。服务要读取的设置都在里面：镜像 digest（第 3 步得到）、容器内的模型路径；1P1D 还需要两台 VM 的地址、RDMA 设备和 GID index。下面的主机命令用 `set -a; . docker/mimo.env; set +a` 加载它，容器则通过 `--env-file` 读取。

**3. 镜像只构建一次，推到镜像仓库。** 在任意一台装有 BuildKit 的 x86-64 机器上执行，用其中一台 VM 也可以：

```bash
export ACR_NAME=          # 你的 Azure Container Registry
REGISTRY="$ACR_NAME.azurecr.io"
az acr login --name "$ACR_NAME"
docker buildx build --platform linux/amd64 --tag "$REGISTRY/mimo-mi300x:$(git rev-parse --short=12 HEAD)" \
  --metadata-file build-meta.json --push docker/
DIGEST=$(python3 -c "import json; print(json.load(open('build-meta.json'))['containerimage.digest'])")
echo "IMAGE=$REGISTRY/mimo-mi300x@$DIGEST"   # 把这一行写进每台 VM 的 docker/mimo.env
```

然后在每台 VM 上：`set -a; . docker/mimo.env; set +a; docker pull "$IMAGE"`。

[Dockerfile](../docker/Dockerfile) 从按 digest 固定的公开基础镜像开始，检出 SGLang `878fff1`、AITER `3f4ab48`（含 Composable Kernel `af7118e` 及其自带 patch），从 PyPI 安装 FlyDSL `0.2.4`，并取 `c99d5cd` 的 FlyDSL kernel；wheel 哈希、CK commit 或最后的 import 检查任何一项对不上，构建都会停止。tag 记录源码 commit；按 digest 运行，保证两台 VM 以及之后每次重启用的都是同一个镜像。基础镜像已缓存时，干净构建约 6 分钟，镜像 27.9 GB。

**4. 渲染启动脚本。** profile 把实测配置转成每个角色一个脚本；主机、路径和设备都保留为变量，由容器从 `docker/mimo.env` 读取。

单机：

```bash
mkdir -p run
python3 tools/render_launch.py --profile rocm-mi300x-single --role server > run/server.sh
```

1P1D（在每台 VM 上渲染，或渲染一次后复制 `run/`）：

```bash
mkdir -p run
python3 tools/render_launch.py --profile rocm-mi300x-pd --role prefill --ablate simulated-acceptance > run/prefill.sh
python3 tools/render_launch.py --profile rocm-mi300x-pd --role decode --ablate simulated-acceptance > run/decode.sh
python3 tools/render_launch.py --profile rocm-mi300x-pd --role router > run/router.sh
```

这些是上线服务用的配置，不是压测配置。`--ablate simulated-acceptance` 去掉了测量时用的固定 MTP 接受长度，MTP 只保留模型真正接受的草稿 token。单机的 `server` 角色本来就按实际接受运行，并且关闭了 Quick Reduce。1P1D 脚本保留了和实测一致的 INT8 Quick Reduce；在你的负载上[核对准确率](optimizations_CN.md#哪些优化可能改变模型输出)之前，可以加 `--ablate int8-quick-reduce` 改为全精度求和。

**5. 启动服务。** 单机：

```bash
set -a; . docker/mimo.env; set +a
bash docker/run-role.sh server run/server.sh
bash docker/wait-ready.sh 127.0.0.1 30001 3600 mimo-server
```

1P1D：

```bash
set -a; . docker/mimo.env; set +a
bash docker/run-role.sh prefill run/prefill.sh                         # 在 VM A 上
bash docker/run-role.sh decode run/decode.sh                           # 在 VM B 上
bash docker/wait-ready.sh "$PREFILL_HOST" 30000 3600 mimo-prefill      # 在 VM A 上
bash docker/wait-ready.sh "$DECODE_HOST" 30001 3600                    # 在 VM A 上
bash docker/run-role.sh router run/router.sh                           # 在 VM A 上，两个都打印 READY 之后再启动
```

[`docker/run-role.sh`](../docker/run-role.sh) 把每个角色启动为一个有名字的容器（`mimo-server`、`mimo-prefill`、`mimo-decode`、`mimo-router`），容器的主进程就是服务本身：失败时最多自动重启 3 次，日志自动轮转，模型只读挂载，设置从 env 文件读取。只有 GPU 角色才拿到 RDMA 和 AITER 需要的宿主机权限（`--privileged`、宿主机 IPC、`/dev/kfd`、`/dev/dri`、`/dev/mem`、`CAP_SYS_ADMIN`），所以只在专用的 GPU 虚拟机上运行。[`docker/wait-ready.sh`](../docker/wait-ready.sh) 轮询不会生成 token 的 `/server_info`，打印服务报告的 KV 容量；如果指定的容器已经退出，就提前结束等待。容器第一次启动时要编译 AITER JIT kernel 并加载权重，需要几十分钟。

所有容器都用宿主机网络，服务监听所有网卡。30000 和 30001 端口只放行两台 VM 私网地址之间的访问；40000 端口同样不要对外开放，VM 之外的客户端需要访问时，在前面加一个带认证的网关。

**6. 确认优化已生效，再发一个请求。** 单机：

```bash
docker logs mimo-server 2>&1 | grep -m1 module_gemm_a8w8_blockscale_bpreshuffle   # CK block-scale GEMM
docker logs mimo-server 2>&1 | grep -m1 mimo_v2_5_pro_b16_tuned_fmoe              # 调优 fused-MoE 表
curl -s --retry 30 --retry-connrefused --retry-delay 10 http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number."}], "max_tokens": 1024, "temperature": 0}'
```

1P1D：prefill 服务的检查和请求在 VM A 上执行，decode 服务的检查在 VM B 上执行。

```bash
docker logs mimo-decode 2>&1 | grep -m1 module_gemm_a8w8_blockscale_bpreshuffle   # CK block-scale GEMM
docker logs mimo-prefill 2>&1 | grep -m1 mimo_v2_5_pro_b16_tuned_fmoe             # 调优 fused-MoE 表
docker logs mimo-prefill 2>&1 | grep -i mooncake | grep -m3 mlx5_ib                # 正在使用的 RDMA 设备
docker logs mimo-prefill 2>&1 | grep -i mooncake | grep -i -m3 tcp                 # 应当没有输出
curl -s --retry 30 --retry-connrefused --retry-delay 10 http://127.0.0.1:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number."}], "max_tokens": 1024, "temperature": 0}'
```

每项检查都输出注释里说的内容，并且回复中 `choices[0].message.content` 含有 `391`、`finish_reason` 为 `stop`，即为完成。

**7. 日常运维。**

- **日志：** `docker logs -f mimo-decode`。
- **重启：** `docker restart mimo-decode` 会保留容器里已编译好的 AITER kernel；`docker rm` 会把它们丢掉，下次启动要重新编译。
- **崩溃、卡死和重启机器：** 服务崩溃后容器会自动重启，最多 3 次。VM 或 Docker 重启后容器不会自动拉起，卡死的服务也不会被重启。无人值守运行时，用 systemd 或带就绪检查的编排工具来执行第 5 步的命令。
- **升级或回滚：** 构建并推送新镜像（第 3 步），更新 `docker/mimo.env` 里的 `IMAGE`。在每台 VM 上用 `docker rm -f` 删掉各角色容器，再重复第 5 步：先启动服务，最后启动路由。回滚就是换回之前的 digest。只有一组服务时，这就是一次停机维护：进行中的请求会失败，新容器就绪之前服务暂停。要避免停机，就用新 digest 再起一组服务，检查通过后再切换路由。
- **停止：** 在 VM A 上 `docker stop mimo-router mimo-prefill`，在 VM B 上 `docker stop mimo-decode`；单机则是 `docker stop mimo-server`。这会保留容器及其可写层。
- **可选的删除或 VM deallocate：** 这是另外需要操作者明确授权的动作，不是自动收尾。删除容器或释放 VM 前先确认持久备份；deallocate 可能清空本地 NVMe，而只在 VM 内部关机不会释放计算分配。

## 哪些步骤实际跑过

在一台只有 CPU、装有 Docker 的虚拟机上跑过：第 3 步的镜像构建（干净环境，用 `--metadata-file` 和 `--load` 代替 `--push`，见 [`evidence/docker-build.json`](../evidence/docker-build.json) 和 [`evidence/deploy-check.json`](../evidence/deploy-check.json)）；用 `run-role.sh` 从渲染出的脚本启动路由角色，路由就是容器的主进程，`docker stop` 0.5 秒返回；`wait-ready.sh` 对一个模拟的 `/server_info` 的检查；以及压测角色的变量传递和只读模型挂载。还没有跑过：推送镜像和按 digest 拉取、GPU 角色（server、prefill、decode），以及第 6 步的检查和请求。这些步骤依据的是记录下来的启动配置。另外，这个 Dockerfile 构建的 runtime 比实测吞吐所用的栈更新，各测量小节都写明了所用的栈。

## 容易误判的故障

下面是在 MI300X 上把这套栈跑起来时的运维观察，不是实测结果；只有图捕获这一项在本仓库有公开的数字。第一组只会让运行变慢，服务照常工作。

- **KV 传输退回 TCP。** 容器没有 `--privileged`、`/dev/mem` 和 `CAP_SYS_ADMIN` 时，Mooncake 会悄悄改用 TCP 而不是 RDMA。token 仍然正确，吞吐却会下降。把"RDMA 已初始化"作为启动检查项。
- **decode 服务关掉了图捕获。** 加了 `--disable-cuda-graph` 后，decode 服务在[基线 A/B](results_CN.md#从基线栈到优化后累计提升) 中只剩三分之一的吞吐，日志里没有任何异常。每次运行前，把每个启动脚本和已知正确的版本逐项比对。
- **镜像里有两份 kernel 库。** 同一个 AITER 版本从不同目录导入时表现不同。要检查导入路径和服务日志里的 kernel 名称，而不只是版本号。
- **调优表从未命中。** 对没有调优记录的 fused-MoE shape，AITER 会在日志里打印 `default`。启动时检查日志里关键 shape 的情况。

第二组会让服务停下或卡住，但原因容易看错：

- **多线程加载权重卡住。** 在 PD 服务上，多线程加载权重时各张量并行 rank 卡在缺页处理里；`--model-loader-extra-config '{"enable_multithread_load": false}'` 解决了这个问题。
- **prefill 服务上的重叠调度。** 它触发了 HIP 非法地址错误，所以吞吐测试都用了 `--disable-overlap-schedule`。
- **会生成 token 的健康检查。** SGLang 的 `/health` 默认会真的生成一个 token（`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`），频繁探测把 prefill 服务的 detokenizer 卡住了。把它设为 `0`，改为轮询 `/server_info`。
- **每步一次的 all-gather 在约 200K token 时超时。** 在数据并行度为 1 时，SGLang 已有的开关 `SGLANG_SCHEDULER_SKIP_ALL_GATHER=1` 可以避开它。
- **照搬其他 GPU 配置里的值。** 从另一套系统照搬的 `--chunked-prefill-size` 在启动时就失败了。这类值要按本 runtime 的限制推算。
