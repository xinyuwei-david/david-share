# 参考

[English](reference.md) · [README](../README_CN.md) · [优化原理](optimizations_CN.md) · [部署细节](deployment_CN.md) · [复测与核对](benchmark_CN.md) · **参考** · [实测结果](results_CN.md)

本页汇集供查阅的资料：本仓库提供什么、目录结构、启动 profile 与 `--ablate` 名称、指标与日志字段的定义、固定的上游提交和资料来源。

## 本仓库做了什么、提供什么

- **推理引擎与 kernel**——归上游 SGLang、AMD AITER、Composable Kernel、FlyDSL 项目；MiMo 专用的提交是 AMD 工程师在公开 fork 中完成的。这里提供固定的 commit 身份，以及文中讨论的每个 commit 的完整 patch（[`upstream/`](../upstream/)）。
- **优化方法与实测数据**——本仓库。MI300X 优化前后的实测对比，附原始压测输出的公开投影（[`evidence/`](../evidence/)）；逐项技术解读和代码摘录；每个数字的适用边界。
- **启动配置**——本仓库。实测 MI300X 栈的机器可读 profile 和一份 NVIDIA 模板（[`profiles/`](../profiles/)），可渲染成启动命令，并支持单项消融（[`tools/render_launch.py`](../tools/render_launch.py)）。
- **部署**——本仓库。从公开源码构建固定版本 runtime 的 Dockerfile、把每个服务角色作为独立容器启动的脚本、就绪检查和设置模板（[`docker/`](../docker/)）。
- **校验**——本仓库。离线测试和 CI，从已提交的证据重新算出每个发布的数字。

你需要自备：Azure ND MI300X v5 容量（PD 路线两台，单机路线一台）、MiMo-V2.5-Pro 权重，以及能访问 RDMA 的容器宿主机。

不提供：模型权重；证据投影背后的私有原始日志（只记录其 SHA-256）；与其他加速卡的比较；打开 FP8 KV cache 时的准确率结果；任何 NVIDIA 上的实测。每项优化会对准确率产生什么影响，见[哪些优化可能改变模型输出](optimizations_CN.md#哪些优化可能改变模型输出)；怎么核对，见[在你的模型上核对有损开关](benchmark_CN.md#在你的模型上核对有损开关)。

## 目录结构

- [`evidence/runs.json`](../evidence/runs.json)——每次运行的身份、拓扑、控制变量和脚本哈希。
- [`evidence/raw/`](../evidence/raw/)——各数据源的投影：`sglang.bench_serving` 输出（每次运行的负载参数和结果块）、调度器日志审计中的 A/B 采样值、长上下文和副本的结果表、逐条回答的准确率评分，以及基线栈的客户端结果和汇总。
- [`evidence/raw-manifest.json`](../evidence/raw-manifest.json)——每份私有原始日志及其公开投影的 SHA-256。
- [`evidence/measurements.json`](../evidence/measurements.json)——全部对比结果，由 `tools/build_evidence.py` 生成。
- [`evidence/docker-build.json`](../evidence/docker-build.json)——干净 Docker 构建的回执：commit、Dockerfile 哈希、构建器、镜像 id 以及日志里的各步记录。
- [`evidence/deploy-check.json`](../evidence/deploy-check.json)——在只有 CPU 的虚拟机上做的容器检查回执：跑了哪些部署步骤、观察到什么、哪些没跑，以及被测脚本的哈希。
- [`upstream/`](../upstream/)——文中讨论的每个 commit 的完整 patch，以及 `SOURCES.lock.json`（哈希、许可证、所属层、是否在固定 runtime 中）。
- [`profiles/`](../profiles/)——技术目录、实测的 MI300X profile 和 NVIDIA 模板。
- [`tools/`](../tools/)——日志投影与解析、证据和 README 生成器、启动命令渲染、画图、公开内容审计。
- [`tests/`](../tests/)——离线测试。
- [`docker/`](../docker/)——Dockerfile、按角色启动容器的脚本、就绪检查和设置模板。
- [`images/`](../images/)——示意图及其哈希台账。

## 启动 profile

`tools/render_launch.py` 按 profile 和角色生成启动脚本，`--ablate <名称>` 去掉一项技术，用来做单变量 A/B。`python tools/render_launch.py --list` 打印同样的 profile、角色和名称。

<!-- BEGIN GENERATED: launch-profiles -->
| Profile | 状态 | 角色 |
|---|---|---|
| `cuda-hopper-pd` | `TEMPLATE_NOT_MEASURED`<br>NVIDIA 对应模板；本仓库未运行 | `prefill` · `decode` · `router` · `bench-decode` · `bench-prefill` |
| `rocm-mi300x-pd` | `MEASURED`<br>阶段对比与并发阶梯（S1–S3）实测所用的 1P1D 配置 | `prefill` · `decode` · `router` · `bench-decode` · `bench-prefill` |
| `rocm-mi300x-single` | `PINNED_RUNTIME`<br>最终固定 runtime；未发布吞吐实测 | `decode` · `server` · `bench-decode` |
<!-- END GENERATED: launch-profiles -->

<!-- BEGIN GENERATED: ablation-names -->
| `--ablate` 名称 | 去掉的优化 | 会改变哪些角色的脚本 |
|---|---|---|
| `attention-dispatch` | [混合 SWA + GQA 的逐层 attention 分派](optimizations_CN.md#混合-swa--gqa-的逐层-attention-分派) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode`, `server` |
| `fp8-kv-5d` | [FP8 KV cache + 向量化 5D 分页布局](optimizations_CN.md#fp8-kv-cache--向量化-5d-分页布局) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode`, `server` |
| `unified-verify` | [MTP target verify 使用 AITER unified attention](optimizations_CN.md#mtp-target-verify-使用-aiter-unified-attention) | `rocm-mi300x-pd`: `decode` |
| `eagle-mtp` | [多层 EAGLE MTP 投机解码及校验修复](optimizations_CN.md#多层-eagle-mtp-投机解码及校验修复) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode`, `server` |
| `memory-sizing` | [Chunked prefill、page size 与 SWA 池容量](optimizations_CN.md#chunked-prefillpage-size-与-swa-池容量) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode`, `server` |
| `decode-graph-capture` | [Decode 图捕获（HIP graph）](optimizations_CN.md#decode-图捕获hip-graph) | `cuda-hopper-pd`: `decode`<br>`rocm-mi300x-pd`: `decode`<br>`rocm-mi300x-single`: `decode` |
| `flydsl-pa-decode` | [FlyDSL paged-attention decode kernel（head 192，page 64）](optimizations_CN.md#flydsl-paged-attention-decode-kernelhead-192page-64) | `rocm-mi300x-single`: `decode`, `server` |
| `ck-a8w8-gemm` | [权重预重排的 block-scale FP8 GEMM](optimizations_CN.md#权重预重排的-block-scale-fp8-gemm) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode`, `server` |
| `int8-quick-reduce` | [张量并行 all-reduce 使用 INT8 Quick Reduce](optimizations_CN.md#张量并行-all-reduce-使用-int8-quick-reduce) | `rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode` |
| `tuned-fused-moe` | [按 shape 调优的 fused-MoE kernel 表](optimizations_CN.md#按-shape-调优的-fused-moe-kernel-表) | `rocm-mi300x-pd`: `prefill`, `decode` |
| `pd-disaggregation` | [Prefill/Decode 分离（1P1D），KV 走 RDMA](optimizations_CN.md#prefilldecode-分离1p1dkv-走-rdma) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode` |
| `fake-prefill` | [Fake prefill：只测 decode](optimizations_CN.md#fake-prefill只测-decode) | `rocm-mi300x-single`: `decode`, `bench-decode` |
| `simulated-acceptance` | [性能测试固定 MTP 接受长度](optimizations_CN.md#性能测试固定-mtp-接受长度) | `cuda-hopper-pd`: `prefill`, `decode`<br>`rocm-mi300x-pd`: `prefill`, `decode`<br>`rocm-mi300x-single`: `decode` |

- 在包含某个名称的 profile 中，表里没有列出的角色也接受这个名称，但渲染出的脚本只有注释行变化。profile 不包含的名称会被 `render_launch.py` 直接拒绝。
- `cuda-hopper-pd` 里的 `tuned-fused-moe` 只是一条注释：Triton fused-MoE 配置要自己生成并放入。
- 在包含 `flydsl-pa-decode` 的 profile 中，消融 `fp8-kv-5d` 时必须同时消融依赖它的 `flydsl-pa-decode`。
<!-- END GENERATED: ablation-names -->

## 指标与日志字段

客户端记录负载参数和 Serving Benchmark Result 块。`tools/bench_log.py` 提取、解析这些字段，`tools/build_evidence.py` 校验哈希及可接受的对照关系。服务返回 HTTP 响应本身不是准确率结果。

RCCL 的消息大小按二进制计算：1G 即 1 GiB。algbw 是消息字节数除以耗时；N 个进程的 AllReduce busbw 等于 algbw × 2(N−1)/N。工具报告的平均值包含 in-place 与 out-of-place，不是多次重跑的均值，也不是网卡利用率。`-c 0` 不校验数据；声明 IB/GDRDMA 前必须有对应 INFO 日志。解析器处理了 INFO 文本与错误计数粘连的情况，不会把两种模式的结果对调。

## 上游提交

<!-- BEGIN GENERATED: upstream-table -->
- [`ROCm/FlyDSL@e46db60`](https://github.com/ROCm/FlyDSL/commit/e46db6020b4560de82a7136d78cc33a5186338f4) — [Kernel][PA] Support BF16 vectorized KV with asymmetric K/V head dim (#1065)  
  算子层 · 上游后续提交，不在固定 runtime 中
- [`ROCm/FlyDSL@ed9885e`](https://github.com/ROCm/FlyDSL/commit/ed9885eca4ffc45e2ec1dc45fa00824baa6b56d3) — [Bugfix][PA] Fix D192 query staging and 64-bit cache offsets (#1064)  
  算子层 · 上游后续提交，不在固定 runtime 中
- [`sammysun0711/FlyDSL@c99d5cd`](https://github.com/sammysun0711/FlyDSL/commit/c99d5cd97864c11e459cff9169d387d312790782) — Fix head-192 PA decode accuracy and long-context for MiMo-V2.5-Pro  
  算子层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/aiter@10a9401`](https://github.com/sammysun0711/aiter/commit/10a94012efc1260dfdf16ba2f52fbda40a518a17) — fix(pa_decode_gluon): correct head-192 persistent MTP decode  
  算子层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/aiter@3f4ab48`](https://github.com/sammysun0711/aiter/commit/3f4ab482a2986919c784e469e23cfac7f93bb153) — feat (MHA): Optimize MiMo FP8 page-64 batch prefill with a head-192 CK specialization  
  算子层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/aiter@d725746`](https://github.com/sammysun0711/aiter/commit/d725746a0f8c233d8e46e2771a7c8dbcd06e40d9) — add tuend moe (#5)  
  算子层 · 在实测 runtime 中
- [`sammysun0711/aiter@fc96a4f`](https://github.com/sammysun0711/aiter/commit/fc96a4f9f5f3e931cbb9de275c8aa01136417500) — Add MiMO-v2.5-Pro ck a8w8 blockscale gemm tuned config on MI300X (gfx942 304 CU)  
  算子层 · 在实测 runtime 中
- [`sammysun0711/sglang@0cfc48b`](https://github.com/sammysun0711/sglang/commit/0cfc48b0e374d7e84c122f739182a39feea56d46) — feat(MHA): Route MiMo cached prefill directly to AITER page-64 attention  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/sglang@1f9bb2b`](https://github.com/sammysun0711/sglang/commit/1f9bb2b4c55cdc7bd5de1ac7977f76afab101a97) — (feat): Add opt-in mixed-precision Triton router GEMM for MiMo prefill  
  算子层 · 后续分支，不在任何 runtime 中
- [`sammysun0711/sglang@2f9b9ae`](https://github.com/sammysun0711/sglang/commit/2f9b9aedf32977bc5d088a86ec0a73bcf432a4d0) — Add env variable SGLANG_USE_AITER_CK_BLOCKSCALE and SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE to use ck a8w8 gemm on gfx942 instead of triton a8w8 gemm  
  推理框架层 · 在实测 runtime 中
- [`sammysun0711/sglang@78cd40c`](https://github.com/sammysun0711/sglang/commit/78cd40c7a5102524536daf9a3178426777174d2d) — fix(speculative): isolate AITER MTP draft KV layout and graph metadata  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/sglang@878fff1`](https://github.com/sammysun0711/sglang/commit/878fff15647fe3dabb32aa3a335b0ad16e3ee878) — bugfix(MTP): Fix HIP non-greedy EAGLE verification for MTP topk=1 (tree_topk) speculative decoding  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/sglang@a2fd773`](https://github.com/sammysun0711/sglang/commit/a2fd773ab43f960f5f2c29b5c592b0ca43c5ba8f) — feat(flydsl pa decode): add configurable FlyDSL partition counts  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/sglang@ba15db1`](https://github.com/sammysun0711/sglang/commit/ba15db1a576dcdc8d51ba15bd069b9fd1f748d97) — feat: Add opt-in FlyDSL paged decode for MiMo EAGLE verification  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/sglang@db840d9`](https://github.com/sammysun0711/sglang/commit/db840d935a9f7097dbeb5f1b0dba4d261057a2bd) — [AMD] Fix aiter SWA handling in draft_extend_v2 for MTP speculative decoding  
  推理框架层 · 在实测 runtime 中
- [`sammysun0711/sglang@e11c515`](https://github.com/sammysun0711/sglang/commit/e11c5155f0845079211c2a4d0b8a4ab3669039f9) — feat(aiter attention backend): use Gluon PA for vectorized-5D target verification  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
- [`sammysun0711/sglang@f26ae30`](https://github.com/sammysun0711/sglang/commit/f26ae30063143411f3ae552af1830fa46e3ee0fd) — fix(EAGLE verification): sync result across TP ranks  
  推理框架层 · 在固定 runtime 中，本仓库未测吞吐
<!-- END GENERATED: upstream-table -->

## 资料

[SGLang](https://github.com/sgl-project/sglang) · [AITER](https://github.com/ROCm/aiter) · [FlyDSL](https://github.com/ROCm/FlyDSL) · [Composable Kernel](https://github.com/ROCm/composable_kernel) · [Mooncake](https://github.com/kvcache-ai/Mooncake) · [Azure ND MI300X v5 系列](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ndmi300xv5-series)。各 patch 保留其上游许可证（SGLang 与 FlyDSL 为 Apache-2.0，AITER 为 MIT），详见 [`upstream/SOURCES.lock.json`](../upstream/SOURCES.lock.json)。
