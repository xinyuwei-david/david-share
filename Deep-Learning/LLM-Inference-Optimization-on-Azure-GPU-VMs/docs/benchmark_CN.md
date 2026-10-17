# 复测与核对

[English](benchmark.md) · [README](../README_CN.md) · [优化原理](optimizations_CN.md) · [部署细节](deployment_CN.md) · **复测与核对** · [参考](reference_CN.md) · [实测结果](results_CN.md)

本页用于复测和核对：复现已发布的测点、测量一项改动、了解这些数字背后的测量要点、在你的模型上核对有损开关，以及不用 GPU 离线核对已发布的数字。

## 复现已发布的数字

已发布的吞吐是在 MTP 固定接受长度 3 的条件下测的。要和它比较，渲染 prefill 和 decode 时不要加 `--ablate simulated-acceptance`，重启这两个容器，然后在 VM A 上把压测客户端作为一次性容器运行：

```bash
python3 tools/render_launch.py --profile rocm-mi300x-pd --role bench-decode > run/bench-decode.sh
python3 tools/render_launch.py --profile rocm-mi300x-pd --role bench-prefill > run/bench-prefill.sh
CONCURRENCY=64 bash docker/run-role.sh bench run/bench-decode.sh | tee decode_c64.log
INPUT_LEN=8192 bash docker/run-role.sh bench run/bench-prefill.sh | tee prefill_8k.log
python3 tools/bench_log.py project decode_c64.log -o my_decode_c64.txt
python3 tools/bench_log.py parse my_decode_c64.txt
```

`Successful requests` 等于 prompt 数量（decode 256、prefill 16）只是必要条件。接受一个测点前，还要排除错误响应，核对实际输入和输出长度、上下文余量，并检查服务日志。固定接受模式不对文本准确率评分，但仍须排除协议或执行错误。不加 `--dataset-path` 时，`bench_serving` 会下载它用来采样文本的 ShareGPT 文件，离线主机需要准备本地副本。这套配置只用于测量。

## 测量一项改动

渲染同一个角色、去掉一项技术，只替换这一个容器，每组用完全相同的客户端命令跑两次。下面的例子在 1P1D 的 decode 服务上去掉已发布 A/B 的两个开关；它只是演示方法，并没有重复 [`evidence/runs.json`](../evidence/runs.json) 里记录的单机 64K 实验：

```bash
python3 tools/render_launch.py --profile rocm-mi300x-pd --role decode --ablate ck-a8w8-gemm --ablate unified-verify > run/decode-baseline.sh
docker rm -f mimo-decode && bash docker/run-role.sh decode run/decode-baseline.sh
```

## 测量要点

并发阶梯怎样设计、decode batch 怎样按 KV 容量估算，和对应技术放在一起，见[按饱和点设计并发阶梯](optimizations_CN.md#按饱和点设计并发阶梯)和[按 KV 容量估算 decode batch](optimizations_CN.md#按-kv-容量估算-decode-batch)。

### 精确输入与上下文余量

随机 prompt 压测会把文本重新分词，服务端的实际长度可能漂移。长上下文测点应直接传 token ID（`--tokenize-prompt`），并把模型的特殊 token 算进去：MiMo 会加 4 个，所以传 65,532 个 ID 才是服务端正好 65,536 个 token。`--context-length` 也要留同样的余量，否则最长的测点可能带着错误内容被记为「成功」，[阶段对比](results_CN.md#阶段对比按-shape-调优的-fused-moe-表)剔除 256K 点正是这个原因。

### 全新服务重复测

每个被采纳的 A/B 组都在全新启动的服务上跑两次。两次结果相差约 1% 以内（见 [A/B 表](results_CN.md#受控-ab64k-上下文下的-block-scale-fp8-gemm-路径)），才能说 25% 的差距是真的；每组只跑一次是说明不了问题的。测一个新的上下文长度时，先测 1 个请求，再测几个依次发送，再测几个同时发送，最后在全新服务上重复；中途失败的点作为无效点发布，不做估算。

## 在你的模型上核对有损开关

**kernel 级数值检查。** 上游提交为每个 MiMo 专用 kernel 加了测试，在 MiMo 的 shape（head 192、page 64、FP8 KV、query 长度 4）上与 PyTorch 参考实现比对，其中包括 2 GiB 偏移的用例。这些测试需要 MI300X，本仓库没有运行：

<!-- BEGIN GENERATED: numerical-tests -->
- [`ROCm/FlyDSL@e46db60`](https://github.com/ROCm/FlyDSL/commit/e46db6020b4560de82a7136d78cc33a5186338f4) `tests/kernels/test_pa.py::test_tile_pa_vectorized_5d_matches_torch`
- [`ROCm/FlyDSL@e46db60`](https://github.com/ROCm/FlyDSL/commit/e46db6020b4560de82a7136d78cc33a5186338f4) `tests/kernels/test_pa.py::test_pa_decode_ps_rejects_unsupported_bf16_asymmetric_paths`
- [`ROCm/FlyDSL@e46db60`](https://github.com/ROCm/FlyDSL/commit/e46db6020b4560de82a7136d78cc33a5186338f4) `tests/kernels/test_pa.py::test_pa_decode_ps_rejects_non_divisible_gqa_heads`
- [`ROCm/FlyDSL@ed9885e`](https://github.com/ROCm/FlyDSL/commit/ed9885eca4ffc45e2ec1dc45fa00824baa6b56d3) `tests/kernels/test_pa.py::test_fp8_head_dim_192_matches_torch`
- [`ROCm/FlyDSL@ed9885e`](https://github.com/ROCm/FlyDSL/commit/ed9885eca4ffc45e2ec1dc45fa00824baa6b56d3) `tests/kernels/test_pa.py::test_fp8_cache_offset_above_2gib`
- [`sammysun0711/FlyDSL@c99d5cd`](https://github.com/sammysun0711/FlyDSL/commit/c99d5cd97864c11e459cff9169d387d312790782) `tests/kernels/test_pa.py::test_mimo_v25_pro_head_192_accuracy`
- [`sammysun0711/aiter@10a9401`](https://github.com/sammysun0711/aiter/commit/10a94012efc1260dfdf16ba2f52fbda40a518a17) `op_tests/triton_tests/test_pa_decode_gluon.py::test_mimo_head_192_full_context_regression`
- [`sammysun0711/aiter@3f4ab48`](https://github.com/sammysun0711/aiter/commit/3f4ab482a2986919c784e469e23cfac7f93bb153) `op_tests/test_batch_prefill.py::test_batch_prefill_mimo_fp8_vectorized_page64`
<!-- END GENERATED: numerical-tests -->

kernel 测试证明的是 kernel 在测过的 shape 上与参考实现一致，并不能衡量 FP8 存储或 INT8 求和对端到端答案的影响；那需要模型级的检查。

**模型级检查（本仓库未运行）。** 用固定 runtime 的准确率角色（真实 MTP 接受、Quick Reduce 关闭）。这组 FP8 KV 对照不是严格单变量：B 组切换为 BF16 KV，同时将 target verify 从 FlyDSL 切回 AITER。

```bash
# A 组：FP8 KV cache（线上配置）
python tools/render_launch.py --profile rocm-mi300x-single --role server > arm_a.sh
# B 组：BF16 KV cache。FlyDSL decode 依赖 FP8 KV，渲染器会要求两者一起去掉。
python tools/render_launch.py --profile rocm-mi300x-single --role server --ablate fp8-kv-5d --ablate flydsl-pa-decode > arm_b.sh
# 独立的 Quick Reduce 对照：A 组不变，只修改副本中生成的 export。
cp arm_a.sh arm_qr_int8.sh
sed -i 's/^export ROCM_QUICK_REDUCE_QUANTIZATION=NONE$/export ROCM_QUICK_REDUCE_QUANTIZATION=INT8/' arm_qr_int8.sh
grep -Fx 'export ROCM_QUICK_REDUCE_QUANTIZATION=NONE' arm_a.sh
grep -Fx 'export ROCM_QUICK_REDUCE_QUANTIZATION=INT8' arm_qr_int8.sh
```

按服务启动流程分别运行各组，评分前检查服务进程的实际环境。只在外层 shell 中 export 不够，因为生成的启动脚本会覆盖它。

```bash
SERVER_CONTAINER=mimo-server
docker exec "$SERVER_CONTAINER" bash -c \
  'tr "\000" "\n" < /proc/1/environ | grep "^ROCM_QUICK_REDUCE_QUANTIZATION="'
```

A 组应输出 `NONE`，Quick Reduce 组应输出 `INT8`；两次值相同就不能比较分数。

对每一组，在对应的 `SERVER_CONTAINER` 内使用固定镜像的 Python 环境运行评测工具，temperature 0，数据集为公开数据集：

```bash
docker exec "$SERVER_CONTAINER" /opt/venv/bin/python -m sglang.test.run_eval --port 30001 --eval-name gsm8k --num-examples 1319
docker exec "$SERVER_CONTAINER" /opt/venv/bin/python -m sglang.test.run_eval --port 30001 --eval-name mmlu --num-examples 2000
docker exec "$SERVER_CONTAINER" /opt/venv/bin/python -m sglang.test.run_eval --port 30001 --eval-name gpqa
```

先把 A 组跑两遍，两遍之间的差可以当作初筛用的粗略噪声下限；两组之间的差落在这个范围内，就不能算是开关带来的影响。要发布结论，每组应重复多次并给出置信区间。打开 `fake-prefill` 或 `simulated-acceptance` 时生成的任何内容都不能拿来算分。

## 测试与离线校验

在任何装有 Python 3.10 及以上版本的机器上都能核对已发布的数字，不需要 GPU；在 [README 第 1 步](../README_CN.md#操作步骤)进入的项目目录里执行：

```bash
python -m unittest discover -s tests -v
python tools/build_evidence.py --check
python tools/build_readme.py --check
```

全部测试通过、两个检查都输出 `PASS` 即为完成。每项检查覆盖的范围：

- `python -m unittest discover -s tests -v`——上游 patch 与 SHA-256 锁一致；每份日志投影都能解析且与清单一致；发布的变化率能从绝对值重算；启动命令渲染和消融在两个平台上都符合文档；容器启动脚本为每个角色生成预期的 `docker run`，就绪检查只接受 JSON（这两项在 Linux 和 macOS 上运行）；README 不含私有内容或超出范围的比较。
- `python tools/build_evidence.py --check`——`evidence/measurements.json` 与从 `evidence/raw/`、`evidence/runs.json` 重新构建的结果完全一致。
- `python tools/build_readme.py --check`——两份 README 里每张生成的表、列表和每段代码摘录，都与从证据和固定 patch 重新渲染的结果一致。
- `python tools/draw_diagrams.py --check`——已提交的 PNG 的 SHA-256 与 `images/SOURCES.json` 记录一致。
- `python tools/check_repo.py`——链接和图片都能解析，标题顺序符合读者动线，每张表最多四列，中英文生成块里的数字一致，没有私有路径、主机名或超出范围的比较。

CI 在 Ubuntu 和 Windows、Python 3.10 与 3.12 上运行同一组命令（[workflow](../../../.github/workflows/llm-inference-optimization-ci.yml)）。这些检查都不会启动 GPU 或服务：它们证明的是「已发布的数字确实来自已提交的证据」，而不是「重新跑一遍能得到同样的结果」。[README 的操作步骤](../README_CN.md#操作步骤)才是真正把这套栈跑起来的途径。
