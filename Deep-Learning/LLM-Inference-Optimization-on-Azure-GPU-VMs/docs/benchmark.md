# Measure and Verify

[中文](benchmark_CN.md) · [README](../README.md) · [Optimizations](optimizations.md) · [Deployment](deployment.md) · **Measure and verify** · [Reference](reference.md) · [Results](results.md)

Use this page to measure and check: reproduce the published measurements, measure one change, follow the measurement practice behind the numbers, check a lossy switch on your own model, and verify the published numbers offline without a GPU.

## Reproduce the published numbers

The published throughput was measured with MTP at a fixed acceptance of 3. To compare with it, render prefill and decode without `--ablate simulated-acceptance`, restart those two containers, and run the benchmark client as a throw-away container on VM A:

```bash
python3 tools/render_launch.py --profile rocm-mi300x-pd --role bench-decode > run/bench-decode.sh
python3 tools/render_launch.py --profile rocm-mi300x-pd --role bench-prefill > run/bench-prefill.sh
CONCURRENCY=64 bash docker/run-role.sh bench run/bench-decode.sh | tee decode_c64.log
INPUT_LEN=8192 bash docker/run-role.sh bench run/bench-prefill.sh | tee prefill_8k.log
python3 tools/bench_log.py project decode_c64.log -o my_decode_c64.txt
python3 tools/bench_log.py parse my_decode_c64.txt
```

`Successful requests` equalling the prompt count (256 for decode, 16 for prefill) is necessary, not sufficient. Also exclude error responses, verify actual input and output lengths, check context headroom and inspect server logs before accepting a point. Fixed-acceptance runs do not score text accuracy, but protocol and execution errors must still be rejected. Without `--dataset-path`, `bench_serving` downloads the ShareGPT file it samples text from, so an offline host needs a local copy. Use this configuration for measurement only.

## Measure one change

Render the same role with one technique removed, replace only that container, and run the identical client command twice per arm. This example removes the two switches of the published A/B on the 1P1D decode server; it illustrates the method and does not repeat the single-VM 64K experiment recorded in [`evidence/runs.json`](../evidence/runs.json):

```bash
python3 tools/render_launch.py --profile rocm-mi300x-pd --role decode --ablate ck-a8w8-gemm --ablate unified-verify > run/decode-baseline.sh
docker rm -f mimo-decode && bash docker/run-role.sh decode run/decode-baseline.sh
```

## Measurement practice

### Exact inputs and context headroom

Random-prompt benchmarks re-tokenize text, so the server-side length can drift. For long-context points, pass token IDs (`--tokenize-prompt`) and account for the model's special tokens: MiMo adds four, so 65,532 supplied IDs give exactly 65,536 server-side tokens. Leave the same headroom in `--context-length`, otherwise the longest point can "succeed" with error payloads, which is why the 256K point is excluded from the [stage pair](results.md#stage-pair-shape-tuned-fused-moe-table).

### Fresh-service repeats

Every accepted A/B arm is run twice on a freshly started server. Agreement within about 1% (see the [A/B table](results.md#controlled-ab-block-scale-fp8-gemm-path-at-64k-context)) is what allows a 25% difference to be called real; a single run per arm would not. For a new context length, test one request first, then several in sequence, then several at once, then a fresh-service repeat; a point that fails on the way is published as rejected, not estimated.

## Check a lossy switch on your model

**Kernel-level numerical checks.** The upstream commits add tests that compare each MiMo-specific kernel with a PyTorch reference at MiMo's shape (head 192, page 64, FP8 KV, query length 4), including a 2 GiB offset case. They need an MI300X to run and were not run for this repository:

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

A kernel test proves the kernel matches its reference on the tested shapes. It does not measure how FP8 storage or INT8 reduction moves end-to-end answers; that needs a model-level check.

**How to check a lossy switch on your model (not run here).** Use the accuracy role of the pinned runtime, which runs with real MTP acceptance and Quick Reduce off. This FP8-KV comparison is not strictly single-variable: arm B switches to BF16 KV and also moves target verification from FlyDSL back to AITER.

```bash
# Arm A: FP8 KV cache (as served)
python tools/render_launch.py --profile rocm-mi300x-single --role server > arm_a.sh
# Arm B: BF16 KV cache. FlyDSL decode requires FP8 KV, so the renderer makes you remove both.
python tools/render_launch.py --profile rocm-mi300x-single --role server --ablate fp8-kv-5d --ablate flydsl-pa-decode > arm_b.sh
# Separate Quick Reduce comparison: keep arm A unchanged and change the generated export in its copy.
cp arm_a.sh arm_qr_int8.sh
sed -i 's/^export ROCM_QUICK_REDUCE_QUANTIZATION=NONE$/export ROCM_QUICK_REDUCE_QUANTIZATION=INT8/' arm_qr_int8.sh
grep -Fx 'export ROCM_QUICK_REDUCE_QUANTIZATION=NONE' arm_a.sh
grep -Fx 'export ROCM_QUICK_REDUCE_QUANTIZATION=INT8' arm_qr_int8.sh
```

Run each arm separately with the serving procedure, and check the actual server process environment before scoring. An export in the outer shell is insufficient: the generated script would overwrite it.

```bash
SERVER_CONTAINER=mimo-server
docker exec "$SERVER_CONTAINER" bash -c \
  'tr "\000" "\n" < /proc/1/environ | grep "^ROCM_QUICK_REDUCE_QUANTIZATION="'
```

Expect `NONE` for arm A and `INT8` for the Quick Reduce arm. Do not compare scores if both runs report the same value.

Against each arm, run the evaluation tools inside its `SERVER_CONTAINER`, using the pinned image's Python environment, at temperature 0 on public datasets:

```bash
docker exec "$SERVER_CONTAINER" /opt/venv/bin/python -m sglang.test.run_eval --port 30001 --eval-name gsm8k --num-examples 1319
docker exec "$SERVER_CONTAINER" /opt/venv/bin/python -m sglang.test.run_eval --port 30001 --eval-name mmlu --num-examples 2000
docker exec "$SERVER_CONTAINER" /opt/venv/bin/python -m sglang.test.run_eval --port 30001 --eval-name gpqa
```

Run arm A twice first. The difference between those two runs is a rough noise floor for screening; a gap between arms that stays inside it is not an effect of the switch. Before publishing a conclusion, repeat each arm several times and report a confidence interval. Do not score anything produced with `fake-prefill` or `simulated-acceptance` switched on.

## Tests and Offline Checks

Check the published numbers on any machine with Python 3.10 or newer, no GPU needed, from the project directory you entered in [step 1 of the README](../README.md#steps):

```bash
python -m unittest discover -s tests -v
python tools/build_evidence.py --check
python tools/build_readme.py --check
```

Done when all tests pass and both checks print `PASS`. What each check covers:

- `python -m unittest discover -s tests -v` — upstream patches match their SHA-256 lock; every projected log parses and matches the manifest; published deltas recompute from absolute values; launch rendering and ablation behave as documented on both platforms; the container start script builds the expected `docker run` for each role and the readiness check accepts only JSON (these two run on Linux and macOS); READMEs contain no private or out-of-scope content.
- `python tools/build_evidence.py --check` — `evidence/measurements.json` equals a fresh build from `evidence/raw/` and `evidence/runs.json`.
- `python tools/build_readme.py --check` — every generated table, list and code excerpt in both READMEs equals a fresh render from the evidence and the pinned patches.
- `python tools/draw_diagrams.py --check` — the committed PNGs have the SHA-256 recorded in `images/SOURCES.json`.
- `python tools/check_repo.py` — links and images resolve, headings follow the reader order, every table has at most four columns, English and Chinese generated blocks carry the same numbers, and no private paths, hosts or out-of-scope comparisons appear.

The same commands run in CI on Ubuntu and Windows with Python 3.10 and 3.12 ([workflow](../../../.github/workflows/llm-inference-optimization-ci.yml)). None of these checks starts a GPU or a server: they prove that the published numbers follow from the committed evidence, not that a new run would reproduce them. The [README steps](../README.md#steps) are the only route that runs the stack.
