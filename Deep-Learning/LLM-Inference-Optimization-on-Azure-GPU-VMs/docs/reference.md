# Reference

[中文](reference_CN.md) · [README](../README.md) · [Optimizations](optimizations.md) · [Deployment](deployment.md) · [Measure and verify](benchmark.md) · **Reference** · [Results](results.md)

Look-up material: metric and log field definitions, what this repository delivers, the directory layout, the pinned upstream commits and sources.

## Metrics and log fields

The client records workload arguments and the Serving Benchmark Result block. `tools/bench_log.py` projects and parses those fields; `tools/build_evidence.py` verifies their hashes and accepted comparisons. A server HTTP response alone is not an accuracy result.

RCCL output uses binary message sizes: 1G is 1 GiB. algbw is message bytes divided by time. For N ranks, AllReduce busbw is algbw multiplied by 2(N-1)/N. The reported average includes in-place and out-of-place rows; it is not a repeat-run average or NIC utilization. `-c 0` does not validate data. An INFO log must show IB selection and GDRDMA before claiming that path. The parser handles INFO text glued to a result's wrong-count without swapping the two modes.

## What This Repository Delivers

- **Serving engine and kernels** — owned by the upstream SGLang, AMD AITER, Composable Kernel and FlyDSL projects; the MiMo-specific commits are AMD engineering work in public forks. Here: pinned commit identities and the full patch of every commit discussed ([`upstream/`](../upstream/)).
- **Optimization method and measurements** — this repository. Measured MI300X before/after comparisons with projected raw benchmark output ([`evidence/`](../evidence/)), a per-technique explanation with code excerpts, and the boundaries of every number.
- **Launch configuration** — this repository. Machine-readable profiles for the measured MI300X stack and an NVIDIA template ([`profiles/`](../profiles/)), rendered into commands with single-technique ablation ([`tools/render_launch.py`](../tools/render_launch.py)).
- **Deployment** — this repository. A Dockerfile that builds the pinned runtime from public sources, a script that runs each server role as its own container, a readiness check and a settings template ([`docker/`](../docker/)).
- **Checks** — this repository. Offline tests and CI that recompute every published number from the committed evidence.

You supply: Azure ND MI300X v5 capacity (two VMs for the PD path, one for the single-VM path), the MiMo-V2.5-Pro checkpoint, and a container host with RDMA access.

Not provided: model weights, the private raw logs behind the projected evidence (their SHA-256 is recorded), comparisons with other accelerators, accuracy results with the FP8 KV cache on, and any measured NVIDIA run. What each optimization can do to accuracy, and how to check it, is covered in [Which optimizations can change model output](optimizations.md#which-optimizations-can-change-model-output).

## Repository layout

- [`evidence/runs.json`](../evidence/runs.json) — run identities, topology, controlled variables and script hashes.
- [`evidence/raw/`](../evidence/raw/) — projected sources: `sglang.bench_serving` output (workload arguments and result block of every run), the A/B samples from the scheduler-log audit, the long-context and replica result tables, the per-response accuracy scores, and the baseline client results and summary.
- [`evidence/raw-manifest.json`](../evidence/raw-manifest.json) — SHA-256 of each private raw log and of its public projection.
- [`evidence/measurements.json`](../evidence/measurements.json) — all comparisons, built by `tools/build_evidence.py`.
- [`evidence/docker-build.json`](../evidence/docker-build.json) — receipt of the clean Docker build: commit, Dockerfile hash, builder, image id and the step lines of the log.
- [`evidence/deploy-check.json`](../evidence/deploy-check.json) — receipt of the container checks on a CPU-only VM: which deployment steps ran, what was observed, which were not run, and the hashes of the scripts tested.
- [`upstream/`](../upstream/) — full patches of every commit discussed and `SOURCES.lock.json` (hash, license, layer, whether it is in the pinned runtime).
- [`profiles/`](../profiles/) — technique catalog, measured MI300X profiles and the NVIDIA template.
- [`tools/`](../tools/) — log projection and parsing, evidence and README builders, launch renderer, diagram generator, public-content audit.
- [`tests/`](../tests/) — offline tests.
- [`docker/`](../docker/) — the Dockerfile, the per-role container start script, the readiness check and the settings template.
- [`images/`](../images/) — diagrams and their hash ledger.

## Upstream commits

<!-- BEGIN GENERATED: upstream-table -->
- [`ROCm/FlyDSL@e46db60`](https://github.com/ROCm/FlyDSL/commit/e46db6020b4560de82a7136d78cc33a5186338f4) — [Kernel][PA] Support BF16 vectorized KV with asymmetric K/V head dim (#1065)  
  Operator (kernel) layer · upstream follow-up, not in the pinned runtime
- [`ROCm/FlyDSL@ed9885e`](https://github.com/ROCm/FlyDSL/commit/ed9885eca4ffc45e2ec1dc45fa00824baa6b56d3) — [Bugfix][PA] Fix D192 query staging and 64-bit cache offsets (#1064)  
  Operator (kernel) layer · upstream follow-up, not in the pinned runtime
- [`sammysun0711/FlyDSL@c99d5cd`](https://github.com/sammysun0711/FlyDSL/commit/c99d5cd97864c11e459cff9169d387d312790782) — Fix head-192 PA decode accuracy and long-context for MiMo-V2.5-Pro  
  Operator (kernel) layer · pinned runtime, not throughput-tested here
- [`sammysun0711/aiter@10a9401`](https://github.com/sammysun0711/aiter/commit/10a94012efc1260dfdf16ba2f52fbda40a518a17) — fix(pa_decode_gluon): correct head-192 persistent MTP decode  
  Operator (kernel) layer · pinned runtime, not throughput-tested here
- [`sammysun0711/aiter@3f4ab48`](https://github.com/sammysun0711/aiter/commit/3f4ab482a2986919c784e469e23cfac7f93bb153) — feat (MHA): Optimize MiMo FP8 page-64 batch prefill with a head-192 CK specialization  
  Operator (kernel) layer · pinned runtime, not throughput-tested here
- [`sammysun0711/aiter@d725746`](https://github.com/sammysun0711/aiter/commit/d725746a0f8c233d8e46e2771a7c8dbcd06e40d9) — add tuend moe (#5)  
  Operator (kernel) layer · in the measured runs
- [`sammysun0711/aiter@fc96a4f`](https://github.com/sammysun0711/aiter/commit/fc96a4f9f5f3e931cbb9de275c8aa01136417500) — Add MiMO-v2.5-Pro ck a8w8 blockscale gemm tuned config on MI300X (gfx942 304 CU)  
  Operator (kernel) layer · in the measured runs
- [`sammysun0711/sglang@0cfc48b`](https://github.com/sammysun0711/sglang/commit/0cfc48b0e374d7e84c122f739182a39feea56d46) — feat(MHA): Route MiMo cached prefill directly to AITER page-64 attention  
  Serving-framework layer · pinned runtime, not throughput-tested here
- [`sammysun0711/sglang@1f9bb2b`](https://github.com/sammysun0711/sglang/commit/1f9bb2b4c55cdc7bd5de1ac7977f76afab101a97) — (feat): Add opt-in mixed-precision Triton router GEMM for MiMo prefill  
  Operator (kernel) layer · later branch, not in any runtime here
- [`sammysun0711/sglang@2f9b9ae`](https://github.com/sammysun0711/sglang/commit/2f9b9aedf32977bc5d088a86ec0a73bcf432a4d0) — Add env variable SGLANG_USE_AITER_CK_BLOCKSCALE and SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE to use ck a8w8 gemm on gfx942 instead of triton a8w8 gemm  
  Serving-framework layer · in the measured runs
- [`sammysun0711/sglang@78cd40c`](https://github.com/sammysun0711/sglang/commit/78cd40c7a5102524536daf9a3178426777174d2d) — fix(speculative): isolate AITER MTP draft KV layout and graph metadata  
  Serving-framework layer · pinned runtime, not throughput-tested here
- [`sammysun0711/sglang@878fff1`](https://github.com/sammysun0711/sglang/commit/878fff15647fe3dabb32aa3a335b0ad16e3ee878) — bugfix(MTP): Fix HIP non-greedy EAGLE verification for MTP topk=1 (tree_topk) speculative decoding  
  Serving-framework layer · pinned runtime, not throughput-tested here
- [`sammysun0711/sglang@a2fd773`](https://github.com/sammysun0711/sglang/commit/a2fd773ab43f960f5f2c29b5c592b0ca43c5ba8f) — feat(flydsl pa decode): add configurable FlyDSL partition counts  
  Serving-framework layer · pinned runtime, not throughput-tested here
- [`sammysun0711/sglang@ba15db1`](https://github.com/sammysun0711/sglang/commit/ba15db1a576dcdc8d51ba15bd069b9fd1f748d97) — feat: Add opt-in FlyDSL paged decode for MiMo EAGLE verification  
  Serving-framework layer · pinned runtime, not throughput-tested here
- [`sammysun0711/sglang@db840d9`](https://github.com/sammysun0711/sglang/commit/db840d935a9f7097dbeb5f1b0dba4d261057a2bd) — [AMD] Fix aiter SWA handling in draft_extend_v2 for MTP speculative decoding  
  Serving-framework layer · in the measured runs
- [`sammysun0711/sglang@e11c515`](https://github.com/sammysun0711/sglang/commit/e11c5155f0845079211c2a4d0b8a4ab3669039f9) — feat(aiter attention backend): use Gluon PA for vectorized-5D target verification  
  Serving-framework layer · pinned runtime, not throughput-tested here
- [`sammysun0711/sglang@f26ae30`](https://github.com/sammysun0711/sglang/commit/f26ae30063143411f3ae552af1830fa46e3ee0fd) — fix(EAGLE verification): sync result across TP ranks  
  Serving-framework layer · pinned runtime, not throughput-tested here
<!-- END GENERATED: upstream-table -->

## Sources

[SGLang](https://github.com/sgl-project/sglang) · [AITER](https://github.com/ROCm/aiter) · [FlyDSL](https://github.com/ROCm/FlyDSL) · [Composable Kernel](https://github.com/ROCm/composable_kernel) · [Mooncake](https://github.com/kvcache-ai/Mooncake) · [Azure ND MI300X v5 series](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ndmi300xv5-series). Patches keep their upstream licenses (Apache-2.0 for SGLang and FlyDSL, MIT for AITER); see [`upstream/SOURCES.lock.json`](../upstream/SOURCES.lock.json).
