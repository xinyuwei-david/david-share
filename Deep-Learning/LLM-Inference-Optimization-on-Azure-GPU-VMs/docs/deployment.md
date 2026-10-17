# Deployment Details and Operations

[中文](deployment_CN.md) · [README](../README.md) · [Optimizations](optimizations.md) · **Deployment** · [Measure and verify](benchmark.md) · [Reference](reference.md) · [Results](results.md)

This page adds detail to the [README steps](../README.md#steps): what each step does and why, how to confirm the optimizations are active, day-to-day operation, what has actually been run, and failures that are easy to misread. Follow the command sequence in the README.

## Deployment step details

Deploy the optimized stack from the repository's Dockerfile: build the image once, push it to your registry, and run every server as its own container from that image. Two routes use the same image. **One VM** runs a single TP8 server with the final runtime's settings. **Two VMs in 1P1D** run the prefill/decode topology behind the measured throughput. Checking the published numbers without a GPU is a separate path, described in [Tests and Offline Checks](benchmark.md#tests-and-offline-checks).

**1. Check each VM.** Azure ND MI300X v5 with the ROCm driver and Docker (with BuildKit) installed.

```bash
rocm-smi --showproductname   # eight GPU[0] ... GPU[7] entries naming MI300X
ibv_devinfo -l               # 1P1D: eight RDMA devices, mlx5_ib0 ... mlx5_ib7
show_gids                    # 1P1D: the GID index for MC_GID_INDEX (3 on the measured VMs)
df -h /mnt/models            # room for the weights (about 1 TB) on the local NVMe volume
docker buildx version
```

Serve the weights from the VM's local NVMe volume: every container start reads them in full. Local NVMe on these VMs is temporary storage and is lost when the VM is deallocated, so keep the durable copy elsewhere (for example in Azure Blob Storage) and stage it to NVMe after each allocation.

**2. Get the deployment files and the model on every VM.** The download is pinned to revision `7208274`, the last commit that changed the weights or the modelling code (later commits change only the model card), so every VM and every later redeploy loads the same files. The launch uses `--trust-remote-code` because the model ships its own modelling code; review the code at the revision you pin.

```bash
git clone --filter=blob:none --sparse https://github.com/david-xinyuwei/david-share.git
cd david-share && git sparse-checkout set Deep-Learning/LLM-Inference-Optimization-on-Azure-GPU-VMs
cd Deep-Learning/LLM-Inference-Optimization-on-Azure-GPU-VMs
export MODELS=/mnt/models
export MODEL_REPO=        # the model's Hugging Face repository id, from its model card
hf download "$MODEL_REPO" --revision 7208274e94787a92d7261675fb1de56ccb0e94b5 --local-dir "$MODELS/MiMo-V2.5-Pro"
cp docker/mimo.env.example docker/mimo.env
```

Fill in [`docker/mimo.env`](../docker/mimo.env.example) and copy the same file to every VM. It holds every setting the servers read: the image digest (from step 3), the model path inside the container and, for 1P1D, the two VM addresses, the RDMA devices and the GID index. The host commands below load it with `set -a; . docker/mimo.env; set +a`; the containers receive it with `--env-file`.

**3. Build the image once and push it.** On any x86-64 machine with BuildKit, one of the VMs included:

```bash
export ACR_NAME=          # your Azure Container Registry
REGISTRY="$ACR_NAME.azurecr.io"
az acr login --name "$ACR_NAME"
docker buildx build --platform linux/amd64 --tag "$REGISTRY/mimo-mi300x:$(git rev-parse --short=12 HEAD)" \
  --metadata-file build-meta.json --push docker/
DIGEST=$(python3 -c "import json; print(json.load(open('build-meta.json'))['containerimage.digest'])")
echo "IMAGE=$REGISTRY/mimo-mi300x@$DIGEST"   # put this line into docker/mimo.env on every VM
```

Then, on every VM: `set -a; . docker/mimo.env; set +a; docker pull "$IMAGE"`.

The [Dockerfile](../docker/Dockerfile) starts from a digest-pinned public base image, checks out SGLang `878fff1`, AITER `3f4ab48` with Composable Kernel `af7118e` and its bundled patch, installs FlyDSL `0.2.4` from PyPI and the FlyDSL kernels at `c99d5cd`, and stops if the wheel hash, the CK commit or the final imports do not match. The tag records the source commit; running by digest means both VMs, and every later restart, use the same image. A clean build takes about six minutes once the base image is cached and produces a 27.9 GB image.

**4. Render the launch scripts.** The profiles turn the measured configuration into one script per role; hosts, paths and devices stay variables that the container reads from `docker/mimo.env`.

One VM:

```bash
mkdir -p run
python3 tools/render_launch.py --profile rocm-mi300x-single --role server > run/server.sh
```

1P1D (render on each VM, or once and copy `run/`):

```bash
mkdir -p run
python3 tools/render_launch.py --profile rocm-mi300x-pd --role prefill --ablate simulated-acceptance > run/prefill.sh
python3 tools/render_launch.py --profile rocm-mi300x-pd --role decode --ablate simulated-acceptance > run/decode.sh
python3 tools/render_launch.py --profile rocm-mi300x-pd --role router > run/router.sh
```

These are serving settings, not benchmark settings. `--ablate simulated-acceptance` removes the fixed MTP acceptance used for the measurements, so MTP keeps only the draft tokens the model actually accepts. The one-VM `server` role already runs with real acceptance and with Quick Reduce off. The 1P1D scripts keep INT8 Quick Reduce as in the measured runs; add `--ablate int8-quick-reduce` to sum in full precision until you have [checked accuracy](optimizations.md#which-optimizations-can-change-model-output) on your workload.

**5. Start the servers.** One VM:

```bash
set -a; . docker/mimo.env; set +a
bash docker/run-role.sh server run/server.sh
bash docker/wait-ready.sh 127.0.0.1 30001 3600 mimo-server
```

1P1D:

```bash
set -a; . docker/mimo.env; set +a
bash docker/run-role.sh prefill run/prefill.sh                         # on VM A
bash docker/run-role.sh decode run/decode.sh                           # on VM B
bash docker/wait-ready.sh "$PREFILL_HOST" 30000 3600 mimo-prefill      # on VM A
bash docker/wait-ready.sh "$DECODE_HOST" 30001 3600                    # on VM A
bash docker/run-role.sh router run/router.sh                           # on VM A, after both print READY
```

[`docker/run-role.sh`](../docker/run-role.sh) starts each role as a named container (`mimo-server`, `mimo-prefill`, `mimo-decode`, `mimo-router`) whose main process is the server itself: it restarts on failure up to three times, rotates its logs, mounts the model read-only and reads its settings from the env file. Only the GPU roles get the host access that RDMA and AITER need (`--privileged`, host IPC, `/dev/kfd`, `/dev/dri`, `/dev/mem`, `CAP_SYS_ADMIN`), so run them only on dedicated GPU VMs. [`docker/wait-ready.sh`](../docker/wait-ready.sh) polls `/server_info`, which does not generate tokens, prints the KV capacity the server reports, and stops early when the named container has exited. The first start of a container compiles AITER JIT kernels and loads the weights; allow tens of minutes.

All containers use host networking, and the servers listen on every interface. Allow ports 30000 and 30001 only between the two VMs' private addresses, keep port 40000 private as well, and put an authenticated gateway in front of it if clients outside the VMs need access.

**6. Confirm the optimizations are active, then send a request.** One VM:

```bash
docker logs mimo-server 2>&1 | grep -m1 module_gemm_a8w8_blockscale_bpreshuffle   # CK block-scale GEMM
docker logs mimo-server 2>&1 | grep -m1 mimo_v2_5_pro_b16_tuned_fmoe              # tuned fused-MoE table
curl -s --retry 30 --retry-connrefused --retry-delay 10 http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number."}], "max_tokens": 1024, "temperature": 0}'
```

1P1D, on VM A for the prefill server and the request, on VM B for the decode server:

```bash
docker logs mimo-decode 2>&1 | grep -m1 module_gemm_a8w8_blockscale_bpreshuffle   # CK block-scale GEMM
docker logs mimo-prefill 2>&1 | grep -m1 mimo_v2_5_pro_b16_tuned_fmoe             # tuned fused-MoE table
docker logs mimo-prefill 2>&1 | grep -i mooncake | grep -m3 mlx5_ib                # RDMA devices in use
docker logs mimo-prefill 2>&1 | grep -i mooncake | grep -i -m3 tcp                 # expect no output
curl -s --retry 30 --retry-connrefused --retry-delay 10 http://127.0.0.1:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number."}], "max_tokens": 1024, "temperature": 0}'
```

Done when every check prints what its comment says and the reply's `choices[0].message.content` contains `391` with `finish_reason` `stop`.

**7. Operate.**

- **Logs:** `docker logs -f mimo-decode`.
- **Restart:** `docker restart mimo-decode` keeps the AITER kernels compiled inside the container; `docker rm` discards them, and the next start compiles again.
- **Crashes, hangs and reboots:** the containers restart after a crash, up to three times. They do not come back after a VM or Docker restart, and a hung server is not restarted. For unattended operation, run the step 5 commands from systemd or an orchestrator that also checks readiness.
- **Upgrade or roll back:** build and push a new image (step 3) and update `IMAGE` in `docker/mimo.env`. On each VM remove the role containers with `docker rm -f`, then repeat step 5: servers first, router last. Roll back by restoring the previous digest. With one pair of servers this is a maintenance window: requests in flight fail and serving pauses until the new containers are ready. To avoid it, start a second pair from the new digest, check it, and then switch the router.
- **Stop:** `docker stop mimo-router mimo-prefill` on VM A and `docker stop mimo-decode` on VM B, or `docker stop mimo-server` on one VM. This retains the containers and their writable layers.
- **Optional removal or VM deallocation:** these are separate operator-authorized actions, not automatic cleanup. Confirm durable backups before removing containers or deallocating VMs. Deallocation can clear local NVMe; a shutdown inside the VM does not release its compute allocation.

## What has been run

On a CPU-only VM with Docker: the image build of step 3 from a clean context, with `--metadata-file` and `--load` instead of `--push` ([`evidence/docker-build.json`](../evidence/docker-build.json), [`evidence/deploy-check.json`](../evidence/deploy-check.json)); the router role started by `run-role.sh` from a rendered script, with the router as the container's main process and `docker stop` returning in 0.5 s; `wait-ready.sh` against a stand-in `/server_info`; and the benchmark role's variable pass-through and read-only model mount. Not run yet: the push and the pull by digest, the GPU roles (server, prefill, decode), and the checks and request of step 6. Those follow the recorded launch configuration. The runtime this Dockerfile builds is also newer than the stack behind the measured throughput, which each measurement section names.

## Failures that are easy to misread

These are operational observations from bringing this stack up on MI300X, not measurements; only the graph-capture item has a number published here. The first group only makes a run slower while the server keeps working.

- **KV transfer falls back to TCP.** Without `--privileged`, `/dev/mem` and `CAP_SYS_ADMIN` in the container, Mooncake quietly uses TCP instead of RDMA. Tokens stay correct and throughput drops. Make "RDMA initialized" a start-up check.
- **Graph capture off on the decode server.** With `--disable-cuda-graph` the decode server ran at a third of its throughput in the [baseline A/B](results.md#baseline-to-optimized-stack-the-cumulative-gain) and printed nothing unusual. Diff every launcher against a known-good one before a run.
- **Two copies of a kernel library in one image.** The same AITER version imported from a different directory behaved differently. Check the import path and the kernel names in the server log, not only the version.
- **A tuned table that is never hit.** AITER logs `default` for fused-MoE shapes without a tuned row. Check the start-up log for the shapes that matter.

The second group stops or stalls the server, but the cause is easy to misread:

- **Multithreaded weight loading hangs.** On the PD servers, tensor-parallel ranks hung in page-fault handling while loading weights with several threads; `--model-loader-extra-config '{"enable_multithread_load": false}'` fixed it.
- **Overlap scheduling on the prefill server.** It hit a HIP illegal-address error, so the throughput runs use `--disable-overlap-schedule`.
- **Health probes that generate tokens.** By default SGLang's `/health` generates a real token (`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`); frequent polling stalled the prefill server's detokenizer. Set it to `0` and poll `/server_info` instead.
- **A per-step all-gather times out near 200K tokens.** At data-parallel size 1, the existing SGLang switch `SGLANG_SCHEDULER_SKIP_ALL_GATHER=1` avoided it.
- **Values copied from another GPU's configuration.** A `--chunked-prefill-size` taken over from a different system failed at start-up. Derive such values from this runtime's limits.
