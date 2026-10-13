#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "ERROR: configuration-2 container test failed at line %s; preserve output logs\n" "$LINENO" >&2' ERR
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_container.sh WORKDIR'
    echo 'Requires the VM to retain loaded AMDGPU 6.12.12 and native ROCm6.4.4/RCCL2.22.3.'
    echo 'Builds and runs ROCm7.2.4/RCCL2.27.7 userspace in Docker without changing the VM driver.'
    exit 0
fi
work="${1:?Provide an absolute output directory for configuration 2}"
[[ "$work" == /* ]]
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f /.dockerenv ]] || { echo 'ERROR: run the wrapper on the VM, not inside another container' >&2; exit 2; }
[[ "$(cat /sys/module/amdgpu/version)" == 6.12.12 ]] || {
    echo 'ERROR: configuration 2 requires the actually loaded VM driver 6.12.12; no driver change will be attempted' >&2
    exit 2
}
[[ "$(cat /opt/rocm/.info/version)" == 6.4.4-129 ]]
[[ "$(dpkg-query -W -f='${Version}' rccl)" == 2.22.3.60404-129~24.04 ]]
test -c /dev/kfd
test -d /dev/dri
command -v docker >/dev/null
result="$work/configuration-2"
[[ ! -e "$result" ]] || { echo 'ERROR: output already exists; select a fresh WORKDIR' >&2; exit 2; }
mkdir -p "$work"
exec 9>"$work/native-benchmark.lock"
flock -n 9
mkdir -p "$result"
image=mi300x-rccl-case:rocm-7.2.4-1284e5d
docker build --progress plain -f "$repo/docker/Dockerfile.rccl-tests" -t "$image" "$repo/docker" \
    > "$result/build.txt" 2>&1
docker image inspect --format '{{.Id}}' "$image" > "$result/image-id.txt"
args=(
    --rm --network host --ipc host
    --device /dev/kfd --device /dev/dri
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1
    --env NCCL_DEBUG=WARN --env HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
)
# Mount only the output directory, never the VM's /opt/rocm over the container SDK.
docker run "${args[@]}" -i --entrypoint python3 "$image" - \
    > "$result/runtime-layers.json" <<'PY'
import ctypes
import json
import pathlib
import re
import subprocess

loaded = pathlib.Path("/sys/module/amdgpu/version").read_text().strip()
rocm = pathlib.Path("/opt/rocm/.info/version").read_text().strip()
libraries = subprocess.check_output(["ldd", "/opt/rccl-tests/build/all_reduce_perf"], text=True)
match = re.search(r"librccl\.so\.1 => (\S+)", libraries)
if match is None or "not found" in libraries:
    raise RuntimeError(f"Container library resolution failed:\n{libraries}")
library = pathlib.Path(match[1]).resolve()
if not str(library).startswith("/opt/rocm-7.2.4/lib/"):
    raise RuntimeError(f"Container is not loading its intended RCCL library: {library}")
version = ctypes.c_int()
status = ctypes.CDLL(str(library)).ncclGetVersion(ctypes.byref(version))
if status != 0 or version.value != 22707 or rocm != "7.2.4" or loaded != "6.12.12":
    raise RuntimeError(f"Runtime-layer mismatch: ROCm={rocm}, RCCL={version.value}, KMD={loaded}")
print(json.dumps({
    "execution": "container",
    "container_rocm": rocm,
    "actual_container_rccl_library": str(library),
    "actual_container_rccl_code": version.value,
    "shared_vm_loaded_amdgpu_version": loaded,
    "linked_libraries": libraries.splitlines(),
}, indent=2))
PY
cat "$result/runtime-layers.json"
for run in 1 2 3; do
    log="$result/run-$run.txt"
    if timeout 180 docker run "${args[@]}" --volume "$result:/results" "$image" \
        -b 1M -e 1G -f 2 -t 1 -g 8 -n 100 -w 20 -c 1 -d float -o sum \
        -J "/results/run-$run.json" > "$log" 2>&1; then
        grep -qE '^# Out of bounds values[[:space:]]*:[[:space:]]+0[[:space:]]+OK' "$log"
        grep -E '^[[:space:]]+1073741824[[:space:]]' "$log"
        test -s "$result/run-$run.json"
    else
        status=$?
        tail -n 60 "$log" >&2
        exit "$status"
    fi
done
[[ "$(cat /sys/module/amdgpu/version)" == 6.12.12 ]]
echo CONFIGURATION_2_CONTAINER_TEST_PASS_VM_DRIVER_UNCHANGED
