#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "ERROR: native benchmark failed at line %s\n" "$LINENO" >&2' ERR
case "${1:-}" in
    before) rocm=6.4.4 ;;
    after) rocm=7.2.4 ;;
    *) echo 'Usage: bash scripts/run_native.sh before|after WORKDIR' >&2; exit 2 ;;
esac
phase="$1"
work="${2:?Provide an absolute output directory outside committed evidence}"
[[ "$work" == /* ]]
[[ -n "${MPI_HOME:-}" ]] || { echo 'ERROR: export the same MPI_HOME for both phases' >&2; exit 2; }
test -x "$MPI_HOME/bin/mpirun"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit=1284e5da11f7904e8a6efd902b9de52253785bd0
hip="/opt/rocm-$rocm"
test -x "$hip/bin/hipcc"
result="$work/$phase"
source="$work/source-$phase"
if [[ -e "$result" || -e "$source" ]]; then
    echo 'ERROR: phase output/source already exists; preserve it and choose a fresh WORKDIR' >&2
    exit 2
fi
mkdir -p "$work" "$result" "$source"
exec 9>"$work/native-benchmark.lock"
flock -n 9
if [[ ! -d "$work/upstream/.git" ]]; then
    git init -q "$work/upstream"
    git -C "$work/upstream" remote add origin https://github.com/ROCm/rocm-systems.git
    git -C "$work/upstream" fetch --depth 1 --filter=blob:none origin "$commit"
    git -C "$work/upstream" sparse-checkout init --cone
    git -C "$work/upstream" sparse-checkout set projects/rccl-tests
    git -C "$work/upstream" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$work/upstream" rev-parse HEAD)" == "$commit" ]]
git -C "$work/upstream" archive --format=tar HEAD:projects/rccl-tests | tar -xf - -C "$source"
export PATH="$hip/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$hip/lib:$MPI_HOME/lib"
export NCCL_DEBUG=WARN
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
make -C "$source" -j8 MPI=1 MPI_HOME="$MPI_HOME" GPU_TARGETS=gfx942 \
    HIP_HOME="$hip" NCCL_HOME="$hip" GIT_REV="$commit" GIT_BRANCH=pinned GIT_DIFF= \
    > "$result/build.txt" 2>&1
binary="$source/build/all_reduce_perf"
python3 "$repo/tools/collect_host.py" --binary "$binary" --phase "$phase" > "$result/host-metadata.json"
if [[ "$phase" == after ]]; then
    python3 - "$work" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
old = json.loads((root / "before/host-metadata.json").read_text())
new = json.loads((root / "after/host-metadata.json").read_text())
if old["boot_id"] == new["boot_id"] or old["loaded_amdgpu_srcversion"] == new["loaded_amdgpu_srcversion"]:
    raise RuntimeError("A real host-driver activation reboot has not been demonstrated")
if old["kernel"] != new["kernel"] or old["mpi_version"] != new["mpi_version"]:
    raise RuntimeError("Kernel or MPI changed; the experiment is no longer matched")
PY
fi
"$hip/bin/rocm-smi" --showcomputepartition --showmemorypartition --showtopo > "$result/topology.txt"
cat "$result/host-metadata.json"
for layout in single mpi; do
    for run in 1 2 3; do
        name="$layout-run-$run"
        if [[ "$layout" == single ]]; then
            command=("$binary" -g 8)
        else
            command=("$MPI_HOME/bin/mpirun" --allow-run-as-root --bind-to none -np 8 "$binary" -g 1)
        fi
        printf 'NATIVE_BENCHMARK_BEGIN=%s/%s\n' "$phase" "$name"
        if timeout 180 "${command[@]}" -b 1M -e 1G -f 2 -t 1 -n 100 -w 20 -c 1 \
            -d float -o sum -J "$result/$name.json" > "$result/$name.txt" 2>&1; then
            grep -qE '^# Out of bounds values[[:space:]]*:[[:space:]]+0[[:space:]]+OK' "$result/$name.txt"
            test -s "$result/$name.json"
            grep -E '^[[:space:]]+1073741824[[:space:]]' "$result/$name.txt"
        else
            status=$?
            tail -n 60 "$result/$name.txt" >&2
            exit "$status"
        fi
    done
done
echo NATIVE_PHASE_BENCHMARK_PASS
