#!/usr/bin/env bash
# Optional communication check. It neither starts nor deallocates a VM.
set -euo pipefail

usage="usage: run-rccl.sh <prepare-ssh|start|host|container|check|stop>"
action="${1:?$usage}"
config="${RCCL_ENV_FILE:-$(dirname "$0")/rccl.env}"
output_override="${RCCL_OUTPUT_DIR:-}"
[ -f "$config" ] || { echo "missing $config; copy rccl.env.example and fill it in" >&2; exit 2; }
set -a
. "$config"
set +a
if [ -n "$output_override" ]; then export RCCL_OUTPUT_DIR="$output_override"; fi
case "$action" in prepare-ssh|start|host|container|check|stop) ;; *) echo "$usage" >&2; exit 2 ;; esac
for key in NODE_A NODE_B; do
  value="${!key:-}"
  [[ "$value" =~ ^[[:alnum:]][[:alnum:].-]*$ ]] || {
    echo "$key must be an IPv4 address or DNS hostname, without SSH user syntax" >&2; exit 2;
  }
done
[ "$NODE_A" != "$NODE_B" ] || { echo "NODE_A and NODE_B must be different" >&2; exit 2; }
port="${RCCL_SSH_PORT:-2222}"
[[ "$port" =~ ^[0-9]+$ ]] && ((port > 0 && port < 65536)) || {
  echo "RCCL_SSH_PORT must be between 1 and 65535" >&2; exit 2;
}
name="${RCCL_CONTAINER:-rccl-benchmark}"
[[ "$name" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || { echo "invalid RCCL_CONTAINER" >&2; exit 2; }
hostssh=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes)

if [ "$action" = prepare-ssh ]; then
  keydir="${RCCL_SSH_DIR:?set RCCL_SSH_DIR in the settings file}"
  [[ "$keydir" =~ ^/[[:alnum:]_./-]+$ ]] && [[ "$keydir" != *"/../"* ]] || {
    echo "RCCL_SSH_DIR must be an absolute Linux path without spaces or parent traversal" >&2; exit 2;
  }
  [[ "$keydir" == */rccl-ssh ]] || { echo "use a dedicated directory ending in /rccl-ssh" >&2; exit 2; }
  if [ -d "$keydir" ] && [ ! -f "$keydir/.rccl-owned" ] && [ -n "$(ls -A "$keydir")" ]; then
    echo "refusing to overwrite unrelated SSH material" >&2; exit 2
  fi
  umask 077
  mkdir -p "$keydir"
  touch "$keydir/.rccl-owned"
  [ -e "$keydir/id_ed25519" ] || ssh-keygen -q -t ed25519 -N '' -f "$keydir/id_ed25519"
  [ -e "$keydir/ssh_host_ed25519_key" ] || ssh-keygen -q -t ed25519 -N '' -f "$keydir/ssh_host_ed25519_key"
  cp "$keydir/id_ed25519.pub" "$keydir/authorized_keys"
  "${hostssh[@]}" "$NODE_B" bash -s -- "$keydir" <<'REMOTE'
set -euo pipefail
umask 077
if [ -d "$1" ] && [ ! -f "$1/.rccl-owned" ] && [ -n "$(ls -A "$1")" ]; then
  echo "refusing to overwrite unrelated remote SSH material" >&2; exit 2
fi
mkdir -p "$1"
chmod 700 "$1"
touch "$1/.rccl-owned"
[ -e "$1/ssh_host_ed25519_key" ] || ssh-keygen -q -t ed25519 -N '' -f "$1/ssh_host_ed25519_key"
REMOTE
  remote_key="$("${hostssh[@]}" "$NODE_B" cat "$keydir/ssh_host_ed25519_key.pub")"
  local_key="$(cat "$keydir/ssh_host_ed25519_key.pub")"
  printf '[%s]:%s %s\n' "$NODE_A" "$port" "$(printf '%s\n' "$local_key" | awk '{print $1" "$2}')" > "$keydir/known_hosts"
  printf '[%s]:%s %s\n' "$NODE_B" "$port" "$(printf '%s\n' "$remote_key" | awk '{print $1" "$2}')" >> "$keydir/known_hosts"
  chmod 700 "$keydir"
  chmod 600 "$keydir/id_ed25519" "$keydir/ssh_host_ed25519_key" "$keydir/authorized_keys" "$keydir/known_hosts"
  scp -p -o BatchMode=yes -o StrictHostKeyChecking=yes \
    "$keydir/id_ed25519" "$keydir/authorized_keys" "$keydir/known_hosts" "$NODE_B:$keydir/"
  echo "SSH material prepared through authenticated host SSH; run start on both hosts"
  exit 0
fi

if [ "$action" = start ]; then
  image="${RCCL_IMAGE:?set the built, pushed RCCL_IMAGE digest}"
  [[ "$image" =~ @sha256:[[:xdigit:]]{64}$ ]] || { echo "RCCL_IMAGE must be digest-pinned" >&2; exit 2; }
  keydir="${RCCL_SSH_DIR:?set RCCL_SSH_DIR}"
  for file in id_ed25519 authorized_keys known_hosts ssh_host_ed25519_key; do
    [ -s "$keydir/$file" ] || { echo "missing $file; run prepare-ssh from node A" >&2; exit 2; }
  done
  exec docker run --detach --name "$name" --hostname "$(hostname)" \
    --network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --ulimit memlock=-1:-1 \
    -e RCCL_SSH_PORT="$port" -v "$keydir":/run/rccl-ssh:ro \
    "$image"
fi

if [ "$action" = stop ]; then
  exec docker stop "$name"
fi

out="${RCCL_OUTPUT_DIR:?set a new RCCL_OUTPUT_DIR for this run}"
[ ! -e "$out" ] || { echo "output path already exists; choose a new RCCL_OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$out"
args=(-np 16 -H "$NODE_A:8,$NODE_B:8" -map-by ppr:8:node --bind-to numa
      -x NCCL_IB_HCA=^mlx5_8:1,mlx5_an0:1 -x NCCL_IB_PCI_RELAXED_ORDERING=1)
perf=(-g 1 -f 2 -b 1G -e 16G -c 0 -n 50 -w 20)
debug=(-x NCCL_DEBUG=VERSION)
if [ "$action" = check ]; then
  debug=(-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET)
  perf=(-g 1 -f 2 -b 1G -e 16G -c 1 -n 20 -w 5)
fi
args+=("${debug[@]}" --mca pml ucx --mca btl '^vader,tcp,openib,uct')
if [ "$action" = host ]; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
  module load "${RCCL_HOST_MODULE:-mpi/hpcx}"
  command=(mpirun "${args[@]}" -x PATH -x LD_LIBRARY_PATH "${RCCL_HOST_BIN:?set RCCL_HOST_BIN}" "${perf[@]}")
else
  docker exec "$name" test -x /opt/rccl-tests/all_reduce_perf
  docker exec "$name" ssh -p "$port" "$NODE_B" test -x /opt/rccl-tests/all_reduce_perf
  command=(docker exec "$name" mpirun --allow-run-as-root "${args[@]}"
           --mca plm_rsh_args "-p $port" -x PATH -x LD_LIBRARY_PATH
           /opt/rccl-tests/all_reduce_perf "${perf[@]}")
fi
printf '%q ' "${command[@]}" > "$out/command.sh"
printf '\n' >> "$out/command.sh"
if "${command[@]}" > "$out/output.log" 2>&1; then
  rc=0
else
  rc=$?
fi
printf '%s\n' "$rc" > "$out/exit-code.txt"
cat "$out/output.log"
[ "$rc" -eq 0 ] || { echo "benchmark failed; see $out/output.log" >&2; exit "$rc"; }
parse_args=(parse "$out/output.log")
if [ "$action" = check ]; then parse_args+=(--require-ib); fi
python3 "$(dirname "$0")/../tools/rccl_log.py" "${parse_args[@]}" > "$out/result.json"
echo "Completed and validated; see $out/result.json"
