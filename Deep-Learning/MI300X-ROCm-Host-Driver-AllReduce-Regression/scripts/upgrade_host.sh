#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "ERROR: host upgrade failed at line %s; preserve local-runs and inspect logs\n" "$LINENO" >&2' ERR
if [[ "${1:-}" != --apply || "$(id -u)" -ne 0 ]]; then
    echo 'Usage: sudo -E bash scripts/upgrade_host.sh --apply WORKDIR' >&2
    exit 2
fi
work="${2:?Provide the same absolute WORKDIR used for the native baseline}"
[[ "$work" == /* ]]
[[ -n "${SNAPSHOT_REFERENCE:-}" ]] || { echo 'ERROR: create and verify an OS rollback snapshot first; export SNAPSHOT_REFERENCE' >&2; exit 2; }
[[ ! -f /.dockerenv ]]
command -v systemd-detect-virt >/dev/null
if systemd-detect-virt --quiet --container; then
    echo 'ERROR: this must upgrade the VM OS, not a container' >&2
    exit 2
fi
. /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]
[[ "$(cat /sys/module/amdgpu/version)" == 6.12.12 ]]
test -d "/lib/modules/$(uname -r)/build"
test -s "$work/before/host-metadata.json"
python3 - "$work" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for layout in ("single", "mpi"):
    for run in (1, 2, 3):
        data = json.loads((root / f"before/{layout}-run-{run}.json").read_text())
        if data["rccl_version"] != 22203 or data["out_of_bounds"]["count"] != 0:
            raise RuntimeError("A complete correct native baseline is required before upgrading")
PY
upgrade="$work/upgrade"
[[ ! -e "$upgrade" ]] || { echo 'ERROR: an upgrade attempt already exists; do not overwrite it' >&2; exit 2; }
mkdir -p "$upgrade/old-debs" "$upgrade/staging"
exec 9>"$work/native-benchmark.lock"
flock -n 9
printf '%s\n' "$SNAPSHOT_REFERENCE" > "$upgrade/private-snapshot-reference.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' |
    grep -E '60404|2202139' > "$upgrade/old-packages.tsv"
tar -czf "$upgrade/old-apt-dkms-boot.tgz" -C / \
    etc/apt etc/modprobe.d etc/modules-load.d var/lib/dpkg/status var/lib/dkms/amdgpu \
    "boot/initrd.img-$(uname -r)" "boot/vmlinuz-$(uname -r)"
mapfile -t old < <(awk -F '\t' '{print $1 "=" $2}' "$upgrade/old-packages.tsv")
(
    cd "$upgrade/old-debs"
    apt-get download "${old[@]}" > "$upgrade/old-package-download.txt" 2>&1
    sha256sum ./*.deb > "$upgrade/old-deb-sha256.txt"
)
curl --fail --show-error --location \
    https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb \
    --output "$upgrade/staging/amdgpu-install.deb"
install -d -m 0755 /var/cache/apt/archives
install -d -o _apt -g root -m 0700 /var/cache/apt/archives/partial
DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    -o Dpkg::Options::=--force-confnew "$upgrade/staging/amdgpu-install.deb"
apt-get update > "$upgrade/apt-update.txt" 2>&1
targets=(
    amdgpu-dkms=1:6.16.13.30300400-2341068.24.04
    rocm=7.2.4.70204-93~24.04
    rccl=2.27.7.70204-93~24.04
    rccl-dev=2.27.7.70204-93~24.04
)
apt-get --simulate install --no-install-recommends "${targets[@]}" > "$upgrade/simulation.txt" 2>&1
python3 - "$upgrade" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
text = (root / "simulation.txt").read_text()
old = {line.split("\t")[0].split(":")[0] for line in (root / "old-packages.tsv").read_text().splitlines()}
removed = re.findall(r"^Remv (\S+)", text, re.MULTILINE)
protected = re.findall(r"^(?:Inst|Remv) ((?:linux-image|linux-modules|openssh|walinuxagent|docker|moby|mlnx|doca|lustre|knem|xpmem)\S*)", text, re.MULTILINE)
if protected or any(name.split(":")[0] not in old for name in removed):
    raise RuntimeError(f"Unsafe upgrade transaction: protected={protected}, removals={removed}")
PY
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l \
    apt-get install --yes --no-install-recommends -o Dpkg::Options::=--force-confold \
    "${targets[@]}" > "$upgrade/installation.txt" 2>&1
depmod -a "$(uname -r)"
update-initramfs -u -k "$(uname -r)" >> "$upgrade/installation.txt" 2>&1
[[ "$(modinfo -F version amdgpu)" == 6.16.13 ]]
sync
echo 'INSTALLATION_COMPLETE_REBOOT_REQUIRED'
echo 'The currently loaded module may still be 6.12.12. Reboot the VM without deallocating it.'
echo 'Then run scripts/run_native.sh after with the same WORKDIR and MPI_HOME.'
