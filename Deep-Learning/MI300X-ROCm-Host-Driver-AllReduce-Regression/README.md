# Azure MI300X: AllReduce regression after a native host-driver upgrade

On one Azure ND MI300X v5 VM, upgrading the **VM operating system's AMDGPU kernel driver together with ROCm/RCCL** reduced eight-GPU AllReduce bandwidth from approximately 315 GB/s to 42 GB/s. The data checks still passed. Updating only the userspace libraries in containers on the old driver did **not** reproduce this regression.

This directory is a reproducible investigation case, **not a production upgrade recommendation or a vendor-confirmed root cause**. “Host” means the VM OS used as the container host, not the underlying Azure physical host or its GIM driver.

Author: Xinyu Wei · [中文](README_CN.md) · [Canonical data](data/native-summary.json) · [Original redacted evidence](evidence/) · [Offline verification](#verify-without-a-gpu)

## Measured native before/after

**1 GiB message, out-of-place AllReduce `busbw`, median of three independent runs. Bandwidth is decimal GB/s.**

<!-- BEGIN GENERATED: native-results -->
| Process layout | Before GB/s | After GB/s | Decrease |
|---|---:|---:|---:|
| 1 process, 8 GPUs | 314.60 | 41.78 | 86.72% |
| 8 processes, 1 GPU each | 314.81 | 41.74 | 86.74% |
<!-- END GENERATED: native-results -->

| Item | Before | After |
|---|---|---|
| Execution | Native VM OS | Native VM OS; real reboot |
| Loaded AMDGPU module | `6.12.12` | `6.16.13` |
| AMD driver package | `1:6.12.12.60404-2202139.24.04` | `1:6.16.13.30300400-2341068.24.04` (AMDGPU 30.30.4) |
| ROCm | `6.4.4` | `7.2.4` |
| RCCL | `2.22.3` | `2.27.7` |
| Linux kernel | `6.8.0-1067-azure` | Unchanged |
| MPI | HPC-X Open MPI `4.1.7rc1` | Unchanged |
| GPU partitioning | 8 MI300X VF GPUs, `SPX / NPS1` | Unchanged |
| Data validation | All `#wrong = 0` | All `#wrong = 0` |

Both stacks used [the same official rccl-tests source](https://github.com/ROCm/rocm-systems/tree/1284e5da11f7904e8a6efd902b9de52253785bd0/projects/rccl-tests), commit `1284e5da11f7904e8a6efd902b9de52253785bd0`, rebuilt against each native HIP/RCCL installation with `MPI=1` and `GPU_TARGETS=gfx942`. The test swept **1 MiB to 1 GiB**, doubling message size, with FP32 `sum`, 20 warmups, 100 iterations, graph capture disabled, and both out-of-place and in-place results.

The primary evidence comprises **12 native runs, 264 phase measurements and 44 before/after comparison rows**. [comparison.csv](data/comparison.csv) contains every message-size comparison. Do not compare the mixed-size `Avg bus bandwidth` with a large-message plateau, or treat normalized AllReduce `busbw` as raw XGMI link throughput.

## Why the container-only result was misleading

Three experiments answer different questions:

1. **Old host driver, userspace-only container A/B:** both ROCm 6.4.4/RCCL 2.22.3 and ROCm 7.2.4/RCCL 2.27.7 remained fast. This did not test activating a new kernel module.
2. **Native host-stack upgrade and reboot:** both process layouts regressed to the low-bandwidth plateau in the primary table.
3. **Retain the same immutable userspace images, run them on the new loaded host driver:** even the old userspace image became slow. The old-driver values below are medians of three runs; the new-driver values are **one diagnostic run each**, not the primary native result.

<!-- BEGIN GENERATED: driver-isolation -->
| Fixed userspace image | Driver 6.12.12, GB/s | Driver 6.16.13, GB/s |
|---|---:|---:|
| ROCm 6.4.4 / RCCL 2.22.3 | 314.76 | 37.44 |
| ROCm 7.2.4 / RCCL 2.27.7 | 321.67 | 41.70 |
<!-- END GENERATED: driver-isolation -->

[The immutable image IDs and diagnostic measurements](data/native-summary.json) and [their raw output](evidence/fixed-images-new-driver/) are retained. This supports investigating the **host-driver transition and its firmware/virtualization/interconnect interaction**, rather than asserting that RCCL 2.27.7 alone must be slow.

## New kernel symptoms, not yet a proven code root cause

The new-driver boot reported:

```text
psp gfx command INVOKE_CMD(0x3) failed and response status is (0xFFFF000F)
index 8 is out of range for type 'xgmi_connected_port_num [8]'
psp_xgmi_get_topology_info
amdgpu_xgmi_add_device
```

<!-- BEGIN GENERATED: kernel-symptoms -->
| Search pattern in the complete private boot journals | Old-driver boot | New-driver boot |
|---|---:|---:|
| `xgmi_connected_port_num` | 0 | 4 |
| `psp_xgmi_get_topology_info` | 0 | 4 |
| `INVOKE_CMD(0x3) failed` | 0 | 8 |
<!-- END GENERATED: kernel-symptoms -->

See [the redacted old boot excerpt](evidence/before/kernel-xgmi-excerpt.txt), [the new boot excerpt](evidence/after/kernel-xgmi-excerpt.txt) and [search counts](data/kernel-symptom-counts.json). Counts came from the complete original journals; the public excerpts omit unrelated lines.

**Both drivers' topology output still labels the links `XGMI`.** A link label, successful GPU enumeration or `#wrong = 0` is not proof of healthy transfer performance. These errors are correlated symptoms, not proof that a particular array access caused the bandwidth loss. Physical-host GIM/firmware details were not collected, and **a driver-only rollback restoring bandwidth has not yet been tested**.

## Reproduce natively, in execution order

### 0. Reserve a disposable, isolated test VM

The measured environment was Ubuntu 24.04 on Azure `Standard_ND96isr_MI300X_v5`. Start with native ROCm 6.4.4/RCCL 2.22.3 and loaded driver 6.12.12. Stop your own GPU workloads before testing. Do not alter another user's running jobs.

The new-driver combination reproduced poor performance. Use a disposable VM, maintenance window, verified recovery access and a verified OS-disk snapshot. Keep exact old package backups. **Do not deallocate a scarce Spot GPU VM merely to reboot it**; capacity may not be recoverable. Snapshot storage and a running VM incur costs.

Example snapshot creation from an already authenticated **isolated** Azure CLI environment:

```powershell
$env:AZURE_CONFIG_DIR = "C:\Temp\az-mi300x-case"
$subscription = "<YOUR_SUBSCRIPTION_ID>"
$resourceGroup = "<YOUR_RESOURCE_GROUP>"
$vm = "<YOUR_VM_NAME>"
$disk = az vm show --subscription $subscription --resource-group $resourceGroup --name $vm --query storageProfile.osDisk.managedDisk.id -o tsv
az snapshot create --subscription $subscription --resource-group $resourceGroup --name mi300x-pre-driver-upgrade --source $disk --incremental true --sku Standard_LRS
az snapshot show --subscription $subscription --resource-group $resourceGroup --name mi300x-pre-driver-upgrade --query provisioningState -o tsv
```

Success: snapshot state `Succeeded`. On failure, **do not install the new driver**. Keep the snapshot reference private; never commit it.

### 1. Record and run the native baseline

Run from this directory on the Linux VM, not from a container:

```bash
export WORKDIR="$PWD/local-runs/native-ab"
# Use the same existing MPI installation for both phases.
export MPI_HOME="/opt/hpcx-v2.23-gcc-doca_ofed-ubuntu24.04-cuda12-x86_64/hpcx-rebuild"
sudo -E bash scripts/run_native.sh before "$WORKDIR"
```

Success: six runs finish with `NATIVE_PHASE_BENCHMARK_PASS`, every data check passes, and `before/host-metadata.json` records actual loaded driver 6.12.12. Adjust `MPI_HOME` to your installed MPI; do not change it between phases. If the source, library, native-execution or data-check guard fails, inspect the files in `WORKDIR/before` and stop before upgrading.

### 2. Opt in to the native upgrade, then reboot

```bash
export SNAPSHOT_REFERENCE="<YOUR_VERIFIED_PRIVATE_OS_SNAPSHOT_REFERENCE>"
sudo -E bash scripts/upgrade_host.sh --apply "$WORKDIR"
```

The script backs up installed old package versions and APT/DKMS/boot state, downloads the official installer, uses signed repositories, pins the measured packages, and rejects simulated removal of unrelated software or changes to the kernel, SSH, Azure Agent or RDMA components. It installs into the **VM OS**:

```text
amdgpu-dkms=1:6.16.13.30300400-2341068.24.04
rocm=7.2.4.70204-93~24.04
rccl=2.27.7.70204-93~24.04
rccl-dev=2.27.7.70204-93~24.04
```

Success before reboot: `INSTALLATION_COMPLETE_REBOOT_REQUIRED`. A newly installed module file is **not** the currently loaded module. If download, transaction safety, package installation or DKMS compilation fails, preserve the logs and do not reboot into an unverified driver.

Then reboot the guest OS in place:

```bash
sudo reboot
```

Reconnect using your existing configured SSH port and credentials. Check:

```bash
cat /sys/module/amdgpu/version
cat /sys/module/amdgpu/srcversion
cat /proc/sys/kernel/random/boot_id
dkms status
/opt/rocm/bin/rocm-smi --showproductname
/opt/rocm/bin/rocm-smi --showcomputepartition --showmemorypartition
```

Success: actual loaded module 6.16.13, a new boot ID, matching module `srcversion`, and eight GPUs still in `SPX / NPS1`. GPU initialization took several minutes in this case. If SSH or GPU initialization fails, use the verified management/recovery path and your snapshot or old packages; a full automatic rollback is deliberately **not** supplied as a tested fix.

### 3. Rebuild and run the native upgraded stack

```bash
sudo -E bash scripts/run_native.sh after "$WORKDIR"
```

Success: the collector confirms actual native RCCL 2.27.7, actual loaded driver 6.16.13 and a real reboot; kernel and MPI match the baseline; six runs pass correctness checks. Low bandwidth is a valid experimental result, **not** a reason to discard a correct run. If GPU partitions or topology differ, record that difference before making a causal claim.

Raw JSON and text logs under `local-runs/` are ignored by Git and may contain identifiers. Redact copies before sharing; never commit snapshots, credentials, backups or private run directories.

## Next controlled experiments

These are **not yet measured conclusions**:

1. Keep ROCm 7.2.4/RCCL 2.27.7, roll back only the kernel-driver stack, reboot and check whether bandwidth recovers. Record dependency changes and firmware packages, not just the driver version label.
2. On a fresh protected test VM, compare intermediate supported driver releases with fixed userspace, kernel, partitioning, code and MPI. Locate the first bad driver.
3. Capture `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM` and compare the actual channel paths and selected algorithms. Do not infer physical path changes merely from an environment-variable value.
4. Ask AMD/Azure engineering to correlate the XGMI topology/PSP symptoms with VF/GIM/BKC compatibility. Do not modify or claim control over Azure's physical hypervisor/GIM from a guest VM.
5. Only after the driver/VF mismatch is understood, isolate a kernel-code fix and demonstrate **A/B/A** restoration on the same hardware.

## Verify without a GPU

Python 3.10+ and the standard library are sufficient:

```bash
python3 tools/verify_case.py --check
python3 -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
```

Success: all original measurements, actual loaded-driver/reboot evidence, fixed-image diagnostics and both README tables agree. To regenerate only derived tables and CSV:

```bash
python3 tools/verify_case.py --write
python3 tools/verify_case.py --check
```

Do not edit the raw evidence to make a result pass.

## Evidence and publication boundaries

The public JSON/text files are **redacted copies of real rccl-tests output and kernel journals**, not mock results. Host/IP/account/cloud-resource identifiers, boot UUIDs, hive/MAC identifiers and memory addresses are masked. Trailing whitespace in the published text copies is normalized. Numeric measurements, software versions, test options and error codes are unchanged. Original private records are retained separately.

No password, SSH key, tenant/subscription ID, snapshot reference, model weights, package archive or private host is published. The original experiment retained the upgraded low-bandwidth state for investigation; it did not claim an untested downgrade restored performance.

## Producer references

- [ROCm 7.2.4 native Ubuntu installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/install-methods/package-manager/package-manager-ubuntu.html)
- [AMDGPU 30.30.4 native Ubuntu installation](https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.30.4/install/detailed-install/package-manager/package-manager-ubuntu.html)
- [User/kernel driver compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/reference/user-kernel-space-compat-matrix.html)
- [RCCL 2.27.7 environment variables](https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.4/api-reference/env-variables.html)
- [Pinned official rccl-tests source](https://github.com/ROCm/rocm-systems/tree/1284e5da11f7904e8a6efd902b9de52253785bd0/projects/rccl-tests)
