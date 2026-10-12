import argparse
import ctypes
import json
import os
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    if Path("/.dockerenv").exists():
        raise RuntimeError("Run the primary experiment in the VM OS, not Docker")
    detector = subprocess.run(["systemd-detect-virt", "--container"], capture_output=True, text=True)
    if detector.returncode == 0:
        raise RuntimeError("A container cannot prove a host kernel-driver upgrade")
    if detector.returncode != 1:
        raise RuntimeError(f"Cannot establish native execution: {detector.stderr}")
    version, rccl, driver = {
        "before": ("6.4.4", 22203, "6.12.12"),
        "after": ("7.2.4", 22707, "6.16.13"),
    }[args.phase]
    loaded = Path("/sys/module/amdgpu/version").read_text().strip()
    srcversion = Path("/sys/module/amdgpu/srcversion").read_text().strip()
    file_srcversion = subprocess.check_output(["modinfo", "-F", "srcversion", "amdgpu"], text=True).strip()
    if loaded != driver or srcversion != file_srcversion:
        raise RuntimeError("The expected native driver is not actually loaded; check reboot and DKMS")
    libraries = subprocess.check_output(["ldd", str(args.binary)], text=True)
    match = re.search(r"librccl\.so\.1 => (\S+)", libraries)
    if match is None or "not found" in libraries:
        raise RuntimeError(f"Native library resolution failed:\n{libraries}")
    actual_library = Path(match[1]).resolve()
    if not str(actual_library).startswith(f"/opt/rocm-{version}/lib/"):
        raise RuntimeError(f"Unexpected native RCCL library: {actual_library}")
    actual = ctypes.c_int()
    status = ctypes.CDLL(str(actual_library)).ncclGetVersion(ctypes.byref(actual))
    if status != 0 or actual.value != rccl:
        raise RuntimeError(f"Actual RCCL runtime mismatch: {status}/{actual.value}")
    print(json.dumps({
        "execution": "native_vm_os_not_container",
        "phase": args.phase,
        "kernel": subprocess.check_output(["uname", "-r"], text=True).strip(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "loaded_amdgpu_version": loaded,
        "loaded_amdgpu_srcversion": srcversion,
        "module_file_srcversion": file_srcversion,
        "driver_package": subprocess.check_output(["dpkg-query", "-W", "amdgpu-dkms"], text=True).strip(),
        "actual_rccl_library": str(actual_library),
        "rccl_runtime_code": actual.value,
        "rocm_default_version": Path("/opt/rocm/.info/version").read_text().strip(),
        "mpi_version": subprocess.check_output(["mpirun", "--version"], text=True).splitlines()[0],
        "collective_environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("NCCL_", "RCCL_", "HIP_", "HSA_", "ROCR_"))
        },
        "linked_libraries": libraries.splitlines(),
    }, indent=2))


if __name__ == "__main__":
    main()
