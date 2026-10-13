import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1284e5da11f7904e8a6efd902b9de52253785bd0"
SIZES = tuple(1 << exponent for exponent in range(20, 31))
CODES = {"before": 22203, "after": 22707}
PHASES = ("out_of_place", "in_place")


def validate_output(data: dict, log: str, layout: str, code: int, sizes: tuple[int, ...] = SIZES) -> None:
    if data["version"] != 3 or data["rccl_version"] != code or data["nccl_version"] != code:
        raise ValueError("Unexpected producer format or actual RCCL/header version")
    cfg = data["config"]
    required = {
        "nthreads": 1, "ngpus": 8 if layout == "single" else 1,
        "iterations": 100, "warmup_iters": 20, "validation": 1, "graph": 0,
        "aggregated_iterations": 1, "minimum_bytes": sizes[0], "maximum_bytes": sizes[-1],
    }
    for key, value in required.items():
        if cfg[key] != value:
            raise ValueError(f"Unexpected test configuration: {key}")
    if len(sizes) > 1 and cfg["step_factor"] != 2:
        raise ValueError("Unexpected message-size progression")
    if data["out_of_bounds"] != {"count": 0, "okay": "true"}:
        raise ValueError("Producer data validation failed")
    if any(error.strip() for error in data["errors"]):
        raise ValueError("Producer reported errors")
    if COMMIT not in log or f"rccl-library={code}" not in log:
        raise ValueError("Source or actual loaded-library provenance mismatch")
    ranks = re.findall(
        r"^#\s+Rank\s+(\d+)\s+Group\s+0\s+Pid\s+(\d+).*?\bdevice\s+(\d+)\b",
        log, re.MULTILINE,
    )
    if len(ranks) != 8 or sorted(int(rank) for rank, _, _ in ranks) != list(range(8)):
        raise ValueError("Expected eight distinct ranks")
    if sorted(int(gpu) for _, _, gpu in ranks) != list(range(8)):
        raise ValueError("Expected eight distinct GPUs")
    if len({pid for _, pid, _ in ranks}) != (1 if layout == "single" else 8):
        raise ValueError("Unexpected process layout")
    if sorted(row["size"] for row in data["results"]) != list(sizes):
        raise ValueError("Missing or duplicate message size")
    for row in data["results"]:
        if row["type"] != "float" or row["redop"] != "sum" or row["count"] != row["size"] // 4:
            raise ValueError("Datatype, reduction or count mismatch")
        if row["actual_iterations"] != 100:
            raise ValueError("Unexpected actual iteration count")
        for phase in PHASES:
            result = row[phase]
            time, alg, bus = result["time"], result["alg_bw"], result["bus_bw"]
            if result["nwrong"] != 0 or not all(math.isfinite(value) and value > 0 for value in (time, alg, bus)):
                raise ValueError("Nonzero #wrong or invalid metrics")
            if not math.isclose(alg, row["size"] / time / 1000, rel_tol=1e-6, abs_tol=5e-6):
                raise ValueError("Time/algbw units mismatch")
            if not math.isclose(bus, alg * 1.75, rel_tol=1e-6, abs_tol=5e-6):
                raise ValueError("Eight-rank AllReduce normalization mismatch")


def compute_comparison() -> list[dict]:
    groups: dict[tuple[str, str, int, str], list[float]] = {}
    for state, code in CODES.items():
        for layout in ("single", "mpi"):
            for run in (1, 2, 3):
                path = ROOT / "evidence" / state / f"{layout}-run-{run}.json"
                data = json.loads(path.read_text())
                validate_output(data, path.with_suffix(".txt").read_text(), layout, code)
                for row in data["results"]:
                    for phase in PHASES:
                        groups.setdefault((state, layout, row["size"], phase), []).append(row[phase]["bus_bw"])
    result = []
    for layout in ("single", "mpi"):
        for size in SIZES:
            for phase in PHASES:
                before = groups[("before", layout, size, phase)]
                after = groups[("after", layout, size, phase)]
                if len(before) != 3 or len(after) != 3:
                    raise ValueError("Expected three independent native runs")
                old, new = median(before), median(after)
                result.append({
                    "layout": layout, "size_bytes": size, "phase": phase,
                    "before_bus_bw_gb_s": old, "after_bus_bw_gb_s": new,
                    "change_percent": (new / old - 1) * 100,
                    "statistic": "median_of_3_native_runs",
                })
    return result


def validate_driver_isolation(summary: dict) -> None:
    for version, code in (("6.4.4", 22203), ("7.2.4", 22707)):
        old = []
        for run in (1, 2, 3):
            path = ROOT / "evidence/fixed-images-old-driver" / f"rocm-{version}-run-{run}.json"
            data = json.loads(path.read_text())
            validate_output(data, path.with_suffix(".txt").read_text(), "single", code)
            old.extend(row for row in data["results"] if row["size"] == 1 << 30)
        path = ROOT / "evidence/fixed-images-new-driver" / f"new-driver-same-container-{version}.json"
        new = json.loads(path.read_text())
        validate_output(new, path.with_suffix(".txt").read_text(), "single", code, (1 << 30,))
        for phase in PHASES:
            saved = next(row for row in summary["driver_isolation_diagnostics"]
                         if row["userspace_rocm"] == version and row["phase"] == phase)
            if saved["old_driver_6_12_12_bus_bw_gb_s"] != median(row[phase]["bus_bw"] for row in old):
                raise ValueError("Old-driver diagnostic does not match original evidence")
            if saved["new_driver_6_16_13_bus_bw_gb_s"] != new["results"][0][phase]["bus_bw"]:
                raise ValueError("New-driver diagnostic does not match original evidence")


def render_table(rows: list[dict], chinese: bool) -> str:
    if chinese:
        lines = ["| 进程布局 | 升级前 GB/s | 升级后 GB/s | 性能下降 |",
                 "|---|---:|---:|---:|"]
        names = {"single": "单进程、8 GPU", "mpi": "8 进程、每进程 1 GPU"}
    else:
        lines = ["| Process layout | Before GB/s | After GB/s | Decrease |",
                 "|---|---:|---:|---:|"]
        names = {"single": "1 process, 8 GPUs", "mpi": "8 processes, 1 GPU each"}
    for row in rows:
        if row["size_bytes"] == 1 << 30 and row["phase"] == "out_of_place":
            lines.append(
                f'| {names[row["layout"]]} | {row["before_bus_bw_gb_s"]:.2f} | '
                f'{row["after_bus_bw_gb_s"]:.2f} | {-row["change_percent"]:.2f}% |'
            )
    return "\n".join(lines)


def render_isolation(summary: dict, chinese: bool) -> str:
    heading = (
        "| 保持不变的用户态镜像 | 驱动 6.12.12，GB/s | 驱动 6.16.13，GB/s |"
        if chinese else "| Fixed userspace image | Driver 6.12.12, GB/s | Driver 6.16.13, GB/s |"
    )
    lines = [heading, "|---|---:|---:|"]
    for rocm, rccl in (("6.4.4", "2.22.3"), ("7.2.4", "2.27.7")):
        row = next(item for item in summary["driver_isolation_diagnostics"]
                   if item["userspace_rocm"] == rocm and item["phase"] == "out_of_place")
        lines.append(
            f'| ROCm {rocm} / RCCL {rccl} | {row["old_driver_6_12_12_bus_bw_gb_s"]:.2f} | '
            f'{row["new_driver_6_16_13_bus_bw_gb_s"]:.2f} |'
        )
    return "\n".join(lines)


def render_configurations(summary: dict, chinese: bool) -> str:
    before, after = summary["before"], summary["after"]
    native = next(row for row in summary["native_comparison"]
                  if row["layout"] == "single" and row["size_bytes"] == 1 << 30
                  and row["phase"] == "out_of_place")
    old_image = next(row for row in summary["driver_isolation_diagnostics"]
                     if row["userspace_rocm"] == "6.4.4" and row["phase"] == "out_of_place")
    new_image = next(row for row in summary["driver_isolation_diagnostics"]
                     if row["userspace_rocm"] == "7.2.4" and row["phase"] == "out_of_place")
    cases = (
        (1, before, None, native["before_bus_bw_gb_s"], True),
        (2, before, ("7.2.4", "2.27.7"), new_image["old_driver_6_12_12_bus_bw_gb_s"], True),
        (3, after, None, native["after_bus_bw_gb_s"], True),
        (4, after, ("6.4.4", "2.22.3"), old_image["new_driver_6_16_13_bus_bw_gb_s"], False),
    )
    if chinese:
        lines = ["| 配置 | VM 内安装的用户态 | 容器内安装的用户态 | 测试位置 | 实际使用的 AMDGPU 内核驱动 | busbw GB/s | 统计口径 |",
                 "|---|---|---|---|---|---:|---|"]
    else:
        lines = ["| Configuration | Userspace installed in the VM | Userspace installed in the container | Test location | Actually used AMDGPU kernel driver | busbw GB/s | Statistic |",
                 "|---|---|---|---|---|---:|---|"]
    for number, host, container, bandwidth, repeated in cases:
        host_rocm = host["rocm_default_version"].split("-")[0]
        host_rccl = "2.22.3" if host["rccl_runtime_code"] == 22203 else "2.27.7"
        vm = f"ROCm `{host_rocm}` / RCCL `{host_rccl}`"
        if container is None:
            image = "**没有容器**" if chinese else "**No container**"
            location = "VM 操作系统原生" if chinese else "Native VM OS"
            driver = f'`{host["loaded_amdgpu_version"]}`（VM）' if chinese else f'`{host["loaded_amdgpu_version"]}` (VM)'
        else:
            image = f"ROCm `{container[0]}` / RCCL `{container[1]}`"
            location = "容器内" if chinese else "Inside container"
            driver = f'`{host["loaded_amdgpu_version"]}`（共享 VM 驱动）' if chinese else f'`{host["loaded_amdgpu_version"]}` (shared VM driver)'
        statistic = ("三轮中位数" if repeated else "单轮额外诊断") if chinese else ("Median of 3 runs" if repeated else "One extra diagnostic run")
        lines.append(f"| {number} | {vm} | {image} | {location} | {driver} | {bandwidth:.2f} | {statistic} |")
    return "\n".join(lines)


def render_kernel_counts(summary: dict, chinese: bool) -> str:
    heading = (
        "| 在完整私有启动日志中检索的模式 | 旧驱动启动 | 新驱动启动 |"
        if chinese else "| Search pattern in the complete private boot journals | Old-driver boot | New-driver boot |"
    )
    lines = [heading, "|---|---:|---:|"]
    counts = summary["kernel_symptom_counts"]
    for pattern in ("xgmi_connected_port_num", "psp_xgmi_get_topology_info", "INVOKE_CMD(0x3) failed"):
        lines.append(f'| `{pattern}` | {counts["before"][pattern]} | {counts["after"][pattern]} |')
    return "\n".join(lines)


def replace_generated_block(text: str, section: str, table: str) -> str:
    pattern = rf"(<!-- BEGIN GENERATED: {section} -->\n).*?(<!-- END GENERATED: {section} -->)"
    updated, count = re.subn(
        pattern, lambda match: match[1] + table + "\n" + match[2], text, flags=re.DOTALL
    )
    if count != 1:
        raise ValueError(f"Expected one canonical {section} table")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Regenerate comparison.csv and README tables")
    parser.add_argument("--check", action="store_true", help="Check evidence and committed generated content")
    args = parser.parse_args()
    summary = json.loads((ROOT / "data/native-summary.json").read_text())
    rows = compute_comparison()
    if rows != summary["native_comparison"]:
        raise ValueError("Committed summary differs from original producer measurements")
    if summary["primary_measurement_count"] != 264 or len(rows) != 44:
        raise ValueError("Unexpected experiment coverage")
    before, after = summary["before"], summary["after"]
    if before["execution"] != "native_vm_os_not_container" or after["execution"] != before["execution"]:
        raise ValueError("A container result was substituted for native host measurements")
    if before["kernel"] != after["kernel"] or before["mpi_version"] != after["mpi_version"]:
        raise ValueError("Kernel or MPI changed during the host comparison")
    if before["loaded_amdgpu_version"] != "6.12.12" or after["loaded_amdgpu_version"] != "6.16.13":
        raise ValueError("Actual loaded-driver change was not demonstrated")
    installed = json.loads((ROOT / "evidence/installed-module-before-reboot.json").read_text())
    boot = json.loads((ROOT / "evidence/before/driver-boot-evidence.json").read_text())
    if after["loaded_amdgpu_srcversion"] != installed["module_file_srcversion"] or after["boot_id"] == boot["boot_id"]:
        raise ValueError("Actual new module loading and a real reboot were not demonstrated")
    validate_driver_isolation(summary)
    csv_path = ROOT / "data/comparison.csv"
    if args.write:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with csv_path.open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != len(rows):
            raise ValueError("CSV coverage mismatch")
        for record, csv_row in zip(rows, csv_rows, strict=True):
            for key in ("before_bus_bw_gb_s", "after_bus_bw_gb_s", "change_percent"):
                if float(csv_row[key]) != record[key]:
                    raise ValueError("CSV/JSON information alignment failed")
    for filename, chinese in (("README.md", False), ("README_CN.md", True)):
        path = ROOT / filename
        text = path.read_text(encoding="utf-8")
        updated = text
        for section, table in (
            ("native-results", render_table(rows, chinese)),
            ("execution-configurations", render_configurations(summary, chinese)),
            ("driver-isolation", render_isolation(summary, chinese)),
            ("kernel-symptoms", render_kernel_counts(summary, chinese)),
        ):
            updated = replace_generated_block(updated, section, table)
        if args.write:
            path.write_text(updated, encoding="utf-8")
        elif updated != text:
            raise ValueError(f"{filename}: published numbers do not match canonical evidence")
    print("PASS: 12 native runs / 264 phase measurements / 44 comparison rows / bilingual tables")
    print("PASS: same-source fixed-image driver diagnostics and actually loaded native driver")


if __name__ == "__main__":
    main()
