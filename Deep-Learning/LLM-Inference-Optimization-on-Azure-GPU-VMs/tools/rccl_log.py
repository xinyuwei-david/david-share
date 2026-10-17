#!/usr/bin/env python3
"""Project RCCL output without host identities, then verify the result contract.

    python tools/rccl_log.py project input.log -o public.txt
    python tools/rccl_log.py parse public.txt

Projection retains the numeric result tokens, workload header and software
version lines. It is not a new GPU run. Interleaved INFO messages are reduced
to anonymous transport markers, never copied as arbitrary log text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

NUMBER = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
ROW_START = re.compile(r"^\s*(\d+)\s+(\d+)\s+(float)\s+(sum)\s+(-1)\s*")
LOGGER_PREFIX = r"[A-Za-z0-9_][\w.-]*:\d+:\d+\s+\[\d+\]\s+NCCL\b"
LOGGER_HOST = re.compile(r"^\s*([A-Za-z0-9_][\w.-]*):\d+:\d+\s+\[\d+\]\s+NCCL\b")
QUAD = re.compile(
    rf"^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+(N/A|\d+)(?=\s|$|{LOGGER_PREFIX})"
)
PARAMETERS = re.compile(
    r"^# nThread (\d+) nGpus (\d+) minBytes (\d+) maxBytes (\d+) "
    r"step: (\d+)\(factor\) warmup iters: (\d+) iters: (\d+) "
    r"agg iters: (\d+) validation: (\d+) graph: (\d+) unalign: (\d+)\s*$"
)
PARAMETER_NAMES = (
    "threads", "gpus_per_thread", "min_bytes", "max_bytes", "factor",
    "warmup_iterations", "iterations", "aggregate_iterations", "validation",
    "graph", "unalign",
)
SOFTWARE = re.compile(
    rf"^(?:{LOGGER_PREFIX}\s+INFO\s+)?"
    r"(RCCL|ROCm|HIP) version\s*:\s*(\d+\.\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9_.:-]+)?)\s*$"
)
RANK = re.compile(r"^#\s+Rank\s+(\d+)\s+Group\s+\d+\b")
RANK_COUNT = re.compile(r"^# ranks: (\d+)$")
AVERAGE = re.compile(rf"^# Avg bus bandwidth\s*:\s*({NUMBER})\s*$")
BOUNDS = re.compile(r"^# Out of bounds values\s*:\s*(\d+)\s+(\S+)\s*$")
FINISHED = "# Collective test concluded: all_reduce_perf"


class RcclError(ValueError):
    pass


def _finish_row(prefix: list[str], quads: list[list[str]]) -> dict:
    if len(quads) != 2:
        raise RcclError(f"message size {prefix[0]} has {len(quads)} result groups, expected two")
    row = {
        "bytes": int(prefix[0]), "count": int(prefix[1]),
        "datatype": prefix[2], "operation": prefix[3], "root": int(prefix[4]),
        "tokens": prefix + quads[0] + quads[1],
    }
    if row["bytes"] != 4 * row["count"]:
        raise RcclError("float element count does not match byte count")
    for key, quad in zip(("out_of_place", "in_place"), quads):
        values = [float(v) for v in quad[:3]]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise RcclError("time and bandwidth must be finite and positive")
        row[key] = dict(zip(("time_us", "algbw_gb_s", "busbw_gb_s"), values))
        row[key]["wrong"] = None if quad[3] == "N/A" else int(quad[3])
    return row


def parse(text: str) -> dict:
    lines = text.splitlines()
    hosts = {match[1] for line in lines if (match := LOGGER_HOST.match(line))}
    if hosts:
        names = "|".join(re.escape(host) for host in sorted(hosts, key=len, reverse=True))
        boundary = rf"(?:{names}):\d+:\d+\s+\[\d+\]\s+NCCL\b"
        quad_pattern = re.compile(
            rf"^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+(N/A|\d+?)(?=\s|$|{boundary})"
        )
    else:
        quad_pattern = QUAD
    parameters, parameter_line, software = None, None, {}
    ranks, projected_ranks, rows = set(), None, []
    prefix, quads = None, []
    average, bounds, concluded = None, None, 0
    network = {"IB": 0, "Socket": 0, "GDRDMA": 0}
    for line in lines:
        if re.search(r"Test (?:NCCL|CUDA) failure|Segmentation fault|MPI_ABORT", line):
            raise RcclError("the log contains a runtime failure")
        if match := PARAMETERS.fullmatch(line):
            if parameters is not None:
                raise RcclError("multiple workload headers are not one run")
            parameters = dict(zip(PARAMETER_NAMES, map(int, match.groups())))
            parameter_line = line
        if match := SOFTWARE.fullmatch(line):
            name, value = match.groups()
            if name in software and software[name] != value:
                raise RcclError(f"conflicting {name} versions")
            software[name] = value
        if match := RANK.match(line):
            ranks.add(int(match[1]))
        if match := RANK_COUNT.fullmatch(line):
            if projected_ranks is not None:
                raise RcclError("duplicate projected rank count")
            projected_ranks = int(match[1])
        for transport in ("IB", "Socket"):
            if re.search(rf"\bNCCL INFO Using network {transport}\s*$", line):
                network[transport] += 1
        if re.search(r"\bvia NET/IB/\d+/GDRDMA\b", line):
            network["GDRDMA"] += 1
        if match := ROW_START.match(line):
            if prefix is not None:
                rows.append(_finish_row(prefix, quads))
            prefix, quads = list(match.groups()), []
            line = line[match.end():]
        if prefix is not None:
            while match := quad_pattern.match(line):
                quads.append(list(match.groups()))
                line = line[match.end():]
        if match := AVERAGE.fullmatch(line):
            if average is not None:
                raise RcclError("duplicate average bandwidth")
            average = float(match[1])
        if match := BOUNDS.fullmatch(line):
            if bounds is not None:
                raise RcclError("duplicate bounds result")
            bounds = (int(match[1]), match[2])
        if line.strip() == FINISHED:
            concluded += 1
    if prefix is not None:
        rows.append(_finish_row(prefix, quads))
    if parameters is None or average is None or concluded != 1:
        raise RcclError("missing workload, average or successful completion marker")
    if bounds != (0, "OK"):
        raise RcclError("bounds check did not report zero errors")
    if not {"RCCL", "ROCm"} <= software.keys():
        raise RcclError("RCCL and ROCm version evidence is required")
    rank_count = projected_ranks if projected_ranks is not None else len(ranks)
    if rank_count < 2 or (ranks and ranks != set(range(rank_count))):
        raise RcclError("ranks must be a complete, contiguous group")
    if parameters["threads"] != 1 or parameters["gpus_per_thread"] != 1:
        raise RcclError("this evidence protocol requires one GPU per rank")
    if parameters["factor"] < 2 or parameters["min_bytes"] <= 0:
        raise RcclError("invalid size sweep")
    expected, size = [], parameters["min_bytes"]
    while size <= parameters["max_bytes"] and len(expected) < 64:
        expected.append(size)
        size *= parameters["factor"]
    if [r["bytes"] for r in rows] != expected or not expected or size <= parameters["max_bytes"]:
        raise RcclError("result rows do not match the complete message-size sweep")
    values = []
    for row in rows:
        for mode in ("out_of_place", "in_place"):
            result = row[mode]
            wrong = result["wrong"]
            if parameters["validation"] and wrong != 0:
                raise RcclError("the data check is enabled but a wrong-count is nonzero or missing")
            if not parameters["validation"] and wrong is not None:
                raise RcclError("performance-only run unexpectedly claims a data check")
            factor = 2 * (rank_count - 1) / rank_count
            if abs(result["busbw_gb_s"] - result["algbw_gb_s"] * factor) > 0.03:
                raise RcclError("bus bandwidth does not match the AllReduce normalization")
            values.append(result["busbw_gb_s"])
    if not math.isfinite(average) or abs(statistics.fmean(values) - average) > 0.01:
        raise RcclError("reported average differs from the printed in-place/out-of-place results")
    return {
        "parameters": parameters, "parameter_line": parameter_line, "ranks": rank_count,
        "software": software, "rows": rows, "average_busbw_gb_s": average,
        "data_check": "PASS" if parameters["validation"] else "NOT_RUN",
        "network_markers": network,
    }


def project(text: str) -> str:
    run = parse(text)
    lines = [
        "# Public numeric projection of one completed all_reduce_perf run.",
        "# Host identifiers and unrelated diagnostic lines were removed.",
        run["parameter_line"], f"# ranks: {run['ranks']}",
    ]
    for name in ("RCCL", "ROCm", "HIP"):
        if name in run["software"]:
            lines.append(f"{name} version : {run['software'][name]}")
    lines.append("# bytes count datatype operation root oop_time_us oop_algbw oop_busbw oop_wrong ip_time_us ip_algbw ip_busbw ip_wrong")
    for row in run["rows"]:
        lines.append(" ".join(row["tokens"]))
    for transport in ("IB", "Socket"):
        lines.extend([f"NCCL INFO Using network {transport}"] * run["network_markers"][transport])
    for line in text.splitlines():
        if match := re.search(r"\bvia NET/IB/\d+/GDRDMA\b", line):
            lines.append("NCCL INFO " + match[0])
    lines.append("# Out of bounds values : 0 OK")
    average = next(line for line in text.splitlines() if AVERAGE.fullmatch(line))
    lines.extend([average.rstrip(), FINISHED])
    output = "\n".join(lines) + "\n"
    parsed = parse(output)
    if parsed != run:
        raise RcclError("projection changed parsed results")
    return output


def build_evidence(directory: Path) -> dict:
    manifest = json.loads((directory / "runs.json").read_text(encoding="utf-8"))
    results, listed, identities = [], set(), set()
    for record in manifest["runs"]:
        filename = record["file"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.txt", filename) or filename in listed:
            raise RcclError("source files must be unique basenames")
        if record["id"] in identities:
            raise RcclError("run identities must be unique")
        identities.add(record["id"])
        listed.add(filename)
        source = directory / "raw" / filename
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise RcclError(f"source hash mismatch: {filename}")
        if not re.fullmatch(r"[0-9a-f]{64}", record["original_sha256"]):
            raise RcclError("missing original-log digest")
        parsed = parse(data.decode("utf-8"))
        expected = record["expected"]
        if parsed["parameters"] != expected["parameters"] or parsed["ranks"] != expected["ranks"]:
            raise RcclError(f"workload identity differs: {filename}")
        if parsed["software"] != expected["software"]:
            raise RcclError(f"software identity differs: {filename}")
        results.append({
            "id": record["id"], "label": record["label"], "comparison": record["comparison"],
            "environment": record["environment"], "command": record["command"],
            "boundary": record["boundary"], "source": "evidence/rccl/raw/" + filename,
            **parsed,
        })
    actual = {p.name for p in (directory / "raw").iterdir() if p.is_file()}
    if actual != listed:
        raise RcclError("RCCL source manifest does not cover the exact raw directory")
    observation = json.loads((directory / "observation.json").read_text(encoding="utf-8"))
    if Path(observation["file"]).name != observation["file"]:
        raise RcclError("invalid diagnostic path")
    diagnostic = (directory / observation["file"]).read_bytes()
    if hashlib.sha256(diagnostic).hexdigest() != observation["sha256"]:
        raise RcclError("diagnostic excerpt hash mismatch")
    retries = observation["completed_retry_source_sha256"]
    if diagnostic.decode("utf-8").count(FINISHED) != len(retries):
        raise RcclError("diagnostic retry count differs")
    for digest in [observation["failure_source_sha256"], *retries]:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RcclError("invalid diagnostic source hash")
    return {
        "protocol": manifest["protocol"], "runs": results,
        "initialization_observation": {
            "failures_recorded": 1, "completed_targeted_retries": len(retries),
            "root_cause_status": observation["root_cause_status"],
            "source": "evidence/rccl/" + observation["file"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("project", "parse"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        if name == "project":
            command.add_argument("-o", "--output", required=True, type=Path)
        else:
            command.add_argument("--require-ib", action="store_true")
    args = parser.parse_args(argv)
    try:
        text = args.input.read_text(encoding="utf-8")
        if args.command == "project":
            output = project(text)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
            print(f"wrote {args.output.name}")
        else:
            record = parse(text)
            network = record["network_markers"]
            if args.require_ib and (
                network["IB"] != record["ranks"] or network["Socket"] or not network["GDRDMA"]
            ):
                raise RcclError("the log does not prove IB and GPU Direct RDMA on every rank")
            print(json.dumps(record, indent=2, ensure_ascii=False))
    except (OSError, RcclError) as error:
        parser.exit(1, f"RCCL_EVIDENCE_ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
