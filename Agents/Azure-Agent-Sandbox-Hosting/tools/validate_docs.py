#!/usr/bin/env python3
"""Validate public scope, canonical claims, run order, bilingual cells, links and privacy guards."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import render_docs as rd

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("README.md", "README_CN.md", "VALIDATION.md", "VALIDATION_CN.md")
PAIRS = (("README.md", "README_CN.md"), ("VALIDATION.md", "VALIDATION_CN.md"))
HEADINGS = {
    "README.md": ["Purpose", "Steps", "Results"],
    "README_CN.md": ["目的", "操作步骤", "实现效果"],
    "VALIDATION.md": ["Architecture and Metrics", "What the Scripts Run", "Measured Runs",
                      "Tests and Offline Checks", "Limits, Assets and Sources"],
    "VALIDATION_CN.md": ["架构与指标口径", "脚本实际执行了什么", "实测记录", "测试与离线校验", "边界、目录与资料"],
}
RUN_PARTS = {"en": ["**Question.**", "**Input.**", "**Varied and fixed.**", "**Result.**", "**Boundary.**"],
             "zh": ["**问题。**", "**输入。**", "**变量与固定项。**", "**结果。**", "**边界。**"]}
RUN_ID_PATTERN = r"[a-c]-\d{4}-\d{2}-\d{2}(?:-[a-z]+)*"
RUN_SECTION_RE = re.compile(rf"(?ms)^### ({RUN_ID_PATTERN})[:：].*?(?=^### |^## |\Z)")


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def strip_code(text: str) -> str:
    return re.sub(r"(?ms)^```.*?^```", "", text)


def run_sections(text: str) -> dict[str, str]:
    return {match.group(1): match.group(0) for match in RUN_SECTION_RE.finditer(strip_code(text))}


def check_sections(pages: dict[str, str]) -> list[str]:
    return [f"{name}: headings differ" for name, expected in HEADINGS.items()
            if re.findall(r"(?m)^## (.+?)\s*$", strip_code(pages[name])) != expected]


def check_public_scope(pages: dict[str, str]) -> list[str]:
    errors = []
    claims = json.loads(read("evidence/claims.json"))
    if claims["publication"]["edition"] != "public" or claims["publication"]["paths"] != ["A", "B", "C"]:
        errors.append("claims: public publication scope must be A/B/C")
    if set(claims["paths"]) != {"A", "B", "C"} or set(claims["letter_mapping"]) != {"A", "B", "C"}:
        errors.append("claims: unpublished paths present")
    index = json.loads(read("evidence/runs/INDEX.json"))
    for relative in index:
        run = json.loads(read(f"evidence/runs/{relative}"))
        if run["path"] not in ("A", "B", "C", "shared"):
            errors.append(f"{relative}: unpublished measurement path")
    metrics = json.loads(read("evidence/measurements.json"))["metrics"]
    if any(not key.startswith(("a.", "b.", "c.", "f.")) for key in metrics):
        errors.append("metrics: unpublished measurement namespace")
    for name, text in pages.items():
        if "<!--v:d." in text or re.search(r"(?m)^### d-\d{4}", text):
            errors.append(f"{name}: unpublished metric or measured-run section")
    return errors


def check_information_alignment(pages: dict[str, str]) -> list[str]:
    errors = []
    claims = json.loads(read("evidence/claims.json"))
    names = " | ".join(f"{letter} · {claims['letter_mapping'][letter]['name']}" for letter in ("A", "B", "C"))
    for name, lang, label in (("README.md", "en", "Question"), ("README_CN.md", "zh", "问题")):
        if f"| {label} | {names} |" not in pages[name]:
            errors.append(f"{name}: canonical path table differs")
        labels = (("status", "Status", "状态"), ("lifecycle_owner", "Who runs the session lifecycle", "谁负责会话生命周期"),
                  ("session_location", "Where the session's micro-VM runs", "会话的 micro-VM 跑在哪里"),
                  ("memory_suspend", "Suspend with memory snapshot", "带内存快照的挂起"),
                  ("max_session", "Largest single session (docs)", "单个会话最大规格（文档）"))
        for field, en, zh in labels:
            values = " | ".join(claims["paths"][letter][f"{field}_{lang}"] for letter in ("A", "B", "C"))
            if f"| {en if lang == 'en' else zh} | {values} |" not in pages[name]:
                errors.append(f"{name}: {field} differs from canonical claims")
        comparison = "Observed or documented" if lang == "en" else "实测或文档项"
        if f"| {comparison} | {names} |" not in pages[name]:
            errors.append(f"{name}: comparison paths differ")
        if f"| | {names} |" not in pages[name]:
            errors.append(f"{name}: isolation paths differ")
    required = {
        "README.md": ["component sums, not a single end-to-end client stopwatch",
                      "stop and idle waiting are excluded", "private preview"],
        "README_CN.md": ["采用分段相加，不是客户端连续端到端计时", "不含停止和空闲等待", "私有预览"],
        "VALIDATION.md": ["component sums, not a single end-to-end client stopwatch",
                          "stop and idle waiting are excluded"],
        "VALIDATION_CN.md": ["采用分段相加，不是客户端连续端到端计时", "不含停止和空闲等待"],
    }
    for name, fragments in required.items():
        for fragment in fragments:
            if fragment not in pages[name]:
                errors.append(f"{name}: missing measurement/publication boundary")
    return errors


def check_run_sections(pages: dict[str, str]) -> list[str]:
    errors = []
    for name, lang in (("VALIDATION.md", "en"), ("VALIDATION_CN.md", "zh")):
        for run, section in run_sections(pages[name]).items():
            positions = [section.find(part) for part in RUN_PARTS[lang]]
            if -1 in positions or positions != sorted(positions):
                errors.append(f"{name} {run}: missing or reordered run parts")
            elif re.search(r"(?m)^\|", section[:positions[1]]):
                errors.append(f"{name} {run}: table before the input")
    return errors


def check_run_order(pages: dict[str, str]) -> list[str]:
    order = json.loads(read("evidence/claims.json"))["run_order"]
    index = json.loads(read("evidence/runs/INDEX.json"))
    measured = {json.loads(read(f"evidence/runs/{path}"))["run_id"] for path in index
                if not path.startswith("abc-")}
    errors = []
    if len(order) != len(measured) or set(order) != measured:
        errors.append("claims: every measured run must appear exactly once")
    if order != sorted(order, key=lambda run: (run[0], run[2:12])):
        errors.append("claims: run order must group by path then ascending date")
    for name, text in pages.items():
        actual = re.findall(rf"(?m)^\| `({RUN_ID_PATTERN})` \|", text) if name.startswith("README") else list(run_sections(text))
        if actual != order:
            errors.append(f"{name}: measured run order differs from canonical claims")
    return errors


def numbers(text: str) -> Counter:
    body = re.sub(r"<!--.*?-->|<img[^>]*>", "", text, flags=re.S)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    body = re.sub(r"`[^`\n]*`", "", strip_code(body))
    return Counter(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", body))


def check_numbers(pages: dict[str, str]) -> list[str]:
    errors = []
    for en, zh in PAIRS:
        if numbers(pages[en]) != numbers(pages[zh]):
            errors.append(f"{en}/{zh}: whole-page numbers differ")
        left = {match.group(2): match.group(3) for match in rd.BLOCK_RE.finditer(pages[en])}
        right = {match.group(2): match.group(3) for match in rd.BLOCK_RE.finditer(pages[zh])}
        if left.keys() != right.keys():
            errors.append(f"{en}/{zh}: generated blocks differ")
        for block in left.keys() & right.keys():
            if numbers(left[block]) != numbers(right[block]):
                errors.append(f"{en}/{zh}: block {block} numbers differ")
            cells = lambda text: [[numbers(cell) for cell in line.split("|")[1:-1]]
                                  for line in text.splitlines() if line.startswith("|")]
            if cells(left[block]) != cells(right[block]):
                errors.append(f"{en}/{zh}: block {block} numeric cells differ")
    return errors


def check_links_images(pages: dict[str, str]) -> list[str]:
    errors = []
    for name, text in pages.items():
        targets = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", strip_code(text))
        targets += re.findall(r'<img[^>]*src="([^"]+)"', text)
        for target in targets:
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            path = (ROOT / target.split("#", 1)[0]).resolve()
            if not path.exists() or not path.is_relative_to(ROOT):
                errors.append(f"{name}: broken or out-of-project link {target}")
        for image in re.findall(r"<img[^>]*>", text):
            if 'width="900"' not in image or not re.search(r'alt=".{20,}"', image):
                errors.append(f"{name}: image must have descriptive alt and width 900")
    return errors


def check_commands(pages: dict[str, str]) -> list[str]:
    blocks = [re.findall(r"(?ms)^```bash\n(.*?)^```", pages[name]) for name in ("README.md", "README_CN.md")]
    normalize = lambda text: re.sub(r"#.*", "", text).strip()
    return [] if [normalize(block) for block in blocks[0]] == [normalize(block) for block in blocks[1]] \
        else ["README command blocks differ between languages"]


def check_private(_pages: dict[str, str], files: list[Path] | None = None) -> list[str]:
    errors = []
    paths = files if files is not None else list(ROOT.rglob("*"))
    for path in paths:
        if not path.is_file() or path.name == ".env":
            continue
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        if relative.startswith(("images/icons/", "evidence-local/", ".pytest_cache/")) or "__pycache__" in relative:
            continue
        if path.name in ("test_guards.py", "validate_docs.py") or path.suffix not in (
            ".md", ".py", ".json", ".svg", ".example", ".txt", ".yml"
        ):
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r'href="data:[^"]*"', "", text)
        patterns = (r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b",
                    r"\b(?:\d{1,3}\.){3}\d{1,3}\b", r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b",
                    r"(?:eyJ[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AccountKey=)")
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(0)
                if value in ("127.0.0.1", "00000000-0000-0000-0000-000000000000") or \
                        value.startswith("10.") and value.endswith(".0") or value.endswith("@example.com"):
                    continue
                errors.append(f"{relative}: private identifier or credential")
    return errors


RULES = {"sections": check_sections, "public-scope": check_public_scope, "information-alignment": check_information_alignment,
         "run-sections": check_run_sections, "run-order": check_run_order, "numbers": check_numbers,
         "links-images": check_links_images, "commands": check_commands, "private": check_private}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    pages = {name: read(name) for name in PAGES}
    failures = 0
    for name, check in RULES.items():
        errors = check(pages)
        print(f"CHECK {name} {'FAIL' if errors else 'PASS'}")
        for error in errors:
            print(f"    {error}")
        failures += bool(errors)
    print(f"VALIDATE {'FAIL' if failures else 'PASS'} {len(RULES) - failures}/{len(RULES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
