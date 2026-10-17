#!/usr/bin/env python3
"""Public-content and structure audit for this repository.

    python tools/check_repo.py

Fails (exit 1) on: a broken relative link, image or in-page anchor; a `##`
heading out of reader order; an undocumented top-level path; a `<details>`
block; English and Chinese generated blocks that carry different numbers; or
content that must not be published (private paths, hosts, addresses, names,
and comparisons with other accelerators).
"""
from __future__ import annotations

import os
import re
import sys
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = {"en": ROOT / "README.md", "cn": ROOT / "README_CN.md"}
VALIDATION_NAMES = {"en": "VALIDATION.md", "cn": "VALIDATION_CN.md"}
READER_ORDER = {
    "en": ["Purpose", "Steps", "Results"],
    "cn": ["目的", "操作步骤", "实现效果"],
}
VALIDATION_ORDER = {
    "en": ["Architecture and Metrics", "Commands and Configuration", "Query and Log Details",
           "Measurements on MI300X", "Tests and Offline Checks", "Limits, Assets and Sources"],
    "cn": ["架构与指标口径", "配置命令实际执行了什么", "查询与日志细节",
           "MI300X 上的实测验证", "测试与离线校验", "边界、目录与资料"],
}
MAX_TABLE_COLUMNS = 4  # five-column tables overflow a 390px viewport on GitHub
TOP_LEVEL_EXEMPT = {".gitattributes", ".gitignore", "README.md", "__pycache__"}
FORBIDDEN = [
    (re.compile(r"\b(?:H20|H100|H200|H800|A100|A800|L20|L40S?|B100|B200|GB200|GB300|MI250X?|MI325X|MI35\dX?|TPU|Gaudi\d?)\b"), "named comparison accelerator"),
    (re.compile(r"exp_stats|swe_flash", re.I), "customer evaluation material"),
    (re.compile(r"(?i)swe-?bench"), "agentic benchmark run that used customer material"),
    (re.compile(r"/data/models/(?!MiMo-V2\.5-Pro\b)", re.I), "private model path"),
    (re.compile(r"(?i)\b[A-Z]:\\Users\\|/home/[\w.-]+/|/Users/[\w.-]+/|\\\\[\w.-]+\\"), "user or network path"),
    (re.compile(r"\b[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[\w.-]+\b"), "e-mail address"),
    (re.compile(r"\b(?!0\.0\.0\.0\b)(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"), "IPv4 address"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "GUID (subscription or tenant)"),
    (re.compile(r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"), "credential-looking value"),
    (re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"), "private key material"),
    (re.compile(r"<details", re.I), "collapsed block"),
    (re.compile(r"\b20\d\d-\d\d-\d\d\b|\b20\d\d(?:0[1-9]|1[0-2])(?:[0-2]\d|3[01])\b|\b(?:January|February|March|April|June|July|August|September|October|November|December)\b|\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"),
     "calendar date that dates the underlying project"),
]
PRIVATE_FOLDED_HASHES = {
    "c608c4787821de1b383d87ca8b2711d3d171b5a6c017c45a8523e9ae904d495e",
    "522d3edeb64a7ca45f5ea6757140aa4e8d95ab73179fe7d290c347e4662ecb74",
    "0c0820afcf101e4cae96d75f63a513ebe186ff6487fe7a896519f477859ed99c",
    "3b4e330eb1c006261b9bdf9cf1f1abd49edc7ca9cb77bfa74eadf31a3a4a0014",
    "647aa77d5912b717fe36e5f809fa8a260850cc150b8c202b9d863ced85d81d23",
    "be94e4b7d4fe080bdfc76f717a2286f237dfb158a857cbd383b44dfe42831e51",
}
PRIVATE_CASED_HASHES = {
    "e0c88e1585b2f70e083ce61dcb1c5a1ebc4e30a2d062e1970db9def0938eb7d0",
    "fb338e53da57ec89c638f24e8f87697a70d0c0d8e4cc3a56f754211d2c6f0444",
    "e85f6b78597ef3b66102610dc4f58094186cb6cd83cf0a729019b58bf660f18a",
    "0c438bb629fafee2031ef90f09e139726894f4301a346424691a73fdbb4aba79",
    "026442b0dce6753aff66787cc664b9b81325c4163ebcea410f5f43bcbc22cf5a",
    "edadc1e65a0def66be3236619588886eaa1c6d2614fc2451a2ff396ef399af61",
    "3d19227ed918d2b0862a2990dc3ce08fdd301a5f89361d82d8b8ebf1864c88cb",
}
SOFTWARE_DECLARATION = re.compile(
    r'(?m)(^ROCm version\s*:\s*|^\s*"ROCm"\s*:\s*")\d+\.\d+\.\d+\.\d+-\d+-[0-9a-f]+'
)
SCAN_SUFFIXES = {".md", ".py", ".json", ".txt", ".sh", ".yml", ".yaml", ".log", ".example", ""}
SCAN_SKIP_DIRS = {"upstream", "__pycache__"}
SCAN_SKIP_FILES = {"check_repo.py", "test_public_content.py", "test_bench_log.py", "test_rccl.py"}  # synthetic guard probes
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)|<img\s+[^>]*src=\"([^\"]+)\"")
GENERATED = re.compile(r"<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n(?P<body>.*?)<!-- END GENERATED: (?P=name) -->", re.S)
LINK_TARGET = re.compile(r"\]\([^)]*\)")
NUMBER = re.compile(r"(?<![A-Za-z_\d])[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def slug(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def headings(text: str) -> list[tuple[int, str]]:
    out, fence = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
            continue
        m = re.match(r"^(#{1,6}) (.+)$", line)
        if m and not fence:
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


def check_links(lang: str, text: str, errors: list[str]) -> None:
    anchors = {slug(h) for _, h in headings(text)}
    for m in LINK.finditer(text):
        target = next(g for g in m.groups() if g)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        if not path:
            if anchor not in anchors:
                errors.append(f"{lang}: BROKEN_ANCHOR #{anchor}")
            continue
        resolved = (ROOT / path).resolve()
        outside = ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve()
        if outside and os.environ.get("CHECK_MONOREPO_LINKS") != "1":
            continue  # sibling projects may be absent in a sparse checkout; CI sets CHECK_MONOREPO_LINKS=1
        if not resolved.exists():
            errors.append(f"{lang}: BROKEN_LOCAL_LINK {target}")
        elif anchor and resolved.is_file() and resolved.suffix == ".md":
            target_anchors = {slug(h) for _, h in headings(resolved.read_text(encoding="utf-8"))}
            if anchor not in target_anchors:
                errors.append(f"{lang}: BROKEN_FILE_ANCHOR {target}")


def check_order(lang: str, text: str, errors: list[str], document: str = "readme") -> None:
    got = [h for level, h in headings(text) if level == 2]
    expected = READER_ORDER if document == "readme" else VALIDATION_ORDER
    if got != expected[lang]:
        errors.append(f"{lang}: HEADING_ORDER {got}")


def _rows(body: str) -> list[list[str]]:
    """Numbers per table row or list item; code blocks compare as whole text."""
    visible = [LINK_TARGET.sub("]", l) for l in body.splitlines()]  # anchors and URLs are not reader-visible numbers
    lines = [l for l in visible if l.startswith("|")]
    if lines:
        return [NUMBER.findall(l) for l in lines[2:]]
    items = [l for l in visible if l.startswith("- ")]
    if items:
        return [NUMBER.findall(l) for l in items]
    return [[body.split("```", 1)[1] if "```" in body else ""]]


def check_bilingual(errors: list[str]) -> None:
    documents = [READMES, {lang: ROOT / name for lang, name in VALIDATION_NAMES.items()}]
    for paths in documents:
        blocks = {lang: {m.group("name"): m.group("body") for m in GENERATED.finditer(p.read_text(encoding="utf-8"))}
                  for lang, p in paths.items()}
        if set(blocks["en"]) != set(blocks["cn"]):
            errors.append(f"BILINGUAL_BLOCK_SET {sorted(set(blocks['en']) ^ set(blocks['cn']))}")
        for name in sorted(set(blocks["en"]) & set(blocks["cn"])):
            en, cn = _rows(blocks["en"][name]), _rows(blocks["cn"][name])
            if len(en) != len(cn):
                errors.append(f"BILINGUAL_ROWS {name}: {len(en)} vs {len(cn)} rows")
                continue
            for i, (a, b) in enumerate(zip(en, cn)):
                if a != b:
                    errors.append(f"BILINGUAL_NUMBERS {name} row {i + 1}: en {a} cn {b}")
            if name in {"rccl-headline", "rccl-runs", "rccl-observation", "run-index", "run-inputs"}:
                numbers = {lang: Counter(NUMBER.findall(LINK_TARGET.sub("]", blocks[lang][name])))
                           for lang in ("en", "cn")}
                if numbers["en"] != numbers["cn"]:
                    errors.append(f"BILINGUAL_PROSE_NUMBERS {name}")


def check_operator_path(lang: str, text: str, errors: list[str]) -> None:
    heading = "Steps" if lang == "en" else "操作步骤"
    section = text.split("## " + heading, 1)[1].split("\n## ", 1)[0] if "## " + heading in text else ""
    first_fence = section.find("```")
    first_table = section.find("\n|")
    if first_table < 0 or (first_fence >= 0 and first_table > first_fence):
        errors.append(f"{lang}: MISSING_DEPLOYMENT_ORDER")
    for entry in ("docker buildx build", "render_launch.py", "run-role.sh", "wait-ready.sh",
                  "v1/chat/completions", "run-rccl.sh", "rccl.Dockerfile"):
        if entry not in section:
            errors.append(f"{lang}: MISSING_OPERATOR_STEP {entry}")
    if "unittest" in section or "build_evidence.py --check" in section:
        errors.append(f"{lang}: OFFLINE_REPLAY_IN_OPERATOR_PATH")
    for old in ("## Start Here", "## Reproduce in Your Environment", "## 从这里开始", "## 客户如何复现"):
        if old in text:
            errors.append(f"{lang}: RETIRED_OPERATOR_HEADING")
    if VALIDATION_NAMES[lang] not in text:
        errors.append(f"{lang}: MISSING_REVIEWER_ENTRY")
    for image in re.finditer(r'<img\b[^>]*src="([^"]+)"[^>]*>', text):
        width = re.search(r'\bwidth="(\d+)"', image[0])
        if not width or int(width[1]) > 900:
            errors.append(f"{lang}: UNBOUNDED_DIAGRAM {image[1]}")


def check_table_shape(lang: str, text: str, errors: list[str]) -> None:
    rows, fence = [], False
    for line in text.splitlines() + [""]:
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("|"):
            rows.append(line)
            continue
        if rows:
            widths = {len(re.split(r"(?<!\\)\|", r)) - 2 for r in rows}
            if len(widths) != 1:
                errors.append(f"{lang}: TABLE_SHAPE {rows[0][:60]!r}")
            elif next(iter(widths)) > MAX_TABLE_COLUMNS:
                errors.append(f"{lang}: TABLE_TOO_WIDE {next(iter(widths))} columns: {rows[0][:60]!r}")
            rows = []


def check_top_level(errors: list[str]) -> None:
    text = "".join(path.read_text(encoding="utf-8") for path in READMES.values())
    text += "".join((ROOT / name).read_text(encoding="utf-8") for name in VALIDATION_NAMES.values())
    for entry in ROOT.iterdir():
        if entry.name in TOP_LEVEL_EXEMPT:
            continue
        ref = entry.name + ("/" if entry.is_dir() else "")
        if f"({ref})" not in text and f"({entry.name})" not in text:
            errors.append(f"UNDOCUMENTED_TOP_LEVEL {ref}")


def private_name_present(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    for token in tokens:
        exact = hashlib.sha256(token.encode("utf-8")).hexdigest()
        folded = hashlib.sha256(token.casefold().encode("utf-8")).hexdigest()
        if exact in PRIVATE_CASED_HASHES or folded in PRIVATE_FOLDED_HASHES:
            return True
    return False


def check_forbidden(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES or path.name in SCAN_SKIP_FILES:
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in SCAN_SKIP_DIRS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if private_name_present(text):
            errors.append(f"FORBIDDEN {rel.as_posix()}: private-name digest matched")
        scan_text = SOFTWARE_DECLARATION.sub(r"\1<software-version>", text)
        for pattern, why in FORBIDDEN:
            m = pattern.search(scan_text)
            if m:
                errors.append(f"FORBIDDEN {rel.as_posix()}: {why}")


def run() -> list[str]:
    errors: list[str] = []
    for lang, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        check_links(lang, text, errors)
        check_order(lang, text, errors)
        check_table_shape(lang, text, errors)
        check_operator_path(lang, text, errors)
        reviewer = ROOT / VALIDATION_NAMES[lang]
        if not reviewer.is_file():
            errors.append(f"{lang}: MISSING_VALIDATION")
            continue
        review_text = reviewer.read_text(encoding="utf-8")
        check_links(reviewer.name, review_text, errors)
        check_order(lang, review_text, errors, "validation")
        check_table_shape(reviewer.name, review_text, errors)
    check_bilingual(errors)
    check_top_level(errors)
    check_forbidden(errors)
    return errors


def main() -> int:
    errors = run()
    if errors:
        print("\n".join(errors))
        return 1
    print("PASS operator/reviewer structure, execution paths, links, bilingual numbers and public-content guards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
