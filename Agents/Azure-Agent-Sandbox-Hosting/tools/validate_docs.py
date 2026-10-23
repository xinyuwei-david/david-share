#!/usr/bin/env python3
"""Check the four pages a reader opens: structure, runs, links, images, bilingual numbers and private content.

    python tools/validate_docs.py            # prints one CHECK line per rule, exit 0 when all PASS

Rules (each prints CHECK <rule> PASS|FAIL <detail>):
  readme-sections     README.md / README_CN.md have exactly the three ## headings, in order
  validation-sections VALIDATION.md / VALIDATION_CN.md have their five ## headings, in order
  run-sections        every measured-run ### section has Question, Input, Varied and fixed, Result, Boundary in
                      that order, and no table before its Input
  runs-listed         every run_id in evidence/runs/ (except the support files) is a ### section in both
                      VALIDATION pages and a row of both README results tables
  run-order           all four pages follow the canonical A-D, ascending-date run order in evidence/claims.json
  links               every relative link and image resolves to a file in the repository
  images              every image is used in both languages, has alt text, and diagrams are embedded at <= 900 px
  numbers             the multiset of numbers in each README equals its Chinese page's (same for VALIDATION)
  information-alignment core identities, statuses, boundaries, recommendations and evidence limits agree
  commands            Steps commands use variables, not subscription IDs, GUIDs or literal resource IDs; they
                      install into a virtual environment, never close the reader's shell (set -e, exit), and
                      keep az group delete alone in its block with its confirmation prompt
  shell-syntax        every bash block in both READMEs parses with bash -n
  private             no GUID, IPv4 address, email, token, internal name or retired claim in the delivered tree
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import render_docs as rd

ROOT = Path(__file__).resolve().parents[1]

README_HEADINGS = {"README.md": ["Purpose", "Results", "Steps"], "README_CN.md": ["目的", "实现效果", "操作步骤"]}
VALIDATION_HEADINGS = {
    "VALIDATION.md": ["Architecture and Metrics", "What the Scripts Run", "Measured Runs", "Tests and Offline Checks",
                      "Limits, Assets and Sources"],
    "VALIDATION_CN.md": ["架构与指标口径", "脚本实际执行了什么", "实测记录", "测试与离线校验", "边界、目录与资料"],
}
RUN_PARTS = {"en": ["**Question.**", "**Input.**", "**Varied and fixed.**", "**Result.**", "**Boundary.**"],
             "zh": ["**问题。**", "**输入。**", "**变量与固定项。**", "**结果。**", "**边界。**"]}
RUN_ID_PATTERN = r"[a-e]-\d{4}-\d{2}-\d{2}(?:-[a-z]+)*"
RUN_SECTION_RE = re.compile(rf"(?ms)^### ({RUN_ID_PATTERN})[:：].*?(?=^### |^## |\Z)")
SUPPORT_RUNS = {"abc-2026-09-27-foundation", "d-2026-09-29-cold-early", "d-2026-09-29-model-direct",
                "d-2026-09-29-capacity", "d-2026-09-22-field"}
PAIRS = [("README.md", "README_CN.md"), ("VALIDATION.md", "VALIDATION_CN.md")]

# Loopback, and the fixed link-local address of the Azure instance metadata service (the same on every Azure VM;
# measure_e2b.py reads the VM's managed identity token from it).
ALLOWED_IPV4 = {"127.0.0.1", "169.254.169.254"}
# Fixed home directory of the default user inside an E2B sandbox, not a machine of the author.
ALLOWED_LOCAL_PATHS = {"/home/user/"}
# The ACR token username the SDK expects; documented as a fixed value, not an identity.
ALLOWED_GUIDS = {"00000000-0000-0000-0000-000000000000"}
PRIVATE_PATTERNS = {
    "guid": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b",
    "token": r"(?:eyJ[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AccountKey=)",
    "local-path": r"(?i)(?:[a-z]:\\\\?users\\|/home/[a-z]+/|onedrive - microsoft)",
}
# Names from the author's environment; stored as plain fragments because they are not secrets but must not ship.
INTERNAL_NAMES = ["7d645c", "5mliuz7m6nn64", "bfe2665ff6", "33546699", "byoaks-v3", "aks-byoaks", "rg-byoaks",
                  "MCAPS", "XINYU-SUR", "adcproxy.io", "caresp_", "a6186e7a", "kylerc", "bowen", "reply-to-ken",
                  "e2b-embed-vm"]
# Matched with case: the author's E2B VMs are lowercase e2b-vm*, and config/.env.example has YOUR-E2B-VM-NAME.
INTERNAL_NAMES_CASED = ["e2b-vm"]
RETIRED_CLAIMS = ["not observed on the cluster", "Sessions on cluster-not observed", "NOT_ON_CLUSTER",
                  "24–26 s", "11–16 s", "Managed, per agent",
                  "same sample agent on all four", "同一个示例 agent 部署到四条路径",
                  "The sample agent is the same everywhere", "示例 agent 在各路径上相同",
                  "Foundry installs 13 runtime pods", "Foundry 在你集群的 `foundry-system` 里装 13 个运行时 pod",
                  "13 runtime pods, installed and patched by Foundry", "13 个运行时 pod，由 Foundry 安装和升级",
                  "no idle timeout observed in 90 min", "Pod lost → ready again, state kept",
                  "pod 丢失 → 重新就绪，状态保留", "Setup in an existing foundation",
                  "在已有基础资源上的部署耗时",
                  "A, B and C use the workshop reflection agent",
                  "A, B and C use the workshop's reflection agent",
                  "A, B and C use the agenthost workshop's reflection agent",
                  "A、B、C 使用 workshop reflection agent",
                  "A、B、C 使用 workshop 的 reflection agent",
                  "A、B、C 用 agenthost workshop 的 reflection agent",
                  # B reached GA in 2026-06 (Azure Update 563596); the preview wording must not return.
                  "and a public preview is acceptable", "且能接受公开预览",
                  "Hosted agents are in public preview today", "no later GA announcement was found",
                  "此后未找到 GA 公告", "B public preview", "B 的公开预览说明"]
SCAN_SUFFIXES = {".md", ".json", ".py", ".yml", ".yaml", ".txt", ".example", ".svg", ".in"}
SKIP_SCAN = {"tests/test_guards.py", "tools/validate_docs.py"}


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def h2(text: str) -> list[str]:
    return re.findall(r"(?m)^## (.+?)\s*$", strip_code(text))


def strip_code(text: str) -> str:
    return re.sub(r"(?ms)^```.*?^```", "", text)


def check_readme_sections(pages: dict[str, str]) -> list[str]:
    return [f"{n}: headings {h2(pages[n])} != {want}" for n, want in README_HEADINGS.items() if h2(pages[n]) != want]


def check_four_path_tables(pages: dict[str, str]) -> list[str]:
    expected = {
        "README.md": "| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS | E · E2B on Azure VM |",
        "README_CN.md": "| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS | E · E2B on Azure VM |",
    }
    errors = []
    for name, header in expected.items():
        if header not in pages[name]:
            errors.append(f"{name}: isolation table must contain paths A, B, C and D (and E) in order")
    return errors


def check_information_alignment(pages: dict[str, str]) -> list[str]:
    claims = json.loads(read("evidence/claims.json"))
    errors = []
    letters = ("A", "B", "C", "D", "E")
    en_names = " | ".join(f"{letter} · {claims['letter_mapping'][letter]['name']}" for letter in letters)
    header = f"| Question | {en_names} |"
    if header not in pages["README.md"]:
        errors.append("README.md: Purpose table names differ from evidence/claims.json")
    for name, label in (("README.md", "Observed or documented"), ("README_CN.md", "实测或文档项")):
        block = next((m.group(3) for m in rd.BLOCK_RE.finditer(pages[name]) if m.group(2) == "comparison"), "")
        if f"| {label} | {en_names} |" not in rd.unwrap_table_layout(block):
            errors.append(f"{name}: Results comparison names differ from evidence/claims.json")
    status_en = " | ".join(claims["paths"][letter]["status_en"] for letter in letters)
    status_zh = " | ".join(claims["paths"][letter]["status_zh"] for letter in letters)
    if f"| Status | {status_en} |" not in pages["README.md"]:
        errors.append("README.md: maturity row differs from evidence/claims.json")
    if f"| 状态 | {status_zh} |" not in pages["README_CN.md"]:
        errors.append("README_CN.md: maturity row differs from evidence/claims.json")
    for field, label_en, label_zh in (
        ("lifecycle_owner", "Who runs the session lifecycle", "谁负责会话生命周期"),
        ("session_location", "Where the session's micro-VM runs", "会话的 micro-VM 跑在哪里"),
        ("memory_suspend", "Suspend with memory snapshot", "带内存快照的挂起"),
        ("state_storage", "External storage to keep data across suspend and resume",
         "挂起再恢复后保留数据，是否必须挂外置存储"),
        ("selectable_compute", "Selectable size / node SKU (docs)", "可选规格 / 节点 SKU（文档）"),
    ):
        values_en = " | ".join(claims["paths"][letter][f"{field}_en"] for letter in letters)
        values_zh = " | ".join(claims["paths"][letter][f"{field}_zh"] for letter in letters)
        if f"| {label_en} | {values_en} |" not in pages["README.md"]:
            errors.append(f"README.md: {field} row differs from evidence/claims.json")
        if f"| {label_zh} | {values_zh} |" not in pages["README_CN.md"]:
            errors.append(f"README_CN.md: {field} row differs from evidence/claims.json")
    boundary_en = " | ".join(claims["paths"][letter]["boundary_en"] for letter in letters)
    boundary_zh = " | ".join(claims["paths"][letter]["boundary_zh"] for letter in letters)
    owner_en = " | ".join(claims["paths"][letter]["node_owner_en"] for letter in letters)
    owner_zh = " | ".join(claims["paths"][letter]["node_owner_zh"] for letter in letters)
    for name, expected, label in (
        ("README.md", f"| Boundary | {boundary_en} |", "boundary"),
        ("README_CN.md", f"| 隔离边界 | {boundary_zh} |", "boundary"),
        ("README.md", f"| Who owns the node | {owner_en} |", "node owner"),
        ("README_CN.md", f"| 节点归谁 | {owner_zh} |", "node owner"),
    ):
        if expected not in pages[name]:
            errors.append(f"{name}: {label} row differs from evidence/claims.json")
    required = {
        "README.md": [
            "Start with B for a general managed enterprise agent when its limits fit.",
            "For a long-running, per-user workload",
            "A = ACA Sandboxes, B = Foundry hosted agent, C = AKS + agent-sandbox, D = BYO-AKS",
            "not the same container",
            "leaves out answer times",
            "stop and idle waiting are excluded",
        ],
        "README_CN.md": [
            "一般企业 agent 在 B 的限制可接受时先选 B。",
            "对于需要按用户长期运行",
            "A = ACA Sandboxes，B = Foundry hosted agent，C = AKS + agent-sandbox，D = BYO-AKS",
            "不是相同容器",
            "不比较回答时间",
            "不含停止和空闲等待",
        ],
        "VALIDATION.md": [
            "The workload contract is comparable, but the container is not identical",
            "9 fixed Deployment replicas",
            "the total grows with the node pool",
            "does **not** establish continuity of the deleted session",
            "component sums, not a single end-to-end client stopwatch",
            "does not show that code ran in the sandbox",
            "stop and idle waiting are excluded",
            "pod age, not idle duration",
        ],
        "VALIDATION_CN.md": [
            "路径的工作负载口径可比，但容器并不完全相同",
            "9 个固定的 Deployment 副本",
            "总数随节点池增长",
            "不能**证明被删会话自身的连续性",
            "采用分段相加，不是客户端连续端到端计时",
            "不能说明代码在沙箱里执行",
            "不含停止和空闲等待",
            "pod 的年龄，不是空闲时长",
        ],
    }
    for name, fragments in required.items():
        for fragment in fragments:
            if fragment not in pages[name]:
                errors.append(f"{name}: missing aligned narrative {fragment!r}")
    for name in ("README.md", "README_CN.md", "VALIDATION.md", "VALIDATION_CN.md"):
        if "ResponsesHostServer" not in pages[name]:
            errors.append(f"{name}: B's distinct workload implementation is missing")
    for letter in ("A", "B", "C"):
        mapping = claims["letter_mapping"][letter]
        en_fragment = f"{mapping['upstream_module']} Solution {mapping['upstream_solution_letter']}"
        zh_fragment = f"{mapping['upstream_module']} 叫 Solution {mapping['upstream_solution_letter']}"
        if en_fragment not in pages["README.md"]:
            errors.append(f"README.md: upstream letter mapping for path {letter} differs from evidence/claims.json")
        if zh_fragment not in pages["README_CN.md"]:
            errors.append(f"README_CN.md: upstream letter mapping for path {letter} differs from evidence/claims.json")
    scope = claims["measurement_scope"]
    badge_fragment = (f"measured-{scope['region'].replace(' ', '%20')}%2C%20"
                      f"{scope['first_date'].replace('-', '--')}%20to%20"
                      f"{scope['last_date'][5:].replace('-', '--')}")
    for name in ("README.md", "README_CN.md"):
        if badge_fragment not in pages[name]:
            errors.append(f"{name}: measured badge differs from evidence/claims.json")
    delivered = "\n".join(pages.values()) + "\n" + read("tools/build_diagrams.py") + "\n" + \
        read("evidence/runs/d-byoaks-2026-09-29/bench.json") + "\n" + read("evidence/measurements.json")
    for retired in RETIRED_CLAIMS:
        if retired in delivered:
            errors.append(f"delivered narrative contains retired claim {retired!r}")
    metrics = json.loads(read("evidence/measurements.json"))["metrics"]
    if "d.pod_lost_followup" in metrics or "d.other_session_followup" not in metrics:
        errors.append("evidence/measurements.json: D recovery metric does not preserve the different-session boundary")
    topology = claims["d_runtime_topology"]
    if topology["total_with_one_matching_node"] != \
            topology["fixed_deployment_pods"] + topology["daemonsets"]:
        errors.append("evidence/claims.json: D one-node runtime total does not add up")
    if topology["total_with_one_matching_node"] != metrics["d.initial_runtime_pods"]["value"]:
        errors.append("evidence/claims.json: D one-node runtime total differs from the measured read-back")
    return errors


def check_validation_sections(pages: dict[str, str]) -> list[str]:
    return [f"{n}: headings {h2(pages[n])} != {want}" for n, want in VALIDATION_HEADINGS.items()
            if h2(pages[n]) != want]


def run_sections(text: str) -> dict[str, str]:
    body = strip_code(text)
    out = {}
    for mo in RUN_SECTION_RE.finditer(body):
        out[mo.group(1)] = mo.group(0)
    return out


def check_run_sections(pages: dict[str, str]) -> list[str]:
    errors = []
    for name, lang in (("VALIDATION.md", "en"), ("VALIDATION_CN.md", "zh")):
        for run, sec in run_sections(pages[name]).items():
            pos = [sec.find(p) for p in RUN_PARTS[lang]]
            if -1 in pos:
                errors.append(f"{name} {run}: missing {RUN_PARTS[lang][pos.index(-1)]}")
                continue
            if pos != sorted(pos):
                errors.append(f"{name} {run}: parts out of order")
            if re.search(r"(?m)^\|", sec[:pos[1]]):
                errors.append(f"{name} {run}: a table appears before the input")
            gen = re.search(r"<!-- BEGIN GENERATED: ([a-z-]+) -->", sec[pos[1]:pos[2]])
            if not gen and not re.search(r"\"[^\"]{8,}\"|“[^”]{8,}”|`[^`]{6,}`", sec[pos[1]:pos[2]]):
                errors.append(f"{name} {run}: the input part quotes no real input")
    return errors


def run_ids() -> list[str]:
    ids = []
    for p in sorted((ROOT / "evidence" / "runs").rglob("*.json")):
        if p.name == "INDEX.json":
            continue
        ids.append(json.loads(p.read_text(encoding="utf-8"))["run_id"])
    return ids


def check_runs_listed(pages: dict[str, str]) -> list[str]:
    errors = []
    for rid in run_ids():
        if rid in SUPPORT_RUNS:
            continue
        for name in ("VALIDATION.md", "VALIDATION_CN.md"):
            if rid not in run_sections(pages[name]):
                errors.append(f"{name}: no ### section for run {rid}")
        for name in ("README.md", "README_CN.md"):
            if f"| `{rid}` |" not in pages[name]:
                errors.append(f"{name}: run {rid} missing from the results table")
    return errors


def check_run_order(pages: dict[str, str]) -> list[str]:
    order = json.loads(read("evidence/claims.json")).get("run_order")
    if not isinstance(order, list) or not all(isinstance(rid, str) for rid in order):
        return ["evidence/claims.json: run_order must be a list of run IDs"]
    expected = set(run_ids()) - SUPPORT_RUNS
    if len(order) != len(expected) or set(order) != expected:
        return ["evidence/claims.json: run_order must list every measured run exactly once"]
    errors = []
    keys = [(rid[0], rid[2:12]) for rid in order]
    if keys != sorted(keys):
        errors.append("evidence/claims.json: run_order must group by path then ascending date")
    for name in ("README.md", "README_CN.md", "VALIDATION.md", "VALIDATION_CN.md"):
        body = strip_code(pages[name])
        if name.startswith("README"):
            actual = re.findall(rf"(?m)^\| `({RUN_ID_PATTERN})` \|", body)
        else:
            actual = [mo.group(1) for mo in RUN_SECTION_RE.finditer(body)]
        if actual != order:
            errors.append(f"{name}: measured run order differs from evidence/claims.json run_order")
    return errors


LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
IMG_HTML_RE = re.compile(r"<img\s+([^>]*)>")


def html_attr(attrs: str, key: str) -> str | None:
    mo = re.search(rf'{key}="([^"]*)"', attrs)
    return mo.group(1) if mo else None


def images(text: str) -> list[dict]:
    out = [{"src": m.group(2), "alt": m.group(1), "width": None} for m in IMG_MD_RE.finditer(text)]
    for m in IMG_HTML_RE.finditer(text):
        w = html_attr(m.group(1), "width")
        out.append({"src": html_attr(m.group(1), "src"), "alt": html_attr(m.group(1), "alt") or "",
                    "width": int(w) if w and w.isdigit() else None})
    return out


def check_links(pages: dict[str, str]) -> list[str]:
    errors = []
    for name, text in pages.items():
        body = strip_code(text)
        targets = [m.group(1) for m in LINK_RE.finditer(body)] + [i["src"] for i in images(body)]
        for t in targets:
            if not t or re.match(r"^(https?:|mailto:|#)", t):
                continue
            path = (ROOT / t.split("#", 1)[0]).resolve()
            if not path.exists() or ROOT.resolve() not in path.parents and path != ROOT.resolve():
                errors.append(f"{name}: broken link {t}")
    return errors


def check_images(pages: dict[str, str]) -> list[str]:
    errors = []
    for en, zh in PAIRS:
        ie, iz = images(strip_code(pages[en])), images(strip_code(pages[zh]))
        norm = lambda lst: [re.sub(r"-(en|zh)\.", "-L.", i["src"]) for i in lst]  # noqa: E731
        if norm(ie) != norm(iz):
            errors.append(f"{en}/{zh}: images differ {norm(ie)} vs {norm(iz)}")
        for name, lst in ((en, ie), (zh, iz)):
            for i in lst:
                if i["src"].startswith("http"):
                    continue  # badges
                if len(i["alt"]) < 20:
                    errors.append(f"{name}: image {i['src']} has no descriptive alt text")
                if "architecture" in i["src"] and (i["width"] is None or i["width"] > 900):
                    errors.append(f"{name}: diagram {i['src']} must be <img width<=900>")
                if "-zh." in i["src"] and name == en or "-en." in i["src"] and name == zh:
                    errors.append(f"{name}: image {i['src']} is in the other language")
    sources = json.loads(read("images/SOURCES.json"))
    used = {i["src"] for n in pages for i in images(strip_code(pages[n]))}
    for fig in sources["figures"]:
        if fig["file"].endswith(".png") and fig["file"] not in used:
            errors.append(f"images/SOURCES.json: {fig['file']} is not used on any page")
    return errors


NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def numbers(text: str) -> Counter:
    body = re.sub(r"<!--.*?-->", "", rd.unwrap_table_layout(text), flags=re.S)
    body = re.sub(r"\(https?://[^)]*\)|https?://\S+", "", body)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    body = re.sub(r"<img[^>]*>", "", body)
    body = re.sub(r"`[^`\n]*`", "", body)
    body = re.sub(r"(?ms)^```.*?^```", "", body)
    return Counter(NUM_RE.findall(body))


def check_numbers(pages: dict[str, str]) -> list[str]:
    errors = []
    for en, zh in PAIRS:
        a, b = numbers(pages[en]), numbers(pages[zh])
        if a != b:
            only_en = sorted((a - b).elements())
            only_zh = sorted((b - a).elements())
            errors.append(f"{en} vs {zh}: numbers only in English {only_en[:12]}, only in Chinese {only_zh[:12]}")
        blocks_en = {mo.group(2): mo.group(3) for mo in rd.BLOCK_RE.finditer(pages[en])}
        blocks_zh = {mo.group(2): mo.group(3) for mo in rd.BLOCK_RE.finditer(pages[zh])}
        if blocks_en.keys() != blocks_zh.keys():
            errors.append(f"{en} vs {zh}: generated blocks differ")
        for name in blocks_en.keys() & blocks_zh.keys():
            left, right = blocks_en[name], blocks_zh[name]
            if numbers(left) != numbers(right):
                errors.append(f"{en} vs {zh}: numbers differ in generated block {name}")
            cells = lambda text: [[numbers(cell) for cell in line.split("|")[1:-1]]
                                  for line in text.splitlines() if line.startswith("|")]  # noqa: E731
            if cells(left) != cells(right):
                errors.append(f"{en} vs {zh}: numeric table cells differ in generated block {name}")
    return errors


def steps_section(text: str, heading: str) -> str:
    mo = re.search(rf"(?ms)^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text)
    return mo.group(1) if mo else ""


def check_commands(pages: dict[str, str]) -> list[str]:
    errors = []
    for name, heading in (("README.md", "Steps"), ("README_CN.md", "操作步骤")):
        blocks = re.findall(r"(?ms)^```bash\n(.*?)^```", steps_section(pages[name], heading))
        commands = "\n".join(blocks)
        for required in ("./setup.sh", "./prepare-agent-sandbox.sh", "kubectl apply -f agent-sandbox.yaml",
                         "./sandbox-deploy.sh", "python tools/aca_sandbox.py create", "azd ai agent init",
                         "azd provision", "azd deploy", "azd ai agent invoke", '"$C_AGENT_URL/reflect"',
                         "python tools/e2b_agent.py"):
            if required not in commands:
                errors.append(f"{name}: missing deployment or real invocation command {required}")
        if 'export HOSTING_REPO="$PWD"' not in commands:
            errors.append(f"{name}: missing stable repository return directory")
        if re.search(r"tools/(?:measure_aca|measure_responses|measure_e2b|check_isolation|build_measurements|render_docs|validate_docs)\.py", commands):
            errors.append(f"{name}: measurement or evidence check used in deployment Steps")
        if "C invocation failed: non-empty response missing" not in commands:
            errors.append(f"{name}: C invocation does not reject a missing application answer")
        for required in ("python3 -m venv .venv", ". .venv/bin/activate"):
            if required not in commands:
                errors.append(f"{name}: Steps must install Python packages into a virtual environment ({required})")
        # Readers paste these blocks into a shell they keep open; set -e or exit would close it on the first failure.
        if re.search(r"(?m)^\s*set\s+-[a-z]*e", commands) or re.search(r"(?m)^\s*exit\b", commands):
            errors.append(f"{name}: Steps commands must not close the reader's interactive shell (set -e or exit)")
        for code in blocks:
            if "az group delete" in code and code.strip().count("\n") > 0:
                errors.append(f"{name}: az group delete must be alone in its own code block")
            if re.search(r"az group delete[^\n]*(?:--yes|\s-y\b)", code):
                errors.append(f"{name}: az group delete must keep its confirmation prompt")
            for bad, why in ((r"/subscriptions/[0-9a-f-]{8}", "literal subscription ID"),
                             (PRIVATE_PATTERNS["guid"], "GUID"),
                             (r"-g\s+rg-[a-z0-9-]+|--resource-group\s+rg-", "literal resource group")):
                if re.search(bad, code):
                    errors.append(f"{name}: {why} in a Steps command")
    en = re.findall(r"(?ms)^```bash\n(.*?)^```", pages["README.md"])
    zh = re.findall(r"(?ms)^```bash\n(.*?)^```", pages["README_CN.md"])
    strip = lambda s: re.sub(r"#.*", "", s).strip()  # noqa: E731
    if [strip(b) for b in en] != [strip(b) for b in zh]:
        errors.append("README.md and README_CN.md command blocks differ (comments aside)")
    return errors


def check_shell_syntax(pages: dict[str, str]) -> list[str]:
    # Readers paste each block into bash; a block that does not parse fails before its first command runs.
    bash = shutil.which("bash")
    if bash is None:
        return ["bash not found; the Steps blocks are bash and are checked with bash -n"]
    errors = []
    for name in ("README.md", "README_CN.md"):
        for i, block in enumerate(re.findall(r"(?ms)^```bash\n(.*?)^```", pages[name]), 1):
            p = subprocess.run([bash, "-n"], input=block, capture_output=True, text=True, timeout=30)
            if p.returncode:
                last = (p.stderr.strip().splitlines() or [""])[-1]
                errors.append(f"{name}: bash block {i} does not parse: {last[:160]}")
    return errors


def delivered_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file() or rel.startswith((".git/", "evidence-local/", ".pytest_cache/")) or "__pycache__" in rel:
            continue
        if p.name == ".env" or rel in SKIP_SCAN:
            continue
        if p.suffix in SCAN_SUFFIXES or p.name in {".gitignore", ".gitattributes", ".env.example"}:
            out.append(p)
    return out


def check_private(_pages: dict[str, str], files: list[Path] | None = None) -> list[str]:
    errors = []
    for p in files if files is not None else delivered_files():
        rel = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name
        text = p.read_text(encoding="utf-8", errors="replace")
        if rel.startswith("images/icons/"):
            continue  # unmodified Microsoft icons, verified by hash in images/SOURCES.json
        if p.suffix == ".svg":
            text = re.sub(r'href="data:[^"]*"', "", text)
        for kind, pat in PRIVATE_PATTERNS.items():
            for m in re.finditer(pat, text):
                hit = m.group(0)
                if kind == "ipv4" and (hit in ALLOWED_IPV4 or (hit.startswith("10.") and hit.endswith(".0"))):
                    continue  # loopback, or the private example subnet prefix 10.x.y.0
                if kind == "guid" and hit in ALLOWED_GUIDS:
                    continue
                if kind == "local-path" and hit in ALLOWED_LOCAL_PATHS:
                    continue
                if kind == "email" and hit.endswith("@example.com"):
                    continue
                errors.append(f"{rel}: {kind} {hit[:40]}")
        for frag in INTERNAL_NAMES:
            if frag.lower() in text.lower():
                errors.append(f"{rel}: internal name {frag}")
        for frag in INTERNAL_NAMES_CASED:
            if frag in text:
                errors.append(f"{rel}: internal name {frag}")
        if rel.endswith(".md"):
            for claim in RETIRED_CLAIMS:
                if claim in text:
                    errors.append(f"{rel}: retired claim {claim!r}")
    return errors


RULES = {
    "readme-sections": check_readme_sections, "four-path-tables": check_four_path_tables,
    "information-alignment": check_information_alignment, "validation-sections": check_validation_sections,
    "run-sections": check_run_sections, "runs-listed": check_runs_listed, "run-order": check_run_order, "links": check_links,
    "images": check_images, "numbers": check_numbers, "commands": check_commands,
    "shell-syntax": check_shell_syntax, "private": check_private,
}


def run(pages: dict[str, str] | None = None) -> dict[str, list[str]]:
    pages = pages or {n: read(n) for n in ("README.md", "README_CN.md", "VALIDATION.md", "VALIDATION_CN.md")}
    return {rule: fn(pages) for rule, fn in RULES.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    results = run()
    for rule, errors in results.items():
        if errors:
            print(f"CHECK {rule} FAIL {len(errors)}")
            for e in errors:
                print(f"    {e}")
        else:
            print(f"CHECK {rule} PASS")
    failed = [r for r, e in results.items() if e]
    print(f"VALIDATE {'FAIL' if failed else 'PASS'} {len(results) - len(failed)}/{len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
