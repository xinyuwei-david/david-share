#!/usr/bin/env python3
"""Fill the generated blocks of the README pair and the docs/ pages.

    python tools/build_readme.py          # rewrite the generated blocks in place
    python tools/build_readme.py --check  # fail if any document differs from a fresh render

Every number, table and code excerpt between `<!-- BEGIN GENERATED: name -->`
and `<!-- END GENERATED: name -->` comes from evidence/, profiles/ or the
pinned upstream patches. Prose outside the markers is written by hand. Blocks
are rendered with project-root links and rebased for pages under docs/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = {"en": ROOT / "README.md", "cn": ROOT / "README_CN.md"}
DOC_PAGES = ("optimizations", "deployment", "benchmark", "reference", "results")
DOCS = {page: {"en": ROOT / "docs" / f"{page}.md", "cn": ROOT / "docs" / f"{page}_CN.md"} for page in DOC_PAGES}
DOCUMENTS = {"readme": READMES, **DOCS}
BLOCK = re.compile(r"(<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)", re.S)
RELATIVE_TARGET = re.compile(r'(\]\(|<img\s[^>]*?src=")(?!https?://|mailto:|#|/)(docs/)?')


def rebase(text: str, prefix: str) -> str:
    """Prefix project-root relative link and image targets, leaving fenced code untouched.

    From a page inside docs/, a `docs/<page>` target becomes the sibling `<page>`.
    """
    def target(m: re.Match) -> str:
        if m.group(2):
            return m.group(1) + ("" if prefix else m.group(2))
        return m.group(1) + prefix

    out, fence = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        out.append(line if fence or line.startswith("```") else RELATIVE_TARGET.sub(target, line))
    return "\n".join(out)


def _json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _n(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _c(value: float) -> str:
    """Compact throughput: no decimals, so before/after pairs fit a phone column."""
    return f"{value:,.0f}"


def _i(value: float) -> str:
    return f"{value:,.0f}"


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]], align: list[str]) -> str:
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(_cell(h) for h in header) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        if len(r) != len(header):
            raise SystemExit(f"TABLE_SHAPE {len(r)} cells for {len(header)} columns: {r}")
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _lock() -> dict:
    return {e["file"]: e for e in _json("upstream/SOURCES.lock.json")["patches"]}


def excerpt(name: str, lang: str) -> str:
    spec = _json("tools/excerpts.json")["excerpts"][name]
    entry = _lock()[spec["patch"]]
    lines = (ROOT / "upstream" / "patches" / spec["patch"]).read_text(encoding="utf-8").split("\n")
    hits = [i for i, line in enumerate(lines) if line == spec["start"]]
    if len(hits) != 1:
        raise SystemExit(f"EXCERPT_ANCHOR {name}: {len(hits)} matches in {spec['patch']}")
    body = lines[hits[0]: hits[0] + spec["lines"]]
    if len(body) != spec["lines"]:
        raise SystemExit(f"EXCERPT_SHORT {name}")
    short = entry["commit"][:7]
    link = f"[`{entry['repository']}@{short}`]({entry['url']})"
    patch_link = f"[{spec['patch']}](upstream/patches/{spec['patch']})"
    if lang == "en":
        caption = f"Diff excerpt from {link}, file `{spec['path']}` (local copy: {patch_link})."
    else:
        caption = f"摘自 {link} 的 diff，文件 `{spec['path']}`（本地副本：{patch_link}）。"
    return caption + "\n\n```diff\n" + "\n".join(body) + "\n```\n"


def moe_table(lang: str) -> str:
    text = (ROOT / "upstream/patches/sammysun0711__aiter__d725746.patch").read_text(encoding="utf-8")
    section = text.split("b/aiter/configs/model_configs/mimo_v2_5_pro_b16_tuned_fmoe.csv\n", 2)[2]
    section = section.split("diff --git", 1)[0]
    rows_csv = [l[1:] for l in section.split("\n") if l.startswith("+") and not l.startswith("+++")]
    reader = list(csv.DictReader(io.StringIO("\n".join(rows_csv))))
    block_m = {r["block_m"] for r in reader}
    if len(block_m) != 1:
        raise SystemExit(f"MOE_BLOCK_M not constant: {block_m}")
    kernels = []
    out = []
    for r in reader:
        kernel = re.sub(r"^_ZN5aiter\d+", "", r["kernelName1"]).rstrip("E")
        if kernel not in kernels:
            kernels.append(kernel)
        out.append([_i(int(r["token"])), chr(ord("A") + kernels.index(kernel)), _n(float(r["us"]), 1), _n(float(r["tflops"]), 1)])
    bm = next(iter(block_m))
    legend = "\n".join(f"- {chr(ord('A') + i)} = `{k}`" for i, k in enumerate(kernels))
    if lang == "en":
        head = ["Tokens", "Kernel", "Time (µs)", "TFLOPS"]
        caption = f"Every row uses block_m = {bm}; time and TFLOPS are the tuner's own measurement at each token count. Kernel letters:"
    else:
        head = ["Token 数", "Kernel", "耗时（µs）", "TFLOPS"]
        caption = f"每一行的 block_m 都是 {bm}；耗时和 TFLOPS 是调优器自己在每个 token 数下测到的值。Kernel 字母含义："
    return _table(head, out, ["r", "l", "r", "r"]) + "\n" + caption + "\n\n" + legend + "\n"


def headline(lang: str) -> str:
    m = _json("evidence/measurements.json")
    ab = m["ab_ck_unified_verify_64k"]
    st = m["tuned_moe_stage"]
    pre = {r["input_tokens"]: r for r in st["prefill"]}
    dec = {r["concurrency"]: r for r in st["decode"]}
    ab_val = f"{_c(ab['baseline_mean_tok_s'])} → {_c(ab['optimized_mean_tok_s'])}"
    pre_val = f"{_c(pre[8192]['before_input_tok_s'])} → {_c(pre[8192]['after_input_tok_s'])}"
    dec_val = f"{_c(dec[128]['before_output_tok_s'])} → {_c(dec[128]['after_output_tok_s'])}"
    if lang == "en":
        head = ["What changed", "Before → after (tok/s)", "Change"]
        rows = [
            ["CK FP8 GEMM + unified verify<br>64K/1K decode, batch 16, one VM<br>*A/B, two switches, N=2*", ab_val, f"**{_pct(ab['throughput_delta_pct'])}**"],
            ["Tuned fused-MoE table<br>8K prefill, concurrency 4, 1P1D<br>*stage pair, N=1*", pre_val, f"**{_pct(pre[8192]['input_tok_s_delta_pct'])}**"],
            ["Tuned fused-MoE table<br>8K/1K decode, concurrency 128, 1P1D<br>*stage pair, N=1*", dec_val, f"**{_pct(dec[128]['output_tok_s_delta_pct'])}**"],
        ]
    else:
        head = ["改了什么", "优化前 → 优化后（tok/s）", "变化"]
        rows = [
            ["CK FP8 GEMM + unified verify<br>64K/1K decode，batch 16，单机<br>*两开关 A/B，各 2 次*", ab_val, f"**{_pct(ab['throughput_delta_pct'])}**"],
            ["调优 fused-MoE 表<br>8K prefill，并发 4，1P1D<br>*阶段对比，各 1 次*", pre_val, f"**{_pct(pre[8192]['input_tok_s_delta_pct'])}**"],
            ["调优 fused-MoE 表<br>8K/1K decode，并发 128，1P1D<br>*阶段对比，各 1 次*", dec_val, f"**{_pct(dec[128]['output_tok_s_delta_pct'])}**"],
        ]
    return _table(head, rows, ["l", "l", "r"])


def _ab_names(lang: str) -> list[str]:
    return ["Baseline", "Optimized", "Change"] if lang == "en" else ["基线", "优化后", "变化"]


def ab_table(lang: str) -> str:
    ab = _json("evidence/measurements.json")["ab_ck_unified_verify_64k"]
    n = _ab_names(lang)
    head = ["Arm", "Run 1, run 2", "Mean"] if lang == "en" else ["组别", "第 1、2 次", "均值"]
    sep = ", " if lang == "en" else "、"
    rows = [
        [n[0], sep.join(_n(x) for x in ab["baseline_run_means_tok_s"]), _n(ab["baseline_mean_tok_s"])],
        [n[1], sep.join(_n(x) for x in ab["optimized_run_means_tok_s"]), _n(ab["optimized_mean_tok_s"])],
        [n[2], "", f"**{_pct(ab['throughput_delta_pct'])}**"],
    ]
    return _table(head, rows, ["l", "r", "r"])


def ab_tpot(lang: str) -> str:
    ab = _json("evidence/measurements.json")["ab_ck_unified_verify_64k"]
    n = _ab_names(lang)
    head = ["Arm", "Implied TPOT (ms)"] if lang == "en" else ["组别", "折算 TPOT（ms）"]
    rows = [[n[0], _n(ab["baseline_implied_tpot_ms"])], [n[1], _n(ab["optimized_implied_tpot_ms"])],
            [n[2], f"**{_pct(ab['implied_tpot_delta_pct'])}**"]]
    return _table(head, rows, ["l", "r"])


def stage_prefill(lang: str) -> str:
    st = _json("evidence/measurements.json")["tuned_moe_stage"]
    head = (["Input tokens", "tok/s<br>before → after", "Change"] if lang == "en" else ["输入 token", "tok/s 优化前 → 后", "变化"])
    rows = [[_i(r["input_tokens"]), f"{_c(r['before_input_tok_s'])} → {_c(r['after_input_tok_s'])}",
             f"**{_pct(r['input_tok_s_delta_pct'])}**<br>TTFT {_pct(r['mean_ttft_delta_pct'])}"] for r in st["prefill"]]
    return _table(head, rows, ["r", "r", "r"])


def stage_decode(lang: str) -> str:
    st = _json("evidence/measurements.json")["tuned_moe_stage"]
    head = (["Concurrency", "tok/s<br>before → after", "Change"] if lang == "en" else ["并发", "tok/s 优化前 → 后", "变化"])
    rows = [[str(r["concurrency"]), f"{_c(r['before_output_tok_s'])} → {_c(r['after_output_tok_s'])}",
             f"**{_pct(r['output_tok_s_delta_pct'])}**<br>TPOT {_pct(r['mean_tpot_delta_pct'])}"] for r in st["decode"]]
    return _table(head, rows, ["r", "r", "r"])


def ladder(lang: str) -> str:
    rows_in = _json("evidence/measurements.json")["concurrency_ladder_8k1k"]
    head = (["In flight<br>(observed)", "Output<br>tok/s", "TPOT<br>(ms)", "TTFT (s)<br>mean / P99"] if lang == "en"
            else ["并发<br>（实测）", "Output<br>tok/s", "TPOT<br>（ms）", "TTFT（s）<br>均值 / P99"])
    rows = [[f"{r['concurrency']}<br>({_n(r['observed_concurrency'], 1)})", _c(r["output_tok_s"]), _n(r["mean_tpot_ms"]),
             f"{_n(r['mean_ttft_ms'] / 1000.0, 1)} / {_n(r['p99_ttft_ms'] / 1000.0, 1)}"] for r in rows_in]
    return _table(head, rows, ["r", "r", "r", "r"])


def _x(value: float) -> str:
    return f"{value:.2f}×"


def cumulative(lang: str) -> str:
    cu = _json("evidence/measurements.json")["cumulative"]
    g = cu["graph_capture"]
    pre = {r["input_tokens"]: r for r in cu["prefill"]}
    d64 = next(r for r in cu["decode"] if r["concurrency"] == 64)
    rows_spec = [
        (("Decode graph capture, one switch<br>16K/1K, 16 in flight, baseline stack", "Decode 图捕获，单开关<br>16K/1K，16 路并发，基线栈"),
         f"{_n(g['off_output_tok_s'], 1)} → {_n(g['on_output_tok_s'], 1)} tok/s", g["output_tok_s_factor"]),
        (("128K prefill, 1 request, 8 GPUs<br>one VM → 1P1D prefill server", "128K prefill，1 个请求，8 张 GPU<br>单机 → 1P1D 的 prefill 服务"),
         f"{_c(pre[131072]['early_input_tok_s'])} → {_c(pre[131072]['late_input_tok_s'])} tok/s", pre[131072]["factor"]),
        (("Decode time per token, 64 in flight<br>MTP at fixed acceptance 3, lower is better", "Decode 每 token 耗时，64 路并发<br>MTP 固定接受长度 3，越低越好"),
         f"{_n(d64['early_mean_tpot_ms'])} → {_n(d64['fixed_mean_tpot_ms'])} ms", d64["fixed_tpot_factor"]),
        (("Decode time per token, 64 in flight<br>MTP at actual acceptance, lower is better", "Decode 每 token 耗时，64 路并发<br>MTP 按实际接受，越低越好"),
         f"{_n(d64['early_mean_tpot_ms'])} → {_n(d64['real_mean_tpot_ms'])} ms", d64["real_tpot_factor"]),
        ((f"64K prefill, 4 in flight<br>baseline prompts averaged {_c(pre[65536]['early_avg_prompt_tokens'])} tokens",
          f"64K prefill，4 路并发<br>基线栈 prompt 平均 {_c(pre[65536]['early_avg_prompt_tokens'])} token"),
         f"{_c(pre[65536]['early_input_tok_s'])} → {_c(pre[65536]['late_input_tok_s'])} tok/s", pre[65536]["factor"]),
        ((f"8K prefill, 4 in flight<br>baseline prompts averaged {_c(pre[8192]['early_avg_prompt_tokens'])} tokens",
          f"8K prefill，4 路并发<br>基线栈 prompt 平均 {_c(pre[8192]['early_avg_prompt_tokens'])} token"),
         f"{_c(pre[8192]['early_input_tok_s'])} → {_c(pre[8192]['late_input_tok_s'])} tok/s", pre[8192]["factor"]),
    ]
    head = ["Measured on MI300X", "Before → after", "Factor"] if lang == "en" else ["MI300X 上实测", "优化前 → 后", "倍数"]
    rows = [[label[0 if lang == "en" else 1], val, f"**{_x(f)}**"] for label, val, f in rows_spec]
    return _table(head, rows, ["l", "l", "r"])


def cumulative_decode(lang: str) -> str:
    cu = _json("evidence/measurements.json")["cumulative"]
    if lang == "en":
        head = ["Metric", "Baseline<br>16K in", "Actual<br>MTP", "Fixed<br>MTP 3"]
        names = ("TPOT ms<br>at {c}", "tok/s<br>at {c}")
    else:
        head = ["指标", "基线栈<br>16K 输入", "按实际<br>接受", "固定接受<br>长度 3"]
        names = ("TPOT ms<br>并发 {c}", "tok/s<br>并发 {c}")
    rows = []
    for r in cu["decode"]:
        rows.append([names[0].format(c=r["concurrency"]), _n(r["early_mean_tpot_ms"]),
                     f"{_n(r['real_mean_tpot_ms'])}<br>{_x(r['real_tpot_factor'])}",
                     f"{_n(r['fixed_mean_tpot_ms'])}<br>{_x(r['fixed_tpot_factor'])}"])
    for r in cu["decode"]:
        rows.append([names[1].format(c=r["concurrency"]), _c(r["early_output_tok_s"]),
                     f"{_c(r['real_output_tok_s'])}<br>{_x(r['real_output_factor'])}",
                     f"{_c(r['fixed_output_tok_s'])}<br>{_x(r['fixed_output_factor'])}"])
    return _table(head, rows, ["l", "r", "r", "r"])


def _k(n: int) -> str:
    return f"{n // 1024}K"


def context_table(lang: str) -> str:
    cs = _json("evidence/measurements.json")["context_scaling"]
    en = lang == "en"
    head = (["Input", "Prefill tok/s<br>1 request", "Decode batch<br>steady / peak", "Decode tok/s<br>total / each"] if en
            else ["上下文", "Prefill tok/s<br>1 个请求", "Decode batch<br>（稳态 / 峰值）", "Decode tok/s<br>合计、每请求"])
    rows = []
    for r in cs["rows"]:
        ctx = _k(r["input_tokens"])
        rows.append([ctx, _c(r["prefill_1req_input_tok_s"]),
                     f"{r['decode_batch_mode']} / {r['decode_batch_max']}<br>" + (f"{r['decode_max_client_concurrency']} in flight" if en else f"并发 {r['decode_max_client_concurrency']}"),
                     f"{_c(r['decode_gen_tok_s'])}<br>{_n(r['decode_gen_tok_s_per_request'], 1)}"])
    return _table(head, rows, ["l", "r", "r", "r"])


def replicas_table(lang: str) -> str:
    rows_in = _json("evidence/measurements.json")["prefill_replicas"]
    en = lang == "en"
    head = (["Input tokens", "One server<br>1 → 2 in flight", "Two replicas<br>1 → 2 in flight"] if en
            else ["输入 token", "一个服务<br>1 → 2 个在途", "两个副本<br>1 → 2 个在途"])
    rows = []
    for r in rows_in:
        one = (f"{_c(r['single_one_tok_s'])} → {_c(r['single_two_tok_s'])}<br>{_x(r['single_factor'])}" if "single_factor" in r
               else ("not measured" if en else "未测"))
        rows.append([_i(r["input_tokens"]), one,
                     f"{_c(r['one_in_flight_tok_s'])} → {_c(r['two_in_flight_tok_s'])}<br>**{_x(r['factor'])}**"])
    return _table(head, rows, ["r", "r", "r"])


def accuracy_table(lang: str) -> str:
    rows_in = _json("evidence/measurements.json")["accuracy"]
    en = lang == "en"
    head = ["Benchmark", "Scored", "Accuracy"] if en else ["基准", "评分范围", "准确率"]
    rows = []
    for r in rows_in:
        passes = (f"{r['passes']} pass" + ("es" if r["passes"] > 1 else "")) if en else f"{r['passes']} 遍"
        scored = (f"first {_i(r['questions'])} questions × {passes}" if en else f"前 {_i(r['questions'])} 题 × {passes}")
        acc = f"**{_n(r['accuracy_pct'])}%**<br>{_i(r['correct'])} / {_i(r['responses'])}"
        rows.append([r["benchmark"], scored, acc])
    return _table(head, rows, ["l", "l", "r"])


def glance(lang: str) -> str:
    m = _json("evidence/measurements.json")
    ab, st = m["ab_ck_unified_verify_64k"], m["tuned_moe_stage"]
    p8 = next(r for r in st["prefill"] if r["input_tokens"] == 8192)
    if lang == "en":
        return "\n".join([
            f"- Single optimizations on an otherwise fixed stack: decode at 64K context **{_pct(ab['throughput_delta_pct'])}** from the block-scale FP8 GEMM "
            f"and unified-verify switches (in-session A/B); 8K prefill **{_pct(p8['input_tok_s_delta_pct'])}** from a shape-tuned fused-MoE table "
            "([details](#what-single-optimizations-added)).",
            "- The factors do not multiply: each row compares a different pair of runs. Workload and topology of every row are in "
            "[Baseline to optimized stack](#baseline-to-optimized-stack-the-cumulative-gain). No performance number compares MI300X with another accelerator.",
        ]) + "\n"
    return "\n".join([
        f"- 在其余不变的栈上单独改一项：block-scale FP8 GEMM 与 unified verify 两个开关让 64K 上下文 decode **{_pct(ab['throughput_delta_pct'])}**（同一会话 A/B）；"
        f"按 shape 调优的 fused-MoE 表让 8K prefill **{_pct(p8['input_tok_s_delta_pct'])}**（[详情](#单项优化各自带来多少)）。",
        "- 这些倍数不能相乘：每一行比较的是不同的一对运行。每一行的负载和拓扑见[从基线栈到优化后](#从基线栈到优化后累计提升)。本页没有任何性能数字拿 MI300X 和其他加速器比较。",
    ]) + "\n"


EVIDENCE_LABEL = {
    "en": {"LOCAL_AB": "measured A/B", "LOCAL_STAGE": "measured stage pair", "LOCAL_LADDER": "measured ladder",
           "ON_IN_MEASURED_RUN": "on in measured runs, not isolated", "PINNED_NOT_MEASURED": "pinned runtime, not throughput-tested here",
           "KERNEL_NOT_ISOLATED": "pinned runtime; kernel not measured in isolation", "NOT_MEASURED": "later commit, not measured"},
    "cn": {"LOCAL_AB": "实测 A/B", "LOCAL_STAGE": "实测阶段对比", "LOCAL_LADDER": "实测并发阶梯",
           "ON_IN_MEASURED_RUN": "实测时已打开，未单独拆分", "PINNED_NOT_MEASURED": "在固定 runtime 中，本仓库未测吞吐",
           "KERNEL_NOT_ISOLATED": "在固定 runtime 中；kernel 未单独测试", "NOT_MEASURED": "后续提交，未实测"},
}


PRECISION_ORDER = ("lossy", "output-contract", "same-math", "measurement-only")
PRECISION_TITLE = {
    "en": {"lossy": "Lossy — check accuracy before production", "output-contract": "Output-preserving only if the implementation is correct",
           "same-math": "Same arithmetic, different kernel or layout", "measurement-only": "Benchmark methods — never score their output"},
    "cn": {"lossy": "有损——上线前必须核对准确率", "output-contract": "只有实现正确时才不改变输出",
           "same-math": "算术不变，只换 kernel 或布局", "measurement-only": "测试方法——生成内容不能算分"},
}
PRECISION_SHORT = {
    "en": {"lossy": "lossy", "output-contract": "exact if correct", "same-math": "same math", "measurement-only": "test method only"},
    "cn": {"lossy": "有损", "output-contract": "实现正确则不变", "same-math": "算术不变", "measurement-only": "仅测试方法"},
}


def slug(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _measured(t: dict, lang: str) -> str:
    """Short 'what it did' cell: a measured number where one exists, otherwise the evidence status."""
    m = _json("evidence/measurements.json")
    ab, st, cu = m["ab_ck_unified_verify_64k"], m["tuned_moe_stage"], m["cumulative"]
    p8 = next(r for r in st["prefill"] if r["input_tokens"] == 8192)
    d128 = next(r for r in st["decode"] if r["concurrency"] == 128)
    rep8 = next(r for r in m["prefill_replicas"] if r["input_tokens"] == 8192)
    en = lang == "en"
    special = {
        "decode-graph-capture": (f"{_x(cu['graph_capture']['output_tok_s_factor'])} decode (A/B)", f"decode {_x(cu['graph_capture']['output_tok_s_factor'])}（A/B）"),
        "ck-a8w8-gemm": (f"{_pct(ab['throughput_delta_pct'])} decode, with unified verify (A/B)", f"decode {_pct(ab['throughput_delta_pct'])}，与 unified verify 合计（A/B）"),
        "unified-verify": (f"{_pct(ab['throughput_delta_pct'])} decode, with CK GEMM (A/B)", f"decode {_pct(ab['throughput_delta_pct'])}，与 CK GEMM 合计（A/B）"),
        "tuned-fused-moe": (f"8K prefill {_pct(p8['input_tok_s_delta_pct'])}, decode {_pct(d128['output_tok_s_delta_pct'])} (stage pair)",
                            f"8K prefill {_pct(p8['input_tok_s_delta_pct'])}，decode {_pct(d128['output_tok_s_delta_pct'])}（阶段对比）"),
        "concurrency-ladder": ("finds the plateau (ladder)", "找到饱和点（并发阶梯）"),
        "prefill-replicas": (f"8K prefill {_x(rep8['factor'])} (two replicas)", f"8K prefill {_x(rep8['factor'])}（两个副本）"),
    }
    if t["id"] in special:
        return special[t["id"]][0 if en else 1]
    if t["id"] == "fake-prefill":
        return "not used in the published runs" if en else "已发布的测试中未使用"
    if t["evidence"] == "ON_IN_MEASURED_RUN":
        partly = t["stack"] == "pinned-runtime"
        return (("partly on, not isolated" if partly else "on, not isolated") if en
                else ("部分已打开，未单独拆分" if partly else "已打开，未单独拆分"))
    short = {"NOT_MEASURED": ("later commit, not measured", "后续提交，未实测")}
    return short.get(t["evidence"], ("final runtime, not measured", "在最终 runtime 中，未实测"))[0 if en else 1]


def technique_overview(lang: str) -> str:
    cat = _json("profiles/techniques.json")
    head = ["Technique", "What it did on MI300X", "Output"] if lang == "en" else ["优化手段", "在 MI300X 上的效果", "对输出"]
    out = []
    for layer in ("framework", "operator", "workload"):
        out.append(f"**{cat['layers'][layer][lang]}**\n")
        rows = [[f"[{t[lang]}](#{slug(t[lang])})", _measured(t, lang), PRECISION_SHORT[lang][t["precision"]["class"]]]
                for t in cat["techniques"] if t["layer"] == layer]
        out.append(_table(head, rows, ["l", "l", "l"]))
    return "\n".join(out)


def card(tid: str, lang: str) -> str:
    cat = _json("profiles/techniques.json")
    t = next(x for x in cat["techniques"] if x["id"] == tid)
    lock = _lock()
    en = lang == "en"
    lab = ({"mi": "Switch on MI300X", "nv": "On NVIDIA", "code": "Code", "ev": "Evidence", "out": "Effect on output",
            "none": "no code change (configuration only)"} if en else
           {"mi": "MI300X 上的开关", "nv": "NVIDIA 上", "code": "代码", "ev": "证据", "out": "对输出的影响",
            "none": "无代码改动（仅配置）"})
    codes = ", ".join(f"[{lock[p]['commit'][:7]}]({lock[p]['url']})" for p in t["patches"]) or lab["none"]
    label = t["evidence_note"][lang] if "evidence_note" in t else EVIDENCE_LABEL[lang][t["evidence"]]
    sep = ": " if en else "："
    cls = t["precision"]["class"]
    return "\n".join([
        f"- **{lab['mi']}**{sep}{t['rocm'] if en else t['rocm_cn']}",
        f"- **{lab['nv']}**{sep}{t['cuda'] if en else t['cuda_cn']}",
        f"- **{lab['code']}**{sep}{codes}",
        f"- **{lab['ev']}**{sep}{label}",
        f"- **{lab['out']}**{sep}{PRECISION_SHORT[lang][cls]}{'. ' if en else '。'}{t['precision'][lang]}",
    ]) + "\n"


def precision_summary(lang: str) -> str:
    cat = _json("profiles/techniques.json")
    classes = cat["precision_classes"]
    out = []
    for cls in PRECISION_ORDER:
        items = [t for t in cat["techniques"] if t["precision"]["class"] == cls]
        if not items:
            continue
        names = ("; " if lang == "en" else "；").join(f"[{t[lang]}](#{slug(t[lang])})" for t in items)
        stop = ". " if lang == "en" else "。"
        out.append(f"- **{PRECISION_TITLE[lang][cls]}**{stop}{classes[cls][lang]} {names}")
    return "\n".join(out) + "\n"


def numerical_tests(lang: str) -> str:
    lock = _json("upstream/SOURCES.lock.json")
    lines = []
    for e in lock["patches"]:
        text = (ROOT / "upstream" / "patches" / e["file"]).read_text(encoding="utf-8")
        current = None
        tests = []
        for line in text.split("\n"):
            m = re.match(r"^\+\+\+ b/(.+)$", line)
            if m:
                current = m.group(1)
            m = re.match(r"^\+def (test_\w+)\(", line)
            if m and current:
                tests.append((current, m.group(1)))
        for path, name in tests:
            lines.append(f"- [`{e['repository']}@{e['commit'][:7]}`]({e['url']}) `{path}::{name}`")
    if not lines:
        raise SystemExit("NO_UPSTREAM_TESTS found in the pinned patches")
    return "\n".join(lines) + "\n"


STACK_LABEL = {
    "en": {"measured_run": "in the measured runs", "pinned_runtime_not_throughput_tested": "pinned runtime, not throughput-tested here",
           "later_branch": "later branch, not in any runtime here", "upstreamed_later": "upstream follow-up, not in the pinned runtime"},
    "cn": {"measured_run": "在实测 runtime 中", "pinned_runtime_not_throughput_tested": "在固定 runtime 中，本仓库未测吞吐",
           "later_branch": "后续分支，不在任何 runtime 中", "upstreamed_later": "上游后续提交，不在固定 runtime 中"},
}


PROFILE_STATUS = {
    "en": {"MEASURED": "1P1D settings of the measured stage pair and concurrency ladder ({runs})",
           "PINNED_RUNTIME": "final pinned runtime; no published throughput run",
           "TEMPLATE_NOT_MEASURED": "mapping to NVIDIA; never run here"},
    "cn": {"MEASURED": "阶段对比与并发阶梯（{runs}）实测所用的 1P1D 配置",
           "PINNED_RUNTIME": "最终固定 runtime；未发布吞吐实测",
           "TEMPLATE_NOT_MEASURED": "NVIDIA 对应模板；本仓库未运行"},
}


def _profiles() -> list[dict]:
    paths = sorted(p for p in (ROOT / "profiles").glob("*.json") if p.name != "techniques.json")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def launch_profiles(lang: str) -> str:
    order = list(_json("evidence/runs.json")["runs"])
    head = ["Profile", "Status", "Roles"] if lang == "en" else ["Profile", "状态", "角色"]
    rows = []
    for p in _profiles():
        runs = "–".join(f"S{order.index(r) + 1}" for r in p["measured_runs"][:: max(len(p["measured_runs"]) - 1, 1)])
        if p["status"] == "MEASURED" and not runs:
            raise SystemExit(f"MEASURED_WITHOUT_RUNS {p['profile']}")
        rows.append([f"`{p['profile']}`", f"`{p['status']}`<br>" + PROFILE_STATUS[lang][p["status"]].format(runs=runs),
                     " · ".join(f"`{role}`" for role in p["roles"])])
    return _table(head, rows, ["l", "l", "l"])


def _effective_roles(profile: dict, technique: str) -> list[str]:
    """Roles whose rendered environment or arguments change when the technique is ablated."""
    from render_launch import compose  # same directory; imported lazily so the module stays importable alone
    roles = []
    for role in profile["roles"]:
        deps = [t["id"] for t in _json("profiles/techniques.json")["techniques"]
                if technique in t.get("requires", []) and t["id"] in profile["techniques"]]
        if compose(profile, role, [technique, *deps])[:2] != compose(profile, role, deps)[:2]:
            roles.append(role)
    return roles


def ablation_names(lang: str) -> str:
    en = lang == "en"
    catalog = _json("profiles/techniques.json")["techniques"]
    page = DOCS["optimizations"][lang].relative_to(ROOT).as_posix()
    rows = []
    for t in catalog:
        cells = [f"`{p['profile']}`: " + ", ".join(f"`{r}`" for r in roles)
                 for p in _profiles() if t["id"] in p["techniques"] for roles in [_effective_roles(p, t["id"])] if roles]
        if cells:
            rows.append([f"`{t['id']}`", f"[{t[lang]}]({page}#{slug(t[lang])})", "<br>".join(cells)])
    head = (["`--ablate` name", "Removes", "Changes the script of"] if en
            else ["`--ablate` 名称", "去掉的优化", "会改变哪些角色的脚本"])
    notes = ["- Within a profile that includes a name, a role not listed for it accepts the name, but the rendered "
             "script changes only in its comment lines. A name that a profile does not include is rejected by "
             "`render_launch.py`." if en
             else "- 在包含某个名称的 profile 中，表里没有列出的角色也接受这个名称，但渲染出的脚本只有注释行变化。"
             "profile 不包含的名称会被 `render_launch.py` 直接拒绝。"]
    if any(t["id"] == "tuned-fused-moe" for t in catalog):
        notes.append("- `cuda-hopper-pd` carries `tuned-fused-moe` only as a note: generate and install the Triton "
                     "fused-MoE configuration yourself." if en else
                     "- `cuda-hopper-pd` 里的 `tuned-fused-moe` 只是一条注释：Triton fused-MoE 配置要自己生成并放入。")
    for t in catalog:
        for dep in t.get("requires", []):
            notes.append(f"- In profiles that include `{t['id']}`, ablating `{dep}` also requires ablating `{t['id']}`, "
                         "which depends on it." if en else
                         f"- 在包含 `{t['id']}` 的 profile 中，消融 `{dep}` 时必须同时消融依赖它的 `{t['id']}`。")
    return _table(head, rows, ["l", "l", "l"]) + "\n" + "\n".join(notes) + "\n"


def upstream_table(lang: str) -> str:
    lock = _json("upstream/SOURCES.lock.json")
    cat = _json("profiles/techniques.json")["layers"]
    lines = [f"- [`{e['repository']}@{e['commit'][:7]}`]({e['url']}) — {e['subject']}  \n  {cat[e['layer']][lang]} · {STACK_LABEL[lang][e['stack_status']]}"
             for e in lock["patches"]]
    return "\n".join(lines) + "\n"


RUN_LABELS = {
    "stage-before-moe-table": ("MoE tuning: before", "MoE 调优前"),
    "stage-after-moe-table": ("MoE tuning: after", "MoE 调优后"),
    "ladder-ck": ("Decode concurrency ladder", "Decode 并发扫描"),
    "ab-64k-bs16": ("Two-switch 64K A/B", "64K 双开关 A/B"),
    "baseline-pd-client": ("Graph capture A/B", "图捕获 A/B"),
    "baseline-pd-summary": ("Baseline stack", "基线软件栈"),
    "actual-acceptance": ("Actual MTP acceptance", "实际 MTP 接受率"),
    "context-matrix": ("Context-length sweep", "上下文长度扫描"),
    "prefill-replicas": ("Prefill replicas", "Prefill 副本"),
    "accuracy-subset": ("Public accuracy subsets", "公开基准子集准确率"),
}


def _run_heading(identity: str, index: int, lang: str) -> str:
    return f"{RUN_LABELS[identity][lang == 'cn']} (S{index})"


# The results-page section that derives its numbers from each run's records.
RUN_SECTIONS = {
    "stage-before-moe-table": ("Stage pair: shape-tuned fused-MoE table", "阶段对比：按 shape 调优的 fused-MoE 表"),
    "stage-after-moe-table": ("Stage pair: shape-tuned fused-MoE table", "阶段对比：按 shape 调优的 fused-MoE 表"),
    "ladder-ck": ("Where decode saturates: concurrency ladder", "Decode 在哪里饱和：并发阶梯"),
    "ab-64k-bs16": ("Controlled A/B: block-scale FP8 GEMM path at 64K context",
                    "受控 A/B：64K 上下文下的 block-scale FP8 GEMM 路径"),
    "baseline-pd-client": ("Baseline to optimized stack: the cumulative gain", "从基线栈到优化后：累计提升"),
    "baseline-pd-summary": ("Baseline to optimized stack: the cumulative gain", "从基线栈到优化后：累计提升"),
    "actual-acceptance": ("Baseline to optimized stack: the cumulative gain", "从基线栈到优化后：累计提升"),
    "context-matrix": ("How throughput changes with context length", "吞吐怎样随上下文长度变化"),
    "prefill-replicas": ("Adding a second prefill replica", "增加第二个 prefill 副本"),
    "accuracy-subset": ("Accuracy measured on the optimized kernels", "在优化后的 kernel 上实测的准确率"),
}


def rccl_headline(lang: str) -> str:
    runs = {r["id"]: r for r in _json("evidence/measurements.json")["rccl"]["runs"]}
    rows = []
    for identity, label in (("host-performance", "Host" if lang == "en" else "宿主机"),
                            ("container-performance", "Moby container" if lang == "en" else "Moby 容器")):
        run = runs[identity]
        largest = next(row for row in run["rows"] if row["bytes"] == 16 * 1024 ** 3)
        rocm = ".".join(run["software"]["ROCm"].split("-")[0].split(".")[:3])
        rccl = run["software"]["RCCL"].split("-")[0]
        rows.append([label, f"ROCm {rocm}<br>RCCL {rccl}", _n(run["average_busbw_gb_s"]),
                     _n(largest["out_of_place"]["busbw_gb_s"])])
    header = (["Route", "Software", "Average<br>GB/s", "16 GiB<br>GB/s"] if lang == "en"
              else ["运行方式", "软件栈", "平均<br>GB/s", "16 GiB<br>GB/s"])
    return _table(header, rows, ["l", "l", "r", "r"])


def run_index(lang: str) -> str:
    en = lang == "en"
    meta = _json("evidence/runs.json")["runs"]
    m = _json("evidence/measurements.json")
    pre = next(r for r in m["tuned_moe_stage"]["prefill"] if r["input_tokens"] == 8192)
    ladder = next(r for r in m["concurrency_ladder_8k1k"] if r["concurrency"] == 64)
    decode = next(r for r in m["cumulative"]["decode"] if r["concurrency"] == 64)
    context = next(r for r in m["context_scaling"]["rows"] if r["input_tokens"] == 65536)
    replicas = next(r for r in m["prefill_replicas"] if r["input_tokens"] == 8192)
    graph = m["cumulative"]["graph_capture"]
    results = {
        "stage-before-moe-table": f"8K prefill: {_n(pre['before_input_tok_s'])} input tok/s",
        "stage-after-moe-table": f"8K prefill: {_n(pre['after_input_tok_s'])} input tok/s",
        "ladder-ck": f"C=64: {_n(ladder['output_tok_s'])} output tok/s; fixed MTP 3",
        "ab-64k-bs16": f"{_pct(m['ab_ck_unified_verify_64k']['throughput_delta_pct'])}; N=2/arm",
        "baseline-pd-client": f"{_n(graph['off_output_tok_s'], 1)} → {_n(graph['on_output_tok_s'], 1)} output tok/s",
        "baseline-pd-summary": f"C=64, 16K: TPOT {_n(decode['early_mean_tpot_ms'])} ms",
        "actual-acceptance": f"C=64, 8K: TPOT {_n(decode['real_mean_tpot_ms'])} ms",
        "context-matrix": f"64K, 1 request: {_n(context['prefill_1req_input_tok_s'])} input tok/s",
        "prefill-replicas": f"8K, 2 in flight: {_n(replicas['two_in_flight_tok_s'])} input tok/s",
        "accuracy-subset": f"{len(m['accuracy'])} benchmark subsets; scores on the results page",
    }
    if not en:
        results.update({
            "ladder-ck": f"并发 64：{_n(ladder['output_tok_s'])} output tok/s；MTP 固定接受 3",
            "ab-64k-bs16": f"{_pct(m['ab_ck_unified_verify_64k']['throughput_delta_pct'])}；每组 2 次",
            "context-matrix": f"64K，1 个请求：{_n(context['prefill_1req_input_tok_s'])} input tok/s",
            "prefill-replicas": f"8K，2 个请求在途：{_n(replicas['two_in_flight_tok_s'])} input tok/s",
            "accuracy-subset": f"{len(m['accuracy'])} 个基准子集；逐项分数见实测结果页",
        })
    page = DOCS["results"][lang].relative_to(ROOT).as_posix()
    rows = []
    for i, identity in enumerate(meta, 1):
        heading = _run_heading(identity, i, lang)
        rows.append([f"[S{i}]({page}#{slug(heading)})", RUN_LABELS[identity][not en], results[identity]])
    for i, run in enumerate(m["rccl"]["runs"], 1):
        result = f"{run['average_busbw_gb_s']:g} GB/s; data check {run['data_check']}"
        if not en:
            result = f"{run['average_busbw_gb_s']:g} GB/s；数据校验 {run['data_check']}"
        rows.append([f"[R{i}]({page}#{slug(run['label'][lang])})", run["label"][lang], result])
    header = ["Run", "Scenario", "Recorded result / scope"] if en else ["运行", "场景", "记录结果 / 范围"]
    return _table(header, rows, ["l", "l", "l"])


def run_inputs(lang: str) -> str:
    en = lang == "en"
    runs = _json("evidence/runs.json")["runs"]
    out = []
    for i, (identity, record) in enumerate(runs.items(), 1):
        title = _run_heading(identity, i, lang)
        links = ", ".join(f"[{name}](evidence/raw/{name})" for name in record["raw"])
        raw = record.get("server_flags")
        if raw:
            input_text = f"`{raw}`"
        else:
            input_text = ("The complete shell invocation was not captured in this bundle; the linked public "
                          "record retains its available workload/configuration fields." if en else
                          "这份记录没有保存完整的 shell 调用；下列公开证据保留了当时可获取的负载和配置字段。")
        if en:
            out.extend([
                f"### {title}\n",
                f"**Question.** What does the recorded {RUN_LABELS[identity][0].lower()} establish?\n",
                f"**Recorded input.** {input_text}\n",
                f"**Variables.** Use the topology, runtime, flags and environment for `{identity}` in "
                "[the frozen run catalog](evidence/runs.json), not another bundle's configuration.\n",
                f"**Result.** {links}. [{RUN_SECTIONS[identity][0]}](#{slug(RUN_SECTIONS[identity][0])}) derives "
                "its values from these records.\n",
                "**Boundary.** A public projection supports the retained values, not a byte-for-byte rerun of "
                "missing private input. Replaying evidence is not a fresh serving experiment.\n",
            ])
        else:
            out.extend([
                f"### {title}\n",
                f"**问题。** 这份“{RUN_LABELS[identity][1]}”记录能够说明什么？\n",
                f"**记录的输入。** {input_text}\n",
                f"**变量。** 使用[固定运行目录](evidence/runs.json)中 `{identity}` 对应的拓扑、runtime、开关和环境，不能借用其他组的配置。\n",
                f"**结果。** {links}。[{RUN_SECTIONS[identity][1]}](#{slug(RUN_SECTIONS[identity][1])})一节从这些记录计算结果。\n",
                "**边界。** 公开投影只能支撑保留下来的数值；缺失的私有输入不能保证逐字节重放。离线复算也不是重新运行推理服务。\n",
            ])
    return "\n".join(out)


def rccl_runs(lang: str) -> str:
    en = lang == "en"
    data = _json("evidence/measurements.json")["rccl"]
    questions = {
        "host-vs-container": ("Does a completed container run remain in the same bandwidth range as the host run?",
                              "容器中完成的运行是否仍与宿主机处于同一带宽水平？"),
        "correctness": ("Are the collective results correct, and does every rank select IB?",
                        "集合通信结果是否正确，每个进程是否都选择了 IB？"),
        "ib-vs-socket": ("Can a run finish while its RCCL traffic falls back to Socket?",
                         "RCCL 退回 Socket 时，程序是否仍可能正常结束？"),
        "single-channels": ("What changes when the minimum-channel setting is removed on one VM?",
                            "单机取消最少通道数设置后，带宽发生了什么变化？"),
    }
    out = []
    for run in data["runs"]:
        p = run["parameters"]
        title = run["label"][lang]
        question = questions[run["comparison"]][not en]
        if run["id"] == "ib-control-validation":
            question = ("With the IB device visible, does RCCL choose IB and pass its data check?" if en else
                        "映射 IB 设备后，RCCL 是否选择 IB 并通过数据校验？")
        out.append(f"### {title}\n")
        if en:
            out.extend([
                f"**Question.** {question}\n",
                "**Recorded input.** The following is the recorded MPI invocation with node addresses and "
                "the executable path replaced by variables; the other arguments are unchanged.\n",
                "```bash\n" + run["command"] + "\n```\n",
                f"**Variables.** {run['environment']['nodes']} VM(s), {run['ranks']} ranks, float/sum; "
                f"{p['iterations']} timed iterations, {p['warmup_iterations']} warmups, validation={p['validation']}. "
                f"{run['boundary'][lang]} Each arm is one sweep, not a repeatability estimate.\n",
                "**Observed result.**\n",
            ])
        else:
            out.extend([
                f"**问题。** {question}\n",
                "**记录的输入。** 以下是当时的 MPI 调用，仅将节点地址和程序路径替换为变量，其余参数不变。\n",
                "```bash\n" + run["command"] + "\n```\n",
                f"**变量。** {run['environment']['nodes']} 台 VM、{run['ranks']} 个进程，float/sum；"
                f"计时 {p['iterations']} 次、预热 {p['warmup_iterations']} 次，validation={p['validation']}。"
                f"{run['boundary'][lang]} 每组仅一次扫描，不能估计重复性。\n",
                "**观测结果。**\n",
            ])
        header = (["Message (GiB)", "Out-of-place<br>GB/s", "In-place<br>GB/s"] if en
                  else ["消息（GiB）", "Out-of-place<br>GB/s", "In-place<br>GB/s"])
        rows = [[f"{row['bytes'] / 1024 ** 3:g}", _n(row["out_of_place"]["busbw_gb_s"]),
                 _n(row["in_place"]["busbw_gb_s"])] for row in run["rows"]]
        out.append(_table(header, rows, ["r", "r", "r"]))
        network = run["network_markers"]
        if run["environment"]["nodes"] == 1:
            transport_boundary = ("This is intra-VM communication, not a validation of the inter-VM IB path." if en else
                                  "这是单机通信结果，不能验证跨机 IB 链路。")
        elif network["IB"] == run["ranks"] and network["GDRDMA"]:
            transport_boundary = ("This run positively records IB selection and GPU Direct RDMA." if en else
                                  "本次运行已正向记录 IB 选择和 GPU Direct RDMA。")
        elif network["Socket"]:
            transport_boundary = ("This run records Socket fallback, not an IB data path." if en else
                                  "本次记录的是 Socket 回退，不能作为走 IB 的证明。")
        else:
            transport_boundary = ("Zero INFO markers in a VERSION-only log mean not captured, not absence of RDMA." if en else
                                  "VERSION 级别日志中没有 INFO 标记，只表示未记录，不能判断没有 RDMA。")
        if en:
            out.append(f"Reported average: **{run['average_busbw_gb_s']:g} GB/s** across both modes and all sizes. "
                       f"Data check: **{run['data_check']}**. INFO markers: IB={network['IB']}, "
                       f"Socket={network['Socket']}, GDRDMA={network['GDRDMA']}.\n")
            out.append(f"**Boundary.** {transport_boundary} "
                       f"busbw is normalized collective bandwidth, not measured NIC utilization or model throughput. "
                       f"[Numeric projection]({run['source']}); [source hashes and conditions](evidence/rccl/runs.json).\n")
        else:
            out.append(f"工具报告平均 **{run['average_busbw_gb_s']:g} GB/s**，包含两种模式和全部消息大小。"
                       f"数据校验：**{run['data_check']}**。INFO 标记：IB={network['IB']}、"
                       f"Socket={network['Socket']}、GDRDMA={network['GDRDMA']}。\n")
            out.append(f"**边界。** {transport_boundary}"
                       f"busbw 是归一化集合通信带宽，不是网卡利用率或模型吞吐率。"
                       f"[数值投影]({run['source']})；[来源哈希与条件](evidence/rccl/runs.json)。\n")
    return "\n".join(out)


def rccl_observation(lang: str) -> str:
    item = _json("evidence/measurements.json")["rccl"]["initialization_observation"]
    failed, retries = item["failures_recorded"], item["completed_targeted_retries"]
    if lang == "en":
        return (f"**Initialization observation:** {failed} recorded attempt failed inside rocm_smi; "
                f"{retries} subsequent targeted retries completed. The root cause is **{item['root_cause_status']}**. "
                f"These observations establish neither a failure rate nor a fix. [Diagnostic excerpts]({item['source']}).\n")
    return (f"**初始化观察：** {failed} 次记录的尝试在 rocm_smi 内失败，随后 {retries} 次专项重试完成。"
            f"根因为 **{item['root_cause_status']}**，不能据此估计失败率或认定已经修复。"
            f"[诊断摘录]({item['source']})。\n")


BLOCKS = {
    "glance": glance,
    "cumulative": cumulative,
    "cumulative-decode": cumulative_decode,
    "headline": headline,
    "context-table": context_table,
    "replicas-table": replicas_table,
    "accuracy-table": accuracy_table,
    "ab-table": ab_table,
    "ab-tpot": ab_tpot,
    "stage-prefill": stage_prefill,
    "stage-decode": stage_decode,
    "ladder": ladder,
    "technique-overview": technique_overview,
    "precision-summary": precision_summary,
    "numerical-tests": numerical_tests,
    "moe-table": moe_table,
    "upstream-table": upstream_table,
}
for _t in _json("profiles/techniques.json")["techniques"]:
    BLOCKS[f"card-{_t['id']}"] = (lambda i: (lambda lang: card(i, lang)))(_t["id"])
for _name in _json("tools/excerpts.json")["excerpts"]:
    BLOCKS[f"excerpt-{_name}"] = (lambda n: (lambda lang: excerpt(n, lang)))(_name)
BLOCKS.update({"rccl-headline": rccl_headline, "rccl-runs": rccl_runs, "rccl-observation": rccl_observation,
               "run-index": run_index, "run-inputs": run_inputs, "launch-profiles": launch_profiles,
               "ablation-names": ablation_names})
REQUIRED_BLOCKS = {
    "readme": {"headline", "rccl-headline", "run-index"},
    "optimizations": {"technique-overview", "precision-summary", "moe-table"}
                     | {name for name in BLOCKS if name.startswith(("card-", "excerpt-"))},
    "deployment": set(),
    "benchmark": {"numerical-tests"},
    "reference": {"upstream-table", "launch-profiles", "ablation-names"},
    "results": {"cumulative", "glance", "cumulative-decode", "headline", "ab-table", "ab-tpot", "stage-prefill",
                "stage-decode", "ladder", "context-table", "replicas-table", "accuracy-table", "run-inputs",
                "rccl-observation", "rccl-runs"},
}


def render(text: str, lang: str, document: str = "readme") -> str:
    seen = set()
    prefix = "" if document == "readme" else "../"

    def fill(m: re.Match) -> str:
        name = m.group("name")
        if name not in BLOCKS:
            raise SystemExit(f"UNKNOWN_BLOCK {name} in {lang} {document}")
        if name in seen:
            raise SystemExit(f"DUPLICATE_BLOCK {name} in {lang} {document}")
        seen.add(name)
        return m.group(1) + rebase(BLOCKS[name](lang), prefix) + m.group(4)

    out = BLOCK.sub(fill, text)
    missing = REQUIRED_BLOCKS[document] - seen
    if missing:
        raise SystemExit(f"MISSING_BLOCKS in {lang} {document}: {sorted(missing)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    for document, paths in DOCUMENTS.items():
        for lang, path in paths.items():
            current = path.read_text(encoding="utf-8")
            fresh = render(current, lang, document)
            if args.check:
                if fresh != current:
                    stale.append(path.name)
            else:
                path.write_text(fresh, encoding="utf-8", newline="\n")
    if args.check:
        if stale:
            print("STALE " + ", ".join(stale) + "; run tools/build_readme.py")
            return 1
        count = sum(len(paths) for paths in DOCUMENTS.values())
        print(f"PASS {len(BLOCKS)} generated blocks are current across {count} README and docs pages")
        return 0
    print("rendered " + ", ".join(p.relative_to(ROOT).as_posix() for paths in DOCUMENTS.values() for p in paths.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
