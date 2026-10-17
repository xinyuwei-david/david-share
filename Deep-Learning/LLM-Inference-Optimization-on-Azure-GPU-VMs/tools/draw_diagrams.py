#!/usr/bin/env python3
"""Draw the architecture and test-topology diagrams (English and Chinese).

    python tools/draw_diagrams.py            # write images/*.png and images/SOURCES.json
    python tools/draw_diagrams.py --check    # verify committed images against the ledger

Measured numbers are shown as generated README tables, not as images.

Chinese figures need a CJK font (Microsoft YaHei, SimHei, Noto Sans CJK or
Source Han Sans). The generator fails instead of falling back to a font that
would render Chinese labels as boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
LEDGER = IMAGES / "SOURCES.json"

TEXT = {
    "en": {
        "arch_title": "Three optimization layers on one serving path",
        "workload": "Workload &\ndeployment\nlayer",
        "framework": "Serving-\nframework layer\n(SGLang)",
        "operator": "Operator\n(kernel) layer",
        "hw": "Azure ND MI300X v5 VM: 8x MI300X (gfx942, 192 GB HBM3), 8x InfiniBand",
        "w": ["Benchmark client\nsglang.bench_serving", "PD router\nsglang_router", "Measurement modes\nfake prefill, fixed acceptance"],
        "f": ["Scheduler\nchunked prefill 64K", "EAGLE MTP\n3 steps, multi-layer", "KV cache manager\nFP8, vectorized 5D\nSWA pool", "Attention dispatch\nfull vs SWA/sink layers", "PD KV transfer\nMooncake over RDMA", "Decode graph capture\nHIP graphs"],
        "o": ["FlyDSL PA decode\nhead 192, page 64", "AITER / Gluon PA\nSWA and plain decode", "CK batch prefill\nhead 192, page 64", "CK A8W8 GEMM\nblock-scale, preshuffled", "Fused-MoE\nshape-tuned table", "Triton router GEMM\nlater, not measured"],
        "note": "CUDA counterparts are adaptation guidance; no NVIDIA deployment or performance is measured here.",
        "topo_title": "Measured test topology",
        "vm_a": "VM A  (prefill)", "vm_b": "VM B  (decode)",
        "client": "Benchmark client\nbench_serving", "router": "PD router :40000",
        "prefill": "Prefill server\nTP8 on 8x MI300X", "decode": "Decode server\nTP8 on 8x MI300X",
        "kv": "KV cache: Mooncake, 8x IB RDMA",
        "m1": "client: TTFT, TPOT,\ninput/output tok/s", "m2": "scheduler log:\ngeneration tok/s, batch",
        "single": "Single-VM mode used for the 64K A/B: one TP8 server runs prefill and decode, 16 requests of 64K in flight",
    },
    "cn": {
        "arch_title": "同一条推理链路上的三层优化",
        "workload": "负载与部署层",
        "framework": "推理框架层（SGLang）",
        "operator": "算子层（kernel）",
        "hw": "Azure ND MI300X v5 虚拟机：8 张 MI300X（gfx942，192 GB HBM3），8 路 InfiniBand",
        "w": ["压测客户端\nsglang.bench_serving", "PD 路由\nsglang_router", "测量模式\nfake prefill、固定接受长度"],
        "f": ["调度器\nchunked prefill 64K", "EAGLE MTP\n3 步、多层", "KV cache 管理\nFP8、向量化 5D\nSWA 池", "Attention 分派\n全注意力层 / SWA 层", "PD KV 传输\nMooncake over RDMA", "Decode 图捕获\nHIP graph"],
        "o": ["FlyDSL PA decode\nhead 192，page 64", "AITER / Gluon PA\nSWA 与普通 decode", "CK batch prefill\nhead 192，page 64", "CK A8W8 GEMM\nblock-scale、权重预重排", "Fused-MoE\n按 shape 调优表", "Triton router GEMM\n后续提交，未实测"],
        "note": "CUDA 对应实现只是迁移线索；本仓库未验证 NVIDIA 上的部署或性能。",
        "topo_title": "实测拓扑",
        "vm_a": "VM A（prefill）", "vm_b": "VM B（decode）",
        "client": "压测客户端\nbench_serving", "router": "PD 路由 :40000",
        "prefill": "Prefill 服务\n8 张 MI300X 上 TP8", "decode": "Decode 服务\n8 张 MI300X 上 TP8",
        "kv": "KV cache：Mooncake，8 路 IB RDMA",
        "m1": "客户端：TTFT、TPOT、\n输入/输出 tok/s", "m2": "调度器日志：\n生成 tok/s、batch",
        "single": "64K A/B 使用单机模式：一个 TP8 服务同时做 prefill 和 decode，16 个 64K 请求同时在跑",
    },
}
CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei")
COLORS = {"workload": "#FFFFFF", "framework": "#FFFFFF", "operator": "#FFFFFF", "hw": "#F2F4F6", "later": "#F2F4F6"}
AZURE_BLUE = "#0078D4"
ICON_LEDGER = IMAGES / "icons" / "SOURCES.json"
SOURCE_FILES = ("evidence/runs.json", "evidence/measurements.json", "evidence/rccl/runs.json",
                "profiles/techniques.json", "images/icons/SOURCES.json")


def verify_icons() -> None:
    ledger = json.loads(ICON_LEDGER.read_text(encoding="utf-8"))
    for item in ledger["icons"]:
        if Path(item["file"]).name != item["file"]:
            raise SystemExit("INVALID_ICON_PATH")
        path = ICON_LEDGER.parent / item["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit("ICON_LEDGER_MISMATCH " + item["file"])
    if {item["file"] for item in ledger["icons"]} != {"vm.png"}:
        raise SystemExit("ICON_USAGE_MISMATCH")


def _vm_icon(ax, x, y):
    from matplotlib import image
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    icon = OffsetImage(image.imread(IMAGES / "icons" / "vm.png"), zoom=0.16)
    ax.add_artist(AnnotationBbox(icon, (x, y), frameon=False))


def _setup(lang: str):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "fixed"
    if lang == "cn":
        names = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((f for f in CJK_FONTS if f in names), None)
        if chosen is None:
            raise SystemExit("NO_CJK_FONT: install one of " + ", ".join(CJK_FONTS))
        matplotlib.rcParams["font.family"] = [chosen]
    else:
        matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    return plt


def _box(ax, x, y, w, h, text, color, size=10, bold=False, edge="#88939D", style="-"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01",
                                facecolor=color, edgecolor=edge, linewidth=1.2, linestyle=style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def _row(ax, y, h, items, color, x0=0.17, x1=0.985, later_last=False):
    n = len(items)
    gap = 0.014
    w = (x1 - x0 - gap * (n - 1)) / n
    for i, item in enumerate(items):
        is_later = later_last and i == n - 1
        _box(ax, x0 + i * (w + gap), y, w, h, item, COLORS["later"] if is_later else color,
             size=9.5, style="--" if is_later else "-")


def architecture(lang: str, out: Path) -> None:
    plt = _setup(lang)
    t = TEXT[lang]
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.955, t["arch_title"], ha="center", va="center", fontsize=16, fontweight="bold", color=AZURE_BLUE)
    layers = [("workload", 0.72, t["w"], False), ("framework", 0.47, t["f"], False), ("operator", 0.22, t["o"], True)]
    for key, y, items, later in layers:
        _box(ax, 0.015, y, 0.14, 0.19, t[key], COLORS[key], size=11, bold=True)
        _row(ax, y + 0.02, 0.15, items, COLORS[key], later_last=later)
    for y in (0.72, 0.47):
        ax.annotate("", xy=(0.085, y - 0.055), xytext=(0.085, y), arrowprops=dict(arrowstyle="-|>", color=AZURE_BLUE, lw=1.4))
    _box(ax, 0.015, 0.09, 0.97, 0.08, t["hw"], COLORS["hw"], size=11, bold=True)
    _vm_icon(ax, 0.045, 0.13)
    ax.text(0.5, 0.035, t["note"], ha="center", va="center", fontsize=10, color="#333333")
    fig.savefig(out, dpi=100, metadata={"Software": None})
    plt.close(fig)


def topology(lang: str, out: Path) -> None:
    plt = _setup(lang)
    t = TEXT[lang]
    fig = plt.figure(figsize=(14, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, t["topo_title"], ha="center", va="center", fontsize=16, fontweight="bold", color=AZURE_BLUE)
    _box(ax, 0.03, 0.2, 0.44, 0.66, "", "#FAFAFA", edge="#888888")
    _box(ax, 0.53, 0.2, 0.44, 0.66, "", "#FAFAFA", edge="#888888")
    ax.text(0.25, 0.82, t["vm_a"], ha="center", fontsize=12, fontweight="bold")
    ax.text(0.75, 0.82, t["vm_b"], ha="center", fontsize=12, fontweight="bold")
    _vm_icon(ax, 0.075, 0.825)
    _vm_icon(ax, 0.575, 0.825)
    _box(ax, 0.06, 0.6, 0.17, 0.14, t["client"], COLORS["workload"], size=10)
    _box(ax, 0.27, 0.6, 0.17, 0.14, t["router"], COLORS["workload"], size=10)
    _box(ax, 0.06, 0.26, 0.38, 0.2, t["prefill"], COLORS["framework"], size=11)
    _box(ax, 0.56, 0.26, 0.38, 0.2, t["decode"], COLORS["framework"], size=11)
    _box(ax, 0.6, 0.6, 0.3, 0.14, t["m2"], "#FFFFFF", size=9.5, edge="#AAAAAA", style="--")
    arrow = dict(arrowstyle="-|>", color=AZURE_BLUE, lw=1.5)
    ax.annotate("", xy=(0.27, 0.67), xytext=(0.23, 0.67), arrowprops=arrow)
    ax.annotate("", xy=(0.3, 0.46), xytext=(0.3, 0.6), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.36), xytext=(0.44, 0.36), arrowprops=dict(arrowstyle="-|>", color=AZURE_BLUE, lw=2.2))
    ax.text(0.5, 0.15, t["kv"], ha="center", va="center", fontsize=10, color=AZURE_BLUE, fontweight="bold")
    ax.annotate("", xy=(0.5, 0.33), xytext=(0.5, 0.17), arrowprops=dict(arrowstyle="-", color=AZURE_BLUE, lw=1, linestyle=":"))
    ax.annotate("", xy=(0.75, 0.6), xytext=(0.75, 0.46), arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=1, linestyle=":"))
    ax.annotate("", xy=(0.44, 0.7), xytext=(0.6, 0.5), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2, linestyle="--", connectionstyle="arc3,rad=0.25"))
    ax.text(0.145, 0.53, t["m1"], ha="center", va="center", fontsize=9.5, color="#1F4E79")
    ax.text(0.5, 0.06, t["single"], ha="center", va="center", fontsize=10.5, color="#333333")
    fig.savefig(out, dpi=100, metadata={"Software": None})
    plt.close(fig)


def rccl_topology(lang: str, out: Path) -> None:
    plt = _setup(lang)
    en = lang == "en"
    evidence = json.loads((ROOT / "evidence/measurements.json").read_text(encoding="utf-8"))["rccl"]
    runs = {r["id"]: r for r in evidence["runs"]}
    fig = plt.figure(figsize=(14, 5.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title = "Two independent AllReduce routes on the same VM pair" if en else "同一对 VM 上的两条独立 AllReduce 路线"
    ax.text(0.5, 0.95, title, ha="center", fontsize=17, fontweight="bold", color=AZURE_BLUE)
    for x, label in ((0.035, "VM A"), (0.565, "VM B")):
        _box(ax, x, 0.16, 0.4, 0.69, "", "#F2F4F6")
        _vm_icon(ax, x + 0.045, 0.805)
        ax.text(x + 0.215, 0.795, label + " | Azure ND MI300X v5",
                ha="center", fontsize=12, fontweight="bold")
        for y, identity, route in (
            (0.55, "host-performance", "Host processes" if en else "宿主机进程"),
            (0.31, "container-performance", "Moby container" if en else "Moby 容器"),
        ):
            run = runs[identity]
            rocm = ".".join(run["software"]["ROCm"].split("-")[0].split(".")[:3])
            rccl = run["software"]["RCCL"].split("-")[0]
            text = f"{route}\nROCm {rocm} | RCCL {rccl}"
            _box(ax, x + 0.025, y, 0.35, 0.16, text, "#FFFFFF", size=13)
        gpu = "8 GPU ranks | 8 IB ports" if en else "8 个 GPU 进程 | 8 个 IB 端口"
        ax.text(x + 0.2, 0.205, gpu, ha="center", fontsize=12)
    for y in (0.63, 0.39):
        ax.annotate("", xy=(0.59, y), xytext=(0.41, y),
                    arrowprops=dict(arrowstyle="<->", color=AZURE_BLUE, lw=2))
        ax.text(0.5, y + 0.035, "AllReduce", ha="center", fontsize=11, color=AZURE_BLUE)
    note = ("Separate runs, not concurrent paths or fallback. Different software stacks do not isolate container overhead."
            if en else "分开运行，不是同时叠加或互为兜底；软件栈不同，不能单独归因于容器开销。")
    ax.text(0.5, 0.07, note, ha="center", fontsize=11)
    fig.savefig(out, dpi=100, metadata={"Software": None})
    plt.close(fig)


FIGURES = {
    "architecture-en.png": (architecture, "en"),
    "architecture-cn.png": (architecture, "cn"),
    "test-topology-en.png": (topology, "en"),
    "test-topology-cn.png": (topology, "cn"),
    "rccl-topology-en.png": (rccl_topology, "en"),
    "rccl-topology-cn.png": (rccl_topology, "cn"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify committed images against images/SOURCES.json")
    args = ap.parse_args(argv)
    verify_icons()
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if args.check:
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        bad = []
        if ledger.get("generator_sha256") != generator_hash or ledger.get("sources") != source_hashes:
            bad.append("generator/source hashes changed; regenerate figures")
        for item in ledger["images"]:
            path = IMAGES / item["file"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                bad.append(item["file"])
        if set(i["file"] for i in ledger["images"]) != set(FIGURES):
            bad.append("ledger/figure list mismatch")
        if bad:
            print("IMAGE_LEDGER_MISMATCH " + ", ".join(bad))
            return 1
        print(f"PASS {len(ledger['images'])} images match images/SOURCES.json")
        return 0
    IMAGES.mkdir(exist_ok=True)
    items = []
    for name, (fn, lang) in FIGURES.items():
        out = IMAGES / name
        fn(lang, out)
        item = {
            "file": name, "language": lang, "kind": "original diagram",
            "generator": "tools/draw_diagrams.py",
            "icons_used": ["icons/vm.png"],
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "shows": ("architecture of the three optimization layers" if name.startswith("architecture") else
                      "independent host and container AllReduce routes" if name.startswith("rccl-") else
                      "measured 1P1D test topology and measurement points"),
            "does_not_show": "measured traffic, latency or production readiness",
        }
        items.append(item)
    LEDGER.write_text(json.dumps({"schema": 2, "generator_sha256": generator_hash,
                                 "sources": source_hashes, "images": items}, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"wrote {len(items)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
