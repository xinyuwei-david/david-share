#!/usr/bin/env python3
"""Draw the four-path architecture figure (English and Chinese) from one layout.

    python tools/build_diagrams.py           # write the SVGs and update images/SOURCES.json
    python tools/build_diagrams.py --png     # also render the PNGs with headless Microsoft Edge (Windows)
    python tools/build_diagrams.py --check   # fail if an SVG, PNG or icon differs from images/SOURCES.json

Icons come only from images/icons/ (Azure Architecture Icons V24, unmodified) and must match the
ledger by SHA-256. Every icon in the ledger must be drawn, and every text must fit its box.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
LEDGER = IMAGES / "SOURCES.json"
CLAIMS = ROOT / "evidence" / "claims.json"

W, H = 1600, 1036
PNG_SCALE = 2
AZURE_BLUE = "#0078D4"
MS_ZONE = "#F3F2F1"
MS_BORDER = "#A19F9D"
CARD_BORDER = "#C8C6C4"
VM_GREEN = "#107C10"
TEXT = "#201F1E"
SUBTEXT = "#605E5C"
STATUS = {"ga": "#107C10", "oss": "#0078D4", "preview": "#5C2D91", "unverified": "#605E5C"}
FONT = "'Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', sans-serif"

GAP, MARGIN = 16, 24
COL_W = (W - 2 * MARGIN - 3 * GAP) // 4
COL_X = [MARGIN + i * (COL_W + GAP) for i in range(4)]
HEAD_Y, ENTRY_Y = 112, 196
MS_Y, MS_H = 244, 236
SUB_Y, SUB_H = 496, 424
CARD_H, CARD_GAP = 86, 10

I_ACA = "02989-icon-service-Container-Apps-Environments.svg"
I_AGENT = "038470523-icon-service-Foundry-Agent-Service.svg"
I_PROJECT = "036509415-icon-service-Foundry-Project.svg"
I_AKS = "10023-icon-service-Kubernetes-Services.svg"
I_ACR = "10105-icon-service-Container-Registries.svg"
I_STORAGE = "10086-icon-service-Storage-Accounts.svg"
I_APPI = "00012-icon-service-Application-Insights.svg"
I_NSG = "10067-icon-service-Network-Security-Groups.svg"

# Each card: (icon or None for the micro-VM boundary, title, detail)
LAYOUT = {
    "en": {
        "title": "Four ways to host an agent sandbox on Azure",
        "subtitle": "Who runs the lifecycle, and where each session's micro-VM runs. A/C/D placement is from our "
                    "read-backs; B's VM-isolated boundary is from Learn.",
        "ms": "MICROSOFT-MANAGED",
        "sub": "YOUR SUBSCRIPTION",
        "cols": [
            {"letter": "A", "name": "ACA Sandboxes", "status": ("GA unverified", "unverified"),
             "entry": "Client → sandbox port (wakes on request) or SDK",
             "ms": [(I_ACA, "ACA Sandboxes service", "create · stop with memory snapshot · resume"),
                    (None, "Sandbox micro-VM", "your disk image · own kernel · read back: Cloud Hypervisor")],
             "sub": [(I_ACA, "Sandbox group", "Microsoft.App resource · disk images · volumes · VNet connection"),
                     (I_ACR, "Container Registry", "source of the sandbox disk image"),
                     (I_STORAGE, "Storage · Blob volume", "agent state, mounted into the sandbox")]},
            {"letter": "B", "name": "Foundry hosted agent", "status": ("GA unverified", "unverified"),
             "entry": "Client → Foundry agent endpoint (Responses)",
             "ms": [(I_AGENT, "Foundry Agent Service", "endpoint · sessions · versions · agent identity"),
                    (None, "Per-session sandbox", "VM-isolated (docs) · $HOME and /files kept across idle")],
             "sub": [(I_PROJECT, "Foundry project", "agent definition and model deployment"),
                     (I_APPI, "Application Insights", "agent traces, when connected")]},
            {"letter": "C", "name": "AKS + agent-sandbox", "status": ("GA unverified · OSS", "unverified"),
             "entry": "Client → your LoadBalancer (not Foundry)",
             "ms": [(I_AKS, "AKS control plane", "managed Kubernetes API")],
             "sub": [(I_AKS, "agent-sandbox controller", "Sandbox resource · operatingMode Running / Suspended"),
                     (None, "Kata pod VM per Sandbox", "runtimeClass kata-vm-isolation · your image · own kernel"),
                     (I_AKS, "Kata node pool", "Azure Linux · Gen2 VM size with nested virtualization"),
                     (I_STORAGE, "Storage · Blob PVC", "agent state survives pod loss and suspend")]},
            {"letter": "D", "name": "BYO-AKS", "status": ("private preview", "preview"),
             "entry": "Client → Foundry agent endpoint (Responses)",
             "ms": [(I_AGENT, "Foundry Agent Service", "endpoint · sessions · dispatch to your cluster")],
             "sub": [(I_NSG, "Subnet NSG", "allow AzureCloud → foundry-router :443"),
                     (I_AKS, "foundry-system on your AKS", "9 fixed pods + 4 DaemonSets that scale with Kata nodes"),
                     (None, "Kata pod VM per session", "runtimeClass kata-vm-isolation · one Deployment each"),
                     (I_ACR, "Container Registry · Storage", "agent image · session state")]},
        ],
        "model": "Model inference stays on Foundry Models in all four paths; the agent calls it from inside its sandbox.",
        "legend": ["Microsoft-managed", "Your subscription", "One micro-VM per sandbox or session"],
        "footer": "Original explanatory diagram. Sources: Microsoft Learn (Container Apps Sandboxes overview, Hosted "
                  "agents in Foundry Agent Service, Pod sandboxing with AKS), agent-sandbox v0.5.2, and the runs in "
                  "evidence/runs/. Icons: Azure Architecture Icons V24, unmodified.",
    },
    "zh": {
        "title": "在 Azure 上托管 agent 沙箱的四种方式",
        "subtitle": "谁负责生命周期，每个会话的 micro-VM 跑在哪里。A/C/D 的落点来自实测读回，B 的 VM 级"
                    "隔离边界来自 Learn。",
        "ms": "微软托管",
        "sub": "你的订阅",
        "cols": [
            {"letter": "A", "name": "ACA Sandboxes", "status": ("GA 未核实", "unverified"),
             "entry": "客户端 → 沙箱端口（请求即唤醒）或 SDK",
             "ms": [(I_ACA, "ACA Sandboxes 服务", "创建 · 带内存快照的停止 · 恢复"),
                    (None, "沙箱 micro-VM", "你的磁盘镜像 · 独立内核 · 读回 VMM：Cloud Hypervisor")],
             "sub": [(I_ACA, "Sandbox group", "Microsoft.App 资源 · 磁盘镜像 · 卷 · VNet 连接"),
                     (I_ACR, "Container Registry", "沙箱磁盘镜像的来源"),
                     (I_STORAGE, "Storage · Blob 卷", "agent 状态，挂载进沙箱")]},
            {"letter": "B", "name": "Foundry hosted agent", "status": ("GA 未核实", "unverified"),
             "entry": "客户端 → Foundry agent 端点（Responses）",
             "ms": [(I_AGENT, "Foundry Agent Service", "端点 · 会话 · 版本 · agent 身份"),
                    (None, "每会话一个沙箱", "VM 级隔离（文档）· 空闲后保留 $HOME 和 /files")],
             "sub": [(I_PROJECT, "Foundry 项目", "agent 定义和模型部署"),
                     (I_APPI, "Application Insights", "接入后接收 agent 的 trace")]},
            {"letter": "C", "name": "AKS + agent-sandbox", "status": ("GA 未核实 · 开源", "unverified"),
             "entry": "客户端 → 你的 LoadBalancer，不经 Foundry 端点",
             "ms": [(I_AKS, "AKS 控制面", "托管的 Kubernetes API")],
             "sub": [(I_AKS, "agent-sandbox 控制器", "Sandbox 资源 · operatingMode Running / Suspended"),
                     (None, "每个 Sandbox 一个 Kata pod VM", "runtimeClass kata-vm-isolation · 你的镜像 · 独立内核"),
                     (I_AKS, "Kata 节点池", "Azure Linux · 支持嵌套虚拟化的 Gen2 规格"),
                     (I_STORAGE, "Storage · Blob PVC", "pod 丢失或挂起后 agent 状态仍在")]},
            {"letter": "D", "name": "BYO-AKS", "status": ("私有预览", "preview"),
             "entry": "客户端 → Foundry agent 端点（Responses）",
             "ms": [(I_AGENT, "Foundry Agent Service", "端点 · 会话 · 把会话派发到你的集群")],
             "sub": [(I_NSG, "子网 NSG", "放行 AzureCloud → foundry-router :443"),
                     (I_AKS, "你的 AKS 上的 foundry-system", "9 个固定 pod + 4 个随 Kata 节点扩展的 DaemonSet"),
                     (None, "每会话一个 Kata pod VM", "runtimeClass kata-vm-isolation · 每个会话一个 Deployment"),
                     (I_ACR, "Container Registry · Storage", "agent 镜像 · 会话状态")]},
        ],
        "model": "四条路径的模型推理都在 Foundry Models 上；agent 在自己的沙箱里调用模型。",
        "legend": ["微软托管", "你的订阅", "每个沙箱或会话一个 micro-VM"],
        "footer": "原创解释图。来源：Microsoft Learn（Container Apps Sandboxes 概览、Foundry Agent Service 的 "
                  "Hosted agents、AKS Pod sandboxing）、agent-sandbox v0.5.2，以及 evidence/runs/ 中的实测记录。"
                  "图标：Azure Architecture Icons V24，未做修改。",
    },
}


def char_em(ch: str) -> float:
    o = ord(ch)
    if o >= 0x2E80 or ch in "，。；：（）、“”":
        return 1.0
    if ch == " ":
        return 0.28
    if ch in "ilj.,:;'|!·()[]/":
        return 0.3
    if ch in "mwMW@":
        return 0.82
    if ch.isupper():
        return 0.63
    if ch.isdigit():
        return 0.55
    return 0.53


def text_width(s: str, size: float, bold: bool = False) -> float:
    return sum(char_em(c) for c in s) * size * (1.06 if bold else 1.0)


def wrap(s: str, size: float, max_w: float, bold: bool = False) -> list[str]:
    tokens: list[str] = []
    word = ""
    for ch in s:
        if ord(ch) >= 0x2E80:
            if word:
                tokens.append(word)
                word = ""
            tokens.append(ch)
        elif ch == " ":
            word += ch
            tokens.append(word)
            word = ""
        else:
            word += ch
    if word:
        tokens.append(word)
    lines, cur = [], ""
    for t in tokens:
        if cur and text_width(cur + t, size, bold) > max_w:
            lines.append(cur.rstrip())
            cur = t.lstrip()
        else:
            cur += t
    if cur:
        lines.append(cur.rstrip())
    return lines


class Svg:
    def __init__(self, icons: dict[str, str]):
        self.parts: list[str] = []
        self.icons = icons
        self.used: set[str] = set()
        self.overflows: list[str] = []

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.0, rx=4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, color=AZURE_BLUE, sw=2.0, arrow=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{marker}/>')

    def text(self, x, y, s, size, color=TEXT, bold=False, anchor="start", max_w=None, max_lines=None, line_h=None):
        lines = wrap(s, size, max_w, bold) if max_w else [s]
        if max_w:
            for ln in lines:
                if text_width(ln, size, bold) > max_w + 0.5:
                    self.overflows.append(ln)
        if max_lines and len(lines) > max_lines:
            self.overflows.append(f"{s!r} needs {len(lines)} lines, room for {max_lines}")
        lh = line_h or size * 1.3
        weight = ' font-weight="600"' if bold else ""
        spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else lh}">{escape(ln)}</tspan>' for i, ln in enumerate(lines))
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{weight}>'
                          f'{spans}</text>')

    def icon(self, name, x, y, size):
        self.used.add(name)
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="{self.icons[name]}"/>')

    def render(self) -> str:
        defs = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{AZURE_BLUE}"/></marker></defs>')
        body = "\n".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                f'font-family="{FONT}">\n{defs}\n<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n{body}\n</svg>\n')


def card(s: Svg, x: int, y: int, w: int, spec) -> None:
    icon, title, detail = spec
    if icon is None:
        s.rect(x, y, w, CARD_H, fill="#FFFFFF", stroke=VM_GREEN, sw=2.2, rx=6, dash="7 4")
        s.text(x + 14, y + 30, title, 15.5, bold=True, max_w=w - 28, max_lines=1)
        s.text(x + 14, y + 53, detail, 12.5, color=SUBTEXT, max_w=w - 28, max_lines=2, line_h=16)
        return
    s.rect(x, y, w, CARD_H, fill="#FFFFFF", stroke=CARD_BORDER, sw=1.2, rx=6)
    s.icon(icon, x + 12, y + 22, 40)
    s.text(x + 64, y + 30, title, 15.5, bold=True, max_w=w - 76, max_lines=1)
    s.text(x + 64, y + 53, detail, 12.5, color=SUBTEXT, max_w=w - 76, max_lines=2, line_h=16)


def draw(lang: str, icons: dict[str, str]) -> Svg:
    t = LAYOUT[lang]
    s = Svg(icons)
    s.rect(0, 0, W, 60, fill=AZURE_BLUE, rx=0)
    s.text(MARGIN, 40, t["title"], 24, color="#FFFFFF", bold=True)
    s.text(MARGIN, 90, t["subtitle"], 14.5, color=TEXT, max_w=W - 2 * MARGIN, max_lines=1)
    for i, col in enumerate(t["cols"]):
        x = COL_X[i]
        status, kind = col["status"]
        s.rect(x, HEAD_Y, COL_W, 70, fill="#FFFFFF", stroke=AZURE_BLUE, sw=1.6, rx=6)
        s.rect(x, HEAD_Y, 44, 70, fill=AZURE_BLUE, rx=6)
        s.text(x + 22, HEAD_Y + 46, col["letter"], 28, color="#FFFFFF", bold=True, anchor="middle")
        s.text(x + 58, HEAD_Y + 30, col["name"], 18, bold=True, max_w=COL_W - 70, max_lines=1)
        chip_w = text_width(status, 12, True) + 20
        s.rect(x + 58, HEAD_Y + 40, chip_w, 22, fill=STATUS[kind], rx=11)
        s.text(x + 58 + chip_w / 2, HEAD_Y + 55.5, status, 12, color="#FFFFFF", bold=True, anchor="middle")
        s.text(x + 12, ENTRY_Y + 16, col["entry"], 13, color=AZURE_BLUE, bold=True, max_w=COL_W - 24, max_lines=2,
               line_h=16)
        s.line(x + COL_W / 2, ENTRY_Y + 26, x + COL_W / 2, MS_Y - 2, arrow=True)
        s.rect(x, MS_Y, COL_W, MS_H, fill=MS_ZONE, stroke=MS_BORDER, sw=1.2, rx=6, dash="6 4")
        s.text(x + 12, MS_Y + 20, t["ms"], 11.5, color=SUBTEXT, bold=True)
        for j, spec in enumerate(col["ms"]):
            card(s, x + 12, MS_Y + 32 + j * (CARD_H + CARD_GAP), COL_W - 24, spec)
        s.rect(x, SUB_Y, COL_W, SUB_H, fill="#FFFFFF", stroke=AZURE_BLUE, sw=1.6, rx=6)
        s.text(x + 12, SUB_Y + 20, t["sub"], 11.5, color=AZURE_BLUE, bold=True)
        for j, spec in enumerate(col["sub"]):
            card(s, x + 12, SUB_Y + 32 + j * (CARD_H + CARD_GAP), COL_W - 24, spec)
        if len(col["ms"]) * (CARD_H + CARD_GAP) + 32 > MS_H or len(col["sub"]) * (CARD_H + CARD_GAP) + 32 > SUB_H:
            s.overflows.append(f"column {col['letter']} has more cards than its zone holds")
    band_y = SUB_Y + SUB_H + 14
    s.rect(MARGIN, band_y, W - 2 * MARGIN, 32, fill="#EFF6FC", stroke=AZURE_BLUE, sw=1.0, rx=6)
    s.text(MARGIN + 14, band_y + 21, t["model"], 13.5, bold=True, max_w=W - 2 * MARGIN - 28, max_lines=1)
    ly = band_y + 54
    lx = MARGIN
    for k, label in enumerate(t["legend"]):
        if k == 0:
            s.rect(lx, ly - 12, 26, 16, fill=MS_ZONE, stroke=MS_BORDER, sw=1.2, rx=3, dash="4 3")
        elif k == 1:
            s.rect(lx, ly - 12, 26, 16, fill="#FFFFFF", stroke=AZURE_BLUE, sw=1.6, rx=3)
        else:
            s.rect(lx, ly - 12, 26, 16, fill="#FFFFFF", stroke=VM_GREEN, sw=2.0, rx=3, dash="5 3")
        s.text(lx + 34, ly + 1, label, 12.5, color=TEXT)
        lx += 34 + text_width(label, 12.5) + 30
    s.text(MARGIN, ly + 26, t["footer"], 11, color=SUBTEXT, max_w=W - 2 * MARGIN, max_lines=2, line_h=14)
    return s


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", head[16:24])


def _render_png_in_process(svg_path: Path, png_path: Path, scale: int) -> None:
    try:
        import playwright
    except ImportError as exc:
        raise RuntimeError("PNG rendering needs playwright; install requirements-dev.txt") from exc
    driver = Path(playwright.__file__).resolve().parent / "driver"
    node = driver / ("node.exe" if sys.platform == "win32" else "node")
    script = """
const [pkg, url, output, width, height, scale] = process.argv.slice(1);
const {chromium} = require(pkg);
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true, timeout: 15000});
  try {
    const page = await browser.newPage({
      viewport: {width: Number(width), height: Number(height)},
      deviceScaleFactor: Number(scale)
    });
    await page.goto(url, {waitUntil: 'load', timeout: 15000});
    await page.screenshot({path: output, animations: 'disabled', timeout: 15000});
  } finally {
    await browser.close();
  }
})().then(() => process.exit(0), error => {console.error(error); process.exit(1);});
"""
    with tempfile.NamedTemporaryFile(dir=png_path.parent, suffix=".png", delete=False) as temporary:
        output = Path(temporary.name)
    try:
        result = subprocess.run(
            [str(node), "-e", script, str(driver / "package"), svg_path.as_uri(), str(output),
             str(W), str(H), str(scale)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"Edge PNG renderer failed: {(result.stderr or result.stdout).strip()}")
        if png_size(output) != (W * scale, H * scale):
            raise RuntimeError(f"unexpected PNG size {png_size(output)} for {png_path.name}")
        output.replace(png_path)
    finally:
        output.unlink(missing_ok=True)


def render_png(svg_path: Path, png_path: Path, scale: int = PNG_SCALE) -> None:
    # A second Edge screenshot in the same Python process can be tiled/corrupted on Windows.
    # Render each language in a fresh process so browser and compositor state cannot leak.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--render-one-svg", str(svg_path), str(png_path),
         "--scale", str(scale)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(f"PNG renderer failed for {svg_path.name}: {(result.stderr or result.stdout).strip()}")
    time.sleep(5)  # Let Edge's compositor process exit before another language is rendered.
    if png_size(png_path) != (W * scale, H * scale):
        raise RuntimeError(f"unexpected PNG size {png_size(png_path)} for {png_path.name}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_icons(ledger: dict) -> dict[str, str]:
    icons: dict[str, str] = {}
    for item in ledger["icons"]:
        p = ROOT / item["file"]
        data = p.read_bytes()
        if sha(data) != item["sha256"] or len(data) != item["bytes"]:
            raise RuntimeError(f"icon differs from images/SOURCES.json: {item['file']}")
        icons[p.name] = "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")
    on_disk = {p.name for p in (IMAGES / "icons").glob("*.svg")}
    if on_disk != set(icons):
        raise RuntimeError(f"icons on disk {sorted(on_disk)} differ from the ledger {sorted(icons)}")
    return icons


def validate_claims() -> None:
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
    mapping = claims["letter_mapping"]
    paths = claims["paths"]
    for lang in ("en", "zh"):
        for col in LAYOUT[lang]["cols"]:
            letter = col["letter"]
            expected_name = mapping[letter]["name"]
            expected_status = paths[letter][f"diagram_status_{lang}"]
            if col["name"] != expected_name:
                raise RuntimeError(f"[{lang}] path {letter} name differs from evidence/claims.json")
            if col["status"][0] != expected_status:
                raise RuntimeError(f"[{lang}] path {letter} status differs from evidence/claims.json")
    topology = claims["d_runtime_topology"]
    expected_en = (f"{topology['fixed_deployment_pods']} fixed pods + {topology['daemonsets']} DaemonSets "
                   "that scale with Kata nodes")
    expected_zh = (f"{topology['fixed_deployment_pods']} 个固定 pod + {topology['daemonsets']} 个随 Kata 节点"
                   "扩展的 DaemonSet")
    for lang, expected in (("en", expected_en), ("zh", expected_zh)):
        d = next(c for c in LAYOUT[lang]["cols"] if c["letter"] == "D")
        if d["sub"][1][2] != expected:
            raise RuntimeError(f"[{lang}] D runtime topology differs from evidence/claims.json")


def build_svgs(ledger: dict) -> dict[str, str]:
    validate_claims()
    icons = load_icons(ledger)
    out = {}
    for lang in ("en", "zh"):
        svg = draw(lang, icons)
        if svg.used != set(icons):
            raise RuntimeError(f"[{lang}] drawn icons {sorted(svg.used)} differ from the ledger {sorted(icons)}")
        if svg.overflows:
            raise RuntimeError(f"[{lang}] text does not fit its box: {svg.overflows}")
        out[f"images/architecture-{lang}.svg"] = svg.render()
    return out


def check(ledger: dict) -> list[str]:
    errors = []
    try:
        svgs = build_svgs(ledger)
    except RuntimeError as exc:
        return [str(exc)]
    figures = {f["file"]: f for f in ledger["figures"]}
    for rel, text in svgs.items():
        p = ROOT / rel
        if not p.exists() or p.read_bytes() != text.encode("utf-8"):
            errors.append(f"{rel} differs from a fresh render of tools/build_diagrams.py")
        if figures.get(rel, {}).get("sha256") != sha(text.encode("utf-8")):
            errors.append(f"{rel} hash differs from images/SOURCES.json")
        png = rel[:-4] + ".png"
        entry = figures.get(png)
        if not entry:
            errors.append(f"{png} missing from images/SOURCES.json")
            continue
        pp = ROOT / png
        if not pp.exists() or sha(pp.read_bytes()) != entry["sha256"]:
            errors.append(f"{png} differs from images/SOURCES.json")
        elif list(png_size(pp)) != [entry["width"], entry["height"]]:
            errors.append(f"{png} size differs from images/SOURCES.json")
        if entry.get("rendered_from_svg_sha256") != sha(text.encode("utf-8")):
            errors.append(f"{png} was rendered from a different SVG; run build_diagrams.py --png")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--png", action="store_true", help="render PNGs with headless Edge (Windows)")
    ap.add_argument("--check", action="store_true", help="verify committed figures against the ledger")
    ap.add_argument("--render-one-svg", nargs=2, metavar=("SVG", "PNG"), help=argparse.SUPPRESS)
    ap.add_argument("--scale", type=int, default=PNG_SCALE, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.render_one_svg:
        _render_png_in_process(Path(args.render_one_svg[0]).resolve(), Path(args.render_one_svg[1]).resolve(),
                               args.scale)
        return 0
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if args.check:
        errors = check(ledger)
        for e in errors:
            print(f"DIAGRAMS FAIL {e}", file=sys.stderr)
        if not errors:
            print(f"DIAGRAMS PASS {len(ledger['figures'])} figures, {len(ledger['icons'])} icons")
        return 1 if errors else 0
    svgs = build_svgs(ledger)
    figures = []
    for rel, text in svgs.items():
        p = ROOT / rel
        p.write_text(text, encoding="utf-8", newline="\n")
        figures.append({"file": rel, "sha256": sha(text.encode("utf-8")), "kind": "original explanatory diagram",
                        "generator": "tools/build_diagrams.py"})
        png = p.with_suffix(".png")
        if args.png:
            render_png(p, png)
        if png.exists():
            w, h = png_size(png)
            old = {f["file"]: f for f in ledger.get("figures", [])}.get(png.relative_to(ROOT).as_posix(), {})
            rendered_from = sha(text.encode("utf-8")) if args.png else old.get("rendered_from_svg_sha256")
            figures.append({"file": png.relative_to(ROOT).as_posix(), "sha256": sha(png.read_bytes()),
                            "width": w, "height": h, "rendered_from_svg_sha256": rendered_from,
                            "renderer": f"Microsoft Edge via isolated Playwright process, device scale factor {PNG_SCALE}"})
    ledger["figures"] = figures
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"DIAGRAMS WRITTEN {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
