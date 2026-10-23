#!/usr/bin/env python3
"""Fill the generated blocks and inline values of README.md, README_CN.md, VALIDATION.md and VALIDATION_CN.md.

    python tools/render_docs.py           # rewrite the four pages in place
    python tools/render_docs.py --check   # fail if any page differs from a fresh render

Two kinds of generated content, both rebuilt from evidence/runs/ (hash-checked) and
evidence/measurements.json (must equal a fresh build):

    <!-- BEGIN GENERATED: name -->...<!-- END GENERATED: name -->   tables and verbatim inputs
    <!--v:metric.field-->value<!--/v-->                             one number inside prose

Hand-written prose around them is left untouched. An unknown block or field fails the render.
The results tables follow the canonical run_order in evidence/claims.json.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_measurements as bm  # noqa: E402

ROOT = bm.ROOT
PAGES = {"README.md": "en", "README_CN.md": "zh", "VALIDATION.md": "en", "VALIDATION_CN.md": "zh"}
BLOCK_RE = re.compile(r"(<!-- BEGIN GENERATED: ([a-z0-9-]+) -->\n)(.*?)(<!-- END GENERATED: \2 -->)", re.S)
VALUE_RE = re.compile(r"<!--v:([a-z0-9_.]+)-->(.*?)<!--/v-->")


class RenderError(RuntimeError):
    pass


def num(v, places: int = 2) -> str:
    """Seconds keep two decimals (three for sub-100ms values; trailing zeros dropped, one kept); prices pass places=3."""
    if isinstance(v, bool):
        raise RenderError(f"boolean where a number was expected: {v}")
    if isinstance(v, int):
        return str(v)
    if float(v).is_integer() and abs(v) >= 100:
        return str(int(v))
    if places == 2 and 0 < abs(float(v)) < 0.1:
        places = 3
    s = f"{v:.{places}f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def rng(rec: dict) -> str:
    return f"{num(rec['min'])}–{num(rec['max'])}"


class Ctx:
    def __init__(self, metrics: dict, runs: dict, lang: str):
        self.m, self.runs, self.lang = metrics, runs, lang

    def t(self, en: str, zh: str) -> str:
        return en if self.lang == "en" else zh

    def med(self, key: str) -> str:
        r = self.m[key]
        if "median" not in r:
            return self.vals(key)
        return self.t(f"{num(r['median'])} s median ({rng(r)}; n={r['n']})",
                      f"中位数 {num(r['median'])} s（{rng(r)}；n={r['n']}）")

    def bmed(self, key: str) -> str:
        r = self.m[key]
        return self.t(f"**{num(r['median'])} s** median ({rng(r)}; n={r['n']})",
                      f"**{num(r['median'])} s** 中位数（{rng(r)}；n={r['n']}）")

    def vals(self, key: str) -> str:
        r = self.m[key]
        joined = ", ".join(num(v) for v in r["values"])
        return f"{joined} s (n={r['n']})" if self.lang == "en" else f"{joined} s（n={r['n']}）"

    def one(self, key: str) -> str:
        r = self.m[key]
        return self.t(f"{num(r['value'])} s (n=1)", f"{num(r['value'])} s（n=1）")

    def rng_n(self, key: str) -> str:
        r = self.m[key]
        return self.t(f"{rng(r)} s (n={r['n']})", f"{rng(r)} s（n={r['n']}）")


def table(head: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(head)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def quote(lines: list[str]) -> str:
    return "\n".join(f"> {ln}" for ln in lines) + "\n"


# --- README blocks --------------------------------------------------------------------------------

def comparison_details(c: Ctx) -> tuple[list[str], list[list[str]]]:
    m = c.m
    head = [c.t("Observed or documented", "实测或文档项"), "A · ACA Sandboxes",
            c.t("B · Foundry hosted agent", "B · Foundry hosted agent"),
            "C · AKS + agent-sandbox", "D · BYO-AKS", "E · E2B on Azure VM"]
    rows = [
        [c.t("**Create → ready, no external volume**", "**创建到就绪（不挂外部卷）**"),
         c.t(f"{c.bmed('a.nv.create')}, SDK create → `Running`, image + VNet + port (2026-10-18); image only "
             f"{c.med('a.attr.image')} (2026-10-09)",
             f"{c.bmed('a.nv.create')}，SDK 创建 → `Running`，镜像 + VNet + 端口（2026-10-18）；仅镜像 "
             f"{c.med('a.attr.image')}（2026-10-09）"),
         c.t(f"{c.bmed('b.sess.create')}, `POST .../endpoint/sessions` → `active`",
             f"{c.bmed('b.sess.create')}，`POST .../endpoint/sessions` → `active`"),
         c.t(f"{c.bmed('c.n.apply_ready')}, `kubectl apply` → Sandbox `Ready`",
             f"{c.bmed('c.n.apply_ready')}，`kubectl apply` → Sandbox `Ready`"),
         c.t(f"**{rng(m['d.pod_ready'])} s** (n={m['d.pod_ready']['n']}), pod created → `Ready` with the image cached; "
             f"{num(m['d.pod_ready_fresh']['min'])} s on a fresh node (n={m['d.pod_ready_fresh']['n']})",
             f"**{rng(m['d.pod_ready'])} s**（n={m['d.pod_ready']['n']}），镜像已缓存时 pod 创建 → `Ready`；新节点 "
             f"{num(m['d.pod_ready_fresh']['min'])} s（n={m['d.pod_ready_fresh']['n']}）"),
         c.t(f"{c.bmed('e.ci.create')} in the agent runs, `Sandbox.create()` → it returns; "
             f"{c.med('e.ci.d_sandbox_ready')} in the D-protocol new sessions; {rng(m['e.h.create'])} s over 20 creates "
             f"in turn, {m['e.h.create']['under_0_1']} of them under 0.1 s (2026-10-18)",
             f"agent 实测中 {c.bmed('e.ci.create')}，`Sandbox.create()` → 返回；D 口径新会话中 "
             f"{c.med('e.ci.d_sandbox_ready')}；连续创建 20 次 {rng(m['e.h.create'])} s，其中 "
             f"{m['e.h.create']['under_0_1']} 次低于 0.1 s（2026-10-18）")],
        [c.t("**Create → ready with a private Blob volume**<br>Optional: to keep data across suspend and resume, "
             "only C needs a volume",
             "**挂私有 Blob 卷时创建到就绪**<br>可选：想在挂起再恢复后保留数据，只有 C 必须挂卷"),
         c.t(f"{c.bmed('a.create')}, SDK create → `Running`, image + VNet + port + volume (2026-10-09)",
             f"{c.bmed('a.create')}，SDK 创建 → `Running`，镜像 + VNet + 端口 + 卷（2026-10-09）"),
         c.t("not needed: the platform restores `$HOME` and `/files`", "不需要：平台恢复 `$HOME` 和 `/files`"),
         c.t(f"**required** to keep data; {c.bmed('c.p.apply_ready')}, `kubectl apply` → Sandbox `Ready`; "
             f"2026-09-27: {c.one('c.apply_ready')}",
             f"**必须挂卷**才能保留数据；{c.bmed('c.p.apply_ready')}，`kubectl apply` → Sandbox `Ready`；"
             f"2026-09-27：{c.one('c.apply_ready')}"),
         c.t("not needed: the file written before `:stop` is kept", "不需要：`:stop` 前写入的文件会保留"),
         c.t("not needed: templates and snapshots stay on the host VM's OS disk",
             "不需要：模板和快照存在宿主 VM 的系统盘上")],
        [c.t("└ volume overhead", "└ 子项：挂卷附加耗时"),
         c.t(f"create → Running in one attribution run (2026-10-09): {rng(m['a.attr.small'])} s without the volume "
             f"(image, VNet or port alone; n={m['a.attr.small']['n']}), {rng(m['a.attr.vnetvolume'])} s with it "
             f"(image + VNet + volume; n={m['a.attr.vnetvolume']['n']})",
             f"同一次归因实测（2026-10-09）里 create → Running：不挂卷 {rng(m['a.attr.small'])} s（只加镜像、VNet 或端口；"
             f"n={m['a.attr.small']['n']}），挂卷 {rng(m['a.attr.vnetvolume'])} s（镜像 + VNet + 卷；"
             f"n={m['a.attr.vnetvolume']['n']}）"),
         c.t("none", "无"),
         c.t(f"none measurable: apply → Ready {c.med('c.p.apply_ready')} with the PVC, {c.med('c.n.apply_ready')} "
             "without",
             f"测不出差别：apply → Ready 挂 PVC {c.med('c.p.apply_ready')}，不挂 {c.med('c.n.apply_ready')}"),
         c.t("none", "无"),
         c.t("none", "无")],
        [c.t("**Resume → ready**", "**恢复到就绪**"),
         c.t(f"{c.bmed('a.nv.resume')}, resume → `Running`, without the Blob volume (2026-10-18)",
             f"{c.bmed('a.nv.resume')}，恢复 → `Running`，不挂 Blob 卷（2026-10-18）"),
         c.t("not timed separately: after `:stop` the session wakes on its next request, and that request's time "
             "includes a model turn; `$HOME` kept, no memory snapshot",
             "无法单独计时：`:stop` 后要等下一次请求才唤醒，这次请求的耗时包含一轮模型回答；保留 `$HOME`，无内存快照"),
         c.t(f"{c.bmed('c.p.resume_ready')}, `operatingMode: Running` → Sandbox `Ready`, with the Blob PVC that C needs "
             "to keep data; no memory snapshot",
             f"{c.bmed('c.p.resume_ready')}，`operatingMode: Running` → Sandbox `Ready`，挂着 C 保留数据所需的 Blob PVC；"
             "无内存快照"),
         c.t("not timed separately: after `:stop` the next request starts a new session pod, and that request's time "
             "includes a model turn; the file written before is kept, no memory snapshot",
             "无法单独计时：`:stop` 后由下一次请求启动新的会话 pod，这次请求的耗时包含一轮模型回答；之前写入的文件保留，"
             "无内存快照"),
         c.t(f"{c.bmed('e.ci.resume')} in the agent runs, `Sandbox.connect` → it returns; {rng(m['e.h.resume'])} s "
             f"over 20 cycles in turn with no model, {m['e.h.resume']['over_1']} of 20 over 1 s (2026-10-18)",
             f"agent 实测中 {c.bmed('e.ci.resume')}，`Sandbox.connect` → 返回；不调模型、连续 20 次循环 "
             f"{rng(m['e.h.resume'])} s，20 次里 {m['e.h.resume']['over_1']} 次超过 1 s（2026-10-18）")],
        [c.t("**Pod or sandbox lost → ready again**", "**pod 或沙箱丢失 → 重新就绪**"),
         c.t("not exercisable: no API injects host loss, and deleting a sandbox ends it",
             "无法从外部制造：没有注入宿主机故障的接口，删除沙箱即结束会话"),
         c.t("not exercisable: no API kills only a session's sandbox; explicit stop and resume are above",
             "无法从外部制造：没有只杀掉会话沙箱的接口；显式停止与恢复见上方"),
         c.t(c.med("c.p.pod_lost") + f"; with the Blob PVC that C needs to keep data, the conversation went on "
             f"{m['c.p.state_kept']['count']}/{m['c.p.state_kept']['n']} (2026-09-27: {num(m['c.pod_lost']['value'])} s, n=1)",
             c.med("c.p.pod_lost") + f"；挂着 C 保留数据所需的 Blob PVC，对话接着继续 "
             f"{m['c.p.state_kept']['count']}/{m['c.p.state_kept']['n']}（2026-09-27：{num(m['c.pod_lost']['value'])} s，n=1）"),
         c.t(c.one("d.pod_lost") + "; deleted session continuity not tested",
             c.one("d.pod_lost") + "；未测试被删会话的连续性"),
         c.t(c.one("e.ci.pod_lost") + ": after `kill`, a replacement sandbox until it is ready",
             c.one("e.ci.pod_lost") + "：`kill` 之后新建替代沙箱到就绪")],
    ]
    return head, rows


# GitHub removes <wbr> and inline CSS. Soft hyphens and split code spans keep the full table wrappable.
TABLE_WORD_BREAKS = {
    "Sandboxes": "Sand|boxes", "continuity": "con|tinuity", "documented": "docu|mented",
    "exercisable": "exer|cisable", "separately": "sep|arately", "snapshots": "snap|shots",
    "templates": "tem|plates",
}


def wrap_table_cell(text: str) -> str:
    parts = re.split(r"(`[^`\n]+`)", text)
    for i, part in enumerate(parts):
        if part.startswith("`"):
            code = part[1:-1]
            if len(code) >= 10:
                pieces = re.split(r"(?<=[ _/\-])(?=.)", code)
                parts[i] = "<br>".join(f"<code>{html.escape(piece)}</code>" for piece in pieces) + "&#8203;"
        else:
            part = re.sub(r"[A-Za-z]{9,}", lambda m: TABLE_WORD_BREAKS.get(m[0], m[0]).replace("|", "&shy;"), part)
            part = re.sub(r"(?<=\S)(?=（)|(?<=）)(?=）)", "&#8203;", part)
            parts[i] = re.sub(r"(?<=\d)–(?=\d)", "–&#8203;", part)
    return "".join(parts)


def unwrap_table_layout(text: str) -> str:
    text = re.sub(r"<code>.*?</code>(?:<br><code>.*?</code>)*", lambda m: "`" + html.unescape(
        re.sub(r"</?code>|<br>", "", m[0])) + "`", text)
    return text.replace("&shy;", "").replace("&#8203;", "")


def b_comparison(c: Ctx) -> str:
    head, rows = comparison_details(c)
    return table([wrap_table_cell(cell) for cell in head],
                 [[wrap_table_cell(cell) for cell in row] for row in rows]) + "\n"


CI_TEMPLATE_URL = "https://github.com/e2b-dev/code-interpreter/tree/162a2a25b6a5ea5806749a437b96cbdc5bfc087e/template"


def ci_passed(m: dict) -> int:
    ci = m["e.latest.ci"]
    return sum(v["passed"] == ci["n"] for v in ci["steps"].values())


def b_runs(c: Ctx) -> str:
    m = c.m
    old_en = "superseded executor (`base` template, `python3` through `commands.run`)"
    old_zh = "已被取代的执行方式（`base` 模板，用 `commands.run` 跑 `python3`）"
    head = [c.t("Run", "实测"), c.t("Path", "路径"), c.t("Scenario", "场景"), c.t("Result", "结果")]
    rows = [
        ["`a-2026-10-09-agent`", "A", c.t("full agent, 3 runs", "完整 agent，跑 3 次"),
         c.t(f"create → Running {num(m['a.create']['median'])} s with the Blob volume (median); vmm "
             f"`{m['a.vmm']['value']}` in {m['a.vmm']['count']}/{m['a.vmm']['n']}; history grew "
             f"{m['a.state_kept']['count']}/{m['a.state_kept']['n']}",
             f"挂 Blob 卷时创建 → Running {num(m['a.create']['median'])} s（中位数）；{m['a.vmm']['count']}/{m['a.vmm']['n']} 次 "
             f"vmm 为 `{m['a.vmm']['value']}`；历史持续增长 {m['a.state_kept']['count']}/{m['a.state_kept']['n']}")],
        ["`a-2026-10-09-attribution`", "A", c.t("create time by configuration", "按配置拆分创建耗时"),
         c.t(f"image, VNet or port alone {rng(m['a.attr.small'])} s; "
             f"adding the private Blob volume {rng(m['a.attr.vnetvolume'])} s",
             f"只加镜像、VNet 或端口 {rng(m['a.attr.small'])} s；"
             f"加私有 Blob 卷后 {rng(m['a.attr.vnetvolume'])} s")],
        ["`a-2026-10-09-late-mount`", "A", c.t("start first, mount Blob later", "先启动、后挂 Blob"),
         c.t(f"start {rng(m['a.late.create'])} s, then mount {rng(m['a.late.mount'])} s (n=2)",
             f"启动 {rng(m['a.late.create'])} s，再挂载 {rng(m['a.late.mount'])} s（n=2）")],
        ["`a-2026-09-27-setup`", "A", c.t("first deployment, wake on request", "首次部署，请求唤醒"),
         c.t(f"kernel `{m['a.isolation']['guest_kernel']}`; deploy and wake times in VALIDATION",
             f"内核 `{m['a.isolation']['guest_kernel']}`；部署和唤醒耗时见 VALIDATION")],
        ["`b-2026-09-27`", "B", c.t("deploy, new session, follow-ups; gateway model route",
                                   "部署、新会话、后续回答；gateway 模型路由"),
         c.t("deploy and answer times only; see VALIDATION", "只测了部署和回答耗时，见 VALIDATION")],
        ["`b-2026-10-15-sessions`", "B", c.t("explicit session lifecycle, 3 runs", "显式会话生命周期，跑 3 次"),
         c.t(f"create → active {num(m['b.sess.create']['median'])} s (median); file kept across stop and resume "
             f"{m['b.sess.state_kept']['files']}/{m['b.sess.state_kept']['n']}",
             f"创建 → active {num(m['b.sess.create']['median'])} s（中位数）；停止再恢复后文件保留 "
             f"{m['b.sess.state_kept']['files']}/{m['b.sess.state_kept']['n']}")],
        ["`c-2026-09-27`", "C", c.t("Kata Sandbox, pod loss, suspend/resume", "Kata Sandbox、pod 丢失、挂起/恢复"),
         c.t(f"pod kernel `{m['c.isolation']['guest_kernel']}` vs runc pod on the Ubuntu system pool "
             f"`{m['c.isolation']['system_node_kernel']}`; "
             f"pod lost → Ready {num(m['c.pod_lost']['value'])} s; resume → Ready {num(m['c.resume_ready']['value'])} s",
             f"pod 内核 `{m['c.isolation']['guest_kernel']}`，Ubuntu 系统节点池的 runc pod 内核 "
             f"`{m['c.isolation']['system_node_kernel']}`；"
             f"pod 丢失后 {num(m['c.pod_lost']['value'])} s 就绪；恢复后 {num(m['c.resume_ready']['value'])} s 就绪")],
        ["`d-2026-09-29-placement`", "D", c.t("where a session runs", "会话跑在哪里"),
         c.t(f"every session pod `{m['d.placement']['runtime_class']}` on your node pool after NSG rule "
             f"`{m['d.placement']['nsg_rule']}`",
             f"加 NSG 规则 `{m['d.placement']['nsg_rule']}` 后，每个会话 pod 都是 `{m['d.placement']['runtime_class']}`，"
             "跑在你的节点池上")],
        ["`d-2026-09-29-perf`", "D", c.t("cold, warm, 4 and 8 at once", "冷启动、后续、4 和 8 并发"),
         c.t(f"session pod created → Ready {rng(m['d.pod_ready'])} s with the image cached, "
             f"{num(m['d.pod_ready_fresh']['min'])} s on a fresh node",
             f"会话 pod 创建 → Ready：镜像已缓存 {rng(m['d.pod_ready'])} s，新节点 "
             f"{num(m['d.pod_ready_fresh']['min'])} s")],
        ["`d-2026-09-29-recovery`", "D", c.t("delete a session pod", "删除一个会话 pod"),
         c.t(f"replacement Ready in {num(m['d.pod_lost']['value'])} s; deleted session continuity not tested",
             f"{num(m['d.pod_lost']['value'])} s 后替代 pod 就绪；未测试被删会话的连续性")],
        ["`e-2026-10-14-agent`", "E", c.t(f"{old_en}: Prompt Agent + E2B, 3 runs + 20-run microbench",
                                          f"{old_zh}：Prompt Agent + E2B，跑 3 次 + 20 次微基准"),
         c.t(f"create {num(m['e.create']['median'])} s (on host {rng(m['e.bench20.create'])} s), "
             f"resume {num(m['e.resume']['median'])} s (medians); "
             f"VMM `{m['e.vmm']['value']}` in the host's process list; "
             f"history grew {m['e.state_kept']['count']}/{m['e.state_kept']['n']}",
             f"创建 {num(m['e.create']['median'])} s（本机 {rng(m['e.bench20.create'])} s）、"
             f"恢复 {num(m['e.resume']['median'])} s（中位数）；"
             f"宿主机进程列表里的 VMM 为 `{m['e.vmm']['value']}`；"
             f"历史持续增长 {m['e.state_kept']['count']}/{m['e.state_kept']['n']}")],
        ["`e-2026-10-14-perf`", "E", c.t(f"{old_en}: cold, warm, 4 and 8 at once, replacement",
                                         f"{old_zh}：冷启动、后续、4 和 8 并发、沙箱替换"),
         c.t(f"replacement Ready in {num(m['e.pod_lost']['value'])} s",
             f"{num(m['e.pod_lost']['value'])} s 后替代沙箱就绪")],
        ["`e-2026-10-15-latest`", "E", c.t(f"newest SDK ({old_en}); E2B's official code-interpreter template",
                                           f"最新 SDK（{old_zh}）；E2B 官方 code-interpreter 模板"),
         c.t(f"e2b 2.56.0: history grew {m['e.latest.sdk.state_kept']['count']}/{m['e.latest.sdk.state_kept']['n']}, "
             f"create {num(m['e.latest.sdk.create']['median'])} s, resume {num(m['e.latest.sdk.resume']['median'])} s "
             f"(medians); code-interpreter built only after raising three host defaults; "
             f"{ci_passed(m)}/{len(m['e.latest.ci']['steps'])} checks passed in "
             f"{m['e.latest.ci']['n']}/{m['e.latest.ci']['n']} runs, JavaScript failed in all",
             f"e2b 2.56.0：历史持续增长 {m['e.latest.sdk.state_kept']['count']}/{m['e.latest.sdk.state_kept']['n']}，"
             f"创建 {num(m['e.latest.sdk.create']['median'])} s、恢复 {num(m['e.latest.sdk.resume']['median'])} s"
             f"（中位数）；code-interpreter 调大三项宿主机默认值后才建成；"
             f"{len(m['e.latest.ci']['steps'])} 项检查中 {ci_passed(m)} 项在 {m['e.latest.ci']['n']}/"
             f"{m['e.latest.ci']['n']} 次都通过，JavaScript 每次都失败")],
        ["`e-2026-10-15-multi`", "E", c.t(f"{old_en}: two independent E2B hosts: second host, at once, idle, host stop",
                                          f"{old_zh}：两台独立 E2B 主机：第二台、并发、空闲容量、停一台"),
         c.t(f"host b: history grew {m['e.multi.b.state_kept']['count']}/{m['e.multi.b.state_kept']['n']}, "
             f"create {num(m['e.multi.b.create']['median'])} s; with host a's stack stopped, host b kept serving and "
             f"host a's paused sandbox could not resume on host b",
             f"host b：历史持续增长 {m['e.multi.b.state_kept']['count']}/{m['e.multi.b.state_kept']['n']}，"
             f"创建 {num(m['e.multi.b.create']['median'])} s；"
             "停掉 host a 的 E2B 后，host b 照常服务，host a 上暂停的沙箱在 host b 上恢复不了")],
        ["`e-2026-10-17-ci-agent`", "E", c.t("E2B Code Interpreter (`run_code`) in `code-interpreter-v1`, 3 runs; "
                                             "template build on host c",
                                             "E2B Code Interpreter（`run_code`）跑在 `code-interpreter-v1` 里，跑 3 次；"
                                             "在 host c 上构建模板"),
         c.t(f"create {num(m['e.ci.create']['median'])} s, resume {num(m['e.ci.resume']['median'])} s "
             f"(medians); history grew {m['e.ci.state_kept']['count']}/{m['e.ci.state_kept']['n']}; "
             f"template built on attempt {m['e.ci.build']['built_attempt']} of {m['e.ci.build']['n']}",
             f"创建 {num(m['e.ci.create']['median'])} s、恢复 {num(m['e.ci.resume']['median'])} s（中位数）；"
             f"历史持续增长 {m['e.ci.state_kept']['count']}/{m['e.ci.state_kept']['n']}；"
             f"模板在 {m['e.ci.build']['n']} 次尝试中的第 {m['e.ci.build']['built_attempt']} 次建成")],
        ["`e-2026-10-17-ci-perf`", "E", c.t("cold, warm, 4 and 8 at once, replacement, idle capacity; "
                                            "`code-interpreter-v1`",
                                            "冷启动、后续、4 和 8 并发、沙箱替换、空闲容量；`code-interpreter-v1`"),
         c.t(f"create {num(m['e.ci.d_sandbox_ready']['median'])} s (median of the {m['e.ci.d_sandbox_ready']['n']} "
             f"new sessions); replacement Ready in {num(m['e.ci.pod_lost']['value'])} s; idle at `HUGEPAGES=4096`: "
             f"{m['e.ci.idle']['ci_4096']} `code-interpreter-v1` or {m['e.ci.idle']['base_4096']} `base` sandboxes",
             f"创建 {num(m['e.ci.d_sandbox_ready']['median'])} s（{m['e.ci.d_sandbox_ready']['n']} 个新会话的中位数）；"
             f"{num(m['e.ci.pod_lost']['value'])} s 后替代沙箱就绪；`HUGEPAGES=4096` 时空闲可放 "
             f"{m['e.ci.idle']['ci_4096']} 个 `code-interpreter-v1` 或 {m['e.ci.idle']['base_4096']} 个 `base` 沙箱")],
        ["`e-2026-10-18-ci-host`", "E", c.t("host mode on host c, no model: 20 create → pause → resume cycles, hugepages "
                                            "per sandbox",
                                            "在 host c 上跑 host 模式、不调模型：20 次创建 → 暂停 → 恢复循环，每个沙箱占多少大页"),
         c.t(f"create {rng(m['e.h.create'])} s ({m['e.h.create']['under_0_1']} of 20 under 0.1 s), "
             f"resume {rng(m['e.h.resume'])} s; `x` kept {m['e.h.state_kept']['count']}/"
             f"{m['e.h.state_kept']['n']}; {m['e.h.reserve']['per_sandbox']} pages per sandbox, {m['e.h.reserve']['fit']} "
             "fit at `HUGEPAGES=4096`, the next refused",
             f"创建 {rng(m['e.h.create'])} s（20 次里 {m['e.h.create']['under_0_1']} 次低于 0.1 s），"
             f"恢复 {rng(m['e.h.resume'])} s；`x` 保留 {m['e.h.state_kept']['count']}/"
             f"{m['e.h.state_kept']['n']}；每个沙箱 {m['e.h.reserve']['per_sandbox']} 页，`HUGEPAGES=4096` 时放下 "
             f"{m['e.h.reserve']['fit']} 个，下一个被拒")],
        ["`e-2026-10-18-ci-multi`", "E", c.t("two hosts with `code-interpreter-v1`: agent runs on host a, at once on one "
                                             "and on two hosts, host stop",
                                             "两台主机跑 `code-interpreter-v1`：host a 上的 agent 实测、单台和两台并发、停一台"),
         c.t(f"host a: history grew {m['e.m.a.state_kept']['count']}/{m['e.m.a.state_kept']['n']}, create "
             f"{', '.join(num(v) for v in m['e.m.a.create']['values'])} s, resume {num(m['e.m.a.resume']['median'])} s "
             "(median); with host a's stack stopped, host c "
             "kept serving and host a's paused sandbox could not resume on host c",
             f"host a：历史持续增长 {m['e.m.a.state_kept']['count']}/{m['e.m.a.state_kept']['n']}，创建 "
             f"{', '.join(num(v) for v in m['e.m.a.create']['values'])} s，恢复 {num(m['e.m.a.resume']['median'])} s"
             "（中位数）；停掉 host a 的 E2B 后，host c 照常服务，"
             "host a 上暂停的沙箱在 host c 上恢复不了")],
        ["`e-2026-10-18-ci-scale`", "E", c.t("new `Standard_D8ds_v7` with `HUGEPAGES=8448`: README step 7, cold, warm, 4 and 8 "
                                             "at once, replacement, hugepages",
                                             "新建 `Standard_D8ds_v7`，`HUGEPAGES=8448`：README 第 7 步、冷启动、后续、4 和 8 并发、"
                                             "沙箱替换、大页"),
         c.t(f"template built on the first attempt; replacement Ready in {num(m['e.s.pod_lost']['value'])} s; "
             f"{m['e.s.reserve']['fit']} sandboxes fit at `HUGEPAGES=8448`, the next refused",
             f"模板第一次就建成；{num(m['e.s.pod_lost']['value'])} s 后替代沙箱就绪；`HUGEPAGES=8448` 时放下 "
             f"{m['e.s.reserve']['fit']} 个沙箱，下一个被拒")],
        *runs_20261018(c),
    ]
    order = json.loads((ROOT / "evidence" / "claims.json").read_text(encoding="utf-8")).get("run_order")
    if not isinstance(order, list) or not all(isinstance(rid, str) for rid in order):
        raise RenderError("evidence/claims.json: run_order must be a list of run IDs")
    rows_by_id = {row[0].strip("`"): row for row in rows}
    if len(order) != len(rows) or set(order) != set(rows_by_id):
        raise RenderError("evidence/claims.json: run_order must list every results-table run exactly once")
    return table(head, [rows_by_id[rid] for rid in order])


# --- VALIDATION blocks ----------------------------------------------------------------------------

A_AGENT, A_NV = "a-aca-2026-10-09/agent-runs.json", "a-aca-2026-10-18/agent-novolume.json"


def v_a_agent_input(c: Ctx, rel: str = A_AGENT) -> str:
    a = c.runs[rel]
    i = a["inputs"]
    spec = a["sandbox_spec"]
    lines = [c.t(f"Configuration: {a['configuration']}; cpu {spec['cpu']}, memory {spec['memory']}, "
                 f"auto-suspend {spec['auto_suspend_seconds']} s in {spec['auto_suspend_mode']} mode.",
                 f"配置：{a['configuration']}；cpu {spec['cpu']}，memory {spec['memory']}，"
                 f"{spec['auto_suspend_mode']} 模式下空闲 {spec['auto_suspend_seconds']} s 自动挂起。"),
             c.t(f"Request: {i['request']}.", f"请求：{i['request']}。"),
             c.t(f"First answer: \"{i['first']}\"", f"第一个问题：\"{i['first']}\""),
             c.t(f"Every later turn: \"{i['warm']}\"", f"之后每一轮：\"{i['warm']}\"")]
    return quote(lines)


def v_a_agent(c: Ctx, rel: str = A_AGENT, prefix: str = "a") -> str:
    a = c.runs[rel]
    wait = "ready_wait_s" in a["runs"][0]["first"]
    head = [c.t("Run", "次"), c.t("Create → Running", "创建 → Running")] + \
        ([c.t("Wait for the agent", "等 agent 就绪")] if wait else []) + \
        [c.t("First answer", "第一个回答"), c.t("Cold", "冷启动"), c.t("Follow-up", "后续"), c.t("Stop", "停止"),
         c.t("Resume", "恢复"), c.t("Answer after resume", "恢复后回答"), c.t("Resume → answer", "恢复 → 回答"),
         c.t("Answers", "回答")]
    rows = []
    for r in a["runs"]:
        answers = " / ".join(str(r[k]["response"]) for k in ("first", "warm", "first_after_resume", "warm_after_resume"))
        rows.append([str(r["run"]), num(r["create_total_s"])] + ([num(r["first"]["ready_wait_s"])] if wait else []) +
                    [num(r["first"]["guest_s"]), num(r["cold_total_s"]), num(r["warm"]["guest_s"]), num(r["stop_total_s"]),
                     num(r["resume_total_s"]), num(r["first_after_resume"]["guest_s"]), num(r["recover_total_s"]),
                     answers])
    m = c.m
    keys = ["create"] + (["ready_wait"] if wait else []) + ["first_answer", "cold", "warm", "stop", "resume",
                                                           "first_after_resume", "recover"]
    rows.append([c.t("**median**", "**中位数**")] + [f"**{num(m[f'{prefix}.{k}']['median'])}**" for k in keys] + [""])
    note = c.t("Seconds; answers are the agent's replies in order. Totals are added per run before taking the median; "
               "the independently reported column medians must not be added.",
               "单位为秒；最后一列是 agent 依次给出的回答。总耗时先在每次实测内相加，再取中位数；"
               "不能把各列独立统计的中位数相加。")
    if wait:
        late = m[f"{prefix}.ready_wait_resume"]["max"]
        en_after = "0 s in every run" if not late else f"at most {num(late)} s"
        zh_after = "每次都是 0 s" if not late else f"最长 {num(late)} s"
        note += c.t(f" Wait for the agent: inside the sandbox, until it accepted a connection on port 8080, before the "
                    f"first request; part of Cold. The same wait before the answer after resume was {en_after} and is "
                    "part of Resume → answer.",
                    f"等 agent 就绪：在沙箱内等到 agent 在 8080 端口接受连接，再发第一个请求，计入冷启动。恢复后回答前的同样"
                    f"等待{zh_after}，计入恢复 → 回答。")
    return note + "\n\n" + table(head, rows, ["---"] + ["---:"] * len(keys) + ["---"])


def v_a_attr(c: Ctx) -> str:
    att = c.runs["a-aca-2026-10-09/create-attribution.json"]
    head = [c.t("Configuration", "配置"), c.t("What is added", "加了什么"), "n", c.t("Create → Running, s", "创建 → Running，s"),
            c.t("Median", "中位数")]
    rows = []
    for cfg in ("builtin", "image", "vnet", "port", "vnetvolume", "full"):
        r = c.m[f"a.attr.{cfg}"]
        rows.append([f"`{cfg}`", att["configs"][cfg], str(r["n"]), ", ".join(num(v) for v in r["values"]),
                     num(r["median"]) if "median" in r else c.t("— (n=2)", "—（n=2）")])
    s = c.m["a.attr.process_survived"]
    tail = c.t(f"\nGuest uptime after resume exceeded its pre-stop value in {s['count']} of {s['n']} sandboxes. "
               "This observation alone does not prove process-memory continuity.\n",
               f"\n{s['n']} 个沙箱中有 {s['count']} 个恢复后的 guest uptime 大于停止前的值。"
               "这个观察本身不能证明进程内存的连续性。\n")
    return table(head, rows, ["---", "---", "---:", "---", "---:"]) + tail


def v_a_late(c: Ctx) -> str:
    late = c.runs["a-aca-2026-10-09/late-mount.json"]
    head = [c.t("Run", "次"), c.t("Create without volume, s", "不带卷创建，s"), c.t("Attach Blob volume, s", "挂载 Blob 卷，s"),
            c.t("State after mount", "挂载后状态")]
    rows = [[str(r["run"]), num(r["create_total_s"]), num(r["add_volume_mount_s"]), r["state_after_mount"]]
            for r in late["runs"]]
    return table(head, rows, ["---", "---:", "---:", "---"])


def v_a_setup(c: Ctx) -> str:
    s = c.runs["a-aca-2026-09-27/setup-and-wake.json"]
    ev = s["events"]
    names = {"vnet_connection": c.t("VNet connection", "VNet 连接"), "disk_image": c.t("disk image from ACR", "从 ACR 构建磁盘镜像"),
             "volume": c.t("Blob volume", "Blob 卷"), "sandbox_create": c.t("sandbox create", "创建沙箱"),
             "reflect_in_sandbox": c.t("answer, timed inside", "回答（沙箱内计时）"),
             "suspend_sdk": c.t("stop (memory snapshot)", "停止（内存快照）"), "resume_sdk": c.t("resume", "恢复"),
             "reflect_after_resume": c.t("answer after resume", "恢复后回答"),
             "reflect_warm_after_resume": c.t("next answer", "下一轮回答"), "suspend_sdk_2": c.t("stop again", "再次停止"),
             "wake_on_http_request": c.t("HTTP request to the stopped sandbox", "向已停止的沙箱发 HTTP 请求"),
             "reflect_external_warm": c.t("next HTTP request", "下一个 HTTP 请求")}
    rows = [[c.t("sandbox group (Bicep)", "sandbox group（Bicep）"), "", num(s["sandbox_group_deploy_seconds"]), ""]]
    for e in ev:
        q = e.get("question") or e.get("query") or ""
        resp = e.get("response", e.get("state", ""))
        rows.append([names[e["step"]], f"\"{q}\"" if q else "", num(e["seconds"]), str(resp)])
    head = [c.t("Step", "步骤"), c.t("Input", "输入"), "s", c.t("Reply or state", "回答或状态")]
    iso = s["isolation_check"]
    tail = c.t(f"\nRead inside the sandbox: vmm_type `{iso['vmm_type']}`, kernel `{iso['guest_kernel']}`, "
               f"{iso['vcpus']} vCPU.\n",
               f"\n在沙箱内读到：vmm_type `{iso['vmm_type']}`，内核 `{iso['guest_kernel']}`，{iso['vcpus']} 个 vCPU。\n")
    return table(head, rows, ["---", "---", "---:", "---"]) + tail


def v_b(c: Ctx) -> str:
    b = c.runs["b-hosted-agent-2026-09-27/run.json"]
    head = [c.t("Step", "步骤"), c.t("Route to the model", "到模型的路由"), "s"]
    g, d = b["invoke_v1_gateway"], b["invoke_v2_direct"]
    rows = [["`azd ai agent init`", "", num(b["azd_ai_agent_init_seconds"])],
            [c.t("`azd deploy`, version 1", "`azd deploy`，版本 1"), "gateway", num(b["deploy_v1_gateway"]["seconds"])],
            [c.t("new session → answer", "新会话 → 回答"), "gateway", num(g["new_session_seconds"])],
            [c.t("follow-up turns", "后续回答"), "gateway", ", ".join(num(v) for v in g["warm_seconds"])],
            [c.t("`azd deploy`, version 2", "`azd deploy`，版本 2"), "direct", num(b["deploy_v2_direct"]["seconds"])],
            [c.t("new session → answer", "新会话 → 回答"), "direct", num(d["new_session_seconds"])],
            [c.t("follow-up turns", "后续回答"), "direct", ", ".join(num(v) for v in d["warm_seconds"])]]
    return table(head, rows, ["---", "---", "---:"])


def v_b_sessions(c: Ctx) -> str:
    return session_table(c, "b-sessions-2026-10-15/sessions.json", "b.sess")


def session_table(c: Ctx, rel: str, prefix: str) -> str:
    s = c.runs[rel]
    head = [c.t("Run", "次"), c.t("Create → active", "创建 → active"), c.t("First answer", "第一个回答"),
            c.t("Follow-up", "后续"), c.t("Stop → idle", "停止 → idle"), c.t("Answer after stop", "停止后回答"),
            c.t("Next answer", "再下一轮"), c.t("File kept", "文件保留"), c.t("Answers", "回答")]
    rows = []
    for r in s["runs"]:
        create = r["create"]["total_s"] + ((r["create"].get("wait_active") or {}).get("seconds") or 0)
        stop = r["stop"]["total_s"] + r["stop"]["wait_stopped"]["seconds"]
        answers = " / ".join(str(r[k]["response"]) for k in ("first", "warm", "resume", "warm_after_resume"))
        rows.append([str(r["run"]), num(round(create, 3)), num(r["first"]["total_s"]), num(r["warm"]["total_s"]),
                     num(round(stop, 3)), num(r["resume"]["total_s"]), num(r["warm_after_resume"]["total_s"]),
                     c.t("yes", "是") if r["marker_after_resume"]["matches"] is True else c.t("no", "否"), answers])
    m = c.m
    rows.append([c.t("**median**", "**中位数**")] + [f"**{num(m[k]['median'])}**" for k in
                 (f"{prefix}.create", f"{prefix}.first", f"{prefix}.warm", f"{prefix}.stop", f"{prefix}.resume",
                  f"{prefix}.warm_after_resume")] + ["", ""])
    codes: dict[str, int] = {}
    for e in s["http_log"]:
        codes[str(e["status"])] = codes.get(str(e["status"]), 0) + 1
    retried = sum(len(v["attempts"]) > 1 for r in s["runs"] for v in r.values()
                  if isinstance(v, dict) and isinstance(v.get("attempts"), list))
    en_codes = ", ".join(f"{code} ×{n}" for code, n in sorted(codes.items()))
    zh_codes = "、".join(f"{code} ×{n}" for code, n in sorted(codes.items()))
    tail = c.t(f"\n{len(s['http_log'])} HTTP attempts: {en_codes}; "
               + ("none retried" if not retried else f"{retried} calls retried") + ".\n",
               f"\n共 {len(s['http_log'])} 次 HTTP 尝试：{zh_codes}；"
               + ("没有重试" if not retried else f"{retried} 次调用有重试") + "。\n")
    return c.t("Seconds, client wall time; the last column is the agent's replies in order.\n\n",
               "单位为秒，客户端计时；最后一列是 agent 依次给出的回答。\n\n") + \
        table(head, rows, ["---"] + ["---:"] * 6 + ["---", "---"]) + tail


def v_c(c: Ctx) -> str:
    r = c.runs["c-aks-agent-sandbox-2026-09-27/run.json"]
    head = [c.t("Step", "步骤"), c.t("Observed", "观察结果")]
    k = r["kata_isolation"]
    rows = [[c.t("`prepare-agent-sandbox.sh`", "`prepare-agent-sandbox.sh`"), f"{num(r['prepare_script_seconds'])} s"],
            [c.t("Blob private link", "Blob 私有链接"), f"{num(r['storage_private_link_seconds'])} s"],
            [c.t("`kubectl apply` → Sandbox Ready", "`kubectl apply` → Sandbox Ready"),
             f"{num(r['sandbox_apply_to_ready_seconds'])} s"],
            [c.t("first answer, second answer", "第一个、第二个回答"),
             f"{num(r['reflect']['first_seconds'])} s, {num(r['reflect']['second_seconds'])} s"
             + c.t(f"; context remembered: {str(r['reflect']['remembers_context']).lower()}",
                   f"；记住上下文：{str(r['reflect']['remembers_context']).lower()}")],
            [c.t("`uname -r` in the Sandbox pod / in a runc pod", "Sandbox pod / runc pod 中的 `uname -r`"),
             f"`{k['sandbox_kernel']}` / `{k['runc_pod_kernel']}`"],
            [c.t("delete the pod → Ready again", "删除 pod → 重新就绪"),
             f"{num(r['pod_delete_to_ready_seconds'])} s" + c.t(f", state restored: {str(r['state_restored_after_pod_delete']).lower()}",
                                                               f"，状态恢复：{str(r['state_restored_after_pod_delete']).lower()}")],
            [c.t("`operatingMode: Suspended` → pod gone", "`operatingMode: Suspended` → pod 消失"),
             f"{num(r['suspend']['pod_gone_seconds'])} s" + c.t(f"; health while suspended HTTP {r['suspend']['health_while_suspended']}",
                                                               f"；挂起期间健康检查 HTTP {r['suspend']['health_while_suspended']}")],
            [c.t("`operatingMode: Running` → Ready / LB 200 / answer", "`operatingMode: Running` → Ready / LB 200 / 回答"),
             f"{num(r['resume']['sandbox_ready_seconds'])} / {num(r['resume']['lb_health_200_seconds'])} / "
             f"{num(r['resume']['first_answer_seconds'])} s"]]
    return table(head, rows)


def v_d_input(c: Ctx) -> str:
    b = c.runs["d-byoaks-2026-09-29/bench.json"]
    i = b["inputs"]
    rtt = c.m["d.client_rtt"]
    lines = [c.t(f"Request: {b['request']}.", f"请求：{b['request']}。"),
             c.t(f"New sessions: \"{i['cold']}\" (n = 1, 2, 3)", f"新会话：\"{i['cold']}\"（n = 1、2、3）"),
             c.t(f"One session: \"{i['warm_first']}\", then \"{i['warm_followup_1']}\", then \"{i['warm_followup_n']}\" "
                 "(n = 1, 2)",
                 f"同一会话：\"{i['warm_first']}\"，然后 \"{i['warm_followup_1']}\"，再 \"{i['warm_followup_n']}\"（n = 1、2）"),
             c.t(f"At once, per session: \"{i['scale_turn1']}\", then \"{i['scale_turn2']}\"",
                 f"并发时每个会话：\"{i['scale_turn1']}\"，然后 \"{i['scale_turn2']}\""),
             c.t(f"Client: {b['client']}; recorded round trip {rng(rtt)} ms.",
                 f"客户端：{b['client']}；记录到的往返时延 {rng(rtt)} ms。")]
    return quote(lines)


def v_d_cold(c: Ctx) -> str:
    b = c.runs["d-byoaks-2026-09-29/bench.json"]
    e = c.runs["d-byoaks-2026-09-29/cold-early.json"]
    fresh = set(e["first_pod_on_fresh_node_s"])
    head = [c.t("Start (UTC)", "开始（UTC）"), c.t("Node", "节点"), c.t("Answer", "回答"), c.t("New session → answer, s", "新会话 → 回答，s"),
            c.t("Pod created → Ready, s", "pod 创建 → Ready，s")]
    rows = []
    for r in e["runs"]:
        rows.append([r["start_utc"], c.t("no image yet", "尚无镜像") if r["wall_s"] in fresh else c.t("image cached", "镜像已缓存"),
                     r["answer"], num(r["wall_s"]), c.t("not paired", "未配对")])
    for t in b["cold_turns"]:
        rows.append([t["started_utc"][:8], c.t("image cached", "镜像已缓存"), t["answer"], num(t["wall_s"]),
                     num(t["pod"]["create_to_ready_s"])])
    w = b["warm_turns"][0]
    rows.append([w["started_utc"][:8], c.t("image cached", "镜像已缓存"), w["answer"], num(w["wall_s"]),
                 c.t("not paired", "未配对")])
    m = c.m
    tail = c.t(f"\nWith the image cached: {c.med('d.cold')}. First pod on a node without the image: "
               f"{c.vals('d.cold_fresh')}. Session pods created → Ready across the run: "
               f"{', '.join(num(v) for v in m['d.pod_ready']['values'])} s with the image cached, "
               f"{', '.join(num(v) for v in m['d.pod_ready_fresh']['values'])} s on a fresh node.\n",
               f"\n镜像已缓存：{c.med('d.cold')}。节点上还没有镜像时的第一个 pod：{c.vals('d.cold_fresh')}。"
               f"本次所有会话 pod 从创建到 Ready：镜像已缓存 {', '.join(num(v) for v in m['d.pod_ready']['values'])} s，"
               f"新节点 {', '.join(num(v) for v in m['d.pod_ready_fresh']['values'])} s。\n")
    return table(head, rows, ["---", "---", "---", "---:", "---:"]) + tail


def v_d_warm(c: Ctx) -> str:
    b = c.runs["d-byoaks-2026-09-29/bench.json"]
    head = [c.t("Turn", "轮次"), c.t("Answer", "回答"), c.t("Client wall, s", "客户端耗时，s")]
    rows = []
    for k, t in enumerate(b["warm_turns"]):
        label = c.t("1 (new session)", "1（新会话）") if k == 0 else str(k + 1)
        rows.append([label, t["answer"], num(t["wall_s"])])
    tail = c.t(f"\nThe same model called directly from the same client, without an agent: {c.med('d.model_direct')}. "
               "This baseline used low reasoning effort; it is not a matched agent-configuration benchmark.\n",
               f"\n同一客户端不经 agent 直接调用同一个模型：{c.med('d.model_direct')}。"
               "该基线使用 low reasoning effort，不是与 agent 配置完全匹配的基准测试。\n")
    return table(head, rows, ["---", "---", "---:"]) + tail


def v_d_scale(c: Ctx) -> str:
    head = [c.t("At once", "并发"), c.t("Answered", "回答"), c.t("Refused", "拒绝"), c.t("Refusal", "拒绝原因"),
            c.t("Refused after, s", "拒绝耗时，s")]
    rows = []
    for n in (4, 8):
        r = c.m[f"d.c{n}"]
        rows.append([str(n), f"{r['answered']}/{r['n']}" + (f" ({', '.join(num(v) for v in r['answered_s'])} s)" if r["answered_s"] else ""),
                     f"{r['refused']}/{r['n']}", ", ".join(f"`{x}`" for x in r["refused_codes"]),
                     f"{num(r['refused_after_s'][0])}–{num(r['refused_after_s'][1])}"])
    cap = c.m["d.capacity"]
    tail = c.t(f"\nWhen the refusals began, {cap['max_sessions']} session pods were alive on {cap['n']} "
               f"`{cap['node_sku']}` nodes, at most {cap['per_node']} per node, and no node had 500m CPU left.\n",
               f"\n开始拒绝时，{cap['n']} 个 `{cap['node_sku']}` 节点上共有 {cap['max_sessions']} 个会话 pod，"
               f"单节点最多 {cap['per_node']} 个，没有节点还剩 500m CPU。\n")
    return table(head, rows, ["---:", "---", "---", "---", "---:"]) + tail


def v_d_recovery(c: Ctx) -> str:
    r = c.runs["d-byoaks-2026-09-29/recovery.json"]
    f = r["followup_after_kill"]
    rows = [[c.t("session pod deleted (UTC)", "删除会话 pod（UTC）"), r["pod_deleted_utc"]],
            [c.t("replacement pod Ready after", "替代 pod 就绪耗时"), f"{num(r['replacement_ready_s'])} s"],
            [c.t("follow-up on a different live session", "另一个存活会话的后续回答"),
             f"HTTP {f['http']}, {num(f['wall_s'])} s, \"{f['answer']}\"" +
             c.t(f", that session kept its state: {str(f['state_preserved']).lower()}",
                 f"，该会话保留了自身状态：{str(f['state_preserved']).lower()}")]]
    return table([c.t("Step", "步骤"), c.t("Observed", "观察结果")], rows)


def v_d_placement(c: Ctx) -> str:
    p = c.runs["d-byoaks-2026-09-29/placement.json"]
    pod, rule = p["session_pod"], p["nsg_rule_added"]
    flows = "; ".join(f"{f['source']} → :443 {f['access']}" for f in p["ip_flow_verify_before_fix"])
    req = pod["containers"][0]["resources"]["requests"]
    rows = [[c.t("IP flow verify before the fix", "修复前 IP flow verify"), flows],
            [c.t("rule added to the subnet NSG", "在子网 NSG 上新增的规则"),
             f"`{rule['name']}`, {c.t('priority', '优先级')} {rule['priority']}, {rule['direction']} {rule['access']} "
             f"{rule['protocol']} {rule['source']} → {rule['destination']}:{rule['destination_port']}"],
            [c.t("first session after the fix", "修复后的第一个会话"),
             f"HTTP {p['first_session_after_fix']['http']}, \"{p['first_session_after_fix']['answer']}\", "
             f"{p['first_session_after_fix']['client_wall_seconds']} s {c.t('on the client', '客户端耗时')}"],
            [c.t("its session pod", "它的会话 pod"),
             f"runtimeClassName `{pod['runtimeClassName']}`, {c.t('owner', '属主')} {pod['owner_kind']}, "
             f"{c.t('requests', 'requests')} {req['cpu']} / {req['memory']}, overhead {pod['overhead']['memory']}, "
             f"{c.t('init containers', 'init 容器')} {', '.join(pod['init_containers'])}"],
            [c.t("scheduled → Ready", "调度 → Ready"), f"{c.m['d.placement']['scheduled_to_ready_s']} s"],
            [c.t("session pods at the 18:15 recheck", "18:15 复查时的会话 pod"),
             c.t(f"{c.m['d.placement']['kata_pods_at_recheck']} of {len(p['session_pods_at_recheck'])} Running with "
                 "`kata-vm-isolation`",
                 f"{len(p['session_pods_at_recheck'])} 个中 {c.m['d.placement']['kata_pods_at_recheck']} 个为 Running 且是 "
                 "`kata-vm-isolation`")],
            [c.t("two turns at the recheck", "复查时的两轮对话"),
             " → ".join(f"\"{t['answer']}\"" for t in p["recheck_two_turns"])],
            [c.t("kernel of the Kata pool nodes", "Kata 节点池的节点内核"), f"`{c.m['d.node_kernel']['value']}`"]]
    return table([c.t("Check", "检查项"), c.t("Observed", "观察结果")], rows)


def v_e_agent_input(c: Ctx) -> str:
    ea = c.runs["e-e2b-2026-10-14/agent-runs.json"]
    i = ea["inputs"]
    iso = ea["isolation_check"]
    lines = [c.t(f"Host VM: {ea['host_vm_sku']}; E2B `base` template ({iso['vcpus']} vCPU, {iso['memory_mb']} MB, "
                 "Firecracker memory + disk snapshot via `beta_pause`).",
                 f"宿主机 VM：{ea['host_vm_sku']}；E2B `base` 模板（{iso['vcpus']} vCPU，{iso['memory_mb']} MB，"
                 "经 `beta_pause` 保存 Firecracker 内存+磁盘快照）。"),
             c.t(f"Versions: E2B Python SDK `{ea['sdk']}`; E2B Embed runtime `{ea['e2b_runtime_commit'][:7]}`.",
                 f"版本：E2B Python SDK `{ea['sdk']}`；E2B Embed 运行时 `{ea['e2b_runtime_commit'][:7]}`。"),
             c.t(f"Request: {i['request']}.", f"请求：{i['request']}。"),
             c.t(f"Agent instructions: \"{ea['agent']['instructions']}\"", f"Agent 指令：\"{ea['agent']['instructions']}\""),
             c.t(f"Its one function tool `{ea['agent']['tools'][0]['name']}(python_code)`: "
                 f"\"{ea['agent']['tools'][0]['description']}\"",
                 f"唯一的函数工具 `{ea['agent']['tools'][0]['name']}(python_code)`："
                 f"\"{ea['agent']['tools'][0]['description']}\""),
             c.t(f"First answer: \"{i['first']}\"", f"第一个问题：\"{i['first']}\""),
             c.t(f"Every later turn: \"{i['warm']}\"", f"之后每一轮：\"{i['warm']}\"")]
    return quote(lines)


def v_e_agent(c: Ctx) -> str:
    ea = c.runs["e-e2b-2026-10-14/agent-runs.json"]
    head = [c.t("Run", "次"), c.t("Create → Running", "创建 → Running"), c.t("First answer", "第一个回答"),
            c.t("Cold", "冷启动"), c.t("Follow-up", "后续"), c.t("Pause", "暂停"), c.t("Resume", "恢复"),
            c.t("Answer after resume", "恢复后回答"), c.t("Resume → answer", "恢复 → 回答"), c.t("Answers", "回答")]
    rows = []
    for r in ea["runs"]:
        answers = " / ".join(str(r[k]["response"]) for k in ("first", "warm", "first_after_resume", "warm_after_resume"))
        rows.append([str(r["run"]), num(r["create_total_s"]), num(r["first"]["guest_s"]), num(r["cold_total_s"]),
                     num(r["warm"]["guest_s"]), num(r["stop_total_s"]), num(r["resume_total_s"]),
                     num(r["first_after_resume"]["guest_s"]), num(r["recover_total_s"]), answers])
    m = c.m
    rows.append([c.t("**median**", "**中位数**")] + [f"**{num(m[k]['median'])}**" for k in
                 ("e.create", "e.first_answer", "e.cold", "e.warm", "e.stop", "e.resume", "e.first_after_resume",
                  "e.recover")] + [""])
    iso = ea["isolation_check"]
    tail = c.t(f"\nOn-host 20-sandbox microbenchmark (zero WAN RTT): create {c.rng_n('e.bench20.create')}, "
               f"warm command {c.rng_n('e.bench20.warm_cmd')}, pause {c.rng_n('e.bench20.pause')}, "
               f"resume {c.rng_n('e.bench20.resume')}. VMM `{iso['vmm_type']}` `{iso['firecracker_version']}` "
               f"from the host's process list; guest kernel `{iso['guest_kernel']}` from `uname -r` inside the "
               f"sandbox; host kernel `{iso['host_kernel']}`.\n",
               f"\n本机 20 次沙箱微基准（零跨网 RTT）：创建 {c.rng_n('e.bench20.create')}，"
               f"热命令 {c.rng_n('e.bench20.warm_cmd')}，暂停 {c.rng_n('e.bench20.pause')}，"
               f"恢复 {c.rng_n('e.bench20.resume')}。VMM `{iso['vmm_type']}` `{iso['firecracker_version']}` "
               f"来自宿主机进程列表；guest 内核 `{iso['guest_kernel']}` 来自沙箱内的 `uname -r`；"
               f"宿主机内核 `{iso['host_kernel']}`。\n")
    return table(head, rows, ["---"] + ["---:"] * 8 + ["---"]) + tail


def v_e_perf_input(c: Ctx) -> str:
    ep = c.runs["e-e2b-2026-10-14/perf.json"]
    i = ep["inputs"]
    driver = next(s for s in ep["_sources"] if s["role"] == "benchmark driver")
    w = i["warm"]
    lines = [c.t(f"Request: {ep['request']}; agent instructions and tool as in `e-2026-10-14-agent`.",
                 f"请求：{ep['request']}；agent 指令和工具与 `e-2026-10-14-agent` 相同。"),
             c.t(f"New sessions: \"{i['cold']}\" (n = 1, 2, 3, 4, 5)", f"新会话：\"{i['cold']}\"（n = 1、2、3、4、5）"),
             c.t(f"One session: \"{w[0]}\", then \"{w[1]}\", then \"{w[2]}\", then \"{w[3]}\"",
                 f"同一会话：\"{w[0]}\"，然后 \"{w[1]}\"，再 \"{w[2]}\"，再 \"{w[3]}\""),
             c.t(f"At once, per session: \"{i['scale'][0]}\", then \"{i['scale'][1]}\"",
                 f"并发时每个会话：\"{i['scale'][0]}\"，然后 \"{i['scale'][1]}\""),
             c.t(f"Replacement: a live session ran \"{i['recovery'][0]}\"; after another sandbox was killed and "
                 f"replaced, it ran \"{i['recovery'][1]}\"",
                 f"替换：一个存活会话先发 \"{i['recovery'][0]}\"；另一个沙箱被删除并替换后，再发 \"{i['recovery'][1]}\""),
             c.t(f"Driver: the author's private script `{driver['file']}` (SHA-256 `{driver['sha256'][:12]}…` in the "
                 "record); `tools/measure_e2b.py perf` sends the same prompts with the same agent and clocks.",
                 f"驱动：作者的私有脚本 `{driver['file']}`（SHA-256 `{driver['sha256'][:12]}…` 记在记录里）；"
                 "`tools/measure_e2b.py perf` 用同样的 agent、问题和计时。")]
    return quote(lines)


def v_e_cold(c: Ctx) -> str:
    ep = c.runs["e-e2b-2026-10-14/perf.json"]
    head = [c.t("Start (UTC)", "开始（UTC）"), c.t("Answer", "回答"),
            c.t("Sandbox created → Ready, s", "沙箱创建 → Ready，s"),
            c.t("Agent turn, s", "Agent 轮次，s"), c.t("New session → answer, s", "新会话 → 回答，s")]
    rows = [[t["started_utc"][:8], t["answer"], num(t["sandbox_create_to_ready_s"]),
             num(t["agent_turn_s"]), num(t["wall_s"])] for t in ep["cold_turns"]]
    tail = c.t(f"\nAcross the 5 cold runs: sandbox create → Ready {c.med('e.d_sandbox_ready')}, "
               f"Prompt Agent turn {c.med('e.d_agent_turn')}, end-to-end new session → answer {c.med('e.d_cold')}.\n",
               f"\n5 次冷启动汇总：沙箱创建 → Ready {c.med('e.d_sandbox_ready')}，"
               f"Prompt Agent 轮次 {c.med('e.d_agent_turn')}，端到端新会话 → 回答 {c.med('e.d_cold')}。\n")
    return table(head, rows, ["---", "---", "---:", "---:", "---:"]) + tail


def v_e_warm(c: Ctx) -> str:
    ep = c.runs["e-e2b-2026-10-14/perf.json"]
    head = [c.t("Turn", "轮次"), c.t("Answer", "回答"), c.t("Sandbox tool, s", "沙箱工具耗时，s"),
            c.t("Turn, s", "轮次耗时，s")]
    rows = []
    for k, t in enumerate(ep["warm_turns"]):
        label = c.t("1 (new session; includes the sandbox create)", "1（新会话；含沙箱创建）") if k == 0 else str(k + 1)
        rows.append([label, t["answer"], num(t["sandbox_tool_s"]), num(t["wall_s"])])
    return table(head, rows, ["---", "---", "---:", "---:"])


def v_e_scale(c: Ctx) -> str:
    head = [c.t("At once", "并发"), c.t("Answered", "回答"), c.t("Refused", "拒绝"),
            c.t("Sandbox create + turn 1, s", "沙箱创建 + 第 1 轮，s")]
    rows = []
    for n in (4, 8):
        r = c.m[f"e.c{n}"]
        rows.append([str(n), f"{r['answered']}/{r['n']}", f"{r['refused']}/{r['n']}",
                     ", ".join(num(v) for v in r["answered_s"])])
    rec = c.runs["e-e2b-2026-10-14/perf.json"]["recovery"]
    f = rec["followup_on_live_session"]
    tail = c.t(f"\nSandbox replacement after `kill`: new Firecracker sandbox Ready in {num(rec['replacement_ready_s'])} s; "
               f"follow-up on a live session HTTP {f['http']}, {num(f['wall_s'])} s, \"{f['answer']}\", "
               f"state preserved: {str(f['state_preserved']).lower()}. No count of live micro-VMs is published for "
               "this run: the driver counted them with `pgrep -fc`, which also counted its own shell.\n",
               f"\n`kill` 删除沙箱后的替换：新 Firecracker 沙箱在 {num(rec['replacement_ready_s'])} s 内就绪；"
               f"另一个存活会话的追问 HTTP {f['http']}，{num(f['wall_s'])} s，\"{f['answer']}\"，"
               f"保留状态：{str(f['state_preserved']).lower()}。本次不公布同时运行的 micro-VM 数：驱动用 `pgrep -fc` "
               "计数，把它自己的 shell 也算了进去。\n")
    return table(head, rows, ["---:", "---", "---", "---"]) + tail


def e_turn_table(c: Ctx, runs_: list[dict], prefix: str) -> str:
    head = [c.t("Run", "次"), c.t("Create → Running", "创建 → Running"), c.t("First answer", "第一个回答"),
            c.t("Cold", "冷启动"), c.t("Follow-up", "后续"), c.t("Pause", "暂停"), c.t("Resume", "恢复"),
            c.t("Answer after resume", "恢复后回答"), c.t("Resume → answer", "恢复 → 回答"), c.t("Answers", "回答")]
    rows = []
    for r in runs_:
        answers = " / ".join(str(r[k]["response"]) for k in ("first", "warm", "first_after_resume", "warm_after_resume"))
        rows.append([str(r["run"]), num(r["create_total_s"]), num(r["first"]["wall_s"]), num(r["cold_total_s"]),
                     num(r["warm"]["wall_s"]), num(r["stop_total_s"]), num(r["resume_total_s"]),
                     num(r["first_after_resume"]["wall_s"]), num(r["recover_total_s"]), answers])
    m = c.m
    rows.append([c.t("**median**", "**中位数**")] + [f"**{num(m[f'{prefix}.{k}']['median'])}**" for k in
                 ("create", "first", "cold", "warm", "pause", "resume", "first_after_resume", "recover")] + [""])
    return table(head, rows, ["---"] + ["---:"] * 8 + ["---"])


def v_e_multi_input(c: Ctx) -> str:
    mu = c.runs["e-e2b-2026-10-15/multi.json"]
    i, t = mu["inputs"], mu["tool"]
    lines = [c.t(f"Tool: `{t['file']}` (SHA-256 `{t['sha256'][:12]}…`); the agent runs on host b used the revision "
                 f"`{t['agent_runs_revision_sha256'][:12]}…`, which did not yet record HTTP statuses. Agent instructions, "
                 "tool schema and prompts as in `e-2026-10-14-agent` and `e-2026-10-14-perf`.",
                 f"工具：`{t['file']}`（SHA-256 `{t['sha256'][:12]}…`）；host b 上的 agent 实测用的是前一版 "
                 f"`{t['agent_runs_revision_sha256'][:12]}…`，那一版还不记录 HTTP 状态。agent 指令、工具定义和问题与 "
                 "`e-2026-10-14-agent`、`e-2026-10-14-perf` 相同。")]
    install = {
        "a": c.t("installed on 2026-10-14 for `e-2026-10-14-agent`, deallocated overnight, then started and brought up "
                 "once with `docker compose up -d --wait`, as E2B's guide asks after a reboot",
                 "2026-10-14 为 `e-2026-10-14-agent` 安装，夜间释放；开机后按 E2B 指南对重启的要求执行一次 "
                 "`docker compose up -d --wait`"),
        "b": c.t(f"installed on 2026-10-15 with README step 7 (`compose.yaml` and `.env` at `588e5b9`, both SHA-256 OK) "
                 f"in {mu['hosts']['b']['install_time']['total_s']} s",
                 f"2026-10-15 按 README 第 7 步安装（`588e5b9` 的 `compose.yaml` 和 `.env`，SHA-256 均校验通过），用时 "
                 f"{mu['hosts']['b']['install_time']['total_s']} s")}
    for name in ("a", "b"):
        h = mu["hosts"][name]
        disk = h["os_disk"].replace(", ", " · ")
        lines.append(c.t(f"Host {name}: `{h['vm_sku']}`, zone `{h['zone']}`, host kernel `{h['host_kernel']}`, Docker "
                         f"`{h['docker']}`, OS disk `{disk}`, `HUGEPAGES={h['hugepages']}`; {install[name]}.",
                         f"Host {name}：`{h['vm_sku']}`，zone `{h['zone']}`，宿主机内核 `{h['host_kernel']}`，Docker "
                         f"`{h['docker']}`，系统盘 `{disk}`，`HUGEPAGES={h['hugepages']}`；{install[name]}。"))
    lines += [c.t("Network: both hosts in one subnet and zone; the tool ran on host a and reached host b's ports `3000` "
                  "and `3002` over the private network; host b's NSG admitted only host a's private address on those "
                  "two ports and no other inbound traffic.",
                  "网络：两台主机在同一子网和可用区；工具在 host a 上运行，经私网访问 host b 的 `3000` 和 `3002` 端口；"
                  "host b 的 NSG 只允许 host a 的私网地址访问这两个端口，不放行其他入站流量。"),
              c.t(f"Agent runs on host b: \"{i['first']}\", then \"{i['count']}\" 3 times, pausing for "
                  f"{i['resume_wait_s']} s after the second turn.",
                  f"host b 上的 agent 实测：\"{i['first']}\"，然后 3 次 \"{i['count']}\"，第二轮后暂停 "
                  f"{i['resume_wait_s']} s。"),
              c.t(f"One host at once, on host a: 8 sessions; new session \"{i['cold'].format(i=1)}\" and one session "
                  f"with \"{i['warm'][0]}\" before it.",
                  f"单台并发（host a）：8 个会话；在此之前先跑一个新会话 \"{i['cold'].format(i=1)}\" 和一个以 "
                  f"\"{i['warm'][0]}\" 开头的会话。"),
              c.t(f"At once, per session: \"{i['scale'][0]}\", then \"{i['scale'][1]}\".",
                  f"并发时每个会话：\"{i['scale'][0]}\"，然后 \"{i['scale'][1]}\"。"),
              c.t(f"Host stop: on each host one live session (\"{i['live'][0]}\") and one paused sandbox "
                  f"(\"{i['paused'][0]}\", `{{h}}` = the host's name); then `docker compose stop` on host a with the VM "
                  f"running, later `docker compose up -d --wait`; reads back \"{i['live'][1]}\" and "
                  f"\"{i['paused'][1]}\".",
                  f"停一台：每台各有一个存活会话（\"{i['live'][0]}\"）和一个暂停的沙箱（\"{i['paused'][0]}\"，"
                  f"`{{h}}` 为主机名）；然后在 host a 上执行 `docker compose stop`（VM 不关），之后再 "
                  f"`docker compose up -d --wait`；读回时发 \"{i['live'][1]}\" 和 \"{i['paused'][1]}\"。"),
              c.t("Idle capacity: `Sandbox.create(\"base\")` on host a, one after another with no model and no "
                  "command, until a create was refused, 3 times.",
                  "空闲容量：在 host a 上逐个 `Sandbox.create(\"base\")`，不调模型、不跑命令，直到创建被拒，共 3 次。")]
    return quote(lines)


def v_e_multi_agent(c: Ctx) -> str:
    mu = c.runs["e-e2b-2026-10-15/multi.json"]
    rr = mu["agent_runs_host_b"]
    slow = max(rr, key=lambda r: r["cold_total_s"])
    tail = c.t(f"\nHistory inside the sandbox grew 1, 2, 3, 4 in {c.m['e.multi.b.state_kept']['count']} of "
               f"{len(rr)} runs. Run {slow['run']}'s slow turns were model turns: its first answer took "
               f"{num(slow['first']['wall_s'])} s, of which the sandbox command took "
               f"{num(slow['first']['sandbox_tool_s'])} s.\n",
               f"\n沙箱内的历史在 {len(rr)} 次中有 {c.m['e.multi.b.state_kept']['count']} 次按 1、2、3、4 增长。"
               f"第 {slow['run']} 次慢在模型轮次：第一个回答用了 {num(slow['first']['wall_s'])} s，其中沙箱命令只占 "
               f"{num(slow['first']['sandbox_tool_s'])} s。\n")
    return e_turn_table(c, rr, "e.multi.b") + tail


def v_e_multi_scale(c: Ctx) -> str:
    mu = c.runs["e-e2b-2026-10-15/multi.json"]
    head = [c.t("Phase", "阶段"), c.t("Sessions at once", "同时会话"), c.t("Answered", "回答"),
            c.t("Refused", "拒绝"), c.t("Refusal cause", "拒绝原因"), c.t("Sandbox create, s", "沙箱创建，s"),
            c.t("Whole session, s", "整个会话，s"), c.t("Model HTTP attempts", "模型 HTTP 尝试")]
    rows = []
    for key, level, label in (("e.multi.one_host", mu["perf_host_a"]["at_once"], c.t("host a alone", "仅 host a")),
                              ("e.multi.two_hosts", mu["two_hosts_at_once"], c.t("hosts a and b", "host a 和 b"))):
        r = c.m[key]
        ok = [s for s in level["sessions"] if s.get("remembered")]
        create = [s["sandbox_create_to_ready_s"] for s in ok]
        sess = [s["session_s"] for s in ok]
        per = c.t(", ".join(f"{h} {v['answered']}/{v['sessions']}" for h, v in r["per_host"].items()),
                  "，".join(f"{h} {v['answered']}/{v['sessions']}" for h, v in r["per_host"].items()))
        codes = ", ".join(f"{code} ×{n}" for code, n in sorted(r["model_http_statuses"].items()))
        rows.append([label, str(r["n"]), f"{r['answered']}/{r['n']} ({per})", f"{r['refused']}/{r['n']}",
                     f"`{r['refusal_cause']}`", f"{num(min(create), 3)}–{num(max(create), 3)}",
                     f"{num(min(sess))}–{num(max(sess))}", codes])
    tail = c.t("\nThe client spread the 16 sessions over the two hosts in turn. Each refusal was E2B's "
               "`500: Failed to place sandbox`, with `cannot allocate memory` from the orchestrator behind it; the "
               "model's 2 HTTP 500 answers were retried within their turns and those sessions answered.\n",
               "\n客户端把 16 个会话轮流分给两台主机。每次拒绝都是 E2B 的 `500: Failed to place sandbox`，"
               "背后是编排器的 `cannot allocate memory`；模型返回的 2 次 HTTP 500 都在同一轮内重试成功，这些会话照常回答。\n")
    return table(head, rows, ["---", "---:", "---", "---", "---", "---:", "---:", "---"]) + tail


def v_e_multi_idle(c: Ctx) -> str:
    mu = c.runs["e-e2b-2026-10-15/multi.json"]
    hp = c.m["e.multi.hugepages"]
    head = [c.t("Run", "次"), c.t("Created before the refusal", "被拒前创建成功"), c.t("Refusal", "拒绝原因"),
            c.t("Hugepages taken: before → at the refusal", "已占大页：开始 → 被拒时"),
            c.t("After killing them", "全部删除后")]
    rows = []
    for r in mu["idle_capacity_host_a"]:
        ok = [x for x in r["creates"] if x["ok"]]
        rows.append([str(r["run"]), str(len(ok)), f"`{r['refusal']}`",
                     f"{r['before']['hugepages_taken']} → {ok[-1]['hugepages_taken']}",
                     c.t(f"{r['after_kill']['hugepages_taken']} pages, {r['after_kill']['sandboxes_left']} sandboxes",
                         f"{r['after_kill']['hugepages_taken']} 页，{r['after_kill']['sandboxes_left']} 个沙箱")])
    free = hp["pool"] - hp["postgres"]
    tail = c.t(f"\nPages are 2 MiB. Each `base` sandbox (512 MiB) reserved {hp['per_sandbox']} pages when it was "
               f"created, idle or not. The pool is {hp['pool']} pages and postgres in the Compose stack holds "
               f"{hp['postgres']}, which leaves {free} pages: room for {hp['fit']} sandboxes. Run 1 stopped earlier "
               "on a different error.\n",
               f"\n每页 2 MiB。每个 `base` 沙箱（512 MiB）在创建时就预留 {hp['per_sandbox']} 页，不论是否空闲。"
               f"大页池共 {hp['pool']} 页，Compose 栈里的 postgres 占 {hp['postgres']} 页，剩下 {free} 页，"
               f"能放下 {hp['fit']} 个沙箱。第 1 次因为另一个错误提前停止。\n")
    return table(head, rows, ["---:", "---:", "---", "---", "---"]) + tail


def v_e_multi_hostloss(c: Ctx, rel: str = "e-e2b-2026-10-15/multi.json", other: str = "b") -> str:
    hl = c.runs[rel]["host_loss"]
    b, a, back = hl["after_loss"][other], hl["after_loss"]["a"], hl["after_restart"]["a"]

    def res(x: dict, value: str | None = None) -> str:
        if x["ok"]:
            return c.t(f"ok, {num(x['seconds'])} s", f"成功，{num(x['seconds'])} s") + (f", \"{value}\"" if value else "")
        err = x["error"]
        short = "connection refused" if "Connection refused" in err else err.split(":")[0] + ": " + \
            err.split(": ", 1)[1].replace("<sandbox id>", "…")
        return c.t(f"failed: `{short}`", f"失败：`{short}`")

    rows = [[c.t(f"host {other}, live session: read its file", f"host {other}，存活会话：读回文件"),
             res(b["live_session_read"], b["live_session_read"].get("value"))],
            [c.t(f"host {other}: new session", f"host {other}：新会话"),
             res(b["new_session"], b["new_session"]["value"]["answer"])],
            [c.t(f"host {other}: resume host a's paused sandbox", f"host {other}：恢复 host a 上暂停的沙箱"),
             res(b["connect_lost_hosts_paused_sandbox"])],
            [c.t("host a, stopped: create", "host a 停止中：创建"), res(a["create"])],
            [c.t("host a, stopped: resume its own paused sandbox", "host a 停止中：恢复自己暂停的沙箱"),
             res(a["connect_own_paused_sandbox"])],
            [c.t("host a, after restart: resume its own paused sandbox", "host a 重启后：恢复自己暂停的沙箱"),
             res(back["resume_own_paused_sandbox"], back["resume_own_paused_sandbox"].get("value"))],
            [c.t("host a, after restart: the sandbox that was running when it stopped",
                 "host a 重启后：停止时正在运行的沙箱"), res(back["live_sandbox_before_loss"])],
            [c.t("host a, after restart: new session", "host a 重启后：新会话"),
             res(back["new_session"], back["new_session"]["value"]["answer"])],
            [c.t("`docker compose stop` / `docker compose up -d --wait`", "`docker compose stop` / `docker compose up -d --wait`"),
             f"{num(hl['stop']['seconds'])} s / {num(hl['start']['seconds'])} s"]]
    return table([c.t("Check", "检查项"), c.t("Observed", "观察结果")], rows)


def v_e_latest_input(c: Ctx) -> str:
    la = c.runs["e-e2b-2026-10-15/latest.json"]
    s, ci = la["sdk_check"], la["code_interpreter"]
    r = ci["template_resources"]
    lines = [c.t(f"SDK check: `{s['tool']['file']} {s['tool']['command']}` with `{s['sdk']}` (Python {s['python']}), "
                 f"tool SHA-256 `{s['tool']['sha256'][:12]}…`; \"{s['inputs']['first']}\", then "
                 f"\"{s['inputs']['count']}\" 3 times, pausing for {s['inputs']['resume_wait_s']} s after the "
                 "second turn.",
                 f"SDK 检查：`{s['tool']['file']} {s['tool']['command']}`，`{s['sdk']}`（Python {s['python']}），"
                 f"工具 SHA-256 `{s['tool']['sha256'][:12]}…`；\"{s['inputs']['first']}\"，然后 3 次 "
                 f"\"{s['inputs']['count']}\"，第二轮后暂停 {s['inputs']['resume_wait_s']} s。"),
             c.t(f"code-interpreter: template `{ci['template']}` ({r['vcpus']} vCPU, {r['memory_mb']} MB) from "
                 f"[`e2b-dev/code-interpreter@162a2a2`]({CI_TEMPLATE_URL}), built with its unchanged "
                 f"`template/build_prod.py` (build environment `e2b 2.32.0`); checked with "
                 f"`e2b-code-interpreter {ci['client']['e2b-code-interpreter']}` and `{ci['check_tool']['file']}`.",
                 f"code-interpreter：模板 `{ci['template']}`（{r['vcpus']} vCPU，{r['memory_mb']} MB），来自 "
                 f"[`e2b-dev/code-interpreter@162a2a2`]({CI_TEMPLATE_URL})，用其原样的 `template/build_prod.py` 构建"
                 f"（构建环境 `e2b 2.32.0`）；用 `e2b-code-interpreter {ci['client']['e2b-code-interpreter']}` 和 "
                 f"`{ci['check_tool']['file']}` 检查。"),
             c.t("Cells, in order, with no model: \"x = 1\", \"x+=1; x\", a pandas sum, a matplotlib plot, JavaScript "
                 "\"[1, 2, 3].reduce((a, b) => a + b)\", \"1/0\", then after pause and resume \"x\".",
                 "按顺序执行的代码单元（不经模型）：\"x = 1\"、\"x+=1; x\"、pandas 求和、matplotlib 画图、JavaScript "
                 "\"[1, 2, 3].reduce((a, b) => a + b)\"、\"1/0\"，暂停并恢复后再执行 \"x\"。")]
    return quote(lines)


def v_e_latest_sdk(c: Ctx) -> str:
    la = c.runs["e-e2b-2026-10-15/latest.json"]
    rr = la["sdk_check"]["runs"]
    codes: dict[str, int] = {}
    for r in rr:
        for k in ("first", "warm", "first_after_resume", "warm_after_resume"):
            for code in r[k]["http_statuses"]:
                codes[str(code)] = codes.get(str(code), 0) + 1
    joined = ", ".join(f"{code} ×{n}" for code, n in sorted(codes.items()))
    tail = c.t(f"\nModel HTTP attempts: {joined}. History grew 1, 2, 3, 4 in "
               f"{c.m['e.latest.sdk.state_kept']['count']} of {len(rr)} runs; no sandbox was left.\n",
               f"\n模型 HTTP 尝试：{joined}。历史在 {len(rr)} 次中有 {c.m['e.latest.sdk.state_kept']['count']} 次按 "
               "1、2、3、4 增长；没有遗留沙箱。\n")
    return e_turn_table(c, rr, "e.latest.sdk") + tail


def v_e_latest_build(c: Ctx) -> str:
    ci = c.runs["e-e2b-2026-10-15/latest.json"]["code_interpreter"]
    head = [c.t("Attempt", "次"), "HUGEPAGES", c.t("Build workspace, MiB", "构建工作空间，MiB"),
            c.t("OS disk, GB", "系统盘，GB"), c.t("Result", "结果"), c.t("Seconds", "秒")]
    why_by_attempt = {
        1: c.t("`mmap memfd: cannot allocate memory` while resuming the build VM",
               "恢复构建 VM 时 `mmap memfd: cannot allocate memory`"),
        2: c.t("`pip install` in the build VM: `[Errno 28] No space left on device`",
               "构建 VM 里的 `pip install`：`[Errno 28] No space left on device`"),
        3: c.t("below 8 GiB free on the OS disk, which also holds the E2B database",
               "系统盘剩余不到 8 GiB，而 E2B 的数据库也在这块盘上"),
    }
    rows = []
    for b in ci["builds"]:
        at = f"{b['step']}/{b['steps']}"
        if b["result"] == "failed":
            what = c.t(f"failed at step {at}: {why_by_attempt[b['attempt']]}",
                       f"第 {at} 步失败：{why_by_attempt[b['attempt']]}")
        elif b["result"] == "stopped":
            what = c.t(f"stopped by us at step {at}: {why_by_attempt[b['attempt']]}",
                       f"我们在第 {at} 步停止：{why_by_attempt[b['attempt']]}")
        else:
            what = c.t(f"built, {at} steps; steps {b['cached_steps'][0]}–{b['cached_steps'][1]} from attempt 3's cache",
                       f"建成，{at} 步；第 {b['cached_steps'][0]}–{b['cached_steps'][1]} 步用第 3 次的缓存")
        rows.append([str(b["attempt"]), str(b["hugepages"]), str(b["build_disk_mb"]), str(b["os_disk_gb"]), what,
                     num(b["seconds"])])
    a3 = next(b for b in ci["builds"] if b["result"] == "stopped")
    a4 = next(b for b in ci["builds"] if b["result"] == "built")
    used = a4["os_disk_free_gib_before"] - a4["os_disk_free_gib_after"]
    ch = {x["to"]: x for x in ci["changes_for_the_build"]}
    hp, ws, disk = ch[4096], ch[8192], ch[128]
    tail = c.t(f"\nBy step {a3['step']} attempt {a3['attempt']} had written {a3['written_mib']['templates']} MiB of "
               f"templates and {a3['written_mib']['orchestrator']} MiB under the orchestrator's directory, with "
               f"{a3['os_disk_free_mib']} MiB left on the OS disk; attempt {a4['attempt']} then used {used} GiB "
               f"(free {a4['os_disk_free_gib_before']} → {a4['os_disk_free_gib_after']} GiB). What we changed: "
               f"`HUGEPAGES` in the Compose `.env` {hp['from']} → {hp['to']}, then `docker compose up -d --wait`; the "
               f"test team's build workspace (`disk_mb`) {ws['from']} → {ws['to']} MiB, by adding one row to Embed's "
               f"internal `project_limits` table; the OS disk {disk['from']} → {disk['to']} GB (deallocate, snapshot, "
               "`az disk update --size-gb`, start).\n",
               f"\n到第 {a3['step']} 步时，第 {a3['attempt']} 次已写入 {a3['written_mib']['templates']} MiB 模板和 "
               f"{a3['written_mib']['orchestrator']} MiB 编排器目录数据，系统盘只剩 {a3['os_disk_free_mib']} MiB；"
               f"第 {a4['attempt']} 次又用掉 {used} GiB（剩余 {a4['os_disk_free_gib_before']} → "
               f"{a4['os_disk_free_gib_after']} GiB）。我们改了这些：Compose `.env` 里的 `HUGEPAGES` {hp['from']} → "
               f"{hp['to']}，再执行 `docker compose up -d --wait`；测试团队的构建工作空间（`disk_mb`）{ws['from']} → "
               f"{ws['to']} MiB，做法是往 Embed 内部的 `project_limits` 表加一行；系统盘 {disk['from']} → {disk['to']} GB"
               "（释放、拍快照、`az disk update --size-gb`、开机）。\n")
    return table(head, rows, ["---:", "---:", "---:", "---:", "---", "---:"]) + tail


def v_e_latest_ci(c: Ctx) -> str:
    ci = c.runs["e-e2b-2026-10-15/latest.json"]["code_interpreter"]
    m = c.m["e.latest.ci"]
    head = [c.t("Step", "步骤"), c.t("Passed", "通过"), c.t("Seconds, min–max", "秒，最小–最大"),
            c.t("Observed in run 4", "第 4 次的观察")]
    last = {s["step"]: s for s in ci["check_runs"][-1]["steps"]}
    rows = []
    for name, v in m["steps"].items():
        s = last[name]
        seen = s.get("error", {}).get("value") if isinstance(s.get("error"), dict) else None
        if seen is None:
            obs = s.get("observed") or {}
            seen = obs.get("text") if obs.get("text") else (obs.get("error") or
                                                              ("png" if obs.get("results") else ""))
        rows.append([f"`{name}`", f"{v['passed']}/{m['n']}", f"{num(v['min'], 3)}–{num(v['max'], 3)}",
                     f"`{seen}`" if seen else ""])
    js = ci["javascript_failure"]
    labels = [c.t("JavaScript as the first cell of a new sandbox, then the same cell again",
                  "新沙箱的第一个单元就是这段 JavaScript，再执行一次同样的单元"),
              c.t("a Python cell first, then the JavaScript cell (the order of the check)",
                  "先执行一个 Python 单元，再执行这段 JavaScript（检查里的顺序）"),
              c.t("the JavaScript cell with a leading semicolon", "在这段 JavaScript 前加一个分号")]
    if len(js["repro"]) != len(labels):
        raise RenderError("e-2026-10-15-latest: expected 3 JavaScript reproduction cases")
    repro = []
    for label, case in zip(labels, js["repro"]):
        cells = " → ".join((f"`{x['text']}`" if x["text"] else f"`{x['error']['name']}`" if x["error"] else c.t("ok", "成功"))
                           for x in case["cells"])
        repro.append(f"{label}: {cells}" if c.lang == "en" else f"{label}：{cells}")
    src = js["source"]
    tail = c.t(f"\nWhy JavaScript failed, reproduced in new sandboxes: " + "; ".join(repro) + f". In `{src['file']}` at "
               f"`{src['repo'].split('@')[1][:7]}`, line 160 writes `process.env['KEY'] = 'VALUE'` with no semicolon, "
               "and lines 351–361 put those lines before the user code on a context's first execution, so code that "
               "starts with `[` continues the last of them. Cells that start with `console.log` or a number ran.\n",
               f"\nJavaScript 失败的原因（在新沙箱里复现）：" + "；".join(repro) + f"。在 `{src['repo'].split('@')[1][:7]}` 的 "
               f"`{src['file']}` 里，第 160 行写出的 `process.env['KEY'] = 'VALUE'` 不带分号，第 351–361 行在一个上下文"
               "第一次执行时把这些语句放在用户代码前面，于是以 `[` 开头的代码接到了最后一句上。以 `console.log` 或数字"
               "开头的代码可以正常执行。\n")
    return table(head, rows, ["---", "---:", "---:", "---"]) + tail


CI_TURNS = ("first", "warm", "first_after_resume", "warm_after_resume")


def v_e_ci_input(c: Ctx) -> str:
    ca, cp = c.runs["e-e2b-2026-10-17/ci-agent.json"], c.runs["e-e2b-2026-10-17/ci-perf.json"]
    i, p, tpl, inst, sdk = ca["inputs"], cp["inputs"], ca["template"], ca["install"], ca["sdk"]
    tool, w, ref = ca["agent"]["tools"][0], p["warm"], tpl["source"].split("@")[1].split(",")[0]
    lines = [c.t(f"Host c: `Standard_D4ds_v7` (4 vCPU, 16 GiB), zone `2`, 128 GB OS disk; E2B Embed "
                 f"`{ca['e2b_runtime_commit'][:7]}` installed on {inst['date_utc']} from the two files README step 7 "
                 "downloads (`compose.yaml` and `.env`, SHA-256 OK), then `HUGEPAGES=4096` appended to `.env`.",
                 f"Host c：`Standard_D4ds_v7`（4 vCPU、16 GiB），zone `2`，系统盘 128 GB；E2B Embed "
                 f"`{ca['e2b_runtime_commit'][:7]}` 于 {inst['date_utc']} 用 README 第 7 步下载的那两个文件安装"
                 "（`compose.yaml` 和 `.env`，SHA-256 均校验通过），之后在 `.env` 末尾加上 `HUGEPAGES=4096`。"),
             c.t(f"Template: `{tpl['name']}` ({tpl['vcpus']} vCPU, {tpl['memory_mb']} MB) from "
                 f"[`e2b-dev/code-interpreter@{ref[:7]}`]({CI_TEMPLATE_URL}), built on host c with its unchanged "
                 "`template/build_prod.py` (build table below).",
                 f"模板：`{tpl['name']}`（{tpl['vcpus']} vCPU、{tpl['memory_mb']} MB），来自 "
                 f"[`e2b-dev/code-interpreter@{ref[:7]}`]({CI_TEMPLATE_URL})，在 host c 上用其未改动的 "
                 "`template/build_prod.py` 构建（见下方构建表）。"),
             c.t(f"Executor: E2B Code Interpreter, `e2b {sdk['e2b']}` and `e2b-code-interpreter "
                 f"{sdk['e2b-code-interpreter']}`; each call of the `{tool['name']}` tool is one `run_code` cell in the "
                 "sandbox's Jupyter kernel: the tool's history step, then the model's Python.",
                 f"执行方式：E2B Code Interpreter，`e2b {sdk['e2b']}` 和 `e2b-code-interpreter "
                 f"{sdk['e2b-code-interpreter']}`；每次调用 `{tool['name']}` 工具就是在沙箱的 Jupyter 内核里执行一个 "
                 "`run_code` 单元：先是工具记录历史的那一步，然后是模型写的 Python。"),
             c.t(f"Tool: `{ca['tool']['file']}` (SHA-256 `{ca['tool']['sha256'][:12]}…`) with `--executor "
                 f"code-interpreter`; model `{ca['model']}`; agent instructions and tool schema as in "
                 "`e-2026-10-14-agent`.",
                 f"工具：`{ca['tool']['file']}`（SHA-256 `{ca['tool']['sha256'][:12]}…`），加 `--executor "
                 f"code-interpreter`；模型 `{ca['model']}`；agent 指令和工具定义与 `e-2026-10-14-agent` 相同。"),
             c.t(f"Agent runs: \"{i['first']}\", then \"{i['count']}\" 3 times, pausing for {i['resume_wait_s']} s "
                 "after the second turn.",
                 f"agent 实测：\"{i['first']}\"，然后 3 次 \"{i['count']}\"，第二轮后暂停 {i['resume_wait_s']} s。"),
             c.t(f"Perf: new sessions \"{p['cold']}\" (n = 1–5); one session \"{w[0]}\", then \"{w[1]}\", then "
                 f"\"{w[2]}\", then \"{w[3]}\"; 4 and 8 sessions at once, each \"{p['scale'][0]}\", then "
                 f"\"{p['scale'][1]}\"; replacement: a live session ran \"{p['live'][0]}\" and, after another sandbox "
                 f"was killed and replaced, \"{p['live'][1]}\".",
                 f"性能：新会话 \"{p['cold']}\"（n = 1–5）；同一会话 \"{w[0]}\"，然后 \"{w[1]}\"，再 \"{w[2]}\"，再 "
                 f"\"{w[3]}\"；4 和 8 个会话同时开始，每个发 \"{p['scale'][0]}\"，然后 \"{p['scale'][1]}\"；替换：一个"
                 f"存活会话先发 \"{p['live'][0]}\"，另一个沙箱被删除并替换后再发 \"{p['live'][1]}\"。"),
             c.t("Idle capacity: `Sandbox.create` of each template one after another, checked with `run_code(\"1+1\")` "
                 "or `echo ok` and no model, until a create was refused; at `HUGEPAGES=4096`, then at the Compose "
                 "default 2048, then 4096 restored. The probe is the author's script, SHA-256 in the record.",
                 "空闲容量：逐个 `Sandbox.create` 每种模板，用 `run_code(\"1+1\")` 或 `echo ok` 检查、不调模型，直到创建被拒；"
                 "先在 `HUGEPAGES=4096` 下测，再在 Compose 默认的 2048 下测，最后恢复 4096。探测脚本是作者的私有脚本，"
                 "SHA-256 记在记录里。")]
    return quote(lines)


def v_e_ci_agent(c: Ctx) -> str:
    ca = c.runs["e-e2b-2026-10-17/ci-agent.json"]
    rr = ca["runs"]
    tool_s = [r[t]["sandbox_tool_s"] for r in rr for t in CI_TURNS]
    kernels = sorted({r[t]["guest_kernel"] for r in rr for t in CI_TURNS})
    codes: dict[int, int] = {}
    for r in rr:
        for t in CI_TURNS:
            for code in r[t]["http_statuses"]:
                codes[code] = codes.get(code, 0) + 1
    shown = ", ".join(f"{code} ×{n}" for code, n in sorted(codes.items()))
    tail = c.t(f"\nHistory inside the sandbox grew 1, 2, 3, 4 in {c.m['e.ci.state_kept']['count']} of {len(rr)} runs. "
               f"Every turn made one `run_code` call, which took {num(min(tool_s), 3)}–{num(max(tool_s), 3)} s of the "
               f"turn; guest kernel `{', '.join(kernels)}`. Model HTTP attempts: {shown}. No sandbox, conversation or "
               "agent version was left.\n",
               f"\n沙箱内的历史在 {len(rr)} 次中有 {c.m['e.ci.state_kept']['count']} 次按 1、2、3、4 增长。"
               f"每一轮都只调用一次 `run_code`，占该轮 {num(min(tool_s), 3)}–{num(max(tool_s), 3)} s；guest 内核 "
               f"`{', '.join(kernels)}`。模型 HTTP 尝试：{shown}。没有遗留沙箱、会话或 agent 版本。\n")
    return e_turn_table(c, rr, "e.ci") + tail


def v_e_ci_build(c: Ctx) -> str:
    b = c.runs["e-e2b-2026-10-17/ci-agent.json"]["build"]
    head = [c.t("Attempt", "尝试"), c.t("Result", "结果"), c.t("Seconds", "秒")]
    rows = []
    for a in b["attempts"]:
        if a["result"] == "built":
            lo, hi = a["cached_steps"]
            res = c.t(f"built, {a['steps']}/{a['steps']} steps; steps {lo}–{hi} from attempt 2's cache "
                      f"(builder: {a['builder_reported']})",
                      f"建成，{a['steps']}/{a['steps']} 步；第 {lo}–{hi} 步来自第 2 次的缓存（构建器显示 "
                      f"{a['builder_reported']}）")
        else:
            res = c.t(f"failed at step {a['step']}/{a['steps']}: `{a['error']}`",
                      f"在第 {a['step']}/{a['steps']} 步失败：`{a['error']}`")
        rows.append([str(a["attempt"]), res, num(a["seconds"])])
    s = {x["setting"].split()[0]: x["value"] for x in b["settings"]}
    tail = c.t(f"\nAll three attempts ran with the settings that let host a build on 2026-10-15: `HUGEPAGES` "
               f"{s['HUGEPAGES']}, the test team's build workspace (`disk_mb`) {s['disk_mb']} MiB in one "
               f"`project_limits` row, and a {s['OS']} GB OS disk; nothing changed between attempts. Attempt 1 started "
               "16 s after the row was added and ran out of space at step 8; attempt 2, started about 5 minutes after "
               "the row, passed that step. That fits the new limit taking effect only after a delay, which we did "
               "not verify further. Before attempt 3 a subscription automation had deallocated the VM and we started "
               "it again.\n",
               f"\n三次尝试都用了 2026-10-15 让 host a 建成的设置：`HUGEPAGES` {s['HUGEPAGES']}、测试团队的构建工作空间"
               f"（`disk_mb`）{s['disk_mb']} MiB（`project_limits` 表里一行）、{s['OS']} GB 系统盘；各次尝试之间没有改任何"
               "设置。第 1 次在加这一行之后 16 s 开始，在第 8 步空间不足；第 2 次在加行约 5 分钟后开始，通过了这一步。"
               "这与新上限要过一段时间才生效相符，我们没有进一步核实。第 3 次之前，一个订阅自动化把 VM 释放了，我们重新开机。\n")
    return table(head, rows, ["---:", "---", "---:"]) + tail


CI_PERF = "e-e2b-2026-10-17/ci-perf.json"


def v_e_ci_cold(c: Ctx, rel: str = CI_PERF, prefix: str = "e.ci") -> str:
    cp = c.runs[rel]
    head = [c.t("Start (UTC)", "开始（UTC）"), c.t("Answer", "回答"),
            c.t("Sandbox created → Ready, s", "沙箱创建 → Ready，s"),
            c.t("Agent turn, s", "Agent 轮次，s"), c.t("New session → answer, s", "新会话 → 回答，s")]
    rows = [[t["started_utc"][:8], t["answer"], num(t["sandbox_create_to_ready_s"]), num(t["agent_turn_s"]),
             num(t["wall_s"])] for t in cp["cold"]]
    tail = c.t(f"\nAcross the 5 new sessions: sandbox create → Ready {c.med(f'{prefix}.d_sandbox_ready')}, Prompt Agent "
               f"turn {c.med(f'{prefix}.d_agent_turn')}, end-to-end new session → answer {c.med(f'{prefix}.d_cold')}.\n",
               f"\n5 个新会话汇总：沙箱创建 → Ready {c.med(f'{prefix}.d_sandbox_ready')}，Prompt Agent 轮次 "
               f"{c.med(f'{prefix}.d_agent_turn')}，端到端新会话 → 回答 {c.med(f'{prefix}.d_cold')}。\n")
    return table(head, rows, ["---", "---", "---:", "---:", "---:"]) + tail


def v_e_ci_warm(c: Ctx, rel: str = CI_PERF, prefix: str = "e.ci") -> str:
    cp = c.runs[rel]
    head = [c.t("Turn", "轮次"), c.t("Answer", "回答"), c.t("Sandbox tool, s", "沙箱工具耗时，s"),
            c.t("Turn, s", "轮次耗时，s")]
    rows = []
    for k, t in enumerate(cp["warm"]["turns"]):
        label = c.t("1 (new session; the sandbox was created before it)", "1（新会话；沙箱在此之前创建）") if k == 0 \
            else str(k + 1)
        rows.append([label, t["answer"], num(t["sandbox_tool_s"]), num(t["wall_s"])])
    tail = c.t(f"\nThe sandbox was created in {num(cp['warm']['sandbox_create_to_ready_s'])} s before turn 1. "
               f"Follow-ups 2–4: {c.med(f'{prefix}.d_warm')}.\n",
               f"\n沙箱在第 1 轮之前创建，用时 {num(cp['warm']['sandbox_create_to_ready_s'])} s。"
               f"后续第 2–4 轮：{c.med(f'{prefix}.d_warm')}。\n")
    return table(head, rows, ["---", "---", "---:", "---:"]) + tail


def v_e_ci_scale(c: Ctx, rel: str = CI_PERF, prefix: str = "e.ci") -> str:
    cp = c.runs[rel]
    head = [c.t("At once", "并发"), c.t("Answered", "回答"), c.t("Refused", "拒绝"),
            c.t("Sandbox create, s", "沙箱创建，s"), c.t("Whole session, s", "整个会话，s"),
            c.t("Model HTTP attempts", "模型 HTTP 尝试")]
    rows = []
    for lv in cp["at_once"]:
        n = lv["sessions_at_once"]
        r = c.m[f"{prefix}.c{n}"]
        ok = [s for s in lv["sessions"] if s.get("remembered")]
        create = [s["sandbox_create_to_ready_s"] for s in ok]
        codes = ", ".join(f"{code} ×{k}" for code, k in sorted(lv["model_http_statuses"].items()))
        rows.append([str(n), f"{r['answered']}/{r['n']}", f"{r['refused']}/{r['n']}",
                     f"{num(min(create), 3)}–{num(max(create), 3)}",
                     f"{num(min(r['session_s']))}–{num(max(r['session_s']))}", codes])
    rec = cp["recovery"]
    f = rec["followup_on_live_session"]
    refused = any(c.m[f"{prefix}.c{lv['sessions_at_once']}"]["refused"] for lv in cp["at_once"])
    tail = c.t(("\nEach refused session got E2B's `500: Failed to place sandbox` at create and never reached the model. "
                if refused else "\nNo session was refused. ") +
               f"Sandbox replacement after `kill`: a new `code-interpreter-v1` sandbox was Ready in "
               f"{num(rec['replacement_ready_s'])} s; follow-up on a live session HTTP {f['http']}, "
               f"{num(f['wall_s'])} s, \"{f['answer']}\", state preserved: {str(f['state_preserved']).lower()}. "
               "No sandbox or conversation was left.\n",
               ("\n每个被拒的会话都在创建沙箱时收到 E2B 的 `500: Failed to place sandbox`，没有走到模型。"
                if refused else "\n没有会话被拒。") +
               f"`kill` 删除沙箱后的替换：新的 `code-interpreter-v1` 沙箱在 {num(rec['replacement_ready_s'])} s 内就绪；"
               f"另一个存活会话的追问 HTTP {f['http']}，{num(f['wall_s'])} s，\"{f['answer']}\"，"
               f"保留状态：{str(f['state_preserved']).lower()}。没有遗留沙箱或会话。\n")
    return table(head, rows, ["---:", "---", "---", "---:", "---:", "---"]) + tail


def v_e_ci_idle(c: Ctx) -> str:
    cap = c.runs["e-e2b-2026-10-17/ci-perf.json"]["idle_capacity"]
    m = c.m["e.ci.idle"]
    head = [c.t("`HUGEPAGES` (2 MiB pages)", "`HUGEPAGES`（2 MiB 页）"), c.t("Free at start", "开始时空闲"),
            c.t(f"`code-interpreter-v1` (2048 MB = {m['ci_pages']} pages)",
                f"`code-interpreter-v1`（2048 MB = {m['ci_pages']} 页）"),
            c.t(f"`base` (512 MB = {m['base_pages']} pages)", f"`base`（512 MB = {m['base_pages']} 页）")]
    rows = [[str(pool), str(m[f"free_{pool}"]), str(m[f"ci_{pool}"]), str(m[f"base_{pool}"])] for pool in (4096, 2048)]
    touched = []
    for ph in cap["phases"]:
        for r in ph["results"]:
            if r["template"] == "code-interpreter-v1":
                free = [r["before"]["hp_free"]] + [row[3] for row in r["rows"]]
                touched += [a - b for a, b in zip(free, free[1:])]
    n = m["ci_4096"]
    tail = c.t(f"\nEach count is the free pages divided by the template's pages, rounded down: with {n} "
               f"`code-interpreter-v1` sandboxes running, the next one needs {m['ci_pages']} pages when "
               f"{m['free_4096']} − {n} × {m['ci_pages']} = {m['free_4096'] - n * m['ci_pages']} are left. That matches each sandbox reserving its whole memory "
               "from the pool at create, as `e-2026-10-15-multi` measured for `base`; this run recorded only "
               f"`HugePages_Free`, which counts pages already touched ({min(touched)}–{max(touched)} per idle "
               "`code-interpreter-v1` sandbox), not reservations. Every create ended in `500: Failed to place sandbox`, "
               f"every sandbox was killed, and the pool went back to {cap['hugepages_env_found']} pages.\n",
               f"\n每个数都等于空闲页数除以模板所需页数后向下取整：已有 {n} 个 `code-interpreter-v1` 沙箱在运行时，"
               f"下一个需要 {m['ci_pages']} 页，而此时只剩 {m['free_4096']} − {n} × {m['ci_pages']} = "
               f"{m['free_4096'] - n * m['ci_pages']} 页。这与每个沙箱"
               "在创建时就从大页池预留全部内存相符，`e-2026-10-15-multi` 对 `base` 实测过这一点；本次只记录了 "
               f"`HugePages_Free`，它只计已经用到的页（每个空闲 `code-interpreter-v1` 沙箱 {min(touched)}–{max(touched)} "
               "页），不计预留。每轮创建都以 `500: Failed to place sandbox` 结束，所有沙箱都已删除，大页池恢复为 "
               f"{cap['hugepages_env_found']} 页。\n")
    return table(head, rows, ["---:", "---:", "---:", "---:"]) + tail


# --- E, 2026-10-18: code-interpreter-v1 on hosts c, a and d -----------------------------------------

E_HOST, E_MULTI, E_SCALE = ("e-e2b-2026-10-18/ci-host.json", "e-e2b-2026-10-18/ci-multi.json",
                            "e-e2b-2026-10-18/ci-scale.json")


def host_line(c: Ctx, name: str, h: dict, note_en: str, note_zh: str) -> str:
    disk = h["os_disk"].replace(", ", " · ")
    return c.t(f"Host {name}: `{h['vm_sku']}` ({h['vcpus']} vCPU, {h['memory_gib']} GiB), zone `{h['zone']}`, OS disk "
               f"`{disk}`, `HUGEPAGES={h['hugepages']}`; {note_en}.",
               f"Host {name}：`{h['vm_sku']}`（{h['vcpus']} vCPU、{h['memory_gib']} GiB），zone `{h['zone']}`，系统盘 "
               f"`{disk}`，`HUGEPAGES={h['hugepages']}`；{note_zh}。")


def tool_line(c: Ctx, doc: dict, commands: str) -> str:
    sdk = doc["sdk"]
    return c.t(f"Tool: `{doc['tool']['file']}` (SHA-256 `{doc['tool']['sha256'][:12]}…`), {commands}; E2B Code "
               f"Interpreter only (`e2b {sdk['e2b']}`, `e2b-code-interpreter {sdk['e2b-code-interpreter']}`, template "
               f"`{doc['template']['name']}`, {doc['template']['vcpus']} vCPU, {doc['template']['memory_mb']} MB).",
               f"工具：`{doc['tool']['file']}`（SHA-256 `{doc['tool']['sha256'][:12]}…`），{commands}；只用 E2B Code "
               f"Interpreter（`e2b {sdk['e2b']}`、`e2b-code-interpreter {sdk['e2b-code-interpreter']}`，模板 "
               f"`{doc['template']['name']}`，{doc['template']['vcpus']} vCPU、{doc['template']['memory_mb']} MB）。")


def v_e_host_input(c: Ctx) -> str:
    h = c.runs[E_HOST]
    i = h["inputs"]
    probe = i["probe_cell"].replace("\n", "; ")
    cells = i["bench_cells"]
    lines = [host_line(c, "c", h["host"], "the install of `e-2026-10-17-ci-agent`, where `code-interpreter-v1` was built",
                       "即 `e-2026-10-17-ci-agent` 的安装，`code-interpreter-v1` 就建在这台上"),
             tool_line(c, h, c.t(f"`{h['command']}` on host c itself against `localhost`; no model and no agent",
                                 f"在 host c 本机上对 `localhost` 运行 `{h['command']}`；不调模型、不建 agent")),
             c.t(f"Read-back: one sandbox runs the cell `{probe}` while the tool lists the host's Firecracker processes "
                 "from `/proc/*/cmdline`.",
                 f"读回：一个沙箱执行单元 `{probe}`，同时工具从 `/proc/*/cmdline` 列出宿主机上的 Firecracker 进程。"),
             c.t(f"Cycles: 20 sandboxes one after another, each `Sandbox.create` → cell `{cells[0]}` → cell `{cells[1]}` → "
                 f"`beta_pause` → wait {i['resume_wait_s']} s → `Sandbox.connect` → cell `{cells[2]}`, which must print "
                 f"`{i['bench_expected']}` → `kill`.",
                 f"循环：20 个沙箱逐个进行，每个 `Sandbox.create` → 单元 `{cells[0]}` → 单元 `{cells[1]}` → `beta_pause` → "
                 f"等待 {i['resume_wait_s']} s → `Sandbox.connect` → 单元 `{cells[2]}`（必须打印 `{i['bench_expected']}`）→ "
                 "`kill`。"),
             c.t(f"Reservation: sandboxes created one by one and kept, each checked with `{i['reserve_cell']}`; "
                 "`/proc/meminfo` read before the first and after each, until a create was refused; then every sandbox "
                 "killed and `/proc/meminfo` read again 5 s later.",
                 f"预留：逐个创建沙箱并保留，每个用 `{i['reserve_cell']}` 检查；在第一个之前和每个之后读 `/proc/meminfo`，"
                 "直到创建被拒；然后删除全部沙箱，5 s 后再读一次 `/proc/meminfo`。")]
    return quote(lines)


def v_e_host_bench(c: Ctx) -> str:
    runs_ = c.runs[E_HOST]["bench"]["runs"]
    head = [c.t("Cycle", "循环"), c.t("Create", "创建"), c.t("First cell", "第一个单元"), c.t("Next cell", "下一个单元"),
            c.t("Pause", "暂停"), c.t("Resume", "恢复"), c.t("Cell after resume", "恢复后单元"),
            c.t("`x` after resume", "恢复后的 `x`")]
    rows = [[str(r["i"]), num(r["create_s"]), num(r["first_cell_s"]), num(r["warm_cell_s"]), num(r["pause_s"]),
             num(r["resume_s"]), num(r["cell_after_resume_s"]), "42" if r["state_kept"] else c.t("lost", "丢失")]
            for r in runs_]
    m = c.m
    resume = [r["resume_s"] for r in runs_]
    k = next(i for i, v in enumerate(resume) if v > 1)
    later = resume[k:]
    if min(later) <= 1:
        raise RenderError("e-2026-10-18-ci-host: the resume pattern changed; rewrite this summary")
    tail = c.t(f"\nSeconds. Over the 20 cycles: create {c.rng_n('e.h.create')}, {m['e.h.create']['under_0_1']} of 20 under "
               f"0.1 s; first cell {c.rng_n('e.h.first_cell')}; next cell {c.rng_n('e.h.warm_cell')}; pause "
               f"{c.rng_n('e.h.pause')}, {m['e.h.pause']['over_1']} of 20 over 1 s; resume {c.rng_n('e.h.resume')}: the "
               f"first {k} cycles at most {num(m['e.h.resume']['first_3_max'])} s, every later one {span(later)} s; "
               f"cell after resume {c.rng_n('e.h.cell_after_resume')}. `print(x)` printed 42 in "
               f"{m['e.h.state_kept']['count']} of {m['e.h.state_kept']['n']} cycles: the kernel's variable came back "
               "with the memory snapshot. No sandbox was left.\n",
               f"\n单位为秒。20 次循环中：创建 {c.rng_n('e.h.create')}，20 次里 {m['e.h.create']['under_0_1']} 次低于 "
               f"0.1 s；第一个单元 {c.rng_n('e.h.first_cell')}；下一个单元 {c.rng_n('e.h.warm_cell')}；暂停 "
               f"{c.rng_n('e.h.pause')}，20 次里 {m['e.h.pause']['over_1']} 次超过 1 s；恢复 {c.rng_n('e.h.resume')}："
               f"前 {k} 次最多 {num(m['e.h.resume']['first_3_max'])} s，之后每次 {span(later)} s；恢复后单元 "
               f"{c.rng_n('e.h.cell_after_resume')}。`print(x)` 在 {m['e.h.state_kept']['n']} 次中有 "
               f"{m['e.h.state_kept']['count']} 次打印 42：内核里的变量随内存快照一起恢复。没有遗留沙箱。\n")
    return table(head, rows, ["---:"] * 7 + ["---"]) + tail


def reservation_table(c: Ctx, rel: str, key: str, host: str) -> str:
    r, m = c.runs[rel]["reservation"], c.m[key]
    head = [c.t("Sandboxes", "沙箱数"), c.t("Create, s", "创建，s"), c.t("`HugePages_Free`", "`HugePages_Free`"),
            c.t("`HugePages_Rsvd`", "`HugePages_Rsvd`"), c.t("Pages taken", "已占页数"),
            c.t("MemAvailable, MiB", "MemAvailable，MiB")]

    def taken(x: dict) -> int:
        return x["hugepages_total"] - x["hugepages_free"] + x["hugepages_rsvd"]

    rows = [[c.t("0 (before)", "0（开始前）"), "", str(r["before"]["hugepages_free"]), str(r["before"]["hugepages_rsvd"]),
             str(taken(r["before"])), str(r["before"]["mem_available_mb"])]]
    rows += [[str(s["sandboxes"]), num(s["create_s"]), str(s["hugepages_free"]), str(s["hugepages_rsvd"]),
              str(taken(s)), str(s["mem_available_mb"])] for s in r["steps"]]
    rows.append([c.t(f"{r['refused']['at']}: refused", f"{r['refused']['at']}：被拒"), num(r["refused"]["seconds"], 3),
                 "", "", "", ""])
    rows.append([c.t("0 (all killed)", "0（全部删除后）"), "", str(r["after"]["hugepages_free"]),
                 str(r["after"]["hugepages_rsvd"]), str(taken(r["after"])), str(r["after"]["mem_available_mb"])])
    left = m["pool"] - m["held_by_others"] - m["fit"] * m["per_sandbox"]
    mem = m["mem_available_mb"]
    tail = c.t(f"\nPages taken = `HugePages_Total` ({m['pool']}) − free + reserved; pages are 2 MiB. Each sandbox took "
               f"{m['per_sandbox']} pages, its whole {m['per_sandbox'] * 2} MiB, as soon as it was created; "
               f"{m['held_by_others']} pages were taken by other processes on host {host}. With {m['fit']} sandboxes, "
               f"{m['pool']} − {m['held_by_others']} − {m['fit']} × {m['per_sandbox']} = {left} pages were left, fewer than "
               f"one sandbox needs, and create {r['refused']['at']} got `500: Failed to place sandbox` in "
               f"{num(m['refused_s'], 3)} s. MemAvailable stayed at {min(mem)}–{max(mem)} MiB: the pool is set aside at "
               "boot, so sandbox memory does not show there.",
               f"\n已占页数 = `HugePages_Total`（{m['pool']}）− 空闲 + 预留；每页 2 MiB。每个沙箱一创建就占 "
               f"{m['per_sandbox']} 页，即它的全部 {m['per_sandbox'] * 2} MiB；host {host} 上其他进程占 "
               f"{m['held_by_others']} 页。有 {m['fit']} 个沙箱时只剩 {m['pool']} − {m['held_by_others']} − {m['fit']} × "
               f"{m['per_sandbox']} = {left} 页，不够再放一个，第 {r['refused']['at']} 次创建在 {num(m['refused_s'], 3)} s "
               "后收到 `500: Failed to place sandbox`。MemAvailable 一直在 "
               f"{min(mem)}–{max(mem)} MiB：大页池在开机时就划出去了，沙箱内存不体现在这里。")
    if m["taken_before"] != m["held_by_others"]:
        tail += c.t(f" The reading before the first create showed {m['taken_before']} pages taken, and after it still "
                    f"{m['taken'][0]}: the read-back sandbox, killed just before, had not yet given its pages back.",
                    f"第一次创建前读到已占 {m['taken_before']} 页，创建后仍是 {m['taken'][0]} 页：刚删除的读回沙箱"
                    "当时还没把它的页还回去。")
    return table(head, rows, ["---", "---:", "---:", "---:", "---:", "---:"]) + tail + "\n"


def v_e_host_reserve(c: Ctx) -> str:
    iso = c.m["e.h.isolation"]
    return (reservation_table(c, E_HOST, "e.h.reserve", "c")
            + c.t(f"\nRead-back: Firecracker `{iso['firecracker_version']}` behind the sandbox in the host's process "
                  f"list; inside it kernel `{iso['guest_kernel']}`, {iso['guest_cpus']} vCPU and MemTotal "
                  f"{iso['guest_mem_total_mb']} MiB for the template's {iso['template_memory_mb']} MB; host kernel "
                  f"`{iso['host_kernel']}`.\n",
                  f"\n读回：宿主机进程列表里沙箱背后是 Firecracker `{iso['firecracker_version']}`；沙箱内内核 "
                  f"`{iso['guest_kernel']}`、{iso['guest_cpus']} vCPU、MemTotal {iso['guest_mem_total_mb']} MiB"
                  f"（模板 {iso['template_memory_mb']} MB）；宿主机内核 `{iso['host_kernel']}`。\n"))


def v_e_cm_input(c: Ctx) -> str:
    cm = c.runs[E_MULTI]
    i, hosts = cm["inputs"], cm["hosts"]
    lines = [host_line(c, "a", hosts["a"],
                       "installed for `e-2026-10-14-agent`; the pool raised to 4096 pages, the build workspace row added, "
                       "the OS disk grown to 128 GB and `code-interpreter-v1` built in `e-2026-10-15-latest`",
                       "为 `e-2026-10-14-agent` 安装；在 `e-2026-10-15-latest` 中大页池调到 4096 页、加了构建工作空间那一行、"
                       "系统盘扩到 128 GB，并建好 `code-interpreter-v1`"),
             host_line(c, "c", hosts["c"], "the install of `e-2026-10-17-ci-agent`", "即 `e-2026-10-17-ci-agent` 的安装"),
             tool_line(c, cm, c.t("every phase run on host a", "每个阶段都在 host a 上运行")),
             c.t("Network: the tool reached host c's ports `3000` and `3002` over the private network; host c's NSG "
                 "admitted only host a's private address on those two ports and denied every other inbound connection. "
                 "Each host's API key came from an environment variable named in the hosts file.",
                 "网络：工具经私网访问 host c 的 `3000` 和 `3002` 端口；host c 的 NSG 只允许 host a 的私网地址访问这两个端口，"
                 "拒绝其他所有入站连接。每台主机的 API key 来自 hosts 文件里指定的环境变量。"),
             c.t(f"Agent runs on host a: \"{i['first']}\", then \"{i['count']}\" 3 times, pausing for {i['resume_wait_s']} s "
                 "after the second turn; agent instructions and tool schema as in `e-2026-10-14-agent`.",
                 f"host a 上的 agent 实测：\"{i['first']}\"，然后 3 次 \"{i['count']}\"，第二轮后暂停 {i['resume_wait_s']} s；"
                 "agent 指令和工具定义与 `e-2026-10-14-agent` 相同。"),
             c.t(f"At once, 4 sessions on host a alone, then 4 on each host: \"{i['scale'][0]}\", then \"{i['scale'][1]}\".",
                 f"并发：先只在 host a 上同时开 4 个会话，再在两台上各开 4 个：\"{i['scale'][0]}\"，然后 \"{i['scale'][1]}\"。"),
             c.t(f"Host stop: on each host one live session (\"{i['live'][0]}\") and one paused sandbox "
                 f"(\"{i['paused'][0]}\", `{{h}}` = the host's name); then `docker compose stop` on host a with the VM "
                 f"running, later `docker compose up -d --wait`; reads back \"{i['live'][1]}\" and \"{i['paused'][1]}\".",
                 f"停一台：每台各有一个存活会话（\"{i['live'][0]}\"）和一个暂停的沙箱（\"{i['paused'][0]}\"，`{{h}}` 为"
                 f"主机名）；然后在 host a 上执行 `docker compose stop`（VM 不关），之后再 `docker compose up -d --wait`；"
                 f"读回时发 \"{i['live'][1]}\" 和 \"{i['paused'][1]}\"。")]
    return quote(lines)


def v_e_cm_agent(c: Ctx) -> str:
    rr = c.runs[E_MULTI]["agent_runs_host_a"]
    tool_s = [r[t]["sandbox_tool_s"] for r in rr for t in CI_TURNS]
    calls = {r[t]["tool_calls"] for r in rr for t in CI_TURNS}
    kernels = sorted({r[t]["guest_kernel"] for r in rr for t in CI_TURNS})
    codes = Counter(code for r in rr for t in CI_TURNS for code in r[t]["http_statuses"])
    shown = ", ".join(f"{code} ×{n}" for code, n in sorted(codes.items()))
    slow = max(rr, key=lambda r: r["create_total_s"])
    rest = ", ".join(num(r["create_total_s"]) for r in rr if r is not slow)
    tail = c.t(f"\nHistory inside the sandbox grew 1, 2, 3, 4 in {c.m['e.m.a.state_kept']['count']} of {len(rr)} runs. "
               f"Run {slow['run']}'s create took {num(slow['create_total_s'])} s, the first `code-interpreter-v1` create "
               f"of this run on host a; the others took {rest} s. The record does not show where that time went. "
               f"{'Every turn made one' if calls == {1} else 'Turns made'} `run_code` call, "
               f"{num(min(tool_s), 3)}–{num(max(tool_s), 3)} s of the turn; guest kernel `{', '.join(kernels)}`. Model "
               "HTTP attempts: " + shown + ". No sandbox, conversation or agent version was left.\n",
               f"\n沙箱内的历史在 {len(rr)} 次中有 {c.m['e.m.a.state_kept']['count']} 次按 1、2、3、4 增长。第 "
               f"{slow['run']} 次创建用了 {num(slow['create_total_s'])} s，是本次在 host a 上第一次创建 "
               f"`code-interpreter-v1`；其余两次 {rest} s。记录里看不出这段时间花在哪里。"
               f"{'每一轮都只调用一次' if calls == {1} else '各轮调用'} `run_code`，占该轮 "
               f"{num(min(tool_s), 3)}–{num(max(tool_s), 3)} s；guest 内核 `{', '.join(kernels)}`。模型 HTTP 尝试："
               + shown + "。没有遗留沙箱、会话或 agent 版本。\n")
    return e_turn_table(c, rr, "e.m.a") + tail


def v_e_cm_scale(c: Ctx) -> str:
    head = [c.t("Phase", "阶段"), c.t("Sessions at once", "同时会话"), c.t("Answered", "回答"), c.t("Refused", "拒绝"),
            c.t("Sandbox create, s", "沙箱创建，s"), c.t("Whole session, s", "整个会话，s"),
            c.t("Model HTTP attempts", "模型 HTTP 尝试")]
    rows = []
    for key, label in (("e.m.one_host", c.t("host a alone", "仅 host a")),
                       ("e.m.two_hosts", c.t("hosts a and c", "host a 和 c"))):
        r = c.m[key]
        per = c.t(", ".join(f"{h} {v['answered']}/{v['sessions']}" for h, v in r["per_host"].items()),
                  "，".join(f"{h} {v['answered']}/{v['sessions']}" for h, v in r["per_host"].items()))
        codes = ", ".join(f"{code} ×{n}" for code, n in sorted(r["model_http_statuses"].items()))
        rows.append([label, str(r["n"]), f"{r['answered']}/{r['n']} ({per})", f"{r['refused']}/{r['n']}",
                     f"{num(min(r['create_s']), 3)}–{num(max(r['create_s']), 3)}",
                     f"{num(min(r['session_s']))}–{num(max(r['session_s']))}", codes])
    fit = c.m["e.h.reserve"]["fit"]
    tail = c.t(f"\nEvery refused session got `500: Failed to place sandbox` at create and never reached the model. At "
               f"`HUGEPAGES=4096` one host fits {fit} `code-interpreter-v1` sandboxes (`e-2026-10-18-ci-host`), so 4 on a "
               "host leaves one refused there; the second host added its own room. The client gave the sessions to the "
               "hosts in turn. No sandbox or conversation was left.\n",
               f"\n每个被拒的会话都在创建沙箱时收到 `500: Failed to place sandbox`，没有走到模型。`HUGEPAGES=4096` 时一台"
               f"主机能放 {fit} 个 `code-interpreter-v1` 沙箱（`e-2026-10-18-ci-host`），所以每台 4 个就有一个被拒；第二台"
               "主机带来它自己的空间。客户端把会话轮流分给两台主机。没有遗留沙箱或会话。\n")
    return table(head, rows, ["---", "---:", "---", "---", "---:", "---:", "---"]) + tail


def v_e_cs_input(c: Ctx) -> str:
    cs = c.runs[E_SCALE]
    h, i = cs["host"], cs["inputs"]
    lines = [c.t(f"Host d: a new `{h['vm_sku']}` ({h['vcpus']} vCPU, {h['memory_gib']} GiB), zone `{h['zone']}`, "
                 f"OS disk `{h['os_disk'].replace(', ', ' · ')}`, host kernel `{h['host_kernel']}`.",
                 f"Host d：新建的 `{h['vm_sku']}`（{h['vcpus']} vCPU、{h['memory_gib']} GiB），zone `{h['zone']}`，系统盘 "
                 f"`{h['os_disk'].replace(', ', ' · ')}`，宿主机内核 `{h['host_kernel']}`。"),
             c.t(f"Install: README step 7 as written, with one difference: `HUGEPAGES={h['hugepages']}` instead of 4096, "
                 "room for 8 sandboxes of 1024 pages plus 256; Docker Engine from Docker's apt repository; `compose.yaml` "
                 "and `.env` at `588e5b9` (SHA-256 OK); then the `project_limits` row, `sleep 300`, `code-interpreter` "
                 "at `162a2a2` and its unchanged `build_prod.py`.",
                 f"安装：按 README 第 7 步原样执行，只有一处不同：`HUGEPAGES={h['hugepages']}` 而不是 4096，可放 8 个"
                 "各 1024 页的沙箱再加 256 页余量；Docker Engine 来自 Docker 的 apt 源；`588e5b9` 的 `compose.yaml` 和 "
                 "`.env`（SHA-256 校验通过）；然后加 `project_limits` 那一行、`sleep 300`、`162a2a2` 的 "
                 "`code-interpreter` 及其未改动的 `build_prod.py`。"),
             tool_line(c, cs, c.t(f"`{cs['commands']['perf']}`, then `{cs['commands']['host']}`, on host d itself "
                                  "against `localhost`",
                                  f"在 host d 本机上对 `localhost` 先运行 `{cs['commands']['perf']}`，再运行 "
                                  f"`{cs['commands']['host']}`")),
             c.t(f"Perf prompts as in `e-2026-10-17-ci-agent`: new sessions \"{i['cold']}\" (n = 1–5); one session "
                 f"\"{i['warm'][0]}\", then \"{i['warm'][1]}\", then \"{i['warm'][2]}\", then \"{i['warm'][3]}\"; 4 and 8 "
                 f"sessions at once, each \"{i['scale'][0]}\", then \"{i['scale'][1]}\"; replacement: a live session ran "
                 f"\"{i['live'][0]}\" and, after another sandbox was killed and replaced, \"{i['live'][1]}\".",
                 f"性能问题与 `e-2026-10-17-ci-agent` 相同：新会话 \"{i['cold']}\"（n = 1–5）；同一会话 \"{i['warm'][0]}\"，"
                 f"然后 \"{i['warm'][1]}\"，再 \"{i['warm'][2]}\"，再 \"{i['warm'][3]}\"；4 和 8 个会话同时开始，每个发 "
                 f"\"{i['scale'][0]}\"，然后 \"{i['scale'][1]}\"；替换：一个存活会话先发 \"{i['live'][0]}\"，另一个沙箱被删除"
                 f"并替换后再发 \"{i['live'][1]}\"。"),
             c.t(f"Reservation: as in `e-2026-10-18-ci-host`, up to 9 sandboxes, each checked with `{i['reserve_cell']}`.",
                 f"预留：与 `e-2026-10-18-ci-host` 相同，最多 9 个沙箱，每个用 `{i['reserve_cell']}` 检查。")]
    return quote(lines)


def v_e_cs_install(c: Ctx) -> str:
    ins = c.runs[E_SCALE]["install"]
    t, it, b = ins["times_utc"], ins["install_time"], ins["build"]
    head = [c.t("Step", "步骤"), c.t("Start (UTC)", "开始（UTC）"), c.t("Seconds", "秒")]
    rows = [[c.t("Docker Engine from Docker's apt repository", "从 Docker 的 apt 源安装 Docker Engine"),
             t["start"][11:19], str(it["docker_engine_s"])],
            [c.t("download and check the two files, `docker compose up -d --wait`",
                 "下载并校验两个文件，`docker compose up -d --wait`"), t["compose_up_start"][11:19], str(it["compose_up_s"])],
            [c.t("`project_limits` row, then `sleep 300`", "加 `project_limits` 那一行，然后 `sleep 300`"),
             t["compose_up_done"][11:19], str(it["project_limits_row_and_wait_s"])],
            [c.t("clone `code-interpreter` at `162a2a2`, build environment", "克隆 `162a2a2` 的 `code-interpreter`，建构建环境"),
             t["sleep_done"][11:19], str(it["clone_and_build_env_s"])],
            [c.t(f"`build_prod.py`, {b['steps']} steps", f"`build_prod.py`，{b['steps']} 步"), t["build_start"][11:19],
             str(it["template_build_s"])],
            [c.t("**total**", "**合计**"), "", f"**{it['total_s']}**"]]
    tail = c.t(f"\n`psql` printed `{ins['project_limits_insert']}`; the build exited {b['exit_code']} on its first attempt, "
               f"its last builder step was {b['steps']}/{b['steps']} and it reported `Build finished, took "
               f"{b['builder_reported']}`.\n",
               f"\n`psql` 输出 `{ins['project_limits_insert']}`；构建第一次尝试就以退出码 {b['exit_code']} 结束，最后一步"
               f"是 {b['steps']}/{b['steps']}，并输出 `Build finished, took {b['builder_reported']}`。\n")
    return table(head, rows, ["---", "---", "---:"]) + tail


def v_e_cs_warm(c: Ctx) -> str:
    turns = c.runs[E_SCALE]["warm"]["turns"]
    slow = max(turns[1:], key=lambda t: t["wall_s"])
    k = turns.index(slow) + 1
    return v_e_ci_warm(c, E_SCALE, "e.s") + c.t(
        f"\nTurn {k} took {num(slow['wall_s'])} s, of which the sandbox tool took {num(slow['sandbox_tool_s'], 3)} s; "
        "the rest was the model's side of the turn.\n",
        f"\n第 {k} 轮用了 {num(slow['wall_s'])} s，其中沙箱工具只占 {num(slow['sandbox_tool_s'], 3)} s；其余是该轮模型那边的时间。\n")


def v_e_cs_reserve(c: Ctx) -> str:
    iso = c.m["e.s.isolation"]
    return (reservation_table(c, E_SCALE, "e.s.reserve", "d")
            + c.t(f"\nRead-back on host d: Firecracker `{iso['firecracker_version']}`, guest kernel `{iso['guest_kernel']}`, "
                  f"{iso['guest_cpus']} vCPU, MemTotal {iso['guest_mem_total_mb']} MiB, as on host c; host kernel "
                  f"`{iso['host_kernel']}`.\n",
                  f"\nhost d 上的读回：Firecracker `{iso['firecracker_version']}`，guest 内核 `{iso['guest_kernel']}`，"
                  f"{iso['guest_cpus']} vCPU，MemTotal {iso['guest_mem_total_mb']} MiB，与 host c 相同；宿主机内核 "
                  f"`{iso['host_kernel']}`。\n"))


# --- 2026-10-18 supplementary runs ---------------------------------------------------------------

C18 = "c-aks-agent-sandbox-2026-10-18/perf.json"
B_IDLE = ("b-sessions-2026-10-18/idle.json", "b-sessions-2026-10-18/idle-control.json")


def span(values: list) -> str:
    return f"{num(min(values))}–{num(max(values))}"


def state_counts(c: Ctx, states: list[str]) -> str:
    counts = Counter(states)
    return c.t(", ", "、").join(f"`{s}` {k}/{len(states)}" for s, k in sorted(counts.items()))


def idle_minutes(m: dict, key: str) -> int:
    return round(m[f"{key}.waited"]["median"] / 60)


def runs_20261018(c: Ctx) -> list[list[str]]:
    m = c.m
    a_create = m["a.c4"]["create_s"] + m["a.c8"]["create_s"]
    mins, mins5 = idle_minutes(m, "b.idle"), idle_minutes(m, "b.idle5")
    k = m["b.idle.state_kept"]
    dk = m["d.sess.state_kept"]
    return [
        ["`a-2026-10-18-scale`", "A", c.t("4 and 8 agent sandboxes at once", "同时创建 4 个和 8 个 agent 沙箱"),
         c.t(f"create → Running {span(a_create)} s with the full specification, 4 and 8 at once; none left",
             f"完整规格、同时创建 4 个和 8 个，创建 → Running {span(a_create)} s；无遗留")],
        ["`a-2026-10-18-novolume`", "A", c.t("full agent without the Blob volume, 3 runs", "不挂 Blob 卷的完整 agent，跑 3 次"),
         c.t(f"create → Running {num(m['a.nv.create']['median'])} s, resume → Running "
             f"{num(m['a.nv.resume']['median'])} s (medians); conversation kept across stop and resume "
             f"{m['a.nv.state_kept']['count']}/{m['a.nv.state_kept']['n']}; none left",
             f"创建 → Running {num(m['a.nv.create']['median'])} s、恢复 → Running "
             f"{num(m['a.nv.resume']['median'])} s（中位数）；停止和恢复后对话接着继续 "
             f"{m['a.nv.state_kept']['count']}/{m['a.nv.state_kept']['n']}；无遗留")],
        ["`b-2026-10-18-perf`", "B", c.t("D's protocol through the gateway: cold, warm, 4 and 8 at once",
                                         "经网关跑 D 口径：冷启动、后续、4 和 8 并发"),
         c.t("answer times and at-once answers only; see VALIDATION", "只测了回答耗时和并发回答，见 VALIDATION")],
        ["`b-2026-10-18-idle`", "B", c.t(f"next turn after {mins} min without requests, and after {mins5} min; 3 runs each",
                                         f"{mins} 分钟没有请求后的下一轮，以及 {mins5} 分钟后的对照；各跑 3 次"),
         c.t(f"status after {mins} min {state_counts(c, m['b.idle.state']['states'])}; file kept {k['files']}/{k['n']}",
             f"{mins} 分钟后状态 {state_counts(c, m['b.idle.state']['states'])}；文件保留 {k['files']}/{k['n']}")],
        ["`c-2026-10-18-perf`", "C", c.t("cold without and with the Blob PVC, pod loss, suspend/resume, 4 and 8 at once",
                                         "不挂卷和挂 Blob PVC 的冷启动、pod 丢失、挂起/恢复、4 和 8 并发"),
         c.t(f"apply → Ready {num(m['c.n.apply_ready']['median'])} s without the volume, "
             f"{num(m['c.p.apply_ready']['median'])} s with the PVC; pod lost → Ready {num(m['c.p.pod_lost']['median'])} s, "
             f"resume → Ready {num(m['c.p.resume_ready']['median'])} s (medians, with the PVC); conversation kept "
             f"{m['c.p.state_kept']['count']}/{m['c.p.state_kept']['n']}",
             f"apply → Ready 不挂卷 {num(m['c.n.apply_ready']['median'])} s、挂 PVC {num(m['c.p.apply_ready']['median'])} s；"
             f"pod 丢失 → Ready {num(m['c.p.pod_lost']['median'])} s、恢复 → Ready "
             f"{num(m['c.p.resume_ready']['median'])} s（中位数，挂 PVC）；对话保留 "
             f"{m['c.p.state_kept']['count']}/{m['c.p.state_kept']['n']}")],
        ["`d-2026-10-18-sessions`", "D", c.t("explicit session lifecycle, 3 runs", "显式会话生命周期，跑 3 次"),
         c.t(f"create → active {num(m['d.sess.create']['median'])} s (median); file kept {dk['files']}/{dk['n']}; "
             "a new session pod after each stop",
             f"创建 → active {num(m['d.sess.create']['median'])} s（中位数）；文件保留 {dk['files']}/{dk['n']}；"
             "每次停止后换一个新的会话 pod")],
    ]


def v_a_scale(c: Ctx) -> str:
    a = c.runs["a-aca-2026-10-18/scale.json"]
    head = [c.t("At once", "并发"), c.t("Answered", "回答"), c.t("Create → Running, s", "创建 → Running，s"),
            c.t("Read turn inside the sandbox, s", "沙箱内的读取轮次，s"), c.t("All sessions, s", "全部会话，s")]
    rows = []
    for lv in a["levels"]:
        r = c.m[f"a.c{lv['sessions_at_once']}"]
        rows.append([str(r["n"]), f"{r['answered']}/{r['n']}", span(r["create_s"]), span(r["read_s"]), num(r["wall_s"])])
    vmm = ", ".join(f"`{v}`" for v in c.m["a.c8"]["vmm"])
    tail = c.t(f"\nEvery sandbox's second answer was its own `SCALE-<k>`; `vmm_type` {vmm} for all; every sandbox was "
               "deleted and none was left. All sessions: from the first create call until the last sandbox was "
               "deleted.\n",
               f"\n每个沙箱的第二个回答都是它自己的 `SCALE-<k>`；`vmm_type` 都是 {vmm}；所有沙箱都已删除，没有遗留。"
               "全部会话：从第一个创建调用到最后一个沙箱删除。\n")
    return table(head, rows, ["---:", "---", "---:", "---:", "---:"]) + tail


def v_b_perf_input(c: Ctx) -> str:
    b = c.runs["b-hosted-agent-2026-10-18/perf.json"]
    i, ag = b["inputs"], b["agent"]
    lines = [c.t(f"Request: {b['request']}; {b['token']}.",
                 f"请求：{b['request']}；认证：{b['token']}。"),
             c.t(f"Agent: `{ag['name']}` version {ag['version']} ({ag['cpu']} vCPU, {ag['memory']}, "
                 f"`MODEL_ROUTING={ag['model_routing']}`, `{ag['model']}`); {ag['version_note']}.",
                 f"Agent：`{ag['name']}` 版本 {ag['version']}（{ag['cpu']} vCPU、{ag['memory']}、"
                 f"`MODEL_ROUTING={ag['model_routing']}`、`{ag['model']}`）；{ag['version_note']}。"),
             c.t(f"New sessions: \"{i['cold']}\" (n = 1–5)", f"新会话：\"{i['cold']}\"（n = 1–5）"),
             c.t(f"One session: \"{i['warm_first']}\", then \"{i['warm_followup_1']}\", then \"{i['warm_followup_n']}\" "
                 "(n = 1, 2)",
                 f"同一会话：\"{i['warm_first']}\"，然后 \"{i['warm_followup_1']}\"，再 \"{i['warm_followup_n']}\"（n = 1、2）"),
             c.t(f"At once, per session: \"{i['scale_turn1']}\", then \"{i['scale_turn2']}\" (4, then 8 sessions)",
                 f"并发时每个会话：\"{i['scale_turn1']}\"，然后 \"{i['scale_turn2']}\"（先 4 个，再 8 个会话）"),
             c.t(f"Client: {b['client']}.", f"客户端：{b['client']}。")]
    return quote(lines)


def v_b_perf(c: Ctx) -> str:
    b = c.runs["b-hosted-agent-2026-10-18/perf.json"]
    head1 = [c.t("Start (UTC)", "开始（UTC）"), c.t("Answer", "回答"), c.t("New session → answer, s", "新会话 → 回答，s"),
             c.t("First byte, s", "首字节，s")]
    rows1 = [[t["started_utc"][:8], t["answer"], num(t["wall_s"]), num(t["first_byte_s"])] for t in b["cold_turns"]]
    head2 = [c.t("Turn", "轮次"), c.t("Answer", "回答"), c.t("Client wall, s", "客户端耗时，s")]
    rows2 = [[c.t("1 (new session)", "1（新会话）") if k == 0 else str(k + 1), t["answer"], num(t["wall_s"])]
             for k, t in enumerate(b["warm_turns"])]
    head3 = [c.t("At once", "并发"), c.t("Answered", "回答"), c.t("Read back its word", "读回自己的词"),
             c.t("Write turn, s", "写入轮次，s")]
    rows3 = []
    for lv in b["scalability"]:
        r = c.m[f"b.d.c{lv['concurrency']}"]
        rows3.append([str(r["n"]), f"{r['answered']}/{r['n']}", f"{r['remembered']}/{r['n']}", span(r["turn1_s"])])
    return (c.t("New sessions:\n\n", "新会话：\n\n") + table(head1, rows1, ["---", "---", "---:", "---:"])
            + c.t(f"\n{c.med('b.d.cold')}.\n\nFollow-ups in one session:\n\n",
                  f"\n{c.med('b.d.cold')}。\n\n同一会话的后续回答：\n\n")
            + table(head2, rows2, ["---", "---", "---:"])
            + c.t(f"\nFollow-ups 2–4: {c.med('b.d.warm')}.\n\nAt once:\n\n", f"\n后续第 2–4 轮：{c.med('b.d.warm')}。\n\n并发：\n\n")
            + table(head3, rows3, ["---:", "---", "---", "---:"])
            + c.t("\nEvery request returned HTTP 200; the tool sends each request once.\n",
                  "\n每个请求都返回 HTTP 200；工具对每个请求只发一次。\n"))


def v_b_idle(c: Ctx) -> str:
    m = c.m
    head = [c.t("Wait, s", "等待，s"), c.t("Run", "次"), c.t("Status after the wait", "等待后的状态"),
            c.t("Follow-up before the wait", "等待前的后续"), c.t("Next turn", "下一轮"),
            c.t("Service time of that turn", "该轮服务端耗时"), c.t("Turn after that", "再下一轮"),
            c.t("File kept", "文件保留"), c.t("Answers", "回答")]
    rows = []
    for rel, key in zip(B_IDLE, ("b.idle", "b.idle5")):
        for r in c.runs[rel]["runs"]:
            answers = " / ".join(str(r[x]["response"]) for x in ("first", "warm", "resume", "warm_after_resume"))
            rows.append([num(r["idle"]["waited_s"]), str(r["run"]), f"`{r['idle']['state_before_resume']}`",
                         num(r["warm"]["total_s"]), num(r["resume"]["total_s"]), num(r["resume"]["server_s"]),
                         num(r["warm_after_resume"]["total_s"]),
                         c.t("yes", "是") if r["marker_after_resume"]["matches"] is True else c.t("no", "否"), answers])
        rows.append([f"**{num(m[key + '.waited']['median'])}**", c.t("**median**", "**中位数**"), ""]
                    + [f"**{num(m[f'{key}.{x}']['median'])}**" for x in ("warm", "next", "next_server", "after")]
                    + ["", ""])
    return c.t("Seconds, client wall time; the service time is the response's `completed_at` minus `created_at` (whole "
               "seconds); the last column is the agent's replies in order.\n\n",
               "单位为秒，客户端计时；服务端耗时是响应的 `completed_at` 减 `created_at`（整秒）；最后一列是 agent 依次给出的回答。\n\n") \
        + table(head, rows, ["---:", "---", "---", "---:", "---:", "---:", "---:", "---", "---"])


def v_c18_input(c: Ctx) -> str:
    r = c.runs[C18]
    cl, i, res = r["cluster"], r["inputs"], r["resources"]
    lines = [c.t(f"Cluster: {cl['note']}; Kubernetes {cl['kubernetes']}, {cl['tier']} tier; system pool "
                 f"{cl['system_pool']}; Kata pool {cl['kata_pool']}; agent-sandbox {cl['agent_sandbox']}.",
                 f"集群：{cl['note']}；Kubernetes {cl['kubernetes']}，{cl['tier']} 层级；系统节点池 {cl['system_pool']}；"
                 f"Kata 节点池 {cl['kata_pool']}；agent-sandbox {cl['agent_sandbox']}。"),
             c.t(f"Each test Sandbox copies the spec of the workshop's `{r['template']}` Sandbox in namespace "
                 f"`{r['namespace']}` under a new name and the label `app=agent-host-measure`: image `{r['image']}`, "
                 f"runtime class `{r['runtime_class']}`, requests {res['requests']['cpu']} CPU and "
                 f"{res['requests']['memory']}, limits {res['limits']['cpu']} CPU and {res['limits']['memory']}. "
                 "`AGENT_ID` comes from the pod name, so each copy keeps its conversation in its own state file.",
                 f"每个测试 Sandbox 复制命名空间 `{r['namespace']}` 里 workshop 的 `{r['template']}` Sandbox 的 spec，"
                 f"换一个新名字并加标签 `app=agent-host-measure`：镜像 `{r['image']}`，运行时类 `{r['runtime_class']}`，"
                 f"requests {res['requests']['cpu']} CPU、{res['requests']['memory']}，limits {res['limits']['cpu']} CPU、"
                 f"{res['limits']['memory']}。`AGENT_ID` 取自 pod 名，所以每个副本把对话存在自己的状态文件里。"),
             c.t(f"Request: {i['request']}.", f"请求：{i['request']}。"),
             c.t(f"First answer: \"{i['first']}\"", f"第一个问题：\"{i['first']}\""),
             c.t(f"Every later turn: \"{i['follow_up']}\"; with the PVC once more after the pod was deleted and once "
                 "after suspend/resume",
                 f"之后每一轮：\"{i['follow_up']}\"；挂 PVC 时在删除 pod 后、挂起/恢复后各再问一次"),
             c.t(f"At once, per Sandbox without the volume: \"{i['scale_write']}\", then \"{i['scale_read']}\" "
                 "(4, then 8 Sandboxes)",
                 f"并发时每个 Sandbox（不挂卷）：\"{i['scale_write']}\"，然后 \"{i['scale_read']}\"（先 4 个，再 8 个 Sandbox）"),
             c.t(f"Client: {r['client']}.", f"客户端：{r['client']}。")]
    return quote(lines)


def v_c18_runs(c: Ctx) -> str:
    r = c.runs[C18]
    head = ["Sandbox", c.t("Volume", "卷"), c.t("`kubectl apply` → Ready, s", "`kubectl apply` → Ready，s"),
            c.t("First answer, s", "第一个回答，s"), c.t("Cold total, s", "冷启动合计，s"), c.t("Follow-up, s", "后续，s"),
            c.t("Conversation length", "对话长度")]
    rows, extra = [], []
    for x in r["cold"] + r["pvc"]:
        keys = [k for k in ("first", "follow_up", "after_pod_delete", "after_resume") if k in x]
        extra += [x[k]["exec_wall_s"] - x[k]["guest_s"] for k in keys]
        rows.append([x["sandbox"], "Blob PVC" if x["pvc"] else c.t("none", "无"), num(x["apply_to_ready_s"]),
                     num(x["first"]["guest_s"]), num(x["cold_total_s"]), num(x["follow_up"]["guest_s"]),
                     ", ".join(str(x[k]["history_length"]) for k in keys)])
    tail = c.t(f"\nWithout the volume: apply → Ready {c.med('c.n.apply_ready')}, first answer {c.med('c.n.first')}, "
               f"cold total {c.med('c.n.cold')}. With the Blob PVC: apply → Ready {c.med('c.p.apply_ready')}, first "
               f"answer {c.med('c.p.first')}, cold total {c.med('c.p.cold')}. Follow-ups with the PVC: "
               f"{c.vals('c.p.warm')}; the record does not show where "
               "the slow one spent its time. Answers are timed inside the pod; the `kubectl exec` round trip around each "
               f"added {span(extra)} s on the client and is excluded.\n",
               f"\n不挂卷：apply → Ready {c.med('c.n.apply_ready')}，第一个回答 {c.med('c.n.first')}，冷启动合计 "
               f"{c.med('c.n.cold')}。挂 Blob PVC：apply → Ready {c.med('c.p.apply_ready')}，第一个回答 "
               f"{c.med('c.p.first')}，冷启动合计 {c.med('c.p.cold')}。"
               f"挂 PVC 时的后续回答：{c.vals('c.p.warm')}；记录里看不出慢的那一次时间花在哪里。"
               f"回答在 pod 内计时；外层的 `kubectl exec` 往返每次在客户端多花 {span(extra)} s，不计入。\n")
    return table(head, rows, ["---", "---", "---:", "---:", "---:", "---:", "---"]) + tail


def v_c18_pvc(c: Ctx) -> str:
    r, m = c.runs[C18], c.m
    head = ["Sandbox", c.t("Pod deleted → Ready, s", "删除 pod → Ready，s"), c.t("Answer, s", "回答，s"),
            c.t("Suspended → pod gone, s", "Suspended → pod 消失，s"), c.t("Running → Ready, s", "Running → Ready，s"),
            c.t("Answer, s", "回答，s"), c.t("Resume → answer, s", "恢复 → 回答，s")]
    rows = [[x["sandbox"], num(x["pod_delete_to_ready_s"]), num(x["after_pod_delete"]["guest_s"]),
             num(x["suspend_to_pod_gone_s"]), num(x["resume_to_ready_s"]), num(x["after_resume"]["guest_s"]),
             num(x["resume_to_answer_s"])] for x in r["pvc"]]
    rows.append([c.t("**median**", "**中位数**"), f"**{num(m['c.p.pod_lost']['median'])}**", "",
                 f"**{num(m['c.p.suspend']['median'])}**", f"**{num(m['c.p.resume_ready']['median'])}**", "",
                 f"**{num(m['c.p.resume_answer']['median'])}**"])
    reasons = sorted({x["suspended_condition"]["reason"] for x in r["pvc"]})
    tail = c.t(f"\nWhile suspended, the Sandbox's Ready condition read `False` ({', '.join(f'`{x}`' for x in reasons)}) "
               "in every run. The conversation length kept growing after the pod was deleted and after suspend/resume "
               "(last column of the table above), so the state came back from the Blob PVC each time. Resume → answer "
               "is the Running patch until Ready plus the answer inside the pod.\n",
               f"\n挂起期间，每次 Sandbox 的 Ready 条件都是 `False`（{'、'.join(f'`{x}`' for x in reasons)}）。删除 pod 后、"
               "挂起/恢复后对话长度都接着增长（见上表最后一列），说明状态每次都从 Blob PVC 恢复。恢复 → 回答是 Running "
               "patch 到 Ready 再加 pod 内的回答。\n")
    return table(head, rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:"]) + tail


def v_c18_scale(c: Ctx) -> str:
    r = c.runs[C18]
    start = {n["name"] for n in r["kata_nodes_at_start"]}
    head = [c.t("At once", "并发"), c.t("Answered", "回答"), c.t("Apply → Ready, s", "Apply → Ready，s"),
            c.t("Sandboxes per Kata node", "每个 Kata 节点上的 Sandbox"), c.t("All Sandboxes, s", "全部 Sandbox，s")]
    rows = []
    for lv in r["scale"]:
        s = c.m[f"c.c{lv['sessions_at_once']}"]
        per = sorted(Counter(x["server"]["node"] for x in lv["sessions"]).items())
        rows.append([str(s["n"]), f"{s['answered']}/{s['n']}", span(s["ready_s"]),
                     c.t(", ".join(f"{k}: {v}" for k, v in per), "、".join(f"{k}：{v}" for k, v in per)), num(s["wall_s"])])
    lv8 = next(lv for lv in r["scale"] if lv["sessions_at_once"] == 8)
    late = [x["apply_to_ready_s"] for x in lv8["sessions"] if x["server"]["node"] not in start]
    on_first = len(lv8["sessions"]) - len(late)
    tail = c.t(f"\nThe run started with {len(start)} Kata node, which also ran the workshop's own Sandbox. With 8 at once, "
               f"{on_first} test Sandboxes fit there (each requests {r['resources']['requests']['cpu']} CPU); the other "
               f"{len(late)} ran on a node the cluster autoscaler added during the run and were Ready after {span(late)} s. "
               f"Kata nodes at the end: {len(r['kata_nodes_at_end'])}. Every Sandbox answered with its own `SCALE-<k>` "
               "and was deleted.\n",
               f"\n开始时只有 {len(start)} 个 Kata 节点，workshop 自己的 Sandbox 也在上面。8 个同时开时，{on_first} 个测试 Sandbox "
               f"放进了这个节点（每个 requests {r['resources']['requests']['cpu']} CPU）；另外 {len(late)} 个跑在集群自动扩缩容"
               f"在测试中新加的节点上，{span(late)} s 后才就绪。结束时 Kata 节点数：{len(r['kata_nodes_at_end'])}。每个 Sandbox "
               "都回答了自己的 `SCALE-<k>`，并已删除。\n")
    return table(head, rows, ["---:", "---", "---:", "---", "---:"]) + tail


def v_d_sessions(c: Ctx) -> str:
    p = c.m["d.sess.pods"]
    tail = c.t(f"\nSession pods during the three runs: {p['n']} new pods, {p['per_session']} per session, each Running "
               f"{span(p['running_after_s'])} s after it appeared (pod age at the change). Both pods of a session ended "
               "(Terminating, then Completed); their ages fit `:stop` ending the first and the delete ending the second. "
               "`kubectl get pods -w` records order and ages, not timestamps.\n",
               f"\n三次实测中的会话 pod：新出现 {p['n']} 个，每个会话 {p['per_session']} 个，每个出现后 "
               f"{span(p['running_after_s'])} s 进入 Running（变化时的 pod 年龄）。同一会话的两个 pod 都结束了"
               "（Terminating，然后 Completed）；年龄与 `:stop` 结束第一个、删除会话结束第二个相符。"
               "`kubectl get pods -w` 只记录顺序和年龄，没有时间戳。\n")
    return session_table(c, "d-byoaks-2026-10-18/sessions.json", "d.sess") + tail


BLOCKS = {
    "comparison": b_comparison, "runs": b_runs,
    "a-agent-input": v_a_agent_input, "a-agent": v_a_agent, "a-attr": v_a_attr, "a-late": v_a_late,
    "a-setup": v_a_setup, "b-run": v_b, "b-sessions": v_b_sessions, "c-run": v_c, "d-input": v_d_input, "d-cold": v_d_cold,
    "d-warm": v_d_warm, "d-scale": v_d_scale, "d-recovery": v_d_recovery, "d-placement": v_d_placement,
    "e-agent-input": v_e_agent_input, "e-agent": v_e_agent, "e-perf-input": v_e_perf_input,
    "e-cold": v_e_cold, "e-warm": v_e_warm, "e-scale": v_e_scale,
    "e-latest-input": v_e_latest_input, "e-latest-sdk": v_e_latest_sdk, "e-latest-build": v_e_latest_build,
    "e-latest-ci": v_e_latest_ci,
    "e-multi-input": v_e_multi_input, "e-multi-agent": v_e_multi_agent, "e-multi-scale": v_e_multi_scale,
    "e-multi-idle": v_e_multi_idle, "e-multi-hostloss": v_e_multi_hostloss,
    "e-ci-input": v_e_ci_input, "e-ci-agent": v_e_ci_agent, "e-ci-build": v_e_ci_build, "e-ci-cold": v_e_ci_cold,
    "e-ci-warm": v_e_ci_warm, "e-ci-scale": v_e_ci_scale, "e-ci-idle": v_e_ci_idle,
    "e-host-input": v_e_host_input, "e-host-bench": v_e_host_bench, "e-host-reserve": v_e_host_reserve,
    "e-cm-input": v_e_cm_input, "e-cm-agent": v_e_cm_agent, "e-cm-scale": v_e_cm_scale,
    "e-cm-hostloss": lambda c: v_e_multi_hostloss(c, E_MULTI, "c"),
    "e-cs-input": v_e_cs_input, "e-cs-install": v_e_cs_install,
    "e-cs-cold": lambda c: v_e_ci_cold(c, E_SCALE, "e.s"), "e-cs-warm": v_e_cs_warm,
    "e-cs-scale": lambda c: v_e_ci_scale(c, E_SCALE, "e.s"), "e-cs-reserve": v_e_cs_reserve,
    "a-scale": v_a_scale, "a-nv-input": lambda c: v_a_agent_input(c, A_NV), "a-nv": lambda c: v_a_agent(c, A_NV, "a.nv"),
    "b-perf-input": v_b_perf_input, "b-perf": v_b_perf, "b-idle": v_b_idle,
    "c18-input": v_c18_input, "c18-runs": v_c18_runs, "c18-pvc": v_c18_pvc, "c18-scale": v_c18_scale,
    "d-sessions": v_d_sessions,
}


def value(metrics: dict, key: str) -> str:
    metric, _, field = key.rpartition(".")
    if metric not in metrics:
        raise RenderError(f"inline value {key}: unknown metric {metric}")
    rec = metrics[metric]
    if field == "range":
        return rng(rec)
    if field not in rec:
        raise RenderError(f"inline value {key}: metric has no field {field}")
    v = rec[field]
    return v if isinstance(v, str) else num(v, 3 if rec.get("unit") == "USD/h" else 2)


def render_page(text: str, lang: str, metrics: dict, runs: dict) -> str:
    c = Ctx(metrics, runs, lang)

    def block(mo: re.Match) -> str:
        name = mo.group(2)
        if name not in BLOCKS:
            raise RenderError(f"unknown generated block {name}")
        return mo.group(1) + BLOCKS[name](c) + mo.group(4)

    text = BLOCK_RE.sub(block, text)
    return VALUE_RE.sub(lambda mo: f"<!--v:{mo.group(1)}-->{value(metrics, mo.group(1))}<!--/v-->", text)


def load_inputs() -> tuple[dict, dict]:
    runs, index = bm.load_runs()
    fresh = bm.render(bm.build(runs, index))
    if not bm.OUT.exists() or bm.OUT.read_text(encoding="utf-8") != fresh:
        raise RenderError("evidence/measurements.json differs from a fresh build; run tools/build_measurements.py")
    return json.loads(fresh)["metrics"], runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    args = ap.parse_args(argv)
    try:
        metrics, runs = load_inputs()
        changed = []
        for name, lang in PAGES.items():
            path = ROOT / name
            old = path.read_text(encoding="utf-8")
            new = render_page(old, lang, metrics, runs)
            if new != old:
                changed.append(name)
                if not args.check:
                    path.write_text(new, encoding="utf-8", newline="\n")
    except (RenderError, bm.BuildError, KeyError, FileNotFoundError) as exc:
        print(f"RENDER FAIL {exc}", file=sys.stderr)
        return 1
    if args.check and changed:
        print(f"RENDER FAIL {', '.join(changed)} differ from a fresh render; run tools/render_docs.py", file=sys.stderr)
        return 1
    print(f"RENDER {'PASS' if args.check else 'WRITTEN'} {len(PAGES)} pages" + (f", updated {changed}" if changed and not args.check else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
