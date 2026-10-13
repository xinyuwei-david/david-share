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
import json
import re
import sys
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
    """Seconds keep two decimals (trailing zeros dropped, one kept); prices pass places=3."""
    if isinstance(v, bool):
        raise RenderError(f"boolean where a number was expected: {v}")
    if isinstance(v, int):
        return str(v)
    if float(v).is_integer() and abs(v) >= 100:
        return str(int(v))
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

    def vals(self, key: str) -> str:
        r = self.m[key]
        joined = ", ".join(num(v) for v in r["values"])
        return f"{joined} s (n={r['n']})" if self.lang == "en" else f"{joined} s（n={r['n']}）"

    def one(self, key: str) -> str:
        r = self.m[key]
        return self.t(f"{num(r['value'])} s (n=1)", f"{num(r['value'])} s（n=1）")


def table(head: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(head)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def quote(lines: list[str]) -> str:
    return "\n".join(f"> {ln}" for ln in lines) + "\n"


# --- README blocks --------------------------------------------------------------------------------

def b_comparison(c: Ctx) -> str:
    m = c.m
    claims = json.loads((ROOT / "evidence" / "claims.json").read_text(encoding="utf-8"))
    head = [c.t("Observed or documented", "实测或文档项"), "A · ACA Sandboxes",
            c.t("B · Foundry hosted agent", "B · Foundry hosted agent"),
            "C · AKS + agent-sandbox", "D · BYO-AKS"]
    rows = [
        [c.t("**New session → first answer**", "**新会话 → 第一个回答**"),
         c.t(c.med("a.cold") + "; SDK create + answer inside sandbox",
             c.med("a.cold") + "；SDK 创建 + 沙箱内回答"),
         c.t(f"{num(m['b.cold_gateway']['value'])} s (n=1, via the Foundry endpoint; model route: gateway (APIM))",
             f"{num(m['b.cold_gateway']['value'])} s（n=1，经 Foundry 端点；模型路由：gateway（APIM））"),
         c.t(f"{num(m['c.cold']['value'])} s (n=1; apply → Ready {num(m['c.apply_ready']['value'])} s + answer "
             f"{num(m['c.first']['value'])} s)",
             f"{num(m['c.cold']['value'])} s（n=1；apply → Ready {num(m['c.apply_ready']['value'])} s + 回答 "
             f"{num(m['c.first']['value'])} s）"),
         c.t(c.med("d.cold") + ", via the Foundry endpoint", c.med("d.cold") + "，经 Foundry 端点")],
        [c.t("**Follow-up turn, same session**", "**同一会话的后续回答**"),
         c.med("a.warm"), c.t(c.vals("b.warm_gateway") + "; model route: gateway (APIM)",
                             c.vals("b.warm_gateway") + "；模型路由：gateway（APIM）"),
         c.one("c.warm"), c.med("d.warm")],
        [c.t("**Sandbox or pod until ready**", "**沙箱或 pod 就绪**"),
         c.t(c.med("a.create") + ", create → Running", c.med("a.create") + "，create → Running"),
         c.t("not visible (managed)", "不可见（平台托管）"),
         c.t(f"{num(m['c.apply_ready']['value'])} s (n=1), apply → Ready",
             f"{num(m['c.apply_ready']['value'])} s（n=1），apply → Ready"),
         c.t(f"{rng(m['d.pod_ready'])} s (n={m['d.pod_ready']['n']}) with the image cached; "
             f"{num(m['d.pod_ready_fresh']['min'])} s on a fresh node (n={m['d.pod_ready_fresh']['n']})",
             f"镜像已缓存 {rng(m['d.pod_ready'])} s（n={m['d.pod_ready']['n']}）；新节点 "
             f"{num(m['d.pod_ready_fresh']['min'])} s（n={m['d.pod_ready_fresh']['n']}）")],
        [c.t("**Resume request → first answer (stop excluded)**", "**发起恢复 → 第一个回答（不含停止）**"),
         c.t(c.med("a.recover") + f"; separately reported resume-only median {num(m['a.resume']['median'])} s",
             c.med("a.recover") + f"；恢复本身另行统计的中位数 {num(m['a.resume']['median'])} s"),
         c.t("not exercised (the platform restores $HOME)", "未测（平台恢复 $HOME）"),
         c.t(f"{num(m['c.resume_answer']['value'])} s (n=1; Ready {num(m['c.resume_ready']['value'])} s, "
             f"LoadBalancer healthy {num(m['c.resume_lb']['value'])} s)",
             f"{num(m['c.resume_answer']['value'])} s（n=1；Ready {num(m['c.resume_ready']['value'])} s，"
             f"LoadBalancer 健康 {num(m['c.resume_lb']['value'])} s）"),
         claims["paths"]["D"][f"memory_suspend_{c.lang}"]],
        [c.t("**Pod lost → ready again**", "**pod 丢失 → 重新就绪**"),
         c.t("not exercised (service-managed)", "未测（服务托管）"),
         c.t("not exercised (managed)", "未测（平台托管）"),
         c.t(c.one("c.pod_lost") + "; same Sandbox state restored",
             c.one("c.pod_lost") + "；同一 Sandbox 的状态已恢复"),
         c.t(c.one("d.pod_lost") + "; deleted session continuity not tested",
             c.one("d.pod_lost") + "；未测试被删会话的连续性")],
        [c.t("**Idle policy (official docs)**", "**空闲策略（官方说明）**"),
         *(claims["paths"][letter][f"idle_policy_{c.lang}"] for letter in ("A", "B", "C", "D"))],
        [c.t("**Capacity**", "**容量**"),
         c.t("serverless; quota is concurrent cores (docs)", "无服务器；配额按并发核数计（文档）"),
         c.t("managed, per session", "平台托管，按会话"),
         c.t("your Kata node pool", "你的 Kata 节点池"),
         c.t(f"observed at most {m['d.capacity']['per_node']} per {m['d.capacity']['node_sku']} node; "
             f"{m['d.capacity']['max_sessions']} on {m['d.capacity']['n']} nodes; "
             + ", ".join(f"`{code}`" for n in (4, 8) for code in m[f'd.c{n}']['refused_codes']),
             f"单个 {m['d.capacity']['node_sku']} 节点观察到最多 {m['d.capacity']['per_node']} 个会话；"
             f"{m['d.capacity']['n']} 个节点共 {m['d.capacity']['max_sessions']} 个；"
             + "、".join(f"`{code}`" for n in (4, 8) for code in m[f'd.c{n}']['refused_codes']))],
        [c.t("**Path setup in our run**", "**本次路径部署耗时**"),
         c.t(f"{num(m['a.setup']['value'])} s (sandbox group, VNet connection, disk image, volume, sandbox)",
             f"{num(m['a.setup']['value'])} s（sandbox group、VNet 连接、磁盘镜像、卷、沙箱）"),
         c.t(f"{num(m['b.deploy']['value'])} s (`azd deploy`; model route: gateway (APIM))",
             f"{num(m['b.deploy']['value'])} s（`azd deploy`；模型路由：gateway（APIM））"),
         c.t(f"{num(m['c.setup']['value'])} s (prepare script, private link, apply)",
             f"{num(m['c.setup']['value'])} s（准备脚本、私有链接、apply）"),
         c.t(f"{num(m['d.onboarding']['value'])} s (roles, `azd provision`, `azd deploy`; author's console, not kept)",
             f"{num(m['d.onboarding']['value'])} s（角色、`azd provision`、`azd deploy`；作者控制台记录，原始输出未保留）")],
    ]
    spec = c.runs["a-aca-2026-10-09/agent-runs.json"]["sandbox_spec"]
    observation = c.t(
        f"**Local configuration and observation, not official idle policy:** A used "
        f"{spec['auto_suspend_mode']} mode with a {spec['auto_suspend_seconds']} s auto-suspend setting. "
        f"D's oldest pod was still Running at least {m['d.idle']['value']} min after creation; "
        "this is pod age, not idle duration or a reclamation timeout.",
        f"**本地配置与观察，不是官网空闲策略：** A 本次使用 {spec['auto_suspend_mode']} 模式，"
        f"自动挂起设置为 {spec['auto_suspend_seconds']} s。D 最早的 pod 在创建至少 "
        f"{m['d.idle']['value']} min 后仍为 Running；这是 pod 的年龄，不是空闲时长或回收超时。")
    return table(head, rows) + "\n" + observation + "\n"


def b_runs(c: Ctx) -> str:
    m = c.m
    head = [c.t("Run", "实测"), c.t("Path", "路径"), c.t("Scenario", "场景"), c.t("Result", "结果")]
    rows = [
        ["`a-2026-10-09-agent`", "A", c.t("full agent, 3 runs", "完整 agent，跑 3 次"),
         c.t(f"cold {num(m['a.cold']['median'])} s, warm {num(m['a.warm']['median'])} s, resume → answer "
             f"{num(m['a.recover']['median'])} s (medians); vmm `{m['a.vmm']['value']}` in "
             f"{m['a.vmm']['count']}/{m['a.vmm']['n']}; history grew {m['a.state_kept']['count']}/{m['a.state_kept']['n']}",
             f"冷启动 {num(m['a.cold']['median'])} s、后续 {num(m['a.warm']['median'])} s、恢复到回答 "
             f"{num(m['a.recover']['median'])} s（中位数）；{m['a.vmm']['count']}/{m['a.vmm']['n']} 次 vmm 为 "
             f"`{m['a.vmm']['value']}`；历史持续增长 {m['a.state_kept']['count']}/{m['a.state_kept']['n']}")],
        ["`a-2026-10-09-attribution`", "A", c.t("create time by configuration", "按配置拆分创建耗时"),
         c.t(f"image, VNet or port alone {rng(m['a.attr.small'])} s; "
             f"adding the private Blob volume {rng(m['a.attr.vnetvolume'])} s",
             f"只加镜像、VNet 或端口 {rng(m['a.attr.small'])} s；"
             f"加私有 Blob 卷后 {rng(m['a.attr.vnetvolume'])} s")],
        ["`a-2026-10-09-late-mount`", "A", c.t("start first, mount Blob later", "先启动、后挂 Blob"),
         c.t(f"start {rng(m['a.late.create'])} s, then mount {rng(m['a.late.mount'])} s (n=2)",
             f"启动 {rng(m['a.late.create'])} s，再挂载 {rng(m['a.late.mount'])} s（n=2）")],
        ["`a-2026-09-27-setup`", "A", c.t("first deployment, wake on request", "首次部署，请求唤醒"),
         c.t(f"setup {num(m['a.setup']['value'])} s; a request to a stopped sandbox answered in "
             f"{num(m['a.wake_http']['value'])} s; kernel `{m['a.isolation']['guest_kernel']}`",
             f"部署 {num(m['a.setup']['value'])} s；向已停止的沙箱发请求，{num(m['a.wake_http']['value'])} s 后回答；"
             f"内核 `{m['a.isolation']['guest_kernel']}`")],
        ["`b-2026-09-27`", "B", c.t("deploy, new session, follow-ups; gateway model route",
                                   "部署、新会话、后续回答；gateway 模型路由"),
         c.t(f"deploy {num(m['b.deploy']['value'])} s; new session {num(m['b.cold_gateway']['value'])} s; "
             f"follow-ups {', '.join(num(v) for v in m['b.warm_gateway']['values'])} s",
             f"部署 {num(m['b.deploy']['value'])} s；新会话 {num(m['b.cold_gateway']['value'])} s；后续 "
             f"{', '.join(num(v) for v in m['b.warm_gateway']['values'])} s")],
        ["`c-2026-09-27`", "C", c.t("Kata Sandbox, pod loss, suspend/resume", "Kata Sandbox、pod 丢失、挂起/恢复"),
         c.t(f"pod kernel `{m['c.isolation']['guest_kernel']}` vs runc pod on the Ubuntu system pool "
             f"`{m['c.isolation']['system_node_kernel']}`; "
             f"pod lost → Ready {num(m['c.pod_lost']['value'])} s; resume → answer {num(m['c.resume_answer']['value'])} s",
             f"pod 内核 `{m['c.isolation']['guest_kernel']}`，Ubuntu 系统节点池的 runc pod 内核 "
             f"`{m['c.isolation']['system_node_kernel']}`；"
             f"pod 丢失后 {num(m['c.pod_lost']['value'])} s 就绪；恢复到回答 {num(m['c.resume_answer']['value'])} s")],
        ["`d-2026-09-29-placement`", "D", c.t("where a session runs", "会话跑在哪里"),
         c.t(f"every session pod `{m['d.placement']['runtime_class']}` on your node pool after NSG rule "
             f"`{m['d.placement']['nsg_rule']}`",
             f"加 NSG 规则 `{m['d.placement']['nsg_rule']}` 后，每个会话 pod 都是 `{m['d.placement']['runtime_class']}`，"
             "跑在你的节点池上")],
        ["`d-2026-09-29-perf`", "D", c.t("cold, warm, 4 and 8 at once", "冷启动、后续、4 和 8 并发"),
         c.t(f"cold {num(m['d.cold']['median'])} s, warm {num(m['d.warm']['median'])} s (medians); "
             f"4 at once {m['d.c4']['answered']}/{m['d.c4']['n']}, 8 at once {m['d.c8']['answered']}/{m['d.c8']['n']} "
             "answered (pool full)",
             f"冷启动 {num(m['d.cold']['median'])} s、后续 {num(m['d.warm']['median'])} s（中位数）；"
             f"4 并发 {m['d.c4']['answered']}/{m['d.c4']['n']}、8 并发 {m['d.c8']['answered']}/{m['d.c8']['n']} 回答（节点池已满）")],
        ["`d-2026-09-29-recovery`", "D", c.t("delete a session pod", "删除一个会话 pod"),
         c.t(f"replacement Ready in {num(m['d.pod_lost']['value'])} s; a different live session answered in "
             f"{num(m['d.other_session_followup']['value'])} s; deleted session continuity not tested",
             f"{num(m['d.pod_lost']['value'])} s 后替代 pod 就绪；另一个存活会话在 "
             f"{num(m['d.other_session_followup']['value'])} s 内回答；未测试被删会话的连续性")],
    ]
    order = json.loads((ROOT / "evidence" / "claims.json").read_text(encoding="utf-8")).get("run_order")
    if not isinstance(order, list) or not all(isinstance(rid, str) for rid in order):
        raise RenderError("evidence/claims.json: run_order must be a list of run IDs")
    rows_by_id = {row[0].strip("`"): row for row in rows}
    if len(order) != len(rows) or set(order) != set(rows_by_id):
        raise RenderError("evidence/claims.json: run_order must list every results-table run exactly once")
    return table(head, [rows_by_id[rid] for rid in order])


# --- VALIDATION blocks ----------------------------------------------------------------------------

def v_a_agent_input(c: Ctx) -> str:
    a = c.runs["a-aca-2026-10-09/agent-runs.json"]
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


def v_a_agent(c: Ctx) -> str:
    a = c.runs["a-aca-2026-10-09/agent-runs.json"]
    head = [c.t("Run", "次"), c.t("Create → Running", "创建 → Running"), c.t("First answer", "第一个回答"),
            c.t("Cold", "冷启动"), c.t("Follow-up", "后续"), c.t("Stop", "停止"), c.t("Resume", "恢复"),
            c.t("Answer after resume", "恢复后回答"), c.t("Resume → answer", "恢复 → 回答"), c.t("Answers", "回答")]
    rows = []
    for r in a["runs"]:
        answers = " / ".join(str(r[k]["response"]) if k == "first" else str(r[k]["response"])
                             for k in ("first", "warm", "first_after_resume", "warm_after_resume"))
        rows.append([str(r["run"]), num(r["create_total_s"]), num(r["first"]["guest_s"]), num(r["cold_total_s"]),
                     num(r["warm"]["guest_s"]), num(r["stop_total_s"]), num(r["resume_total_s"]),
                     num(r["first_after_resume"]["guest_s"]), num(r["recover_total_s"]), answers])
    m = c.m
    rows.append([c.t("**median**", "**中位数**")] + [f"**{num(m[k]['median'])}**" for k in
                 ("a.create", "a.first_answer", "a.cold", "a.warm", "a.stop", "a.resume", "a.first_after_resume",
                  "a.recover")] + [""])
    return c.t("Seconds; answers are the agent's replies in order. Totals are added per run before taking the median; "
               "the independently reported column medians must not be added.\n\n",
               "单位为秒；最后一列是 agent 依次给出的回答。总耗时先在每次实测内相加，再取中位数；"
               "不能把各列独立统计的中位数相加。\n\n") + \
        table(head, rows, ["---"] + ["---:"] * 8 + ["---"])


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
    d = c.m["d.model_direct"]
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


BLOCKS = {
    "comparison": b_comparison, "runs": b_runs,
    "a-agent-input": v_a_agent_input, "a-agent": v_a_agent, "a-attr": v_a_attr, "a-late": v_a_late,
    "a-setup": v_a_setup, "b-run": v_b, "c-run": v_c, "d-input": v_d_input, "d-cold": v_d_cold,
    "d-warm": v_d_warm, "d-scale": v_d_scale, "d-recovery": v_d_recovery, "d-placement": v_d_placement,
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
