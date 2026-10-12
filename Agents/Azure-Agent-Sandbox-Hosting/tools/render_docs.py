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
    head = [c.t("Observed or documented", "实测或文档项"), "A · ACA Sandboxes",
            c.t("B · Foundry hosted agent", "B · Foundry hosted agent"),
            "C · AKS + agent-sandbox"]
    rows = [
        [c.t("**New session → first answer**", "**新会话 → 第一个回答**"),
         c.t(c.med("a.cold") + "; SDK create + answer inside sandbox",
             c.med("a.cold") + "；SDK 创建 + 沙箱内回答"),
         c.t(f"{num(m['b.cold_gateway']['value'])} s (n=1, via the Foundry endpoint; model route: gateway (APIM))",
             f"{num(m['b.cold_gateway']['value'])} s（n=1，经 Foundry 端点；模型路由：gateway（APIM））"),
         c.t(f"{num(m['c.cold']['value'])} s (n=1; apply → Ready {num(m['c.apply_ready']['value'])} s + answer "
             f"{num(m['c.first']['value'])} s)",
             f"{num(m['c.cold']['value'])} s（n=1；apply → Ready {num(m['c.apply_ready']['value'])} s + 回答 "
             f"{num(m['c.first']['value'])} s）")],
        [c.t("**Follow-up turn, same session**", "**同一会话的后续回答**"),
         c.med("a.warm"), c.t(c.vals("b.warm_gateway") + "; model route: gateway (APIM)",
                             c.vals("b.warm_gateway") + "；模型路由：gateway（APIM）"),
         c.one("c.warm")],
        [c.t("**Sandbox or pod until ready**", "**沙箱或 pod 就绪**"),
         c.t(c.med("a.create") + ", create → Running", c.med("a.create") + "，create → Running"),
         c.t("not visible (managed)", "不可见（平台托管）"),
         c.t(f"{num(m['c.apply_ready']['value'])} s (n=1), apply → Ready",
             f"{num(m['c.apply_ready']['value'])} s（n=1），apply → Ready")],
        [c.t("**Resume request → first answer (stop excluded)**", "**发起恢复 → 第一个回答（不含停止）**"),
         c.t(c.med("a.recover") + f"; separately reported resume-only median {num(m['a.resume']['median'])} s",
             c.med("a.recover") + f"；恢复本身另行统计的中位数 {num(m['a.resume']['median'])} s"),
         c.t("not exercised (the platform restores $HOME)", "未测（平台恢复 $HOME）"),
         c.t(f"{num(m['c.resume_answer']['value'])} s (n=1; Ready {num(m['c.resume_ready']['value'])} s, "
             f"LoadBalancer healthy {num(m['c.resume_lb']['value'])} s)",
             f"{num(m['c.resume_answer']['value'])} s（n=1；Ready {num(m['c.resume_ready']['value'])} s，"
             f"LoadBalancer 健康 {num(m['c.resume_lb']['value'])} s）")],
        [c.t("**Pod lost → ready again**", "**pod 丢失 → 重新就绪**"),
         c.t("not exercised (service-managed)", "未测（服务托管）"),
         c.t("not exercised (managed)", "未测（平台托管）"),
         c.t(c.one("c.pod_lost") + "; same Sandbox state restored",
             c.one("c.pod_lost") + "；同一 Sandbox 的状态已恢复")],
        [c.t("**Idle behaviour**", "**空闲时的行为**"),
         c.t("suspends after the idle time you set (900 s here) with a memory snapshot",
             "到你设置的空闲时间（本次 900 s）自动挂起，保存内存快照"),
         c.t("docs: compute released after 2–60 min idle (default 15), $HOME and /files restored",
             "文档：空闲 2–60 min（默认 15）后释放算力，恢复时还原 $HOME 和 /files"),
         c.t("none built in; you patch operatingMode", "无内置机制；需要你修改 operatingMode")],
        [c.t("**Capacity**", "**容量**"),
         c.t("serverless; quota is concurrent cores (docs)", "无服务器；配额按并发核数计（文档）"),
         c.t("managed, per session", "平台托管，按会话"),
         c.t("your Kata node pool", "你的 Kata 节点池")],
        [c.t("**Path setup in our run**", "**本次路径部署耗时**"),
         c.t(f"{num(m['a.setup']['value'])} s (sandbox group, VNet connection, disk image, volume, sandbox)",
             f"{num(m['a.setup']['value'])} s（sandbox group、VNet 连接、磁盘镜像、卷、沙箱）"),
         c.t(f"{num(m['b.deploy']['value'])} s (`azd deploy`; model route: gateway (APIM))",
             f"{num(m['b.deploy']['value'])} s（`azd deploy`；模型路由：gateway（APIM））"),
         c.t(f"{num(m['c.setup']['value'])} s (prepare script, private link, apply)",
             f"{num(m['c.setup']['value'])} s（准备脚本、私有链接、apply）")],
    ]
    return table(head, rows)


def b_runs(c: Ctx) -> str:
    m = c.m
    head = [c.t("Run", "实测"), c.t("Path", "路径"), c.t("Scenario", "场景"), c.t("Result", "结果")]
    rows = [
        ["`a-2026-10-09-agent`", "A", c.t("full agent, 3 runs", "完整 agent，跑 3 次"),
         c.t(f"cold {num(m['a.cold']['median'])} s, warm {num(m['a.warm']['median'])} s, resume → answer "
             f"{num(m['a.recover']['median'])} s (medians); vmm `{m['a.vmm']['value']}` in "
             f"{m['a.vmm']['count']}/{m['a.vmm']['n']}; state kept {m['a.state_kept']['count']}/{m['a.state_kept']['n']}",
             f"冷启动 {num(m['a.cold']['median'])} s、后续 {num(m['a.warm']['median'])} s、恢复到回答 "
             f"{num(m['a.recover']['median'])} s（中位数）；{m['a.vmm']['count']}/{m['a.vmm']['n']} 次 vmm 为 "
             f"`{m['a.vmm']['value']}`；状态保留 {m['a.state_kept']['count']}/{m['a.state_kept']['n']}")],
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














BLOCKS = {
    "comparison": b_comparison, "runs": b_runs,
    "a-agent-input": v_a_agent_input, "a-agent": v_a_agent, "a-attr": v_a_attr, "a-late": v_a_late,
    "a-setup": v_a_setup, "b-run": v_b, "c-run": v_c,
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
