"""The live tools start, explain themselves and keep their offline logic, without the Azure SDK or Azure."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import aca_sandbox
import build_diagrams
import check_isolation
import measure_aca
import measure_responses

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
TOOLS = ["aca_sandbox.py", "measure_aca.py", "measure_responses.py", "measure_e2b.py", "e2b_agent.py", "measure_sessions.py",
         "measure_code_interpreter.py", "measure_c.py", "check_isolation.py", "build_measurements.py", "render_docs.py",
         "build_diagrams.py", "export_evidence.py", "validate_docs.py"]


@pytest.mark.parametrize("tool", TOOLS)
def test_help_runs_without_azure(tool):
    out = subprocess.run([sys.executable, str(ROOT / "tools" / tool), "--help"], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


@pytest.mark.parametrize("failure", ("renderer-error", "bad-size", "timeout"))
def test_failed_diagram_render_preserves_existing_png(monkeypatch, tmp_path, failure):
    svg, png = tmp_path / "figure.svg", tmp_path / "figure.png"
    svg.write_text("<svg/>", encoding="utf-8")
    png.write_bytes(b"previous-image")

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 60)
        return SimpleNamespace(returncode=1 if failure == "renderer-error" else 0,
                               stdout="", stderr="renderer failed" if failure == "renderer-error" else "")

    monkeypatch.setattr(build_diagrams.subprocess, "run", run)
    monkeypatch.setattr(build_diagrams, "png_size", lambda _: (1, 1))
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        build_diagrams._render_png_in_process(svg, png, 2)
    assert png.read_bytes() == b"previous-image"
    assert list(tmp_path.glob("tmp*.png")) == []


def test_isolation_pass_and_fail_from_saved_json():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "check_isolation.py"), "--pods-json",
                          str(FIX / "pods.json"), "--nodes-json", str(FIX / "nodes.json"),
                          "--uname", "session-pod-1=6.6.137.mshv2-2.azl3"], capture_output=True, text=True)
    lines = out.stdout.splitlines()
    assert out.returncode == 1
    assert lines[0].startswith("CHECK isolation PASS pod=session-pod-1 runtimeClass=kata-vm-isolation")
    assert "kernel=same-version-string-as-node-(inconclusive)" in lines[0]
    assert lines[1].startswith("CHECK isolation FAIL pod=runc-pod-1 runtimeClass=runc")


def test_isolation_kernel_differs():
    import json

    pods = json.loads((FIX / "pods.json").read_text(encoding="utf-8"))
    nodes = json.loads((FIX / "nodes.json").read_text(encoding="utf-8"))
    pods["items"] = pods["items"][:1]
    r = check_isolation.evaluate(pods, nodes, {"session-pod-1": "6.6.137.mshv1-1.azl3"})
    assert r[0]["verdict"] == "PASS" and r[0]["kernel"] == "differs from node"


def test_isolation_no_pod_is_blocked(tmp_path):
    empty = tmp_path / "pods.json"
    empty.write_text('{"items": []}', encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "check_isolation.py"), "--pods-json", str(empty)],
                         capture_output=True, text=True)
    assert out.returncode == 3 and "BLOCKED" in out.stdout


@pytest.mark.parametrize("failure", ("denied", "empty", "timeout"))
def test_requested_kernel_read_failure_cannot_report_pass(monkeypatch, capsys, failure):
    pods = json.loads((FIX / "pods.json").read_text(encoding="utf-8"))
    pods["items"] = pods["items"][:1]
    nodes = json.loads((FIX / "nodes.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(check_isolation, "kubectl_json",
                        lambda *args: pods if args[1] == "pods" else nodes)

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 60)
        return SimpleNamespace(returncode=1 if failure == "denied" else 0,
                               stdout="", stderr="exec denied" if failure == "denied" else "")

    monkeypatch.setattr(check_isolation.subprocess, "run", run)
    assert check_isolation.main(["-n", "example", "--exec-uname"]) == 1
    output = capsys.readouterr()
    assert "BLOCKED kernel read failed" in output.err
    assert "CHECK isolation PASS" not in output.out


def test_missing_setting_exits_2(monkeypatch):
    for name in ("SUBSCRIPTION_ID", "RESOURCE_GROUP"):
        monkeypatch.delenv(name, raising=False)
    assert aca_sandbox.main(["show"]) == 2
    monkeypatch.setenv("SUBSCRIPTION_ID", "YOUR-SUBSCRIPTION-ID")
    assert aca_sandbox.main(["show"]) == 2


def test_reflect_code_quotes_the_question_and_times_inside():
    code = aca_sandbox.reflect_code('Where is "Bangkok"?')
    compile(code, "<reflect>", "exec")
    assert "127.0.0.1:8080/reflect" in code and "time.time()" in code


@pytest.mark.parametrize("body", ({"error": "model unavailable"}, {"response": ""}, {"response": None}, []))
def test_aca_request_without_an_answer_cannot_report_success(body):
    sandbox = SimpleNamespace(exec=lambda _: SimpleNamespace(
        exit_code=0, stdout=json.dumps({"seconds": 0.1, "body": body})))
    with pytest.raises(RuntimeError, match="non-empty answer"):
        aca_sandbox.ask(sandbox, "Hi")


def test_aca_request_returns_the_real_answer():
    sandbox = SimpleNamespace(exec=lambda _: SimpleNamespace(
        exit_code=0, stdout=json.dumps({"seconds": 0.1, "body": {"response": "Hello", "history_length": 1}})))
    answer = aca_sandbox.ask(sandbox, "Hi")
    assert answer["response"] == "Hello" and answer["history_length"] == 1


def test_aca_request_waits_for_the_agent_port_and_reports_the_wait():
    code = aca_sandbox.reflect_code("Hi")
    assert "connect_ex(('127.0.0.1',8080))" in code and code.index("connect_ex") < code.index("urlopen")
    sandbox = SimpleNamespace(exec=lambda _: SimpleNamespace(exit_code=0, stdout=json.dumps(
        {"seconds": 1.0, "wait": 2.5, "body": {"response": "Hello", "history_length": 1}})))
    assert aca_sandbox.ask(sandbox, "Hi")["ready_wait_s"] == 2.5


def test_builtin_config_needs_no_sdk_and_full_needs_an_image():
    cfg = {"vnet_connection": "aca-vnet", "volume": "v", "allow_cidr": ""}
    kw = aca_sandbox.config_kwargs("builtin", cfg, None, None)
    assert kw["disk"] == "ubuntu" and kw["cpu"] == "1000m" and kw["auto_suspend_mode"] == "Memory"
    with pytest.raises(aca_sandbox.SettingMissing):
        aca_sandbox.config_kwargs("full", cfg, None, {})
    with pytest.raises(ValueError):
        aca_sandbox.config_kwargs("everything", cfg, "img", {})


def test_novolume_config_is_full_without_the_blob_volume(monkeypatch):
    monkeypatch.setattr(aca_sandbox, "port_request", lambda cfg: "port-8080")
    cfg = {"vnet_connection": "aca-vnet", "volume": "v", "allow_cidr": ""}
    kw = aca_sandbox.config_kwargs("novolume", cfg, "img", {"AGENT_ID": "a"})
    assert kw["volumes"] is None and kw["ports"] == ["port-8080"]
    assert kw["customer_vnet_connection_name"] == "aca-vnet" and kw["disk_id"] == "img"
    assert kw["auto_suspend_mode"] == "Memory" and kw["environment"] == {"AGENT_ID": "a"}


@pytest.mark.parametrize("config", ("image", "full,novolume"))
def test_aca_agent_mode_takes_one_agent_config(config):
    with pytest.raises(SystemExit) as exc:
        measure_aca.main(["agent", "--config", config])
    assert exc.value.code == 2


def test_measure_responses_inputs_match_the_measured_run():
    import json

    src = (ROOT / "tools" / "measure_responses.py").read_text(encoding="utf-8")
    bench = json.loads((ROOT / "evidence/runs/d-byoaks-2026-09-29/bench.json").read_text(encoding="utf-8"))
    for text in bench["inputs"].values():
        for part in text.split("{i}"):
            assert part in src, part


def response(answer="OK"):
    return {"http": 200, "status": "completed", "response_id": "response-1", "answer": answer,
            "wall_s": 0.1, "error": None}


@pytest.fixture
def responses_main(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_RESPONSES_URL", "https://example.com/responses")
    monkeypatch.setenv("SESSION_NAMESPACE", "example-sessions")
    monkeypatch.setattr(measure_responses, "token", lambda: "test-token")
    monkeypatch.setattr(measure_responses.time, "sleep", lambda _: None)
    monkeypatch.setattr(measure_responses, "run_cold", lambda *args: [response("COLD-1")])
    monkeypatch.setattr(measure_responses, "run_warm",
                        lambda *args: {"turns": [response(), response("LANTERN-7")], "remembered": True})

    def run(*args):
        return measure_responses.main(["--cold", "1", "--warm-turns", "1", "--pause", "0",
                                       "--out-dir", str(tmp_path), *args])

    return run


@pytest.mark.parametrize("phase", ("scale-first-turn", "scale-followup", "scale-memory", "recovery-skipped",
                                   "recovery-memory"))
def test_responses_fail_if_any_requested_phase_fails(monkeypatch, responses_main, tmp_path, phase):
    if phase.startswith("scale"):
        first, followup = response(), response("SCALE-0")
        if phase == "scale-first-turn":
            first = {**first, "http": 429, "error": "quota exceeded"}
        elif phase == "scale-followup":
            followup = {**followup, "http": 500, "error": "follow-up failed"}
        result = {"concurrency": 1, "answered": 1,
                  "sessions": [{"t1": first, "t2": followup, "remembered": phase != "scale-memory"}]}
        monkeypatch.setattr(measure_responses, "run_scale", lambda *args: result)
        code = responses_main("--scale", "1")
    else:
        result = {"skipped": "no unique session pod"} if phase == "recovery-skipped" else {
            "turn_after_delete": response("RECOVER-1"), "state_preserved": False}
        monkeypatch.setattr(measure_responses, "run_recovery", lambda *args: result)
        code = responses_main("--recovery")
    assert code == 1
    assert len(list(tmp_path.glob("responses-*.json"))) == 1


def test_responses_refresh_token_before_scale_and_recovery(monkeypatch, responses_main):
    calls = []

    def token():
        calls.append("token")
        return f"test-token-{len(calls)}"

    seen = []
    monkeypatch.setattr(measure_responses, "token", token)
    monkeypatch.setattr(measure_responses, "run_scale", lambda endpoint, tok, pods, n: (
        seen.append(tok) or {"concurrency": n, "answered": n,
                             "sessions": [{"t1": response(), "t2": response("SCALE-0"), "remembered": True}]}))
    monkeypatch.setattr(measure_responses, "run_recovery", lambda endpoint, tok, pods: (
        seen.append(tok) or {"turn_after_delete": response("RECOVER-1"), "state_preserved": True}))
    assert responses_main("--scale", "1", "--recovery") == 0
    assert seen == ["test-token-2", "test-token-3"]


def test_responses_missing_setting_exits_2(monkeypatch):
    monkeypatch.delenv("AGENT_RESPONSES_URL", raising=False)
    assert measure_responses.main(["--pause", "0"]) == 2


def test_failed_pod_observation_is_not_an_empty_success(monkeypatch):
    monkeypatch.setattr(measure_responses.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="Forbidden"))
    with pytest.raises(RuntimeError, match="Forbidden"):
        measure_responses.Pods("example-sessions").list()


def test_failed_observation_is_saved_and_exits_1(monkeypatch, responses_main, tmp_path):
    def fail(*args):
        raise RuntimeError("pod observation failed")

    monkeypatch.setattr(measure_responses, "run_cold", fail)
    assert responses_main("--observe") == 1
    report = json.loads(next(tmp_path.glob("responses-*.json")).read_text(encoding="utf-8"))
    assert "pod observation failed" in report["error"]


def test_failed_pod_deletion_does_not_claim_recovery(monkeypatch):
    lists = iter([[], [{"name": "session-pod-1", "runtime": "kata-vm-isolation"}]])
    pods = measure_responses.Pods("example-sessions")
    monkeypatch.setattr(pods, "list", lambda: next(lists))
    invoked = []
    monkeypatch.setattr(measure_responses, "invoke", lambda *args: invoked.append(args) or response())
    monkeypatch.setattr(measure_responses.time, "sleep", lambda _: None)
    monkeypatch.setattr(measure_responses.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="delete denied"))
    with pytest.raises(RuntimeError, match="delete denied"):
        measure_responses.run_recovery("https://example.com/responses", "test-token", pods)
    assert len(invoked) == 1


def test_failed_warm_start_does_not_open_unrelated_followup_sessions(monkeypatch):
    calls = []
    monkeypatch.setattr(measure_responses, "invoke", lambda *args: calls.append(args) or {
        "http": 500, "status": None, "response_id": None, "answer": None, "wall_s": 0.1, "error": "creation failed"})
    result = measure_responses.run_warm("https://example.com/responses", "test-token", 3)
    assert len(calls) == 1 and not result["remembered"]


@pytest.mark.parametrize("lengths", ([1, 2, 3, 4], [1, 2, 1, 2]))
def test_aca_agent_measurement_checks_state_across_resume(monkeypatch, lengths):
    sandbox = SimpleNamespace(stop=lambda: None, resume=lambda: None, delete=lambda: None, get=lambda: None)
    group = SimpleNamespace(_do_create_sandbox=lambda **kwargs: (sandbox, SimpleNamespace(id="sandbox-1",
                                                                                       state="Running")),
                            get_sandbox=lambda _: None)
    turns = iter({"guest_s": 1.0, "ready_wait_s": 0.5, "history_length": length} for length in lengths)
    monkeypatch.setattr(aca_sandbox, "agent_environment", lambda *args, **kwargs: {})
    used = []
    monkeypatch.setattr(aca_sandbox, "config_kwargs", lambda name, *args: used.append(name) or {})
    monkeypatch.setattr(aca_sandbox, "poll_state", lambda getter, targets: (
        0, SimpleNamespace(state=targets[0], vmm_type="cloudhypervisor"), 1))
    monkeypatch.setattr(aca_sandbox, "ask", lambda *args: next(turns))
    monkeypatch.setattr(measure_aca.time, "sleep", lambda _: None)
    result = measure_aca.agent_run(group, {}, "image-1", 1, "novolume")
    assert used == ["novolume"] and result["config"] == "novolume"
    if lengths == [1, 2, 3, 4]:
        assert result["state_kept"] is True and "error" not in result
        assert result["cold_total_s"] == pytest.approx(result["create_total_s"] + 1.5, abs=0.011)
        assert result["recover_total_s"] == pytest.approx(result["resume_total_s"] + 1.5, abs=0.011)
    else:
        assert result["state_kept"] is False and "error" in result


def test_aca_empty_measurement_is_rejected():
    with pytest.raises(SystemExit) as exc:
        measure_aca.main(["agent", "--runs", "0"])
    assert exc.value.code == 2



# --- measure_e2b: offline fakes for the E2B SDK and the Foundry Responses API -------------------------------

import io  # noqa: E402
import itertools  # noqa: E402
import re  # noqa: E402

import measure_e2b  # noqa: E402


class FakeHost:
    def __init__(self, name, root):
        self.name, self.root, self.sandboxes, self.down = name, Path(root), {}, False
        self.lossy_resume, self.ids, self.capacity = False, itertools.count(1), None


class FakeSandbox:
    """One code-interpreter-v1 sandbox: its /home/user is a temporary folder."""

    def __init__(self, host, sandbox_id):
        self.host, self.sandbox_id = host, sandbox_id
        self.home = host.root / f"{sandbox_id}-{next(host.ids)}"
        self.home.mkdir(parents=True)
        self.kernel = {}

    def get_info(self):
        return SimpleNamespace(name="code-interpreter-v1", cpu_count=2, memory_mb=2048)

    def run_code(self, code, timeout=None):
        """Code Interpreter: one namespace per sandbox, kept between cells like the Jupyter kernel."""
        if self.host.down:
            raise RuntimeError(f"host {self.host.name} unreachable")
        out, error = io.StringIO(), None
        # A per-call print: redirect_stdout is process-wide, and the scale test runs sessions in threads.
        self.kernel["print"] = lambda *args, **kw: print(*args, **{**kw, "file": out})
        try:
            exec(code.replace("/home/user", self.home.as_posix()), self.kernel)
        except Exception as exc:  # noqa: BLE001 - returned like the SDK's ExecutionError
            error = SimpleNamespace(name=type(exc).__name__, value=str(exc))
        return SimpleNamespace(logs=SimpleNamespace(stdout=[out.getvalue()]), text=None, error=error)

    def beta_pause(self):
        if self.host.down:
            raise RuntimeError("connection refused")

    def kill(self):
        self.host.sandboxes.pop(self.sandbox_id, None)


class FakeSDK:
    """The (Sandbox, SandboxQuery, SandboxState) triple; every call is routed by the api_url it names."""

    def __init__(self, hosts):
        sdk = self
        self.hosts = hosts

        def host(api_url):
            h = sdk.hosts[api_url]
            if h.down:
                raise RuntimeError(f"connection refused: {api_url}")
            return h

        class Sandbox:
            @staticmethod
            def create(template, timeout=300, api_url=None, sandbox_url=None, api_key=None):
                h = host(api_url)
                if h.capacity is not None and len(h.sandboxes) >= h.capacity:
                    raise RuntimeError("500: Failed to place sandbox")
                s = FakeSandbox(h, f"{h.name}-sbx-{next(h.ids)}")
                h.sandboxes[s.sandbox_id] = s
                return s

            @staticmethod
            def connect(sandbox_id, api_url=None, sandbox_url=None, api_key=None):
                h = host(api_url)
                if sandbox_id not in h.sandboxes:
                    raise RuntimeError(f"404: sandbox {sandbox_id} not found on {h.name}")
                if h.lossy_resume:
                    fresh = FakeSandbox(h, sandbox_id)
                    h.sandboxes[sandbox_id] = fresh
                return h.sandboxes[sandbox_id]

            @staticmethod
            def kill(sandbox_id, api_url=None, sandbox_url=None, api_key=None):
                host(api_url).sandboxes.pop(sandbox_id, None)

            @staticmethod
            def list(query=None, api_url=None, sandbox_url=None, api_key=None):
                items = [SimpleNamespace(sandbox_id=i) for i in host(api_url).sandboxes]
                pages = [items]

                class Pager:
                    @property
                    def has_next(self):
                        return bool(pages)

                    def next_items(self, **opts):
                        return pages.pop()
                return Pager()

        self.triple = (Sandbox, lambda state=None: SimpleNamespace(state=state),
                       SimpleNamespace(RUNNING="running", PAUSED="paused"))


class FakeFoundry:
    """Plays the Prompt Agent: one tool call with the code a model would write, then a reply from its output."""

    def __init__(self):
        self.convs, self.versions, self.pending, self.ids = set(), [], {}, itertools.count(1)

    def begin(self):
        pass

    def statuses(self):
        return [200, 200]

    def create_agent(self, model):
        self.versions.append("1")
        return "1"

    def delete_agent(self, version):
        self.versions.remove(version)

    def conversation(self):
        c = f"conv-{next(self.ids)}"
        self.convs.add(c)
        return c

    def drop(self, conv_id):
        self.convs.discard(conv_id)
        return True

    @staticmethod
    def code_for(q):
        m = re.search(r"(?:save the code word|write) (\S+) to (/home/user/\S+?)\.? ", q + " ")
        if m:
            return f"open({m.group(2)!r}, 'w').write({m.group(1)!r}); print('saved')"
        m = re.search(r"read (/home/user/\S+?)\.? ", q + " ")
        if m:
            return f"print(open({m.group(1)!r}).read())"
        m = re.search(r"print (COLD-\d+)", q)
        return f"print({m.group(1)!r})" if m else "print('alive')"

    @staticmethod
    def reply(q, out):
        if "Number only" in q:
            return str(out["prior_questions_in_sandbox"])
        if "Reply only OK" in q:
            return "OK"
        m = re.search(r"reply with exactly (\S+)", q)
        if m:
            return m.group(1)
        return out["stdout"].strip() if "read /home/user" in q else "Bangkok is the capital of Thailand."

    def req(self, method, path, body=None, timeout=180):
        user = [i for i in body["input"] if i.get("type") == "message"]
        if user:
            self.pending[body["conversation"]] = user[0]["content"]
            return {"output": [{"type": "function_call", "call_id": "call-1",
                                "arguments": json.dumps({"python_code": self.code_for(user[0]["content"])})}]}
        text = self.reply(self.pending[body["conversation"]], json.loads(body["input"][0]["output"]))
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def e2b_hosts(names, monkeypatch, root):
    fakes = {f"http://{n}:3000": FakeHost(n, root / n) for n in names}
    sdk = FakeSDK(fakes)
    hosts = {n: measure_e2b.Host(n, f"http://{n}:3000", f"http://{n}:3002", "test-key", sdk.triple) for n in names}
    for h in hosts.values():
        monkeypatch.setattr(h, "health", lambda fake=fakes[h.api_url]: 0 if fake.down else 200)
    return hosts, fakes


def test_e2b_prompts_match_the_measured_run():
    perf = json.loads((ROOT / "evidence/runs/e-e2b-2026-10-14/perf.json").read_text(encoding="utf-8"))
    agent = json.loads((ROOT / "evidence/runs/e-e2b-2026-10-14/agent-runs.json").read_text(encoding="utf-8"))
    assert agent["inputs"]["first"] == measure_e2b.FIRST_Q and agent["inputs"]["warm"] == measure_e2b.COUNT_Q
    assert perf["inputs"] == {"cold": measure_e2b.COLD_Q, "warm": measure_e2b.WARM_QS,
                              "scale": [measure_e2b.SCALE_WRITE, measure_e2b.SCALE_READ],
                              "recovery": [measure_e2b.LIVE_WRITE, measure_e2b.LIVE_READ]}
    for doc in (agent, perf):
        assert doc["agent"] == {"instructions": measure_e2b.INSTRUCTIONS, "tools": measure_e2b.TOOLS_SPEC}


@pytest.mark.parametrize("lossy", (False, True))
def test_e2b_agent_run_checks_history_across_pause(monkeypatch, tmp_path, lossy):
    monkeypatch.setattr(measure_e2b.time, "sleep", lambda _: None)
    hosts, fakes = e2b_hosts(["a"], monkeypatch, tmp_path)
    fakes["http://a:3000"].lossy_resume = lossy
    rec = measure_e2b.agent_run(FakeFoundry(), "1", hosts["a"], 1)
    if lossy:
        assert rec["history_lengths"] == [1, 2, 1, 2] and rec["state_kept"] is False and "error" in rec
    else:
        assert rec["history_lengths"] == [1, 2, 3, 4] and rec["state_kept"] is True and "error" not in rec
        assert [rec[k]["response"] for k in ("warm", "first_after_resume", "warm_after_resume")] == ["1", "2", "3"]
    assert rec["sandbox_deleted"] and rec["conversation_deleted"] and not fakes["http://a:3000"].sandboxes


def test_e2b_code_interpreter_cells_share_one_kernel(monkeypatch, tmp_path):
    hosts, fakes = e2b_hosts(["a"], monkeypatch, tmp_path)
    sbx = hosts["a"].create()
    first = measure_e2b.sandbox_run_code(sbx, "x = 41", "q1")
    second = measure_e2b.sandbox_run_code(sbx, "print(x + 1)", "q2")
    assert second["stdout"] == "42" and second["state"]["prior_questions"] == 1
    assert [first["state"]["history_length"], second["state"]["history_length"]] == [1, 2]
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        measure_e2b.sandbox_run_code(sbx, "1/0", "q3")


def test_e2b_scale_spreads_sessions_and_counts_per_host(monkeypatch, tmp_path):
    hosts, fakes = e2b_hosts(["a", "b"], monkeypatch, tmp_path)
    res = measure_e2b.at_once(FakeFoundry(), "1", [(hosts["ab"[k % 2]], k + 1) for k in range(6)])
    assert res["sessions_at_once"] == 6 and res["answered"] == 6
    assert res["per_host"] == {"a": {"sessions": 3, "answered": 3}, "b": {"sessions": 3, "answered": 3}}
    assert all(s["turn2"]["answer"] == f"SCALE-{s['slot']}" for s in res["sessions"])
    assert all(s["turn1"]["wall_s"] >= s["turn1"]["agent_turn_s"] for s in res["sessions"])
    assert res["model_http_statuses"] == {"200": 24}
    assert all(not f.sandboxes for f in fakes.values())


def test_e2b_hostloss_records_what_survives(monkeypatch, tmp_path):
    monkeypatch.setattr(measure_e2b.time, "sleep", lambda _: None)
    hosts, fakes = e2b_hosts(["a", "b"], monkeypatch, tmp_path)

    def shell(cmd):
        fakes["http://a:3000"].down = cmd == "stop"
        return {"exit_code": 0, "seconds": 0.0, "tail": ""}

    monkeypatch.setattr(measure_e2b, "run_shell", shell)
    out = measure_e2b.hostloss(FakeFoundry(), "1", hosts, "a", "stop", "start")
    assert "error" not in out
    b = out["after_loss"]["b"]
    assert b["live_session_read"]["ok"] and b["live_session_read"]["value"] == "LANTERN-7"
    assert b["new_session"]["ok"] and b["new_session"]["value"]["correct"]
    assert b["connect_lost_hosts_paused_sandbox"]["ok"] is False
    assert out["after_loss"]["a"]["create"]["ok"] is False
    assert out["after_loss"]["a"]["connect_own_paused_sandbox"]["ok"] is False
    resumed = out["after_restart"]["a"]["resume_own_paused_sandbox"]
    assert resumed["ok"] and resumed["value"] == "PAUSED-a"
    assert out["left_after_cleanup"] == {"a": 0, "b": 0} and out["conversations_deleted"]
    assert not measure_e2b.failed({"mode": "hostloss", "hostloss": out})


def test_e2b_main_fails_on_lost_state_or_a_leftover(monkeypatch, tmp_path):
    for k, v in {"PROJECT_ENDPOINT": "https://example.com/api/projects/p", "ACA_MODEL": "gpt-5.4-mini",
                 "E2B_API_URL": "http://a:3000", "E2B_SANDBOX_URL": "http://a:3002", "E2B_API_KEY": "test-key"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FOUNDRY_TOKEN_SOURCE", raising=False)
    monkeypatch.setattr(measure_e2b.time, "sleep", lambda _: None)
    fakes = {"http://a:3000": FakeHost("a", tmp_path / "a")}
    monkeypatch.setattr(measure_e2b, "load_sdk", lambda: FakeSDK(fakes).triple)
    monkeypatch.setattr(measure_e2b.Host, "health", lambda self: 200)
    monkeypatch.setattr(measure_e2b, "token", lambda: "test-token")
    fd = FakeFoundry()
    monkeypatch.setattr(measure_e2b, "Foundry", lambda endpoint, tok: fd)
    assert measure_e2b.main(["agent", "--runs", "1", "--out-dir", str(tmp_path / "ok")]) == 0
    assert fd.versions == [] and fd.convs == set()
    doc = json.loads(next((tmp_path / "ok").glob("e2b-agent-*.json")).read_text(encoding="utf-8"))
    assert doc["agent"]["instructions"] == measure_e2b.INSTRUCTIONS and "test-key" not in json.dumps(doc)
    assert (doc["executor"], doc["template"]) == ("code-interpreter", "code-interpreter-v1")
    fakes["http://a:3000"].lossy_resume = True
    assert measure_e2b.main(["agent", "--runs", "1", "--out-dir", str(tmp_path / "lossy")]) == 1
    fakes["http://a:3000"].lossy_resume = False
    stray = FakeSandbox(fakes["http://a:3000"], "stray")
    fakes["http://a:3000"].sandboxes["stray"] = stray
    assert measure_e2b.main(["agent", "--runs", "1", "--out-dir", str(tmp_path / "left")]) == 1


def test_e2b_host_mode_reads_the_vmm_benchmarks_and_reserves(monkeypatch, tmp_path):
    for k, v in {"E2B_API_URL": "http://a:3000", "E2B_SANDBOX_URL": "http://a:3002", "E2B_API_KEY": "test-key"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("PROJECT_ENDPOINT", raising=False)
    monkeypatch.setattr(measure_e2b.time, "sleep", lambda _: None)
    monkeypatch.setattr(measure_e2b.os, "uname", lambda: SimpleNamespace(release="test-host-kernel"), raising=False)
    fake = FakeHost("a", tmp_path / "a")
    fake.capacity = 2
    monkeypatch.setattr(measure_e2b, "load_sdk", lambda: FakeSDK({"http://a:3000": fake}).triple)
    monkeypatch.setattr(measure_e2b.Host, "health", lambda self: 200)
    monkeypatch.setattr(measure_e2b, "GUEST_PROBE", "print('6.1.177+', 2, 1985)")
    monkeypatch.setattr(measure_e2b, "vmm_processes", lambda: ["/fc-versions/v1.14-0.2.0/amd64/firecracker"])
    monkeypatch.setattr(measure_e2b, "meminfo", lambda: {"hugepages_total": 4096, "hugepages_rsvd": 62,
                                                         "hugepages_free": 4083 - 1024 * len(fake.sandboxes),
                                                         "mem_available_mb": 5700})
    assert measure_e2b.main(["host", "--bench", "2", "--reserve", "3", "--out-dir", str(tmp_path / "out")]) == 0
    doc = json.loads(next((tmp_path / "out").glob("e2b-host-*.json")).read_text(encoding="utf-8"))
    iso = doc["isolation"]
    assert iso["host_kernel"] == "test-host-kernel"
    assert (iso["guest_kernel"], iso["guest_cpus"], iso["sandbox_info"]["memory_mb"]) == ("6.1.177+", 2, 2048)
    assert iso["vmm_processes"] and [r["state_kept"] for r in doc["bench"]["runs"]] == [True, True]
    res = doc["reservation"]
    assert [s["hugepages_free"] for s in res["steps"]] == [3059, 2035] and res["refused"]["at"] == 3
    assert res["sandboxes_deleted"] and not fake.sandboxes and doc["leftover_sandboxes"] == {"a": 0}
    assert "agent" not in doc and "test-key" not in json.dumps(doc)
    fake.lossy_resume = True
    assert measure_e2b.main(["host", "--bench", "1", "--out-dir", str(tmp_path / "lossy")]) == 1


def test_e2b_agent_answers_and_cleans_up(monkeypatch, tmp_path, capsys):
    import e2b_agent
    for k, v in {"PROJECT_ENDPOINT": "https://example.com/api/projects/p", "ACA_MODEL": "gpt-5.4-mini",
                 "E2B_API_URL": "http://a:3000", "E2B_SANDBOX_URL": "http://a:3002", "E2B_API_KEY": "test-key"}.items():
        monkeypatch.setenv(k, v)
    fakes = {"http://a:3000": FakeHost("a", tmp_path / "a")}
    monkeypatch.setattr(measure_e2b, "load_sdk", lambda: FakeSDK(fakes).triple)
    monkeypatch.setattr(measure_e2b, "token", lambda: "test-token")
    fd = FakeFoundry()
    monkeypatch.setattr(measure_e2b, "Foundry", lambda endpoint, tok: fd)
    assert e2b_agent.main(["Where is Bangkok? Answer in one short sentence."]) == 0
    assert "Bangkok" in capsys.readouterr().out
    assert fd.versions == [] and fd.convs == set() and not fakes["http://a:3000"].sandboxes


def test_e2b_lost_state_or_a_leak_is_not_exit_0():
    perf = {"mode": "perf", "warm": {"remembered": True}, "at_once": [{"answered": 1, "sessions_at_once": 1}],
            "recovery": {"followup_on_live_session": {"state_preserved": False}, "sandboxes_deleted": True,
                         "conversation_deleted": True}}
    assert measure_e2b.failed(perf)
    perf["recovery"]["followup_on_live_session"]["state_preserved"] = True
    assert not measure_e2b.failed(perf)
    session = {"ok": True, "value": {"sandbox_deleted": True, "conversation_deleted": True}}
    hl = {"lost_host": "a", "stop": {"exit_code": 0}, "start": {"exit_code": 0, "health_up": {"status": 200}},
          "after_loss": {"b": {"live_session_read": {"ok": True, "value": "LANTERN-7"}, "new_session": session,
                               "connect_lost_hosts_paused_sandbox": {"ok": False, "error": "not found"}},
                         "a": {"create": {"ok": False, "error": "refused"},
                               "connect_own_paused_sandbox": {"ok": False, "error": "refused"}}},
          "after_restart": {"a": {"resume_own_paused_sandbox": {"ok": True, "value": "PAUSED-a"},
                                  "live_sandbox_before_loss": {"ok": False, "error": "not found"},
                                  "new_session": session}},
          "left_after_cleanup": {"a": 0, "b": 0}, "conversations_deleted": True}
    assert not measure_e2b.failed({"mode": "hostloss", "hostloss": hl})
    breaks = [lambda h: h.update(conversations_deleted=False),
              lambda h: h.update(left_after_cleanup={"a": 1, "b": 0}),
              lambda h: h.update(left_after_cleanup={"a": "unreachable", "b": 0}),
              lambda h: h["after_loss"]["b"]["new_session"]["value"].update(sandbox_deleted=False),
              lambda h: h["after_loss"]["b"]["live_session_read"].update(ok=False),
              lambda h: h["after_loss"]["b"]["live_session_read"].update(value="LANTERN-8"),
              lambda h: h["after_loss"]["b"]["new_session"].update(ok=False),
              lambda h: h["after_restart"]["a"]["new_session"].update(ok=False),
              lambda h: h["after_restart"]["a"]["resume_own_paused_sandbox"].update(ok=False)]
    for i, brk in enumerate(breaks):
        h = json.loads(json.dumps(hl))
        brk(h)
        assert measure_e2b.failed({"mode": "hostloss", "hostloss": h}), i
    mu = json.loads((ROOT / "evidence/runs/e-e2b-2026-10-15/multi.json").read_text(encoding="utf-8"))
    assert not measure_e2b.failed({"mode": "hostloss", "hostloss": mu["host_loss"]})
    assert measure_e2b.session_attempt(lambda: {"http": 0, "error": "SandboxException: 500"})["ok"] is False
    assert measure_e2b.session_attempt(lambda: {"http": 200, "correct": True})["ok"] is True


def test_e2b_recovery_and_hostloss_clean_up_when_a_call_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(measure_e2b.time, "sleep", lambda _: None)

    def second_call_fails(fn):
        n = itertools.count(1)
        return lambda: (_ for _ in ()).throw(RuntimeError("HTTP 500")) if next(n) == 2 else fn()

    hosts, fakes = e2b_hosts(["a"], monkeypatch, tmp_path / "r")
    monkeypatch.setattr(hosts["a"], "create", second_call_fails(hosts["a"].create))
    fd = FakeFoundry()
    out = measure_e2b.recovery(fd, "1", hosts["a"])
    assert "error" in out and out["sandboxes_deleted"] and out["conversation_deleted"]
    assert not fakes["http://a:3000"].sandboxes and fd.convs == set()

    monkeypatch.setattr(measure_e2b, "run_shell", lambda cmd: {"exit_code": 0, "seconds": 0.0, "tail": ""})
    hosts, fakes = e2b_hosts(["a", "b"], monkeypatch, tmp_path / "h")
    fd = FakeFoundry()
    fd.conversation = second_call_fails(fd.conversation)
    out = measure_e2b.hostloss(fd, "1", hosts, "a", "stop", "start")
    assert "error" in out and out["conversations_deleted"] and fd.convs == set()
    assert measure_e2b.failed({"mode": "hostloss", "hostloss": out})


def test_e2b_bad_hosts_file_is_a_setting_error(monkeypatch, tmp_path):
    for k, v in {"PROJECT_ENDPOINT": "https://example.com/api/projects/p", "ACA_MODEL": "gpt-5.4-mini"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(measure_e2b, "load_sdk", lambda: FakeSDK({}).triple)
    for name, content in (("missing.json", None), ("broken.json", "{not json"), ("flat.json", '{"a": "http://a:3000"}')):
        path = tmp_path / name
        if content is not None:
            path.write_text(content, encoding="utf-8")
        assert measure_e2b.main(["agent", "--hosts", str(path), "--out-dir", str(tmp_path / "out")]) == 2, name


import measure_sessions  # noqa: E402


class FakeSessionsAPI:
    """The hosted-agent session endpoints of one agent version, for measure_sessions.Client."""

    def __init__(self, lose_marker=False, flat_count=False, stray=False):
        self.sessions, self.files, self.turns, self.ids = {}, {}, {}, itertools.count(1)
        self.lose_marker, self.flat_count, self.stray = lose_marker, flat_count, stray

    def call(self, op, method, path, *, body=None, raw=None):
        rec = {"status": 200, "attempts": [{"status": 200, "seconds": 0.01}], "total_s": 0.01, "success_s": 0.01}
        sid = next((s for s in self.sessions if f"/sessions/{s}" in path), None)
        out: dict = {}
        if op == "read_version":
            out = {"definition": {"kind": "hosted"}}
        elif op == "create_session":
            sid = f"s{next(self.ids)}"
            self.sessions[sid], rec["status"] = "active", 201
            out = {"agent_session_id": sid, "status": "active"}
            if self.stray:
                self.sessions["orphan"] = "active"
        elif op == "upload_file":
            self.files[sid] = raw.decode("utf-8")
        elif op == "download_file":
            return rec, (b"" if self.lose_marker else self.files[sid].encode("utf-8"))
        elif op == "stop_session":
            self.sessions[sid] = "idle"
        elif op == "get_session":
            out = {"status": self.sessions[sid]}
        elif op == "responses":
            sid = body["agent_session_id"]
            self.sessions[sid] = "active"
            n = self.turns.get(sid, 0)
            self.turns[sid] = n + 1
            text = "Bangkok is in Thailand." if n == 0 else ("1" if self.flat_count else str(n))
            out = {"id": f"r{n}", "agent_session_id": sid, "created_at": 1, "completed_at": 2,
                   "output": [{"content": [{"text": text}]}]}
        elif op == "delete_session":
            self.sessions.pop(sid or path.rsplit("/", 1)[-1], None)
            rec["status"] = 204
        elif op == "list_sessions":
            out = {"data": [{"agent_session_id": s, "created_at": 4e9, "version_indicator": {"agent_version": "2"}}
                            for s in self.sessions]}
        return rec, json.dumps(out).encode("utf-8")


@pytest.mark.parametrize("fault,code", [(None, 0), ("lose_marker", 1), ("flat_count", 1), ("stray", 0)])
def test_sessions_exit_code_needs_state_and_cleanup(monkeypatch, tmp_path, fault, code):
    api = FakeSessionsAPI(**({fault: True} if fault else {}))
    monkeypatch.setattr(measure_sessions, "auth_header", lambda: ("Authorization", "Bearer test-token"))
    monkeypatch.setattr(measure_sessions.Client, "call", lambda self, *a, **k: api.call(*a, **k))
    monkeypatch.setattr(measure_sessions.time, "sleep", lambda _: None)
    args = ["--project-endpoint", "https://example.com/api/projects/p", "--agent", "maf-agent", "--version", "2",
            "--runs", "1", "--out", str(tmp_path / "raw.json")]
    assert measure_sessions.main(args) == code
    raw = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    assert raw["cleanup"]["extra_deleted"] == raw["cleanup"]["extra_sessions"] == (1 if fault == "stray" else 0)
    assert not api.sessions and "test-token" not in json.dumps(raw)
    with pytest.raises(SystemExit):
        measure_sessions.main(args[:-4] + ["--runs", "0", "--out", str(tmp_path / "zero.json")])


def test_e2b_supplementary_record_matches_the_tools():
    latest = json.loads((ROOT / "evidence/runs/e-e2b-2026-10-15/latest.json").read_text(encoding="utf-8"))
    sdk = latest["sdk_check"]
    assert sdk["agent"] == {"instructions": measure_e2b.INSTRUCTIONS, "tools": measure_e2b.TOOLS_SPEC}
    assert sdk["inputs"] == {"first": measure_e2b.FIRST_Q, "count": measure_e2b.COUNT_Q,
                             "resume_wait_s": measure_e2b.RESUME_WAIT_S}
    runs = latest["code_interpreter"]["check_runs"]
    want = ["create (default template)", "README: x = 1", "README: x+=1; x", "pandas sum", "matplotlib png",
            "javascript kernel", "exception", "pause", "connect (resume)", "x after resume", "kill"]
    assert all([s["step"] for s in r["steps"]] == want for r in runs)
    assert mci.RESUME_WAIT_S == 5 and mci.JS_CODE == "[1, 2, 3].reduce((a, b) => a + b)"
    for r in runs:
        observed = {s["step"]: s["observed"] for s in r["steps"]}
        assert set(observed["create (default template)"]) == {"sdk_default_template", "template_name", "cpu_count",
                                                              "memory_mb"}
        assert set(observed["matplotlib png"]) == {"results", "png_bytes_base64"}


def test_e2b_code_interpreter_records_match_the_tools():
    for name in ("ci-agent", "ci-perf"):
        doc = json.loads((ROOT / f"evidence/runs/e-e2b-2026-10-17/{name}.json").read_text(encoding="utf-8"))
        assert doc["executor"] == "code-interpreter"
        assert doc["template"]["name"] == measure_e2b.TEMPLATE
        assert doc["agent"] == {"instructions": measure_e2b.INSTRUCTIONS, "tools": measure_e2b.TOOLS_SPEC}
        assert doc["tool"]["file"] == "tools/measure_e2b.py"
    agent = json.loads((ROOT / "evidence/runs/e-e2b-2026-10-17/ci-agent.json").read_text(encoding="utf-8"))
    assert agent["inputs"] == {"first": measure_e2b.FIRST_Q, "count": measure_e2b.COUNT_Q,
                               "resume_wait_s": measure_e2b.RESUME_WAIT_S}
    perf = json.loads((ROOT / "evidence/runs/e-e2b-2026-10-17/ci-perf.json").read_text(encoding="utf-8"))
    assert perf["inputs"] == {"cold": measure_e2b.COLD_Q, "warm": measure_e2b.WARM_QS,
                              "scale": [measure_e2b.SCALE_WRITE, measure_e2b.SCALE_READ],
                              "live": [measure_e2b.LIVE_WRITE, measure_e2b.LIVE_READ]}


# --- measure_code_interpreter: an offline fake of the code-interpreter SDK -----------------------------------

import measure_code_interpreter as mci  # noqa: E402


class FakeKernel:
    """Executes run_code cells with the official template server's JavaScript behaviour switchable."""

    def __init__(self, js_env_bug):
        self.vars, self.js_env_bug, self.js_cells = {}, js_env_bug, 0

    def run(self, code, language="python"):
        err, out, results = None, "", []
        if language == "javascript":
            self.js_cells += 1
            if self.js_env_bug and self.js_cells == 1 and code.startswith("["):
                err = SimpleNamespace(name="TypeError", value="\"tpl\"[(1 , 2 , 3)].reduce is not a function",
                                      traceback="TypeError: \"tpl\"[(1 , 2 , 3)].reduce is not a function")
            else:
                out = "6"
        elif code == "1/0":
            err = SimpleNamespace(name="ZeroDivisionError", value="division by zero",
                                  traceback="ZeroDivisionError: division by zero")
        elif "plt.show()" in code:
            results = [SimpleNamespace(png="iVBOR", formats=lambda: ["text", "png", "chart"])]
        elif "pd.DataFrame" in code:
            out = "10"
        elif code == "x = 1":
            self.vars["x"] = 1
        elif code == "x+=1; x":
            self.vars["x"] += 1
            out = str(self.vars["x"])
        elif code == "x":
            out = str(self.vars.get("x", ""))
        return SimpleNamespace(text=out or None, error=err, results=results,
                               logs=SimpleNamespace(stdout=[], stderr=[]))


def fake_ci_sdk(js_env_bug=False, lossy_pause=False, leave_one=False, info_fails=False):
    live = {}

    class Sbx:
        def __init__(self, sid):
            self.sandbox_id, self.kernel = sid, FakeKernel(js_env_bug)

        def get_info(self):
            if info_fails:
                raise RuntimeError("GET /sandboxes/{id}: HTTP 500")
            return SimpleNamespace(template_id="tpl-1", name="code-interpreter-v1", cpu_count=2, memory_mb=2048)

        def run_code(self, code, language="python"):
            return self.kernel.run(code, language)

        def pause(self):
            if lossy_pause:
                self.kernel = FakeKernel(js_env_bug)
            return True

        def kill(self):
            if not leave_one:
                live.pop(self.sandbox_id, None)
            return True

    class Sandbox:
        default_template = "code-interpreter-v1"

        @staticmethod
        def create(timeout=600, **opts):
            assert opts == {"api_url": "http://a:3000", "sandbox_url": "http://a:3002", "api_key": "test-key"}
            s = Sbx(f"sbx-{len(live) + 1}")
            live[s.sandbox_id] = s
            return s

        @staticmethod
        def connect(sid, **opts):
            return live[sid]

        @staticmethod
        def list(query=None, **opts):
            pages = [[SimpleNamespace(sandbox_id=i) for i in live]]

            class Pager:
                @property
                def has_next(self):
                    return bool(pages)

                def next_items(self, **o):
                    return pages.pop()
            return Pager()

    return Sandbox, (lambda state=None: state), SimpleNamespace(RUNNING="running", PAUSED="paused")


@pytest.mark.parametrize("case,code,failed", [("ok", 0, []), ("js", 1, ["javascript kernel"]),
                                              ("lossy", 1, ["x after resume"]), ("leftover", 1, [])])
def test_code_interpreter_check(monkeypatch, tmp_path, case, code, failed):
    for k, v in {"E2B_API_URL": "http://a:3000", "E2B_SANDBOX_URL": "http://a:3002", "E2B_API_KEY": "test-key"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(mci.time, "sleep", lambda _: None)
    monkeypatch.setattr(mci, "load_sdk", lambda: fake_ci_sdk(js_env_bug=case == "js", lossy_pause=case == "lossy",
                                                            leave_one=case == "leftover"))
    assert mci.main(["--out-dir", str(tmp_path)]) == code
    doc = json.loads(next(tmp_path.glob("ci-test-*.json")).read_text(encoding="utf-8"))
    assert [s["step"] for s in doc["steps"] if not s["ok"]] == failed
    assert doc["leftover_sandboxes"] == (1 if case == "leftover" else 0)
    assert "test-key" not in json.dumps(doc)
    zero = next(s for s in doc["steps"] if s["step"] == "exception")
    assert zero["ok"] and zero["execution"]["error"]["name"] == "ZeroDivisionError"
    if case == "js":
        js = next(s for s in doc["steps"] if s["step"] == "javascript kernel")
        assert js["execution"]["error"]["name"] == "TypeError" and js["observed"]["error"] == "TypeError"


def test_code_interpreter_kills_a_sandbox_whose_create_check_failed(monkeypatch, tmp_path):
    for k, v in {"E2B_API_URL": "http://a:3000", "E2B_SANDBOX_URL": "http://a:3002", "E2B_API_KEY": "test-key"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(mci.time, "sleep", lambda _: None)
    monkeypatch.setattr(mci, "load_sdk", lambda: fake_ci_sdk(info_fails=True))
    assert mci.main(["--out-dir", str(tmp_path)]) == 1
    doc = json.loads(next(tmp_path.glob("ci-test-*.json")).read_text(encoding="utf-8"))
    assert [(s["step"], s["ok"]) for s in doc["steps"]] == [("create (default template)", False), ("kill", True)]
    assert doc["leftover_sandboxes"] == 0


def test_code_interpreter_check_needs_settings(monkeypatch, tmp_path):
    for k in ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert mci.main(["--out-dir", str(tmp_path)]) == 2


# --- 2026-10-18 supplementary modes: header auth, idle resume, A and C concurrency, C lifecycle ---------------

def test_responses_header_auth_is_sent_and_malformed_header_is_rejected(monkeypatch):
    monkeypatch.setenv("RESPONSES_AUTH_HEADER", "Ocp-Apim-Subscription-Key: test-key")
    assert measure_responses.token() == ("Ocp-Apim-Subscription-Key", "test-key")
    sent = {}

    def urlopen(req, timeout):
        sent.update({k.lower(): v for k, v in req.header_items()})
        raise measure_responses.urllib.error.URLError("offline")

    monkeypatch.setattr(measure_responses.urllib.request, "urlopen", urlopen)
    out = measure_responses.invoke("https://example.com/responses", measure_responses.token(), "Hi")
    assert sent["ocp-apim-subscription-key"] == "test-key" and "authorization" not in sent
    assert out["http"] is None and "test-key" not in json.dumps(out)
    measure_responses.invoke("https://example.com/responses", "entra-token", "Hi")
    assert sent["authorization"] == "Bearer entra-token"
    monkeypatch.setenv("RESPONSES_AUTH_HEADER", "no-colon-here")
    with pytest.raises(RuntimeError, match="Header-Name: value"):
        measure_responses.token()


@pytest.mark.parametrize("goes_idle,code", [(True, 0), (False, 1)])
def test_sessions_idle_resume_needs_an_idle_session(monkeypatch, tmp_path, goes_idle, code):
    api = FakeSessionsAPI()
    waits = []

    def sleep(seconds):
        if seconds >= 60:
            waits.append(seconds)
            if goes_idle:
                api.sessions.update({sid: "idle" for sid in api.sessions})

    monkeypatch.setattr(measure_sessions, "auth_header", lambda: ("Ocp-Apim-Subscription-Key", "test-key"))
    monkeypatch.setattr(measure_sessions.Client, "call", lambda self, *a, **k: api.call(*a, **k))
    monkeypatch.setattr(measure_sessions.time, "sleep", sleep)
    args = ["--project-endpoint", "https://example.com/api/projects/p", "--agent", "maf-agent", "--version", "2",
            "--runs", "1", "--resume-after", "idle", "--idle-wait", "1020", "--path", "D",
            "--out", str(tmp_path / "raw.json")]
    assert measure_sessions.main(args) == code
    raw = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    run = raw["runs"][0]
    assert waits == [1020] and "stop" not in run and raw["resume_after"] == "idle"
    assert run["idle"]["state_before_resume"] == ("idle" if goes_idle else "active")
    assert raw["path"] == "D" and raw["what"].startswith("BYO-AKS") and "1020 s without requests" in raw["what"]
    assert raw["summary"]["runs_complete"]["count"] == 1 and raw["summary"]["stop_to_idle_s"]["n"] == 0
    assert not api.sessions and "test-key" not in json.dumps(raw)


@pytest.mark.parametrize("forget", (set(), {1}))
def test_aca_scale_counts_only_sessions_that_remember(monkeypatch, forget):
    class Box:
        def __init__(self, k):
            self.k, self.memory, self.deleted = k, None, False

        def delete(self):
            self.deleted = True

    boxes = []

    def create(**kwargs):
        boxes.append(Box(int(kwargs["agent_id"].rsplit("-", 2)[-2])))
        return boxes[-1], SimpleNamespace(id=f"sandbox-{boxes[-1].k}", state="Running")

    def ask(box, question):
        if question.startswith("Remember "):
            box.memory = None if box.k in forget else question.split()[1].rstrip(".")
            return {"guest_s": 1.0, "response": "OK", "history_length": 1}
        return {"guest_s": 1.0, "response": box.memory or "I don't know", "history_length": 2}

    group = SimpleNamespace(_do_create_sandbox=create, get_sandbox=lambda _: None)
    monkeypatch.setattr(aca_sandbox, "agent_environment", lambda cfg, agent_id: {"agent_id": agent_id})
    monkeypatch.setattr(aca_sandbox, "config_kwargs", lambda mode, cfg, image, env: env)
    monkeypatch.setattr(aca_sandbox, "poll_state", lambda getter, targets: (
        0, SimpleNamespace(state="running", vmm_type="cloudhypervisor"), 1))
    monkeypatch.setattr(aca_sandbox, "ask", ask)
    out = measure_aca.at_once(group, {}, "image-1", 3)
    assert out["sessions_at_once"] == 3 and out["answered"] == 3 - len(forget)
    assert sorted(b.k for b in boxes) == [0, 1, 2] and all(b.deleted for b in boxes)
    assert [s["remembered"] for s in out["sessions"]] == [k not in forget for k in range(3)]
    assert out["sessions"][0]["turn1"]["response"] == "OK"


import measure_c  # noqa: E402

C_TEMPLATE = {
    "apiVersion": "agents.x-k8s.io/v1alpha1", "kind": "Sandbox",
    "metadata": {"name": "agent-host", "namespace": "agent", "uid": "workshop-uid"},
    "spec": {"operatingMode": "Running", "podTemplate": {
        "metadata": {"labels": {"app": "agent-host"}},
        "spec": {"runtimeClassName": "kata-vm-isolation", "serviceAccountName": "agent-host",
                 "containers": [{"name": "agent-host", "image": "example.azurecr.io/agent-host:latest",
                                 "env": [{"name": "AGENT_ID",
                                          "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                                         {"name": "STATE_MOUNT_PATH", "value": "/data"}],
                                 "volumeMounts": [{"name": "state", "mountPath": "/data"},
                                                  {"name": "tmp", "mountPath": "/tmp"}]}],
                 "volumes": [{"name": "state", "persistentVolumeClaim": {"claimName": "agent-state"}},
                             {"name": "tmp", "emptyDir": {}}]}}},
}


@pytest.mark.parametrize("pvc", (False, True))
def test_c_manifest_copies_the_workshop_spec_under_a_new_label(pvc):
    before = json.dumps(C_TEMPLATE, sort_keys=True)
    m = measure_c.sandbox_manifest(C_TEMPLATE, "agent-host-measure-cold-1-1", "agent", pvc, "20261018t000000")
    assert json.dumps(C_TEMPLATE, sort_keys=True) == before
    assert m["metadata"] == {"name": "agent-host-measure-cold-1-1", "namespace": "agent",
                             "labels": {"app": "agent-host-measure", "measure-run": "20261018t000000"}}
    pod = m["spec"]["podTemplate"]
    assert pod["metadata"]["labels"]["app"] == "agent-host-measure" and m["spec"]["operatingMode"] == "Running"
    assert pod["spec"]["runtimeClassName"] == "kata-vm-isolation"
    names = [v["name"] for v in pod["spec"]["volumes"]]
    mounts = [v["name"] for v in pod["spec"]["containers"][0]["volumeMounts"]]
    assert names == mounts == (["state", "tmp"] if pvc else ["tmp"])
    # AGENT_ID comes from the pod name, so every copy keeps its conversation in its own state file
    assert pod["spec"]["containers"][0]["env"] == C_TEMPLATE["spec"]["podTemplate"]["spec"]["containers"][0]["env"]


class FakeCSandbox:
    """measure_c.Sandbox for one pod whose conversation survives a new pod only when it is on the volume."""

    def __init__(self, keeps_state, history=0):
        self.keeps_state, self.uid, self.history, self.mode, self.deleted = keeps_state, 1, history, "Running", False

    def new_pod(self):
        self.uid += 1
        if not self.keeps_state:
            self.history = 0

    def get(self):
        ready = "True" if self.mode == "Running" else "False"
        return {"status": {"conditions": [{"type": "Ready", "status": ready, "reason": self.mode}]}}

    def pod(self):
        return {"metadata": {"uid": str(self.uid), "name": "pod"}} if self.mode == "Running" else None

    def wait(self, test, limit, what):
        assert test(), what
        return 0.0

    def ready(self, old_uid=None):
        return self.mode == "Running" and str(self.uid) != old_uid

    def ask(self, question):
        self.history += 1
        return {"guest_s": 1.0, "exec_wall_s": 2.0, "response": "2", "history_length": self.history}

    def server_times(self):
        return {"node": "aks-kata-0"}

    def patch_mode(self, mode):
        self.mode = mode
        if mode == "Running":
            self.new_pod()

    def delete(self):
        self.deleted = True
        return True


@pytest.mark.parametrize("keeps_state", (True, False))
def test_c_pvc_run_needs_the_conversation_to_survive_pod_loss_and_suspend(monkeypatch, keeps_state):
    box = FakeCSandbox(keeps_state, history=7 if keeps_state else 0)
    monkeypatch.setattr(measure_c, "Sandbox", lambda ns, name, container: box)
    monkeypatch.setattr(measure_c, "apply", lambda objs: None)
    monkeypatch.setattr(measure_c, "kubectl", lambda *args, **kwargs: box.new_pod() if args[:2] == ("delete", "pod")
                        else None)
    monkeypatch.setattr(measure_c.time, "sleep", lambda _: None)
    rec = measure_c.cold_run(C_TEMPLATE, "agent", "agent-host", "20261018t000000", 1, pvc=True)
    assert rec["deleted"] and box.deleted
    if keeps_state:
        assert "error" not in rec and rec["history_grew"]
        assert [rec[k]["history_length"] for k in ("first", "follow_up", "after_pod_delete", "after_resume")] == [8, 9, 10, 11]
        assert rec["suspended_condition"] == {"status": "False", "reason": "Suspended"}
        assert rec["resume_to_answer_s"] == pytest.approx(rec["resume_to_ready_s"] + 1.0)
    else:
        assert rec["history_grew"] is False and "did not keep growing" in rec["error"]


def test_c_history_must_grow_at_every_turn():
    assert measure_c.history_grew([{"history_length": n} for n in (1, 2, 3, 4)])
    assert not measure_c.history_grew([{"history_length": n} for n in (1, 2, 1, 2)])
    assert not measure_c.history_grew([{"history_length": 1}, {"history_length": None}])


def test_c_scale_counts_only_sessions_that_remember(monkeypatch):
    boxes = {}

    class Box(FakeCSandbox):
        def __init__(self, name):
            super().__init__(True)
            self.name, self.word = name, None

        def ask(self, question):
            if question.startswith("Remember "):
                self.word = None if self.name.endswith("-1-" + "000000") else question.split()[1].rstrip(".")
                return {"guest_s": 1.0, "exec_wall_s": 2.0, "response": "OK", "history_length": 1}
            return {"guest_s": 1.0, "exec_wall_s": 2.0, "response": self.word or "nothing", "history_length": 2}

    monkeypatch.setattr(measure_c, "Sandbox", lambda ns, name, container: boxes.setdefault(name, Box(name)))
    applied = []
    monkeypatch.setattr(measure_c, "apply", lambda objs: applied.extend(o["metadata"]["name"] for o in objs))
    out = measure_c.at_once(C_TEMPLATE, "agent", "agent-host", "20261018t000000", 3)
    assert len(applied) == 3 and out["sessions_at_once"] == 3 and out["answered"] == 2 and out["deleted"] == 3
    assert [s["remembered"] for s in out["sessions"]] == [True, False, True] and out["nodes"] == ["aks-kata-0"]
