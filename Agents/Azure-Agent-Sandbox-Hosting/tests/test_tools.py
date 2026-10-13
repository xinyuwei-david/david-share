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
TOOLS = ["aca_sandbox.py", "measure_aca.py", "measure_responses.py", "check_isolation.py", "build_measurements.py",
         "render_docs.py", "build_diagrams.py", "export_evidence.py", "validate_docs.py"]


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


def test_builtin_config_needs_no_sdk_and_full_needs_an_image():
    cfg = {"vnet_connection": "aca-vnet", "volume": "v", "allow_cidr": ""}
    kw = aca_sandbox.config_kwargs("builtin", cfg, None, None)
    assert kw["disk"] == "ubuntu" and kw["cpu"] == "1000m" and kw["auto_suspend_mode"] == "Memory"
    with pytest.raises(aca_sandbox.SettingMissing):
        aca_sandbox.config_kwargs("full", cfg, None, {})
    with pytest.raises(ValueError):
        aca_sandbox.config_kwargs("everything", cfg, "img", {})


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
    turns = iter({"guest_s": 1.0, "history_length": length} for length in lengths)
    monkeypatch.setattr(aca_sandbox, "agent_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(aca_sandbox, "config_kwargs", lambda *args: {})
    monkeypatch.setattr(aca_sandbox, "poll_state", lambda getter, targets: (
        0, SimpleNamespace(state=targets[0], vmm_type="cloudhypervisor"), 1))
    monkeypatch.setattr(aca_sandbox, "ask", lambda *args: next(turns))
    monkeypatch.setattr(measure_aca.time, "sleep", lambda _: None)
    result = measure_aca.agent_run(group, {}, "image-1", 1)
    if lengths == [1, 2, 3, 4]:
        assert result["state_kept"] is True and "error" not in result
    else:
        assert result["state_kept"] is False and "error" in result


def test_aca_empty_measurement_is_rejected():
    with pytest.raises(SystemExit) as exc:
        measure_aca.main(["agent", "--runs", "0"])
    assert exc.value.code == 2
