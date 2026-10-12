"""Exercise live tools offline. Pod/node fixtures are synthetic, not placement evidence."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import aca_sandbox
import check_isolation
import measure_aca
import measure_responses

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
TOOLS = ("aca_sandbox.py", "measure_aca.py", "measure_responses.py", "check_isolation.py",
         "build_measurements.py", "render_docs.py", "build_diagrams.py", "validate_docs.py")


@pytest.mark.parametrize("tool", TOOLS)
def test_help_runs_without_azure(tool):
    result = subprocess.run([sys.executable, str(ROOT / "tools" / tool), "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_isolation_pass_and_fail_from_saved_json():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_isolation.py"), "--pods-json", str(FIX / "pods.json"),
         "--nodes-json", str(FIX / "nodes.json"), "--uname", "session-pod-1=fixture-host-kernel"],
        capture_output=True, text=True, timeout=60)
    lines = result.stdout.splitlines()
    assert result.returncode == 1
    assert lines[0].startswith("CHECK isolation PASS pod=session-pod-1 runtimeClass=kata-vm-isolation")
    assert "kernel=same-version-string-as-node-(inconclusive)" in lines[0]
    assert lines[1].startswith("CHECK isolation FAIL pod=runc-pod-1 runtimeClass=runc")


def test_isolation_kernel_differs():
    pods = json.loads((FIX / "pods.json").read_text(encoding="utf-8"))
    nodes = json.loads((FIX / "nodes.json").read_text(encoding="utf-8"))
    pods["items"] = pods["items"][:1]
    result = check_isolation.evaluate(pods, nodes, {"session-pod-1": "fixture-guest-kernel"})
    assert result[0]["verdict"] == "PASS" and result[0]["kernel"] == "differs from node"


def test_isolation_no_pod_is_blocked(tmp_path):
    empty = tmp_path / "pods.json"
    empty.write_text('{"items": []}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_isolation.py"), "--pods-json", str(empty)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 3 and "BLOCKED" in result.stdout


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
    config = {"vnet_connection": "aca-vnet", "volume": "v", "allow_cidr": ""}
    kwargs = aca_sandbox.config_kwargs("builtin", config, None, None)
    assert kwargs["disk"] == "ubuntu" and kwargs["cpu"] == "1000m" and kwargs["auto_suspend_mode"] == "Memory"
    with pytest.raises(aca_sandbox.SettingMissing):
        aca_sandbox.config_kwargs("full", config, None, {})
    with pytest.raises(ValueError):
        aca_sandbox.config_kwargs("everything", config, "image", {})


def response(answer="OK"):
    return {"http": 200, "status": "completed", "response_id": "response-1",
            "answer": answer, "wall_s": 0.1, "error": None}


@pytest.fixture
def responses_main(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_RESPONSES_URL", "https://example.com/responses")
    monkeypatch.setattr(measure_responses, "token", lambda: "test-token")
    monkeypatch.setattr(measure_responses.time, "sleep", lambda _: None)
    monkeypatch.setattr(measure_responses, "run_cold", lambda *args: [response("COLD-1")])
    monkeypatch.setattr(measure_responses, "run_warm",
                        lambda *args: {"turns": [response(), response("LANTERN-7")], "remembered": True})

    def run(*args):
        return measure_responses.main(
            ["--cold", "1", "--warm-turns", "1", "--pause", "0", "--out-dir", str(tmp_path), *args])

    return run


def test_responses_success_is_saved(monkeypatch, responses_main, tmp_path):
    assert responses_main() == 0
    report = json.loads(next(tmp_path.glob("responses-*.json")).read_text(encoding="utf-8"))
    assert report["warm"]["remembered"]
    assert measure_responses.report_passes(report)


@pytest.mark.parametrize("phase", ("scale-first-turn", "scale-followup", "scale-memory", "scale-missing-session"))
def test_responses_fail_if_any_requested_scale_check_fails(monkeypatch, responses_main, tmp_path, phase):
    first, followup = response(), response("SCALE-0")
    if phase == "scale-first-turn":
        first = {**first, "http": 429, "error": "quota exceeded"}
    elif phase == "scale-followup":
        followup = {**followup, "http": 500, "error": "follow-up failed"}
    result = {"concurrency": 1, "sessions": [
        {"t1": first, "t2": followup, "remembered": phase != "scale-memory"}]}
    if phase == "scale-missing-session":
        result["sessions"] = []
    monkeypatch.setattr(measure_responses, "run_scale", lambda *args: result)
    assert responses_main("--scale", "1") == 1
    assert len(list(tmp_path.glob("responses-*.json"))) == 1


def test_responses_refresh_token_before_each_scale_level(monkeypatch, responses_main):
    calls, seen = [], []

    def token():
        calls.append("token")
        return f"test-token-{len(calls)}"

    def scale(endpoint, auth, count):
        seen.append(auth)
        return {"concurrency": count, "sessions": [
            {"t1": response(), "t2": response(f"SCALE-{i}"), "remembered": True} for i in range(count)]}

    monkeypatch.setattr(measure_responses, "token", token)
    monkeypatch.setattr(measure_responses, "run_scale", scale)
    assert responses_main("--scale", "1,2") == 0
    assert seen == ["test-token-2", "test-token-3"]


def test_responses_missing_setting_exits_2(monkeypatch):
    monkeypatch.delenv("AGENT_RESPONSES_URL", raising=False)
    assert measure_responses.main(["--pause", "0"]) == 2


@pytest.mark.parametrize("arguments", (
    ["--cold", "0"], ["--warm-turns", "0"], ["--pause", "-1"],
    ["--scale", "0"], ["--scale", "invalid"], ["--observe"], ["--recovery"],
))
def test_invalid_or_unpublished_operations_are_rejected(arguments):
    with pytest.raises(SystemExit) as exception:
        measure_responses.main(arguments)
    assert exception.value.code == 2


def test_failed_request_is_saved_and_exits_1(monkeypatch, responses_main, tmp_path):
    def fail(*args):
        raise RuntimeError("token refresh failed")

    monkeypatch.setattr(measure_responses, "run_cold", fail)
    assert responses_main() == 1
    report = json.loads(next(tmp_path.glob("responses-*.json")).read_text(encoding="utf-8"))
    assert "token refresh failed" in report["error"]


def test_failed_warm_start_does_not_open_unrelated_followup_sessions(monkeypatch):
    calls = []
    monkeypatch.setattr(measure_responses, "invoke", lambda *args: calls.append(args) or {
        "http": 500, "status": None, "response_id": None, "answer": None, "wall_s": 0.1, "error": "creation failed"})
    result = measure_responses.run_warm("https://example.com/responses", "test-token", 3)
    assert len(calls) == 1 and not result["remembered"]


def test_warm_followups_reuse_the_previous_response(monkeypatch):
    replies = iter([{**response(), "response_id": "first"},
                    {**response("LANTERN-7"), "response_id": "second"}, response("WARM-1")])
    calls = []

    def invoke(*args):
        calls.append(args)
        return next(replies)

    monkeypatch.setattr(measure_responses, "invoke", invoke)
    result = measure_responses.run_warm("https://example.com/responses", "test-token", 2)
    assert result["remembered"]
    assert len(calls[0]) == 3
    assert calls[1][-1] == "first" and calls[2][-1] == "second"


@pytest.mark.parametrize("answer", ("WRONG", ""))
def test_incorrect_cold_answer_cannot_report_success(monkeypatch, responses_main, answer):
    monkeypatch.setattr(measure_responses, "run_cold", lambda *args: [response(answer)])
    assert responses_main() == 1


def test_incorrect_warm_answer_cannot_report_success(monkeypatch, responses_main):
    monkeypatch.setattr(measure_responses, "run_warm", lambda *args: {
        "turns": [response(), response("LANTERN-7"), response("WRONG")], "remembered": True})
    assert responses_main("--warm-turns", "2") == 1


def test_incomplete_warm_run_cannot_report_success(monkeypatch, responses_main):
    assert responses_main("--warm-turns", "3") == 1


@pytest.mark.parametrize("lengths", ([1, 2, 3, 4], [1, 2, 1, 2]))
def test_aca_agent_measurement_checks_state_across_resume(monkeypatch, lengths):
    sandbox = SimpleNamespace(stop=lambda: None, resume=lambda: None, delete=lambda: None, get=lambda: None)
    group = SimpleNamespace(
        _do_create_sandbox=lambda **kwargs: (sandbox, SimpleNamespace(id="sandbox-1", state="Running")),
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
    with pytest.raises(SystemExit) as exception:
        measure_aca.main(["agent", "--runs", "0"])
    assert exception.value.code == 2
