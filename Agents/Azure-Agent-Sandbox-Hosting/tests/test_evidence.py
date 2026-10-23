"""The published numbers are recomputed from the committed run records, and the build fails closed."""
import json
import shutil
import statistics

import pytest

import build_measurements as bm


@pytest.fixture(scope="module")
def built():
    runs, index = bm.load_runs()
    return runs, index, bm.build(runs, index)["metrics"]


def test_committed_measurements_equal_a_fresh_build(built):
    runs, index, _ = built
    assert bm.OUT.read_text(encoding="utf-8") == bm.render(bm.build(runs, index))


def test_a_medians_recomputed_from_raw_runs(built):
    runs, _, m = built
    raw = runs["a-aca-2026-10-09/agent-runs.json"]["runs"]
    assert m["a.cold"]["median"] == statistics.median(r["cold_total_s"] for r in raw) == 28.0
    assert m["a.warm"]["median"] == statistics.median(r["warm"]["guest_s"] for r in raw) == 1.05
    assert m["a.recover"]["median"] == statistics.median(r["recover_total_s"] for r in raw) == 14.45
    assert m["a.create"]["median"] == 24.25
    assert m["a.vmm"]["value"] == "cloudhypervisor" and m["a.vmm"]["count"] == 3


def test_composite_medians_are_not_sums_of_component_medians(built):
    _, _, m = built
    assert m["a.cold"]["median"] != round(m["a.create"]["median"] + m["a.first_answer"]["median"], 2)
    assert m["a.recover"]["median"] != round(m["a.resume"]["median"] + m["a.first_after_resume"]["median"], 2)


def test_d_cold_set_is_the_five_cached_image_sessions(built):
    runs, _, m = built
    assert m["d.cold"]["n"] == 5
    assert m["d.cold"]["median"] == 25.494
    assert (m["d.cold"]["min"], m["d.cold"]["max"]) == (23.532, 26.7)
    assert sorted(m["d.cold_fresh"]["values"]) == [40.6, 41.9]
    bench = runs["d-byoaks-2026-09-29/bench.json"]
    assert all(t["pod"]["runtime"] == "kata-vm-isolation" for t in bench["cold_turns"])


def test_d_capacity_and_refusals(built):
    _, _, m = built
    assert m["d.capacity"]["per_node"] == 5 and m["d.capacity"]["max_sessions"] == 12
    assert m["d.c4"]["answered"] == 0 and m["d.c4"]["refused_codes"] == ["500 session_creation_failed"]
    assert m["d.c8"]["answered"] == 2 and m["d.c8"]["refused_codes"] == ["429 aks_quota_exceeded"]


def test_d_recovery_keeps_the_different_session_boundary(built):
    _, _, m = built
    assert "d.pod_lost_followup" not in m
    assert m["d.other_session_followup"]["value"] == 14.8
    assert "different live session" in m["d.other_session_followup"]["what"]


def test_even_samples_have_no_median(built):
    _, _, m = built
    for key, rec in m.items():
        if "min" in rec and isinstance(rec.get("values"), list) and rec.get("n") == len(rec["values"]):
            assert ("median" in rec) == (rec["n"] % 2 == 1), key


def test_every_run_file_names_its_private_sources(built):
    runs, _, _ = built
    for rel, doc in runs.items():
        assert doc["_sources"], rel
        for s in doc["_sources"]:
            assert len(s["sha256"]) == 64 and s["bytes"] > 0, (rel, s)


def copy_runs(tmp_path):
    dst = tmp_path / "runs"
    shutil.copytree(bm.RUNS, dst)
    return dst


def test_tampered_run_file_fails(tmp_path):
    runs_dir = copy_runs(tmp_path)
    p = runs_dir / "a-aca-2026-10-09" / "agent-runs.json"
    p.write_text(p.read_text(encoding="utf-8").replace('"cold_total_s": 28.0', '"cold_total_s": 18.0'), encoding="utf-8")
    with pytest.raises(bm.BuildError, match="SHA-256"):
        bm.load_runs(runs_dir)


def test_extra_unindexed_run_file_fails(tmp_path):
    runs_dir = copy_runs(tmp_path)
    (runs_dir / "stray.json").write_text("{}", encoding="utf-8")
    with pytest.raises(bm.BuildError, match="INDEX"):
        bm.load_runs(runs_dir)


def test_inconsistent_cold_total_fails():
    runs, index = bm.load_runs()
    runs = json.loads(json.dumps(runs))
    runs["a-aca-2026-10-09/agent-runs.json"]["runs"][0]["cold_total_s"] += 5
    with pytest.raises(bm.BuildError, match="cold_total_s"):
        bm.build(runs, index)


def test_a_operational_error_cannot_enter_the_metrics():
    runs, index = bm.load_runs()
    runs["a-aca-2026-10-09/agent-runs.json"]["runs"][0]["error"] = "request failed"
    with pytest.raises(bm.BuildError, match="recorded an error"):
        bm.build(runs, index)


def test_attribution_uptime_flag_must_match_the_observation():
    runs, index = bm.load_runs()
    runs["a-aca-2026-10-09/create-attribution.json"]["runs"][0]["process_survived"] = False
    with pytest.raises(bm.BuildError, match="uptime observation"):
        bm.build(runs, index)


def test_forgotten_code_word_fails():
    runs, index = bm.load_runs()
    runs = json.loads(json.dumps(runs))
    runs["d-byoaks-2026-09-29/bench.json"]["warm_turns"][1]["answer"] = "I don't remember"
    with pytest.raises(bm.BuildError, match="LANTERN-7"):
        bm.build(runs, index)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf"), True, None, "1.0"])
@pytest.mark.parametrize("summarize", [bm.stat, bm.single])
def test_invalid_measurement_numbers_fail(value, summarize):
    with pytest.raises(bm.BuildError, match="finite non-negative number"):
        summarize([value] if summarize is bm.stat else value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_invalid_duration_in_raw_runs_fails(value):
    runs, index = bm.load_runs()
    runs["a-aca-2026-10-09/agent-runs.json"]["runs"][0]["warm"]["guest_s"] = value
    with pytest.raises(bm.BuildError, match=r"agent-runs.json.*warm.guest_s"):
        bm.build(runs, index)


def test_invalid_setup_component_fails():
    runs, index = bm.load_runs()
    runs["a-aca-2026-09-27/setup-and-wake.json"]["events"][0]["seconds"] = -1.0
    with pytest.raises(bm.BuildError, match=r"setup-and-wake.json.*seconds"):
        bm.build(runs, index)


def test_clock_descriptions_are_not_measurement_samples():
    runs, index = bm.load_runs()
    runs["a-aca-2026-10-09/agent-runs.json"]["clocks"]["guest_s"] = "inside the sandbox, not client wall time"
    assert bm.build(runs, index)["metrics"]["a.warm"]["median"] == 1.05


@pytest.mark.parametrize("field,value", [("lb_health_200_seconds", 100.0), ("sandbox_ready_seconds", 30.0)])
def test_c_resume_milestones_cannot_run_backwards(field, value):
    runs, index = bm.load_runs()
    runs["c-aks-agent-sandbox-2026-09-27/run.json"]["resume"][field] = value
    with pytest.raises(bm.BuildError, match="C resume"):
        bm.build(runs, index)


def test_d_pod_ready_cannot_exceed_its_paired_request():
    runs, index = bm.load_runs()
    runs["d-byoaks-2026-09-29/bench.json"]["cold_turns"][0]["wall_s"] = 1.0
    with pytest.raises(bm.BuildError, match="paired request wall time"):
        bm.build(runs, index)


def test_d_concurrency_requires_every_session_record():
    runs, index = bm.load_runs()
    runs["d-byoaks-2026-09-29/bench.json"]["scalability"][0]["sessions"].pop()
    with pytest.raises(bm.BuildError, match="expected 4 session records"):
        bm.build(runs, index)


def test_e_code_interpreter_capacity_matches_the_free_pages(built):
    _, _, m = built
    idle = m["e.ci.idle"]
    assert (idle["ci_pages"], idle["base_pages"]) == (1024, 256)
    for pool in (4096, 2048):
        assert idle[f"ci_{pool}"] == idle[f"free_{pool}"] // idle["ci_pages"]
        assert idle[f"base_{pool}"] == idle[f"free_{pool}"] // idle["base_pages"]
    assert (m["e.ci.c4"]["answered"], m["e.ci.c8"]["answered"]) == (idle["ci_4096"], idle["ci_4096"])


def test_e_code_interpreter_capacity_count_cannot_drift():
    runs, index = bm.load_runs()
    runs["e-e2b-2026-10-17/ci-perf.json"]["idle_capacity"]["phases"][0]["results"][0]["running_at_once"] = 4
    with pytest.raises(bm.BuildError, match="free pages / template pages"):
        bm.build(runs, index)


def test_e_code_interpreter_refusal_must_be_a_placement_refusal():
    runs, index = bm.load_runs()
    level = runs["e-e2b-2026-10-17/ci-perf.json"]["at_once"][0]
    refused = next(s for s in level["sessions"] if not s.get("remembered"))
    refused["error"] = "ReadTimeout: the model did not answer"
    with pytest.raises(bm.BuildError, match="another reason than a refused sandbox"):
        bm.build(runs, index)


def test_render_rejects_non_standard_json_numbers():
    with pytest.raises(ValueError, match="JSON compliant"):
        bm.render({"value": float("nan")})
