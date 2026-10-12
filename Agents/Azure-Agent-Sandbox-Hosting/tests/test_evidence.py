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
