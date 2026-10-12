"""Recompute public metrics from unchanged run records and reject invalid evidence."""
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
    runs, _, metrics = built
    raw = runs["a-aca-2026-10-09/agent-runs.json"]["runs"]
    assert metrics["a.cold"]["median"] == statistics.median(r["cold_total_s"] for r in raw) == 28.0
    assert metrics["a.warm"]["median"] == statistics.median(r["warm"]["guest_s"] for r in raw) == 1.05
    assert metrics["a.recover"]["median"] == statistics.median(r["recover_total_s"] for r in raw) == 14.45
    assert metrics["a.create"]["median"] == 24.25
    assert metrics["a.vmm"]["value"] == "cloudhypervisor" and metrics["a.vmm"]["count"] == 3


def test_composite_medians_are_not_sums_of_component_medians(built):
    _, _, metrics = built
    assert metrics["a.cold"]["median"] != round(
        metrics["a.create"]["median"] + metrics["a.first_answer"]["median"], 2)
    assert metrics["a.recover"]["median"] != round(
        metrics["a.resume"]["median"] + metrics["a.first_after_resume"]["median"], 2)


def test_public_dataset_contains_only_seven_abc_records(built):
    runs, index, metrics = built
    assert len(runs) == len(index) == 7
    assert all(run["path"] in ("A", "B", "C", "shared") for run in runs.values())
    assert all(key.startswith(("a.", "b.", "c.", "f.")) for key in metrics)


def test_even_samples_have_no_median(built):
    _, _, metrics = built
    for key, record in metrics.items():
        if "min" in record and isinstance(record.get("values"), list) and record.get("n") == len(record["values"]):
            assert ("median" in record) == (record["n"] % 2 == 1), key


def test_every_run_file_names_its_private_sources(built):
    runs, _, _ = built
    for relative, document in runs.items():
        assert document["_sources"], relative
        for source in document["_sources"]:
            assert len(source["sha256"]) == 64 and source["bytes"] > 0, (relative, source)


def copy_runs(tmp_path):
    destination = tmp_path / "runs"
    shutil.copytree(bm.RUNS, destination)
    return destination


def test_tampered_run_file_fails(tmp_path):
    runs_dir = copy_runs(tmp_path)
    path = runs_dir / "a-aca-2026-10-09" / "agent-runs.json"
    path.write_text(path.read_text(encoding="utf-8").replace(
        '"cold_total_s": 28.0', '"cold_total_s": 18.0'), encoding="utf-8")
    with pytest.raises(bm.BuildError, match="SHA-256"):
        bm.load_runs(runs_dir)


def test_extra_unindexed_run_file_fails(tmp_path):
    runs_dir = copy_runs(tmp_path)
    (runs_dir / "stray.json").write_text("{}", encoding="utf-8")
    with pytest.raises(bm.BuildError, match="INDEX"):
        bm.load_runs(runs_dir)


def test_inconsistent_cold_total_fails():
    runs, index = bm.load_runs()
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
