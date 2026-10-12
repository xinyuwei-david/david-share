"""Published pages and figures must match a fresh render of the canonical inputs."""
import json
import re

import pytest

import build_diagrams as bd
import render_docs as rd
import validate_docs as vd


def committed():
    return {name: (rd.ROOT / name).read_text(encoding="utf-8") for name in vd.PAGES}


def test_pages_equal_a_fresh_render_and_render_is_idempotent():
    metrics, runs = rd.load_inputs()
    for name, language in rd.PAGES.items():
        text = committed()[name]
        once = rd.render_page(text, language, metrics, runs)
        assert once == text, f"{name} is stale; run tools/render_docs.py"
        assert rd.render_page(once, language, metrics, runs) == once


@pytest.mark.parametrize("name", vd.PAGES)
def test_runs_are_grouped_by_path_then_date(name):
    text = committed()[name]
    order = re.findall(rf"(?m)^\| `({vd.RUN_ID_PATTERN})` \|", text) \
        if name.startswith("README") else list(vd.run_sections(text))
    assert order
    assert order == sorted(order, key=lambda run: (run[0], run[2:12]))


@pytest.mark.parametrize("rule", list(vd.RULES))
def test_page_rule_passes(rule):
    assert vd.RULES[rule](committed()) == []


def test_figures_match_ledger():
    ledger = json.loads(bd.LEDGER.read_text(encoding="utf-8"))
    assert bd.check(ledger) == []
    assert len(ledger["icons"]) == 7
    assert len(ledger["figures"]) == 4
    for figure in ledger["figures"]:
        if figure["file"].endswith(".png"):
            assert (figure["width"], figure["height"]) == (bd.W * bd.PNG_SCALE, bd.H * bd.PNG_SCALE)


def test_readme_headline_numbers_come_from_evidence():
    metrics, _ = rd.load_inputs()
    for name in ("README.md", "README_CN.md"):
        text = committed()[name]
        assert "28.0" in text and "14.45" in text
        for key in re.findall(r"<!--v:([a-z0-9_.]+)-->", text):
            assert key.rpartition(".")[0] in metrics, key


def test_every_generated_block_is_known():
    for name, text in committed().items():
        for block in re.findall(r"<!-- BEGIN GENERATED: ([a-z0-9-]+) -->", text):
            assert block in rd.BLOCKS, (name, block)


def test_unknown_block_or_value_fails():
    metrics, runs = rd.load_inputs()
    with pytest.raises(rd.RenderError):
        rd.render_page("<!-- BEGIN GENERATED: nope -->\n<!-- END GENERATED: nope -->", "en", metrics, runs)
    with pytest.raises(rd.RenderError):
        rd.render_page("<!--v:a.cold.nothing-->1<!--/v-->", "en", metrics, runs)
