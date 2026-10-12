"""Every guard fails once when the content it protects is broken (one deliberate break per guard)."""
import json
import re

import pytest

import validate_docs as vd

PAGES = ("README.md", "README_CN.md", "VALIDATION.md", "VALIDATION_CN.md")


def pages():
    return {n: (vd.ROOT / n).read_text(encoding="utf-8") for n in PAGES}


def test_removed_readme_heading_fails():
    p = pages()
    p["README_CN.md"] = p["README_CN.md"].replace("## 操作步骤", "## 部署")
    assert vd.check_readme_sections(p)


def test_extra_readme_section_fails():
    p = pages()
    p["README.md"] += "\n## FAQ\n\nMore.\n"
    assert vd.check_readme_sections(p)


def test_missing_b_from_isolation_table_fails():
    p = pages()
    p["README_CN.md"] = p["README_CN.md"].replace(
        "| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS |",
        "| | A · ACA Sandboxes | C · AKS + agent-sandbox | D · BYO-AKS |",
        1,
    )
    assert any("A, B, C and D" in e for e in vd.check_four_path_tables(p))


def test_same_container_claim_fails_information_alignment():
    p = pages()
    p["README.md"] = p["README.md"].replace(
        "This repository applies one comparable stateful interaction contract to all four paths",
        "This repository puts the same sample agent on all four",
        1,
    )
    assert vd.check_information_alignment(p)


def test_deleted_session_state_overclaim_fails_information_alignment():
    p = pages()
    p["VALIDATION.md"] = p["VALIDATION.md"].replace(
        "does **not** establish continuity of the deleted session",
        "proves that the deleted session kept its state",
        1,
    )
    assert vd.check_information_alignment(p)


def test_upstream_letter_drift_fails_information_alignment():
    p = pages()
    p["README.md"] = p["README.md"].replace("module-04 Solution C", "module-04 Solution A", 1)
    assert vd.check_information_alignment(p)


def test_table_before_input_fails():
    p = pages()
    p["VALIDATION.md"] = p["VALIDATION.md"].replace(
        "**Input.** Each run created", "| a | b |\n|---|---|\n\n**Input.** Each run created", 1)
    assert any("before the input" in e for e in vd.check_run_sections(p))


def test_boundary_missing_fails():
    p = pages()
    sec_start = p["VALIDATION_CN.md"].index("### d-2026-09-29-recovery")
    head, tail = p["VALIDATION_CN.md"][:sec_start], p["VALIDATION_CN.md"][sec_start:]
    p["VALIDATION_CN.md"] = head + tail.replace("**边界。**", "**说明。**", 1)
    assert any("边界" in e for e in vd.check_run_sections(p))


def test_run_missing_from_results_table_fails():
    p = pages()
    p["README.md"] = p["README.md"].replace("| `d-2026-09-29-recovery` |", "| `d-recovery` |")
    assert any("d-2026-09-29-recovery" in e for e in vd.check_runs_listed(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
@pytest.mark.parametrize("pair", ((0, 1), (1, 2), (4, 5)))
def test_reordered_run_rows_fail(name, pair):
    p = pages()
    lines = p[name].splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if re.match(rf"\| `({vd.RUN_ID_PATTERN})` \|", line)]
    first, second = (indices[i] for i in pair)
    lines[first], lines[second] = lines[second], lines[first]
    p[name] = "".join(lines)
    assert any(name in e for e in vd.check_run_order(p))


@pytest.mark.parametrize("name", ("VALIDATION.md", "VALIDATION_CN.md"))
def test_reordered_run_sections_fail(name):
    p = pages()
    sections = list(vd.run_sections(p[name]).values())
    p[name] = p[name].replace(sections[0] + sections[1], sections[1] + sections[0], 1)
    assert any(name in e for e in vd.check_run_order(p))


@pytest.mark.parametrize("fault", ("date-regression", "duplicate"))
def test_invalid_canonical_run_order_fails(monkeypatch, fault):
    p = pages()
    read = vd.read
    claims = json.loads(read("evidence/claims.json"))
    if fault == "date-regression":
        claims["run_order"][0], claims["run_order"][1] = claims["run_order"][1], claims["run_order"][0]
    else:
        claims["run_order"][1] = claims["run_order"][0]
    monkeypatch.setattr(vd, "read", lambda name: json.dumps(claims) if name == "evidence/claims.json" else read(name))
    assert any("evidence/claims.json" in e for e in vd.check_run_order(p))


def test_edited_number_in_one_language_fails():
    p = pages()
    p["README_CN.md"] = p["README_CN.md"].replace("每个沙箱一个 micro-VM", "每 2 个沙箱一个 micro-VM", 1)
    assert vd.check_numbers(p)


def test_swapped_numbers_with_unchanged_totals_fail():
    p = pages()
    start = p["README_CN.md"].index("<!-- BEGIN GENERATED: comparison -->")
    end = p["README_CN.md"].index("<!-- END GENERATED: comparison -->", start)
    block = p["README_CN.md"][start:end]
    changed = block.replace("28.0", "__SWAP__", 1).replace("25.49", "28.0", 1).replace("__SWAP__", "25.49", 1)
    p["README_CN.md"] = p["README_CN.md"][:start] + changed + p["README_CN.md"][end:]
    assert vd.numbers(p["README.md"]) == vd.numbers(p["README_CN.md"])
    assert vd.check_numbers(p)


@pytest.mark.parametrize("name", PAGES)
def test_resume_clock_drift_fails(name):
    p = pages()
    if name.endswith("_CN.md"):
        p[name] = p[name].replace("不含停止和空闲等待", "包含停止和空闲等待")
    else:
        p[name] = p[name].replace("stop and idle waiting are excluded", "stop and idle waiting are included")
    assert any(name in error for error in vd.check_information_alignment(p))


@pytest.mark.parametrize("name", PAGES)
def test_pod_age_is_not_an_idle_duration(name):
    p = pages()
    if name.endswith("_CN.md"):
        p[name] = p[name].replace("pod 的年龄，不是空闲时长", "实测的空闲时长")
    else:
        p[name] = p[name].replace("pod age, not idle duration", "measured idle duration")
    assert any(name in error for error in vd.check_information_alignment(p))


def test_broken_link_fails():
    p = pages()
    p["README.md"] += "\nSee [missing](tools/missing.py).\n"
    assert vd.check_links(p)


def test_wide_diagram_fails():
    p = pages()
    p["README.md"] = p["README.md"].replace('architecture-en.png" width="900"', 'architecture-en.png" width="1600"')
    assert vd.check_images(p)


def test_literal_subscription_in_steps_fails():
    p = pages()
    p["README.md"] = p["README.md"].replace(
        'az account set --subscription "$SUBSCRIPTION_ID"',
        "az account set --subscription 12345678-1234-1234-1234-123456789abc", 1)
    assert vd.check_commands(p)


def test_private_content_fails(tmp_path):
    for leak in ("subscription 12345678-1234-1234-1234-123456789abc", "router at 20.242.33.10",
                 "mail someone@contoso.com", "cluster aks-byoaks", "sessions were not observed on the cluster"):
        f = tmp_path / "leak.md"
        f.write_text(f"# x\n\n{leak}\n", encoding="utf-8")
        assert vd.check_private({}, [f]), leak


def test_clean_text_passes_private(tmp_path):
    f = tmp_path / "ok.md"
    f.write_text("# x\n\nPOST http://127.0.0.1:8080/reflect with 10.225.1.0/24 and user@example.com\n", encoding="utf-8")
    assert vd.check_private({}, [f]) == []
