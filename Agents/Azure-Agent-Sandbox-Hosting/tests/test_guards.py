"""Deliberately break each publication guard and require it to report the error."""
import json
import re

import pytest

import validate_docs as vd


def pages():
    return {name: (vd.ROOT / name).read_text(encoding="utf-8") for name in vd.PAGES}


def test_removed_readme_heading_fails():
    documents = pages()
    documents["README.md"] = documents["README.md"].replace("## Steps", "## Deployment")
    assert vd.check_sections(documents)


def test_extra_readme_section_fails():
    documents = pages()
    documents["README.md"] += "\n## FAQ\n\nMore.\n"
    assert vd.check_sections(documents)


def test_missing_b_from_isolation_table_fails():
    documents = pages()
    documents["README.md"] = documents["README.md"].replace(
        "| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox |",
        "| | A · ACA Sandboxes | C · AKS + agent-sandbox |", 1)
    assert vd.check_information_alignment(documents)


def test_table_before_input_fails():
    documents = pages()
    documents["VALIDATION.md"] = documents["VALIDATION.md"].replace(
        "**Input.** Each run created", "| a | b |\n|---|---|\n\n**Input.** Each run created", 1)
    assert any("before the input" in error for error in vd.check_run_sections(documents))


def test_boundary_missing_fails():
    documents = pages()
    documents["VALIDATION.md"] = documents["VALIDATION.md"].replace("**Boundary.**", "**Notes.**", 1)
    assert vd.check_run_sections(documents)


@pytest.mark.parametrize("name", vd.PAGES)
def test_reordered_runs_fail(name):
    documents = pages()
    if name.startswith("README"):
        lines = documents[name].splitlines(keepends=True)
        indices = [i for i, line in enumerate(lines) if re.match(rf"\| `({vd.RUN_ID_PATTERN})` \|", line)]
        first, second = indices[:2]
        lines[first], lines[second] = lines[second], lines[first]
        documents[name] = "".join(lines)
    else:
        sections = list(vd.run_sections(documents[name]).values())
        documents[name] = documents[name].replace(sections[0] + sections[1], sections[1] + sections[0], 1)
    assert any(name in error for error in vd.check_run_order(documents))


@pytest.mark.parametrize("fault", ("date-regression", "duplicate"))
def test_invalid_canonical_run_order_fails(monkeypatch, fault):
    read = vd.read
    claims = json.loads(read("evidence/claims.json"))
    if fault == "date-regression":
        claims["run_order"][0], claims["run_order"][1] = claims["run_order"][1], claims["run_order"][0]
    else:
        claims["run_order"][1] = claims["run_order"][0]
    monkeypatch.setattr(vd, "read", lambda name: json.dumps(claims) if name == "evidence/claims.json" else read(name))
    assert vd.check_run_order(pages())


def test_edited_number_in_one_language_fails():
    documents = pages()
    documents["README_CN.md"] += "\n12345\n"
    assert vd.check_numbers(documents)


def test_swapped_numbers_with_unchanged_totals_fail():
    documents = pages()
    start = documents["README_CN.md"].index("<!-- BEGIN GENERATED: comparison -->")
    end = documents["README_CN.md"].index("<!-- END GENERATED: comparison -->", start)
    block = documents["README_CN.md"][start:end]
    changed = block.replace("28.0", "__SWAP__", 1).replace("14.45", "28.0", 1).replace("__SWAP__", "14.45", 1)
    assert changed != block
    documents["README_CN.md"] = documents["README_CN.md"][:start] + changed + documents["README_CN.md"][end:]
    assert vd.numbers(documents["README.md"]) == vd.numbers(documents["README_CN.md"])
    assert vd.check_numbers(documents)


@pytest.mark.parametrize("name", vd.PAGES)
def test_resume_clock_drift_fails(name):
    documents = pages()
    if name.endswith("_CN.md"):
        documents[name] = documents[name].replace("不含停止和空闲等待", "包含停止和空闲等待")
    else:
        documents[name] = documents[name].replace(
            "stop and idle waiting are excluded", "stop and idle waiting are included")
    assert any(name in error for error in vd.check_information_alignment(documents))


def test_broken_link_fails():
    documents = pages()
    documents["README.md"] += "\nSee [missing](tools/missing.py).\n"
    assert vd.check_links_images(documents)


def test_wide_diagram_fails():
    documents = pages()
    documents["README.md"] = documents["README.md"].replace(
        'architecture-en.png" width="900"', 'architecture-en.png" width="1600"')
    assert vd.check_links_images(documents)


def test_bilingual_command_drift_fails():
    documents = pages()
    documents["README.md"] = documents["README.md"].replace(
        'azd deploy && azd ai agent invoke "Hi"', 'azd deploy && azd ai agent invoke "Hello"', 1)
    assert vd.check_commands(documents)


@pytest.mark.parametrize("leak", (
    "subscription 12345678-1234-1234-1234-123456789abc",
    "router at 20.242.33.10",
    "mail someone@contoso.com",
    "AccountKey=not-a-real-key",
))
def test_private_content_fails(tmp_path, leak):
    path = tmp_path / "leak.md"
    path.write_text(f"# x\n\n{leak}\n", encoding="utf-8")
    assert vd.check_private({}, [path])


def test_clean_text_passes_private(tmp_path):
    path = tmp_path / "ok.md"
    path.write_text(
        "# x\n\nPOST http://127.0.0.1:8080/reflect with 10.225.1.0/24 and user@example.com\n",
        encoding="utf-8")
    assert vd.check_private({}, [path]) == []


@pytest.mark.parametrize("fault", ("claim-path", "run-path", "metric", "page"))
def test_unpublished_details_fail_public_scope(monkeypatch, fault):
    read = vd.read
    documents = pages()

    def changed(name):
        value = read(name)
        if fault == "claim-path" and name == "evidence/claims.json":
            document = json.loads(value)
            document["paths"]["D"] = {}
            return json.dumps(document)
        if fault == "run-path" and name.startswith("evidence/runs/") and name != "evidence/runs/INDEX.json":
            document = json.loads(value)
            document["path"] = "D"
            return json.dumps(document)
        if fault == "metric" and name == "evidence/measurements.json":
            document = json.loads(value)
            document["metrics"]["d.unpublished"] = {"value": 1}
            return json.dumps(document)
        return value

    if fault == "page":
        documents["README.md"] += "\n<!--v:d.unpublished.value-->1<!--/v-->\n"
    monkeypatch.setattr(vd, "read", changed)
    assert vd.check_public_scope(documents)
