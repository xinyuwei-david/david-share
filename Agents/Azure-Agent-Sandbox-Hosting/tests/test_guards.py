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
        "| | A · ACA Sandboxes | B · Foundry hosted agent | C · AKS + agent-sandbox | D · BYO-AKS | E · E2B on Azure VM |",
        "| | A · ACA Sandboxes | C · AKS + agent-sandbox | D · BYO-AKS | E · E2B on Azure VM |",
        1,
    )
    assert any("A, B, C and D" in e for e in vd.check_four_path_tables(p))


def test_same_container_claim_fails_information_alignment():
    p = pages()
    p["README.md"] = p["README.md"].replace(
        "This repository applies one comparable stateful interaction contract to all five paths",
        "This repository puts the same sample agent on all four",
        1,
    )
    assert vd.check_information_alignment(p)


@pytest.mark.parametrize("name", ("README.md", "README_CN.md", "VALIDATION.md", "VALIDATION_CN.md"))
def test_b_workload_implementation_cannot_disappear(name):
    p = pages()
    p[name] = p[name].replace("ResponsesHostServer", "reflection service")
    assert any("distinct workload" in error for error in vd.check_information_alignment(p))


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


@pytest.mark.parametrize("name,lang", (("README.md", "en"), ("README_CN.md", "zh")))
def test_selectable_compute_drift_fails_information_alignment(name, lang):
    p = pages()
    claims = json.loads(vd.read("evidence/claims.json"))
    value = claims["paths"]["B"][f"selectable_compute_{lang}"]
    assert value in p[name]
    p[name] = p[name].replace(value, "Choose any Azure VM SKU", 1)
    assert any("selectable_compute" in error for error in vd.check_information_alignment(p))


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
    changed = block.replace("24.25", "__SWAP__", 1).replace("15.93", "24.25", 1).replace("__SWAP__", "15.93", 1)
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


@pytest.mark.parametrize("name", ("VALIDATION.md", "VALIDATION_CN.md"))
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


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
def test_deployment_steps_cannot_reintroduce_measurement_as_acceptance(name):
    p = pages()
    p[name] = p[name].replace('python tools/aca_sandbox.py ask', 'python tools/measure_aca.py ask', 1)
    assert any("measurement or evidence" in error for error in vd.check_commands(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
def test_c_deployment_requires_a_real_application_request(name):
    p = pages()
    p[name] = p[name].replace('"$C_AGENT_URL/reflect"', '"$C_AGENT_URL/ready"', 1)
    assert any("real invocation" in error for error in vd.check_commands(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
def test_e_deployment_requires_a_real_application_call(name):
    p = pages()
    p[name] = p[name].replace("python tools/e2b_agent.py", "python -c 'import e2b'", 1)
    assert any("real invocation" in error for error in vd.check_commands(p))


@pytest.mark.parametrize("name,old,new", (("README.md", "| GA (2026-06) |", "| Public preview |"),
                                          ("README_CN.md", "| GA（2026-06） |", "| 公开预览 |")))
def test_b_status_cannot_revert_to_preview(name, old, new):
    p = pages()
    assert old in p[name]
    p[name] = p[name].replace(old, new, 1)
    assert any("maturity row" in e for e in vd.check_information_alignment(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
@pytest.mark.parametrize("closing", ("set -euo pipefail\n", "exit 1\n"))
def test_steps_cannot_close_the_readers_shell(name, closing):
    p = pages()
    p[name] = p[name].replace("git clone --filter=blob:none", closing + "git clone --filter=blob:none", 1)
    assert any("interactive shell" in e for e in vd.check_commands(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
def test_steps_bash_blocks_parse(name):
    p = pages()
    assert vd.check_shell_syntax(p) == []
    guard = '  echo "C_ALLOW_CIDR is not set: put your public egress CIDR in .env, reload it and run this block again" >&2\n'
    assert guard in p[name]
    p[name] = p[name].replace(guard, "", 1)
    assert any(f"{name}: bash block" in e for e in vd.check_shell_syntax(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
def test_steps_install_into_a_virtual_environment(name):
    p = pages()
    p[name] = p[name].replace(". .venv/bin/activate\n", "", 1)
    assert any("virtual environment" in e for e in vd.check_commands(p))


@pytest.mark.parametrize("name", ("README.md", "README_CN.md"))
@pytest.mark.parametrize("fault", ("no-prompt", "shared-block"))
def test_resource_group_delete_stays_alone_and_confirmed(name, fault):
    p = pages()
    cmd = 'az group delete -n "$RESOURCE_GROUP" --no-wait'
    assert cmd in p[name]
    if fault == "no-prompt":
        p[name] = p[name].replace(cmd, 'az group delete -n "$RESOURCE_GROUP" --yes --no-wait', 1)
    else:
        p[name] = p[name].replace(cmd, 'az aks stop -g "$RESOURCE_GROUP" -n "$AKS_NAME"\n' + cmd, 1)
    assert any("az group delete" in e for e in vd.check_commands(p))


def test_private_content_fails(tmp_path):
    for leak in ("subscription 12345678-1234-1234-1234-123456789abc", "router at 20.242.33.10",
                 "mail someone@contoso.com", "cluster aks-byoaks", "sessions were not observed on the cluster",
                 "ran from /home/alice/work", "metadata at 169.254.169.253", "resource group e2b-embed-vm",
                 "source host-check-e2b-vm.txt", "evidence e2b-vm2-refusal.txt"):
        f = tmp_path / "leak.md"
        f.write_text(f"# x\n\n{leak}\n", encoding="utf-8")
        assert vd.check_private({}, [f]), leak


def test_clean_text_passes_private(tmp_path):
    f = tmp_path / "ok.md"
    f.write_text("# x\n\nPOST http://127.0.0.1:8080/reflect with 10.225.1.0/24 and user@example.com\n"
                 "token from http://169.254.169.254/metadata, file /home/user/scale.txt in the sandbox\n"
                 "E2B_VM_NAME=YOUR-E2B-VM-NAME\n", encoding="utf-8")
    assert vd.check_private({}, [f]) == []
