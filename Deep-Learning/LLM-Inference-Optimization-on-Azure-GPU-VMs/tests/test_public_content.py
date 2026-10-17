"""Tests for the rendered READMEs, diagrams and public-content guards."""
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_readme  # noqa: E402
import check_repo  # noqa: E402
import draw_diagrams  # noqa: E402
import render_launch  # noqa: E402


class ReadmeTests(unittest.TestCase):
    def test_committed_readmes_are_refresh_clean(self):
        for document, paths in build_readme.DOCUMENTS.items():
            for lang, path in paths.items():
                text = path.read_text(encoding="utf-8")
                self.assertEqual(build_readme.render(text, lang, document), text, path.name)

    def test_audit_passes(self):
        self.assertEqual(check_repo.run(), [])

    def test_each_measurement_section_states_its_input_before_its_table(self):
        text = build_readme.VALIDATIONS["en"].read_text(encoding="utf-8")
        section = text.split("### Measured Results on MI300X", 1)[1].split("\n### ", 1)[0]
        for sub in re.split(r"\n#### ", section)[1:5]:
            with self.subTest(section=sub.splitlines()[0]):
                first_table = sub.find("\n| ")
                self.assertGreater(first_table, 0)
                self.assertIn("**Input.**", sub[:first_table])
                self.assertIn("**Boundary.**", sub)

    def test_every_technique_has_a_card_under_its_own_heading(self):
        import json
        cat = json.loads((ROOT / "profiles" / "techniques.json").read_text(encoding="utf-8"))
        for lang, path in build_readme.VALIDATIONS.items():
            text = path.read_text(encoding="utf-8")
            for t in cat["techniques"]:
                with self.subTest(lang=lang, technique=t["id"]):
                    self.assertRegex(text, rf"(?m)^#{{3,6}} {re.escape(t[lang])}\n\n<!-- BEGIN GENERATED: card-{t['id']} -->")

    def test_images_match_ledger(self):
        self.assertEqual(draw_diagrams.main(["--check"]), 0)

    def test_retired_or_private_content_is_rejected(self):
        """Break one guarded fact and confirm the audit turns red."""
        probes = {
            "README.md": "\nThroughput versus H200 was 70%.\n",
            "README_CN.md": "\nPrivateCustomerProbe\n",
        }
        for name, injected in probes.items():
            with self.subTest(file=name):
                with tempfile.TemporaryDirectory() as tmp:
                    copy = Path(tmp) / ROOT.name
                    shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
                    target = copy / name
                    target.write_text(target.read_text(encoding="utf-8") + injected, encoding="utf-8")
                    original_root = check_repo.ROOT
                    try:
                        check_repo.ROOT = copy
                        check_repo.READMES = {"en": copy / "README.md", "cn": copy / "README_CN.md"}
                        errors = check_repo.run()
                    finally:
                        check_repo.ROOT = original_root
                        check_repo.READMES = {"en": original_root / "README.md", "cn": original_root / "README_CN.md"}
                    self.assertTrue(any(e.startswith("FORBIDDEN") for e in errors), errors)

    def test_each_forbidden_category_is_detected(self):
        samples = ["versus A100", "on MI350X", "on MI355X", "SWE-bench Verified", "swebench run", "PrivateCustomerProbe", "C:\\Users\\someone\\", "person@example.com", "10.2.3.4",
                   "12345678-1234-1234-1234-123456789abc", "<details>",
                   "run of 2026-05-08", "stage-20260713-tuned", "from May to July", "5 月 → 7 月"]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(any(p.search(sample) for p, _ in check_repo.FORBIDDEN)
                                or check_repo.private_name_present(sample), sample)

    def test_generated_tables_keep_a_constant_column_count(self):
        for paths in build_readme.DOCUMENTS.values():
            for lang, path in paths.items():
                errors = []
                check_repo.check_table_shape(lang, path.read_text(encoding="utf-8"), errors)
                self.assertEqual(errors, [])

    def test_accuracy_section_lists_every_lossy_switch_in_both_languages(self):
        for lang, path in build_readme.VALIDATIONS.items():
            text = path.read_text(encoding="utf-8")
            block = text.split("<!-- BEGIN GENERATED: precision-summary -->", 1)[1].split("<!-- END GENERATED: precision-summary -->", 1)[0]
            lossy = [l for l in block.splitlines() if l.startswith("- **")][0]
            block = lossy
            for needle in ("FP8", "INT8 Quick Reduce", "router"):
                self.assertIn(needle, block, (lang, needle))
            self.assertIn("ROCM_QUICK_REDUCE_QUANTIZATION=INT8", text, lang)

    def test_tables_wider_than_four_columns_are_rejected(self):
        wide = "| a | b | c | d | e |\n|---|---|---|---|---|\n| 1 | 2 | 3 | 4 | 5 |\n"
        errors = []
        check_repo.check_table_shape("en", wide, errors)
        self.assertTrue(any(e.startswith("en: TABLE_TOO_WIDE") for e in errors), errors)

    def test_bilingual_number_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / ROOT.name
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
            cn = copy / "README_CN.md"
            cn.write_text(cn.read_text(encoding="utf-8").replace("359.24", "459.24", 1), encoding="utf-8")
            original_root = check_repo.ROOT
            try:
                check_repo.ROOT = copy
                check_repo.READMES = {"en": copy / "README.md", "cn": cn}
                errors = check_repo.run()
            finally:
                check_repo.ROOT = original_root
                check_repo.READMES = {"en": original_root / "README.md", "cn": original_root / "README_CN.md"}
            self.assertTrue(any(e.startswith("BILINGUAL_NUMBERS") for e in errors), errors)

    def test_operator_documents_have_only_the_three_reader_sections(self):
        for lang, path in build_readme.READMES.items():
            text = path.read_text(encoding="utf-8")
            errors = []
            check_repo.check_order(lang, text, errors)
            check_repo.check_operator_path(lang, text, errors)
            self.assertEqual(errors, [])
            retired = text + "\n## Start Here\n"
            check_repo.check_order(lang, retired, errors)
            self.assertTrue(errors)

    def test_missing_operator_command_fails_the_contract(self):
        text = build_readme.READMES["en"].read_text(encoding="utf-8")
        errors = []
        check_repo.check_operator_path("en", text.replace("wait-ready.sh", "missing.sh"), errors)
        self.assertIn("en: MISSING_OPERATOR_STEP wait-ready.sh", errors)

    def test_rccl_run_sections_have_input_variables_result_and_boundary_in_order(self):
        labels = {
            "en": ("**Question.**", "**Recorded input.**", "**Variables.**", "**Observed result.**", "**Boundary.**"),
            "cn": ("**问题。**", "**记录的输入。**", "**变量。**", "**观测结果。**", "**边界。**"),
        }
        for lang, path in build_readme.VALIDATIONS.items():
            text = path.read_text(encoding="utf-8")
            block = text.split("<!-- BEGIN GENERATED: rccl-runs -->", 1)[1].split("<!-- END GENERATED: rccl-runs -->", 1)[0]
            sections = re.split(r"\n#### ", block)[1:]
            self.assertEqual(len(sections), 7)
            for section in sections:
                positions = [section.find(label) for label in labels[lang]]
                self.assertTrue(all(p >= 0 for p in positions))
                self.assertEqual(positions, sorted(positions))
                self.assertLess(positions[2], section.find("\n| "))

    def test_cross_document_anchor_is_checked(self):
        errors = []
        check_repo.check_links("probe", "[review](VALIDATION.md#not-a-real-heading)", errors)
        self.assertEqual(errors, ["probe: BROKEN_FILE_ANCHOR VALIDATION.md#not-a-real-heading"])

    def test_software_version_exception_does_not_hide_a_real_address(self):
        text = 'ROCm version : 6.4.4.0-129-e7d83f5\nendpoint=10.2.3.4\n'
        cleaned = check_repo.SOFTWARE_DECLARATION.sub(r"\1<software-version>", text)
        self.assertNotIn("6.4.4.0", cleaned)
        self.assertIn("10.2.3.4", cleaned)

    def test_private_name_rules_contain_only_hashes(self):
        self.assertTrue(check_repo.private_name_present("PrivateCustomerProbe"))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value)
                            for value in check_repo.PRIVATE_FOLDED_HASHES | check_repo.PRIVATE_CASED_HASHES))

    def test_concurrency_ladder_is_not_a_synthetic_output_mode(self):
        import json
        data = json.loads((ROOT / "profiles/techniques.json").read_text(encoding="utf-8"))
        ladder = next(t for t in data["techniques"] if t["id"] == "concurrency-ladder")
        self.assertEqual(ladder["precision"]["class"], "same-math")
        for lang in ("en", "cn"):
            block = build_readme.precision_summary(lang)
            measurement_line = block.splitlines()[-1]
            self.assertNotIn(ladder[lang], measurement_line)

    def test_quick_reduce_example_changes_the_generated_export(self):
        arm = render_launch.render(render_launch.load_profile("rocm-mi300x-single"), "server", [])
        before = "export ROCM_QUICK_REDUCE_QUANTIZATION=NONE"
        after = "export ROCM_QUICK_REDUCE_QUANTIZATION=INT8"
        self.assertEqual(arm.splitlines().count(before), 1)
        changed = arm.replace(before, after)
        self.assertIn(after, changed)
        self.assertNotIn(before, changed)
        for path in build_readme.VALIDATIONS.values():
            text = path.read_text(encoding="utf-8")
            self.assertIn("cp arm_a.sh arm_qr_int8.sh", text)
            self.assertIn("s/^" + before + "$/" + after + "/", text)
            self.assertIn("/proc/1/environ", text)

    def test_rccl_boundaries_follow_the_observed_transport(self):
        block = build_readme.rccl_runs("en")
        sections = re.split(r"\n#### ", "\n" + block)[1:]
        by_name = {s.splitlines()[0]: s for s in sections}
        ib = by_name["IB device mapped: IB transport"]
        socket = by_name["No IB device: Socket fallback"]
        self.assertIn("does RCCL choose IB and pass its data check", ib)
        self.assertNotIn("VERSION-only", ib)
        self.assertNotIn("VERSION-only", socket)
        self.assertIn("positively records IB", ib)
        self.assertIn("records Socket fallback", socket)
        self.assertIn("VERSION-only", by_name["Host AllReduce"])

    def test_model_evaluation_uses_the_pinned_container_interpreter(self):
        for path in build_readme.VALIDATIONS.values():
            text = path.read_text(encoding="utf-8")
            lines = [line for line in text.splitlines() if "-m sglang.test.run_eval" in line]
            self.assertEqual(len(lines), 3)
            self.assertTrue(all(line.startswith('docker exec "$SERVER_CONTAINER" /opt/venv/bin/python')
                                for line in lines))
            self.assertNotIn("The FP8 KV arm is not strictly single-variable", text)


if __name__ == "__main__":
    unittest.main()
