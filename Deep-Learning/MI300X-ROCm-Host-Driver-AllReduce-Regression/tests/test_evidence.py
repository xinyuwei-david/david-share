import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_case", ROOT / "tools/verify_case.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the evidence validator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / "evidence/before/single-run-1.json"
        self.data = json.loads(path.read_text())
        self.log = path.with_suffix(".txt").read_text()

    def test_redacted_real_measurements_are_valid(self) -> None:
        module.validate_output(self.data, self.log, "single", 22203)

    def test_wrong_library_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["rccl_version"] = 22707
        with self.assertRaisesRegex(ValueError, "RCCL"):
            module.validate_output(data, self.log, "single", 22203)

    def test_nonzero_wrong_count_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["results"][0]["out_of_place"]["nwrong"] = 1
        with self.assertRaisesRegex(ValueError, "#wrong"):
            module.validate_output(data, self.log, "single", 22203)

    def test_normalized_units_are_checked(self) -> None:
        data = copy.deepcopy(self.data)
        data["results"][0]["out_of_place"]["bus_bw"] *= 8
        with self.assertRaisesRegex(ValueError, "normalization"):
            module.validate_output(data, self.log, "single", 22203)

    def test_native_comparison_has_expected_coverage(self) -> None:
        rows = module.compute_comparison()
        self.assertEqual(len(rows), 44)
        for row in rows:
            if row["size_bytes"] == 1 << 30 and row["phase"] == "out_of_place":
                self.assertGreater(row["before_bus_bw_gb_s"], 300)
                self.assertLess(row["after_bus_bw_gb_s"], 50)
                self.assertLess(row["change_percent"], -80)

    def test_generated_table_has_single_rounding(self) -> None:
        rows = module.compute_comparison()
        table = module.render_table(rows, False)
        self.assertIn("314.60 | 41.78 | 86.72%", table)
        self.assertIn("314.81 | 41.74 | 86.74%", table)

    def test_configurations_explicitly_describe_vm_and_container(self) -> None:
        summary = json.loads((ROOT / "data/native-summary.json").read_text())
        table = module.render_configurations(summary, True)
        self.assertEqual(table.count("**没有容器**"), 2)
        case2 = next(line for line in table.splitlines() if line.startswith("| 2 |"))
        self.assertIn("ROCm `6.4.4` / RCCL `2.22.3`", case2)
        self.assertIn("ROCm `7.2.4` / RCCL `2.27.7`", case2)
        self.assertIn("`6.12.12`（共享 VM 驱动）", case2)
        self.assertIn("321.67", case2)
        case4 = next(line for line in table.splitlines() if line.startswith("| 4 |"))
        self.assertIn("`6.16.13`（共享 VM 驱动）", case4)
        self.assertIn("单轮额外诊断", case4)
        self.assertIn("37.44", case4)

    def test_english_configurations_have_explicit_no_container(self) -> None:
        summary = json.loads((ROOT / "data/native-summary.json").read_text())
        table = module.render_configurations(summary, False)
        self.assertEqual(table.count("**No container**"), 2)
        self.assertIn("shared VM driver", table)

    def test_empty_generated_block_can_be_populated(self) -> None:
        text = "<!-- BEGIN GENERATED: native-results -->\n<!-- END GENERATED: native-results -->"
        updated = module.replace_generated_block(text, "native-results", "table")
        self.assertIn("-->\ntable\n<!-- END", updated)

    def test_generated_block_replacement_is_idempotent(self) -> None:
        text = "<!-- BEGIN GENERATED: native-results -->\nold table\n<!-- END GENERATED: native-results -->"
        updated = module.replace_generated_block(text, "native-results", "new table")
        self.assertEqual(updated, module.replace_generated_block(updated, "native-results", "new table"))

    def test_evidence_manifest_checks_real_published_files(self) -> None:
        module.validate_evidence_manifest()

    def test_evidence_manifest_rejects_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence").mkdir()
            (root / "data").mkdir()
            names = ("evidence/run.txt", "data/native-summary.json",
                     "data/kernel-symptom-counts.json", "data/comparison.csv")
            for name in names:
                (root / name).write_text("original\n", encoding="utf-8")
            manifest = {
                "hash_algorithm": "sha256",
                "normalization": "UTF-8 text with LF line endings",
                "files": {name: module.evidence_digest(root / name) for name in names},
            }
            (root / "data/evidence-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            module.validate_evidence_manifest(root)
            (root / "evidence/run.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                module.validate_evidence_manifest(root)

    def test_redaction_gate_catches_unknown_ip_and_identifier(self) -> None:
        uuid = "12345678" + "-1234" * 3 + "-123456789abc"
        self.assertIn("UUID", module.private_identifier_findings(uuid))
        self.assertIn("IPv4 address", module.private_identifier_findings("address=" + "192.0." + "2.15"))

    def test_version_strings_are_not_misidentified_as_ip(self) -> None:
        self.assertEqual(module.private_identifier_findings("ROCm 7.2.4.0-93; installer 30.30.4.0.30300400"), [])

    def test_public_readme_links_and_three_section_contract(self) -> None:
        module.validate_publication()

    def test_bilingual_version_facts_are_aligned(self) -> None:
        facts = ("2.22.3", "2.26.6", "2.27.3", "2.27.7", "22703", "6.12.12", "6.16.13",
                 "amdgpu/6.4.4", "6.4.x", "30.30.x", "6.2.x", "7.0.x", "7.2.x", "nvidia-nccl-cu12",
                 "c1faf5053567d0fb97ba5825f51bfd6874ac9527", "AITER")
        for name in ("README.md", "README_CN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            for fact in facts:
                self.assertIn(fact, text, f"{name} lacks {fact}")

    def test_container_build_is_pinned_and_does_not_install_host_driver(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile.rccl-tests").read_text(encoding="utf-8")
        self.assertIn("7.2.4-complete@sha256:92f309", dockerfile)
        self.assertIn(module.COMMIT, dockerfile)
        self.assertIn("MPI=0 GPU_TARGETS=gfx942", dockerfile)
        self.assertNotIn("amdgpu-dkms", dockerfile)
        wrapper = (ROOT / "scripts/run_container.sh").read_text(encoding="utf-8")
        self.assertIn('shared_vm_loaded_amdgpu_version', wrapper)
        self.assertIn('version.value != 22707', wrapper)
        self.assertNotIn('reboot', wrapper)


if __name__ == "__main__":
    unittest.main()
