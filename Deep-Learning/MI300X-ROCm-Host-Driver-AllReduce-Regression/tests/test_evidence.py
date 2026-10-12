import copy
import importlib.util
import json
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

    def test_empty_generated_block_can_be_populated(self) -> None:
        text = "<!-- BEGIN GENERATED: native-results -->\n<!-- END GENERATED: native-results -->"
        updated = module.replace_generated_block(text, "native-results", "table")
        self.assertIn("-->\ntable\n<!-- END", updated)

    def test_generated_block_replacement_is_idempotent(self) -> None:
        text = "<!-- BEGIN GENERATED: native-results -->\nold table\n<!-- END GENERATED: native-results -->"
        updated = module.replace_generated_block(text, "native-results", "new table")
        self.assertEqual(updated, module.replace_generated_block(updated, "native-results", "new table"))


if __name__ == "__main__":
    unittest.main()
