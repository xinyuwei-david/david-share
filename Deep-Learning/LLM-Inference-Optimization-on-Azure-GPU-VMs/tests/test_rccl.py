"""RCCL evidence contracts; offline checks do not execute a GPU benchmark."""
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rccl_log  # noqa: E402

EVIDENCE = ROOT / "evidence" / "rccl"


def sample(name):
    return (EVIDENCE / "raw" / f"{name}.txt").read_text(encoding="utf-8")


class RcclParseTests(unittest.TestCase):
    def test_all_committed_runs_round_trip(self):
        result = rccl_log.build_evidence(EVIDENCE)
        self.assertEqual(len(result["runs"]), 7)
        for record in result["runs"]:
            with self.subTest(run=record["id"]):
                text = sample(record["id"])
                self.assertEqual(rccl_log.parse(text), rccl_log.parse(rccl_log.project(text)))

    def test_headline_performance_is_not_a_correctness_check(self):
        run = rccl_log.parse(sample("container-performance"))
        self.assertEqual(run["data_check"], "NOT_RUN")
        self.assertEqual(run["parameters"]["iterations"], 50)
        self.assertEqual(run["parameters"]["warmup_iterations"], 20)
        self.assertEqual(run["ranks"], 16)
        self.assertEqual(run["average_busbw_gb_s"], 359.24)
        self.assertEqual(run["rows"][-1]["out_of_place"]["busbw_gb_s"], 364.28)
        self.assertIsNone(run["rows"][0]["out_of_place"]["wrong"])

    def test_check_run_preserves_out_of_place_and_in_place_order(self):
        run = rccl_log.parse(sample("container-correctness"))
        self.assertEqual(run["data_check"], "PASS")
        self.assertEqual(run["rows"][0]["out_of_place"]["busbw_gb_s"], 337.40)
        self.assertEqual(run["rows"][0]["in_place"]["busbw_gb_s"], 337.27)
        self.assertEqual(run["network_markers"], {"IB": 16, "Socket": 0, "GDRDMA": 472})
        self.assertEqual(run["parameters"]["iterations"], 20)

    def test_interleaved_info_does_not_lose_a_numeric_group(self):
        original = sample("container-correctness")
        lines = original.splitlines()
        index = next(i for i, line in enumerate(lines) if rccl_log.ROW_START.match(line))
        tokens = lines[index].split()
        logger = "node-a:1:1 [0] NCCL INFO diagnostic"
        lines[index] = (
            " ".join(tokens[:5]) + logger + "\n"
            + " ".join(tokens[5:9]) + logger + "\n"
            + " ".join(tokens[9:])
        )
        self.assertEqual(rccl_log.parse(logger + "\n" + "\n".join(lines)), rccl_log.parse(original))

    def test_numeric_container_hostname_does_not_extend_the_wrong_count(self):
        original = sample("container-correctness")
        lines = original.splitlines()
        index = next(i for i, line in enumerate(lines) if rccl_log.ROW_START.match(line))
        tokens = lines[index].split()
        logger = "01cafe:1:1 [0] NCCL INFO diagnostic"
        lines[index] = " ".join(tokens[:9]) + logger + "\n" + " ".join(tokens[9:])
        parsed = rccl_log.parse(logger + "\n" + "\n".join(lines))
        self.assertEqual(parsed, rccl_log.parse(original))
        self.assertEqual(parsed["rows"][0]["tokens"][8], "0")

    def test_prefixed_version_is_parsed_without_publishing_host_identity(self):
        text = sample("container-performance")
        text = text.replace("RCCL version :", "node-a:1:1 [0] NCCL INFO RCCL version :")
        result = rccl_log.project(text)
        self.assertNotIn("node-a", result)
        self.assertEqual(rccl_log.parse(result)["software"]["RCCL"], "2.27.7-HEAD:0d2c4fd")

    def test_missing_result_group_is_rejected(self):
        text = sample("container-correctness")
        line = next(line for line in text.splitlines() if rccl_log.ROW_START.match(line))
        broken = text.replace(line, " ".join(line.split()[:9]), 1)
        with self.assertRaisesRegex(rccl_log.RcclError, "result groups"):
            rccl_log.parse(broken)

    def test_missing_completion_and_late_failure_are_rejected(self):
        text = sample("container-performance")
        for broken in (text.replace(rccl_log.FINISHED, ""), text + "Segmentation fault\n"):
            with self.subTest(input=broken[-80:]):
                with self.assertRaises(rccl_log.RcclError):
                    rccl_log.parse(broken)

    def test_bad_data_check_is_rejected(self):
        text = sample("container-correctness")
        line = next(line for line in text.splitlines() if rccl_log.ROW_START.match(line))
        tokens = line.split()
        tokens[8] = "1"
        with self.assertRaisesRegex(rccl_log.RcclError, "wrong-count"):
            rccl_log.parse(text.replace(line, " ".join(tokens), 1))

    def test_tampered_average_and_missing_size_are_rejected(self):
        text = sample("host-performance")
        broken_average = text.replace("354.57", "454.57", 1)
        last = next(line for line in text.splitlines() if line.startswith("17179869184 "))
        for broken in (broken_average, text.replace(last + "\n", "")):
            with self.assertRaises(rccl_log.RcclError):
                rccl_log.parse(broken)

    def test_socket_control_does_not_claim_ib(self):
        socket = rccl_log.parse(sample("socket-fallback"))
        ib = rccl_log.parse(sample("ib-transport"))
        self.assertEqual(socket["parameters"], ib["parameters"])
        self.assertEqual(socket["data_check"], "PASS")
        self.assertEqual(socket["network_markers"]["Socket"], 16)
        self.assertEqual(socket["network_markers"]["IB"], 0)
        self.assertEqual(ib["network_markers"]["IB"], 16)
        self.assertEqual(ib["network_markers"]["Socket"], 0)
        self.assertEqual(len(socket["rows"]), 1)

    def test_software_versions_are_not_private_ipv4_addresses(self):
        self.assertIsNotNone(rccl_log.SOFTWARE.fullmatch("ROCm version : 6.4.4.0-129-e7d83f5"))
        self.assertIsNone(rccl_log.SOFTWARE.fullmatch("endpoint: 10.2.3.4"))
        for path in (EVIDENCE / "raw").glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            without_versions = "\n".join(line for line in text.splitlines() if not rccl_log.SOFTWARE.fullmatch(line))
            self.assertNotRegex(without_versions, r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
            self.assertNotRegex(text, r"/home/|/root/|[A-Z]:\\|[\w.-]+:\d+:\d+ \[\d+\]")


class RcclManifestTests(unittest.TestCase):
    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "rccl"
            shutil.copytree(EVIDENCE, copy)
            target = copy / "raw" / "host-performance.txt"
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(rccl_log.RcclError, "hash mismatch"):
                rccl_log.build_evidence(copy)

    def test_workload_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "rccl"
            shutil.copytree(EVIDENCE, copy)
            path = copy / "runs.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["runs"][0]["expected"]["ranks"] = 8
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(rccl_log.RcclError, "workload identity"):
                rccl_log.build_evidence(copy)

    def test_unlisted_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "rccl"
            shutil.copytree(EVIDENCE, copy)
            (copy / "raw" / "extra.txt").write_text("not a run", encoding="utf-8")
            with self.assertRaisesRegex(rccl_log.RcclError, "exact raw directory"):
                rccl_log.build_evidence(copy)

    def test_source_is_the_rocm_monorepo_not_the_retired_project(self):
        data = json.loads((EVIDENCE / "runs.json").read_text(encoding="utf-8"))
        source = data["protocol"]["container_source"]
        self.assertEqual(source["repository"], "https://github.com/ROCm/rocm-systems")
        self.assertEqual(source["subdirectory"], "projects/rccl-tests")
        self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
