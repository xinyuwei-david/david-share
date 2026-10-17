"""Static contracts and Linux command-wiring tests; no GPU or real SSH is used."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RcclBuildContractTests(unittest.TestCase):
    def test_recipe_uses_the_recorded_public_source_and_digest(self):
        source = (ROOT / "docker/rccl.Dockerfile").read_text(encoding="utf-8")
        recorded = json.loads((ROOT / "evidence/rccl/runs.json").read_text(encoding="utf-8"))["protocol"]
        self.assertIn(recorded["container_base"], source)
        self.assertIn(recorded["container_source"]["commit"], source)
        self.assertIn(recorded["container_source"]["repository"] + ".git", source)
        self.assertIn(recorded["container_source"]["subdirectory"], source)
        self.assertIn("/usr/share/doc/rccl-tests/LICENSE.txt", source)
        self.assertIn("ARG PIN_RCCL_TESTS_COMMIT=", source)
        self.assertNotIn("COPY /", source)
        self.assertNotIn(":latest", source)
        receipt = json.loads((ROOT / "evidence/rccl/build-status.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["image_build_status"], "NOT_BUILT")
        self.assertEqual(receipt["fresh_gpu_replay_status"], "NOT_RUN")

    def test_ssh_and_output_fail_closed(self):
        launcher = (ROOT / "docker/run-rccl.sh").read_text(encoding="utf-8")
        entrypoint = (ROOT / "docker/rccl-entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn("StrictHostKeyChecking=yes", launcher)
        self.assertIn("StrictHostKeyChecking yes", entrypoint)
        self.assertIn("refusing to overwrite unrelated", launcher)
        self.assertIn('exec /usr/sbin/sshd -D', entrypoint)
        self.assertIn("chmod 600", launcher)
        self.assertIn("--require-ib", launcher)
        self.assertIn("output path already exists", launcher)
        self.assertNotIn("az vm", launcher)
        self.assertNotIn("pkill", launcher)

    def test_build_context_excludes_settings_and_keys(self):
        text = (ROOT / "docker/.dockerignore").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines(), ["*", "!Dockerfile", "!rccl.Dockerfile", "!rccl-entrypoint.sh"])
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("docker/rccl.env", ignored)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Linux/macOS command wiring, not a live deployment")
class RcclShellTests(unittest.TestCase):
    def test_shell_files_parse(self):
        for name in ("run-rccl.sh", "rccl-entrypoint.sh"):
            done = subprocess.run(["bash", "-n", str(ROOT / "docker" / name)],
                                  capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(done.returncode, 0, done.stderr)

    def test_minimal_start_arguments_and_output_override(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            keydir = temp / "rccl-ssh"
            keydir.mkdir()
            for name in ("id_ed25519", "authorized_keys", "known_hosts", "ssh_host_ed25519_key"):
                (keydir / name).write_text("synthetic test input\n", encoding="utf-8")
            config = temp / "rccl.env"
            config.write_text(
                f"NODE_A=node-a\nNODE_B=node-b\nRCCL_IMAGE=registry.example/rccl@sha256:{'1' * 64}\n"
                f"RCCL_SSH_DIR={keydir}\nRCCL_OUTPUT_DIR=\n", encoding="utf-8"
            )
            fake = temp / "docker"
            fake.write_text(
                '#!/usr/bin/env bash\n'
                'if [ "$1" = run ]; then printf "%s\\n" "$@" > "$ARGV_OUT"; exit 0; fi\n'
                'if [ "$1" = exec ] && [ "$3" = mpirun ]; then cat "$FIXTURE"; fi\n',
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f"{temp}:{os.environ['PATH']}", RCCL_ENV_FILE=str(config),
                       ARGV_OUT=str(temp / "argv"), FIXTURE=str(ROOT / "evidence/rccl/raw/container-performance.txt"))
            env.pop("RCCL_OUTPUT_DIR", None)
            command = ["bash", str(ROOT / "docker/run-rccl.sh")]
            done = subprocess.run(command + ["start"], env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(done.returncode, 0, done.stderr)
            argv = (temp / "argv").read_text(encoding="utf-8").splitlines()
            for item in ("/dev/kfd", "/dev/dri", "/dev/infiniband", "memlock=-1:-1"):
                self.assertIn(item, argv)
            self.assertNotIn("--privileged", argv)
            out = temp / "new-run"
            env["RCCL_OUTPUT_DIR"] = str(out)
            done = subprocess.run(command + ["container"], env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(done.returncode, 0, done.stderr)
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["data_check"], "NOT_RUN")
            done = subprocess.run(command + ["container"], env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(done.returncode, 2)
            self.assertIn("already exists", done.stderr)
            env["RCCL_OUTPUT_DIR"] = str(temp / "bad-transport")
            env["FIXTURE"] = str(ROOT / "evidence/rccl/raw/socket-fallback.txt")
            done = subprocess.run(command + ["check"], env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("does not prove IB", done.stderr)


if __name__ == "__main__":
    unittest.main()
