"""Integration tests for container model integration and deployment verification.

Validates:
1. Docker container read-only volume mounting (:ro) prevents file modification.
2. Fail-closed safety behavior when --require-model-inference is passed without model mount.
3. Fail-closed behavior on corrupted or symlinked model artifacts inside container.
4. Host SHA-256 digests remain bitwise unchanged after container inference.
5. End-to-end container execution produces valid model_prediction.json adhering to frozen schema.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.integration.model_verifier import (
    FROZEN_CLASSIFIER_SHA256,
    FROZEN_PREPROCESSING_PIPELINE_SHA256,
    compute_file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKER_IMAGE = "neuro_adhd_pipeline:v2"
TEST_HMAC_KEY = "test_synthetic_integrity_key_container_model_phase_a42"


def is_docker_available() -> bool:
    """Check if Docker CLI is installed, daemon is responding, and pipeline image is built."""
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if res.returncode != 0:
            return False
        img_res = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return img_res.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_authentic_model_dir() -> Path | None:
    """Retrieve and validate authentic NEUROADHD_MODEL_DIR if configured."""
    env_dir = os.environ.get("NEUROADHD_MODEL_DIR")
    if not env_dir:
        return None
    p = Path(env_dir).resolve()
    pipe_file = p / "final_preprocessing_pipeline.joblib"
    model_file = p / "final_model.joblib"
    if pipe_file.is_file() and model_file.is_file():
        return p
    return None


class TestContainerModelIntegration(unittest.TestCase):
    """Container model integration and read-only volume enforcement test suite."""

    def setUp(self):
        if not is_docker_available():
            self.skipTest("Docker daemon is not running or 'neuro_adhd_pipeline:v2' image is not built.")

    def test_container_model_mount_read_only_enforcement(self):
        """Task 4: Verify Docker container cannot modify model artifacts when mounted :ro."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tmp.chmod(0o777)
            dummy_pipe = tmp / "final_preprocessing_pipeline.joblib"
            dummy_model = tmp / "final_model.joblib"
            dummy_pipe.write_bytes(b"pipeline_bytes")
            dummy_model.write_bytes(b"model_bytes")
            dummy_pipe.chmod(0o666)
            dummy_model.chmod(0o666)

            # 1. Attempt to create a new file inside read-only mount (using root to test VFS :ro enforcement)
            cmd_touch = [
                "docker", "run", "--rm",
                "--user", "root",
                "--entrypoint", "/bin/sh",
                "-v", f"{tmp}:/opt/neuro_adhd/models:ro",
                DOCKER_IMAGE,
                "-c", "touch /opt/neuro_adhd/models/unauthorized.tmp",
            ]
            res_touch = subprocess.run(cmd_touch, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(res_touch.returncode, 0, "Container should not be able to write to :ro mount.")
            self.assertIn("Read-only file system", res_touch.stderr)

            # 2. Attempt to append/overwrite existing file inside read-only mount
            cmd_overwrite = [
                "docker", "run", "--rm",
                "--user", "root",
                "--entrypoint", "/bin/sh",
                "-v", f"{tmp}:/opt/neuro_adhd/models:ro",
                DOCKER_IMAGE,
                "-c", "echo 'tamper' >> /opt/neuro_adhd/models/final_model.joblib",
            ]
            res_overwrite = subprocess.run(cmd_overwrite, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(res_overwrite.returncode, 0, "Container should not be able to modify file in :ro mount.")
            self.assertIn("Read-only file system", res_overwrite.stderr)

            # 3. Attempt to delete existing file inside read-only mount
            cmd_rm = [
                "docker", "run", "--rm",
                "--user", "root",
                "--entrypoint", "/bin/sh",
                "-v", f"{tmp}:/opt/neuro_adhd/models:ro",
                DOCKER_IMAGE,
                "-c", "rm /opt/neuro_adhd/models/final_model.joblib",
            ]
            res_rm = subprocess.run(cmd_rm, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(res_rm.returncode, 0, "Container should not be able to delete file in :ro mount.")
            self.assertIn("Read-only file system", res_rm.stderr)

    def test_container_require_model_inference_fails_closed_without_mount(self):
        """Task 4: Running container with --require-model-inference without model mount fails closed."""
        cmd = [
            "docker", "run", "--rm",
            "-e", f"NEUROADHD_HMAC_KEY={TEST_HMAC_KEY}",
            DOCKER_IMAGE,
            "--mode", "synthetic",
            "--require-model-inference",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = res.stderr + res.stdout
        self.assertNotEqual(res.returncode, 0, "Container must fail closed when --require-model-inference is unfulfilled.")
        self.assertIn("SAFETY VIOLATION", output)
        self.assertIn("NEUROADHD_MODEL_DIR", output)

    def test_container_fails_closed_on_corrupted_model_artifact(self):
        """Task 4: Corrupted model artifact mounted into container triggers fail-closed error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tmp.chmod(0o777)
            corrupt_pipe = tmp / "final_preprocessing_pipeline.joblib"
            corrupt_model = tmp / "final_model.joblib"
            corrupt_pipe.write_bytes(b"corrupted_pipeline_payload")
            corrupt_model.write_bytes(b"corrupted_model_payload")
            corrupt_pipe.chmod(0o666)
            corrupt_model.chmod(0o666)

            cmd = [
                "docker", "run", "--rm",
                "-e", f"NEUROADHD_HMAC_KEY={TEST_HMAC_KEY}",
                "-e", "NEUROADHD_MODEL_DIR=/opt/neuro_adhd/models",
                "-v", f"{tmp}:/opt/neuro_adhd/models:ro",
                DOCKER_IMAGE,
                "--mode", "synthetic",
                "--model-dir", "/opt/neuro_adhd/models",
                "--require-model-inference",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(res.returncode, 0, "Container must fail closed on corrupted model artifacts.")
            output = res.stderr + res.stdout
            self.assertIn("SAFETY VIOLATION", output)
            self.assertTrue("SHA-256 mismatch" in output or "failed verification" in output)

    def test_container_fails_closed_on_symlinked_artifact(self):
        """Task 4: Symlink substitution inside mounted model directory triggers fail-closed error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tmp.chmod(0o777)
            real_file = tmp / "real.bin"
            real_file.write_bytes(b"some_bytes")
            real_file.chmod(0o666)
            sym_pipe = tmp / "final_preprocessing_pipeline.joblib"
            sym_pipe.symlink_to("real.bin")
            real_model = tmp / "final_model.joblib"
            real_model.write_bytes(b"model_bytes")
            real_model.chmod(0o666)

            cmd = [
                "docker", "run", "--rm",
                "-e", f"NEUROADHD_HMAC_KEY={TEST_HMAC_KEY}",
                "-e", "NEUROADHD_MODEL_DIR=/opt/neuro_adhd/models",
                "-v", f"{tmp}:/opt/neuro_adhd/models:ro",
                DOCKER_IMAGE,
                "--mode", "synthetic",
                "--model-dir", "/opt/neuro_adhd/models",
                "--require-model-inference",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(res.returncode, 0, "Container must fail closed on symlinked artifact.")
            output = res.stderr + res.stdout
            self.assertIn("SAFETY VIOLATION", output)
            self.assertTrue("Symlink substitution detected" in output or "failed verification" in output)

    def test_container_full_synthetic_inference_leaves_host_checksums_identical(self):
        """Task 3 & 4: Full container inference with authentic artifacts leaves host files bitwise identical."""
        model_dir = get_authentic_model_dir()
        if model_dir is None:
            self.skipTest(
                "NEUROADHD_MODEL_DIR is not set or authentic artifacts are unavailable. "
                "Full container model integration test was skipped."
            )

        pipe_file = model_dir / "final_preprocessing_pipeline.joblib"
        model_file = model_dir / "final_model.joblib"

        # Record pre-execution checksums
        pipe_sha_before = compute_file_sha256(pipe_file)
        model_sha_before = compute_file_sha256(model_file)
        self.assertEqual(pipe_sha_before, FROZEN_PREPROCESSING_PIPELINE_SHA256)
        self.assertEqual(model_sha_before, FROZEN_CLASSIFIER_SHA256)

        with tempfile.TemporaryDirectory() as out_tmpdir:
            out_path = Path(out_tmpdir) / "test_docker_out"
            out_path.mkdir(parents=True, exist_ok=True)

            # Execute container runner via docker_run_synthetic.sh
            env = os.environ.copy()
            env["NEUROADHD_HMAC_KEY"] = TEST_HMAC_KEY
            env["NEUROADHD_MODEL_DIR"] = str(model_dir)

            script_path = REPO_ROOT / "scripts" / "docker_run_synthetic.sh"
            cmd = [
                str(script_path),
                str(out_path),
                "--require-model-inference",
            ]
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(
                res.returncode, 0,
                f"Container execution failed (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}",
            )

            # Verify post-execution checksums on host - bitwise immutability
            pipe_sha_after = compute_file_sha256(pipe_file)
            model_sha_after = compute_file_sha256(model_file)
            self.assertEqual(pipe_sha_after, pipe_sha_before, "Host pipeline artifact was modified during container run!")
            self.assertEqual(model_sha_after, model_sha_before, "Host model artifact was modified during container run!")

            # Verify model_prediction.json output
            pred_files = list(out_path.glob("**/model_prediction.json"))
            self.assertEqual(len(pred_files), 1, "model_prediction.json was not generated.")
            pred_json_path = pred_files[0]
            with open(pred_json_path, encoding="utf-8") as f:
                pred_data = json.load(f)

            # Validate prediction payload schema
            self.assertEqual(pred_data.get("status"), "SUCCESS")
            self.assertIn(pred_data.get("predicted_class"), [0, 1])
            self.assertIn(pred_data.get("class_label"), ["Typical Control Pattern", "ADHD Pattern"])
            self.assertTrue(0.0 <= pred_data.get("adhd_probability", -1.0) <= 1.0)
            self.assertTrue(0.0 <= pred_data.get("control_probability", -1.0) <= 1.0)
            prob_sum = pred_data.get("adhd_probability", 0) + pred_data.get("control_probability", 0)
            self.assertAlmostEqual(prob_sum, 1.0, places=3)
            self.assertEqual(pred_data.get("features_in"), 19900)
            self.assertEqual(pred_data.get("features_selected"), 100)
            self.assertIs(pred_data.get("refit_performed"), False)

            # Verify 19,900 Fisher-z connectivity vector was also generated
            feat_files = list(out_path.glob("**/features_19900_fisher_z.npy"))
            self.assertEqual(len(feat_files), 1, "features_19900_fisher_z.npy missing.")


if __name__ == "__main__":
    unittest.main()
