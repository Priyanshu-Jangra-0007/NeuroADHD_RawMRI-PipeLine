"""Tests for deterministic, resumable synthetic cohort orchestration."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_synthetic_cohort import run_synthetic_cohort


def _write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def _fake_generator(work_dir: Path, **kwargs):
    subject = kwargs["subject_id"]
    session = kwargs["session_id"]
    root = work_dir / subject / session
    return {
        "bold_nii": _write(root / "bold.nii.gz", "synthetic-bold"),
        "bold_json": _write(root / "bold.json", '{"RepetitionTime": 2.0}'),
        "t1w_nii": _write(root / "t1w.nii.gz", "synthetic-t1w"),
        "mni_template": _write(work_dir / "MNI.nii.gz", "synthetic-template"),
        "cc200_atlas": _write(work_dir / "atlas.nii.gz", "synthetic-atlas"),
        "synthetic_manifest": _write(root / "synthetic_manifest.json", '{"synthetic": true}'),
    }


def _fake_pipeline(**kwargs):
    subject = kwargs["subject_id"]
    run_dir = Path(kwargs["output_dir"]) / subject / "run_test"
    result_file = _write(run_dir / "qc.txt", f"synthetic result for {subject}")
    digest = hashlib.sha256(result_file.read_bytes()).hexdigest()
    _write(
        run_dir / "output_manifest.json",
        json.dumps(
            {
                "status": "SUCCESS",
                "artifacts": {
                    "qc_report": {
                        "filename": "qc.txt",
                        "sha256": digest,
                    }
                },
            }
        ),
    )
    return {
        "status": "SUCCESS",
        "output_dir": run_dir,
        "runtime_seconds": 1.0,
        "peak_rss_gb": 0.25,
    }


class TestSyntheticCohortRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "cohort"

        self.gate_patcher = mock.patch(
            "scripts.run_synthetic_cohort.ResearchAuthorizationGate"
        )
        self.generator_patcher = mock.patch(
            "scripts.run_synthetic_cohort.generate_synthetic_phantom_session",
            side_effect=_fake_generator,
        )
        self.pipeline_patcher = mock.patch(
            "scripts.run_synthetic_cohort.run_poc_v2_pipeline",
            side_effect=_fake_pipeline,
        )
        self.gate_class = self.gate_patcher.start()
        self.generator = self.generator_patcher.start()
        self.pipeline = self.pipeline_patcher.start()
        self.addCleanup(self.gate_patcher.stop)
        self.addCleanup(self.generator_patcher.stop)
        self.addCleanup(self.pipeline_patcher.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runs_subjects_sequentially_and_writes_checkpoints(self):
        result = run_synthetic_cohort(self.output_dir, n_subjects=2, seed=100)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["summary"]["succeeded"], 2)
        self.assertEqual(self.generator.call_count, 2)
        self.assertEqual(self.pipeline.call_count, 2)

        manifest = json.loads((self.output_dir / "cohort_manifest.json").read_text())
        self.assertEqual(manifest["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertIs(manifest["real_data_authorized"], False)
        self.assertEqual(
            [item["subject_id"] for item in manifest["subjects"]],
            ["sub-synthetic001", "sub-synthetic002"],
        )
        self.assertEqual([item["seed"] for item in manifest["subjects"]], [100, 101])
        self.assertTrue(all(item["status"] == "SUCCESS" for item in manifest["subjects"]))
        self.assertTrue(all(item["artifact_count"] == 1 for item in manifest["subjects"]))

    def test_resume_skips_completed_subjects_after_checksum_verification(self):
        first = run_synthetic_cohort(self.output_dir, n_subjects=2)
        self.assertEqual(first["status"], "SUCCESS")
        self.assertEqual(self.pipeline.call_count, 2)

        second = run_synthetic_cohort(self.output_dir, n_subjects=2, resume=True)

        self.assertEqual(second["status"], "SUCCESS")
        self.assertEqual(self.pipeline.call_count, 2)
        self.assertEqual(self.generator.call_count, 2)

    def test_resume_reruns_subject_when_artifact_is_tampered(self):
        run_synthetic_cohort(self.output_dir, n_subjects=1)
        subject = json.loads((self.output_dir / "cohort_manifest.json").read_text())["subjects"][0]
        output_dir = Path(subject["run_output_dir"])
        (output_dir / "qc.txt").write_text("tampered", encoding="utf-8")

        result = run_synthetic_cohort(self.output_dir, n_subjects=1, resume=True)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.pipeline.call_count, 2)

    def test_resume_rejects_changed_configuration(self):
        run_synthetic_cohort(self.output_dir, n_subjects=1, seed=10)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            run_synthetic_cohort(self.output_dir, n_subjects=1, seed=11, resume=True)

    def test_pipeline_failure_is_recorded_and_does_not_claim_success(self):
        self.pipeline.side_effect = RuntimeError("synthetic test failure")
        result = run_synthetic_cohort(self.output_dir, n_subjects=2)

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["summary"]["failed"], 1)
        self.assertEqual(result["summary"]["pending"], 1)
        manifest = json.loads((self.output_dir / "cohort_manifest.json").read_text())
        self.assertEqual(manifest["subjects"][0]["status"], "FAILED")
        self.assertEqual(manifest["subjects"][1]["status"], "PENDING")

    def test_rejects_oversized_cohort_and_memory_limit(self):
        with self.assertRaisesRegex(ValueError, "n_subjects"):
            run_synthetic_cohort(self.output_dir, n_subjects=21)
        with self.assertRaisesRegex(ValueError, "max_memory_gb"):
            run_synthetic_cohort(self.output_dir, n_subjects=1, max_memory_gb=3.0)

    def test_gate_is_checked_before_cohort_execution(self):
        result = run_synthetic_cohort(self.output_dir, n_subjects=1)
        gate = self.gate_class.return_value
        gate.check_execution_allowed.assert_called_once_with(
            requested_mode="synthetic", allow_real_scans=False
        )
        self.assertEqual(result["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
