"""Tests for safe dashboard reads over synthetic pipeline outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.dashboard.service import DashboardDataError, DashboardDataService


class TestDashboardDataService(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "cohorts"
        self.cohort_id = "cohort_20261103T123000_123456Z"
        self.subject_id = "sub-synthetic001"
        self.cohort_dir = self.root / self.cohort_id
        self.run_dir = self.cohort_dir / "runs" / self.subject_id / "run_001"
        self.run_dir.mkdir(parents=True)
        self.service = DashboardDataService(self.root)
        self._build_outputs()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _build_outputs(self):
        (self.run_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "validation_gates": {
                        "gate1_resources": {"status": "PASSED", "peak_rss_gb": 0.2},
                        "gate2_bids_input": {"status": "PASSED", "tr_seconds": 2.0, "t_vol": 3},
                        "gate5_roi_coverage": {"status": "PASSED", "n_rois": 200, "zero_variance_rois": 0},
                        "gate6_feature_schema_conformance": {"status": "PASSED", "n_features": 19900},
                    },
                    "hardware_execution": {"peak_rss_gb": 0.2, "runtime_seconds": 3.5},
                }
            ),
            encoding="utf-8",
        )
        (self.run_dir / "confound_report.json").write_text(
            json.dumps({"num_columns": 31, "matrix_rank": 31, "condition_number": 12.0}),
            encoding="utf-8",
        )
        (self.run_dir / "framewise_displacement.tsv").write_text(
            "framewise_displacement_mm\n0.0\n0.1\n0.3\n", encoding="utf-8"
        )
        with (self.run_dir / "roi_CC200.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node/volume", "0", "1", "2"])
            for roi in range(1, 201):
                writer.writerow([roi, roi * 0.1, roi * 0.2, roi * 0.3])

        feature_path = self.run_dir / "features_19900_fisher_z.csv"
        with feature_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["feature_name", "fisher_z"])
            for left in range(1, 201):
                for right in range(left + 1, 201):
                    writer.writerow([f"ROI_{left:03d}__ROI_{right:03d}", (right - left) / 100.0])

        artifact_names = [
            "provenance.json",
            "confound_report.json",
            "framewise_displacement.tsv",
            "roi_CC200.csv",
            "features_19900_fisher_z.csv",
        ]
        artifacts = {
            name: {"filename": name, "sha256": self._sha(self.run_dir / name)}
            for name in artifact_names
        }
        (self.run_dir / "output_manifest.json").write_text(
            json.dumps({"status": "SUCCESS", "artifacts": artifacts}), encoding="utf-8"
        )

        subject = {
            "subject_id": self.subject_id,
            "seed": 42,
            "status": "SUCCESS",
            "run_output_dir": str(self.run_dir),
            "output_manifest_sha256": self._sha(self.run_dir / "output_manifest.json"),
            "artifact_count": len(artifacts),
            "runtime_seconds": 3.5,
            "peak_rss_gb": 0.2,
        }
        manifest = {
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "cohort_run_id": self.cohort_id,
            "created_utc": "2026-11-03T12:00:00+00:00",
            "updated_utc": "2026-11-03T12:00:00+00:00",
            "configuration": {"n_subjects": 1},
            "status": "SUCCESS",
            "subjects": [subject],
            "summary": {"total_subjects": 1, "succeeded": 1, "failed": 0, "pending": 0},
        }
        (self.cohort_dir / "cohort_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_list_cohorts_only_includes_synthetic_only_manifests(self):
        other = self.root / "cohort_20261103T123001_123456Z"
        other.mkdir()
        (other / "cohort_manifest.json").write_text(
            json.dumps({
                "run_type": "REAL_COHORT",
                "execution_policy": "MIXED",
                "real_data_authorized": True,
                "subjects": [],
            }),
            encoding="utf-8",
        )
        runs = self.service.list_cohorts()
        self.assertEqual([item["cohort_id"] for item in runs], [self.cohort_id])
        self.assertFalse(runs[0]["real_data_authorized"])

    def test_get_subject_detail_computes_qc_measurements(self):
        detail = self.service.get_subject_detail(self.cohort_id, self.subject_id)
        self.assertAlmostEqual(detail["mean_fd_mm"], 0.1333333333, places=6)
        self.assertAlmostEqual(detail["max_fd_mm"], 0.3)
        self.assertEqual(detail["n_rois"], 200)
        self.assertEqual(detail["n_connectivity_features"], 19900)
        self.assertEqual(detail["confound_rank"], 31)
        self.assertEqual(detail["qc_threshold_status"], "QC_THRESHOLDS_PENDING_SUPERVISOR_APPROVAL")

    def test_roi_timeseries_uses_exported_roi_rows_and_time_columns(self):
        result = self.service.get_timeseries(self.cohort_id, self.subject_id, 7)
        self.assertEqual(result["n_timepoints"], 3)
        self.assertEqual(len(result["values"]), 3)
        for actual, expected in zip(result["values"], [0.7, 1.4, 2.1]):
            self.assertAlmostEqual(actual, expected)

    def test_connectivity_reconstructs_symmetric_200_by_200_matrix(self):
        result = self.service.get_connectivity(self.cohort_id, self.subject_id)
        self.assertEqual(result["n_edges"], 19900)
        self.assertEqual(len(result["matrix"]), 200)
        self.assertEqual(len(result["matrix"][0]), 200)
        self.assertEqual(result["matrix"][0][0], 0.0)
        self.assertAlmostEqual(result["matrix"][0][1], 0.01)
        self.assertAlmostEqual(result["matrix"][1][0], 0.02)

    def test_artifact_is_verified_against_output_manifest(self):
        content, content_type = self.service.get_artifact(
            self.cohort_id, self.subject_id, "features_19900_fisher_z.csv"
        )
        self.assertIn(b"feature_name,fisher_z", content)
        self.assertTrue(content_type.startswith("text/csv"))

    def test_tampered_artifact_is_rejected(self):
        path = self.run_dir / "roi_CC200.csv"
        path.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(DashboardDataError, "integrity"):
            self.service.get_timeseries(self.cohort_id, self.subject_id, 1)

    def test_unsafe_cohort_and_artifact_paths_are_rejected(self):
        with self.assertRaises(DashboardDataError):
            self.service.get_cohort("../outside")
        with self.assertRaises(DashboardDataError):
            self.service.get_artifact(self.cohort_id, self.subject_id, "../../etc/passwd")

    def test_run_output_must_remain_inside_cohort_runs_directory(self):
        manifest_path = self.cohort_dir / "cohort_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["subjects"][0]["run_output_dir"] = str(Path(self.temp.name))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(DashboardDataError, "escapes"):
            self.service.get_subject_detail(self.cohort_id, self.subject_id)

    def test_symlink_artifacts_are_rejected(self):
        real_file = self.run_dir / "framewise_displacement.tsv"
        symlink = self.run_dir / "motion_parameters.tsv"
        symlink.symlink_to(real_file)
        with self.assertRaises(DashboardDataError):
            self.service.get_artifact(self.cohort_id, self.subject_id, "motion_parameters.tsv")

    def test_timeseries_roi_range_is_validated(self):
        with self.assertRaisesRegex(DashboardDataError, "between 1 and 200"):
            self.service.get_timeseries(self.cohort_id, self.subject_id, 201)


if __name__ == "__main__":
    unittest.main()
