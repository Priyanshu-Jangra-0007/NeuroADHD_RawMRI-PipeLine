"""Unit & integration-style tests for Step 9: N=6 Pilot Readiness & Synthetic Dry Run."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema
import numpy as np

from scripts.run_poc_v2 import (
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from scripts.run_synthetic_cohort import (
    PILOT_N6_PROFILES,
    _verify_run_artifacts,
    run_synthetic_cohort,
)
from scripts.serve_dashboard import DashboardApplication
from src.dashboard.service import DashboardDataService
from src.validation.authorization_gate import (
    AuthorizationDeniedError,
    ResearchAuthorizationGate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_PATH = REPO_ROOT / "docs" / "pilot" / "N6_PILOT_TRACKER.csv"
PROTOCOL_PATH = REPO_ROOT / "docs" / "pilot" / "N6_PILOT_EXECUTION_PROTOCOL.md"
QC_DOC_PATH = REPO_ROOT / "docs" / "pilot" / "N6_PILOT_QC_CRITERIA.md"
CPAC_DOC_PATH = REPO_ROOT / "docs" / "pilot" / "CPAC_CONCORDANCE_PLAN.md"
PROVENANCE_SCHEMA_PATH = REPO_ROOT / "configs" / "provenance_schema.json"

REQUIRED_TRACKER_COLUMNS = [
    "pilot_slot",
    "placeholder_id",
    "session_id",
    "authorization_status",
    "permitted_operations",
    "dataset_scope",
    "modality_completeness",
    "defacing_status",
    "sdc_status",
    "stc_status",
    "pipeline_status",
    "qc_status",
    "cpac_comparison_status",
    "manual_review_status",
    "notes",
]


class TestStep9PilotDocumentationAndFailClosedGuards(unittest.TestCase):
    """Validate Step 9 documentation invariants and fail-closed authorization boundaries."""

    def test_01_pilot_docs_and_six_placeholder_slots(self):
        for doc in (TRACKER_PATH, PROTOCOL_PATH, QC_DOC_PATH, CPAC_DOC_PATH):
            self.assertTrue(doc.is_file(), f"Missing Step 9 deliverable: {doc}")

        with TRACKER_PATH.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, REQUIRED_TRACKER_COLUMNS)
            rows = list(reader)

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row["placeholder_id"] for row in rows],
            [f"sub-pilot-{i:02d}" for i in range(1, 7)],
        )
        self.assertEqual([int(row["pilot_slot"]) for row in rows], [1, 2, 3, 4, 5, 6])

        protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8")
        for idx in range(1, 7):
            self.assertIn(f"sub-pilot-{idx:02d}", protocol_text)
        self.assertIn("NOT_AUTHORIZED", protocol_text)
        self.assertIn("QC_THRESHOLDS_PENDING_SUPERVISOR_APPROVAL", QC_DOC_PATH.read_text(encoding="utf-8"))
        self.assertIn("proposed", CPAC_DOC_PATH.read_text(encoding="utf-8").lower())

    def test_02_pilot_tracker_contains_no_real_ids_or_diagnoses(self):
        raw_tracker = TRACKER_PATH.read_text(encoding="utf-8")
        with TRACKER_PATH.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))

        for row in rows:
            self.assertEqual(row["authorization_status"], "PENDING_NOT_AUTHORIZED")
            self.assertIn("real_ops=NONE", row["permitted_operations"])
            self.assertNotIn("APPROVED", row["authorization_status"])
            self.assertNotIn("diagnosis", " ".join(row.keys()).lower())

        lowered = raw_tracker.lower()
        for forbidden in ("adhd-200", "peking", "nyu_", "kki_", "ohsu_", "neuroimage", "sub-00", "mrn"):
            self.assertNotIn(forbidden, lowered)

    def test_03_attempted_real_data_pilot_execution_is_denied(self):
        gate = ResearchAuthorizationGate(base_dir=REPO_ROOT)
        report = gate.evaluate_register()
        self.assertEqual(report["overall_authorization_state"], "NOT_AUTHORIZED")
        self.assertFalse(report["real_data_authorized"])
        with self.assertRaises(AuthorizationDeniedError):
            gate.check_execution_allowed(requested_mode="real", allow_real_scans=True)

    def test_04_denied_execution_never_opens_real_mri_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_real_dir = Path(tmp) / "real_mri_do_not_touch"
            fake_real_dir.mkdir(parents=True)
            fake_bold = fake_real_dir / "sub-pilot-01_task-rest_bold.nii.gz"
            fake_t1w = fake_real_dir / "sub-pilot-01_T1w.nii.gz"
            fake_bold.write_bytes(b"SENTINEL_UNREAD_BOLD")
            fake_t1w.write_bytes(b"SENTINEL_UNREAD_T1W")

            with mock.patch("nibabel.load", side_effect=AssertionError("nibabel.load must not be called on denied run")):
                with self.assertRaises(AuthorizationDeniedError):
                    run_poc_v2_pipeline(
                        bold_path=fake_bold,
                        bold_json_path=fake_real_dir / "missing.json",
                        t1w_path=fake_t1w,
                        atlas_path=fake_real_dir / "atlas.nii.gz",
                        template_path=fake_real_dir / "tpl.nii.gz",
                        output_dir=Path(tmp) / "out",
                        subject_id="sub-pilot-01",
                        session_id="ses-1",
                        allow_real_scans=True,
                    )

            self.assertEqual(fake_bold.read_bytes(), b"SENTINEL_UNREAD_BOLD")
            self.assertEqual(fake_t1w.read_bytes(), b"SENTINEL_UNREAD_T1W")

    def test_05_synthetic_execution_permitted_only_under_strict_synthetic_policy(self):
        gate = ResearchAuthorizationGate(base_dir=REPO_ROOT)
        verdict = gate.check_execution_allowed(requested_mode="synthetic", allow_real_scans=False)
        self.assertEqual(verdict["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertEqual(verdict["authorized_mode"], "synthetic")
        self.assertFalse(verdict["real_data_authorized"])
        self.assertEqual(verdict["overall_authorization_state"], "NOT_AUTHORIZED")

    def test_06_per_subject_failure_isolation_in_pilot_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "pilot_isolation"

            def _fake_gen(work_dir: Path, **kwargs):
                subj = kwargs["subject_id"]
                sess = kwargs["session_id"]
                sdir = work_dir / subj / sess
                sdir.mkdir(parents=True, exist_ok=True)
                files = {}
                for key, fname in (
                    ("bold_nii", "b.nii.gz"),
                    ("bold_json", "b.json"),
                    ("t1w_nii", "t1.nii.gz"),
                    ("mni_template", "mni.nii.gz"),
                    ("cc200_atlas", "atlas.nii.gz"),
                    ("synthetic_manifest", "synthetic_manifest.json"),
                ):
                    p = sdir / fname
                    p.write_text(f"{subj}:{key}", encoding="utf-8")
                    files[key] = p
                return files

            calls = {"count": 0}

            def _fake_pipe(**kwargs):
                calls["count"] += 1
                subj = kwargs["subject_id"]
                if subj == "sub-synthetic003":
                    raise RuntimeError("Induced synthetic slot 03 failure for isolation check")
                rdir = Path(kwargs["output_dir"]) / subj / "run_ok"
                rdir.mkdir(parents=True, exist_ok=True)
                prov = rdir / "provenance.json"
                prov_data = {
                    "pipeline_stages_total": 23,
                    "slice_timing_correction": {"status": "APPLIED"},
                    "susceptibility_distortion_correction": {"status": "SKIPPED_TOOL_UNAVAILABLE"},
                    "defacing": {"status": "SKIPPED_TOOL_UNAVAILABLE"},
                    "validation_gates": {
                        "gate2_bids_input": {"tr_seconds": 2.0, "t_vol": 60},
                        "gate5_roi_coverage": {"n_rois": 200, "zero_variance_rois": 0},
                        "gate6_feature_schema_conformance": {"n_features": 19900, "finite_features": 19900},
                    },
                }
                prov.write_text(json.dumps(prov_data), encoding="utf-8")
                import hashlib

                digest = hashlib.sha256(prov.read_bytes()).hexdigest()
                (rdir / "output_manifest.json").write_text(
                    json.dumps({"status": "SUCCESS", "artifacts": {"provenance": {"filename": "provenance.json", "sha256": digest}}}),
                    encoding="utf-8",
                )
                return {"status": "SUCCESS", "output_dir": rdir, "runtime_seconds": 0.4, "peak_rss_gb": 0.2, "provenance": prov_data}

            with mock.patch("scripts.run_synthetic_cohort.generate_synthetic_phantom_session", side_effect=_fake_gen), mock.patch(
                "scripts.run_synthetic_cohort.run_poc_v2_pipeline", side_effect=_fake_pipe
            ):
                res = run_synthetic_cohort(
                    out_dir,
                    n_subjects=6,
                    cohort_profile="pilot_n6",
                    continue_on_error=True,
                )

            self.assertEqual(calls["count"], 6)
            self.assertEqual(res["status"], "PARTIAL_FAILURE")
            self.assertEqual(res["summary"]["succeeded"], 5)
            self.assertEqual(res["summary"]["failed"], 1)
            self.assertEqual(res["summary"]["pending"], 0)
            self.assertEqual(res["subjects"][2]["status"], "FAILED")
            self.assertIn("Induced synthetic slot 03 failure", res["subjects"][2]["error"])
            self.assertEqual(res["subjects"][5]["status"], "SUCCESS")

    def test_13_dashboard_pilot_readiness_endpoint_and_ui_warnings(self):
        app = DashboardApplication()
        payload = app.pilot_readiness_status()
        self.assertEqual(payload["pilot_scope_limit"], 6)
        self.assertEqual(payload["pilot_slots_count"], 6)
        self.assertEqual(payload["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertFalse(payload["real_data_authorized"])
        self.assertEqual(payload["authorization_status"], "NOT_AUTHORIZED")
        self.assertEqual(
            payload["persistent_warning"],
            "REAL-DATA PILOT BLOCKED — INSTITUTIONAL AUTHORIZATION NOT GRANTED",
        )
        self.assertEqual(payload["qc_threshold_status"], "QC_THRESHOLDS_PENDING_SUPERVISOR_APPROVAL")
        self.assertEqual(len(payload["slots"]), 6)

        serialized = json.dumps(payload)
        self.assertNotIn("cryptographic_signature", serialized)
        self.assertNotIn("NEUROADHD_HMAC_KEY", serialized)
        self.assertNotIn("/Users/", serialized)

        html = (REPO_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn("pilot-readiness", html)
        self.assertIn("REAL-DATA PILOT BLOCKED — INSTITUTIONAL AUTHORIZATION NOT GRANTED", html)

    def test_14_cli_and_programmatic_mode_real_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AuthorizationDeniedError):
                run_synthetic_cohort(
                    Path(tmp) / "should_never_exist",
                    n_subjects=1,
                    mode="real",
                    allow_real_scans=True,
                )
            self.assertFalse((Path(tmp) / "should_never_exist").exists())

            cohort_proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "run_synthetic_cohort.py"), "--mode", "real"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(cohort_proc.returncode, 0)
            self.assertIn("AuthorizationDeniedError", cohort_proc.stderr)

            poc_proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "run_poc_v2.py"), "--mode", "real"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(poc_proc.returncode, 0)


class TestStep9RealSyntheticPipelineAndReproducibility(unittest.TestCase):
    """Execute authentic synthetic runs verifying stages, outputs, skips, schema, and determinism."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.base = Path(cls._tmpdir.name)
        cls.cohort_dir = cls.base / "dashboard_cohorts" / "cohort_20261112T000000_000001Z"

        # Generate deterministic single-subject synthetic sessions verifying varied modes:
        # Run A & Run B: identical seed=42 interleaved_k + direct_hz fieldmap -> determinism check (#11)
        cls.run_a_in = cls.base / "inputs_a"
        cls.run_a_out = cls.base / "outputs_a"
        paths_a = generate_synthetic_phantom_session(
            cls.run_a_in,
            subject_id="sub-synthetic001",
            session_id="ses-1",
            t_vol=60,
            seed=42,
            slice_encoding_direction="k",
            phase_encoding_direction="j",
            total_readout_time=0.05,
            fieldmap_mode="direct_hz",
        )
        cls.res_a = run_poc_v2_pipeline(
            bold_path=paths_a["bold_nii"],
            bold_json_path=paths_a["bold_json"],
            t1w_path=paths_a["t1w_nii"],
            atlas_path=paths_a["cc200_atlas"],
            template_path=paths_a["mni_template"],
            output_dir=cls.run_a_out,
            subject_id="sub-synthetic001",
            session_id="ses-1",
            manifest_path=paths_a["synthetic_manifest"],
            allow_real_scans=False,
        )

        cls.run_b_in = cls.base / "inputs_b"
        cls.run_b_out = cls.base / "outputs_b"
        paths_b = generate_synthetic_phantom_session(
            cls.run_b_in,
            subject_id="sub-synthetic001",
            session_id="ses-1",
            t_vol=60,
            seed=42,
            slice_encoding_direction="k",
            phase_encoding_direction="j",
            total_readout_time=0.05,
            fieldmap_mode="direct_hz",
        )
        cls.res_b = run_poc_v2_pipeline(
            bold_path=paths_b["bold_nii"],
            bold_json_path=paths_b["bold_json"],
            t1w_path=paths_b["t1w_nii"],
            atlas_path=paths_b["cc200_atlas"],
            template_path=paths_b["mni_template"],
            output_dir=cls.run_b_out,
            subject_id="sub-synthetic001",
            session_id="ses-1",
            manifest_path=paths_b["synthetic_manifest"],
            allow_real_scans=False,
        )

        # Control run C: PEPOLAR + missing SliceTiming -> honest SKIPPED reporting (#7)
        cls.run_c_in = cls.base / "inputs_c"
        cls.run_c_out = cls.base / "outputs_c"
        paths_c = generate_synthetic_phantom_session(
            cls.run_c_in,
            subject_id="sub-synthetic005",
            session_id="ses-1",
            t_vol=60,
            seed=46,
            include_slice_timing=False,
            slice_encoding_direction="k",
            phase_encoding_direction="j",
            total_readout_time=0.05,
            fieldmap_mode="pepolar",
        )
        cls.res_c = run_poc_v2_pipeline(
            bold_path=paths_c["bold_nii"],
            bold_json_path=paths_c["bold_json"],
            t1w_path=paths_c["t1w_nii"],
            atlas_path=paths_c["cc200_atlas"],
            template_path=paths_c["mni_template"],
            output_dir=cls.run_c_out,
            subject_id="sub-synthetic005",
            session_id="ses-1",
            manifest_path=paths_c["synthetic_manifest"],
            allow_real_scans=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_07_missing_tool_and_missing_metadata_skip_honestly(self):
        prov_c = self.res_c["provenance"]
        self.assertEqual(prov_c["slice_timing_correction"]["status"], "SKIPPED_MISSING_METADATA")
        # Unless FSL topup is installed on host, PEPOLAR must skip honestly with SKIPPED_TOOL_UNAVAILABLE
        self.assertIn(
            prov_c["susceptibility_distortion_correction"]["status"],
            ("SKIPPED_TOOL_UNAVAILABLE", "APPLIED_PEPOLAR_TOPUP"),
        )
        self.assertEqual(prov_c["defacing"]["status"], "SKIPPED_TOOL_UNAVAILABLE")

    def test_08_and_09_cc200_timeseries_and_19900_finite_features(self):
        roi_csv = Path(self.res_a["artifacts"]["roi_csv"])
        with roi_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 200)
        # First column is node/volume followed by 60 volume columns
        vol_cols = [c for c in rows[0].keys() if c != "node/volume"]
        self.assertEqual(len(vol_cols), 60)

        vec = np.load(self.res_a["artifacts"]["features_npy"])
        self.assertEqual(vec.shape, (19900,))
        self.assertTrue(np.all(np.isfinite(vec)))

        with Path(self.res_a["artifacts"]["features_csv"]).open("r", encoding="utf-8", newline="") as stream:
            feat_rows = list(csv.DictReader(stream))
        self.assertEqual(len(feat_rows), 19900)

    def test_10_provenance_matches_schema_and_executed_stages(self):
        schema = json.loads(PROVENANCE_SCHEMA_PATH.read_text(encoding="utf-8"))
        prov = self.res_a["provenance"]
        jsonschema.validate(instance=prov, schema=schema)
        self.assertEqual(prov["pipeline_stages_total"], 23)
        self.assertEqual(prov["slice_timing_correction"]["status"], "APPLIED")
        self.assertEqual(prov["susceptibility_distortion_correction"]["status"], "APPLIED_DIRECT_FIELDMAP")
        self.assertEqual(prov["defacing"]["status"], "SKIPPED_TOOL_UNAVAILABLE")
        self.assertEqual(prov["validation_gates"]["gate5_roi_coverage"]["n_rois"], 200)
        self.assertEqual(prov["validation_gates"]["gate6_feature_schema_conformance"]["n_features"], 19900)

    def test_11_identical_seed_reproducibility(self):
        vec_a = np.load(self.res_a["artifacts"]["features_npy"])
        vec_b = np.load(self.res_b["artifacts"]["features_npy"])
        max_diff = float(np.max(np.abs(vec_a - vec_b)))
        self.assertLessEqual(max_diff, 1e-7)

    def test_12_output_manifest_checksums_and_dashboard_compatibility(self):
        out_dir_a = Path(self.res_a["output_dir"])
        verified = _verify_run_artifacts(out_dir_a)
        self.assertGreaterEqual(len(verified["artifacts"]), 12)

        # Wrap Run A into a synthetic cohort manifest under DashboardDataService contract
        cohort_root = self.base / "dash_root"
        cohort_dir = cohort_root / "cohort_20261112T000001_000001Z"
        run_link_dir = cohort_dir / "runs" / "sub-synthetic001" / "run_test"
        run_link_dir.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copytree(out_dir_a, run_link_dir)
        manifest = {
            "manifest_version": "1.0.0",
            "run_type": "SYNTHETIC_COHORT_ONLY",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "cohort_run_id": cohort_dir.name,
            "created_utc": "2026-11-12T00:00:00Z",
            "updated_utc": "2026-11-12T00:00:01Z",
            "pilot_dry_run": True,
            "status": "SUCCESS",
            "configuration": {"cohort_profile": "pilot_n6"},
            "summary": {"total_subjects": 1, "succeeded": 1, "failed": 0, "pending": 0},
            "subjects": [
                {
                    "subject_id": "sub-synthetic001",
                    "pilot_slot": "sub-pilot-01",
                    "status": "SUCCESS",
                    "run_output_dir": str(run_link_dir),
                    "output_manifest_sha256": verified["manifest_sha256"],
                    "artifact_count": len(verified["artifacts"]),
                    "runtime_seconds": self.res_a["runtime_seconds"],
                    "peak_rss_gb": self.res_a["peak_rss_gb"],
                }
            ],
        }
        (cohort_dir / "cohort_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        service = DashboardDataService(cohort_root)
        detail = service.get_subject_detail(cohort_dir.name, "sub-synthetic001")
        self.assertEqual(detail["n_rois"], 200)
        self.assertEqual(detail["n_connectivity_features"], 19900)
        self.assertEqual(detail["execution_mode"], "SYNTHETIC_DEMONSTRATION_ONLY")
        conn = service.get_connectivity(cohort_dir.name, "sub-synthetic001")
        self.assertEqual(conn["n_edges"], 19900)


if __name__ == "__main__":
    unittest.main()

