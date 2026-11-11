"""Integration tests for automated DICOM-to-BIDS ingestion (Milestone 1).

Verifies all 18 requirements:
1. External storage directory creation
2. External storage path configuration (NEUROADHD_DATA_ROOT)
3. No MRI data written inside the project root
4. dcm2niix detection (shutil.which)
5. Series inspection
6. Patient identifier redaction
7. T1w/BOLD selection
8. Ambiguous series confirmation
9. Successful mocked conversion
10. Missing converter handling
11. Invalid conversion output handling
12. Conversion timeout handling
13. Path traversal protection
14. Original DICOM files remain unchanged
15. Valid BIDS structure generation
16. Existing intake validation integration
17. Multiple BOLD runs support
18. Research authorization remains strictly locked
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from src.intake.service import DEFAULT_DATA_ROOT, IntakeError, IntakeService, get_data_root


class TestDicomBidsIngestion(unittest.TestCase):
    @staticmethod
    def _allow_test_conversion(dataset_id: str, operation: str) -> dict:
        return {
            "is_valid": True,
            "dataset_id": dataset_id,
            "status": "APPROVED",
            "signature_status": "VALID",
            "expiry_status": "VALID_PERIOD",
            "permitted_operations": [operation],
            "issues": [],
        }

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.external_root = Path(self.temp_dir.name) / "NeuroADHD_Converted"
        self.service = IntakeService(
            self.external_root,
            authorization_checker=self._allow_test_conversion,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_nifti_gz(self, path: Path, shape: tuple[int, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
        nib.save(img, str(path))

    def _create_mock_dicom_session(self) -> tuple[str, Path, list[Path]]:
        created = self.service.create_session("scan")
        session_id = created["session_id"]
        session_dir = self.service.staged_sessions_dir / session_id
        data_dir = session_dir / "data"

        # Create two dummy DICOM files for series 1 and series 2
        dcm1 = data_dir / "series1_001.dcm"
        dcm2 = data_dir / "series2_001.dcm"
        dcm1.write_bytes(b"DICM_PLACEHOLDER_SERIES_1_T1")
        dcm2.write_bytes(b"DICM_PLACEHOLDER_SERIES_2_BOLD")

        # Update manifest to register these files
        manifest = self.service._read_manifest(session_dir)
        manifest["uploaded_files"] = [
            {"filename": dcm1.name, "storage_path": f"data/{dcm1.name}", "sha256": hashlib.sha256(dcm1.read_bytes()).hexdigest(), "size_bytes": dcm1.stat().st_size, "uploaded_utc": "2026-11-10T00:00:00Z"},
            {"filename": dcm2.name, "storage_path": f"data/{dcm2.name}", "sha256": hashlib.sha256(dcm2.read_bytes()).hexdigest(), "size_bytes": dcm2.stat().st_size, "uploaded_utc": "2026-11-10T00:00:00Z"},
        ]
        self.service._save_manifest(session_dir, manifest)
        return session_id, session_dir, [dcm1, dcm2]

    # 1. External storage directory creation
    def test_01_external_storage_directory_creation(self):
        custom_root = Path(self.temp_dir.name) / "custom_external"
        self.assertFalse(custom_root.exists())
        svc = IntakeService(custom_root)
        self.assertTrue(custom_root.is_dir())
        self.assertTrue(svc.dicom_input_dir.is_dir())
        self.assertTrue(svc.staged_sessions_dir.is_dir())
        self.assertTrue(svc.bids_dir.is_dir())
        self.assertTrue(svc.validated_sessions_dir.is_dir())
        self.assertTrue(svc.conversion_logs_dir.is_dir())

    # 2. External storage path configuration
    def test_02_external_storage_path_configuration(self):
        env_root = Path(self.temp_dir.name) / "env_mri_storage"
        with mock.patch.dict(os.environ, {"NEUROADHD_DATA_ROOT": str(env_root)}):
            resolved = get_data_root()
            self.assertEqual(resolved, env_root.resolve())
            svc = IntakeService()
            self.assertEqual(svc.data_root, env_root.resolve())

    # 3. No MRI data written inside project root
    def test_03_no_mri_data_written_inside_project_root(self):
        project_root = Path(__file__).resolve().parents[2]
        internal_attempt = project_root / "outputs" / "internal_data_forbidden"
        try:
            with mock.patch.dict(os.environ, {"NEUROADHD_ALLOW_INTERNAL_ROOT": "0"}):
                with self.assertRaises(IntakeError):
                    IntakeService(internal_attempt)
        finally:
            shutil.rmtree(internal_attempt, ignore_errors=True)

    # 4. dcm2niix detection
    def test_04_dcm2niix_detection(self):
        with mock.patch("src.intake.service.shutil.which", return_value="/usr/local/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=["/usr/local/bin/dcm2niix", "-v"],
                    returncode=0,
                    stdout="Chris Rorden's dcm2niiX version v1.0.20261029 Clang21.0.0 ARM\n",
                    stderr="",
                )
                ver = IntakeService._get_dcm2niix_version("/usr/local/bin/dcm2niix")
                self.assertIn("v1.0.20261029", ver)

        with mock.patch("src.intake.service.shutil.which", return_value=None):
            session_id, _, _ = self._create_mock_dicom_session()
            insp = self.service.inspect_dicom_series(session_id)
            self.assertFalse(insp["dcm2niix_available"])
            self.assertIn("not installed", insp["errors"][0])

    # 5. Series inspection
    def test_05_series_inspection(self):
        session_id, session_dir, _ = self._create_mock_dicom_session()
        mock_series_json = [
            {"SeriesNumber": 2, "CRC": "1002", "SeriesDescription": "MPRAGE_T1", "ProtocolName": "t1_mprage", "Modality": "MR", "Files": 176, "Rows": 256, "Columns": 256},
            {"SeriesNumber": 3, "CRC": "1003", "SeriesDescription": "RESTING_STATE_BOLD", "ProtocolName": "ep2d_bold_rest", "Modality": "MR", "Files": 180, "Rows": 64, "Columns": 64},
        ]

        def fake_run(args, **kwargs):
            if "-n" in args and "-2" in args:
                out_dir = Path(args[args.index("-o") + 1])
                (out_dir / "dcm2niix_series.json").write_text(json.dumps(mock_series_json), encoding="utf-8")
            if "-b" in args and "o" in args:
                out_dir = Path(args[args.index("-o") + 1])
                (out_dir / "series_3.json").write_text(json.dumps({"SeriesNumber": 3, "RepetitionTime": 2.0, "EchoTime": 0.03}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                result = self.service.inspect_dicom_series(session_id)

        self.assertTrue(result["dcm2niix_available"])
        self.assertEqual(result["series_count"], 2)
        s1, s2 = result["series"][0], result["series"][1]
        self.assertEqual(s1["candidate_scan_type"], "T1w")
        self.assertEqual(s2["candidate_scan_type"], "BOLD")
        self.assertEqual(s2["repetition_time"], 2.0)
        self.assertEqual(s2["echo_time"], 0.03)

    # 6. Patient identifier redaction
    def test_06_patient_identifier_redaction(self):
        session_id, _, _ = self._create_mock_dicom_session()
        mock_series_json = [
            {
                "SeriesNumber": 1,
                "CRC": "5001",
                "SeriesDescription": "T1_ANAT",
                "PatientName": "JOHN^DOE",
                "PatientID": "HOSPITAL-998811",
                "PatientBirthDate": "19750512",
                "PatientSex": "M",
                "InstitutionAddress": "500 Clinic Way",
            }
        ]

        def fake_run(args, **kwargs):
            if "-n" in args and "-2" in args:
                out_dir = Path(args[args.index("-o") + 1])
                (out_dir / "dcm2niix_series.json").write_text(json.dumps(mock_series_json), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                result = self.service.inspect_dicom_series(session_id)

        rec = result["series"][0]
        self.assertNotIn("PatientName", rec)
        self.assertNotIn("PatientID", rec)
        self.assertNotIn("PatientBirthDate", rec)
        self.assertNotIn("PatientSex", rec)
        self.assertNotIn("InstitutionAddress", rec)
        json_dump = json.dumps(result)
        self.assertNotIn("JOHN^DOE", json_dump)
        self.assertNotIn("HOSPITAL-998811", json_dump)

    # 7. T1w/BOLD selection
    def test_07_t1w_bold_selection(self):
        session_id, _, _ = self._create_mock_dicom_session()

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                conv = self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-test01",
                    "t1w_series": "1",
                    "bold_series": "2",
                })

        self.assertEqual(conv["status"], "CONVERSION_SUCCESSFUL")
        self.assertTrue((self.service.bids_dir / "sub-test01" / "anat" / "sub-test01_T1w.nii.gz").is_file())
        self.assertTrue((self.service.bids_dir / "sub-test01" / "func" / "sub-test01_task-rest_bold.nii.gz").is_file())

    # 8. Ambiguous series confirmation
    def test_08_ambiguous_series_confirmation(self):
        session_id, session_dir, _ = self._create_mock_dicom_session()
        # Mark an ambiguous inspection in manifest
        manifest = self.service._read_manifest(session_dir)
        manifest["dicom_inspection"] = {
            "series": [
                {"series_id": "1", "series_number": 1, "series_description": "ep2d_task_ambiguous", "candidate_scan_type": "BOLD", "classification_ambiguous": True, "ambiguity_reason": "Task not explicitly rest."},
                {"series_id": "2", "series_number": 2, "series_description": "t1_mprage", "candidate_scan_type": "T1w", "classification_ambiguous": False},
            ]
        }
        self.service._save_manifest(session_dir, manifest)

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with self.assertRaisesRegex(IntakeError, "Confirmation required for ambiguous series"):
                self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-ambig",
                    "t1w_series": "2",
                    "bold_series": "1",
                    "confirm_ambiguous": False,
                })

    # 9. Successful mocked conversion
    def test_09_successful_mocked_conversion(self):
        session_id, _, _ = self._create_mock_dicom_session()

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.5, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                result = self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-mock99",
                    "session_id_label": "ses-01",
                    "t1w_series": "1",
                    "bold_series": [{"series_id": "2", "run": 1, "task": "rest"}],
                    "confirm_ambiguous": True,
                })

        self.assertEqual(result["status"], "CONVERSION_SUCCESSFUL")
        self.assertEqual(result["validation"]["status"], "VALIDATED_WAITING_FOR_RESEARCH_AUTHORIZATION")
        self.assertTrue((self.service.validated_sessions_dir / result["download_zip_filename"]).is_file())

    # 10. Missing converter
    def test_10_missing_converter(self):
        session_id, _, _ = self._create_mock_dicom_session()
        with mock.patch("src.intake.service.shutil.which", return_value=None):
            with self.assertRaisesRegex(IntakeError, "dcm2niix converter is not installed"):
                self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-01",
                    "t1w_series": "1",
                    "bold_series": "2",
                })

    # 11. Invalid conversion output
    def test_11_invalid_conversion_output(self):
        session_id, _, _ = self._create_mock_dicom_session()
        # Mock dcm2niix returning 0 but writing no files
        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")):
                with self.assertRaisesRegex(IntakeError, "failed to produce T1w NIfTI"):
                    self.service.convert_dicom_to_bids(session_id, {
                        "subject_id": "sub-01",
                        "t1w_series": "1",
                        "bold_series": "2",
                    })

    # 12. Conversion timeout
    def test_12_conversion_timeout(self):
        session_id, _, _ = self._create_mock_dicom_session()
        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["dcm2niix"], timeout=120)):
                with self.assertRaisesRegex(IntakeError, "conversion timed out"):
                    self.service.convert_dicom_to_bids(session_id, {
                        "subject_id": "sub-01",
                        "t1w_series": "1",
                        "bold_series": "2",
                    })

    # 13. Path traversal protection
    def test_13_path_traversal_protection(self):
        session_id, session_dir, _ = self._create_mock_dicom_session()
        temp_root = Path(self.temp_dir.name)
        before_entries = set(temp_root.iterdir())

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                result = self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "../../../escaped",
                    "t1w_series": "1",
                    "bold_series": "2",
                })

        # The sanitizer turns "../../../escaped" into "sub-escaped" and does not raise
        self.assertEqual(result["subject_id"], "sub-escaped")
        self.assertEqual(result["status"], "CONVERSION_SUCCESSFUL")
        after_entries = set(temp_root.iterdir())
        self.assertEqual(before_entries, after_entries)
        self.assertFalse((temp_root / "escaped").exists())
        self.assertFalse((temp_root / "sub-escaped").exists())

        # Test download zip traversal
        with self.assertRaises(IntakeError):
            self.service.get_bids_session_zip("intake_00000000000000000000000000000000")

    # 14. Original DICOM files remain unchanged
    def test_14_original_dicom_files_remain_unchanged(self):
        session_id, session_dir, dcm_files = self._create_mock_dicom_session()
        original_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in dcm_files}

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-origcheck",
                    "t1w_series": "1",
                    "bold_series": "2",
                })

        for f in dcm_files:
            current_hash = hashlib.sha256(f.read_bytes()).hexdigest()
            self.assertEqual(current_hash, original_hashes[f.name])

    # 15. Valid BIDS structure generation
    def test_15_valid_bids_structure_generation(self):
        session_id, _, _ = self._create_mock_dicom_session()

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-bidsval",
                    "session_id_label": "ses-01",
                    "t1w_series": "1",
                    "bold_series": "2",
                })

        desc = self.service.bids_dir / "dataset_description.json"
        self.assertTrue(desc.is_file())
        desc_data = json.loads(desc.read_text(encoding="utf-8"))
        self.assertEqual(desc_data["BIDSVersion"], "1.8.0")

        t1w_file = self.service.bids_dir / "sub-bidsval" / "ses-01" / "anat" / "sub-bidsval_ses-01_T1w.nii.gz"
        bold_file = self.service.bids_dir / "sub-bidsval" / "ses-01" / "func" / "sub-bidsval_ses-01_task-rest_bold.nii.gz"
        bold_json = self.service.bids_dir / "sub-bidsval" / "ses-01" / "func" / "sub-bidsval_ses-01_task-rest_bold.json"
        self.assertTrue(t1w_file.is_file())
        self.assertTrue(bold_file.is_file())
        self.assertTrue(bold_json.is_file())

    # 16. Existing intake validation
    def test_16_existing_intake_validation(self):
        session_id, _, _ = self._create_mock_dicom_session()
        # Initial scan session with DICOM staged
        checked = self.service.validate_scan_session(session_id)
        self.assertEqual(checked["validation"]["status"], "DICOM_SERIES_STAGED")
        self.assertEqual(checked["validation"]["dicom_file_count"], 2)

    # 17. Multiple BOLD runs
    def test_17_multiple_bold_runs(self):
        session_id, _, _ = self._create_mock_dicom_session()

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 55))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                result = self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-multirun",
                    "t1w_series": "1",
                    "bold_series": [
                        {"series_id": "2", "run": 1, "task": "rest"},
                        {"series_id": "3", "run": 2, "task": "rest"},
                    ],
                })

        self.assertEqual(result["status"], "CONVERSION_SUCCESSFUL")
        func_dir = self.service.bids_dir / "sub-multirun" / "func"
        run1_nii = func_dir / "sub-multirun_task-rest_run-01_bold.nii.gz"
        run2_nii = func_dir / "sub-multirun_task-rest_run-02_bold.nii.gz"
        self.assertTrue(run1_nii.is_file())
        self.assertTrue(run2_nii.is_file())

    # 18. Authorization remains locked
    def test_18_authorization_remains_locked(self):
        session_id, _, _ = self._create_mock_dicom_session()

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run):
                result = self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-authlock",
                    "t1w_series": "1",
                    "bold_series": "2",
                })

        self.assertFalse(result["processing_enabled"])
        self.assertFalse(result["real_data_authorized"])
        self.assertEqual(result["validation"]["status"], "VALIDATED_WAITING_FOR_RESEARCH_AUTHORIZATION")
        self.assertFalse(result["validation"]["processing_enabled"])
        self.assertFalse(result["validation"]["real_data_authorized"])

    # 19. Dual raw/ and bids/ ZIP detection (Regression Test)
    def test_19_dual_raw_and_bids_zip_detection(self):
        created = self.service.create_session("scan")
        session_id = created["session_id"]
        session_dir = self.service.staged_sessions_dir / session_id

        # Build mock ZIP containing both raw/sub-01/ and bids/sub-01/
        zip_buffer = io.BytesIO()
        valid_dicom_content = b"\x00" * 128 + b"DICM" + b"TEST_DICOM_DATA" * 8
        mock_series_specs = [
            ("raw/sub-01/0002_MR000004.dcm", 2, "413754352", "T1w_MPRAGE_1mm", 192),
            ("raw/sub-01/0005_MR000580.dcm", 5, "3837671540", "ep2d_bold_3mm", 240),
            ("raw/sub-01/0004_MR000388.dcm", 4, "2337236158", "T2w_TSE_1mm_iPAT_3D", 192),
            ("raw/sub-01/0006_MR000820.dcm", 6, "3019797074", "ep2d_diff_2mm", 72),
            ("raw/sub-01/0007_MR000892.dcm", 7, "3190052478", "ep2d_diff_2mm_ADC", 72),
            ("raw/sub-01/0001_MR000001.dcm", 1, "3676135602", "localizer", 1),
            ("raw/sub-01/0001_MR000002.dcm", 1, "3676135602", "localizer", 1),
            ("raw/sub-01/0001_MR000003.dcm", 1, "3676135602", "localizer", 1),
            ("raw/sub-01/0003_MR000196.dcm", 3, "3999717435", "T2w_TSE_1mm", 192),
            ("raw/sub-01/0008_MR000964.dcm", 8, "4054646043", "ep2d_diff_2mm_FA", 72),
        ]

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # 1. Add raw DICOM files
            for arc_path, _, _, _, _ in mock_series_specs:
                zf.writestr(arc_path, valid_dicom_content)

            # 2. Add reference BIDS files
            dummy_img = nib.Nifti1Image(np.zeros((10, 10, 10, 60), dtype=np.float32), np.eye(4))
            nii_bytes = io.BytesIO()
            with gzip.GzipFile(fileobj=nii_bytes, mode="wb") as gz:
                gz.write(dummy_img.to_bytes())
            zf.writestr("bids/sub-01/func/sub-01_task-rest_bold.nii.gz", nii_bytes.getvalue())
            zf.writestr("bids/sub-01/func/sub-01_task-rest_bold.json", json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}))

            dummy_t1 = nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.float32), np.eye(4))
            t1_bytes = io.BytesIO()
            with gzip.GzipFile(fileobj=t1_bytes, mode="wb") as gz:
                gz.write(dummy_t1.to_bytes())
            zf.writestr("bids/sub-01/anat/sub-01_T1w.nii.gz", t1_bytes.getvalue())
            zf.writestr("bids/sub-01/anat/sub-01_T1w.json", json.dumps({"EchoTime": 0.002}))
            zf.writestr("bids/dataset_description.json", json.dumps({"BIDSVersion": "1.8.0", "Name": "Ref"}))
            zf.writestr("bids/CHANGES", "Changelog text")

        zip_bytes = zip_buffer.getvalue()
        self.service.upload_file(session_id, "bids-base-DICOM.zip", io.BytesIO(zip_bytes), len(zip_bytes))

        # Check validation: raw/ must be detected, reference NIfTIs must not be counted as DICOM
        val = self.service.validate_scan_session(session_id)
        self.assertEqual(val["validation"]["status"], "DICOM_SERIES_STAGED")
        self.assertEqual(val["validation"]["dicom_file_count"], len(mock_series_specs))
        self.assertEqual(val["validation"]["dicom_directory"], "data/raw/sub-01")
        self.assertFalse(val["validation"]["real_data_authorized"])
        self.assertFalse(val["validation"]["processing_enabled"])

        # Check inspection
        mock_series_json = [
            {"SeriesNumber": s_num, "CRC": crc, "SeriesDescription": desc, "ProtocolName": desc, "Modality": "MR", "Files": files, "Rows": 256, "Columns": 256}
            for _, s_num, crc, desc, files in mock_series_specs
        ]

        dcm2niix_input_paths = []
        def fake_inspect_run(args, **kwargs):
            if "-v" in args:
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="Chris Rorden's dcm2niiX version v1.0.20261029\n", stderr="")
            # Record the directory passed to dcm2niix
            target_input = args[-1]
            dcm2niix_input_paths.append(target_input)
            if "-n" in args and "-2" in args:
                out_dir = Path(args[args.index("-o") + 1])
                (out_dir / "dcm2niix_series.json").write_text(json.dumps(mock_series_json), encoding="utf-8")
            if "-b" in args and "o" in args:
                out_dir = Path(args[args.index("-o") + 1])
                (out_dir / "series_5.json").write_text(json.dumps({"SeriesNumber": 5, "RepetitionTime": 2.0, "EchoTime": 0.03}), encoding="utf-8")
                (out_dir / "series_2.json").write_text(json.dumps({"SeriesNumber": 2, "RepetitionTime": 2.3, "EchoTime": 0.002}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_inspect_run):
                insp = self.service.inspect_dicom_series(session_id)

        # DICOM inspection must only target raw/sub-01, never bids/
        for p in dcm2niix_input_paths:
            self.assertTrue(p.endswith("raw/sub-01"))
            self.assertNotIn("bids", p)

        # 10 series detected
        self.assertEqual(insp["series_count"], 10)
        self.assertEqual(insp["dicom_file_count"], 10)

        # Verify T1w MPRAGE (192 files, CRC 413754352, Series #2)
        t1w = next(s for s in insp["series"] if s["series_crc"] == "413754352")
        self.assertEqual(t1w["series_number"], 2)
        self.assertEqual(t1w["image_count"], 192)
        self.assertEqual(t1w["candidate_scan_type"], "T1w")

        # Verify BOLD (240 files, CRC 3837671540, Series #5)
        bold = next(s for s in insp["series"] if s["series_crc"] == "3837671540")
        self.assertEqual(bold["series_number"], 5)
        self.assertEqual(bold["image_count"], 240)
        self.assertEqual(bold["candidate_scan_type"], "BOLD")

        # Test conversion: verify raw files unchanged and reference NIfTIs not confused
        raw_files = list((session_dir / "data" / "raw" / "sub-01").glob("*.dcm"))
        raw_hashes_before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in raw_files}

        def fake_convert_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"EchoTime": 0.002}), encoding="utf-8")
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_convert_run):
                conv = self.service.convert_dicom_to_bids(session_id, {
                    "subject_id": "sub-dualtest",
                    "t1w_series": "413754352",
                    "bold_series": "3837671540",
                    "confirm_ambiguous": True,
                })

        self.assertEqual(conv["status"], "CONVERSION_SUCCESSFUL")
        self.assertFalse(conv["real_data_authorized"])
        self.assertFalse(conv["processing_enabled"])

        # Verify original DICOM files remain unchanged
        for f in raw_files:
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), raw_hashes_before[f.name])

        # Verify reference BIDS files remain preserved
        ref_bold = session_dir / "data" / "bids" / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"
        self.assertTrue(ref_bold.is_file())


