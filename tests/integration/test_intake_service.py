"""Tests for safe scan-session and separate report intake."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from src.intake.service import IntakeError, IntakeService


class TestIntakeService(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "intake"
        self.service = IntakeService(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _nifti_bytes(self, shape, filename):
        path = Path(self.temp.name) / filename
        image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
        nib.save(image, str(path))
        return path.read_bytes()

    def _session_zip(self):
        bold = self._nifti_bytes((3, 3, 3, 60), "sub-01_task-rest_bold.nii.gz")
        t1w = self._nifti_bytes((3, 3, 3), "sub-01_T1w.nii.gz")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sub-01/func/sub-01_task-rest_bold.nii.gz", bold)
            archive.writestr(
                "sub-01/func/sub-01_task-rest_bold.json",
                json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}),
            )
            archive.writestr("sub-01/anat/sub-01_T1w.nii.gz", t1w)
        return buffer.getvalue()

    @staticmethod
    def _upload(service, session_id, filename, payload):
        return service.upload_file(session_id, filename, io.BytesIO(payload), len(payload))

    def test_complete_bids_zip_validates_headers_and_stays_locked(self):
        created = self.service.create_session("scan")
        result = self._upload(self.service, created["session_id"], "sub-01_session.zip", self._session_zip())
        self.assertEqual(result["status"], "UPLOADED")

        checked = self.service.validate_scan_session(created["session_id"])
        validation = checked["validation"]

        self.assertEqual(validation["status"], "VALIDATED_WAITING_FOR_RESEARCH_AUTHORIZATION")
        self.assertEqual(validation["bold"]["dimensionality"], 4)
        self.assertEqual(validation["bold"]["dimensions"][-1], 60)
        self.assertEqual(validation["t1w"]["dimensionality"], 3)
        self.assertEqual(validation["repetition_time_seconds"], 2.0)
        self.assertFalse(validation["processing_enabled"])
        self.assertFalse(validation["real_data_authorized"])

    def test_incomplete_bids_session_reports_missing_files(self):
        created = self.service.create_session("scan")
        payload = self._nifti_bytes((3, 3, 3, 60), "sub-01_task-rest_bold.nii.gz")
        self._upload(self.service, created["session_id"], "sub-01_task-rest_bold.nii.gz", payload)

        result = self.service.validate_scan_session(created["session_id"])

        self.assertEqual(result["validation"]["status"], "INCOMPLETE_INPUTS")
        self.assertTrue(any("T1w" in item for item in result["validation"]["missing"]))

    def test_zip_path_traversal_is_rejected(self):
        created = self.service.create_session("scan")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../outside.nii.gz", b"not a NIfTI")
        with self.assertRaisesRegex(IntakeError, "path traversal"):
            self._upload(self.service, created["session_id"], "unsafe.zip", buffer.getvalue())
        self.assertFalse((Path(self.temp.name) / "outside.nii.gz").exists())

    def test_zip_symlinks_are_rejected(self):
        created = self.service.create_session("scan")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            info = zipfile.ZipInfo("sub-01/func/link.nii.gz")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, "target")
        with self.assertRaisesRegex(IntakeError, "symlinks"):
            self._upload(self.service, created["session_id"], "linked.zip", buffer.getvalue())

    def test_bold_and_t1w_subject_ids_must_match(self):
        created = self.service.create_session("scan")
        bold = self._nifti_bytes((3, 3, 3, 60), "sub-01_task-rest_bold.nii.gz")
        t1w = self._nifti_bytes((3, 3, 3), "sub-02_T1w.nii.gz")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sub-01/func/sub-01_task-rest_bold.nii.gz", bold)
            archive.writestr("sub-01/func/sub-01_task-rest_bold.json", json.dumps({"RepetitionTime": 2.0}))
            archive.writestr("sub-02/anat/sub-02_T1w.nii.gz", t1w)
        self._upload(self.service, created["session_id"], "mixed-subjects.zip", buffer.getvalue())

        result = self.service.validate_scan_session(created["session_id"])

        self.assertEqual(result["validation"]["status"], "VALIDATION_FAILED")
        self.assertTrue(any("same BIDS subject" in item for item in result["validation"]["errors"]))

    def test_non_rest_bold_task_is_rejected(self):
        created = self.service.create_session("scan")
        bold = self._nifti_bytes((3, 3, 3, 60), "sub-01_task-movie_bold.nii.gz")
        t1w = self._nifti_bytes((3, 3, 3), "sub-01_T1w.nii.gz")
        self._upload(self.service, created["session_id"], "sub-01_task-movie_bold.nii.gz", bold)
        self._upload(self.service, created["session_id"], "sub-01_T1w.nii.gz", t1w)
        self._upload(
            self.service, created["session_id"], "sub-01_task-movie_bold.json",
            json.dumps({"RepetitionTime": 2.0}).encode(),
        )

        result = self.service.validate_scan_session(created["session_id"])

        self.assertEqual(result["validation"]["status"], "VALIDATION_FAILED")
        self.assertTrue(any("task-rest" in item for item in result["validation"]["errors"]))

    def test_invalid_nifti_header_fails_validation(self):
        created = self.service.create_session("scan")
        self._upload(self.service, created["session_id"], "sub-01_task-rest_bold.nii.gz", b"not nifti")
        self._upload(self.service, created["session_id"], "sub-01_T1w.nii.gz", b"not nifti")
        self._upload(
            self.service, created["session_id"], "sub-01_task-rest_bold.json",
            json.dumps({"RepetitionTime": 2.0}).encode(),
        )

        result = self.service.validate_scan_session(created["session_id"])

        self.assertEqual(result["validation"]["status"], "VALIDATION_FAILED")
        self.assertTrue(result["validation"]["errors"])

    def test_report_upload_is_separate_and_does_not_claim_ocr_without_engine(self):
        created = self.service.create_session("report")
        self._upload(self.service, created["session_id"], "radiology-report.pdf", b"%PDF-1.4 placeholder")
        with mock.patch("src.intake.service.shutil.which", return_value=None):
            result = self.service.process_report_session(created["session_id"])

        self.assertEqual(result["report_extraction"]["status"], "REPORTS_STAGED_ENGINE_REQUIRED")
        self.assertEqual(result["report_extraction"]["files"][0]["status"], "PDF_TEXT_OR_OCR_ENGINE_NOT_AVAILABLE")
        self.assertIn("not used as fMRI input", result["report_extraction"]["notice"])

    def test_report_artifact_download_rejects_original_report_and_traversal(self):
        created = self.service.create_session("report")
        with self.assertRaises(IntakeError):
            self.service.get_report_artifact(created["session_id"], "../../report.pdf")

    def test_scan_session_rejects_report_files(self):
        created = self.service.create_session("scan")
        with self.assertRaisesRegex(IntakeError, "Unsupported scan file type"):
            self._upload(self.service, created["session_id"], "report.pdf", b"%PDF")

    def test_report_session_rejects_scan_files(self):
        created = self.service.create_session("report")
        with self.assertRaisesRegex(IntakeError, "Unsupported report file type"):
            self._upload(self.service, created["session_id"], "bold.nii.gz", b"data")

    def test_list_sessions_exposes_only_session_metadata(self):
        created = self.service.create_session("scan")
        sessions = self.service.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], created["session_id"])
        self.assertNotIn("storage_path", sessions[0])


if __name__ == "__main__":
    unittest.main()
