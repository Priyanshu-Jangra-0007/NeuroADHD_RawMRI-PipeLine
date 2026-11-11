"""Integration tests enforcing the fail-closed research authorization gate on DICOM conversion."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from src.intake.service import IntakeError, IntakeService
from src.validation.authorization_gate import ResearchAuthorizationGate

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTER_PATH = REPO_ROOT / "configs" / "research_authorization_register.json"
REAL_SCHEMA_PATH = REPO_ROOT / "configs" / "authorization_register_schema.json"


class TestDicomConversionAuthorizationGate(unittest.TestCase):
    """Fail-closed DICOM conversion authorization checks against ResearchAuthorizationGate."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_tmp = Path(self.temp_dir.name).resolve()
        self.external_root = self.base_tmp / "NeuroADHD_Converted"
        self.fake_project_root = self.base_tmp / "temp_repo"
        (self.fake_project_root / "configs").mkdir(parents=True, exist_ok=True)
        (self.fake_project_root / "configs" / "authorization_register_schema.json").write_text(
            REAL_SCHEMA_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.register_path = self.fake_project_root / "configs" / "research_authorization_register.json"
        self.test_hmac_key = "test-gate-hmac-key-0123456789abcdef"
        self.gate = ResearchAuthorizationGate(base_dir=self.fake_project_root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_nifti_gz(self, path: Path, shape: tuple[int, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
        nib.save(img, str(path))

    def _build_record(self, sign_with_key: str | None = None, **overrides) -> dict:
        record = {
            "authorization_id": "AUTH-DS-SUB01-TEST",
            "project_id": "NeuroADHD RawMRI PoC V2",
            "dataset_id": "sub-01",
            "approved_purpose": "Controlled feasibility testing for DICOM normalization.",
            "institutional_approval_reference": "IEC-2026-TEST-001",
            "approval_date": "2026-01-15",
            "expiry_date": "2028-12-31",
            "permitted_operations": ["dicom_conversion", "bids_validation"],
            "authorized_personnel": "Principal Investigator",
            "authorization_status": "APPROVED",
            "cryptographic_signature": None,
            "notes": "Test dataset authorization record.",
        }
        record.update(overrides)
        if sign_with_key is not None:
            record["cryptographic_signature"] = self.gate.compute_authorization_signature(
                record, key=sign_with_key
            )
        return record

    def _write_temp_register(self, records: list[dict]) -> Path:
        base_register = json.loads(REAL_REGISTER_PATH.read_text(encoding="utf-8"))
        base_register["dataset_authorizations"] = records
        self.register_path.write_text(json.dumps(base_register, indent=2), encoding="utf-8")
        return self.register_path

    def _new_empty_scan_session(self, service: IntakeService) -> tuple[str, Path]:
        created = service.create_session("scan")
        session_id = created["session_id"]
        session_dir = service.staged_sessions_dir / session_id
        return session_id, session_dir

    def _new_staged_dicom_session(self, service: IntakeService) -> tuple[str, Path, list[Path]]:
        session_id, session_dir = self._new_empty_scan_session(service)
        data_dir = session_dir / "data"
        dcm1 = data_dir / "series1_001.dcm"
        dcm2 = data_dir / "series2_001.dcm"
        dcm1.write_bytes(b"\x00" * 128 + b"DICM" + b"T1_SLICE_DATA")
        dcm2.write_bytes(b"\x00" * 128 + b"DICM" + b"BOLD_SLICE_DATA")
        manifest = service._read_manifest(session_dir)
        manifest["uploaded_files"] = [
            {
                "filename": dcm1.name,
                "storage_path": f"data/{dcm1.name}",
                "sha256": hashlib.sha256(dcm1.read_bytes()).hexdigest(),
                "size_bytes": dcm1.stat().st_size,
                "uploaded_utc": "2026-11-11T00:00:00Z",
            },
            {
                "filename": dcm2.name,
                "storage_path": f"data/{dcm2.name}",
                "sha256": hashlib.sha256(dcm2.read_bytes()).hexdigest(),
                "size_bytes": dcm2.stat().st_size,
                "uploaded_utc": "2026-11-11T00:00:00Z",
            },
        ]
        service._save_manifest(session_dir, manifest)
        return session_id, session_dir, [dcm1, dcm2]

    def _assert_blocked(
        self,
        service: IntakeService,
        spec: dict,
        *,
        expected_reason_fragment: str,
        secrets_never_logged: list[str] | None = None,
        with_staged_dicoms: bool = False,
    ) -> IntakeError:
        if with_staged_dicoms:
            session_id, session_dir, _ = self._new_staged_dicom_session(service)
        else:
            session_id, session_dir = self._new_empty_scan_session(service)

        data_dir = session_dir / "data"
        before_data = sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*"))
        before_bids = sorted(str(p.relative_to(service.bids_dir)) for p in service.bids_dir.rglob("*"))
        before_zips = sorted(
            str(p.relative_to(service.validated_sessions_dir)) for p in service.validated_sessions_dir.rglob("*")
        )

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run") as mock_run:
                with self.assertRaises(IntakeError) as ctx:
                    service.convert_dicom_to_bids(session_id, spec)

        mock_run.assert_not_called()

        after_data = sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*"))
        after_bids = sorted(str(p.relative_to(service.bids_dir)) for p in service.bids_dir.rglob("*"))
        after_zips = sorted(
            str(p.relative_to(service.validated_sessions_dir)) for p in service.validated_sessions_dir.rglob("*")
        )
        self.assertEqual(before_data, after_data)
        if not with_staged_dicoms:
            self.assertEqual(after_data, [])
        self.assertEqual(before_bids, after_bids)
        self.assertEqual(before_zips, after_zips)

        msg = str(ctx.exception)
        self.assertRegex(msg.lower(), expected_reason_fragment.lower())
        for secret in secrets_never_logged or []:
            if secret:
                self.assertNotIn(secret, msg)
        return ctx.exception

    def test_01_real_register_sub01_pending_is_blocked(self):
        service = IntakeService(self.external_root)
        spec = {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"}

        with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
            # Unpopulated scan session leaves 0 files under data/
            self._assert_blocked(
                service,
                spec,
                expected_reason_fragment=r"pending",
                secrets_never_logged=[self.test_hmac_key],
                with_staged_dicoms=False,
            )
            # Staged DICOM session leaves data/ unchanged and bids/ empty
            self._assert_blocked(
                service,
                spec,
                expected_reason_fragment=r"pending",
                secrets_never_logged=[self.test_hmac_key],
                with_staged_dicoms=True,
            )

    def test_02_valid_approved_signed_record_allows_conversion(self):
        record = self._build_record(sign_with_key=self.test_hmac_key)
        self._write_temp_register([record])

        service = IntakeService(self.external_root)
        session_id, _, _ = self._new_staged_dicom_session(service)

        def fake_run(args, **kwargs):
            out_dir = Path(args[args.index("-o") + 1])
            prefix = args[args.index("-f") + 1]
            if "T1w" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (20, 20, 20))
            elif "bold" in prefix:
                self._create_synthetic_nifti_gz(out_dir / f"{prefix}.nii.gz", (10, 10, 10, 60))
                (out_dir / f"{prefix}.json").write_text(
                    json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}),
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
                    with mock.patch("src.intake.service.subprocess.run", side_effect=fake_run) as mock_run:
                        result = service.convert_dicom_to_bids(
                            session_id,
                            {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"},
                        )

        self.assertGreaterEqual(mock_run.call_count, 2)
        self.assertEqual(result["status"], "CONVERSION_SUCCESSFUL")
        self.assertEqual(result["subject_id"], "sub-01")
        self.assertFalse(result["processing_enabled"])
        self.assertFalse(result["real_data_authorized"])
        self.assertTrue((service.bids_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz").is_file())
        self.assertTrue((service.bids_dir / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz").is_file())

    def test_03_tampered_field_is_blocked(self):
        record = self._build_record(sign_with_key=self.test_hmac_key)
        original_sig = record["cryptographic_signature"]
        record["approved_purpose"] = "Tampered purpose after signing."
        self._write_temp_register([record])

        service = IntakeService(self.external_root)
        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                self._assert_blocked(
                    service,
                    {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"},
                    expected_reason_fragment=r"signature",
                    secrets_never_logged=[self.test_hmac_key, original_sig],
                )

    def test_04_wrong_key_and_missing_key_are_blocked(self):
        record = self._build_record(sign_with_key=self.test_hmac_key)
        sig = record["cryptographic_signature"]
        self._write_temp_register([record])
        service = IntakeService(self.external_root)
        spec = {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"}

        wrong_key = "wrong-hmac-key-9999999999999999"
        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": wrong_key}):
                self._assert_blocked(
                    service,
                    spec,
                    expected_reason_fragment=r"signature",
                    secrets_never_logged=[self.test_hmac_key, wrong_key, sig],
                )

            cleaned_env = {
                k: v for k, v in os.environ.items() if k not in ("NEUROADHD_HMAC_KEY", "NEUROADHD_INTEGRITY_KEY")
            }
            with mock.patch.dict(os.environ, cleaned_env, clear=True):
                self._assert_blocked(
                    service,
                    spec,
                    expected_reason_fragment=r"hmac key",
                    secrets_never_logged=[self.test_hmac_key, sig],
                )

    def test_05_expired_and_non_approved_statuses_are_blocked(self):
        expired = self._build_record(sign_with_key=self.test_hmac_key, expiry_date="2025-01-01")
        self._write_temp_register([expired])
        service = IntakeService(self.external_root)
        spec = {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"}

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                self._assert_blocked(
                    service,
                    spec,
                    expected_reason_fragment=r"expired",
                    secrets_never_logged=[self.test_hmac_key, expired["cryptographic_signature"]],
                )

                for status in ("PENDING", "REVOKED", "REJECTED"):
                    rec = self._build_record(sign_with_key=self.test_hmac_key, authorization_status=status)
                    self._write_temp_register([rec])
                    self._assert_blocked(
                        service,
                        spec,
                        expected_reason_fragment=status.lower(),
                        secrets_never_logged=[self.test_hmac_key, rec["cryptographic_signature"]],
                    )

    def test_06_wrong_dataset_id_and_exact_dataset_id_matching(self):
        # Only exact dataset_id "sub-01" is approved; subject aliases/traversal must NEVER borrow it.
        record = self._build_record(sign_with_key=self.test_hmac_key, dataset_id="sub-01")
        self._write_temp_register([record])
        service = IntakeService(self.external_root)

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                for bad_subject in (
                    "sub-02",
                    "01",
                    "sub01",
                    "SUB-01",
                    "sub_01",
                    "../../../sub-01",
                    " sub-01",
                    "sub-01 ",
                ):
                    self._assert_blocked(
                        service,
                        {"subject_id": bad_subject, "t1w_series": "1", "bold_series": "2"},
                        expected_reason_fragment=r"no matching authorization record|dataset|whitespace|invalid",
                        secrets_never_logged=[self.test_hmac_key, record["cryptographic_signature"]],
                    )

                # Explicit dataset_id field conflicting with subject_id is also blocked
                self._assert_blocked(
                    service,
                    {
                        "subject_id": "sub-01",
                        "dataset_id": "sub-02",
                        "t1w_series": "1",
                        "bold_series": "2",
                    },
                    expected_reason_fragment=r"does not match",
                    secrets_never_logged=[self.test_hmac_key, record["cryptographic_signature"]],
                )

    def test_07_operation_not_in_permitted_operations_is_blocked(self):
        record = self._build_record(
            sign_with_key=self.test_hmac_key,
            permitted_operations=["bids_validation"],
        )
        self._write_temp_register([record])
        service = IntakeService(self.external_root)

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                self._assert_blocked(
                    service,
                    {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"},
                    expected_reason_fragment=r"not permitted",
                    secrets_never_logged=[self.test_hmac_key, record["cryptographic_signature"]],
                )

    def test_08_missing_or_unreadable_register_is_blocked(self):
        service = IntakeService(self.external_root)
        spec = {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"}

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                if self.register_path.exists():
                    self.register_path.unlink()
                self._assert_blocked(
                    service,
                    spec,
                    expected_reason_fragment=r"not found|missing|unreadable",
                    secrets_never_logged=[self.test_hmac_key],
                )

                self.register_path.write_text("{not valid json", encoding="utf-8")
                self._assert_blocked(
                    service,
                    spec,
                    expected_reason_fragment=r"json|parse|unreadable|malformed",
                    secrets_never_logged=[self.test_hmac_key],
                )

                target = self.fake_project_root / "configs" / "real_target.json"
                target.write_text(REAL_REGISTER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
                self.register_path.unlink()
                self.register_path.symlink_to(target)
                self._assert_blocked(
                    service,
                    spec,
                    expected_reason_fragment=r"symlink",
                    secrets_never_logged=[self.test_hmac_key],
                )

                # Unreadable register (e.g. permission denied)
                self.register_path.unlink()
                self.register_path.write_text(REAL_REGISTER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
                real_open = open

                def _deny_register_open(file, *args, **kwargs):
                    if str(self.register_path) in str(file):
                        raise PermissionError("Permission denied")
                    return real_open(file, *args, **kwargs)

                with mock.patch("builtins.open", side_effect=_deny_register_open):
                    self._assert_blocked(
                        service,
                        spec,
                        expected_reason_fragment=r"permission|unreadable|parse",
                        secrets_never_logged=[self.test_hmac_key],
                    )

    def test_09_gate_exception_and_bypass_attempt_are_blocked(self):
        record = self._build_record(sign_with_key=self.test_hmac_key)
        self._write_temp_register([record])
        service = IntakeService(self.external_root)
        spec = {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"}

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                with mock.patch(
                    "src.intake.service.ResearchAuthorizationGate.verify_dataset_authorization",
                    side_effect=RuntimeError(f"unexpected failure with {self.test_hmac_key}"),
                ):
                    self._assert_blocked(
                        service,
                        spec,
                        expected_reason_fragment=r"runtimeerror",
                        secrets_never_logged=[self.test_hmac_key],
                    )

                with mock.patch(
                    "src.intake.service.ResearchAuthorizationGate",
                    side_effect=OSError("gate constructor crash"),
                ):
                    self._assert_blocked(
                        service,
                        spec,
                        expected_reason_fragment=r"oserror",
                        secrets_never_logged=[self.test_hmac_key],
                    )

                with mock.patch.dict(os.environ, {"NEUROADHD_BYPASS_AUTH_GATE": "1"}):
                    self._assert_blocked(
                        service,
                        spec,
                        expected_reason_fragment=r"bypass",
                        secrets_never_logged=[self.test_hmac_key, record["cryptographic_signature"]],
                    )

    def test_10_missing_signature_is_blocked(self):
        record = self._build_record(sign_with_key=None, cryptographic_signature=None)
        self._write_temp_register([record])
        service = IntakeService(self.external_root)

        with mock.patch("src.intake.service.PROJECT_ROOT", self.fake_project_root):
            with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": self.test_hmac_key}):
                self._assert_blocked(
                    service,
                    {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"},
                    expected_reason_fragment=r"signature is missing",
                    secrets_never_logged=[self.test_hmac_key],
                )

    def test_11_serve_dashboard_never_passes_authorization_checker_and_inspect_stays_allowed(self):
        serve_path = REPO_ROOT / "scripts" / "serve_dashboard.py"
        tree = ast.parse(serve_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "IntakeService":
                kw_names = [kw.arg for kw in node.keywords]
                self.assertNotIn("authorization_checker", kw_names)
                self.assertLessEqual(len(node.args), 1)

        import scripts.serve_dashboard as serve_dashboard

        with mock.patch.object(serve_dashboard, "INTAKE_ROOT", self.external_root):
            with mock.patch.object(serve_dashboard, "OUTPUT_ROOT", self.base_tmp / "dashboard_cohorts"):
                app = serve_dashboard.DashboardApplication()
        self.assertIsNone(app.intake._authorization_checker)

        # Inspect-only stays allowed (lists series metadata without writing NIfTI images)
        session_id, _, _ = self._new_staged_dicom_session(app.intake)
        mock_series_json = [
            {
                "SeriesNumber": 1,
                "CRC": "1001",
                "SeriesDescription": "MPRAGE_T1",
                "ProtocolName": "t1_mprage",
                "Modality": "MR",
                "Files": 176,
                "Rows": 256,
                "Columns": 256,
            }
        ]

        def fake_inspect(args, **kwargs):
            if "-n" in args and "-2" in args:
                out_dir = Path(args[args.index("-o") + 1])
                (out_dir / "dcm2niix_series.json").write_text(json.dumps(mock_series_json), encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with mock.patch("src.intake.service.shutil.which", return_value="/opt/homebrew/bin/dcm2niix"):
            with mock.patch("src.intake.service.subprocess.run", side_effect=fake_inspect):
                insp = app.intake.inspect_dicom_series(session_id)
        self.assertEqual(insp["series_count"], 1)
        self.assertEqual(list(app.intake.bids_dir.rglob("*.nii*")), [])

        # Conversion via DashboardApplication intake remains blocked against real register
        with self.assertRaisesRegex(IntakeError, "Conversion blocked"):
            app.intake.convert_dicom_to_bids(
                session_id,
                {"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"},
            )

        # HTTP endpoint POST /api/intake/sessions/<id>/convert_dicom must fail closed with clear error
        with mock.patch.object(serve_dashboard, "APP", app):
            server = ThreadingHTTPServer(("127.0.0.1", 0), serve_dashboard.DashboardHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/intake/sessions/{session_id}/convert_dicom"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({"subject_id": "sub-01", "t1w_series": "1", "bold_series": "2"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as http_ctx:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(http_ctx.exception.code, 400)
                body = json.loads(http_ctx.exception.read().decode("utf-8"))
                self.assertIn(
                    "Conversion blocked: research authorization not approved for this dataset",
                    body["error"],
                )
            finally:
                server.shutdown()
                server.server_close()

        # Verify frontend/app.js surfaces the message without hiding the convert button
        app_js = (REPO_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        self.assertIn("Conversion blocked: research authorization not approved for this dataset", app_js)
        self.assertNotIn("btn.classList.add(\"hidden\")", app_js)


if __name__ == "__main__":
    unittest.main()
