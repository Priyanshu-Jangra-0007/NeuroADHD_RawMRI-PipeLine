"""Unit tests for Step 2: Research Authorization Gate & Cryptographic Signature Validation.

Covers the 12 required test scenarios:
1. Valid synthetic authorization
2. Invalid HMAC signature
3. Modified authorization register (tampered canonical fields)
4. Expired authorization
5. Missing research approval reference
6. Unauthorized dataset identifier
7. Unauthorized processing operation
8. Missing HMAC key
9. Invalid signature format
10. Attempted authorization bypass
11. Authorization revocation
12. Default synthetic-only execution
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import unittest

from src.validation.authorization_gate import (
    AuthorizationBypassAttemptError,
    AuthorizationDeniedError,
    AuthorizationError,
    AuthorizationSchemaError,
    ResearchAuthorizationGate,
)


class TestStep2ResearchAuthorizationGate(unittest.TestCase):
    """Exhaustive test suite for dataset-level authorization and HMAC-SHA256 validation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name).resolve()
        self.configs_dir = self.base_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[2]
        self.schema_source = repo_root / "configs" / "authorization_register_schema.json"
        self.register_source = repo_root / "configs" / "research_authorization_register.json"

        self.schema_path = self.configs_dir / "authorization_register_schema.json"
        self.schema_path.write_text(self.schema_source.read_text(encoding="utf-8"), encoding="utf-8")

        self.register_path = self.configs_dir / "research_authorization_register.json"
        self.register_path.write_text(self.register_source.read_text(encoding="utf-8"), encoding="utf-8")

        self.gate = ResearchAuthorizationGate(
            register_path=self.register_path,
            schema_path=self.schema_path,
            base_dir=self.base_dir,
        )

        self.test_hmac_key = "test-secret-key-32-bytes-long-for-hmac!!"
        os.environ["NEUROADHD_HMAC_KEY"] = self.test_hmac_key

        # Valid baseline synthetic dataset authorization record template
        self.valid_dataset_record = {
            "authorization_id": "AUTH-DS-SYNTH-001",
            "project_id": "NeuroADHD RawMRI PoC V2",
            "dataset_id": "sub-synth01",
            "approved_purpose": "Synthetic phantom verification and automated CC200 validation.",
            "institutional_approval_reference": "REF-IEC-2026-SYNTH-001",
            "approval_date": "2026-01-01",
            "expiry_date": "2027-12-31",
            "permitted_operations": [
                "dicom_conversion",
                "bids_validation",
                "preprocessing",
                "cc200_extraction",
            ],
            "authorized_personnel": "Principal Investigator & Authorized Research Personnel",
            "authorization_status": "APPROVED",
            "cryptographic_signature": None,
            "notes": "Synthetic testing dataset record.",
        }
        # Compute valid signature
        sig = self.gate.compute_authorization_signature(self.valid_dataset_record, key=self.test_hmac_key)
        self.valid_dataset_record["cryptographic_signature"] = sig

    def tearDown(self):
        self.tmp_dir.cleanup()
        for k in (
            "NEUROADHD_ALLOW_REAL_SCANS",
            "NEUROADHD_BYPASS_AUTH_GATE",
            "NEUROADHD_BYPASS_SAFETY",
            "NEUROADHD_OVERRIDE_SAFETY",
            "NEUROADHD_FORCE_REAL_DATA",
            "NEUROADHD_DISABLE_AUTH_GATE",
            "NEUROADHD_AUTH_OVERRIDE",
            "NEUROADHD_BYPASS_IEC",
        ):
            os.environ.pop(k, None)

    # 1. Valid synthetic authorization
    def test_01_valid_synthetic_authorization(self):
        """A properly signed and configured synthetic authorization record must verify cleanly."""
        res = self.gate.verify_dataset_authorization(
            self.valid_dataset_record,
            key=self.test_hmac_key,
            current_date=datetime.date(2026, 6, 1),
        )
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["signature_status"], "VALID")
        self.assertEqual(res["expiry_status"], "VALID_PERIOD")
        self.assertEqual(len(res["issues"]), 0)

    # 2. Invalid HMAC signature
    def test_02_invalid_hmac_signature(self):
        """A tampered HMAC signature must be detected and rejected."""
        tampered = copy.deepcopy(self.valid_dataset_record)
        tampered["cryptographic_signature"] = "a" * 64  # Invalid signature

        res = self.gate.verify_dataset_authorization(
            tampered,
            key=self.test_hmac_key,
            current_date=datetime.date(2026, 6, 1),
        )
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["signature_status"], "INVALID_SIGNATURE")
        self.assertTrue(any("signature mismatch" in iss.lower() for iss in res["issues"]))

    # 3. Modified authorization register (tampered fields invalidate signature)
    def test_03_modified_record_invalidates_signature(self):
        """Mutating any canonical field in the record must invalidate the HMAC signature."""
        fields_to_tamper = [
            ("approved_purpose", "Altered clinical purpose without approval"),
            ("dataset_id", "sub-02"),
            ("expiry_date", "2028-12-31"),
            ("institutional_approval_reference", "FAKE-REF-999"),
            ("permitted_operations", ["dicom_conversion"]),
            ("project_id", "Hacked Project"),
            ("authorized_personnel", "Unauthorized External Entity"),
            ("approval_date", "2025-01-01"),
        ]
        for field, new_val in fields_to_tamper:
            tampered = copy.deepcopy(self.valid_dataset_record)
            tampered[field] = new_val
            # Signature remains the original one computed before tampering
            res = self.gate.verify_dataset_authorization(
                tampered,
                key=self.test_hmac_key,
                current_date=datetime.date(2026, 6, 1),
            )
            self.assertFalse(res["is_valid"], f"Tampering field '{field}' must invalidate authorization")
            self.assertEqual(
                res["signature_status"],
                "INVALID_SIGNATURE",
                f"Tampering field '{field}' must produce INVALID_SIGNATURE",
            )

    # 4. Expired authorization
    def test_04_expired_authorization(self):
        """An authorization record past its expiry date must be rejected."""
        expired_record = copy.deepcopy(self.valid_dataset_record)
        expired_record["expiry_date"] = "2025-12-31"
        # Re-sign with the expired date
        expired_record["cryptographic_signature"] = self.gate.compute_authorization_signature(
            expired_record, key=self.test_hmac_key
        )

        # Evaluate against a date after expiry
        res = self.gate.verify_dataset_authorization(
            expired_record,
            key=self.test_hmac_key,
            current_date=datetime.date(2026, 6, 1),
        )
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["expiry_status"], "EXPIRED")
        self.assertTrue(any("expired" in iss.lower() for iss in res["issues"]))

    # 5. Missing research approval reference
    def test_05_missing_research_approval_reference(self):
        """Missing institutional approval reference must be rejected."""
        for bad_ref in [None, "", "   "]:
            rec = copy.deepcopy(self.valid_dataset_record)
            rec["institutional_approval_reference"] = bad_ref
            rec["cryptographic_signature"] = self.gate.compute_authorization_signature(rec, key=self.test_hmac_key)
            res = self.gate.verify_dataset_authorization(
                rec,
                key=self.test_hmac_key,
                current_date=datetime.date(2026, 6, 1),
            )
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("institutional_approval_reference" in iss.lower() for iss in res["issues"]))

    # 6. Unauthorized dataset identifier
    def test_06_unauthorized_dataset(self):
        """Requesting processing for a dataset ID different from the authorized ID must fail."""
        res = self.gate.verify_dataset_authorization(
            self.valid_dataset_record,
            requested_dataset="sub-unauthorized-99",
            key=self.test_hmac_key,
            current_date=datetime.date(2026, 6, 1),
        )
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("dataset id mismatch" in iss.lower() for iss in res["issues"]))

    # 7. Unauthorized processing operation
    def test_07_unauthorized_operation(self):
        """Requesting an operation not listed in permitted_operations must be rejected."""
        res = self.gate.verify_dataset_authorization(
            self.valid_dataset_record,
            requested_operation="clinical_diagnostic_inference",
            key=self.test_hmac_key,
            current_date=datetime.date(2026, 6, 1),
        )
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("not permitted" in iss.lower() for iss in res["issues"]))

    # 8. Missing HMAC key
    def test_08_missing_hmac_key(self):
        """When HMAC key is missing from environment and not supplied, verification must fail closed."""
        saved_hmac = os.environ.pop("NEUROADHD_HMAC_KEY", None)
        saved_integrity = os.environ.pop("NEUROADHD_INTEGRITY_KEY", None)
        try:
            with self.assertRaises(AuthorizationError):
                self.gate.get_hmac_key()

            res = self.gate.verify_dataset_authorization(
                self.valid_dataset_record,
                key=None,
                current_date=datetime.date(2026, 6, 1),
            )
            self.assertFalse(res["is_valid"])
            self.assertEqual(res["signature_status"], "MISSING_KEY")
        finally:
            if saved_hmac:
                os.environ["NEUROADHD_HMAC_KEY"] = saved_hmac
            if saved_integrity:
                os.environ["NEUROADHD_INTEGRITY_KEY"] = saved_integrity

    # 9. Invalid signature format
    def test_09_invalid_signature_format(self):
        """Malformed signatures (wrong length, non-hex chars, empty) must be rejected."""
        for bad_sig in [
            "not-a-valid-hex-signature",
            "12345",
            "z" * 64,  # Non-hex characters
            "a" * 63,  # Too short
            "a" * 65,  # Too long
            "",
            None,
        ]:
            rec = copy.deepcopy(self.valid_dataset_record)
            rec["cryptographic_signature"] = bad_sig
            res = self.gate.verify_dataset_authorization(
                rec,
                key=self.test_hmac_key,
                current_date=datetime.date(2026, 6, 1),
            )
            self.assertFalse(res["is_valid"])
            self.assertIn(res["signature_status"], ("INVALID_FORMAT", "MISSING_SIGNATURE"))

    # 10. Attempted authorization bypass
    def test_10_attempted_authorization_bypass(self):
        """Prohibited bypass environment variables must be trapped and rejected immediately."""
        bypass_attempts = [
            ("NEUROADHD_ALLOW_REAL_SCANS", "true"),
            ("NEUROADHD_BYPASS_AUTH_GATE", "1"),
            ("NEUROADHD_BYPASS_SAFETY", "yes"),
            ("NEUROADHD_OVERRIDE_SAFETY", "TRUE"),
            ("NEUROADHD_FORCE_REAL_DATA", "on"),
            ("NEUROADHD_DISABLE_AUTH_GATE", "1"),
            ("NEUROADHD_AUTH_OVERRIDE", "enabled"),
            ("NEUROADHD_BYPASS_IEC", "true"),
        ]
        for env_k, env_v in bypass_attempts:
            os.environ[env_k] = env_v
            try:
                with self.assertRaises(AuthorizationBypassAttemptError) as ctx:
                    self.gate.check_execution_allowed(requested_mode="synthetic")
                self.assertIn("Unauthorized safety bypass attempt detected", str(ctx.exception))
            finally:
                os.environ.pop(env_k, None)

    # 11. Authorization revocation
    def test_11_authorization_revocation(self):
        """Records with status REVOKED, REJECTED, EXPIRED, or PENDING must not be valid."""
        for rev_status in ["REVOKED", "REJECTED", "EXPIRED", "PENDING"]:
            rec = copy.deepcopy(self.valid_dataset_record)
            rec["authorization_status"] = rev_status
            rec["cryptographic_signature"] = self.gate.compute_authorization_signature(rec, key=self.test_hmac_key)
            res = self.gate.verify_dataset_authorization(
                rec,
                key=self.test_hmac_key,
                current_date=datetime.date(2026, 6, 1),
            )
            self.assertFalse(res["is_valid"])
            self.assertEqual(res["status"], rev_status)
            self.assertTrue(any(f"'{rev_status}', not 'APPROVED'" in iss for iss in res["issues"]))

    # 12. Default synthetic-only execution
    def test_12_default_synthetic_only_execution(self):
        """Default execution policy is synthetic only; real data execution is denied."""
        # 1. Default call
        decision = self.gate.check_execution_allowed()
        self.assertEqual(decision["authorized_mode"], "synthetic")
        self.assertFalse(decision["real_data_authorized"])
        self.assertEqual(decision["overall_authorization_state"], "NOT_AUTHORIZED")
        self.assertEqual(decision["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")

        # 2. Explicit request for real data on default register must fail closed
        with self.assertRaises(AuthorizationDeniedError) as ctx:
            self.gate.check_execution_allowed(requested_mode="real", allow_real_scans=True)
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("NOT_AUTHORIZED", str(ctx.exception))

    # Bonus: Constant-time comparison & PHI detection
    def test_13_phi_sanitization_in_dataset_record(self):
        """Dataset authorization records containing PHI keywords must be rejected."""
        for phi_term in ["patient", "hospital", "mrn", "birthdate", "physician"]:
            rec = copy.deepcopy(self.valid_dataset_record)
            rec["notes"] = f"Clinical scan from {phi_term} database."
            rec["cryptographic_signature"] = self.gate.compute_authorization_signature(rec, key=self.test_hmac_key)
            res = self.gate.verify_dataset_authorization(rec, key=self.test_hmac_key)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("phi/pii violation" in iss.lower() for iss in res["issues"]))

    def test_14_future_approval_date_rejected(self):
        """Approval date occurring in the future must be rejected."""
        rec = copy.deepcopy(self.valid_dataset_record)
        rec["approval_date"] = "2027-01-01"
        rec["cryptographic_signature"] = self.gate.compute_authorization_signature(rec, key=self.test_hmac_key)
        res = self.gate.verify_dataset_authorization(
            rec,
            key=self.test_hmac_key,
            current_date=datetime.date(2026, 6, 1),
        )
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("in the future" in iss.lower() for iss in res["issues"]))


if __name__ == "__main__":
    unittest.main()
