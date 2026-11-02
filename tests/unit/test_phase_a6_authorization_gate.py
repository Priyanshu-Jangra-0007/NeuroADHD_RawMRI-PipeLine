"""Unit tests for Phase A.6 Research Authorization Gate & Supervisor Feedback Validation Engine."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.run_poc_v2 import check_safety_guards
from src.validation.authorization_gate import (
    AuthorizationBypassAttemptError,
    AuthorizationDeniedError,
    AuthorizationError,
    AuthorizationSchemaError,
    ResearchAuthorizationGate,
)


class TestPhaseA6AuthorizationGate(unittest.TestCase):
    """Test suite verifying fail-closed research authorization, evidence validation, and bypass prevention."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name).resolve()

        # Create dummy evidence directory
        self.evidence_dir = self.base_dir / "docs" / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # Create sample valid evidence file
        self.sample_evidence_file = self.evidence_dir / "sample_iec_approval.pdf"
        self.sample_evidence_file.write_bytes(b"%PDF-1.4 Mock IEC Ethical Clearance Letter Ref #2026/IEC/001")
        self.sample_evidence_sha256 = hashlib.sha256(self.sample_evidence_file.read_bytes()).hexdigest()

        # Copy authentic register and schema into tmp environment
        repo_root = Path(__file__).resolve().parents[2]
        self.schema_source = repo_root / "configs" / "authorization_register_schema.json"
        self.register_source = repo_root / "configs" / "research_authorization_register.json"

        self.configs_dir = self.base_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.schema_path = self.configs_dir / "authorization_register_schema.json"
        self.schema_path.write_text(self.schema_source.read_text(encoding="utf-8"), encoding="utf-8")

        self.register_path = self.configs_dir / "research_authorization_register.json"
        self.register_path.write_text(self.register_source.read_text(encoding="utf-8"), encoding="utf-8")

        self.gate = ResearchAuthorizationGate(
            register_path=self.register_path,
            schema_path=self.schema_path,
            base_dir=self.base_dir,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()
        # Clean up any potential bypass env vars
        for var in [
            "NEUROADHD_ALLOW_REAL_SCANS",
            "NEUROADHD_BYPASS_AUTH_GATE",
            "NEUROADHD_BYPASS_SAFETY",
            "NEUROADHD_OVERRIDE_SAFETY",
            "NEUROADHD_FORCE_REAL_DATA",
            "NEUROADHD_DISABLE_AUTH_GATE",
        ]:
            os.environ.pop(var, None)

    # 1. Baseline & Synthetic-Only Execution Tests
    def test_default_register_all_pending_results_in_denied(self):
        """Current baseline register with all entries PENDING must evaluate to real_data_authorized=False."""
        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertEqual(report["overall_authorization_state"], "NOT_AUTHORIZED")
        self.assertEqual(report["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertGreater(report["total_items"], 0)
        self.assertEqual(report["status_counts"]["PENDING"], report["total_items"])
        self.assertEqual(report["status_counts"]["APPROVED"], 0)

        # Checking execution for real data must fail closed
        with self.assertRaises(AuthorizationDeniedError) as ctx:
            self.gate.check_execution_allowed(requested_mode="real")
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("NOT_AUTHORIZED", str(ctx.exception))

    def test_synthetic_only_execution_is_allowed(self):
        """Synthetic execution is explicitly permitted under the fail-closed gate."""
        decision = self.gate.check_execution_allowed(requested_mode="synthetic", allow_real_scans=False)
        self.assertEqual(decision["authorized_mode"], "synthetic")
        self.assertFalse(decision["real_data_authorized"])
        self.assertEqual(decision["overall_authorization_state"], "NOT_AUTHORIZED")
        self.assertEqual(decision["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")

    # 2. Supervisor Approval Missing Tests
    def test_supervisor_approval_missing_denies_real_data(self):
        """If supervisor approvals remain PENDING, real data access must be denied."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Approve institutional items but leave supervisor items PENDING
        for item in data["categories"]["institutional_legal"]:
            item["status"] = "APPROVED"
            item["responsible_reviewer"] = "Institutional Reviewer"
            item["decision_date"] = "2026-11-01"
            item["selected_option"] = "OPTION_APPROVED"
            item["evidence_reference"] = "docs/evidence/sample_iec_approval.pdf"
            item["evidence_sha256"] = self.sample_evidence_sha256

        data["real_data_authorized"] = True
        data["overall_authorization_state"] = "FULLY_AUTHORIZED"
        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        with self.assertRaises(AuthorizationDeniedError):
            self.gate.check_execution_allowed(allow_real_scans=True)

    # 3. Institutional Ethics Committee (IEC) Clearance Missing Tests
    def test_iec_approval_missing_denies_real_data(self):
        """Missing or unapproved IEC clearance must strictly block real data authorization."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Find AUTH-IEC-01 and ensure it is PENDING while others are approved
        for cat in data["categories"].values():
            for item in cat:
                if item["id"] == "AUTH-IEC-01":
                    item["status"] = "PENDING"
                else:
                    item["status"] = "APPROVED"
                    item["responsible_reviewer"] = "Authorized Signatory"
                    item["decision_date"] = "2026-11-01"
                    item["selected_option"] = "OPTION_APPROVED"
                    item["evidence_reference"] = "docs/evidence/sample_iec_approval.pdf"
                    item["evidence_sha256"] = self.sample_evidence_sha256

        data["real_data_authorized"] = True
        data["overall_authorization_state"] = "FULLY_AUTHORIZED"
        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertEqual(report["status_counts"]["PENDING"], 1)
        with self.assertRaises(AuthorizationDeniedError) as ctx:
            self.gate.check_execution_allowed(allow_real_scans=True)
        self.assertIn("Pending supervisor/governance decisions: 1", str(ctx.exception))

    # 4. Data Use Agreement (DUA) Missing Tests
    def test_dua_missing_denies_real_data(self):
        """Missing or unapproved DUA (AUTH-DUA-02) must strictly block real data authorization."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cat in data["categories"].values():
            for item in cat:
                if item["id"] == "AUTH-DUA-02":
                    item["status"] = "REJECTED"
                else:
                    item["status"] = "APPROVED"
                    item["responsible_reviewer"] = "Legal Officer"
                    item["decision_date"] = "2026-11-01"
                    item["selected_option"] = "OPTION_APPROVED"
                    item["evidence_reference"] = "docs/evidence/sample_iec_approval.pdf"
                    item["evidence_sha256"] = self.sample_evidence_sha256

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        with self.assertRaises(AuthorizationDeniedError):
            self.gate.check_execution_allowed(allow_real_scans=True)

    # 5. Missing Evidence File Tests
    def test_missing_evidence_file_denies_real_data(self):
        """An item marked APPROVED with a non-existent evidence file must fail verification."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Mark item as APPROVED but reference non-existent file
        data["categories"]["supervisor_decisions"][0]["status"] = "APPROVED"
        data["categories"]["supervisor_decisions"][0]["responsible_reviewer"] = "Dr. Supervisor"
        data["categories"]["supervisor_decisions"][0]["decision_date"] = "2026-11-01"
        data["categories"]["supervisor_decisions"][0]["selected_option"] = "OPTION_B"
        data["categories"]["supervisor_decisions"][0]["evidence_reference"] = "docs/evidence/non_existent_file.pdf"

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        found_missing_ev_issue = any("does not exist" in iss for iss in report["issues"])
        self.assertTrue(found_missing_ev_issue, "Must identify missing evidence file in issues list")

    # 6. Evidence SHA-256 Hash Mismatch Tests
    def test_evidence_hash_mismatch_denies_real_data(self):
        """Tampered evidence file with mismatched SHA-256 must fail verification."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["categories"]["supervisor_decisions"][0]["status"] = "APPROVED"
        data["categories"]["supervisor_decisions"][0]["responsible_reviewer"] = "Dr. Supervisor"
        data["categories"]["supervisor_decisions"][0]["decision_date"] = "2026-11-01"
        data["categories"]["supervisor_decisions"][0]["selected_option"] = "OPTION_B"
        data["categories"]["supervisor_decisions"][0]["evidence_reference"] = "docs/evidence/sample_iec_approval.pdf"
        data["categories"]["supervisor_decisions"][0]["evidence_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        found_sha_mismatch = any("Evidence SHA-256 mismatch" in iss for iss in report["issues"])
        self.assertTrue(found_sha_mismatch, "Must report SHA-256 mismatch")

    # 7. Invalid Approval Status & Malformed Register Tests
    def test_invalid_approval_status_denies_authorization(self):
        """Register containing unsupported status strings must fail validation."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["categories"]["supervisor_decisions"][0]["status"] = "CONDITIONALLY_ALLOWED"
        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(AuthorizationSchemaError):
            self.gate.evaluate_register()

    def test_malformed_json_fails_closed(self):
        """A corrupt or unparseable JSON file must fail closed immediately."""
        self.register_path.write_text("{ this is malformed JSON ...", encoding="utf-8")
        with self.assertRaises(AuthorizationSchemaError):
            self.gate.evaluate_register()

    # 8. Conflicting Decisions Tests
    def test_conflicting_decision_comments_rejected(self):
        """Status APPROVED with contradictory rejection comments must fail verification."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["categories"]["supervisor_decisions"][0]["status"] = "APPROVED"
        data["categories"]["supervisor_decisions"][0]["responsible_reviewer"] = "Dr. Supervisor"
        data["categories"]["supervisor_decisions"][0]["decision_date"] = "2026-11-01"
        data["categories"]["supervisor_decisions"][0]["selected_option"] = "OPTION_B"
        data["categories"]["supervisor_decisions"][0]["evidence_reference"] = "docs/evidence/sample_iec_approval.pdf"
        data["categories"]["supervisor_decisions"][0]["evidence_sha256"] = self.sample_evidence_sha256
        data["categories"]["supervisor_decisions"][0]["reviewer_comments"] = "This protocol is rejected due to safety concerns."

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("Conflicting decision" in iss for iss in report["issues"]))

    def test_conflicting_root_claims_rejected(self):
        """Root claiming real_data_authorized=True while items are PENDING must be rejected."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["real_data_authorized"] = True
        data["overall_authorization_state"] = "FULLY_AUTHORIZED"
        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("CONFLICT DETECTED" in iss for iss in report["issues"]))

    # 9. Unauthorized Environment & CLI Overrides
    def test_unauthorized_env_override_rejected(self):
        """Attempting to bypass authorization via environment variables must raise AuthorizationBypassAttemptError."""
        for var in [
            "NEUROADHD_ALLOW_REAL_SCANS",
            "NEUROADHD_BYPASS_AUTH_GATE",
            "NEUROADHD_BYPASS_SAFETY",
            "NEUROADHD_OVERRIDE_SAFETY",
            "NEUROADHD_FORCE_REAL_DATA",
        ]:
            os.environ[var] = "1"
            with self.assertRaises(AuthorizationBypassAttemptError) as ctx:
                self.gate.check_execution_allowed(requested_mode="synthetic")
            self.assertIn("Unauthorized safety bypass attempt detected", str(ctx.exception))
            os.environ.pop(var, None)

    def test_allow_real_scans_in_check_safety_guards_denied_by_gate(self):
        """Calling check_safety_guards with allow_real_scans=True must raise AuthorizationDeniedError."""
        with self.assertRaises(AuthorizationDeniedError) as ctx:
            check_safety_guards(
                subject_id="sub-synthetic001",
                allow_real_scans=True,
            )
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("allow_real_scans=True", str(ctx.exception))

    # 10. NOT_APPLICABLE Safeguard Tests
    def test_core_requirements_cannot_be_marked_not_applicable(self):
        """Core requirements (IEC clearance, DUA, De-ID) cannot be bypassed with NOT_APPLICABLE."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data["categories"]["institutional_legal"]:
            if item["id"] == "AUTH-IEC-01":
                item["status"] = "NOT_APPLICABLE"
                item["responsible_reviewer"] = "Investigator"
                item["reviewer_comments"] = "We claim IEC approval is not applicable for our study."

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("cannot be marked NOT_APPLICABLE" in iss for iss in report["issues"]))

    # 11. Genuine but Incomplete Approvals Tests
    def test_genuine_but_incomplete_approvals_denied(self):
        """Register where 22 out of 23 items are approved but 1 is pending must be denied."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Approve all except the last item
        all_items = []
        for cat in data["categories"].values():
            all_items.extend(cat)

        for it in all_items[:-1]:
            it["status"] = "APPROVED"
            it["responsible_reviewer"] = "Signatory"
            it["decision_date"] = "2026-11-01"
            it["selected_option"] = "OPTION_APPROVED"
            it["evidence_reference"] = "docs/evidence/sample_iec_approval.pdf"
            it["evidence_sha256"] = self.sample_evidence_sha256

        all_items[-1]["status"] = "PENDING"

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertEqual(report["status_counts"]["PENDING"], 1)
        with self.assertRaises(AuthorizationDeniedError):
            self.gate.check_execution_allowed(allow_real_scans=True)

    # 12. Audit Trail Logging Tests
    def test_audit_trail_logging_and_phi_rejection(self):
        """Audit trail must record valid governance events and reject PHI/PII strings."""
        self.gate.record_audit_event(
            event="SUPERVISOR_REVIEW_SUBMITTED",
            target_id="DEC-STC-01",
            previous_state="PENDING",
            new_state="PENDING",
            changed_by="Research Coordinator",
            reason="Submitted slice-timing recommendation for supervisor evaluation.",
            evidence_reference="docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md",
        )

        with open(self.register_path, "r", encoding="utf-8") as f:
            updated = json.load(f)

        self.assertGreaterEqual(len(updated["audit_trail"]), 2)
        last_event = updated["audit_trail"][-1]
        self.assertEqual(last_event["event"], "SUPERVISOR_REVIEW_SUBMITTED")
        self.assertEqual(last_event["target_id"], "DEC-STC-01")

        # Attempting to log PHI must fail
        with self.assertRaises(ValueError) as ctx:
            self.gate.record_audit_event(
                event="AUDIT_NOTE",
                target_id="SAFE-DEID-01",
                previous_state="PENDING",
                new_state="PENDING",
                changed_by="Investigator",
                reason="Patient name John Doe hospital record updated.",
            )
        self.assertIn("PHI/PII keyword", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
