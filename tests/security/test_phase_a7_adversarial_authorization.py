"""Independent Adversarial Authorization & Security Test Suite for Phase A.7.

Verifies that no supported execution path, environment variable, CLI argument,
symlink substitution, or path traversal can bypass the research authorization gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_poc_v2 import check_safety_guards
from src.parcellation.extractor import CANONICAL_CC200_SHA256, verify_canonical_cc200_atlas
from src.validation.authorization_gate import (
    AuthorizationBypassAttemptError,
    AuthorizationDeniedError,
    AuthorizationError,
    AuthorizationSchemaError,
    ResearchAuthorizationGate,
)


class TestPhaseA7AdversarialAuthorization(unittest.TestCase):
    """Adversarial security audit test harness for Phase A.7."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name).resolve()

        # Create structured evidence directory
        self.evidence_dir = self.base_dir / "docs" / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # Create authentic sample evidence file
        self.sample_evidence = self.evidence_dir / "authentic_iec_letter.pdf"
        self.sample_evidence.write_bytes(b"%PDF-1.4 Institutional Ethics Committee Formal Clearance Letter #2026/01")
        self.sample_sha256 = hashlib.sha256(self.sample_evidence.read_bytes()).hexdigest()

        # Copy authentic register and schema into sandbox
        repo_root = Path(__file__).resolve().parents[2]
        self.configs_dir = self.base_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        schema_src = repo_root / "configs" / "authorization_register_schema.json"
        self.schema_path = self.configs_dir / "authorization_register_schema.json"
        self.schema_path.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

        reg_src = repo_root / "configs" / "research_authorization_register.json"
        self.register_path = self.configs_dir / "research_authorization_register.json"
        self.register_path.write_text(reg_src.read_text(encoding="utf-8"), encoding="utf-8")

        self.gate = ResearchAuthorizationGate(
            register_path=self.register_path,
            schema_path=self.schema_path,
            base_dir=self.base_dir,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()
        # Clean any potential bypass variables
        for k in list(os.environ.keys()):
            if "NEUROADHD" in k.upper() and ("BYPASS" in k.upper() or "REAL" in k.upper() or "OVERRIDE" in k.upper()):
                os.environ.pop(k, None)

    # 1. Environment-Variable Bypass Tests
    def test_case_insensitive_and_alternative_truthy_env_bypasses(self):
        """Security Property: Prohibited environment overrides in any case or truthy form must fail closed."""
        bypass_test_cases = [
            ("neuroadhd_allow_real_scans", "1"),
            ("NEUROADHD_ALLOW_REAL_SCANS", "true"),
            ("NeuroADHD_Bypass_Auth_Gate", "yes"),
            ("NEUROADHD_BYPASS_SAFETY", "force"),
            ("neuroadhd_override_safety", "on"),
            ("NEUROADHD_FORCE_REAL_DATA", "enabled"),
            ("NEUROADHD_AUTH_OVERRIDE", "True"),
            ("NEUROADHD_BYPASS_SAFETY", " 1 "),
        ]
        for env_key, env_val in bypass_test_cases:
            os.environ[env_key] = env_val
            with self.assertRaises(AuthorizationBypassAttemptError, msg=f"Failed to catch bypass: {env_key}={env_val}") as ctx:
                self.gate.check_for_bypass_attempts()
            self.assertIn("Unauthorized safety bypass attempt detected", str(ctx.exception))
            os.environ.pop(env_key, None)

    def test_falsy_env_vars_do_not_trigger_false_positive(self):
        """Security Property: Explicitly falsy environment values must not falsely raise bypass error."""
        falsy_cases = [
            ("NEUROADHD_ALLOW_REAL_SCANS", "0"),
            ("NEUROADHD_ALLOW_REAL_SCANS", "false"),
            ("NEUROADHD_BYPASS_SAFETY", "no"),
            ("NEUROADHD_BYPASS_SAFETY", "off"),
            ("NEUROADHD_BYPASS_AUTH_GATE", "none"),
            ("NEUROADHD_BYPASS_AUTH_GATE", ""),
        ]
        for env_key, env_val in falsy_cases:
            os.environ[env_key] = env_val
            # Should not raise
            self.gate.check_for_bypass_attempts()
            os.environ.pop(env_key, None)

    # 2. Path Traversal & Boundary Escape Tests
    def test_path_traversal_in_evidence_reference_rejected(self):
        """Security Property: Evidence references attempting path traversal (..) must be rejected."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["supervisor_decisions"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_B"
        item["evidence_reference"] = "../../etc/passwd"
        item["evidence_sha256"] = "a" * 64

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("Path traversal detected" in iss for iss in report["issues"]))

    def test_absolute_path_outside_repository_rejected(self):
        """Security Property: Evidence references to arbitrary absolute system paths must be rejected."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["supervisor_decisions"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_B"
        item["evidence_reference"] = "/etc/hosts"
        item["evidence_sha256"] = "b" * 64

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("Path traversal detected" in iss for iss in report["issues"]))

    # 3. Symlink Substitution & Intermediate Symlink Directory Tests
    def test_symlink_evidence_file_rejected(self):
        """Security Property: Symlink substitution of an evidence file must be detected and rejected."""
        symlink_target = self.sample_evidence
        symlink_file = self.evidence_dir / "symlinked_approval.pdf"
        if symlink_file.exists():
            symlink_file.unlink()
        symlink_file.symlink_to(symlink_target)

        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["institutional_legal"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_APPROVED"
        item["evidence_reference"] = "docs/evidence/symlinked_approval.pdf"
        item["evidence_sha256"] = self.sample_sha256

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("Symlink substitution detected" in iss for iss in report["issues"]))

    def test_symlinked_parent_directory_rejected(self):
        """Security Property: Evidence within a symlinked directory must be rejected."""
        real_sub = self.base_dir / "real_storage"
        real_sub.mkdir(parents=True, exist_ok=True)
        real_file = real_sub / "document.pdf"
        real_file.write_bytes(b"content")

        symlinked_dir = self.base_dir / "docs" / "linked_dir"
        if symlinked_dir.exists():
            symlinked_dir.unlink()
        symlinked_dir.symlink_to(real_sub)

        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["institutional_legal"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_APPROVED"
        item["evidence_reference"] = "docs/linked_dir/document.pdf"
        item["evidence_sha256"] = hashlib.sha256(b"content").hexdigest()

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("Symlink substitution detected" in iss for iss in report["issues"]))

    # 4. Evidence File Type & Regular File Assertion Tests
    def test_directory_as_evidence_rejected(self):
        """Security Property: Referencing a directory as documentary evidence must be rejected."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["supervisor_decisions"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_B"
        item["evidence_reference"] = "docs/evidence"  # directory, not file
        item["evidence_sha256"] = "c" * 64

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("must be a regular file" in iss for iss in report["issues"]))

    def test_unsupported_evidence_file_extension_rejected(self):
        """Security Property: Non-document file extensions (.sh, .exe, .py) must be rejected."""
        bad_file = self.evidence_dir / "approval_script.sh"
        bad_file.write_bytes(b"#!/bin/bash\necho approved\n")

        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["supervisor_decisions"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_B"
        item["evidence_reference"] = "docs/evidence/approval_script.sh"
        item["evidence_sha256"] = hashlib.sha256(bad_file.read_bytes()).hexdigest()

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("extension '.sh' is not permitted" in iss for iss in report["issues"]))

    # 5. Mandatory SHA-256 for Approved Items
    def test_approved_item_missing_mandatory_sha256_rejected(self):
        """Security Property: An approved item without a 64-character hex SHA-256 digest must fail."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data["categories"]["supervisor_decisions"][0]
        item["status"] = "APPROVED"
        item["responsible_reviewer"] = "Reviewer"
        item["decision_date"] = "2026-11-01"
        item["selected_option"] = "OPTION_B"
        item["evidence_reference"] = "docs/evidence/authentic_iec_letter.pdf"
        item["evidence_sha256"] = None  # Missing mandatory hash

        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        report = self.gate.evaluate_register()
        self.assertFalse(report["real_data_authorized"])
        self.assertTrue(any("requires a valid 64-character hex 'evidence_sha256'" in iss for iss in report["issues"]))

    # 6. Alternate Entrypoint Hardening
    def test_alternate_entrypoint_run_single_scan_poc_fails_closed(self):
        """Security Property: Alternate legacy entrypoint scripts/run_single_scan_poc.py must fail closed."""
        from scripts.run_single_scan_poc import run_poc
        with self.assertRaises(AuthorizationDeniedError) as ctx:
            run_poc()
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("NOT_AUTHORIZED", str(ctx.exception))

    # 7. Audit Log Injection & Sanitization
    def test_audit_log_newline_injection_rejected(self):
        """Security Property: Newline injection in audit log fields must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.gate.record_audit_event(
                event="AUDIT_EVENT\nINJECTED_LINE: fake",
                target_id="DEC-STC-01",
                previous_state="PENDING",
                new_state="PENDING",
                changed_by="Reviewer",
                reason="Regular update",
            )
        self.assertIn("Log injection detected", str(ctx.exception))

    def test_audit_log_clinical_subject_keywords_rejected(self):
        """Security Property: Historical clinical identifiers (e.g. sub-2344857, KKI) in audit reason are blocked."""
        with self.assertRaises(ValueError) as ctx:
            self.gate.record_audit_event(
                event="AUDIT_EVENT",
                target_id="DEC-STC-01",
                previous_state="PENDING",
                new_state="PENDING",
                changed_by="Reviewer",
                reason="Attempting to process historical scan sub-2344857 from KKI",
            )
        self.assertIn("Restricted identifier or PHI/PII keyword", str(ctx.exception))

    # 8. Scientific Reproducibility & Model Immutability Verification
    def test_canonical_atlas_hash_reproducibility(self):
        """Scientific Property: Canonical CC200 atlas must strictly match authoritative SHA-256."""
        atlas_path = Path("data/atlases/cc200_roi_atlas.nii.gz")
        if atlas_path.exists():
            h = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
            self.assertEqual(h, CANONICAL_CC200_SHA256)
            verify_canonical_cc200_atlas(str(atlas_path))

    def test_register_schema_rejects_unknown_root_properties(self):
        """Security Property: Authorization register schema must strictly forbid additional properties."""
        with open(self.register_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["unauthorized_backdoor_field"] = "bypass_true"
        with open(self.register_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(AuthorizationSchemaError) as ctx:
            self.gate.evaluate_register()
        self.assertIn("failed schema validation", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
