"""Unit tests for Step 8: Documentary Evidence Validation, Scoped Authorization & Two-Phase Readiness."""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from src.validation.authorization_evidence import (
    AuthorizationEvidenceError,
    AuthorizationEvidenceValidator,
    DEFAULT_PROTOCOL_TITLE,
    DEFAULT_PROTOCOL_VERSION,
)
from src.validation.authorization_gate import (
    AuthorizationDeniedError,
    ResearchAuthorizationGate,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestStep8AuthorizationEvidence(unittest.TestCase):
    """Comprehensive synthetic security & governance readiness tests (no real MRI access)."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name).resolve()
        self.configs_dir = self.base_dir / "configs"
        self.evidence_dir = self.base_dir / "docs" / "verified_evidence"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[2]
        self.schema_path = self.configs_dir / "authorization_register_schema.json"
        self.register_path = self.configs_dir / "research_authorization_register.json"
        self.schema_path.write_text(
            (repo_root / "configs" / "authorization_register_schema.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.register_path.write_text(
            (repo_root / "configs" / "research_authorization_register.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        self.gate = ResearchAuthorizationGate(
            register_path=self.register_path,
            schema_path=self.schema_path,
            base_dir=self.base_dir,
        )
        self.validator = AuthorizationEvidenceValidator(gate=self.gate, base_dir=self.base_dir)
        self.hmac_key = "step8-synthetic-test-hmac-secret-key-32bytes"
        self._saved_key = os.environ.get("NEUROADHD_HMAC_KEY")
        os.environ["NEUROADHD_HMAC_KEY"] = self.hmac_key
        self.ref_date = datetime.date(2026, 11, 12)

        # Synthetic non-placeholder documentary evidence files
        self.evidence_files = {}
        for cat in (
            "institutional_authorization",
            "ethics_iec",
            "dataset_dua",
            "privacy_infosec",
            "participant_authorization",
        ):
            p = self.evidence_dir / f"{cat}_signed.pdf"
            p.write_bytes(f"%PDF-1.4 Synthetic Authority Signed Record for {cat}".encode("utf-8"))
            self.evidence_files[cat] = (f"docs/verified_evidence/{p.name}", _sha256_file(p))

        self.complete_package_unverified = {
            "authority_verification_status": "PENDING_REVIEW",
            "evidence": {
                "institutional_authorization": {
                    "evidence_file": self.evidence_files["institutional_authorization"][0],
                    "evidence_sha256": self.evidence_files["institutional_authorization"][1],
                    "document_status": "APPROVED",
                    "institution_name": "Synthetic Biomedical Research Institute",
                    "approving_authority": "Dean of Research (Synthetic Test Fixture)",
                    "approval_reference": "INST-SYNTH-2026-001",
                    "effective_date": "2026-10-01",
                    "expiry_date": "2027-10-01",
                    "protocol_title": DEFAULT_PROTOCOL_TITLE,
                    "protocol_version": DEFAULT_PROTOCOL_VERSION,
                    "dataset_id": "sub-synth-pilot",
                    "study_tier": "retrospective_secondary",
                    "approved_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
                    "approved_participants": ["sub-synth-p01"],
                    "storage_conditions": "AES-256 isolated external volume outside repo",
                    "sharing_conditions": "Non-commercial research only; no raw MRI export",
                    "authority_verified": False,
                },
                "ethics_iec": {
                    "evidence_file": self.evidence_files["ethics_iec"][0],
                    "evidence_sha256": self.evidence_files["ethics_iec"][1],
                    "document_status": "APPROVED",
                    "institution_name": "Synthetic Biomedical Research Institute",
                    "approving_authority": "Synthetic Institutional Ethics Committee",
                    "approval_reference": "IEC-SYNTH-2026-01",
                    "effective_date": "2026-10-01",
                    "expiry_date": "2027-10-01",
                    "protocol_title": DEFAULT_PROTOCOL_TITLE,
                    "protocol_version": DEFAULT_PROTOCOL_VERSION,
                    "dataset_id": "sub-synth-pilot",
                    "study_tier": "retrospective_secondary",
                    "approved_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
                    "approved_participants": ["sub-synth-p01"],
                    "storage_conditions": "AES-256 encrypted local storage; zero cloud sync",
                    "sharing_conditions": "Restricted to authorized roster under secondary waiver",
                    "authority_verified": False,
                },
                "dataset_dua": {
                    "evidence_file": self.evidence_files["dataset_dua"][0],
                    "evidence_sha256": self.evidence_files["dataset_dua"][1],
                    "document_status": "EXECUTED",
                    "institution_name": "Synthetic Biomedical Research Institute",
                    "approving_authority": "Legal Office & Synthetic Repository Steward",
                    "approval_reference": "DUA-SYNTH-2026-01",
                    "effective_date": "2026-10-01",
                    "expiry_date": "2027-10-01",
                    "protocol_title": DEFAULT_PROTOCOL_TITLE,
                    "protocol_version": DEFAULT_PROTOCOL_VERSION,
                    "dataset_id": "sub-synth-pilot",
                    "study_tier": "retrospective_secondary",
                    "approved_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
                    "approved_participants": ["sub-synth-p01"],
                    "storage_conditions": "Controlled local directory, no onward redistribution",
                    "sharing_conditions": "CC-BY-NC non-commercial academic use with citation",
                    "non_commercial_confirmed": True,
                    "downloaded_only_inference": False,
                    "authority_verified": False,
                },
                "privacy_infosec": {
                    "evidence_file": self.evidence_files["privacy_infosec"][0],
                    "evidence_sha256": self.evidence_files["privacy_infosec"][1],
                    "document_status": "APPROVED",
                    "institution_name": "Synthetic Biomedical Research Institute",
                    "approving_authority": "Institutional DPO & Information Security Officer",
                    "approval_reference": "DPO-SEC-SYNTH-2026-01",
                    "effective_date": "2026-10-01",
                    "expiry_date": "2027-10-01",
                    "protocol_title": DEFAULT_PROTOCOL_TITLE,
                    "protocol_version": DEFAULT_PROTOCOL_VERSION,
                    "dataset_id": "sub-synth-pilot",
                    "study_tier": "retrospective_secondary",
                    "approved_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
                    "approved_participants": ["sub-synth-p01"],
                    "storage_conditions": "AES-256 disk, non-root container UID 10001, 2.5 GB limit",
                    "sharing_conditions": "Coded pseudonyms only; no raw header export",
                    "authority_verified": False,
                },
                "participant_authorization": {
                    "evidence_file": self.evidence_files["participant_authorization"][0],
                    "evidence_sha256": self.evidence_files["participant_authorization"][1],
                    "document_status": "WAIVED",
                    "institution_name": "Synthetic Biomedical Research Institute",
                    "approving_authority": "Synthetic Institutional Ethics Committee",
                    "approval_reference": "IEC-SYNTH-2026-01",
                    "effective_date": "2026-10-01",
                    "expiry_date": "2027-10-01",
                    "protocol_title": DEFAULT_PROTOCOL_TITLE,
                    "protocol_version": DEFAULT_PROTOCOL_VERSION,
                    "dataset_id": "sub-synth-pilot",
                    "study_tier": "retrospective_secondary",
                    "approved_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
                    "approved_participants": ["sub-synth-p01"],
                    "storage_conditions": "Coded session directory isolated outside repo",
                    "sharing_conditions": "No clinical reporting or onward raw sharing",
                    "authority_verified": False,
                },
            },
        }

        self.pilot_records = [
            {
                "participant_id": "sub-synth-p01",
                "authorization_status": "APPROVED",
                "protocol_version": DEFAULT_PROTOCOL_VERSION,
                "dataset_id": "sub-synth-pilot",
                "study_tier": "retrospective_secondary",
                "session_scope": ["ses-01"],
                "permitted_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
                "effective_date": "2026-10-01",
                "expiry_date": "2027-10-01",
                "evidence_references": ["IEC-SYNTH-2026-01", "DUA-SYNTH-2026-01"],
            }
        ]

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
        if self._saved_key is not None:
            os.environ["NEUROADHD_HMAC_KEY"] = self._saved_key
        else:
            os.environ.pop("NEUROADHD_HMAC_KEY", None)

    def _activate_synthetic_register_and_package(self, sign: bool = True, custom_sig: str | None = None) -> dict:
        """Create a fully signed synthetic fixture inside TemporaryDirectory ONLY."""
        item_ev_path = self.evidence_dir / "register_item_signed.pdf"
        item_ev_path.write_bytes(b"%PDF-1.4 Synthetic Register Item Signoff")
        item_sha = _sha256_file(item_ev_path)

        reg = json.loads(self.register_path.read_text(encoding="utf-8"))
        for cat in reg["categories"].values():
            for item in cat:
                item["status"] = "APPROVED"
                item["responsible_reviewer"] = "Synthetic Reviewer"
                item["decision_date"] = "2026-10-01"
                item["selected_option"] = "SYNTHETIC_FIXTURE_APPROVED"
                item["evidence_reference"] = "docs/verified_evidence/register_item_signed.pdf"
                item["evidence_sha256"] = item_sha

        ds_rec = {
            "authorization_id": "AUTH-DS-SYNTH-STEP8-001",
            "project_id": "NeuroADHD RawMRI PoC V2",
            "dataset_id": "sub-synth-pilot",
            "approved_purpose": "Synthetic Step 8 scoped authorization fixture.",
            "institutional_approval_reference": "IEC-SYNTH-2026-01",
            "approval_date": "2026-10-01",
            "expiry_date": "2027-10-01",
            "permitted_operations": ["bids_validation", "preprocessing", "cc200_extraction", "quality_control"],
            "authorized_personnel": "Synthetic Authorized Personnel",
            "authorization_status": "APPROVED",
            "cryptographic_signature": None,
            "notes": "Synthetic authorization test fixture only.",
        }
        if sign:
            ds_rec["cryptographic_signature"] = self.gate.compute_authorization_signature(
                ds_rec, key=self.hmac_key
            )
        if custom_sig is not None:
            ds_rec["cryptographic_signature"] = custom_sig

        reg["dataset_authorizations"] = [ds_rec]
        reg["overall_authorization_state"] = "FULLY_AUTHORIZED"
        reg["execution_policy"] = "AUTHORIZED_PILOT_ONLY"
        reg["real_data_authorized"] = True
        self.register_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

        pkg = copy.deepcopy(self.complete_package_unverified)
        pkg["authority_verification_status"] = "VERIFIED"
        for rec in pkg["evidence"].values():
            rec["authority_verified"] = True
        return pkg

    def test_baseline_repo_remains_pending_and_not_authorized(self) -> None:
        """Default production register and missing evidence package must stay locked."""
        repo_root = Path(__file__).resolve().parents[2]
        prod_validator = AuthorizationEvidenceValidator(base_dir=repo_root)
        res = prod_validator.evaluate_evidence_package(package=None)
        self.assertEqual(res["readiness_state"], "PENDING_EVIDENCE")
        self.assertEqual(res["overall_authorization_state"], "NOT_AUTHORIZED")
        self.assertEqual(res["execution_policy"], "STRICTLY_SYNTHETIC_ONLY")
        self.assertFalse(res["real_data_authorized"])
        self.assertFalse(res["execution_permitted"])

    def test_valid_synthetic_signed_scoped_authorization(self) -> None:
        """Complete authority-verified synthetic evidence + valid HMAC register unlocks exact scope."""
        pkg = self._activate_synthetic_register_and_package(sign=True)
        res = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p01",
            session_id="ses-01",
            requested_operation="preprocessing",
            protocol_version=DEFAULT_PROTOCOL_VERSION,
            dataset_id="sub-synth-pilot",
            study_tier="retrospective_secondary",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertTrue(res["authorized"])
        self.assertEqual(res["readiness_state"], "APPROVED_SCOPED")
        self.assertEqual(res["signature_status"], "VALID")

    def test_missing_signature_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=False)
        res = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p01",
            session_id="ses-01",
            requested_operation="preprocessing",
            dataset_id="sub-synth-pilot",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res["authorized"])
        self.assertEqual(res["signature_status"], "MISSING_SIGNATURE")

    def test_incorrect_signature_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=False, custom_sig="b" * 64)
        res = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p01",
            session_id="ses-01",
            requested_operation="preprocessing",
            dataset_id="sub-synth-pilot",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res["authorized"])
        self.assertEqual(res["signature_status"], "INVALID_SIGNATURE")

    def test_modified_approval_fields_and_evidence_tamper_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=True)
        # Tamper register approval field after signing
        reg = json.loads(self.register_path.read_text(encoding="utf-8"))
        reg["dataset_authorizations"][0]["expiry_date"] = "2030-12-31"
        self.register_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

        res = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p01",
            session_id="ses-01",
            requested_operation="preprocessing",
            dataset_id="sub-synth-pilot",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res["authorized"])
        self.assertEqual(res["signature_status"], "INVALID_SIGNATURE")

        # Tamper evidence file hash
        pkg2 = copy.deepcopy(pkg)
        (self.evidence_dir / "ethics_iec_signed.pdf").write_bytes(b"%PDF-1.4 TAMPERED")
        eval2 = self.validator.evaluate_evidence_package(
            package=pkg2,
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(eval2["execution_permitted"])
        self.assertEqual(eval2["readiness_state"], "EVIDENCE_INCOMPLETE")

    def test_expired_authorization_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=True)
        pkg["evidence"]["ethics_iec"]["expiry_date"] = "2026-10-31"
        eval_res = self.validator.evaluate_evidence_package(
            package=pkg,
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(eval_res["execution_permitted"])
        self.assertEqual(eval_res["readiness_state"], "EXPIRED")

    def test_revoked_authorization_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=True)
        pkg["evidence"]["ethics_iec"]["document_status"] = "REVOKED"
        pkg["evidence"]["ethics_iec"]["revoked"] = True
        eval_res = self.validator.evaluate_evidence_package(
            package=pkg,
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(eval_res["execution_permitted"])
        self.assertEqual(eval_res["readiness_state"], "REVOKED")

    def test_unlisted_participant_and_unapproved_session_and_operation_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=True)

        # Unlisted participant (cohort-wide inference must fail; initializes PENDING)
        res_part = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p02",
            session_id="ses-01",
            requested_operation="preprocessing",
            dataset_id="sub-synth-pilot",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_part["authorized"])
        self.assertEqual(res_part["participant_status"], "PENDING")
        self.assertEqual(res_part["readiness_state"], "NOT_AUTHORIZED")

        # Wrong session
        res_ses = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p01",
            session_id="ses-02",
            requested_operation="preprocessing",
            dataset_id="sub-synth-pilot",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_ses["authorized"])

        # Unauthorized operation
        res_op = self.validator.verify_participant_authorization(
            pilot_records=self.pilot_records,
            participant_id="sub-synth-p01",
            session_id="ses-01",
            requested_operation="dicom_conversion",
            dataset_id="sub-synth-pilot",
            evidence_package=pkg,
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_op["authorized"])

    def test_mismatched_protocol_version_and_prospective_escalation_blocked(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=True)
        res_ver = self.validator.evaluate_evidence_package(
            package=pkg,
            expected_protocol_version="PoC-V1-19stage",
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_ver["execution_permitted"])
        self.assertEqual(res_ver["readiness_state"], "EVIDENCE_INCOMPLETE")

        # Retrospective IEC cannot authorize prospective recruitment
        res_tier = self.validator.evaluate_evidence_package(
            package=pkg,
            expected_protocol_version=DEFAULT_PROTOCOL_VERSION,
            expected_dataset_id="sub-synth-pilot",
            expected_study_tier="prospective_pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_tier["execution_permitted"])
        self.assertEqual(res_tier["readiness_state"], "EVIDENCE_INCOMPLETE")
        self.assertTrue(any("prospective" in iss.lower() for iss in res_tier["issues"]))

    def test_mismatched_dataset_missing_dua_and_missing_iec_rejected(self) -> None:
        pkg = self._activate_synthetic_register_and_package(sign=True)

        # Mismatched dataset
        res_ds = self.validator.evaluate_evidence_package(
            package=pkg,
            expected_dataset_id="sub-other",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_ds["execution_permitted"])

        # Missing DUA
        pkg_no_dua = copy.deepcopy(pkg)
        pkg_no_dua["evidence"].pop("dataset_dua")
        res_no_dua = self.validator.evaluate_evidence_package(
            package=pkg_no_dua,
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_no_dua["execution_permitted"])
        self.assertEqual(res_no_dua["category_results"]["dataset_dua"]["readiness_state"], "PENDING_EVIDENCE")

        # Download presence cannot imply DUA
        pkg_dl = copy.deepcopy(pkg)
        pkg_dl["evidence"]["dataset_dua"]["downloaded_only_inference"] = True
        res_dl = self.validator.evaluate_evidence_package(
            package=pkg_dl,
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_dl["execution_permitted"])

        # Missing IEC determination
        pkg_no_iec = copy.deepcopy(pkg)
        pkg_no_iec["evidence"].pop("ethics_iec")
        res_no_iec = self.validator.evaluate_evidence_package(
            package=pkg_no_iec,
            expected_dataset_id="sub-synth-pilot",
            hmac_key=self.hmac_key,
            current_date=self.ref_date,
        )
        self.assertFalse(res_no_iec["execution_permitted"])
        self.assertEqual(res_no_iec["category_results"]["ethics_iec"]["readiness_state"], "PENDING_EVIDENCE")

    def test_draft_self_pdf_and_request_email_rejected_and_phase_a_does_not_unlock(self) -> None:
        # Phase A complete documentary evidence (unverified by authority) -> READY_FOR_AUTHORIZATION_REVIEW, still denied!
        res_phase_a = self.validator.evaluate_evidence_package(
            package=self.complete_package_unverified,
            expected_dataset_id="sub-synth-pilot",
            requested_participant_id="sub-synth-p01",
            require_participant_authorization=True,
            current_date=self.ref_date,
        )
        self.assertEqual(res_phase_a["readiness_state"], "READY_FOR_AUTHORIZATION_REVIEW")
        self.assertFalse(res_phase_a["real_data_authorized"])
        self.assertFalse(res_phase_a["execution_permitted"])

        # Draft, self-created PDF, or request email rejected
        for bad_flag in ("is_draft", "self_created", "is_request_email"):
            bad_pkg = copy.deepcopy(self.complete_package_unverified)
            bad_pkg["evidence"]["ethics_iec"][bad_flag] = True
            bad_res = self.validator.evaluate_evidence_package(
                package=bad_pkg,
                expected_dataset_id="sub-synth-pilot",
                current_date=self.ref_date,
            )
            self.assertEqual(bad_res["readiness_state"], "EVIDENCE_INCOMPLETE")
            self.assertFalse(bad_res["execution_permitted"])

        # Two-phase dry-run activation preview never mutates register
        before = self.register_path.read_text(encoding="utf-8")
        preview = self.validator.preview_activation_plan(
            evidence_package=self.complete_package_unverified,
            target_dataset_id="sub-synth-pilot",
            participant_scope=["sub-synth-p01"],
            permitted_operations=["preprocessing"],
            approval_start_date="2026-10-01",
            approval_expiry_date="2027-10-01",
            current_date=self.ref_date,
        )
        after = self.register_path.read_text(encoding="utf-8")
        self.assertFalse(preview["activation_permitted"])
        self.assertFalse(preview["register_modified"])
        self.assertEqual(before, after)

    def test_enforce_before_data_access_denies_without_opening_mri_files(self) -> None:
        """Prove: authorization check -> DENY -> zero MRI file access."""
        called = {"count": 0}

        def fake_read_mri():
            called["count"] += 1
            return "SHOULD_NEVER_EXECUTE"

        with self.assertRaises(AuthorizationDeniedError):
            self.validator.enforce_before_data_access(
                access_callback=fake_read_mri,
                dataset_id="sub-01",
                requested_operation="preprocessing",
                participant_id="sub-synth-p01",
                session_id="ses-01",
                pilot_records=self.pilot_records,
                evidence_package=self.complete_package_unverified,
                current_date=self.ref_date,
            )
        self.assertEqual(called["count"], 0)


if __name__ == "__main__":
    unittest.main()

