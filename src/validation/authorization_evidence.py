"""Formal Documentary Evidence Validation & Two-Phase Authorization Readiness Engine.

Step 8: Connects institutional documentary evidence (IEC clearance, DUA, institutional
authorization, privacy/infosec sign-offs, and participant/session scoping) to the
existing fail-closed ResearchAuthorizationGate without duplicating or replacing its
HMAC-SHA256 verification mechanism.
"""
from __future__ import annotations

import copy
import csv
import datetime
import hashlib
import hmac
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.validation.authorization_gate import (
    ALLOWED_EVIDENCE_EXTENSIONS,
    PERMITTED_DATASET_OPERATIONS,
    AuthorizationDeniedError,
    AuthorizationError,
    ResearchAuthorizationGate,
)

AUTHORIZATION_READINESS_STATES: Tuple[str, ...] = (
    "NOT_AUTHORIZED",
    "PENDING_EVIDENCE",
    "EVIDENCE_INCOMPLETE",
    "READY_FOR_AUTHORIZATION_REVIEW",
    "AUTHORITY_VERIFICATION_PENDING",
    "APPROVED_SCOPED",
    "EXPIRED",
    "REVOKED",
    "REJECTED",
)

REQUIRED_EVIDENCE_CATEGORIES: Tuple[str, ...] = (
    "institutional_authorization",
    "ethics_iec",
    "dataset_dua",
    "privacy_infosec",
)

PARTICIPANT_EVIDENCE_CATEGORY = "participant_authorization"
ALL_EVIDENCE_CATEGORIES: Tuple[str, ...] = REQUIRED_EVIDENCE_CATEGORIES + (
    PARTICIPANT_EVIDENCE_CATEGORY,
)

NON_APPROVAL_DOCUMENT_STATUSES = frozenset({
    "DRAFT",
    "DRAFT_INCOMPLETE",
    "REQUEST_EMAIL",
    "SELF_GENERATED_PDF",
    "PROPOSED",
    "PENDING",
    "SUBMITTED",
    "UNVERIFIED",
})

VALID_APPROVAL_DOCUMENT_STATUSES = frozenset({
    "APPROVED",
    "EXECUTED",
    "SIGNED",
    "ACTIVE",
    "EXEMPTED",
    "WAIVED",
})

TERMINAL_DENIAL_STATUSES = frozenset({"REVOKED", "REJECTED", "EXPIRED"})

STUDY_TIERS = ("retrospective_secondary", "prospective_pilot")
MAX_PILOT_PARTICIPANTS = 6

DEFAULT_PROTOCOL_ID = "DOC-GOV-PROT-001"
DEFAULT_PROTOCOL_TITLE = (
    "NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity "
    "and Machine Learning"
)
DEFAULT_PROTOCOL_VERSION = "PoC-V2-23stage"

# Disqualifying text markers that indicate an unsubmitted draft or unpopulated template
_DRAFT_OR_PLACEHOLDER_MARKERS = (
    "DRAFT — NOT SUBMITTED — NOT APPROVED",
    "DRAFT - NOT SUBMITTED - NOT APPROVED",
    "[INSTITUTION_NAME]",
    "[IEC_NAME]",
    "[PI_NAME]",
    "[SUPERVISOR_NAME]",
    "[DPO_NAME]",
    "[APPROVAL_DATE]",
    "[EXPIRY_DATE]",
    "[EXECUTED_DUA_PDF]",
    "[IEC_PROTOCOL_NUMBER]",
    "REQUIRES_AUTHORITY_DECISION",
)

_PLACEHOLDER_BRACKET_RE = re.compile(r"^\[[A-Z0-9_ :/-]+\]$")


class AuthorizationEvidenceError(AuthorizationDeniedError):
    """Raised when documentary evidence or scoped participant authorization fails closed."""


def _is_placeholder_value(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    cleaned = value.strip()
    if not cleaned:
        return True
    if _PLACEHOLDER_BRACKET_RE.match(cleaned):
        return True
    upper = cleaned.upper()
    if any(tok in upper for tok in ("[PENDING", "TO_BE_DEFINED", "UNRESOLVED", "PLACEHOLDER")):
        return True
    return False


def _parse_iso_date(raw: Any, field_name: str) -> Tuple[Optional[datetime.date], Optional[str]]:
    if not isinstance(raw, str) or not raw.strip():
        return None, f"Missing mandatory '{field_name}'."
    try:
        return datetime.date.fromisoformat(raw.strip()), None
    except ValueError:
        return None, f"Invalid ISO date for '{field_name}': '{raw}'."


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class AuthorizationEvidenceValidator:
    """Validates documentary approval evidence and two-phase authorization readiness.

    Composes `ResearchAuthorizationGate` rather than re-implementing cryptographic
    HMAC verification.
    """

    def __init__(
        self,
        gate: Optional[ResearchAuthorizationGate] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if gate is not None:
            self.gate = gate
            self.base_dir = gate.base_dir
        else:
            self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
            self.gate = ResearchAuthorizationGate(base_dir=self.base_dir)

    def _validate_evidence_file(
        self,
        evidence_path: Any,
        expected_sha256: Any,
    ) -> Dict[str, Any]:
        issues: List[str] = []
        if not isinstance(evidence_path, str) or not evidence_path.strip():
            return {
                "file_exists": False,
                "sha256_verified": False,
                "actual_sha256": None,
                "has_draft_markers": False,
                "issues": ["Missing mandatory 'evidence_file' path."],
            }

        rel_str = evidence_path.strip()
        raw = Path(rel_str)
        if raw.is_absolute():
            issues.append(
                f"Path traversal blocked: absolute evidence path is prohibited ('{rel_str}')."
            )
            return {
                "file_exists": False,
                "sha256_verified": False,
                "actual_sha256": None,
                "has_draft_markers": False,
                "issues": issues,
            }

        candidate = self.base_dir / raw
        has_symlink = False
        try:
            curr = candidate
            while True:
                if curr.is_symlink():
                    has_symlink = True
                    break
                if curr == self.base_dir or curr == curr.parent:
                    break
                curr = curr.parent
        except OSError:
            has_symlink = True

        if has_symlink:
            issues.append(f"Symlink detected in evidence path '{rel_str}'.")
            return {
                "file_exists": False,
                "sha256_verified": False,
                "actual_sha256": None,
                "has_draft_markers": False,
                "issues": issues,
            }

        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.base_dir)
        except Exception:
            issues.append(
                f"Path traversal blocked: evidence path '{rel_str}' escapes allowed root."
            )
            return {
                "file_exists": False,
                "sha256_verified": False,
                "actual_sha256": None,
                "has_draft_markers": False,
                "issues": issues,
            }

        if not resolved.exists():
            issues.append(f"Evidence file does not exist: '{rel_str}'.")
            return {
                "file_exists": False,
                "sha256_verified": False,
                "actual_sha256": None,
                "has_draft_markers": False,
                "issues": issues,
            }

        if not resolved.is_file():
            issues.append(f"Evidence reference is not a regular file: '{rel_str}'.")
            return {
                "file_exists": False,
                "sha256_verified": False,
                "actual_sha256": None,
                "has_draft_markers": False,
                "issues": issues,
            }

        if resolved.suffix.lower() not in ALLOWED_EVIDENCE_EXTENSIONS:
            issues.append(
                f"Unsupported evidence extension '{resolved.suffix}' "
                f"(allowed: {sorted(ALLOWED_EVIDENCE_EXTENSIONS)})."
            )

        actual_sha = _compute_file_sha256(resolved)
        sha_verified = False
        if not isinstance(expected_sha256, str) or len(expected_sha256.strip()) != 64:
            issues.append("Evidence record requires a 64-character hex 'evidence_sha256'.")
        else:
            exp_clean = expected_sha256.strip().lower()
            try:
                int(exp_clean, 16)
                if hmac.compare_digest(actual_sha, exp_clean):
                    sha_verified = True
                else:
                    issues.append(
                        f"Evidence document SHA-256 mismatch for '{rel_str}'."
                    )
            except ValueError:
                issues.append("Provided 'evidence_sha256' contains non-hex characters.")

        has_draft_markers = False
        if resolved.suffix.lower() in {".md", ".txt", ".json"}:
            try:
                content = resolved.read_text(encoding="utf-8", errors="ignore")
                if any(marker in content for marker in _DRAFT_OR_PLACEHOLDER_MARKERS):
                    has_draft_markers = True
                    issues.append(
                        f"Evidence file '{rel_str}' contains draft or unresolved placeholder markers "
                        "and cannot serve as approval evidence."
                    )
            except OSError as exc:
                issues.append(f"Unable to read text evidence file '{rel_str}': {exc}")

        return {
            "file_exists": True,
            "sha256_verified": sha_verified,
            "actual_sha256": actual_sha,
            "has_draft_markers": has_draft_markers,
            "issues": issues,
        }

    def validate_evidence_record(
        self,
        category: str,
        record: Optional[Dict[str, Any]],
        expected_protocol_title: str = DEFAULT_PROTOCOL_TITLE,
        expected_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        expected_dataset_id: str = "sub-01",
        requested_operation: Optional[str] = None,
        expected_study_tier: str = "retrospective_secondary",
        requested_participant_id: Optional[str] = None,
        current_date: Optional[Union[datetime.date, datetime.datetime]] = None,
    ) -> Dict[str, Any]:
        """Validate a single documentary approval evidence item across all 14 checks."""
        if category not in ALL_EVIDENCE_CATEGORIES:
            raise AuthorizationEvidenceError(
                f"Unrecognized documentary evidence category '{category}'."
            )

        if not record or not isinstance(record, dict):
            return {
                "category": category,
                "readiness_state": "PENDING_EVIDENCE",
                "evidence_complete": False,
                "authority_verified": False,
                "revoked": False,
                "rejected": False,
                "expired": False,
                "issues": [f"Missing evidence record for category '{category}'."],
                "approval_reference": None,
                "effective_date": None,
                "expiry_date": None,
                "sha256_verified": False,
            }

        ref_date = (
            current_date.date()
            if isinstance(current_date, datetime.datetime)
            else (current_date or datetime.date.today())
        )

        issues: List[str] = []
        raw_status = str(record.get("document_status") or "PENDING").strip().upper()
        is_revoked = bool(record.get("revoked", False)) or raw_status == "REVOKED"
        is_rejected = raw_status == "REJECTED"
        is_expired = False

        # Non-approval artifact guards (Section 5/8)
        if record.get("is_draft") is True:
            issues.append("Draft documents cannot serve as real-data approval evidence.")
        if record.get("self_created") is True:
            issues.append(
                "Self-created PDFs or developer-generated summaries are not institutional approval evidence."
            )
        if record.get("is_request_email") is True:
            issues.append("Approval request emails are not approval evidence.")

        if raw_status in NON_APPROVAL_DOCUMENT_STATUSES:
            issues.append(
                f"Document status '{raw_status}' is not an executed institutional approval."
            )
        elif raw_status not in VALID_APPROVAL_DOCUMENT_STATUSES and not is_revoked and not is_rejected:
            if raw_status == "EXPIRED":
                is_expired = True
                issues.append("Documentary evidence is marked EXPIRED.")
            else:
                issues.append(f"Unsupported document_status '{raw_status}'.")

        if is_revoked:
            issues.append(f"Documentary evidence for '{category}' has been REVOKED.")
        if is_rejected:
            issues.append(f"Documentary evidence for '{category}' has been REJECTED.")

        # File existence, containment, SHA-256, and placeholder/draft scan
        file_check = self._validate_evidence_file(
            record.get("evidence_file"),
            record.get("evidence_sha256"),
        )
        issues.extend(file_check["issues"])

        # Institutional identity, approving authority, reference
        inst_name = record.get("institution_name")
        if _is_placeholder_value(inst_name):
            issues.append("Missing or placeholder 'institution_name'.")

        approving_auth = record.get("approving_authority")
        if _is_placeholder_value(approving_auth):
            issues.append("Missing or placeholder 'approving_authority'.")

        approval_ref = record.get("approval_reference")
        if _is_placeholder_value(approval_ref):
            issues.append("Missing or placeholder 'approval_reference'.")

        # Dates & expiry check
        eff_date, eff_err = _parse_iso_date(record.get("effective_date"), "effective_date")
        if eff_err:
            issues.append(eff_err)
        elif eff_date is not None and eff_date > ref_date:
            issues.append(
                f"Effective date '{eff_date.isoformat()}' is in the future (ref: {ref_date.isoformat()})."
            )

        exp_date, exp_err = _parse_iso_date(record.get("expiry_date"), "expiry_date")
        if exp_err:
            issues.append(exp_err)
        elif exp_date is not None:
            if eff_date is not None and exp_date < eff_date:
                issues.append(
                    f"Expiry date '{exp_date.isoformat()}' precedes effective date '{eff_date.isoformat()}'."
                )
            elif exp_date < ref_date:
                is_expired = True
                issues.append(
                    f"Documentary evidence expired on '{exp_date.isoformat()}' (ref: {ref_date.isoformat()})."
                )

        # Protocol title & version check
        doc_title = record.get("protocol_title")
        if not isinstance(doc_title, str) or doc_title.strip() != expected_protocol_title.strip():
            issues.append(
                f"Protocol title mismatch: expected '{expected_protocol_title}', got '{doc_title}'."
            )

        doc_version = record.get("protocol_version")
        if not isinstance(doc_version, str) or doc_version.strip() != expected_protocol_version.strip():
            issues.append(
                f"Protocol version mismatch: expected '{expected_protocol_version}', got '{doc_version}'."
            )

        # Approved dataset check
        doc_dataset = record.get("dataset_id")
        if not isinstance(doc_dataset, str) or doc_dataset.strip() != expected_dataset_id.strip():
            issues.append(
                f"Dataset mismatch in '{category}': expected '{expected_dataset_id}', got '{doc_dataset}'."
            )

        # Study tier separation: retrospective secondary approval NEVER authorizes prospective pilot
        doc_tier = record.get("study_tier")
        if doc_tier is not None:
            if doc_tier not in STUDY_TIERS:
                issues.append(f"Unsupported study_tier '{doc_tier}'.")
            elif doc_tier == "retrospective_secondary" and expected_study_tier == "prospective_pilot":
                issues.append(
                    "Scope violation: retrospective secondary-analysis approval does not authorize "
                    "prospective N=6 participant recruitment."
                )
            elif doc_tier != expected_study_tier:
                issues.append(
                    f"Study tier mismatch: expected '{expected_study_tier}', got '{doc_tier}'."
                )
        elif expected_study_tier == "prospective_pilot" and category == "ethics_iec":
            issues.append("Prospective study IEC evidence must explicitly specify study_tier='prospective_pilot'.")

        # Approved operations check
        ops = record.get("approved_operations")
        if not isinstance(ops, list) or not ops:
            issues.append("Field 'approved_operations' must be a non-empty list.")
        else:
            invalid_ops = [op for op in ops if op not in PERMITTED_DATASET_OPERATIONS]
            if invalid_ops:
                issues.append(f"Unrecognized operation(s) in documentary evidence: {invalid_ops}.")
            if requested_operation and requested_operation not in ops:
                issues.append(
                    f"Requested operation '{requested_operation}' is not authorized by '{category}'."
                )

        # Participant scope check (no cohort-wide inference)
        approved_participants = record.get("approved_participants")
        if requested_participant_id is not None:
            if not isinstance(approved_participants, list) or requested_participant_id not in approved_participants:
                issues.append(
                    f"Participant '{requested_participant_id}' is not included in '{category}' participant scope."
                )

        # Storage and sharing conditions
        if _is_placeholder_value(record.get("storage_conditions")):
            issues.append("Missing or placeholder 'storage_conditions'.")
        if _is_placeholder_value(record.get("sharing_conditions")):
            issues.append("Missing or placeholder 'sharing_conditions'.")

        # Category-specific rules
        if category == "dataset_dua":
            if record.get("downloaded_only_inference") is True:
                issues.append("Dataset download presence cannot be used to infer DUA permission.")
            if record.get("non_commercial_confirmed") is not True:
                issues.append("ADHD-200 / INDI DUA evidence must confirm non_commercial_confirmed=True.")

        authority_verified = bool(record.get("authority_verified", False))

        if is_revoked:
            state = "REVOKED"
        elif is_rejected:
            state = "REJECTED"
        elif is_expired:
            state = "EXPIRED"
        elif issues:
            state = "EVIDENCE_INCOMPLETE"
        elif not authority_verified:
            state = "READY_FOR_AUTHORIZATION_REVIEW"
        else:
            state = "APPROVED_SCOPED"

        return {
            "category": category,
            "readiness_state": state,
            "evidence_complete": (len(issues) == 0 and not is_revoked and not is_rejected and not is_expired),
            "authority_verified": authority_verified,
            "revoked": is_revoked,
            "rejected": is_rejected,
            "expired": is_expired,
            "issues": issues,
            "approval_reference": approval_ref if isinstance(approval_ref, str) else None,
            "effective_date": eff_date.isoformat() if eff_date else None,
            "expiry_date": exp_date.isoformat() if exp_date else None,
            "dataset_id": doc_dataset,
            "protocol_version": doc_version,
            "approved_operations": list(ops) if isinstance(ops, list) else [],
            "approved_participants": list(approved_participants) if isinstance(approved_participants, list) else [],
            "register_item_ids": list(record.get("register_item_ids") or []),
            "sha256_verified": file_check["sha256_verified"],
        }

    def evaluate_evidence_package(
        self,
        package: Optional[Dict[str, Any]] = None,
        expected_protocol_title: str = DEFAULT_PROTOCOL_TITLE,
        expected_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        expected_dataset_id: str = "sub-01",
        requested_operation: Optional[str] = None,
        expected_study_tier: str = "retrospective_secondary",
        requested_participant_id: Optional[str] = None,
        require_participant_authorization: bool = False,
        register_path: Optional[Union[str, Path]] = None,
        hmac_key: Optional[Union[str, bytes]] = None,
        current_date: Optional[Union[datetime.date, datetime.datetime]] = None,
    ) -> Dict[str, Any]:
        """Run Phase A documentary validation & Phase B register/HMAC consistency check."""
        self.gate.check_for_bypass_attempts()

        ref_date = (
            current_date.date()
            if isinstance(current_date, datetime.datetime)
            else (current_date or datetime.date.today())
        )

        register_report = self.gate.evaluate_register(register_path=register_path)
        register_dict = self.gate.load_register(register_path=register_path)

        # Baseline repo mode when no completed evidence bundle has been presented
        if package is None or not package.get("evidence"):
            outstanding = []
            for item in register_report.get("item_evaluations", []):
                if item.get("status") != "APPROVED":
                    outstanding.append(f"{item['id']}: {item['status']}")
            return {
                "readiness_state": "PENDING_EVIDENCE",
                "overall_authorization_state": register_report["overall_authorization_state"],
                "execution_policy": register_report["execution_policy"],
                "real_data_authorized": False,
                "execution_permitted": False,
                "phase_a_complete": False,
                "authority_verified": False,
                "dataset_id": expected_dataset_id,
                "protocol_version": expected_protocol_version,
                "category_results": {
                    cat: {
                        "category": cat,
                        "readiness_state": "PENDING_EVIDENCE",
                        "evidence_complete": False,
                        "authority_verified": False,
                        "issues": [f"No institutional evidence package provided for '{cat}'."],
                    }
                    for cat in REQUIRED_EVIDENCE_CATEGORIES
                },
                "dataset_verification": None,
                "register_summary": {
                    "register_id": register_report["register_id"],
                    "status_counts": register_report["status_counts"],
                    "total_items": register_report["total_items"],
                    "dataset_authorizations_approved": register_report["dataset_authorizations_approved"],
                },
                "issues": [
                    "No verified institutional documentary approval package is registered; execution remains strictly synthetic."
                ],
                "outstanding_requirements": outstanding[:10],
            }

        evidence_map = package.get("evidence", {})
        if not isinstance(evidence_map, dict):
            raise AuthorizationEvidenceError("Evidence package 'evidence' field must be a JSON object.")

        categories_to_check = list(REQUIRED_EVIDENCE_CATEGORIES)
        if require_participant_authorization or PARTICIPANT_EVIDENCE_CATEGORY in evidence_map:
            categories_to_check.append(PARTICIPANT_EVIDENCE_CATEGORY)

        category_results: Dict[str, Dict[str, Any]] = {}
        issues: List[str] = []

        for cat in categories_to_check:
            res = self.validate_evidence_record(
                category=cat,
                record=evidence_map.get(cat),
                expected_protocol_title=expected_protocol_title,
                expected_protocol_version=expected_protocol_version,
                expected_dataset_id=expected_dataset_id,
                requested_operation=requested_operation,
                expected_study_tier=expected_study_tier,
                requested_participant_id=requested_participant_id,
                current_date=ref_date,
            )
            category_results[cat] = res
            for iss in res["issues"]:
                issues.append(f"[{cat}] {iss}")

        # Consistency check between documentary evidence and runtime register
        matching_ds_records = [
            rec
            for rec in register_dict.get("dataset_authorizations", [])
            if isinstance(rec, dict) and rec.get("dataset_id") == expected_dataset_id
        ]
        dataset_verification = None
        register_sync_notes: List[str] = []
        if not matching_ds_records:
            register_sync_notes.append(
                f"Register lacks dataset_authorizations entry for '{expected_dataset_id}' (Phase B activation required)."
            )
        elif len(matching_ds_records) > 1:
            issues.append(
                f"Register contains multiple conflicting dataset_authorizations entries for '{expected_dataset_id}'."
            )
        else:
            ds_rec = matching_ds_records[0]
            dataset_verification = self.gate.verify_dataset_authorization(
                ds_rec,
                requested_dataset=expected_dataset_id,
                requested_operation=requested_operation,
                key=hmac_key,
                current_date=ref_date,
            )
            # Cross-check documentary approval reference & operations against dataset record when ds_rec is populated
            ds_ref = ds_rec.get("institutional_approval_reference")
            iec_res = category_results.get("ethics_iec", {})
            if (
                ds_rec.get("authorization_status") == "APPROVED"
                and iec_res.get("evidence_complete")
                and isinstance(ds_ref, str)
                and ds_ref.strip()
                and iec_res.get("approval_reference")
                and ds_ref.strip() != iec_res["approval_reference"].strip()
            ):
                issues.append(
                    f"Register reference mismatch: dataset record cites '{ds_ref}', "
                    f"but ethics_iec evidence cites '{iec_res['approval_reference']}'."
                )

            ds_ops = set(ds_rec.get("permitted_operations") or [])
            for cat_name, cat_res in category_results.items():
                if cat_res.get("evidence_complete") and cat_res.get("approved_operations"):
                    cat_ops = set(cat_res["approved_operations"])
                    extra_in_register = ds_ops - cat_ops
                    if ds_rec.get("authorization_status") == "APPROVED" and extra_in_register:
                        issues.append(
                            f"Operation scope mismatch: register permits {sorted(extra_in_register)} "
                            f"not covered by '{cat_name}' documentary approval."
                        )

        any_revoked = any(r["revoked"] for r in category_results.values()) or (
            dataset_verification is not None and dataset_verification.get("status") == "REVOKED"
        )
        any_rejected = any(r["rejected"] for r in category_results.values()) or (
            dataset_verification is not None and dataset_verification.get("status") == "REJECTED"
        )
        any_expired = any(r["expired"] for r in category_results.values()) or (
            dataset_verification is not None and dataset_verification.get("expiry_status") == "EXPIRED"
        )
        all_evidence_complete = all(r["evidence_complete"] for r in category_results.values()) and len(issues) == 0
        all_authority_verified = all(r["authority_verified"] for r in category_results.values())

        ds_valid = bool(dataset_verification and dataset_verification.get("is_valid"))
        register_ready = bool(register_report.get("real_data_authorized") and ds_valid)

        # Strict two-phase state machine
        if any_revoked:
            readiness_state = "REVOKED"
        elif any_rejected:
            readiness_state = "REJECTED"
        elif any_expired:
            readiness_state = "EXPIRED"
        elif not all_evidence_complete:
            readiness_state = "EVIDENCE_INCOMPLETE"
        elif not all_authority_verified:
            if str(package.get("authority_verification_status", "")).upper() == "SUBMITTED":
                readiness_state = "AUTHORITY_VERIFICATION_PENDING"
            else:
                readiness_state = "READY_FOR_AUTHORIZATION_REVIEW"
        elif register_ready:
            readiness_state = "APPROVED_SCOPED"
        else:
            readiness_state = "AUTHORITY_VERIFICATION_PENDING"
            for note in register_sync_notes:
                issues.append(f"[register] {note}")
            for iss in register_report.get("issues", []):
                issues.append(f"[register] {iss}")
            if dataset_verification and dataset_verification.get("issues"):
                for iss in dataset_verification["issues"]:
                    issues.append(f"[dataset_hmac] {iss}")

        execution_permitted = (readiness_state == "APPROVED_SCOPED") and register_ready

        outstanding: List[str] = list(register_sync_notes)
        for cat_name, res in category_results.items():
            if not res["evidence_complete"]:
                outstanding.append(f"{cat_name}: {res['readiness_state']}")
            elif not res["authority_verified"]:
                outstanding.append(f"{cat_name}: AWAITING_AUTHORITY_VERIFICATION")
        if register_report["status_counts"]["PENDING"] > 0:
            outstanding.append(
                f"register_items: {register_report['status_counts']['PENDING']} PENDING"
            )
        if dataset_verification and not dataset_verification.get("is_valid"):
            outstanding.append(
                f"dataset_hmac: {dataset_verification.get('signature_status')} / {dataset_verification.get('status')}"
            )

        return {
            "readiness_state": readiness_state,
            "overall_authorization_state": (
                register_report["overall_authorization_state"]
                if execution_permitted
                else "NOT_AUTHORIZED"
            ),
            "execution_policy": register_report["execution_policy"],
            "real_data_authorized": bool(execution_permitted and register_report["real_data_authorized"]),
            "execution_permitted": execution_permitted,
            "phase_a_complete": all_evidence_complete,
            "authority_verified": all_authority_verified,
            "dataset_id": expected_dataset_id,
            "protocol_version": expected_protocol_version,
            "study_tier": expected_study_tier,
            "category_results": category_results,
            "dataset_verification": dataset_verification,
            "register_summary": {
                "register_id": register_report["register_id"],
                "status_counts": register_report["status_counts"],
                "total_items": register_report["total_items"],
                "dataset_authorizations_approved": register_report["dataset_authorizations_approved"],
            },
            "issues": issues,
            "outstanding_requirements": outstanding,
        }

    def verify_participant_authorization(
        self,
        pilot_records: Sequence[Dict[str, Any]],
        participant_id: str,
        session_id: str,
        requested_operation: str,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        dataset_id: str = "sub-01",
        study_tier: str = "retrospective_secondary",
        evidence_package: Optional[Dict[str, Any]] = None,
        pilot_cap: Optional[int] = MAX_PILOT_PARTICIPANTS,
        register_path: Optional[Union[str, Path]] = None,
        hmac_key: Optional[Union[str, bytes]] = None,
        current_date: Optional[Union[datetime.date, datetime.datetime]] = None,
    ) -> Dict[str, Any]:
        """Verify participant-scoped authorization for N=6 feasibility pilot runs."""
        self.gate.check_for_bypass_attempts()

        ref_date = (
            current_date.date()
            if isinstance(current_date, datetime.datetime)
            else (current_date or datetime.date.today())
        )
        reasons: List[str] = []

        if pilot_cap is not None and len(pilot_records) > pilot_cap:
            reasons.append(
                f"Pilot roster exceeds maximum feasibility cap ({len(pilot_records)} > {pilot_cap})."
            )

        matches = [
            rec
            for rec in pilot_records
            if isinstance(rec, dict) and rec.get("participant_id") == participant_id
        ]
        if not matches:
            return {
                "authorized": False,
                "participant_id": participant_id,
                "participant_status": "PENDING",
                "readiness_state": "NOT_AUTHORIZED",
                "reasons": reasons + [
                    f"Participant '{participant_id}' is not listed in the authorized roster (initialized as PENDING)."
                ],
            }

        if len(matches) > 1:
            return {
                "authorized": False,
                "participant_id": participant_id,
                "participant_status": "CONFLICT",
                "readiness_state": "NOT_AUTHORIZED",
                "reasons": [f"Duplicate authorization records found for '{participant_id}'."],
            }

        p_rec = matches[0]
        p_status = str(p_rec.get("authorization_status") or "PENDING").strip().upper()
        if p_status == "REVOKED":
            return {
                "authorized": False,
                "participant_id": participant_id,
                "participant_status": "REVOKED",
                "readiness_state": "REVOKED",
                "reasons": [f"Participant '{participant_id}' authorization has been REVOKED."],
            }
        if p_status == "REJECTED":
            return {
                "authorized": False,
                "participant_id": participant_id,
                "participant_status": "REJECTED",
                "readiness_state": "REJECTED",
                "reasons": [f"Participant '{participant_id}' authorization has been REJECTED."],
            }
        if p_status != "APPROVED":
            reasons.append(
                f"Participant '{participant_id}' authorization_status is '{p_status}', not 'APPROVED'."
            )

        # Protocol, dataset, tier, session, operation, and date validation
        if p_rec.get("protocol_version") != protocol_version:
            reasons.append(
                f"Protocol version mismatch for '{participant_id}': "
                f"expected '{protocol_version}', got '{p_rec.get('protocol_version')}'."
            )
        if p_rec.get("dataset_id") != dataset_id:
            reasons.append(
                f"Dataset mismatch for '{participant_id}': "
                f"expected '{dataset_id}', got '{p_rec.get('dataset_id')}'."
            )
        if p_rec.get("study_tier") == "retrospective_secondary" and study_tier == "prospective_pilot":
            reasons.append(
                "Retrospective secondary-use record cannot authorize prospective recruitment."
            )
        elif p_rec.get("study_tier") and p_rec.get("study_tier") != study_tier:
            reasons.append(
                f"Study tier mismatch for '{participant_id}': expected '{study_tier}', got '{p_rec.get('study_tier')}'."
            )

        sessions = p_rec.get("session_scope") or []
        if not isinstance(sessions, list) or session_id not in sessions:
            reasons.append(
                f"Session '{session_id}' is not authorized in session_scope for '{participant_id}'."
            )

        p_ops = p_rec.get("permitted_operations") or []
        if not isinstance(p_ops, list) or requested_operation not in p_ops:
            reasons.append(
                f"Operation '{requested_operation}' is not permitted for '{participant_id}'."
            )

        eff_d, eff_err = _parse_iso_date(p_rec.get("effective_date"), "effective_date")
        if eff_err:
            reasons.append(eff_err)
        elif eff_d is not None and eff_d > ref_date:
            reasons.append(f"Participant authorization effective_date '{eff_d.isoformat()}' is in the future.")

        exp_d, exp_err = _parse_iso_date(p_rec.get("expiry_date"), "expiry_date")
        expired_participant = False
        if exp_err:
            reasons.append(exp_err)
        elif exp_d is not None:
            if eff_d is not None and exp_d < eff_d:
                reasons.append("Participant expiry_date precedes effective_date.")
            elif exp_d < ref_date:
                expired_participant = True
                reasons.append(f"Participant authorization expired on '{exp_d.isoformat()}'.")

        pkg_eval = self.evaluate_evidence_package(
            package=evidence_package,
            expected_protocol_version=protocol_version,
            expected_dataset_id=dataset_id,
            requested_operation=requested_operation,
            expected_study_tier=study_tier,
            requested_participant_id=participant_id,
            require_participant_authorization=True,
            register_path=register_path,
            hmac_key=hmac_key,
            current_date=ref_date,
        )

        if not pkg_eval["execution_permitted"]:
            reasons.extend(pkg_eval["issues"])

        final_state = pkg_eval["readiness_state"]
        if expired_participant and final_state not in ("REVOKED", "REJECTED"):
            final_state = "EXPIRED"
        elif reasons and final_state == "APPROVED_SCOPED":
            final_state = "NOT_AUTHORIZED"

        authorized = (len(reasons) == 0) and pkg_eval["execution_permitted"] and (final_state == "APPROVED_SCOPED")

        return {
            "authorized": authorized,
            "participant_id": participant_id,
            "session_id": session_id,
            "participant_status": p_status,
            "readiness_state": "APPROVED_SCOPED" if authorized else final_state,
            "dataset_id": dataset_id,
            "protocol_version": protocol_version,
            "requested_operation": requested_operation,
            "signature_status": (
                pkg_eval["dataset_verification"].get("signature_status")
                if pkg_eval.get("dataset_verification")
                else "MISSING_SIGNATURE"
            ),
            "reasons": reasons,
        }

    def preview_activation_plan(
        self,
        evidence_package: Optional[Dict[str, Any]],
        target_dataset_id: str,
        participant_scope: Sequence[str],
        permitted_operations: Sequence[str],
        approval_start_date: str,
        approval_expiry_date: str,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        study_tier: str = "retrospective_secondary",
        register_path: Optional[Union[str, Path]] = None,
        current_date: Optional[Union[datetime.date, datetime.datetime]] = None,
    ) -> Dict[str, Any]:
        """Generate a Phase B activation dry-run preview without modifying the register."""
        pkg_eval = self.evaluate_evidence_package(
            package=evidence_package,
            expected_protocol_version=protocol_version,
            expected_dataset_id=target_dataset_id,
            expected_study_tier=study_tier,
            register_path=register_path,
            current_date=current_date,
        )

        evidence_refs = {
            cat: {
                "approval_reference": res.get("approval_reference"),
                "sha256_verified": res.get("sha256_verified"),
                "authority_verified": res.get("authority_verified"),
                "readiness_state": res.get("readiness_state"),
            }
            for cat, res in pkg_eval.get("category_results", {}).items()
        }

        can_activate = bool(
            pkg_eval.get("phase_a_complete")
            and pkg_eval.get("authority_verified")
            and pkg_eval["readiness_state"] in ("AUTHORITY_VERIFICATION_PENDING", "APPROVED_SCOPED")
            and len(participant_scope) > 0
            and len(permitted_operations) > 0
        )

        intended_register_changes = {
            "root_fields": {
                "overall_authorization_state": "CONDITIONALLY_AUTHORIZED"
                if study_tier == "prospective_pilot" or len(participant_scope) <= MAX_PILOT_PARTICIPANTS
                else "FULLY_AUTHORIZED",
                "execution_policy": "AUTHORIZED_PILOT_ONLY"
                if len(participant_scope) <= MAX_PILOT_PARTICIPANTS
                else "AUTHORIZED_RESEARCH_ONLY",
                "real_data_authorized": True,
            },
            "dataset_authorizations_entry": {
                "dataset_id": target_dataset_id,
                "authorization_status": "APPROVED",
                "approval_date": approval_start_date,
                "expiry_date": approval_expiry_date,
                "permitted_operations": sorted(set(permitted_operations)),
                "cryptographic_signature": "GENERATED_VIA_ResearchAuthorizationGate.compute_authorization_signature",
            },
        }

        return {
            "activation_permitted": can_activate,
            "register_modified": False,
            "readiness_state": pkg_eval["readiness_state"],
            "intended_register_changes": intended_register_changes if can_activate else None,
            "evidence_references": evidence_refs,
            "participant_scope": list(participant_scope),
            "permitted_operations": sorted(set(permitted_operations)),
            "authorization_window": {
                "approval_date": approval_start_date,
                "expiry_date": approval_expiry_date,
            },
            "signing_workflow": "ResearchAuthorizationGate.compute_authorization_signature (NEUROADHD_HMAC_KEY)",
            "blockers": pkg_eval["issues"] + pkg_eval["outstanding_requirements"],
        }

    def enforce_before_data_access(
        self,
        access_callback: Callable[[], Any],
        dataset_id: str,
        requested_operation: str,
        participant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        pilot_records: Optional[Sequence[Dict[str, Any]]] = None,
        evidence_package: Optional[Dict[str, Any]] = None,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        study_tier: str = "retrospective_secondary",
        register_path: Optional[Union[str, Path]] = None,
        hmac_key: Optional[Union[str, bytes]] = None,
        current_date: Optional[Union[datetime.date, datetime.datetime]] = None,
    ) -> Any:
        """Fail-closed wrapper ensuring no MRI data callback runs before full authorization."""
        self.gate.check_for_bypass_attempts()

        # First run runtime gate check for real mode
        self.gate.check_execution_allowed(
            requested_mode="real",
            allow_real_scans=True,
            dataset_id=dataset_id,
            requested_operation=requested_operation,
            register_path=register_path,
            key=hmac_key,
        )

        # Next enforce Step 8 documentary evidence & participant scope verification
        if participant_id is not None:
            p_res = self.verify_participant_authorization(
                pilot_records=pilot_records or [],
                participant_id=participant_id,
                session_id=session_id or "",
                requested_operation=requested_operation,
                protocol_version=protocol_version,
                dataset_id=dataset_id,
                study_tier=study_tier,
                evidence_package=evidence_package,
                register_path=register_path,
                hmac_key=hmac_key,
                current_date=current_date,
            )
            if not p_res["authorized"]:
                raise AuthorizationEvidenceError(
                    "SAFETY VIOLATION: Participant-scoped authorization denied "
                    f"({p_res['readiness_state']}): {'; '.join(p_res['reasons'])}"
                )
        else:
            pkg_eval = self.evaluate_evidence_package(
                package=evidence_package,
                expected_protocol_version=protocol_version,
                expected_dataset_id=dataset_id,
                requested_operation=requested_operation,
                expected_study_tier=study_tier,
                register_path=register_path,
                hmac_key=hmac_key,
                current_date=current_date,
            )
            if not pkg_eval["execution_permitted"]:
                raise AuthorizationEvidenceError(
                    "SAFETY VIOLATION: Documentary evidence verification denied execution "
                    f"({pkg_eval['readiness_state']}): {'; '.join(pkg_eval['issues'])}"
                )

        # Only reached after both runtime HMAC gate and documentary checks succeed
        return access_callback()

    def summarize_dashboard_readiness(
        self,
        evidence_package: Optional[Dict[str, Any]] = None,
        register_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Return a sanitized, read-only authorization readiness summary for the dashboard."""
        reg_report = self.gate.evaluate_register(register_path=register_path)
        pkg_eval = self.evaluate_evidence_package(
            package=evidence_package,
            register_path=register_path,
        )

        items_by_id = {
            item["id"]: item["status"]
            for item in reg_report.get("item_evaluations", [])
        }
        dataset_evals = reg_report.get("dataset_evaluations", [])
        first_ds = dataset_evals[0] if dataset_evals else {}

        # Derive sanitized statuses (never include secrets, raw signatures, keys, or PII)
        def _combine_status(item_ids: Iterable[str], evidence_cat: str) -> str:
            cat_state = (
                pkg_eval.get("category_results", {})
                .get(evidence_cat, {})
                .get("readiness_state", "PENDING_EVIDENCE")
            )
            item_statuses = [items_by_id.get(iid, "PENDING") for iid in item_ids]
            if any(st == "REJECTED" for st in item_statuses) or cat_state == "REJECTED":
                return "REJECTED"
            if cat_state == "REVOKED":
                return "REVOKED"
            if cat_state == "EXPIRED":
                return "EXPIRED"
            if all(st == "APPROVED" for st in item_statuses) and cat_state == "APPROVED_SCOPED":
                return "APPROVED_SCOPED"
            if cat_state == "READY_FOR_AUTHORIZATION_REVIEW":
                return "READY_FOR_AUTHORIZATION_REVIEW"
            return "PENDING"

        tracker_summary = {"total": 0, "pending": 0}
        tracker_path = self.base_dir / "docs" / "governance" / "APPROVAL_TRACKER.csv"
        if tracker_path.is_file() and not tracker_path.is_symlink():
            try:
                with tracker_path.open("r", encoding="utf-8", newline="") as cf:
                    rows = list(csv.DictReader(cf))
                tracker_summary["total"] = len(rows)
                tracker_summary["pending"] = sum(
                    1 for row in rows if (row.get("status") or "").strip().upper() == "PENDING"
                )
            except OSError:
                pass

        outstanding: List[str] = []
        if reg_report["status_counts"]["PENDING"] > 0:
            outstanding.append(
                f"{reg_report['status_counts']['PENDING']} / {reg_report['total_items']} register decisions remain PENDING"
            )
        if first_ds.get("status") != "APPROVED":
            outstanding.append(
                f"Dataset '{first_ds.get('dataset_id', 'sub-01')}' authorization is {first_ds.get('status', 'PENDING')}"
            )
        if first_ds.get("signature_status") != "VALID":
            outstanding.append(
                f"Dataset HMAC verification state is {first_ds.get('signature_status', 'MISSING_SIGNATURE')}"
            )
        if tracker_summary["total"] > 0:
            outstanding.append(
                f"{tracker_summary['pending']} / {tracker_summary['total']} institutional tracker items remain PENDING"
            )
        outstanding.append("Host runtime lacks pydeface and FSL (topup/prelude/fugue/flirt) for real-MRI mode")

        readiness_state = pkg_eval["readiness_state"]
        if reg_report["overall_authorization_state"] == "NOT_AUTHORIZED" and readiness_state == "PENDING_EVIDENCE":
            ui_readiness = "PENDING"
        else:
            ui_readiness = readiness_state

        return {
            "overall_authorization_state": reg_report["overall_authorization_state"],
            "readiness_state": readiness_state,
            "display_readiness_state": ui_readiness,
            "execution_policy": reg_report["execution_policy"],
            "real_data_authorized": False if not pkg_eval["real_data_authorized"] else True,
            "readiness_permits_execution": bool(pkg_eval["execution_permitted"]),
            "iec_evidence_status": _combine_status(("AUTH-IEC-01", "AUTH-DSU-03"), "ethics_iec"),
            "dataset_dua_status": _combine_status(("AUTH-DUA-02",), "dataset_dua"),
            "institutional_authorization_status": _combine_status(("AUTH-SUA-04", "AUTH-DPP-05"), "institutional_authorization"),
            "privacy_infosec_status": _combine_status(
                ("SAFE-DEID-01", "SAFE-FACE-02", "SAFE-STOR-03", "SAFE-ACC-04", "SAFE-AUD-05"),
                "privacy_infosec",
            ),
            "participant_scope_status": (
                f"{first_ds.get('status', 'PENDING')} (sub-01; 0/6 N=6 pilot participants approved)"
            ),
            "protocol_version": DEFAULT_PROTOCOL_VERSION,
            "protocol_id": DEFAULT_PROTOCOL_ID,
            "evidence_expiry_state": first_ds.get("expiry_status", "NO_EXPIRY_DATE"),
            "governance_tracker_pending": tracker_summary["pending"],
            "governance_tracker_total": tracker_summary["total"],
            "outstanding_requirements": copy.deepcopy(outstanding),
        }
