"""Research Authorization Gate & Supervisor Feedback Validation Engine.

Phase A.6 / A.7 Governance & Research Authorization Gatekeeper.
Enforces fail-closed safety policy, validates machine-readable authorization registers,
verifies physical evidence references, and guarantees that real participant MRI processing
remains strictly disabled until all external institutional and supervisory approvals are verified.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class AuthorizationError(RuntimeError):
    """Base exception for authorization gate errors."""


class AuthorizationDeniedError(AuthorizationError):
    """Raised when real participant MRI execution is requested without complete authorization."""


class AuthorizationSchemaError(AuthorizationError):
    """Raised when the authorization register is malformed or schema-invalid."""


class AuthorizationBypassAttemptError(AuthorizationDeniedError):
    """Raised when an unauthorized environment variable or CLI bypass attempt is detected."""


# Canonical list of bypass environment variables to strictly detect and reject
PROHIBITED_BYPASS_ENV_VARS = (
    "NEUROADHD_ALLOW_REAL_SCANS",
    "NEUROADHD_BYPASS_AUTH_GATE",
    "NEUROADHD_BYPASS_SAFETY",
    "NEUROADHD_OVERRIDE_SAFETY",
    "NEUROADHD_FORCE_REAL_DATA",
    "NEUROADHD_DISABLE_AUTH_GATE",
    "NEUROADHD_AUTH_OVERRIDE",
)

# Core requirements that can NEVER be marked NOT_APPLICABLE
NON_EXEMPT_REQUIREMENTS: Set[str] = {
    "AUTH-IEC-01",  # Institutional Ethics Committee Approval
    "AUTH-DUA-02",  # Data Use Agreement
    "SAFE-DEID-01", # De-identification protocol
    "SAFE-FACE-02", # Facial defacing audit
    "SAFE-STOR-03", # Encrypted storage & memory isolation
}

# Approved file extensions for documentary evidence
ALLOWED_EVIDENCE_EXTENSIONS: Set[str] = {
    ".pdf",
    ".md",
    ".txt",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
}


class ResearchAuthorizationGate:
    """Validation engine and fail-closed gatekeeper for research authorization."""

    DEFAULT_REGISTER_PATH = Path("configs/research_authorization_register.json")
    DEFAULT_SCHEMA_PATH = Path("configs/authorization_register_schema.json")

    def __init__(
        self,
        register_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self.register_path = Path(register_path).resolve() if register_path else (self.base_dir / self.DEFAULT_REGISTER_PATH).resolve()
        self.schema_path = Path(schema_path).resolve() if schema_path else (self.base_dir / self.DEFAULT_SCHEMA_PATH).resolve()

    def check_for_bypass_attempts(self) -> None:
        """Detect and reject any attempt to bypass safety guards via environment variables.

        Performs case-insensitive inspection across all environment keys and catches
        non-standard truthy representations.
        """
        for env_k, env_v in os.environ.items():
            k_clean = env_k.strip().upper()
            if k_clean in PROHIBITED_BYPASS_ENV_VARS:
                # Reject any non-empty value that is not explicitly falsy
                if env_v is not None and env_v.strip().lower() not in ("", "0", "false", "no", "none", "off"):
                    raise AuthorizationBypassAttemptError(
                        f"SAFETY VIOLATION: Unauthorized safety bypass attempt detected via environment variable '{env_k}={env_v}'. "
                        "Environment variable overrides are strictly prohibited. The authorization gate fails closed."
                    )

    def load_register(self, register_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load and parse the machine-readable authorization register."""
        target_path = Path(register_path).resolve() if register_path else self.register_path
        if not target_path.exists():
            raise AuthorizationSchemaError(
                f"SAFETY VIOLATION: Authorization register not found at '{target_path}'. "
                "The authorization gate fails closed when the register is missing."
            )
        if target_path.is_symlink():
            raise AuthorizationSchemaError(
                f"SAFETY VIOLATION: Symlink detected for authorization register at '{target_path}'. "
                "Symlinks are strictly prohibited for governance records."
            )
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise AuthorizationSchemaError(
                f"SAFETY VIOLATION: Failed to parse authorization register JSON: {exc}"
            ) from exc
        return data

    def validate_schema(self, register_dict: Dict[str, Any]) -> None:
        """Validate the register dictionary against the formal JSON schema and internal integrity rules."""
        if not isinstance(register_dict, dict):
            raise AuthorizationSchemaError("SAFETY VIOLATION: Authorization register root must be a JSON object.")

        # If schema file exists, validate using jsonschema
        if self.schema_path.exists():
            try:
                import jsonschema
                with open(self.schema_path, "r", encoding="utf-8") as sf:
                    schema_data = json.load(sf)
                jsonschema.validate(instance=register_dict, schema=schema_data)
            except jsonschema.ValidationError as err:
                raise AuthorizationSchemaError(
                    f"SAFETY VIOLATION: Authorization register failed schema validation: {err.message}"
                ) from err
            except Exception as exc:
                if not isinstance(exc, AuthorizationSchemaError):
                    raise AuthorizationSchemaError(
                        f"SAFETY VIOLATION: Unexpected error during schema validation: {exc}"
                    ) from exc

        # Internal strict structural assertions
        required_root_keys = [
            "schema_version",
            "register_id",
            "project",
            "phase",
            "overall_authorization_state",
            "execution_policy",
            "real_data_authorized",
            "last_updated",
            "categories",
            "audit_trail",
        ]
        for rk in required_root_keys:
            if rk not in register_dict:
                raise AuthorizationSchemaError(
                    f"SAFETY VIOLATION: Authorization register missing required root key '{rk}'."
                )

        categories = register_dict.get("categories", {})
        for cat in ["supervisor_decisions", "institutional_legal", "operational_safeguards"]:
            if cat not in categories or not isinstance(categories[cat], list):
                raise AuthorizationSchemaError(
                    f"SAFETY VIOLATION: Categories must contain a list for '{cat}'."
                )

    def verify_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect and verify an individual decision or requirement item with conservative evidence checks."""
        item_id = item.get("id", "UNKNOWN_ID")
        status = item.get("status")
        valid_statuses = {"PENDING", "REVISION_REQUIRED", "REJECTED", "APPROVED", "NOT_APPLICABLE"}

        issues: List[str] = []
        if status not in valid_statuses:
            issues.append(f"Invalid status '{status}'. Must be one of {sorted(valid_statuses)}.")

        is_approved = (status == "APPROVED")
        evidence_verified = False

        if status == "APPROVED":
            # 1. Reviewer must be specified
            reviewer = item.get("responsible_reviewer")
            if not reviewer or not isinstance(reviewer, str) or not reviewer.strip():
                issues.append("Approved item missing non-empty 'responsible_reviewer'.")

            # 2. Decision date must be valid ISO format
            d_date = item.get("decision_date")
            if not d_date or not isinstance(d_date, str):
                issues.append("Approved item missing 'decision_date'.")
            else:
                try:
                    datetime.date.fromisoformat(d_date)
                except ValueError:
                    issues.append(f"Approved item has invalid ISO date '{d_date}'.")

            # 3. Selected option must be present
            selected_option = item.get("selected_option")
            if not selected_option or not isinstance(selected_option, str) or not selected_option.strip():
                issues.append("Approved item missing 'selected_option'.")

            # 4. Mandatory evidence reference & cryptographic digest
            ev_ref = item.get("evidence_reference")
            expected_sha = item.get("evidence_sha256")

            if not ev_ref or not isinstance(ev_ref, str) or not ev_ref.strip():
                issues.append("Approved item missing mandatory 'evidence_reference'.")
            else:
                ev_ref_str = ev_ref.strip()
                raw_path = Path(ev_ref_str)
                if raw_path.is_absolute():
                    issues.append(
                        f"Path traversal detected: Absolute paths are strictly prohibited for evidence references: '{ev_ref_str}'."
                    )
                else:
                    candidate_path = self.base_dir / raw_path

                    # Symlink detection on raw candidate path and intermediate directories
                    has_symlink = False
                    try:
                        if candidate_path.is_symlink():
                            has_symlink = True
                        curr = candidate_path
                        while curr != curr.parent and curr != self.base_dir:
                            if curr.is_symlink():
                                has_symlink = True
                                break
                            curr = curr.parent
                    except Exception:
                        has_symlink = True

                    if has_symlink:
                        issues.append(
                            f"Symlink substitution detected in evidence path '{ev_ref_str}'. "
                            "Symlinks are strictly prohibited for approval evidence."
                        )
                    else:
                        try:
                            resolved_path = candidate_path.resolve(strict=False)
                        except Exception as res_err:
                            issues.append(f"Could not resolve evidence path '{ev_ref_str}': {res_err}")
                            resolved_path = None

                        if resolved_path is not None:
                            # Path traversal assertion: must be strictly inside self.base_dir
                            try:
                                resolved_path.relative_to(self.base_dir)
                                is_inside_repo = True
                            except ValueError:
                                is_inside_repo = False

                            if not is_inside_repo:
                                issues.append(
                                    f"Path traversal detected: Evidence reference '{ev_ref_str}' resolves outside repository root."
                                )
                            elif not resolved_path.exists():
                                issues.append(f"Referenced evidence file does not exist: '{ev_ref_str}'.")
                            elif not resolved_path.is_file():
                                issues.append(f"Referenced evidence must be a regular file, not a directory or device: '{ev_ref_str}'.")
                            elif resolved_path.suffix.lower() not in ALLOWED_EVIDENCE_EXTENSIONS:
                                issues.append(
                                    f"Evidence file extension '{resolved_path.suffix}' is not permitted. "
                                    f"Must be one of {sorted(ALLOWED_EVIDENCE_EXTENSIONS)}."
                                )
                            else:
                                # Mandatory SHA-256 digest validation for approved item
                                if not expected_sha or not isinstance(expected_sha, str) or len(expected_sha.strip()) != 64:
                                    issues.append("Approved item requires a valid 64-character hex 'evidence_sha256'.")
                                else:
                                    try:
                                        h = hashlib.sha256()
                                        with open(resolved_path, "rb") as ef:
                                            for chunk in iter(lambda: ef.read(65536), b""):
                                                h.update(chunk)
                                        actual_sha = h.hexdigest().lower()
                                        if not hmac.compare_digest(actual_sha, expected_sha.strip().lower()):
                                            issues.append(
                                                f"Evidence SHA-256 mismatch for '{ev_ref_str}'. Expected '{expected_sha}', got '{actual_sha}'."
                                            )
                                        else:
                                            evidence_verified = True
                                    except Exception as ef_err:
                                        issues.append(f"Failed to read evidence file '{resolved_path}': {ef_err}")

            # 5. Check for conflicting comments
            comments = str(item.get("reviewer_comments") or "").lower()
            conflict_words = ["reject", "denied", "unacceptable", "requires revision", "not approved", "prohibited"]
            for cw in conflict_words:
                if cw in comments:
                    issues.append(f"Conflicting decision: Status is APPROVED but reviewer comments contain '{cw}'.")

        elif status == "NOT_APPLICABLE":
            # Verify formal justification
            if item_id in NON_EXEMPT_REQUIREMENTS:
                issues.append(f"Item '{item_id}' is a mandatory legal/safety safeguard and cannot be marked NOT_APPLICABLE.")
            reviewer = item.get("responsible_reviewer")
            if not reviewer or not isinstance(reviewer, str) or not reviewer.strip():
                issues.append("NOT_APPLICABLE item missing 'responsible_reviewer'.")
            comments = str(item.get("reviewer_comments") or "").strip()
            if len(comments) < 15:
                issues.append("NOT_APPLICABLE item requires comprehensive scientific/ethical justification in 'reviewer_comments'.")

        elif status in ("PENDING", "REVISION_REQUIRED", "REJECTED"):
            # Evidence cannot claim verified approval
            evidence_verified = False

        return {
            "id": item_id,
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "status": status,
            "is_approved": is_approved and evidence_verified and len(issues) == 0,
            "evidence_verified": evidence_verified,
            "issues": issues,
        }

    def evaluate_register(self, register_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the research authorization register."""
        self.check_for_bypass_attempts()
        reg_dict = self.load_register(register_path)
        self.validate_schema(reg_dict)

        categories = reg_dict.get("categories", {})
        all_items: List[Dict[str, Any]] = []
        for cat_name, cat_list in categories.items():
            for item in cat_list:
                all_items.append(item)

        total_items = len(all_items)
        status_counts = {
            "PENDING": 0,
            "REVISION_REQUIRED": 0,
            "REJECTED": 0,
            "APPROVED": 0,
            "NOT_APPLICABLE": 0,
        }
        item_evaluations: List[Dict[str, Any]] = []
        global_issues: List[str] = []

        seen_ids: Set[str] = set()
        for it in all_items:
            iid = it.get("id")
            if iid in seen_ids:
                global_issues.append(f"Duplicate item ID '{iid}' detected in register.")
            seen_ids.add(iid)

            res = self.verify_item(it)
            st = res["status"]
            if st in status_counts:
                status_counts[st] += 1
            if res["issues"]:
                for iss in res["issues"]:
                    global_issues.append(f"[{res['id']}] {iss}")
            item_evaluations.append(res)

        # Conflict check between root flags and item states
        claimed_authorized = reg_dict.get("real_data_authorized", False)
        claimed_overall = reg_dict.get("overall_authorization_state", "NOT_AUTHORIZED")
        execution_policy = reg_dict.get("execution_policy", "STRICTLY_SYNTHETIC_ONLY")

        # In order for real_data_authorized to be True:
        # 1. Total items > 0
        # 2. PENDING == 0
        # 3. REVISION_REQUIRED == 0
        # 4. REJECTED == 0
        # 5. Zero global issues or evidence failures
        # 6. overall_authorization_state == FULLY_AUTHORIZED
        # 7. execution_policy == AUTHORIZED_RESEARCH_ONLY or AUTHORIZED_PILOT_ONLY
        is_fully_authorized = (
            total_items > 0
            and status_counts["PENDING"] == 0
            and status_counts["REVISION_REQUIRED"] == 0
            and status_counts["REJECTED"] == 0
            and len(global_issues) == 0
            and claimed_authorized is True
            and claimed_overall == "FULLY_AUTHORIZED"
            and execution_policy in ("AUTHORIZED_RESEARCH_ONLY", "AUTHORIZED_PILOT_ONLY")
        )

        if claimed_authorized and not is_fully_authorized:
            global_issues.append(
                "CONFLICT DETECTED: Root claims 'real_data_authorized=true', but unresolved items, "
                f"pending reviews ({status_counts['PENDING']}), or invalid evidence exist."
            )

        if claimed_overall == "FULLY_AUTHORIZED" and not is_fully_authorized:
            global_issues.append(
                "CONFLICT DETECTED: Root claims 'overall_authorization_state=FULLY_AUTHORIZED' "
                f"while items remain unapproved (Pending: {status_counts['PENDING']})."
            )

        report = {
            "register_id": reg_dict.get("register_id"),
            "phase": reg_dict.get("phase"),
            "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "overall_authorization_state": claimed_overall if not global_issues else "NOT_AUTHORIZED",
            "execution_policy": execution_policy,
            "real_data_authorized": is_fully_authorized and len(global_issues) == 0,
            "total_items": total_items,
            "status_counts": status_counts,
            "issues_count": len(global_issues),
            "issues": global_issues,
            "item_evaluations": item_evaluations,
            "audit_trail_entries": len(reg_dict.get("audit_trail", [])),
        }
        return report

    def check_execution_allowed(
        self,
        requested_mode: str = "synthetic",
        allow_real_scans: bool = False,
        register_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Enforce the fail-closed real-data authorization gate before pipeline execution."""
        # Step 1: Detect and reject any environment variable bypass attempt
        self.check_for_bypass_attempts()

        # Step 2: Evaluate the authorization register
        report = self.evaluate_register(register_path)

        # Step 3: Check for real-data execution requests
        is_real_data_requested = (requested_mode.lower() != "synthetic") or (allow_real_scans is True)

        if is_real_data_requested:
            if not report["real_data_authorized"]:
                pending_details = (
                    f"Pending supervisor/governance decisions: {report['status_counts']['PENDING']}, "
                    f"Revision required: {report['status_counts']['REVISION_REQUIRED']}, "
                    f"Rejected: {report['status_counts']['REJECTED']}, "
                    f"Issues count: {report['issues_count']}."
                )
                first_issues = "; ".join(report["issues"][:3]) if report["issues"] else "No active external authorization."
                raise AuthorizationDeniedError(
                    "SAFETY VIOLATION: Execution on real MRI data is strictly denied. "
                    f"The Phase A.6 research authorization gate is NOT_AUTHORIZED. {pending_details} "
                    f"Specific blockers: {first_issues}"
                )

        # Step 4: Synthetic execution is permitted
        return {
            "authorized_mode": "synthetic",
            "real_data_authorized": False,
            "overall_authorization_state": report["overall_authorization_state"],
            "execution_policy": report["execution_policy"],
            "pending_decisions_count": report["status_counts"]["PENDING"],
            "register_id": report["register_id"],
        }

    def record_audit_event(
        self,
        event: str,
        target_id: str,
        previous_state: str,
        new_state: str,
        changed_by: str,
        reason: str,
        evidence_reference: Optional[str] = None,
        register_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Append an auditable event to the authorization register audit trail with strict sanitization."""
        target_file = Path(register_path).resolve() if register_path else self.register_path
        reg_data = self.load_register(target_file)
        self.validate_schema(reg_data)

        # Comprehensive sanitization against PHI/PII and historical subject identifiers
        prohibited_terms = [
            "patient",
            "hospital",
            "mrn",
            "birthdate",
            "ssn",
            "aadhaar",
            "address",
            "phone",
            "clinical",
            "doctor",
            "physician",
            "medical_record",
            "diagnosis",
            "sub-2344857",
            "2344857",
            "peking",
            "kki",
            "nyu",
            "abide",
            "adhd200",
            "adhd-200",
        ]
        text_to_check = f"{reason} {changed_by}".lower()
        for term in prohibited_terms:
            if term in text_to_check:
                raise ValueError(
                    f"AUDIT VIOLATION: Restricted identifier or PHI/PII keyword '{term}' detected. "
                    "Audit log entries must contain only de-identified, governance-related metadata."
                )

        # Sanitize against control character and newline injection in audit log fields
        for field_name, field_val in [
            ("event", event),
            ("target_id", target_id),
            ("previous_state", previous_state),
            ("new_state", new_state),
            ("changed_by", changed_by),
            ("reason", reason),
        ]:
            if not isinstance(field_val, str) or not field_val.strip():
                raise ValueError(f"AUDIT VIOLATION: Field '{field_name}' must be a non-empty string.")
            if "\n" in field_val or "\r" in field_val or "\x00" in field_val:
                raise ValueError(f"AUDIT VIOLATION: Log injection detected: Field '{field_name}' contains newline or control characters.")

        new_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            "target_id": target_id,
            "previous_state": previous_state,
            "new_state": new_state,
            "changed_by": changed_by,
            "reason": reason,
            "evidence_reference": evidence_reference,
        }
        reg_data.setdefault("audit_trail", []).append(new_entry)
        reg_data["last_updated"] = new_entry["timestamp"]

        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, indent=2)
            f.write("\n")
