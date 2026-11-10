"""Safe local intake for MRI sessions and separate radiology-report extraction.

This module stages and validates uploads and automates DICOM-to-BIDS conversion.
It never runs the real-scan preprocessing pipeline or the ADHD model; those remain
behind the research authorization gate.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

SESSION_RE = re.compile(r"^intake_[0-9a-f]{32}$")
MAX_SCAN_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_REPORT_FILE_BYTES = 50 * 1024 * 1024
MAX_SESSION_BYTES = 8 * 1024 * 1024 * 1024
MAX_ZIP_FILES = 5000
MAX_ZIP_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_ZIP_RATIO = 150
CHUNK_BYTES = 1024 * 1024

SCAN_SUFFIXES = (".zip", ".nii.gz", ".nii", ".dcm", ".ima", ".json", ".tsv", ".bval", ".bvec", ".txt", ".csv")
REPORT_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png")

DEFAULT_DATA_ROOT = Path("/Users/apple/Documents/Codex/2026-09-29/NeuroADHD_Converted")


class IntakeError(RuntimeError):
    """Raised for invalid or unsafe intake operations."""


def get_data_root() -> Path:
    raw = os.environ.get("NEUROADHD_DATA_ROOT")
    if raw and raw.strip():
        p = Path(raw.strip()).expanduser().resolve()
    else:
        p = DEFAULT_DATA_ROOT.expanduser().resolve()
    return p


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _clean_filename(raw_name: str) -> str:
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise IntakeError("Upload filename is required.")
    name = raw_name.replace("\\", "/").split("/")[-1].strip()
    if name in ("", ".", "..") or "\x00" in name or len(name) > 240:
        raise IntakeError("Upload filename is invalid.")
    if any(ord(ch) < 32 for ch in name):
        raise IntakeError("Upload filename contains control characters.")
    return name


def _suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower).suffix


class IntakeService:
    """Local, path-contained intake, DICOM conversion and validation service."""

    def __init__(self, root: Optional[Union[str, Path]] = None):
        if root is None:
            self.data_root = get_data_root()
        else:
            self.data_root = Path(root).expanduser().resolve()

        if ".." in self.data_root.parts:
            raise IntakeError("Data root contains invalid path traversal components.")
        if self.data_root.is_symlink():
            raise IntakeError("Data root cannot be a symlink.")

        project_root = Path(__file__).resolve().parents[2]
        try:
            self.data_root.relative_to(project_root)
            allow_internal = os.environ.get("NEUROADHD_ALLOW_INTERNAL_ROOT") == "1"
            if not allow_internal:
                raise IntakeError(
                    f"Data root cannot be inside the project repository directory ({project_root}). "
                    "MRI data must be stored externally (e.g. NeuroADHD_Converted)."
                )
        except ValueError:
            pass

        self.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.data_root, 0o700)
        except OSError:
            pass

        self.dicom_input_dir = self.data_root / "dicom_input"
        self.staged_sessions_dir = self.data_root / "staged_sessions"
        self.bids_dir = self.data_root / "neuroadhd_bids"
        self.validated_sessions_dir = self.data_root / "validated_sessions"
        self.conversion_logs_dir = self.data_root / "conversion_logs"

        for sub_dir in (
            self.dicom_input_dir,
            self.staged_sessions_dir,
            self.bids_dir,
            self.validated_sessions_dir,
            self.conversion_logs_dir,
        ):
            sub_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(sub_dir, 0o700)
            except OSError:
                pass

        # Backwards-compatible root alias (points to staged_sessions_dir)
        self.root = self.staged_sessions_dir

    def _session_dir(self, session_id: str) -> Path:
        if not SESSION_RE.fullmatch(session_id):
            raise IntakeError("Invalid intake session identifier.")
        candidate = self.staged_sessions_dir / session_id
        if not candidate.is_dir() and (self.data_root / session_id).is_dir():
            candidate = self.data_root / session_id
        if candidate.is_symlink() or not candidate.is_dir():
            raise IntakeError("Intake session was not found or is not a safe directory.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise IntakeError("Intake session path escapes the storage root.") from exc
        return resolved

    def _read_manifest(self, session_dir: Path) -> Dict[str, Any]:
        path = session_dir / "intake_manifest.json"
        if path.is_symlink() or not path.is_file():
            raise IntakeError("Intake manifest is missing or unsafe.")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntakeError("Intake manifest is unreadable.") from exc
        if not isinstance(value, dict):
            raise IntakeError("Intake manifest must contain a JSON object.")
        return value

    def _save_manifest(self, session_dir: Path, manifest: Dict[str, Any]) -> None:
        manifest["updated_utc"] = _utc_now()
        _atomic_json(session_dir / "intake_manifest.json", manifest)

    def create_session(self, kind: str) -> Dict[str, Any]:
        if kind not in ("scan", "report"):
            raise IntakeError("Session kind must be 'scan' or 'report'.")
        session_id = f"intake_{uuid.uuid4().hex}"
        session_dir = self.staged_sessions_dir / session_id
        session_dir.mkdir(mode=0o700)
        (session_dir / "uploads").mkdir(mode=0o700)
        (session_dir / "data").mkdir(mode=0o700)
        manifest = {
            "manifest_version": "1.0.0",
            "session_id": session_id,
            "kind": kind,
            "status": "COLLECTING",
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "execution_policy": "INTAKE_AND_VALIDATION_ONLY",
            "real_data_authorized": False,
            "processing_started": False,
            "uploaded_files": [],
            "extracted_files": [],
            "validation": None,
            "report_extraction": None,
            "dicom_inspection": None,
            "dicom_conversion": None,
        }
        self._save_manifest(session_dir, manifest)
        return {"session_id": session_id, "kind": kind, "status": manifest["status"]}

    def _verify_staged_files(self, session_dir: Path, manifest: Dict[str, Any]) -> None:
        """Verify upload and extraction hashes before consuming staged files."""
        records = list(manifest.get("uploaded_files", [])) + list(manifest.get("extracted_files", []))
        for record in records:
            relative_raw = record.get("storage_path") or record.get("relative_path")
            expected_hash = record.get("sha256")
            if not isinstance(relative_raw, str) or not isinstance(expected_hash, str):
                raise IntakeError("Intake manifest contains a malformed file-integrity record.")
            relative = Path(relative_raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise IntakeError("Intake manifest contains an unsafe file path.")
            candidate = session_dir / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise IntakeError("A staged upload file is missing or unsafe.")
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(session_dir)
            except ValueError as exc:
                raise IntakeError("A staged upload file escapes its intake session.") from exc
            if _sha256(resolved) != expected_hash:
                raise IntakeError(f"Staged file integrity check failed for {candidate.name}.")

    def upload_file(
        self,
        session_id: str,
        filename: str,
        stream: BinaryIO,
        content_length: int,
    ) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("status") not in ("COLLECTING", "UPLOADED", "VALIDATION_INCOMPLETE", "DICOM_SERIES_STAGED"):
            raise IntakeError("This intake session is no longer accepting files.")

        safe_name = _clean_filename(filename)
        suffix = _suffix(safe_name)
        kind = manifest.get("kind")
        allowed = SCAN_SUFFIXES if kind == "scan" else REPORT_SUFFIXES
        if not any(safe_name.lower().endswith(item) for item in allowed):
            raise IntakeError(f"Unsupported {kind} file type: {suffix or 'unknown'}.")
        max_file = MAX_SCAN_FILE_BYTES if kind == "scan" else MAX_REPORT_FILE_BYTES
        if content_length <= 0 or content_length > max_file:
            raise IntakeError(f"File size must be between 1 byte and {max_file // (1024 * 1024)} MB.")
        if kind == "scan" and suffix == ".json" and content_length > 10 * 1024 * 1024:
            raise IntakeError("BIDS JSON sidecars must be 10 MB or smaller.")
        prior_bytes = sum(int(item.get("size_bytes", 0)) for item in manifest.get("uploaded_files", []))
        if prior_bytes + content_length > MAX_SESSION_BYTES:
            raise IntakeError("The total upload size for this session exceeds the 8 GB limit.")

        upload_dir = session_dir / "uploads"
        target = upload_dir / safe_name
        if target.exists() or target.is_symlink():
            raise IntakeError(f"A file named '{safe_name}' has already been uploaded to this session.")

        digest = hashlib.sha256()
        written = 0
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as out:
                while written < content_length:
                    chunk = stream.read(min(CHUNK_BYTES, content_length - written))
                    if not chunk:
                        raise IntakeError("Upload ended before Content-Length bytes were received.")
                    written += len(chunk)
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        except Exception:
            target.unlink(missing_ok=True)
            raise

        record = {
            "filename": safe_name,
            "size_bytes": written,
            "sha256": digest.hexdigest(),
            "storage_path": f"uploads/{safe_name}",
            "uploaded_utc": _utc_now(),
        }
        manifest.setdefault("uploaded_files", []).append(record)

        if suffix == ".zip" and kind == "scan":
            extracted = self._extract_zip(session_dir, target)
            manifest.setdefault("extracted_files", []).extend(extracted)
        elif kind == "scan":
            data_target = session_dir / "data" / safe_name
            if data_target.exists() or data_target.is_symlink():
                raise IntakeError(f"A scan file named '{safe_name}' already exists.")
            os.replace(target, data_target)
            os.chmod(data_target, 0o600)
            record["storage_path"] = f"data/{safe_name}"

        manifest["status"] = "UPLOADED"
        self._save_manifest(session_dir, manifest)
        return {
            "session_id": session_id,
            "filename": safe_name,
            "size_bytes": written,
            "sha256": digest.hexdigest(),
            "status": manifest["status"],
        }

    def _extract_zip(self, session_dir: Path, archive_path: Path) -> List[Dict[str, Any]]:
        data_root = session_dir / "data"
        records: List[Dict[str, Any]] = []
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise IntakeError("The uploaded ZIP file is invalid or corrupted.") from exc
        with archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_FILES:
                raise IntakeError(f"ZIP contains more than {MAX_ZIP_FILES} entries.")
            declared_total = 0
            for info in members:
                name = info.filename
                if not name or "\\" in name or name.startswith("/"):
                    raise IntakeError("ZIP contains an unsafe absolute or malformed path.")
                pure = PurePosixPath(name)
                if any(part in ("", ".", "..") for part in pure.parts) or (pure.parts and ":" in pure.parts[0]):
                    raise IntakeError("ZIP contains a path traversal entry.")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise IntakeError("ZIP symlinks are not accepted.")
                if info.flag_bits & 0x1:
                    raise IntakeError("Encrypted ZIP files are not accepted.")
                if info.is_dir():
                    continue
                declared_total += info.file_size
                if info.file_size > MAX_SCAN_FILE_BYTES or declared_total > MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise IntakeError("ZIP expands beyond the configured safety limit.")
                if info.compress_size and info.file_size / info.compress_size > MAX_ZIP_RATIO:
                    raise IntakeError("ZIP compression ratio exceeds the safety limit.")

            extracted_total = 0
            for info in members:
                if info.is_dir():
                    continue
                relative = Path(*PurePosixPath(info.filename).parts)
                destination = data_root / relative
                resolved_parent = destination.parent.resolve()
                try:
                    resolved_parent.relative_to(data_root.resolve())
                except ValueError as exc:
                    raise IntakeError("ZIP entry escapes the intake data directory.") from exc
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if destination.exists() or destination.is_symlink():
                    raise IntakeError("ZIP contains duplicate file paths.")
                written = 0
                digest = hashlib.sha256()
                with archive.open(info, "r") as source:
                    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as out:
                        while True:
                            chunk = source.read(CHUNK_BYTES)
                            if not chunk:
                                break
                            written += len(chunk)
                            extracted_total += len(chunk)
                            if written > info.file_size or extracted_total > MAX_ZIP_UNCOMPRESSED_BYTES:
                                raise IntakeError("ZIP expanded beyond its declared safety limits.")
                            digest.update(chunk)
                            out.write(chunk)
                        out.flush()
                        os.fsync(out.fileno())
                if written != info.file_size:
                    raise IntakeError("ZIP entry size does not match its directory record.")
                records.append({
                    "relative_path": f"data/{relative.as_posix()}",
                    "size_bytes": written,
                    "sha256": digest.hexdigest(),
                })
        return records

    @staticmethod
    def _nifti_stem(path: Path) -> str:
        name = path.name
        return name[:-7] if name.lower().endswith(".nii.gz") else path.stem

    @staticmethod
    def _bids_entity(stem: str, key: str) -> Optional[str]:
        prefix = f"{key}-"
        for token in stem.split("_"):
            if token.startswith(prefix) and len(token) > len(prefix):
                return token[len(prefix):]
        return None

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_name(IntakeService._nifti_stem(path) + ".json")

    def _inspect_nifti(self, path: Path, expected_dims: int) -> Dict[str, Any]:
        try:
            import nibabel as nib
            image = nib.load(str(path))
            shape = tuple(int(value) for value in image.shape)
        except Exception as exc:
            raise IntakeError(f"Could not read NIfTI header for {path.name}: {type(exc).__name__}.") from exc
        if len(shape) != expected_dims:
            raise IntakeError(
                f"{path.name} must be {expected_dims}D; found a {len(shape)}D image."
            )
        return {"filename": path.name, "dimensions": list(shape), "dimensionality": len(shape)}

    def validate_scan_session(self, session_id: str) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "scan":
            raise IntakeError("This session is not a scan-upload session.")
        self._verify_staged_files(session_dir, manifest)
        data_root = session_dir / "data"
        nifti_files = [p for p in data_root.rglob("*") if p.is_file() and not p.is_symlink()
                       and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]
        bold_files = [p for p in nifti_files if self._nifti_stem(p).lower().endswith("_bold")]
        t1w_files = [p for p in nifti_files if self._nifti_stem(p).lower().endswith("_t1w")]
        dicom_files = [p for p in data_root.rglob("*") if p.is_file() and not p.is_symlink()
                       and (p.suffix.lower() in (".dcm", ".ima") or (not p.suffix and not p.name.startswith(".")))]

        findings: Dict[str, Any] = {
            "status": "INCOMPLETE_INPUTS",
            "bold": None,
            "t1w": None,
            "repetition_time_seconds": None,
            "dicom_file_count": len(dicom_files),
            "missing": [],
            "errors": [],
            "warnings": [],
            "processing_enabled": False,
            "real_data_authorized": False,
            "next_action": "Upload one BIDS-compatible resting-state BOLD NIfTI, its JSON sidecar, and one T1w NIfTI.",
        }

        if len(bold_files) > 1 or len(t1w_files) > 1:
            findings["errors"].append("More than one BOLD or T1w candidate was found. Upload one participant/session at a time.")
        elif bold_files and t1w_files:
            bold, t1w = bold_files[0], t1w_files[0]
            try:
                bold_stem = self._nifti_stem(bold)
                t1w_stem = self._nifti_stem(t1w)
                if not any(token.lower() == "task-rest" for token in bold_stem.split("_")):
                    raise IntakeError("BOLD input must be a resting-state file with the BIDS task-rest entity.")
                bold_subject = self._bids_entity(bold_stem, "sub")
                t1w_subject = self._bids_entity(t1w_stem, "sub")
                if not bold_subject or not t1w_subject or bold_subject != t1w_subject:
                    raise IntakeError("BOLD and T1w files must have the same BIDS subject identifier.")
                bold_session = self._bids_entity(bold_stem, "ses")
                t1w_session = self._bids_entity(t1w_stem, "ses")
                if bold_session and t1w_session and bold_session != t1w_session:
                    raise IntakeError("BOLD and T1w files have different BIDS session identifiers.")
                bold_info = self._inspect_nifti(bold, 4)
                if bold_info["dimensions"][3] < 50:
                    raise IntakeError(f"BOLD scan has {bold_info['dimensions'][3]} volumes; at least 50 are required.")
                t1w_info = self._inspect_nifti(t1w, 3)
                sidecar = self._sidecar(bold)
                if sidecar.is_symlink() or not sidecar.is_file():
                    raise IntakeError(f"Missing BOLD JSON sidecar: {sidecar.name}.")
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise IntakeError("BOLD JSON sidecar is invalid.") from exc
                tr = metadata.get("RepetitionTime") if isinstance(metadata, dict) else None
                if isinstance(tr, bool) or not isinstance(tr, (int, float)) or not 0.5 <= float(tr) <= 5.0:
                    raise IntakeError("BOLD JSON must contain RepetitionTime between 0.5 and 5.0 seconds.")
                findings["bold"] = {**bold_info, "relative_path": bold.relative_to(session_dir).as_posix()}
                findings["t1w"] = {**t1w_info, "relative_path": t1w.relative_to(session_dir).as_posix()}
                findings["repetition_time_seconds"] = float(tr)
                findings["status"] = "VALIDATED_WAITING_FOR_RESEARCH_AUTHORIZATION"
                findings["next_action"] = (
                    "Input structure and image headers passed intake validation. Real-scan preprocessing remains locked "
                    "until the research authorization register is approved."
                )
            except IntakeError as exc:
                findings["errors"].append(str(exc))
        else:
            if not bold_files:
                findings["missing"].append("4D resting-state BOLD NIfTI ending in _bold.nii or _bold.nii.gz")
            if not t1w_files:
                findings["missing"].append("3D anatomical T1w NIfTI ending in _T1w.nii or _T1w.nii.gz")
            if dicom_files:
                findings["status"] = "DICOM_SERIES_STAGED"
                findings["warnings"].append(
                    "DICOM files are staged in this session. Run series inspection and select T1w and BOLD series for automated BIDS conversion."
                )
                findings["next_action"] = (
                    "Inspect staged DICOM series using the DICOM series inspector, select T1w and BOLD series, and convert to BIDS."
                )

        if findings["errors"]:
            findings["status"] = "VALIDATION_FAILED"
        elif findings["missing"] and not dicom_files:
            findings["status"] = "INCOMPLETE_INPUTS"

        manifest["validation"] = findings
        manifest["status"] = findings["status"]
        self._save_manifest(session_dir, manifest)
        return {"session_id": session_id, "kind": "scan", "validation": findings}

    # =========================================================================
    # DICOM Inspection and Conversion
    # =========================================================================

    @staticmethod
    def _get_dcm2niix_version(bin_path: str) -> Optional[str]:
        try:
            res = subprocess.run([bin_path, "-v"], capture_output=True, text=True, timeout=5, check=False)
            output = (res.stdout + "\n" + res.stderr).strip()
            for line in output.splitlines():
                if "version" in line.lower() or line.strip().startswith("v1."):
                    return line.strip()
        except Exception:
            pass
        return "installed"

    @staticmethod
    def _sanitize_string(val: Any) -> str:
        if val is None:
            return ""
        # Clean special chars, reject non-ascii control chars
        cleaned = re.sub(r"[^\w\s\-_.,/()^]", "", str(val)).strip()
        return cleaned

    @staticmethod
    def _classify_series(
        desc: str,
        protocol: str,
        modality: str,
        files_count: int,
        tr: Optional[float],
        sidecar: Dict[str, Any],
    ) -> Tuple[str, bool, Optional[str]]:
        combined = f"{desc} {protocol}".lower()

        # Reject localizer / scout / calibration
        if any(term in combined for term in ("localizer", "scout", "survey", "aascout", "phoenix")):
            return "LOCALIZER", False, None
        if any(term in combined for term in ("dti", "dwi", "diff", "diffusion")):
            return "DWI", False, None

        # T1-weighted indicators
        is_t1 = any(term in combined for term in ("t1", "mprage", "spgr", "bravo", "tfl"))
        # Functional BOLD indicators
        is_bold = any(term in combined for term in ("bold", "rest", "fmri", "func", "ep2d"))

        if is_t1 and not is_bold:
            if files_count < 10 and sidecar.get("MRAcquisitionType") != "3D":
                return "T1w", True, "T1w candidate has fewer than 10 images or is not clearly 3D."
            return "T1w", False, None

        if is_bold and not is_t1:
            has_rest = "rest" in combined or "rs" in combined
            if not has_rest:
                return "BOLD", True, "Scan appears to be BOLD fMRI but does not explicitly designate resting-state ('rest'). Confirmation required."
            if tr is not None and not (0.5 <= tr <= 5.0):
                return "BOLD", True, f"RepetitionTime ({tr}s) is outside typical fMRI range [0.5, 5.0]s. Confirmation required."
            return "BOLD", False, None

        if is_t1 and is_bold:
            return "AMBIGUOUS", True, "Series description contains both T1 and functional/BOLD markers."

        return "OTHER", True, "Could not automatically classify series modality. Manual review required."

    def _format_series_record(self, series_item: Dict[str, Any], sidecar: Dict[str, Any]) -> Dict[str, Any]:
        s_num = series_item.get("SeriesNumber") or sidecar.get("SeriesNumber", 0)
        crc = str(series_item.get("CRC") or s_num)
        desc = self._sanitize_string(series_item.get("SeriesDescription") or sidecar.get("SeriesDescription", ""))
        protocol = self._sanitize_string(series_item.get("ProtocolName") or sidecar.get("ProtocolName", ""))
        modality = self._sanitize_string(series_item.get("Modality") or sidecar.get("Modality", "MR"))
        files_count = int(series_item.get("Files", 1))

        rows = series_item.get("Rows") or sidecar.get("AcquisitionMatrixPE") or sidecar.get("ReconMatrixPE")
        cols = series_item.get("Columns") or sidecar.get("BaseResolution")
        dims = [int(rows), int(cols)] if (rows and cols) else None

        tr = sidecar.get("RepetitionTime")
        if tr is not None:
            try:
                tr = round(float(tr), 4)
            except (ValueError, TypeError):
                tr = None

        te = sidecar.get("EchoTime")
        if te is not None:
            try:
                te = round(float(te), 4)
            except (ValueError, TypeError):
                te = None

        scan_type, is_ambiguous, reason = self._classify_series(desc, protocol, modality, files_count, tr, sidecar)

        # Redact patient identifiers strictly: never copy PatientName, PatientID, PatientBirthDate, etc.
        return {
            "series_id": crc,
            "series_number": int(s_num) if s_num is not None else 0,
            "series_description": desc or protocol or f"Series {s_num}",
            "protocol_name": protocol,
            "image_count": files_count,
            "dimensions": dims,
            "modality": modality,
            "repetition_time": tr,
            "echo_time": te,
            "candidate_scan_type": scan_type,
            "classification_ambiguous": is_ambiguous,
            "ambiguity_reason": reason,
        }

    def _parse_n1_stdout(
        self,
        stdout: str,
        sidecars_by_num: Dict[int, Dict[str, Any]],
        file_count: int,
    ) -> List[Dict[str, Any]]:
        records = []
        pattern = re.compile(r"^\s*(\d+)\s+(.+)$")
        for line in stdout.splitlines():
            m = pattern.match(line)
            if m:
                s_id = m.group(1)
                desc = Path(m.group(2).strip()).name
                s_num = int(s_id) if len(s_id) <= 6 else 1
                sc = sidecars_by_num.get(s_num, {})
                item = {
                    "SeriesNumber": s_num,
                    "CRC": s_id,
                    "SeriesDescription": desc,
                    "Files": file_count,
                }
                records.append(self._format_series_record(item, sc))
        return records

    def inspect_dicom_series(self, session_id: str) -> Dict[str, Any]:
        """Inspect staged DICOM files without converting them into NIfTI images."""
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "scan":
            raise IntakeError("Only scan intake sessions support DICOM inspection.")

        self._verify_staged_files(session_dir, manifest)
        data_root = session_dir / "data"

        dicom_files = [
            p for p in data_root.rglob("*")
            if p.is_file() and not p.is_symlink()
            and (p.suffix.lower() in (".dcm", ".ima") or (not p.suffix and not p.name.startswith(".")))
        ]

        dcm2niix_bin = shutil.which("dcm2niix")
        if not dcm2niix_bin:
            return {
                "session_id": session_id,
                "dcm2niix_available": False,
                "dcm2niix_version": None,
                "series_count": 0,
                "series": [],
                "errors": ["dcm2niix converter is not installed or not in PATH."],
            }

        if not dicom_files:
            return {
                "session_id": session_id,
                "dcm2niix_available": True,
                "dcm2niix_version": self._get_dcm2niix_version(dcm2niix_bin),
                "series_count": 0,
                "series": [],
                "errors": ["No DICOM files found in session."],
            }

        # Run dcm2niix -n -1 inspection
        try:
            inspect_proc = subprocess.run(
                [dcm2niix_bin, "-n", "-1", str(data_root)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise IntakeError("dcm2niix inspection timed out after 60 seconds.") from exc
        except OSError as exc:
            raise IntakeError(f"Failed to execute dcm2niix: {exc}") from exc

        # Extract structured series JSON & sidecars using temporary inspect directory
        series_records: List[Dict[str, Any]] = []
        with tempfile.TemporaryDirectory(dir=self.conversion_logs_dir, prefix=".inspect_") as tmp_inspect_str:
            tmp_inspect = Path(tmp_inspect_str)
            try:
                subprocess.run(
                    [dcm2niix_bin, "-n", "-2", "-o", str(tmp_inspect), str(data_root)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except Exception:
                pass

            try:
                subprocess.run(
                    [dcm2niix_bin, "-b", "o", "-ba", "y", "-o", str(tmp_inspect), str(data_root)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except Exception:
                pass

            series_json_path = tmp_inspect / "dcm2niix_series.json"
            series_json_data = []
            if series_json_path.is_file():
                try:
                    series_json_data = json.loads(series_json_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            sidecars_by_num: Dict[int, Dict[str, Any]] = {}
            for sidecar_p in tmp_inspect.glob("*.json"):
                if sidecar_p.name == "dcm2niix_series.json":
                    continue
                try:
                    sc_data = json.loads(sidecar_p.read_text(encoding="utf-8"))
                    s_num = sc_data.get("SeriesNumber")
                    if s_num is not None:
                        sidecars_by_num[int(s_num)] = sc_data
                except Exception:
                    pass

            if series_json_data and isinstance(series_json_data, list):
                for item in series_json_data:
                    s_num = item.get("SeriesNumber", 0)
                    sc = sidecars_by_num.get(int(s_num), {})
                    rec = self._format_series_record(item, sc)
                    series_records.append(rec)
            else:
                series_records = self._parse_n1_stdout(inspect_proc.stdout, sidecars_by_num, len(dicom_files))

        bold_candidates = [s for s in series_records if s.get("candidate_scan_type") == "BOLD"]
        if len(bold_candidates) > 1:
            for s in bold_candidates:
                s["classification_ambiguous"] = True
                s["ambiguity_reason"] = "Multiple BOLD candidates detected. User selection and confirmation required."

        result = {
            "session_id": session_id,
            "dcm2niix_available": True,
            "dcm2niix_version": self._get_dcm2niix_version(dcm2niix_bin),
            "series_count": len(series_records),
            "series": series_records,
            "errors": [],
        }

        manifest["dicom_inspection"] = result
        self._save_manifest(session_dir, manifest)
        return result

    def convert_dicom_to_bids(
        self,
        session_id: str,
        conversion_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert confirmed T1w and BOLD DICOM series into standardized BIDS NIfTI structure."""
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "scan":
            raise IntakeError("Only scan intake sessions support DICOM-to-BIDS conversion.")

        self._verify_staged_files(session_dir, manifest)
        data_root = session_dir / "data"

        dcm2niix_bin = shutil.which("dcm2niix")
        if not dcm2niix_bin:
            raise IntakeError("dcm2niix converter is not installed or not in PATH.")

        raw_subject = conversion_spec.get("subject_id", "")
        if not raw_subject or not isinstance(raw_subject, str):
            raise IntakeError("A de-identified BIDS subject identifier is required (e.g. 'sub-001').")
        sub_clean = re.sub(r"[^a-zA-Z0-9]", "", raw_subject)
        if sub_clean.lower().startswith("sub"):
            sub_clean = sub_clean[3:]
        if not sub_clean:
            raise IntakeError("BIDS subject identifier is invalid.")
        sub_entity = f"sub-{sub_clean}"

        raw_session = conversion_spec.get("session_id_label", "")
        ses_entity = ""
        if raw_session and isinstance(raw_session, str):
            ses_clean = re.sub(r"[^a-zA-Z0-9]", "", raw_session)
            if ses_clean.lower().startswith("ses"):
                ses_clean = ses_clean[3:]
            if ses_clean:
                ses_entity = f"ses-{ses_clean}"

        t1w_series = conversion_spec.get("t1w_series")
        if t1w_series is None:
            raise IntakeError("An anatomical T1w series must be selected for conversion.")

        bold_input = conversion_spec.get("bold_series")
        if not bold_input:
            raise IntakeError("At least one resting-state BOLD series must be selected for conversion.")

        bold_specs: List[Dict[str, Any]] = []
        if isinstance(bold_input, list):
            for idx, item in enumerate(bold_input):
                if isinstance(item, dict):
                    bold_specs.append(item)
                else:
                    bold_specs.append({"series_id": str(item), "run": idx + 1 if len(bold_input) > 1 else None, "task": "rest"})
        elif isinstance(bold_input, dict):
            bold_specs.append(bold_input)
        else:
            bold_specs.append({"series_id": str(bold_input), "run": None, "task": "rest"})

        confirm_ambiguous = bool(conversion_spec.get("confirm_ambiguous", False))
        if not confirm_ambiguous:
            inspection = manifest.get("dicom_inspection") or {}
            inspected_series: Dict[str, Dict[str, Any]] = {}
            for s in inspection.get("series", []):
                inspected_series[str(s.get("series_id"))] = s
                inspected_series[str(s.get("series_number"))] = s

            ambiguous_selected = []
            for s_id in [str(t1w_series)] + [str(b.get("series_id")) for b in bold_specs]:
                rec = inspected_series.get(s_id)
                if rec and rec.get("classification_ambiguous"):
                    ambiguous_selected.append(f"Series {s_id} ({rec.get('series_description')}: {rec.get('ambiguity_reason')})")

            if ambiguous_selected:
                raise IntakeError(
                    f"Confirmation required for ambiguous series: {'; '.join(ambiguous_selected)}. "
                    "Check the confirmation box to proceed."
                )

        sub_dir = self.bids_dir / sub_entity
        ses_dir = sub_dir / ses_entity if ses_entity else sub_dir
        anat_dir = ses_dir / "anat"
        func_dir = ses_dir / "func"
        anat_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        func_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        desc_path = self.bids_dir / "dataset_description.json"
        if not desc_path.is_file():
            _atomic_json(desc_path, {
                "Name": "NeuroADHD Converted BIDS Dataset",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "License": "Restricted Research Use Only",
                "Authors": ["NeuroADHD Automated DICOM Ingestion"],
            })

        converted_files: List[Dict[str, Any]] = []

        # Convert T1w series
        prefix_t1w = f"{sub_entity}_{ses_entity}_T1w" if ses_entity else f"{sub_entity}_T1w"
        with tempfile.TemporaryDirectory(dir=self.conversion_logs_dir, prefix=".conv_t1w_") as tmp_t1w_str:
            tmp_t1w = Path(tmp_t1w_str)
            cmd_t1w = [
                dcm2niix_bin,
                "-z", "y",
                "-b", "y",
                "-ba", "y",
                "-n", str(t1w_series),
                "-o", str(tmp_t1w),
                "-f", prefix_t1w,
                str(data_root),
            ]
            try:
                proc = subprocess.run(cmd_t1w, capture_output=True, text=True, timeout=120, check=False)
            except subprocess.TimeoutExpired as exc:
                raise IntakeError(f"dcm2niix conversion timed out for T1w series {t1w_series}.") from exc

            nii_candidates = list(tmp_t1w.glob("*.nii.gz"))
            if not nii_candidates:
                for uncompressed in tmp_t1w.glob("*.nii"):
                    gz_path = uncompressed.with_suffix(".nii.gz")
                    with uncompressed.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    nii_candidates.append(gz_path)

            if not nii_candidates:
                raise IntakeError(f"dcm2niix failed to produce T1w NIfTI for series {t1w_series}. (Return code: {proc.returncode})")

            target_t1w = anat_dir / f"{prefix_t1w}.nii.gz"
            shutil.copy2(nii_candidates[0], target_t1w)
            os.chmod(target_t1w, 0o600)
            converted_files.append({"type": "T1w_NIFTI", "path": str(target_t1w.relative_to(self.bids_dir))})

            json_candidates = list(tmp_t1w.glob("*.json"))
            if json_candidates:
                target_t1w_json = anat_dir / f"{prefix_t1w}.json"
                shutil.copy2(json_candidates[0], target_t1w_json)
                os.chmod(target_t1w_json, 0o600)
                converted_files.append({"type": "T1w_JSON", "path": str(target_t1w_json.relative_to(self.bids_dir))})

        # Convert each BOLD series
        target_bold = None
        target_json = None
        for idx, b_spec in enumerate(bold_specs):
            b_series_id = b_spec.get("series_id")
            run_num = b_spec.get("run")
            if run_num is None and len(bold_specs) > 1:
                run_num = idx + 1
            task_name = b_spec.get("task", "rest")
            run_entity = f"_run-{int(run_num):02d}" if run_num is not None else ""
            prefix_bold = f"{sub_entity}_{ses_entity}_task-{task_name}{run_entity}_bold" if ses_entity else f"{sub_entity}_task-{task_name}{run_entity}_bold"

            with tempfile.TemporaryDirectory(dir=self.conversion_logs_dir, prefix=".conv_bold_") as tmp_bold_str:
                tmp_bold = Path(tmp_bold_str)
                cmd_bold = [
                    dcm2niix_bin,
                    "-z", "y",
                    "-b", "y",
                    "-ba", "y",
                    "-n", str(b_series_id),
                    "-o", str(tmp_bold),
                    "-f", prefix_bold,
                    str(data_root),
                ]
                try:
                    proc = subprocess.run(cmd_bold, capture_output=True, text=True, timeout=120, check=False)
                except subprocess.TimeoutExpired as exc:
                    raise IntakeError(f"dcm2niix conversion timed out for BOLD series {b_series_id}.") from exc

                nii_candidates = list(tmp_bold.glob("*.nii.gz"))
                if not nii_candidates:
                    for uncompressed in tmp_bold.glob("*.nii"):
                        gz_path = uncompressed.with_suffix(".nii.gz")
                        with uncompressed.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        nii_candidates.append(gz_path)

                if not nii_candidates:
                    raise IntakeError(f"dcm2niix failed to produce BOLD NIfTI for series {b_series_id}. (Return code: {proc.returncode})")

                json_candidates = list(tmp_bold.glob("*.json"))
                if not json_candidates:
                    raise IntakeError(f"dcm2niix failed to produce BOLD JSON sidecar for series {b_series_id}.")

                sidecar_data = json.loads(json_candidates[0].read_text(encoding="utf-8"))
                tr = sidecar_data.get("RepetitionTime")
                if tr is None:
                    raise IntakeError(f"BOLD JSON sidecar for series {b_series_id} is missing required 'RepetitionTime'.")
                try:
                    tr_val = float(tr)
                    if not (0.5 <= tr_val <= 5.0):
                        raise IntakeError(f"RepetitionTime {tr_val}s is outside valid range [0.5, 5.0]s.")
                except (ValueError, TypeError) as exc:
                    raise IntakeError(f"Invalid RepetitionTime in BOLD JSON: {tr}") from exc

                if not sidecar_data.get("TaskName"):
                    sidecar_data["TaskName"] = task_name

                # Do NOT fabricate missing SliceTiming or PhaseEncodingDirection

                target_bold = func_dir / f"{prefix_bold}.nii.gz"
                target_json = func_dir / f"{prefix_bold}.json"

                shutil.copy2(nii_candidates[0], target_bold)
                os.chmod(target_bold, 0o600)

                _atomic_json(target_json, sidecar_data)
                os.chmod(target_json, 0o600)

                converted_files.append({"type": "BOLD_NIFTI", "path": str(target_bold.relative_to(self.bids_dir))})
                converted_files.append({"type": "BOLD_JSON", "path": str(target_json.relative_to(self.bids_dir))})

        if target_bold is None or target_json is None:
            raise IntakeError("No BOLD files were successfully converted.")

        val_result = self._validate_bids_output(target_t1w, target_bold, target_json)

        log_file = self.conversion_logs_dir / f"conversion_{_utc_now().replace(':', '-')}_{session_id}.json"
        log_payload = {
            "session_id": session_id,
            "timestamp_utc": _utc_now(),
            "subject_id": sub_entity,
            "session_id_label": ses_entity or None,
            "t1w_series": str(t1w_series),
            "bold_series": bold_specs,
            "bids_directory": str(ses_dir),
            "converted_files": converted_files,
            "validation_status": val_result["status"],
            "dcm2niix_version": self._get_dcm2niix_version(dcm2niix_bin),
        }
        _atomic_json(log_file, log_payload)

        zip_path = self.create_bids_session_zip(session_id, sub_entity, ses_entity)

        conversion_summary = {
            "session_id": session_id,
            "status": "CONVERSION_SUCCESSFUL",
            "subject_id": sub_entity,
            "session_id_label": ses_entity or None,
            "converted_files": converted_files,
            "bids_directory": str(ses_dir),
            "download_zip_filename": zip_path.name,
            "validation": val_result,
            "processing_enabled": False,
            "real_data_authorized": False,
            "notice": (
                "DICOM conversion and BIDS session validation completed. Real-scan preprocessing remains locked "
                "under research authorization."
            ),
        }

        manifest["dicom_conversion"] = conversion_summary
        manifest["validation"] = val_result
        manifest["status"] = val_result["status"]
        self._save_manifest(session_dir, manifest)
        return conversion_summary

    def _validate_bids_output(self, t1w_path: Path, bold_path: Path, bold_json_path: Path) -> Dict[str, Any]:
        findings: Dict[str, Any] = {
            "status": "VALIDATION_FAILED",
            "bold": None,
            "t1w": None,
            "repetition_time_seconds": None,
            "errors": [],
            "warnings": [],
            "processing_enabled": False,
            "real_data_authorized": False,
            "next_action": "BIDS conversion complete. Real-participant preprocessing remains locked under research authorization.",
        }

        try:
            bold_info = self._inspect_nifti(bold_path, 4)
            if bold_info["dimensions"][3] < 50:
                findings["errors"].append(f"BOLD scan has {bold_info['dimensions'][3]} volumes; at least 50 are required.")
            t1w_info = self._inspect_nifti(t1w_path, 3)

            sidecar = json.loads(bold_json_path.read_text(encoding="utf-8"))
            tr = float(sidecar.get("RepetitionTime", 0))
            if not (0.5 <= tr <= 5.0):
                findings["errors"].append(f"RepetitionTime {tr}s outside valid range [0.5, 5.0]s.")

            findings["bold"] = bold_info
            findings["t1w"] = t1w_info
            findings["repetition_time_seconds"] = tr

            if not findings["errors"]:
                findings["status"] = "VALIDATED_WAITING_FOR_RESEARCH_AUTHORIZATION"
        except Exception as exc:
            findings["errors"].append(str(exc))

        return findings

    def create_bids_session_zip(
        self,
        session_id: str,
        subject_id: str,
        session_id_label: Optional[str] = None,
    ) -> Path:
        """Package converted BIDS session into a clean downloadable ZIP archive."""
        sub_entity = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
        ses_entity = f"ses-{session_id_label}" if session_id_label and not session_id_label.startswith("ses-") else session_id_label

        sub_dir = self.bids_dir / sub_entity
        if not sub_dir.is_dir():
            raise IntakeError(f"BIDS directory for {sub_entity} not found.")

        zip_stem = f"{sub_entity}_{ses_entity}_bids" if ses_entity else f"{sub_entity}_bids"
        zip_path = self.validated_sessions_dir / f"{zip_stem}.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            desc_p = self.bids_dir / "dataset_description.json"
            if desc_p.is_file():
                archive.write(desc_p, arcname="dataset_description.json")

            for f in sub_dir.rglob("*"):
                if f.is_file() and not f.is_symlink():
                    rel_p = f.relative_to(self.bids_dir)
                    archive.write(f, arcname=str(rel_p))

        os.chmod(zip_path, 0o600)
        return zip_path

    def get_bids_session_zip(self, session_id: str) -> Tuple[bytes, str, str]:
        """Retrieve the validated BIDS session ZIP bytes for client download."""
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        conv = manifest.get("dicom_conversion")
        if not conv or not conv.get("download_zip_filename"):
            raise IntakeError("No converted BIDS session ZIP available for this session.")

        zip_name = _clean_filename(conv["download_zip_filename"])
        zip_path = self.validated_sessions_dir / zip_name
        if zip_path.is_symlink() or not zip_path.is_file():
            raise IntakeError("Validated BIDS session ZIP is missing or unsafe.")

        resolved = zip_path.resolve(strict=True)
        try:
            resolved.relative_to(self.validated_sessions_dir)
        except ValueError as exc:
            raise IntakeError("Session ZIP escapes validated sessions directory.") from exc

        return resolved.read_bytes(), "application/zip", zip_name

    # =========================================================================
    # Report Extraction Pipeline
    # =========================================================================

    def process_report_session(self, session_id: str) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "report":
            raise IntakeError("This session is not a report-upload session.")
        self._verify_staged_files(session_dir, manifest)
        extracted = []
        report_dir = session_dir / "reports"
        if report_dir.is_symlink():
            raise IntakeError("Report output directory is a symlink and is not safe.")
        report_dir.mkdir(exist_ok=True, mode=0o700)
        try:
            report_dir.resolve(strict=True).relative_to(session_dir)
        except ValueError as exc:
            raise IntakeError("Report output directory escapes its intake session.") from exc
        for record in manifest.get("uploaded_files", []):
            path = session_dir / record["storage_path"]
            if path.is_symlink() or not path.is_file():
                raise IntakeError("A report file is missing or unsafe.")
            suffix = _suffix(path.name)
            text = ""
            method = "NOT_AVAILABLE"
            if suffix == ".pdf":
                text, method = self._extract_pdf(path, report_dir)
            elif suffix in (".jpg", ".jpeg", ".png"):
                text, method = self._extract_image(path)
            text_name = None
            if text.strip():
                text_name = f"{path.stem}-{record['sha256'][:10]}.extracted.txt"
                text_path = report_dir / text_name
                if text_path.is_symlink():
                    raise IntakeError("Extracted report target is a symlink.")
                fd, temporary_name = tempfile.mkstemp(prefix=".extract-", suffix=".tmp", dir=report_dir)
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as out:
                        out.write(text)
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(temporary_name, text_path)
                    os.chmod(text_path, 0o600)
                except Exception:
                    try:
                        os.unlink(temporary_name)
                    except OSError:
                        pass
                    raise
            extracted.append({
                "filename": path.name,
                "status": "TEXT_EXTRACTED" if text.strip() else method,
                "text_characters": len(text),
                "text_artifact": f"reports/{text_name}" if text_name else None,
            })
        result = {
            "status": "REPORT_EXTRACTION_COMPLETE" if any(x["text_characters"] for x in extracted)
                      else "REPORTS_STAGED_ENGINE_REQUIRED",
            "files": extracted,
            "engines": {
                "pdftotext_available": bool(shutil.which("pdftotext")),
                "tesseract_available": bool(shutil.which("tesseract")),
                "pdftoppm_available": bool(shutil.which("pdftoppm")),
            },
            "notice": "Report text is extracted for research review only. It is not used as fMRI input and is not a diagnosis.",
        }
        manifest["report_extraction"] = result
        manifest["status"] = result["status"]
        self._save_manifest(session_dir, manifest)
        return {"session_id": session_id, "kind": "report", "report_extraction": result}

    def _extract_pdf(self, path: Path, report_dir: Path) -> Tuple[str, str]:
        pdftotext = shutil.which("pdftotext")
        if pdftotext:
            try:
                result = subprocess.run(
                    [pdftotext, "-layout", str(path), "-"],
                    capture_output=True, text=True, timeout=45, check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout[:2_000_000], "PDF_TEXT_LAYER"
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path), strict=False)
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages[:100])
            if text.strip():
                return text[:2_000_000], "PDF_TEXT_LAYER"
        except Exception:
            pass

        pdftoppm, tesseract = shutil.which("pdftoppm"), shutil.which("tesseract")
        if pdftoppm and tesseract:
            with tempfile.TemporaryDirectory(prefix="neuroadhd-pdf-") as temp_dir:
                prefix = Path(temp_dir) / "page"
                try:
                    render = subprocess.run(
                        [pdftoppm, "-f", "1", "-l", "5", "-scale-to", "1800", "-png", str(path), str(prefix)],
                        capture_output=True, timeout=60, check=False,
                    )
                    if render.returncode == 0:
                        texts = []
                        for image_path in sorted(Path(temp_dir).glob("page-*.png")):
                            try:
                                ocr = subprocess.run(
                                    [tesseract, str(image_path), "stdout"],
                                    capture_output=True, text=True, timeout=45, check=False,
                                )
                            except (OSError, subprocess.TimeoutExpired):
                                continue
                            if ocr.returncode == 0:
                                texts.append(ocr.stdout)
                        joined = "\n\n".join(texts)
                        if joined.strip():
                            return joined[:2_000_000], "PDF_PAGE_OCR_FIRST_FIVE"
                except (OSError, subprocess.TimeoutExpired):
                    pass
        return "", "PDF_TEXT_OR_OCR_ENGINE_NOT_AVAILABLE"

    def _extract_image(self, path: Path) -> Tuple[str, str]:
        tesseract = shutil.which("tesseract")
        if not tesseract:
            return "", "TESSERACT_NOT_AVAILABLE"
        try:
            result = subprocess.run(
                [tesseract, str(path), "stdout"],
                capture_output=True, text=True, timeout=45, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "", "OCR_FAILED"
        if result.returncode != 0:
            return "", "OCR_FAILED"
        return result.stdout[:2_000_000], "IMAGE_OCR"

    def get_session(self, session_id: str) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        return manifest

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for search_dir in (self.staged_sessions_dir, self.data_root):
            if not search_dir.is_dir():
                continue
            for candidate in search_dir.iterdir():
                if candidate.is_symlink() or not candidate.is_dir() or not SESSION_RE.fullmatch(candidate.name):
                    continue
                try:
                    manifest = self._read_manifest(candidate)
                    if any(s["session_id"] == manifest["session_id"] for s in sessions):
                        continue
                    sessions.append({
                        "session_id": manifest["session_id"],
                        "kind": manifest["kind"],
                        "status": manifest["status"],
                        "created_utc": manifest["created_utc"],
                        "uploaded_file_count": len(manifest.get("uploaded_files", [])),
                    })
                except IntakeError:
                    continue
        return sorted(sessions, key=lambda item: item["created_utc"], reverse=True)

    def get_report_artifact(self, session_id: str, filename: str) -> Tuple[bytes, str]:
        """Return only a generated text-extraction artifact, never the original report."""
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "report":
            raise IntakeError("Only report sessions contain extracted report artifacts.")
        safe_name = _clean_filename(filename)
        if not safe_name.endswith(".extracted.txt"):
            raise IntakeError("Only extracted text artifacts can be downloaded.")
        report_root = session_dir / "reports"
        if report_root.is_symlink() or not report_root.is_dir():
            raise IntakeError("Extracted report directory is missing or unsafe.")
        candidate = report_root / safe_name
        if candidate.is_symlink() or not candidate.is_file():
            raise IntakeError("Extracted report artifact was not found.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to((session_dir / "reports").resolve(strict=True))
        except ValueError as exc:
            raise IntakeError("Report artifact escapes its storage directory.") from exc
        if resolved.stat().st_size > 2_000_000:
            raise IntakeError("Extracted report artifact exceeds the download limit.")
        return resolved.read_bytes(), "text/plain; charset=utf-8"
