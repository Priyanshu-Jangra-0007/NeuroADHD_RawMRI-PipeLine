"""Safe local intake for MRI sessions and separate radiology-report extraction.

This module stages and validates uploads only. It never runs the real-scan
preprocessing pipeline or the ADHD model; those remain behind the research
authorization gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional

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


class IntakeError(RuntimeError):
    """Raised for invalid or unsafe intake operations."""


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
    """Local, path-contained intake and validation service."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def _session_dir(self, session_id: str) -> Path:
        if not SESSION_RE.fullmatch(session_id):
            raise IntakeError("Invalid intake session identifier.")
        candidate = self.root / session_id
        if candidate.is_symlink() or not candidate.is_dir():
            raise IntakeError("Intake session was not found or is not a safe directory.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
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
        session_dir = self.root / session_id
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
        }
        self._save_manifest(session_dir, manifest)
        return {"session_id": session_id, "kind": kind, "status": manifest["status"]}

    def upload_file(
        self,
        session_id: str,
        filename: str,
        stream: BinaryIO,
        content_length: int,
    ) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("status") not in ("COLLECTING", "UPLOADED", "VALIDATION_INCOMPLETE"):
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
        data_root = session_dir / "data"
        nifti_files = [p for p in data_root.rglob("*") if p.is_file() and not p.is_symlink()
                       and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]
        bold_files = [p for p in nifti_files if self._nifti_stem(p).lower().endswith("_bold")]
        t1w_files = [p for p in nifti_files if self._nifti_stem(p).lower().endswith("_t1w")]
        dicom_files = [p for p in data_root.rglob("*") if p.is_file() and not p.is_symlink()
                       and p.suffix.lower() in (".dcm", ".ima")]

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
                    "DICOM files were staged, but automatic series identification and DICOM-to-NIfTI conversion "
                    "are not performed in this intake-only release."
                )
                findings["next_action"] = (
                    "Provide a BIDS-compatible NIfTI pair, or have the DICOM series converted and reviewed with dcm2niix "
                    "before adding the resulting BOLD/T1w files."
                )

        if findings["errors"]:
            findings["status"] = "VALIDATION_FAILED"
        elif findings["missing"] and not dicom_files:
            findings["status"] = "INCOMPLETE_INPUTS"

        manifest["validation"] = findings
        manifest["status"] = findings["status"]
        self._save_manifest(session_dir, manifest)
        return {"session_id": session_id, "kind": "scan", "validation": findings}

    def process_report_session(self, session_id: str) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "report":
            raise IntakeError("This session is not a report-upload session.")
        extracted = []
        report_dir = session_dir / "reports"
        report_dir.mkdir(exist_ok=True, mode=0o700)
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
                text_name = f"{path.stem}.extracted.txt"
                text_path = report_dir / text_name
                text_path.write_text(text, encoding="utf-8")
                os.chmod(text_path, 0o600)
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

    def _extract_pdf(self, path: Path, report_dir: Path) -> tuple[str, str]:
        pdftotext = shutil.which("pdftotext")
        if pdftotext:
            result = subprocess.run(
                [pdftotext, "-layout", str(path), "-"],
                capture_output=True, text=True, timeout=45, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout[:2_000_000], "PDF_TEXT_LAYER"
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
                render = subprocess.run(
                    [pdftoppm, "-f", "1", "-l", "5", "-scale-to", "1800", "-png", str(path), str(prefix)],
                    capture_output=True, timeout=60, check=False,
                )
                if render.returncode == 0:
                    texts = []
                    for image_path in sorted(Path(temp_dir).glob("page-*.png")):
                        ocr = subprocess.run(
                            [tesseract, str(image_path), "stdout"],
                            capture_output=True, text=True, timeout=45, check=False,
                        )
                        if ocr.returncode == 0:
                            texts.append(ocr.stdout)
                    joined = "\n\n".join(texts)
                    if joined.strip():
                        return joined[:2_000_000], "PDF_PAGE_OCR_FIRST_FIVE"
        return "", "PDF_TEXT_OR_OCR_ENGINE_NOT_AVAILABLE"

    def _extract_image(self, path: Path) -> tuple[str, str]:
        tesseract = shutil.which("tesseract")
        if not tesseract:
            return "", "TESSERACT_NOT_AVAILABLE"
        result = subprocess.run(
            [tesseract, str(path), "stdout"],
            capture_output=True, text=True, timeout=45, check=False,
        )
        if result.returncode != 0:
            return "", "OCR_FAILED"
        return result.stdout[:2_000_000], "IMAGE_OCR"

    def get_session(self, session_id: str) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        return manifest

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for candidate in self.root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir() or not SESSION_RE.fullmatch(candidate.name):
                continue
            try:
                manifest = self._read_manifest(candidate)
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


    def get_report_artifact(self, session_id: str, filename: str) -> tuple[bytes, str]:
        """Return only a generated text-extraction artifact, never the original report."""
        session_dir = self._session_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if manifest.get("kind") != "report":
            raise IntakeError("Only report sessions contain extracted report artifacts.")
        safe_name = _clean_filename(filename)
        if not safe_name.endswith(".extracted.txt"):
            raise IntakeError("Only extracted text artifacts can be downloaded.")
        candidate = session_dir / "reports" / safe_name
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
