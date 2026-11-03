"""Read-only data access for the local NeuroADHD research dashboard.

The service exposes only synthetic cohort outputs generated under outputs/dashboard_cohorts.
It rejects non-synthetic manifests, path traversal, symlinks and unlisted artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

COHORT_ID_RE = re.compile(r"^cohort_[A-Za-z0-9_TZ]+$")
SUBJECT_ID_RE = re.compile(r"^sub-synthetic[0-9]{3}$")
EDGE_RE = re.compile(r"^ROI_([0-9]{3})__ROI_([0-9]{3})$")
ARTIFACTS = {
    "qc_registration_epi_to_t1w.png": "image/png",
    "qc_registration_t1w_to_mni.png": "image/png",
    "qc_tissue_masks.png": "image/png",
    "qc_parcellation_cc200.png": "image/png",
    "roi_CC200.csv": "text/csv; charset=utf-8",
    "features_19900_fisher_z.csv": "text/csv; charset=utf-8",
    "motion_parameters.tsv": "text/tab-separated-values; charset=utf-8",
    "framewise_displacement.tsv": "text/tab-separated-values; charset=utf-8",
    "confounds.tsv": "text/tab-separated-values; charset=utf-8",
    "confound_report.json": "application/json",
    "provenance.json": "application/json",
    "model_prediction.json": "application/json",
}
MAX_CONNECTIVITY_BYTES = 8 * 1024 * 1024


class DashboardDataError(RuntimeError):
    """Raised when a dashboard request refers to invalid or unsafe run data."""


class DashboardDataService:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _json(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise DashboardDataError(f"Cannot read valid JSON from {path.name}.") from exc
        if not isinstance(value, dict):
            raise DashboardDataError(f"{path.name} must contain a JSON object.")
        return value

    def _cohort_dir(self, cohort_id: str) -> Path:
        if not COHORT_ID_RE.fullmatch(cohort_id):
            raise DashboardDataError("Invalid cohort identifier.")
        candidate = self.output_root / cohort_id
        if candidate.is_symlink() or not candidate.is_dir():
            raise DashboardDataError("Cohort was not found or is not a safe directory.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise DashboardDataError("Cohort path escapes the dashboard output root.") from exc
        return resolved

    def _load_manifest(self, cohort_id: str) -> Dict[str, Any]:
        cohort_dir = self._cohort_dir(cohort_id)
        manifest_path = cohort_dir / "cohort_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DashboardDataError("Cohort manifest is missing or unsafe.")
        manifest = self._json(manifest_path)
        if manifest.get("execution_policy") != "STRICTLY_SYNTHETIC_ONLY":
            raise DashboardDataError("Dashboard only displays strictly synthetic cohorts.")
        if manifest.get("real_data_authorized") is not False:
            raise DashboardDataError("Cohort manifest has an invalid real-data authorization state.")
        if manifest.get("run_type") != "SYNTHETIC_COHORT_ONLY":
            raise DashboardDataError("Unexpected cohort run type.")
        if not isinstance(manifest.get("subjects"), list):
            raise DashboardDataError("Cohort manifest has no valid subject list.")
        return manifest

    def list_cohorts(self) -> List[Dict[str, Any]]:
        cohorts = []
        for candidate in self.output_root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir() or not COHORT_ID_RE.fullmatch(candidate.name):
                continue
            try:
                manifest = self._load_manifest(candidate.name)
            except DashboardDataError:
                continue
            summary = manifest.get("summary", {})
            subjects = manifest["subjects"]
            cohorts.append({
                "cohort_run_id": manifest.get("cohort_run_id", candidate.name),
                "cohort_id": candidate.name,
                "status": manifest.get("status", "UNKNOWN"),
                "created_utc": manifest.get("created_utc"),
                "updated_utc": manifest.get("updated_utc"),
                "total_subjects": summary.get("total_subjects", len(subjects)),
                "succeeded": summary.get("succeeded", sum(x.get("status") == "SUCCESS" for x in subjects)),
                "failed": summary.get("failed", sum(x.get("status") == "FAILED" for x in subjects)),
                "pending": summary.get("pending", sum(x.get("status") in ("PENDING", "RUNNING") for x in subjects)),
                "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
                "real_data_authorized": False,
            })
        return sorted(cohorts, key=lambda x: x.get("created_utc") or "", reverse=True)

    def get_cohort(self, cohort_id: str) -> Dict[str, Any]:
        manifest = self._load_manifest(cohort_id)
        # Do not expose absolute filesystem paths from the persisted manifest.
        public_subjects = []
        for subject in manifest["subjects"]:
            public_subjects.append({
                key: value for key, value in subject.items()
                if key not in ("run_output_dir", "input_sha256", "synthetic_manifest_sha256")
            })
        return {
            "cohort_run_id": manifest.get("cohort_run_id"),
            "cohort_id": cohort_id,
            "status": manifest.get("status"),
            "created_utc": manifest.get("created_utc"),
            "updated_utc": manifest.get("updated_utc"),
            "configuration": manifest.get("configuration", {}),
            "summary": manifest.get("summary", {}),
            "subjects": public_subjects,
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
        }

    def _subject_record(self, cohort_id: str, subject_id: str):
        if not SUBJECT_ID_RE.fullmatch(subject_id):
            raise DashboardDataError("Invalid synthetic subject identifier.")
        manifest = self._load_manifest(cohort_id)
        records = [x for x in manifest["subjects"] if x.get("subject_id") == subject_id]
        if len(records) != 1:
            raise DashboardDataError("Synthetic subject was not found in this cohort.")
        return self._cohort_dir(cohort_id), records[0]

    def _run_dir(self, cohort_id: str, subject_id: str) -> Path:
        cohort_dir, record = self._subject_record(cohort_id, subject_id)
        if record.get("status") != "SUCCESS" or not record.get("run_output_dir"):
            raise DashboardDataError("This subject does not have a completed successful run yet.")
        raw = Path(record["run_output_dir"])
        if raw.is_symlink():
            raise DashboardDataError("Run directory symlinks are not permitted.")
        run_dir = raw.resolve(strict=True)
        allowed_root = (cohort_dir / "runs").resolve()
        try:
            run_dir.relative_to(allowed_root)
        except ValueError as exc:
            raise DashboardDataError("Run output path escapes this cohort's runs directory.") from exc
        if not run_dir.is_dir():
            raise DashboardDataError("Run output directory is missing.")
        return run_dir

    def _artifact_path(self, cohort_id: str, subject_id: str, filename: str) -> Path:
        if filename not in ARTIFACTS:
            raise DashboardDataError("Artifact is not available through the dashboard.")
        run_dir = self._run_dir(cohort_id, subject_id)
        candidate = run_dir / filename
        if candidate.is_symlink() or not candidate.is_file():
            raise DashboardDataError("Artifact is missing or unsafe.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(run_dir)
        except ValueError as exc:
            raise DashboardDataError("Artifact path escapes the run directory.") from exc

        manifest = self._json(run_dir / "output_manifest.json")
        entries = manifest.get("artifacts", {})
        expected = next(
            (item.get("sha256") for item in entries.values() if item.get("filename") == filename),
            None,
        )
        if not isinstance(expected, str):
            raise DashboardDataError("Artifact is not listed in the output integrity manifest.")
        actual = self._sha256(resolved)
        if not hmac.compare_digest(actual, expected):
            raise DashboardDataError("Artifact integrity verification failed.")
        return resolved

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def get_artifact(self, cohort_id: str, subject_id: str, filename: str) -> tuple[bytes, str]:
        path = self._artifact_path(cohort_id, subject_id, filename)
        if path.stat().st_size > 25 * 1024 * 1024:
            raise DashboardDataError("Artifact exceeds the dashboard download limit.")
        return path.read_bytes(), ARTIFACTS[filename]

    @staticmethod
    def _read_numeric_column(path: Path, column_name: str, delimiter: str = "\t") -> List[float]:
        values = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            if column_name not in (reader.fieldnames or []):
                raise DashboardDataError(f"Expected data column {column_name} is missing.")
            for row in reader:
                try:
                    value = float(row[column_name])
                except (TypeError, ValueError) as exc:
                    raise DashboardDataError("Numeric artifact contains invalid values.") from exc
                if not math.isfinite(value):
                    raise DashboardDataError("Numeric artifact contains non-finite values.")
                values.append(value)
        if not values:
            raise DashboardDataError("Numeric artifact is empty.")
        return values

    def get_subject_detail(self, cohort_id: str, subject_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(cohort_id, subject_id)
        provenance = self._json(self._artifact_path(cohort_id, subject_id, "provenance.json"))
        confound_report = self._json(self._artifact_path(cohort_id, subject_id, "confound_report.json"))
        fd = self._read_numeric_column(
            self._artifact_path(cohort_id, subject_id, "framewise_displacement.tsv"),
            "framewise_displacement_mm",
        )
        gates = provenance.get("validation_gates", {})
        hardware = provenance.get("hardware_execution", {})
        gate1 = gates.get("gate1_resources", {})
        gate2 = gates.get("gate2_bids_input", {})
        gate5 = gates.get("gate5_roi_coverage", {})
        gate6 = gates.get("gate6_feature_schema_conformance", {})
        record_dir = self._subject_record(cohort_id, subject_id)[1]
        model_prediction = None
        try:
            model_prediction = self._json(self._artifact_path(cohort_id, subject_id, "model_prediction.json"))
            model_prediction = {
                key: model_prediction.get(key)
                for key in ("class_label", "adhd_probability", "control_probability", "evaluation_mode", "diagnostic_disclaimer")
            }
        except DashboardDataError:
            pass

        motion = fd
        max_points = 600
        if len(motion) > max_points:
            stride = math.ceil(len(motion) / max_points)
            motion = motion[::stride]
        return {
            "subject_id": subject_id,
            "status": record_dir.get("status"),
            "runtime_seconds": record_dir.get("runtime_seconds") or hardware.get("runtime_seconds"),
            "peak_rss_gb": record_dir.get("peak_rss_gb") or hardware.get("peak_rss_gb"),
            "tr_seconds": gate2.get("tr_seconds"),
            "n_volumes": gate2.get("t_vol"),
            "mean_fd_mm": sum(fd) / len(fd),
            "max_fd_mm": max(fd),
            "fd_series": motion,
            "n_rois": gate5.get("n_rois"),
            "zero_variance_rois": gate5.get("zero_variance_rois"),
            "n_connectivity_features": gate6.get("n_features"),
            "confound_columns": confound_report.get("num_columns"),
            "confound_rank": confound_report.get("matrix_rank"),
            "confound_condition_number": confound_report.get("condition_number"),
            "validation_gates": gates,
            "qc_threshold_status": "QC_THRESHOLDS_PENDING_SUPERVISOR_APPROVAL",
            "model_prediction": model_prediction,
            "execution_mode": "SYNTHETIC_DEMONSTRATION_ONLY",
            "diagnostic_disclaimer": "Synthetic research demonstration only; not clinically validated and not for diagnosis.",
        }

    def get_timeseries(self, cohort_id: str, subject_id: str, roi: int) -> Dict[str, Any]:
        if not 1 <= int(roi) <= 200:
            raise DashboardDataError("ROI must be between 1 and 200.")
        path = self._artifact_path(cohort_id, subject_id, "roi_CC200.csv")
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or reader.fieldnames[0] != "node/volume":
                raise DashboardDataError("ROI time-series CSV has an unexpected orientation.")
            matches = [row for row in reader if row.get("node/volume") == str(int(roi))]
        if len(matches) != 1:
            raise DashboardDataError(f"Expected exactly one time series for ROI {roi}.")
        row = matches[0]
        values = []
        for name in list(row.keys())[1:]:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise DashboardDataError("ROI time series contains invalid values.") from exc
            if not math.isfinite(value):
                raise DashboardDataError("ROI time series contains non-finite values.")
            values.append(value)
        return {"roi": int(roi), "n_timepoints": len(values), "values": values}

    def get_connectivity(self, cohort_id: str, subject_id: str) -> Dict[str, Any]:
        path = self._artifact_path(cohort_id, subject_id, "features_19900_fisher_z.csv")
        if path.stat().st_size > MAX_CONNECTIVITY_BYTES:
            raise DashboardDataError("Connectivity artifact exceeds the dashboard parsing limit.")
        matrix = [[0.0 for _ in range(200)] for _ in range(200)]
        seen = set()
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["feature_name", "fisher_z"]:
                raise DashboardDataError("Connectivity CSV has an unexpected schema.")
            for row in reader:
                match = EDGE_RE.fullmatch(row.get("feature_name", ""))
                if not match:
                    raise DashboardDataError("Connectivity edge name is malformed.")
                left, right = int(match.group(1)), int(match.group(2))
                if not (1 <= left < right <= 200):
                    raise DashboardDataError("Connectivity edge has invalid ROI indices.")
                pair = (left, right)
                if pair in seen:
                    raise DashboardDataError("Connectivity artifact contains duplicate edges.")
                seen.add(pair)
                try:
                    value = float(row["fisher_z"])
                except (TypeError, ValueError) as exc:
                    raise DashboardDataError("Connectivity contains invalid numeric values.") from exc
                if not math.isfinite(value):
                    raise DashboardDataError("Connectivity contains non-finite values.")
                matrix[left - 1][right - 1] = value
                matrix[right - 1][left - 1] = value
        if len(seen) != 19900:
            raise DashboardDataError(f"Expected 19,900 connectivity edges; found {len(seen)}.")
        return {
            "n_rois": 200,
            "n_edges": len(seen),
            "matrix": matrix,
            "display": "Fisher-z transformed Pearson connectivity; symmetric display matrix with zero diagonal.",
        }
