#!/usr/bin/env python3
"""Run a reproducible, synthetic-only cohort through the existing NeuroADHD pipeline.

This orchestrator deliberately reuses scripts.run_poc_v2 rather than duplicating
MRI preprocessing. It processes one synthetic subject at a time to keep memory
usage bounded and writes an auditable cohort manifest after every state change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_poc_v2 import (  # noqa: E402
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from src.validation.authorization_gate import ResearchAuthorizationGate  # noqa: E402

MAX_COHORT_SIZE = 20
MAX_MEMORY_GB = 2.5
MANIFEST_NAME = "cohort_manifest.json"
MANIFEST_VERSION = "1.0.0"
VALID_COHORT_PROFILES = ("standard", "pilot_n6")

# Step 9 N=6 technical feasibility synthetic dry-run profiles.
# These exercise diverse valid synthetic acquisition parameters without clinical claims or real IDs.
PILOT_N6_PROFILES: List[Dict[str, Any]] = [
    {
        "pilot_slot": "sub-pilot-01",
        "label": "interleaved_k_direct_hz_pe_j",
        "tr": 2.0,
        "include_slice_timing": True,
        "slice_encoding_direction": "k",
        "phase_encoding_direction": "j",
        "total_readout_time": 0.05,
        "fieldmap_mode": "direct_hz",
    },
    {
        "pilot_slot": "sub-pilot-02",
        "label": "descending_k_minus_direct_hz_pe_j_minus",
        "tr": 2.0,
        "include_slice_timing": True,
        "slice_encoding_direction": "k-",
        "phase_encoding_direction": "j-",
        "total_readout_time": 0.045,
        "fieldmap_mode": "direct_hz",
    },
    {
        "pilot_slot": "sub-pilot-03",
        "label": "ascending_k_no_fieldmap_skip_control",
        "tr": 2.0,
        "include_slice_timing": True,
        "slice_encoding_direction": "k",
        "phase_encoding_direction": None,
        "total_readout_time": None,
        "fieldmap_mode": None,
    },
    {
        "pilot_slot": "sub-pilot-04",
        "label": "missing_slice_timing_skip_control_direct_hz",
        "tr": 2.0,
        "include_slice_timing": False,
        "slice_encoding_direction": "k",
        "phase_encoding_direction": "j",
        "total_readout_time": 0.05,
        "fieldmap_mode": "direct_hz",
    },
    {
        "pilot_slot": "sub-pilot-05",
        "label": "interleaved_k_pepolar_tool_check",
        "tr": 2.0,
        "include_slice_timing": True,
        "slice_encoding_direction": "k",
        "phase_encoding_direction": "j",
        "total_readout_time": 0.05,
        "fieldmap_mode": "pepolar",
    },
    {
        "pilot_slot": "sub-pilot-06",
        "label": "custom_tr_2_5s_axis_i_direct_rads",
        "tr": 2.5,
        "include_slice_timing": True,
        "slice_encoding_direction": "i",
        "phase_encoding_direction": "i",
        "total_readout_time": 0.04,
        "fieldmap_mode": "direct_rad_s",
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Replace a JSON manifest atomically so interrupted runs do not truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _normalise_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalised = dict(config)
    for key in ("canonical_atlas_path", "model_dir"):
        value = normalised.get(key)
        normalised[key] = str(Path(value).expanduser().resolve()) if value else None
    return normalised


def _validate_config(config: Dict[str, Any]) -> None:
    n_subjects = int(config["n_subjects"])
    t_vol = int(config["t_vol"])
    memory_gb = float(config["max_memory_gb"])
    if not 1 <= n_subjects <= MAX_COHORT_SIZE:
        raise ValueError(f"n_subjects must be between 1 and {MAX_COHORT_SIZE}.")
    if t_vol < 50:
        raise ValueError("t_vol must be at least 50 for the existing pipeline.")
    if not 0 < memory_gb <= MAX_MEMORY_GB:
        raise ValueError(f"max_memory_gb must be > 0 and <= {MAX_MEMORY_GB}.")
    if config["atlas_mode"] == "canonical" and not config.get("canonical_atlas_path"):
        raise ValueError("canonical_atlas_path is required in canonical atlas mode.")
    if config["require_model_inference"] and not config.get("model_dir"):
        raise ValueError("model_dir is required when require_model_inference is enabled.")
    profile = config.get("cohort_profile", "standard")
    if profile not in VALID_COHORT_PROFILES:
        raise ValueError(f"Unsupported cohort_profile '{profile}'.")
    if profile == "pilot_n6" and n_subjects != len(PILOT_N6_PROFILES):
        raise ValueError(f"pilot_n6 profile requires n_subjects={len(PILOT_N6_PROFILES)}.")


def _detect_host_tool_availability() -> Dict[str, bool]:
    return {
        "dcm2niix": shutil.which("dcm2niix") is not None,
        "pydeface": shutil.which("pydeface") is not None,
        "fsl_flirt": shutil.which("flirt") is not None,
        "fsl_topup": shutil.which("topup") is not None,
        "fsl_prelude": shutil.which("prelude") is not None,
        "fsl_fugue": shutil.which("fugue") is not None,
        "afni_3dtshift": shutil.which("3dTshift") is not None,
    }


def _extract_subject_stage_summary(result: Dict[str, Any], run_output_dir: Path) -> Dict[str, Any]:
    """Extract concise non-clinical stage, QC, and missing-tool telemetry from a subject run."""
    provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    if not provenance:
        prov_path = run_output_dir / "provenance.json"
        if prov_path.is_file() and not prov_path.is_symlink():
            try:
                provenance = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                provenance = {}

    gates = provenance.get("validation_gates") or {}
    gate2 = gates.get("gate2_bids_input") or {}
    gate5 = gates.get("gate5_roi_coverage") or {}
    gate6 = gates.get("gate6_feature_schema_conformance") or {}
    stc = result.get("slice_timing") or provenance.get("slice_timing_correction") or {}
    sdc = result.get("susceptibility_distortion") or provenance.get("susceptibility_distortion_correction") or {}
    defacing = result.get("defacing") or provenance.get("defacing") or {}

    mean_fd_mm: Optional[float] = None
    max_fd_mm: Optional[float] = None
    fd_path = run_output_dir / "framewise_displacement.tsv"
    if fd_path.is_file() and not fd_path.is_symlink():
        try:
            with fd_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                vals = [float(row["framewise_displacement_mm"]) for row in reader if "framewise_displacement_mm" in row]
            if vals:
                mean_fd_mm = round(sum(vals) / len(vals), 6)
                max_fd_mm = round(max(vals), 6)
        except Exception:
            mean_fd_mm = None
            max_fd_mm = None

    return {
        "pipeline_stages_total": provenance.get("pipeline_stages_total"),
        "tr_seconds": gate2.get("tr_seconds"),
        "n_volumes": gate2.get("t_vol"),
        "n_rois": gate5.get("n_rois"),
        "zero_variance_rois": gate5.get("zero_variance_rois"),
        "n_connectivity_features": gate6.get("n_features"),
        "finite_features": gate6.get("finite_features"),
        "stc_status": stc.get("status"),
        "stc_direction": stc.get("slice_encoding_direction"),
        "sdc_status": sdc.get("status"),
        "sdc_method": sdc.get("method"),
        "defacing_status": defacing.get("status"),
        "privacy_export_status": (defacing.get("privacy_export_gate") or {}).get("overall_status"),
        "mean_fd_mm": mean_fd_mm,
        "max_fd_mm": max_fd_mm,
    }


def _new_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    now = _utc_now()
    is_pilot = config.get("cohort_profile") == "pilot_n6"
    subjects = []
    for index in range(1, int(config["n_subjects"]) + 1):
        entry: Dict[str, Any] = {
            "subject_id": f"sub-synthetic{index:03d}",
            "seed": int(config["seed"]) + index - 1,
            "status": "PENDING",
            "input_sha256": {},
            "synthetic_manifest_sha256": None,
            "run_output_dir": None,
            "output_manifest_sha256": None,
            "artifact_count": 0,
            "runtime_seconds": None,
            "peak_rss_gb": None,
            "error": None,
        }
        if is_pilot:
            profile = PILOT_N6_PROFILES[index - 1]
            entry["pilot_slot"] = profile["pilot_slot"]
            entry["acquisition_profile"] = profile["label"]
        subjects.append(entry)
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "run_type": "SYNTHETIC_COHORT_ONLY",
        "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
        "real_data_authorized": False,
        "cohort_run_id": datetime.now(timezone.utc).strftime("cohort_%Y%m%dT%H%M%S_%fZ"),
        "created_utc": now,
        "updated_utc": now,
        "configuration": _normalise_config(config),
        "status": "INITIALIZED",
        "subjects": subjects,
    }
    if is_pilot:
        manifest["pilot_dry_run"] = True
        manifest["pilot_label"] = "SYNTHETIC_ONLY_N6_PILOT_DRY_RUN"
    return manifest


def _verify_run_artifacts(run_output_dir: Path) -> Dict[str, Any]:
    """Verify each output listed by the single-subject runner; reject path escapes."""
    root = run_output_dir.resolve(strict=True)
    manifest_path = root / "output_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"Missing or unsafe output manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "SUCCESS":
        raise RuntimeError("Single-subject output manifest is not marked SUCCESS.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("Output manifest contains no artifact records.")

    verified = {}
    for artifact_name, entry in artifacts.items():
        filename = entry.get("filename")
        expected_hash = entry.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"Malformed artifact entry: {artifact_name}")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe artifact path in output manifest: {filename}")
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"Missing or unsafe pipeline artifact: {candidate}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Artifact escapes run directory: {filename}") from exc
        actual_hash = _sha256(resolved)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise RuntimeError(f"Artifact SHA-256 mismatch: {artifact_name}")
        verified[artifact_name] = actual_hash

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "artifacts": verified,
    }


def _load_or_create_manifest(
    manifest_path: Path, config: Dict[str, Any], resume: bool
) -> Dict[str, Any]:
    normalised = _normalise_config(config)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"{manifest_path} already exists. Use --resume to continue this cohort."
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError("Unsupported cohort manifest version.")
        if manifest.get("execution_policy") != "STRICTLY_SYNTHETIC_ONLY":
            raise RuntimeError("Existing cohort manifest is not synthetic-only.")
        if manifest.get("real_data_authorized") is not False:
            raise RuntimeError("Cohort manifest contains an invalid real-data authorization state.")
        if manifest.get("configuration") != normalised:
            raise ValueError("Resume configuration differs from the original cohort configuration.")
        expected_ids = [f"sub-synthetic{i:03d}" for i in range(1, config["n_subjects"] + 1)]
        actual_ids = [subject.get("subject_id") for subject in manifest.get("subjects", [])]
        if actual_ids != expected_ids:
            raise ValueError("Cohort manifest subject list does not match its configuration.")
        return manifest
    if resume:
        raise FileNotFoundError(f"Cannot resume; cohort manifest not found: {manifest_path}")
    return _new_manifest(config)


def run_synthetic_cohort(
    output_dir: Path,
    *,
    n_subjects: int = 2,
    t_vol: int = 60,
    seed: int = 42,
    atlas_mode: str = "synthetic_phantom",
    canonical_atlas_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    require_model_inference: bool = False,
    max_memory_gb: float = 2.5,
    enforce_container_memory: bool = False,
    resume: bool = False,
    continue_on_error: bool = False,
    cohort_profile: str = "standard",
    mode: str = "synthetic",
    allow_real_scans: bool = False,
) -> Dict[str, Any]:
    """Execute a deterministic synthetic cohort sequentially and checkpoint every subject."""
    # Authorization is checked first at the cohort boundary before creating files or inspecting data.
    authorization = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
    authorization.check_execution_allowed(
        requested_mode=mode,
        allow_real_scans=allow_real_scans,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    config: Dict[str, Any] = {
        "n_subjects": int(n_subjects),
        "t_vol": int(t_vol),
        "seed": int(seed),
        "atlas_mode": atlas_mode,
        "canonical_atlas_path": str(canonical_atlas_path) if canonical_atlas_path else None,
        "model_dir": str(model_dir) if model_dir else None,
        "require_model_inference": bool(require_model_inference),
        "max_memory_gb": float(max_memory_gb),
        "enforce_container_memory": bool(enforce_container_memory),
    }
    if cohort_profile != "standard":
        config["cohort_profile"] = cohort_profile
    _validate_config(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = _load_or_create_manifest(manifest_path, config, resume=resume)
    manifest["status"] = "RUNNING"
    manifest["updated_utc"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)

    input_root = output_dir / "inputs"
    run_root = output_dir / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    is_pilot = cohort_profile == "pilot_n6"
    tool_availability = _detect_host_tool_availability() if is_pilot else None

    failures = []
    for index, subject in enumerate(manifest["subjects"]):
        subject_id = subject["subject_id"]
        if subject["status"] == "SUCCESS":
            try:
                stored_dir = Path(subject["run_output_dir"])
                verified = _verify_run_artifacts(stored_dir)
                if verified["manifest_sha256"] != subject["output_manifest_sha256"]:
                    raise RuntimeError("Output manifest checksum changed since the cohort was recorded.")
                continue
            except Exception:
                # A previously completed run is no longer trustworthy; re-run this synthetic subject.
                subject["status"] = "PENDING"
                subject["error"] = "Previous output failed resume integrity verification; rerunning."

        subject["status"] = "RUNNING"
        subject["error"] = None
        # A retry must not retain references to an older, potentially invalid run.
        subject["run_output_dir"] = None
        subject["output_manifest_sha256"] = None
        subject["artifact_count"] = 0
        subject["input_sha256"] = {}
        subject["synthetic_manifest_sha256"] = None
        subject.pop("stage_summary", None)
        manifest["updated_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        started = time.monotonic()

        try:
            gen_kwargs: Dict[str, Any] = {
                "subject_id": subject_id,
                "session_id": "ses-1",
                "t_vol": config["t_vol"],
                "seed": subject["seed"],
                "canonical_atlas_path": config["canonical_atlas_path"]
                if config["atlas_mode"] == "canonical"
                else None,
            }
            if is_pilot:
                profile = PILOT_N6_PROFILES[index]
                gen_kwargs.update(
                    {
                        "tr": profile["tr"],
                        "include_slice_timing": profile["include_slice_timing"],
                        "slice_encoding_direction": profile["slice_encoding_direction"],
                        "phase_encoding_direction": profile["phase_encoding_direction"],
                        "total_readout_time": profile["total_readout_time"],
                        "fieldmap_mode": profile["fieldmap_mode"],
                    }
                )

            synthetic_paths = generate_synthetic_phantom_session(
                input_root / subject_id,
                **gen_kwargs,
            )
            input_hashes: Dict[str, str] = {}
            for key, path in synthetic_paths.items():
                if key == "synthetic_manifest" or path is None:
                    continue
                p_obj = Path(path)
                if p_obj.is_dir():
                    for child in sorted(p_obj.iterdir(), key=lambda c: c.name):
                        if child.is_file() and not child.is_symlink():
                            input_hashes[f"{key}/{child.name}"] = _sha256(child)
                elif p_obj.is_file():
                    input_hashes[key] = _sha256(p_obj)
            subject["input_sha256"] = input_hashes
            subject["synthetic_manifest_sha256"] = _sha256(
                Path(synthetic_paths["synthetic_manifest"])
            )
            pipeline_kwargs: Dict[str, Any] = {
                "bold_path": synthetic_paths["bold_nii"],
                "bold_json_path": synthetic_paths["bold_json"],
                "t1w_path": synthetic_paths["t1w_nii"],
                "atlas_path": synthetic_paths["cc200_atlas"],
                "template_path": synthetic_paths["mni_template"],
                "output_dir": run_root,
                "subject_id": subject_id,
                "session_id": "ses-1",
                "max_memory_gb": config["max_memory_gb"],
                "allow_real_scans": False,
                "manifest_path": synthetic_paths["synthetic_manifest"],
                "atlas_mode": config["atlas_mode"],
                "enforce_container_memory": config["enforce_container_memory"],
                "model_dir": config["model_dir"],
                "require_model_inference": config["require_model_inference"],
            }
            if synthetic_paths.get("fmap_dir") is not None:
                pipeline_kwargs["fmap_dir"] = synthetic_paths["fmap_dir"]
            result = run_poc_v2_pipeline(**pipeline_kwargs)
            if result.get("status") != "SUCCESS":
                raise RuntimeError(f"Pipeline returned unexpected status: {result.get('status')}")
            run_output_dir = Path(result["output_dir"]).resolve(strict=True)
            verified = _verify_run_artifacts(run_output_dir)

            subject_update: Dict[str, Any] = {
                "status": "SUCCESS",
                "run_output_dir": str(run_output_dir),
                "output_manifest_sha256": verified["manifest_sha256"],
                "artifact_count": len(verified["artifacts"]),
                "runtime_seconds": round(time.monotonic() - started, 3),
                "peak_rss_gb": float(result.get("peak_rss_gb", 0.0)),
                "error": None,
            }
            stage_summary = _extract_subject_stage_summary(result, run_output_dir)
            if any(v is not None for v in stage_summary.values()):
                subject_update["stage_summary"] = stage_summary
            subject.update(subject_update)
        except Exception as exc:
            subject.update(
                {
                    "status": "FAILED",
                    "runtime_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append({"subject_id": subject_id, "error": subject["error"]})
            manifest["updated_utc"] = _utc_now()
            manifest["status"] = "PARTIAL_FAILURE"
            _atomic_write_json(manifest_path, manifest)
            if not continue_on_error:
                break

        manifest["updated_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)

    successes = sum(item["status"] == "SUCCESS" for item in manifest["subjects"])
    failed_count = sum(item["status"] == "FAILED" for item in manifest["subjects"])
    pending_count = sum(item["status"] == "PENDING" for item in manifest["subjects"])
    if failed_count or pending_count:
        manifest["status"] = "PARTIAL_FAILURE" if successes else "FAILED"
    else:
        manifest["status"] = "SUCCESS"
    manifest["updated_utc"] = _utc_now()

    rss_values = [float(item["peak_rss_gb"]) for item in manifest["subjects"] if item.get("peak_rss_gb") is not None]
    runtime_values = [float(item["runtime_seconds"]) for item in manifest["subjects"] if item.get("runtime_seconds") is not None]
    summary_payload: Dict[str, Any] = {
        "total_subjects": len(manifest["subjects"]),
        "succeeded": successes,
        "failed": failed_count,
        "pending": pending_count,
        "failures": failures,
    }
    if is_pilot or rss_values or runtime_values:
        summary_payload["max_peak_rss_gb"] = round(max(rss_values), 4) if rss_values else None
        summary_payload["total_runtime_seconds"] = round(sum(runtime_values), 3) if runtime_values else None
    if tool_availability is not None:
        summary_payload["tool_availability"] = tool_availability
        summary_payload["cohort_profile"] = cohort_profile

    manifest["summary"] = summary_payload
    _atomic_write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest_path": manifest_path,
        "summary": manifest["summary"],
        "subjects": manifest["subjects"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sequential, resumable synthetic NeuroADHD MRI cohort."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "synthetic_cohort")
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--allow-real-scans", action="store_true", default=False)
    parser.add_argument("--n-subjects", type=int, default=2, help=f"Synthetic subjects (1-{MAX_COHORT_SIZE}).")
    parser.add_argument("--t-vol", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atlas-mode", choices=("synthetic_phantom", "canonical"), default="synthetic_phantom")
    parser.add_argument("--canonical-atlas-path", type=Path, default=PROJECT_ROOT / "data" / "atlases" / "cc200_roi_atlas.nii.gz")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--require-model-inference", action="store_true")
    parser.add_argument("--max-memory-gb", type=float, default=2.5)
    parser.add_argument("--enforce-container-memory", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume the cohort recorded in output-dir.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--pilot-profile", action="store_true", help="Execute the 6-slot N=6 synthetic pilot dry-run profile.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cohort_profile = "pilot_n6" if args.pilot_profile else "standard"
    n_subjects = 6 if args.pilot_profile and args.n_subjects == 2 else args.n_subjects
    try:
        result = run_synthetic_cohort(
            args.output_dir,
            n_subjects=n_subjects,
            t_vol=args.t_vol,
            seed=args.seed,
            atlas_mode=args.atlas_mode,
            canonical_atlas_path=args.canonical_atlas_path,
            model_dir=args.model_dir,
            require_model_inference=args.require_model_inference,
            max_memory_gb=args.max_memory_gb,
            enforce_container_memory=args.enforce_container_memory,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            cohort_profile=cohort_profile,
            mode=args.mode,
            allow_real_scans=args.allow_real_scans,
        )
    except Exception as exc:
        print(f"COHORT FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = result["summary"]
    print("=" * 72)
    print("NEUROADHD SYNTHETIC COHORT EXECUTION")
    print(f"Status     : {result['status']}")
    print(f"Subjects   : {summary['succeeded']}/{summary['total_subjects']} succeeded")
    print(f"Failed     : {summary['failed']}")
    print(f"Pending    : {summary['pending']}")
    print(f"Manifest   : {result['manifest_path']}")
    print("=" * 72)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
