#!/usr/bin/env python3
"""Run a reproducible, synthetic-only cohort through the existing NeuroADHD pipeline.

This orchestrator deliberately reuses scripts.run_poc_v2 rather than duplicating
MRI preprocessing. It processes one synthetic subject at a time to keep memory
usage bounded and writes an auditable cohort manifest after every state change.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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


def _new_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    now = _utc_now()
    subjects = []
    for index in range(1, int(config["n_subjects"]) + 1):
        subjects.append(
            {
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
        )
    return {
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
) -> Dict[str, Any]:
    """Execute a deterministic synthetic cohort sequentially and checkpoint every subject."""
    output_dir = Path(output_dir).expanduser().resolve()
    config = {
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
    _validate_config(config)

    # Authorization is checked at the cohort boundary and again by every single-scan run.
    authorization = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
    authorization.check_execution_allowed(requested_mode="synthetic", allow_real_scans=False)

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

    failures = []
    for subject in manifest["subjects"]:
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
        manifest["updated_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        started = time.monotonic()

        try:
            synthetic_paths = generate_synthetic_phantom_session(
                input_root / subject_id,
                subject_id=subject_id,
                session_id="ses-1",
                t_vol=config["t_vol"],
                seed=subject["seed"],
                canonical_atlas_path=config["canonical_atlas_path"]
                if config["atlas_mode"] == "canonical"
                else None,
            )
            subject["input_sha256"] = {
                key: _sha256(Path(path))
                for key, path in synthetic_paths.items()
                if key != "synthetic_manifest"
            }
            subject["synthetic_manifest_sha256"] = _sha256(
                Path(synthetic_paths["synthetic_manifest"])
            )
            result = run_poc_v2_pipeline(
                bold_path=synthetic_paths["bold_nii"],
                bold_json_path=synthetic_paths["bold_json"],
                t1w_path=synthetic_paths["t1w_nii"],
                atlas_path=synthetic_paths["cc200_atlas"],
                template_path=synthetic_paths["mni_template"],
                output_dir=run_root,
                subject_id=subject_id,
                session_id="ses-1",
                max_memory_gb=config["max_memory_gb"],
                allow_real_scans=False,
                manifest_path=synthetic_paths["synthetic_manifest"],
                atlas_mode=config["atlas_mode"],
                enforce_container_memory=config["enforce_container_memory"],
                model_dir=config["model_dir"],
                require_model_inference=config["require_model_inference"],
            )
            if result.get("status") != "SUCCESS":
                raise RuntimeError(f"Pipeline returned unexpected status: {result.get('status')}")
            run_output_dir = Path(result["output_dir"]).resolve(strict=True)
            verified = _verify_run_artifacts(run_output_dir)

            subject.update(
                {
                    "status": "SUCCESS",
                    "run_output_dir": str(run_output_dir),
                    "output_manifest_sha256": verified["manifest_sha256"],
                    "artifact_count": len(verified["artifacts"]),
                    "runtime_seconds": round(time.monotonic() - started, 3),
                    "peak_rss_gb": float(result.get("peak_rss_gb", 0.0)),
                    "error": None,
                }
            )
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
    manifest["summary"] = {
        "total_subjects": len(manifest["subjects"]),
        "succeeded": successes,
        "failed": failed_count,
        "pending": pending_count,
        "failures": failures,
    }
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_synthetic_cohort(
            args.output_dir,
            n_subjects=args.n_subjects,
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
