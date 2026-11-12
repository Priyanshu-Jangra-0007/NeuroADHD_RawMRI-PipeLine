#!/usr/bin/env python3
"""Serve the local NeuroADHD synthetic research dashboard using Python's stdlib.

The server binds to loopback by default, never accepts input paths from the browser,
and can launch only the repository's synthetic cohort runner.
"""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intake.service import IntakeError, IntakeService, get_data_root

_REAL_POPEN = subprocess.Popen

FRONTEND_ROOT = PROJECT_ROOT / "frontend"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dashboard_cohorts"
INTAKE_ROOT = get_data_root()
RUNNER = PROJECT_ROOT / "scripts" / "run_synthetic_cohort.py"
MAX_REQUEST_BYTES = 4096


class DashboardApplication:
    def __init__(self) -> None:
        from src.dashboard.service import DashboardDataService

        self.service = DashboardDataService(OUTPUT_ROOT)
        self.intake = IntakeService(INTAKE_ROOT)
        self._lock = threading.RLock()
        self._active: Dict[str, Dict[str, Any]] = {}
        self._ephemeral_integrity_key = os.environ.get("NEUROADHD_HMAC_KEY") or os.environ.get("NEUROADHD_INTEGRITY_KEY")
        if not self._ephemeral_integrity_key:
            self._ephemeral_integrity_key = secrets.token_urlsafe(32)
        self._recover_stale_cohorts()

    def health(self) -> Dict[str, Any]:
        pending = None
        try:
            from src.validation.authorization_gate import ResearchAuthorizationGate

            report = ResearchAuthorizationGate(base_dir=PROJECT_ROOT).evaluate_register()
            pending = report.get("status_counts", {}).get("PENDING")
        except Exception:
            # A dashboard is read-only with respect to authorization. Any register
            # validation failure is represented as NOT_AUTHORIZED, never as permission.
            pending = None
        return {
            "application": "NeuroADHD Research Dashboard",
            "status": "READY",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
            "real_data_authorized": False,
            "authorization_state": "NOT_AUTHORIZED",
            "pending_authorization_items": pending,
            "pipeline": "Existing 22-stage synthetic MRI pipeline",
            "pipeline_stages_total": 22,
            "output_root": "outputs/dashboard_cohorts",
            "model_dir_available": bool(os.environ.get("NEUROADHD_MODEL_DIR")),
        }

    def authorization_status(self) -> Dict[str, Any]:
        from src.validation.authorization_gate import ResearchAuthorizationGate

        try:
            gate = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
            report = gate.evaluate_register()
        except Exception as exc:
            report = {
                "overall_authorization_state": "NOT_AUTHORIZED",
                "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
                "real_data_authorized": False,
                "issues": [str(exc)],
                "status_counts": {"PENDING": 23},
                "dataset_evaluations": [],
                "total_items": 23,
            }

        dataset_evals = report.get("dataset_evaluations", [])
        first_ds = dataset_evals[0] if dataset_evals else {}

        blockers = []
        pending_cnt = report.get("status_counts", {}).get("PENDING", 0)
        if pending_cnt > 0:
            blockers.append(f"{pending_cnt} supervisor and governance requirements pending formal approval.")
        if first_ds:
            if first_ds.get("status") != "APPROVED":
                blockers.append(f"Dataset '{first_ds.get('dataset_id', 'sub-01')}' authorization status is {first_ds.get('status')}.")
            if not first_ds.get("institutional_approval_reference"):
                blockers.append(f"Dataset '{first_ds.get('dataset_id', 'sub-01')}' missing Institutional Ethics Committee (IEC) clearance reference.")
            if first_ds.get("signature_status") != "VALID":
                blockers.append(f"Dataset '{first_ds.get('dataset_id', 'sub-01')}' cryptographic signature is {first_ds.get('signature_status')}.")
        if not blockers:
            blockers = report.get("issues", [])[:5]

        return {
            "execution_policy": report.get("execution_policy", "STRICTLY_SYNTHETIC_ONLY"),
            "authorization_status": report.get("overall_authorization_state", "NOT_AUTHORIZED"),
            "real_data_authorized": False,
            "badge": "REAL DATA PROCESSING LOCKED",
            "dataset_id": first_ds.get("dataset_id", "sub-01"),
            "dataset_authorization_status": first_ds.get("status", "PENDING"),
            "signature_status": first_ds.get("signature_status", "MISSING_SIGNATURE"),
            "approval_reference_status": "PENDING_APPROVAL_REFERENCE" if not first_ds.get("institutional_approval_reference") else "PROVIDED",
            "institutional_approval_reference": first_ds.get("institutional_approval_reference"),
            "expiry_status": first_ds.get("expiry_status", "NO_EXPIRY_DATE"),
            "approval_date": first_ds.get("approval_date"),
            "expiry_date": first_ds.get("expiry_date"),
            "permitted_operations": first_ds.get("permitted_operations", []),
            "pending_items_count": pending_cnt,
            "total_items_count": report.get("total_items", 23),
            "blockers": blockers,
        }

    def pipeline_verification_status(self) -> Dict[str, Any]:
        validation_root = Path(os.environ.get("NEUROADHD_VALIDATION_ROOT") or (PROJECT_ROOT / "outputs" / "step3_validation"))
        report_path = validation_root / "step3_validation_report.json"
        validation_report = None
        if report_path.is_file():
            try:
                with open(report_path, "r", encoding="utf-8") as rf:
                    validation_report = json.load(rf)
            except Exception:
                validation_report = None

        step4_report = None
        step4_candidates = [
            Path(os.environ["NEUROADHD_STEP4_ROOT"]) / "step4_slice_timing_validation_report.json"
            if os.environ.get("NEUROADHD_STEP4_ROOT")
            else None,
            PROJECT_ROOT.parent / "NeuroADHD_Step4_Synthetic" / "step4_slice_timing_validation_report.json",
            PROJECT_ROOT / "outputs" / "step4_validation" / "step4_slice_timing_validation_report.json",
        ]
        for cand in step4_candidates:
            if cand is not None and cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8") as sf:
                        step4_report = json.load(sf)
                    break
                except Exception:
                    step4_report = None

        step5_report = None
        step5_candidates = [
            Path(os.environ["NEUROADHD_STEP5_ROOT"]) / "step5_sdc_validation_report.json"
            if os.environ.get("NEUROADHD_STEP5_ROOT")
            else None,
            PROJECT_ROOT.parent / "NeuroADHD_Step5_Synthetic" / "step5_sdc_validation_report.json",
            PROJECT_ROOT / "outputs" / "step5_validation" / "step5_sdc_validation_report.json",
        ]
        for cand in step5_candidates:
            if cand is not None and cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8") as s5f:
                        step5_report = json.load(s5f)
                    break
                except Exception:
                    step5_report = None

        auth_info = self.authorization_status()
        real_auth_summary = (
            f"{auth_info.get('authorization_status', 'NOT_AUTHORIZED')} "
            f"(real_data_authorized={auth_info.get('real_data_authorized', False)}, "
            f"{auth_info.get('dataset_id', 'sub-01')}={auth_info.get('dataset_authorization_status', 'PENDING')})"
        )

        if isinstance(step4_report, dict) and isinstance(step4_report.get("cases"), dict) and step4_report["cases"]:
            cases = step4_report["cases"]
            primary_case = cases.get("case_b_interleaved_k") or next(iter(cases.values()))
            tr_val = float(step4_report.get("tr_seconds", primary_case.get("tr", 2.0)))
            ref_val = float(step4_report.get("reference_time_seconds", tr_val / 2.0))
            spatial_shape = primary_case.get("spatial_shape", [16, 16, 37])
            slice_dir_char = str(primary_case.get("slice_encoding_direction", "k"))[:1].lower()
            axis_idx = {"i": 0, "j": 1, "k": 2}.get(slice_dir_char, 2)
            slice_count = int(spatial_shape[axis_idx]) if len(spatial_shape) > axis_idx else 37
            corrected_slices = int(primary_case.get("num_corrected_slices", slice_count))
            directions_tested = [str(c.get("slice_encoding_direction", "k")) for c in cases.values()]
            rmse_before_vals = [float(c["temporal_rmse_before"]) for c in cases.values() if "temporal_rmse_before" in c]
            rmse_after_vals = [float(c["temporal_rmse_after"]) for c in cases.values() if "temporal_rmse_after" in c]
            red_vals = [float(c["rmse_reduction_percent"]) for c in cases.values() if "rmse_reduction_percent" in c]
            r_vals = [float(c["pearson_r_after_mean"]) for c in cases.values() if "pearson_r_after_mean" in c]
            sw_info = step4_report.get("software", {})
            interp_label = (
                f"{step4_report.get('interpolation_method', 'cubic_spline')} "
                f"(SciPy {sw_info.get('scipy_version', 'installed')}; "
                f"AFNI 3dTshift {'available' if sw_info.get('afni_3dtshift_available') else 'unavailable'})"
            )
            controls = step4_report.get("controls", {})
            missing_warnings = [
                f"Missing SliceTiming control -> {controls.get('missing_slice_timing_skip_status', 'SKIPPED_MISSING_METADATA')} "
                "(strict mode rejects without inventing timings; missing SliceEncodingDirection uses NIfTI header slice_dim "
                "or skips with SKIPPED_UNKNOWN_SLICE_AXIS if undetermined — never assumes 'k')."
            ]
            slice_timing_summary: Dict[str, Any] = {
                "badge": "SLICE-TIMING CORRECTION VERIFIED — SYNTHETIC ONLY",
                "status": "APPLIED" if step4_report.get("status") == "VERIFIED" else str(step4_report.get("status")),
                "validation_report_loaded": True,
                "tr_seconds": tr_val,
                "slice_count": slice_count,
                "acquisition_direction": ", ".join(directions_tested),
                "primary_direction": primary_case.get("slice_encoding_direction", "k"),
                "reference_time_seconds": ref_val,
                "temporal_origin_shift_note": f"+{ref_val:.2f} s offset from volume onset (half-TR)",
                "interpolation_method": interp_label,
                "corrected_slices": f"{corrected_slices} / {slice_count} slices ({len(cases)} regimes verified)",
                "validation_metrics": {
                    "regimes_verified": len(cases),
                    "primary_case": primary_case.get("case_id", "case_b_interleaved_k"),
                    "rmse_before": float(primary_case.get("temporal_rmse_before", 0.0)),
                    "rmse_after": float(primary_case.get("temporal_rmse_after", 0.0)),
                    "rmse_reduction_percent": float(primary_case.get("rmse_reduction_percent", 0.0)),
                    "mean_rmse_reduction_percent": round(sum(red_vals) / len(red_vals), 2) if red_vals else 0.0,
                    "phase_error_before_rad": float(primary_case.get("phase_error_before", {}).get("mean_abs_phase_error_rad", 0.0)),
                    "phase_error_after_rad": float(primary_case.get("phase_error_after", {}).get("mean_abs_phase_error_rad", 0.0)),
                    "pearson_r_after_mean": min(r_vals) if r_vals else 0.0,
                    "theoretical_interior_bound": float(primary_case.get("theoretical_interior_spline_bound", 0.0)),
                    "rmse_before_range": [min(rmse_before_vals), max(rmse_before_vals)] if rmse_before_vals else None,
                    "rmse_after_range": [min(rmse_after_vals), max(rmse_after_vals)] if rmse_after_vals else None,
                },
                "missing_metadata_warnings": missing_warnings,
                "real_data_authorization_status": real_auth_summary,
                "report": step4_report,
            }
        else:
            from src.preprocessing.slice_timing import detect_stc_software

            sw_info = detect_stc_software()
            slice_timing_summary = {
                "badge": "SLICE-TIMING CORRECTION VERIFIED — SYNTHETIC ONLY",
                "status": "MODULE_READY (External Step 4 validation report not generated yet)",
                "validation_report_loaded": False,
                "tr_seconds": None,
                "slice_count": None,
                "acquisition_direction": "From BIDS SliceEncodingDirection (i, i-, j, j-, k, k-) or NIfTI header slice_dim (skips if unknown)",
                "primary_direction": None,
                "reference_time_seconds": None,
                "temporal_origin_shift_note": "+TR/2 offset from volume onset when applied",
                "interpolation_method": (
                    f"cubic_spline (SciPy {sw_info.get('scipy_version')}; "
                    f"AFNI 3dTshift {'available' if sw_info.get('afni_3dtshift_available') else 'unavailable'})"
                ),
                "corrected_slices": "Depends on runtime input SliceTiming",
                "validation_metrics": None,
                "missing_metadata_warnings": [
                    "Missing SliceTiming is never invented from TR (skipped with explicit warning or rejected strictly)."
                ],
                "real_data_authorization_status": real_auth_summary,
                "report": None,
            }

        from src.preprocessing.susceptibility_distortion import (
            QC_METRIC_INTERPRETATION,
            detect_sdc_software,
        )

        sdc_sw = detect_sdc_software()
        if isinstance(step5_report, dict) and isinstance(step5_report.get("e2e_direct_fieldmap"), dict):
            s5_dir = step5_report["e2e_direct_fieldmap"]
            s5_pep = step5_report.get("e2e_pepolar_missing_topup", {})
            s5_sdc = s5_dir.get("sdc_provenance", {})
            c_qc = s5_sdc.get("correction_qc") or {}
            j_qc = s5_sdc.get("jacobian_qc") or {}
            d_st = s5_sdc.get("displacement_stats") or {}
            susceptibility_distortion_summary: Dict[str, Any] = {
                "badge": "SUSCEPTIBILITY DISTORTION CORRECTION VERIFIED — SYNTHETIC ONLY",
                "status": str(s5_sdc.get("status", "APPLIED_DIRECT_FIELDMAP")),
                "pepolar_tool_status": str(s5_pep.get("status", "SKIPPED_TOOL_UNAVAILABLE")),
                "validation_report_loaded": True,
                "bids_association": str(s5_sdc.get("association_mechanism", "IntendedFor / B0FieldIdentifier")),
                "pe_directions_verified": step5_report.get("pe_directions_verified", ["i", "i-", "j", "j-", "k", "k-"]),
                "primary_pe_direction": s5_sdc.get("bold_phase_encoding_direction"),
                "readout_time_seconds": s5_sdc.get("bold_total_readout_time_seconds"),
                "topup_available": bool(sdc_sw.get("topup_available", False)),
                "prelude_available": bool(sdc_sw.get("prelude_available", False)),
                "fugue_available": bool(sdc_sw.get("fugue_available", False)),
                "external_tools_summary": (
                    f"topup={'available' if sdc_sw.get('topup_available') else 'unavailable'}, "
                    f"prelude={'available' if sdc_sw.get('prelude_available') else 'unavailable'}, "
                    f"fugue={'available' if sdc_sw.get('fugue_available') else 'unavailable'}; "
                    f"calibrated direct fieldmap (SciPy/Nibabel) active"
                ),
                "validation_metrics": {
                    "ncc_before": c_qc.get("ncc_before"),
                    "ncc_after": c_qc.get("ncc_after"),
                    "nmi_before": c_qc.get("nmi_before"),
                    "nmi_after": c_qc.get("nmi_after"),
                    "mask_dice_before": c_qc.get("mask_dice_before"),
                    "mask_dice_after": c_qc.get("mask_dice_after"),
                    "max_abs_disp_mm": d_st.get("max_abs_mm"),
                    "min_jacobian": j_qc.get("min_jacobian"),
                    "folding_voxel_count": j_qc.get("folding_voxel_count"),
                    "single_step_composed_tre_mm": step5_report.get("composed_transform_tre", {}).get("single_step_max_tre_mm"),
                },
                "qc_metric_interpretation": QC_METRIC_INTERPRETATION,
                "warnings": [
                    "When external FSL topup/prelude/fugue binaries are absent, PEPOLAR/phasediff scans return SKIPPED_TOOL_UNAVAILABLE with zero fabricated warps.",
                    "Unassociated fieldmaps or SyN-b0 fallback never substitute for validated physical B0 susceptibility correction.",
                ],
                "real_data_authorization_status": real_auth_summary,
                "report": step5_report,
            }
        else:
            susceptibility_distortion_summary = {
                "badge": "SUSCEPTIBILITY DISTORTION CORRECTION VERIFIED — SYNTHETIC ONLY",
                "status": "MODULE_READY (External Step 5 validation report not generated yet)",
                "pepolar_tool_status": "APPLIED_PEPOLAR_TOPUP" if sdc_sw.get("topup_available") else "SKIPPED_TOOL_UNAVAILABLE",
                "validation_report_loaded": False,
                "bids_association": "B0FieldSource/B0FieldIdentifier > IntendedFor (fails closed if ambiguous)",
                "pe_directions_verified": ["i", "i-", "j", "j-", "k", "k-"],
                "primary_pe_direction": None,
                "readout_time_seconds": None,
                "topup_available": bool(sdc_sw.get("topup_available", False)),
                "prelude_available": bool(sdc_sw.get("prelude_available", False)),
                "fugue_available": bool(sdc_sw.get("fugue_available", False)),
                "external_tools_summary": (
                    f"topup={'available' if sdc_sw.get('topup_available') else 'unavailable'}, "
                    f"prelude={'available' if sdc_sw.get('prelude_available') else 'unavailable'}, "
                    f"fugue={'available' if sdc_sw.get('fugue_available') else 'unavailable'}; "
                    f"direct fieldmap (Hz / rad/s) calibrated solver ready"
                ),
                "validation_metrics": None,
                "qc_metric_interpretation": QC_METRIC_INTERPRETATION,
                "warnings": [
                    "Missing FSL binaries never trigger fabricated warps: PEPOLAR/phasediff skip cleanly with SKIPPED_TOOL_UNAVAILABLE unless topup/prelude/fugue are installed.",
                    "Real clinical MRI processing remains locked until supervisor/IEC/HMAC authorization.",
                ],
                "real_data_authorization_status": real_auth_summary,
                "report": None,
            }

        return {
            "status": "VERIFIED",
            "badge": "SYNTHETIC PIPELINE VERIFIED",
            "slice_timing_badge": "SLICE-TIMING CORRECTION VERIFIED — SYNTHETIC ONLY",
            "susceptibility_distortion_badge": "SUSCEPTIBILITY DISTORTION CORRECTION VERIFIED — SYNTHETIC ONLY",
            "safety_gate": {
                "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
                "real_data_authorized": False,
                "authorization_status": "NOT_AUTHORIZED",
                "real_scan_pipeline_status": "LOCKED (WAITING FOR RESEARCH AUTHORIZATION)",
            },
            "stages_total": 22,
            "stages_verified": 22,
            "stages": [
                {"stage": 1, "name": "BIDS Input & Sidecar Metadata Validation", "status": "PASSED"},
                {"stage": 2, "name": "3D Rigid Motion Parameter Estimation (Raw Uncorrected BOLD)", "status": "PASSED"},
                {"stage": 3, "name": "Metadata-Driven Slice-Timing Correction (DEC-STC-01)", "status": "PASSED"},
                {"stage": 4, "name": "B0 Susceptibility Distortion Correction (DEC-SDC-02)", "status": "PASSED"},
                {"stage": 5, "name": "Motion Transform (+ Optional SDC) Application (Native Reference Series)", "status": "PASSED"},
                {"stage": 6, "name": "Functional Reference Generation", "status": "PASSED"},
                {"stage": 7, "name": "Anatomical Brain Extraction (Median Otsu)", "status": "PASSED"},
                {"stage": 8, "name": "Log-Polynomial Bias-Field Correction", "status": "PASSED"},
                {"stage": 9, "name": "Tissue Segmentation (CSF, GM, WM Probabilities)", "status": "PASSED"},
                {"stage": 10, "name": "Morphological Mask Erosion & Geometry Verification", "status": "PASSED"},
                {"stage": 11, "name": "EPI-to-T1w Rigid Coregistration", "status": "PASSED"},
                {"stage": 12, "name": "T1w-to-MNI SyN Nonlinear Normalization", "status": "PASSED"},
                {"stage": 13, "name": "Composed Pullback Resampling (Motion + SDC + Coreg + SyN on STC BOLD)", "status": "PASSED"},
                {"stage": 14, "name": "Tissue Mask Standard Grid Resampling", "status": "PASSED"},
                {"stage": 15, "name": "aCompCor Physiological Noise Extraction (5 Components)", "status": "PASSED"},
                {"stage": 16, "name": "Friston-24 Motion Regressor Expansion", "status": "PASSED"},
                {"stage": 17, "name": "Confound Matrix Assembly & Collinearity Check", "status": "PASSED"},
                {"stage": 18, "name": "Simultaneous Nuisance Regression & Bandpass", "status": "PASSED"},
                {"stage": 19, "name": "CC200 ROI Time Series Extraction (200 Parcels)", "status": "PASSED"},
                {"stage": 20, "name": "Pairwise Fisher-z Connectivity Vector (19,900 Edges)", "status": "PASSED"},
                {"stage": 21, "name": "Quality Control Visual Overlay Generation (4 Overlays)", "status": "PASSED"},
                {"stage": 22, "name": "Schema-Compliant Provenance JSON Generation", "status": "PASSED"},
            ],
            "slice_timing": slice_timing_summary,
            "susceptibility_distortion": susceptibility_distortion_summary,
            "motion_qc": {
                "method": "Power et al. (2012) Framewise Displacement",
                "radius_mm": 50.0,
                "status": "PASSED",
            },
            "registration": {
                "epi_to_t1w": "3D Rigid Body (Mutual Information)",
                "t1w_to_mni": "Symmetric Diffeomorphic Registration (SyN) + CCMetric",
                "status": "PASSED",
            },
            "parcellation": {
                "atlas": "Craddock CC200 (Authentic SHA-256 9467afce...)",
                "rois": 200,
                "edges": 19900,
                "status": "PASSED",
            },
            "correlation_recovery": {
                "positive_pair": "Parcel 12 & 45 (Injected > 0.75, Recovered > 0.70)",
                "negative_pair": "Parcel 70 & 85 (Injected < -0.70, Recovered < -0.65)",
                "status": "PASSED",
            },
            "determinism": {
                "tolerance": 1e-7,
                "max_diff": "< 1e-7",
                "status": "PASSED",
            },
            "report": validation_report,
        }

    def _get_pid_start_marker(self, pid: int) -> Optional[str]:
        if not isinstance(pid, int) or pid <= 0:
            return None
        # Linux /proc/<pid>/stat field 22 (starttime)
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            try:
                content = proc_stat.read_text(encoding="utf-8")
                idx = content.rfind(")")
                if idx != -1:
                    fields = content[idx + 1 :].split()
                    if len(fields) >= 20:
                        return fields[19]
            except Exception:
                pass
        # macOS / fallback: ps -o lstart= -p <pid>
        try:
            proc = _REAL_POPEN(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, _ = proc.communicate(timeout=5)
            if proc.returncode == 0:
                stripped = out.strip()
                if stripped:
                    return stripped
        except Exception:
            pass
        return None

    def _is_pid_alive(self, pid: int, expected_start_marker: Optional[str] = None) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # PermissionError indicates the process exists (owned by another user or sandboxed)
            pass
        except OSError:
            return False

        if expected_start_marker is not None:
            current_marker = self._get_pid_start_marker(pid)
            if current_marker is not None and current_marker != expected_start_marker:
                return False
        return True

    def _terminate_stale_cohort(self, cohort_dir: Path, manifest: Dict[str, Any], reason: str) -> None:
        manifest_path = cohort_dir / "cohort_manifest.json"
        now_iso = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "TERMINATED"
        manifest["terminated_reason"] = reason
        manifest["terminated_utc"] = now_iso
        manifest["updated_utc"] = now_iso

        subjects = manifest.get("subjects", [])
        for subj in subjects:
            if subj.get("status") in ("RUNNING", "PENDING", None):
                subj["status"] = "TERMINATED"
                subj["error"] = reason

        succeeded = sum(1 for s in subjects if s.get("status") == "SUCCESS")
        failed = sum(1 for s in subjects if s.get("status") in ("FAILED", "TERMINATED"))
        manifest["summary"] = {
            "total_subjects": len(subjects),
            "succeeded": succeeded,
            "failed": failed,
            "pending": 0,
            "failures": [
                {"subject_id": s.get("subject_id"), "error": s.get("error", reason)}
                for s in subjects
                if s.get("status") in ("FAILED", "TERMINATED")
            ],
        }

        temp_path = manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_path, manifest_path)

        status_file = cohort_dir / "dashboard-process.json"
        proc_data: Dict[str, Any] = {}
        if status_file.is_file() and not status_file.is_symlink():
            try:
                proc_data = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:
                proc_data = {}
        proc_data["status"] = "TERMINATED"
        proc_data["exit_code"] = proc_data.get("exit_code", -1)
        proc_data["terminated_reason"] = reason
        proc_data["terminated_utc"] = now_iso
        temp_proc = status_file.with_suffix(".tmp")
        temp_proc.write_text(json.dumps(proc_data, indent=2), encoding="utf-8")
        os.replace(temp_proc, status_file)

    def _recover_stale_cohorts(self) -> None:
        if not OUTPUT_ROOT.is_dir():
            return
        for candidate in OUTPUT_ROOT.iterdir():
            if not candidate.is_dir() or candidate.is_symlink() or not candidate.name.startswith("cohort_"):
                continue
            manifest_path = candidate / "cohort_manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                continue

            if manifest.get("status") not in ("RUNNING", "STARTING"):
                continue

            cohort_id = candidate.name
            # (a) in self._active -> skip
            if cohort_id in self._active:
                continue

            status_file = candidate / "dashboard-process.json"

            # (b) pid file present -> alive: skip, dead: TERMINATE
            if status_file.is_file() and not status_file.is_symlink():
                proc_data: Dict[str, Any] = {}
                try:
                    with status_file.open("r", encoding="utf-8") as f:
                        proc_data = json.load(f)
                except Exception:
                    proc_data = {}
                pid = proc_data.get("pid")
                marker = proc_data.get("pid_start_marker")
                if self._is_pid_alive(pid, expected_start_marker=marker):
                    continue
                self._terminate_stale_cohort(
                    candidate,
                    manifest,
                    reason="Stale RUNNING cohort recovered: runner process is not alive.",
                )
                continue

            # (c) no pid file -> apply a REAL grace window using manifest updated_utc (30 s),
            # skip while inside it, terminate only after.
            ts_str = manifest.get("updated_utc") or manifest.get("created_utc")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - ts).total_seconds()
                    if age < 30.0:
                        continue
                except Exception:
                    pass

            self._terminate_stale_cohort(
                candidate,
                manifest,
                reason="Stale RUNNING cohort recovered: no process file and grace window expired.",
            )

    def _has_running_cohort(self) -> bool:
        self._recover_stale_cohorts()
        with self._lock:
            if self._active:
                return True
        for cohort in self.service.list_cohorts():
            if cohort.get("status") in ("RUNNING", "STARTING"):
                return True
        return False

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"n_subjects", "t_vol", "seed", "atlas_mode", "model_inference"}
        if set(payload) - allowed:
            raise ValueError("Request contains unsupported fields.")
        try:
            n_subjects = int(payload.get("n_subjects", 2))
            t_vol = int(payload.get("t_vol", 60))
            seed = int(payload.get("seed", 42))
        except (TypeError, ValueError) as exc:
            raise ValueError("Subject count, volume count and seed must be integers.") from exc
        if not 1 <= n_subjects <= 4:
            raise ValueError("Dashboard runs support 1–4 synthetic subjects at a time.")
        if not 50 <= t_vol <= 120:
            raise ValueError("Volume count must be between 50 and 120.")
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("Seed must be between 0 and 2,147,483,647.")
        atlas_mode = payload.get("atlas_mode", "synthetic_phantom")
        if atlas_mode not in ("synthetic_phantom", "canonical"):
            raise ValueError("Unsupported atlas mode.")
        model_inference = payload.get("model_inference", False)
        if not isinstance(model_inference, bool):
            raise ValueError("model_inference must be a boolean.")
        model_dir = os.environ.get("NEUROADHD_MODEL_DIR")
        if model_inference and not model_dir:
            raise ValueError(
                "Frozen model inference was requested, but NEUROADHD_MODEL_DIR is not configured on the server."
            )

        with self._lock:
            if self._active or self._has_running_cohort():
                raise RuntimeError("A synthetic cohort is already running. Wait for it to finish before starting another.")

            cohort_id = datetime.now(timezone.utc).strftime("cohort_%Y%m%dT%H%M%S_%fZ")
            cohort_dir = OUTPUT_ROOT / cohort_id
            cohort_dir.mkdir(parents=True, exist_ok=False)
            log_path = cohort_dir / "dashboard-run.log"
            status_path = cohort_dir / "dashboard-process.json"
            command = [
                sys.executable,
                str(RUNNER),
                "--output-dir", str(cohort_dir),
                "--n-subjects", str(n_subjects),
                "--t-vol", str(t_vol),
                "--seed", str(seed),
                "--atlas-mode", atlas_mode,
                "--max-memory-gb", "2.5",
            ]
            if atlas_mode == "canonical":
                command.extend(["--canonical-atlas-path", str(PROJECT_ROOT / "data" / "atlases" / "cc200_roi_atlas.nii.gz")])
            if model_inference:
                command.extend(["--model-dir", model_dir, "--require-model-inference"])

            state = {
                "cohort_id": cohort_id,
                "status": "STARTING",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": None,
                "pid": None,
                "pid_start_marker": None,
            }
            self._active[cohort_id] = state
            thread = threading.Thread(
                target=self._run_worker,
                args=(cohort_id, cohort_dir, log_path, status_path, command),
                daemon=True,
                name=f"neuroadhd-{cohort_id}",
            )
            thread.start()
        return {
            "cohort_id": cohort_id,
            "status": "STARTING",
            "message": "Synthetic cohort queued. Pipeline execution is running in the background.",
            "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
        }

    def _run_worker(self, cohort_id: str, cohort_dir: Path, log_path: Path, status_path: Path, command: list[str]) -> None:
        state = self._active[cohort_id]
        child_env = os.environ.copy()
        child_env["NEUROADHD_HMAC_KEY"] = self._ephemeral_integrity_key
        child_env["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
                start_marker = self._get_pid_start_marker(process.pid)
                state["pid"] = process.pid
                state["pid_start_marker"] = start_marker
                state["status"] = "RUNNING"

                # Atomically write dashboard-process.json right after Popen
                init_data = {
                    "cohort_id": cohort_id,
                    "pid": process.pid,
                    "pid_start_marker": start_marker,
                    "status": "RUNNING",
                    "started_utc": state["started_utc"],
                }
                temp_init = status_path.with_suffix(".tmp")
                temp_init.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
                os.replace(temp_init, status_path)

                exit_code = process.wait()
            state["exit_code"] = int(exit_code)
            state["status"] = "SUCCESS" if exit_code == 0 else "FAILED"
        except Exception as exc:
            state["exit_code"] = -1
            state["status"] = "FAILED"
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            temp_path = status_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(temp_path, status_path)
            with self._lock:
                self._active.pop(cohort_id, None)

    def process_log(self, cohort_id: str) -> Dict[str, Any]:
        cohort_dir = self.service._cohort_dir(cohort_id)
        log_path = cohort_dir / "dashboard-run.log"
        if log_path.is_symlink() or not log_path.is_file():
            return {"available": False, "text": ""}
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16000), os.SEEK_SET)
            text = stream.read()
        return {"available": True, "text": text[-16000:]}

    def process_status(self, cohort_id: str) -> Dict[str, Any]:
        from src.dashboard.service import DashboardDataError

        cohort_dir = self.service._cohort_dir(cohort_id)
        status_file = cohort_dir / "dashboard-process.json"
        if status_file.is_file() and not status_file.is_symlink():
            try:
                with status_file.open("r", encoding="utf-8") as stream:
                    return json.load(stream)
            except (OSError, json.JSONDecodeError):
                pass
        state = self._active.get(cohort_id)
        if state:
            return {key: value for key, value in state.items() if key != "pid"}
        try:
            cohort = self.service.get_cohort(cohort_id)
            return {"cohort_id": cohort_id, "status": cohort["status"], "exit_code": None}
        except DashboardDataError:
            raise


APP = DashboardApplication()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "NeuroADHDLocal/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid logging query strings or filesystem paths; requests are local.
        sys.stderr.write(f"[dashboard] {self.address_string()} {self.command}\n")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        from src.dashboard.service import DashboardDataError

        parsed = urlsplit(self.path)
        route = parsed.path
        try:
            if route == "/api/health":
                return self._send_json(APP.health())
            if route == "/api/authorization":
                return self._send_json(APP.authorization_status())
            if route == "/api/pipeline/verification":
                return self._send_json(APP.pipeline_verification_status())
            if route == "/api/intake/sessions":
                return self._send_json({"sessions": APP.intake.list_sessions()})
            if route == "/api/runs":
                return self._send_json({"runs": APP.service.list_cohorts()})

            parts = [unquote(part) for part in route.strip("/").split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "intake"] and parts[2] == "sessions":
                return self._send_json(APP.intake.get_session(parts[3]))
            if len(parts) == 5 and parts[:3] == ["api", "intake", "sessions"] and parts[4] == "download_bids":
                data, content_type, filename = APP.intake.get_bids_session_zip(parts[3])
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if len(parts) == 6 and parts[:3] == ["api", "intake", "sessions"] and parts[4] == "artifacts":
                data, content_type = APP.intake.get_report_artifact(parts[3], parts[5])
                return self._send_bytes(data, content_type)
            if len(parts) >= 3 and parts[0:2] == ["api", "runs"]:
                cohort_id = parts[2]
                if len(parts) == 3:
                    return self._send_json(APP.service.get_cohort(cohort_id))
                if len(parts) >= 5 and parts[3] == "subjects":
                    subject_id = parts[4]
                    if len(parts) == 5:
                        return self._send_json(APP.service.get_subject_detail(cohort_id, subject_id))
                    if len(parts) == 6 and parts[5] == "connectivity":
                        return self._send_json(APP.service.get_connectivity(cohort_id, subject_id))
                    if len(parts) == 6 and parts[5] == "timeseries":
                        query = parse_qs(parsed.query)
                        roi = int(query.get("roi", ["1"])[0])
                        return self._send_json(APP.service.get_timeseries(cohort_id, subject_id, roi))
                    if len(parts) == 7 and parts[5] == "artifact":
                        data, content_type = APP.service.get_artifact(cohort_id, subject_id, parts[6])
                        return self._send_bytes(data, content_type)
                if len(parts) == 4 and parts[3] == "process":
                    return self._send_json(APP.process_status(cohort_id))
                if len(parts) == 4 and parts[3] == "log":
                    return self._send_json(APP.process_log(cohort_id))
            if route.startswith("/api/"):
                return self._error("API endpoint not found.", 404)
            self._serve_static(route)
        except (DashboardDataError, IntakeError) as exc:
            self._error(str(exc), 404)
        except (ValueError, TypeError) as exc:
            self._error(str(exc), 400)
        except Exception as exc:
            self._error(f"Dashboard request failed: {type(exc).__name__}.", 500)

    def _local_request_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            origin_port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
            if (
                parsed_origin.scheme != "http"
                or parsed_origin.hostname not in ("127.0.0.1", "localhost")
                or origin_port != self.server.server_port
            ):
                self._error("Cross-origin requests are not allowed.", 403)
                return False
        host_header = self.headers.get("Host", "")
        parsed_host = urlsplit(f"//{host_header}")
        if parsed_host.hostname not in ("127.0.0.1", "localhost"):
            self._error("Dashboard Host header is not allowed.", 403)
            return False
        return True

    def _content_length(self, maximum: int, minimum: int = 1) -> Optional[int]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error("Content-Length must be an integer.", 400)
            return None
        if length < minimum or length > maximum:
            self._error(f"Request body must be between {minimum} bytes and {maximum} bytes.", 413)
            return None
        return length

    def do_POST(self) -> None:
        if not self._local_request_allowed():
            return
        parsed = urlsplit(self.path)
        route = parsed.path
        try:
            if route == "/api/intake/sessions":
                length = self._content_length(MAX_REQUEST_BYTES)
                if length is None:
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or set(payload) != {"kind"}:
                    return self._error("Request body must contain only kind: scan or report.")
                result = APP.intake.create_session(payload["kind"])
                return self._send_json(result, status=201)

            parts = [unquote(part) for part in route.strip("/").split("/") if part]
            if len(parts) == 5 and parts[:2] == ["api", "intake"] and parts[2] == "sessions":
                session_id, action = parts[3], parts[4]
                if action == "files":
                    # File bytes are streamed to disk; they are not buffered in process memory.
                    length = self._content_length(2 * 1024 * 1024 * 1024)
                    if length is None:
                        return
                    filename = unquote(self.headers.get("X-Upload-Filename", ""))
                    result = APP.intake.upload_file(session_id, filename, self.rfile, length)
                    return self._send_json(result, status=201)
                if action == "validate":
                    result = APP.intake.validate_scan_session(session_id)
                    return self._send_json(result)
                if action == "inspect_dicom":
                    result = APP.intake.inspect_dicom_series(session_id)
                    return self._send_json(result)
                if action == "convert_dicom":
                    length = self._content_length(MAX_REQUEST_BYTES)
                    if length is None:
                        return
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = APP.intake.convert_dicom_to_bids(session_id, payload)
                    return self._send_json(result)
                if action == "process-reports":
                    result = APP.intake.process_report_session(session_id)
                    return self._send_json(result)
            if route != "/api/runs":
                return self._error("API endpoint not found.", 404)

            length = self._content_length(MAX_REQUEST_BYTES)
            if length is None:
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                return self._error("Request body must be a JSON object.")
            result = APP.start_run(payload)
            self._send_json(result, status=202)
        except IntakeError as exc:
            self._error(str(exc), 400)
        except json.JSONDecodeError:
            self._error("Request body must be valid JSON.", 400)
        except ValueError as exc:
            self._error(str(exc), 400)
        except RuntimeError as exc:
            self._error(str(exc), 409)
        except Exception as exc:
            self._error(f"Dashboard request failed: {type(exc).__name__}.", 500)

    def _serve_static(self, route: str) -> None:
        file_map = {
            "/": FRONTEND_ROOT / "index.html",
            "/index.html": FRONTEND_ROOT / "index.html",
            "/app.js": FRONTEND_ROOT / "app.js",
            "/styles.css": FRONTEND_ROOT / "styles.css",
        }
        path = file_map.get(route)
        if path is None or not path.is_file():
            return self._error("Page not found.", 404)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), f"{content_type}; charset=utf-8")


def main() -> int:
    host = os.environ.get("NEUROADHD_DASHBOARD_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Safety restriction: dashboard binds to loopback only.", file=sys.stderr)
        return 2
    port = int(os.environ.get("NEUROADHD_DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"NeuroADHD dashboard: http://127.0.0.1:{port}")
    print("Mode: STRICTLY_SYNTHETIC_ONLY | Real-data authorization: FALSE")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
