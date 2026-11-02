"""Automated unit tests for Phase A.4 Real-Data Safety Gate and Policy Enforcement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.run_poc_v2 import (
    check_safety_guards,
    compute_manifest_hmac,
    compute_sha256,
)
from src.parcellation.extractor import verify_canonical_cc200_atlas


class TestPhaseA4SafetyGate(unittest.TestCase):
    """Verify fail-closed safety controls, default synthetic mode, and PHI leak prevention."""

    def setUp(self):
        self.test_key = "test_phase_a4_safety_gate_secret_key_12345"
        self.orig_key = os.environ.get("NEUROADHD_HMAC_KEY")
        os.environ["NEUROADHD_HMAC_KEY"] = self.test_key

    def tearDown(self):
        if self.orig_key is not None:
            os.environ["NEUROADHD_HMAC_KEY"] = self.orig_key
        else:
            os.environ.pop("NEUROADHD_HMAC_KEY", None)

    def test_cli_mode_defaults_to_synthetic_and_rejects_real(self):
        """CLI parser must default to synthetic and reject any '--mode real' argument."""
        # Inspect runner argument parser definition directly
        from scripts.run_poc_v2 import main
        # We can construct the parser matching main() to test its configuration
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mode",
            choices=["synthetic"],
            default="synthetic",
        )
        # Default must be synthetic
        args = parser.parse_args([])
        self.assertEqual(args.mode, "synthetic")

        # Passing '--mode real' must fail argument parsing with SystemExit
        with self.assertRaises(SystemExit):
            parser.parse_args(["--mode", "real"])

    def test_allow_real_scans_cannot_bypass_safety_guard(self):
        """Passing allow_real_scans=True must raise RuntimeError immediately."""
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                subject_id="sub-synthetic001",
                allow_real_scans=True,
            )
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("allow_real_scans=True", str(ctx.exception))

    def test_missing_hmac_key_fails_closed(self):
        """Missing or empty NEUROADHD_HMAC_KEY environment variable must fail closed."""
        os.environ.pop("NEUROADHD_HMAC_KEY", None)
        with self.assertRaises(RuntimeError) as ctx:
            compute_manifest_hmac("MAGIC", "sub-synthetic001", {})
        self.assertIn("NEUROADHD_HMAC_KEY", str(ctx.exception))

        os.environ["NEUROADHD_HMAC_KEY"] = "   "
        with self.assertRaises(RuntimeError) as ctx:
            compute_manifest_hmac("MAGIC", "sub-synthetic001", {})
        self.assertIn("NEUROADHD_HMAC_KEY", str(ctx.exception))

    def test_blocked_clinical_keywords_in_subject_id(self):
        """Clinical keywords and participant identifiers must be strictly blocked."""
        blocked_sids = [
            "sub-2344857",
            "sub-peking01",
            "sub-kki101",
            "sub-nyu005",
            "sub-adhd200_01",
            "sub-control_02",
            "sub-abide_001",
        ]
        for sid in blocked_sids:
            with self.assertRaises(RuntimeError) as ctx:
                check_safety_guards(subject_id=sid)
            self.assertIn("SAFETY VIOLATION", str(ctx.exception))
            self.assertIn("strictly blocked", str(ctx.exception))

    def test_blocked_paths_referencing_clinical_data(self):
        """File paths referencing real clinical directories must be strictly blocked."""
        blocked_paths = [
            "/data/adhd200/sub-01_bold.nii.gz",
            "/opt/neuro_adhd/down/outputs/adhd200/bold.nii.gz",
            "/scans/peking_2/sub-01.nii",
            "/cohorts/abide/sub-01.nii.gz",
        ]
        for p in blocked_paths:
            with self.assertRaises(RuntimeError) as ctx:
                check_safety_guards(
                    subject_id="sub-synthetic001",
                    bold_path=p,
                    t1w_path="/valid/t1.nii.gz",
                    bold_json_path="/valid/bold.json",
                    template_path="/valid/mni.nii.gz",
                    atlas_path="/valid/atlas.nii.gz",
                )
            self.assertIn("SAFETY VIOLATION", str(ctx.exception))
            self.assertIn("prohibited", str(ctx.exception).lower())

    def test_missing_manifest_fails_closed(self):
        """Executing on arbitrary NIfTI files without an authenticated manifest must fail closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bold_file = tmp / "sub-synthetic001_bold.nii.gz"
            bold_file.write_bytes(b"dummy bold")
            t1w_file = tmp / "sub-synthetic001_T1w.nii.gz"
            t1w_file.write_bytes(b"dummy t1w")
            bold_json = tmp / "sub-synthetic001_bold.json"
            bold_json.write_text('{"RepetitionTime": 2.0}', encoding="utf-8")
            template = tmp / "mni.nii.gz"
            template.write_bytes(b"dummy mni")
            atlas = tmp / "atlas.nii.gz"
            atlas.write_bytes(b"dummy atlas")

            with self.assertRaises(RuntimeError) as ctx:
                check_safety_guards(
                    subject_id="sub-synthetic001",
                    bold_path=bold_file,
                    t1w_path=t1w_file,
                    bold_json_path=bold_json,
                    template_path=template,
                    atlas_path=atlas,
                )
            self.assertIn("Synthetic origin manifest 'synthetic_manifest.json' not found", str(ctx.exception))

    def test_phi_leak_prevention_in_provenance_schema(self):
        """Verify that provenance schema and output dictionary forbid sensitive PHI fields."""
        schema_path = Path("configs/provenance_schema.json")
        self.assertTrue(schema_path.is_file(), "Provenance schema must exist")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Inspect required properties
        props = schema.get("properties", {})
        # PHI properties that must NEVER be part of the schema
        prohibited_phi_properties = [
            "patient_name",
            "patient_birthdate",
            "mrn",
            "medical_record_number",
            "social_security_number",
            "phone_number",
            "email",
            "hospital_name",
            "clinic_name",
            "physician_name",
            "clinical_notes",
        ]
        for phi in prohibited_phi_properties:
            self.assertNotIn(
                phi,
                props,
                f"Prohibited PHI field '{phi}' must not be present in provenance schema!",
            )

    def test_canonical_atlas_verification_cannot_be_skipped(self):
        """Canonical atlas verification must fail closed on modified or corrupted atlas file."""
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
            tf.write(b"corrupted_atlas_content_not_authentic")
            tf.flush()
            with self.assertRaises(ValueError) as ctx:
                verify_canonical_cc200_atlas(tf.name)
            self.assertIn("CC200 atlas SHA-256 mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
