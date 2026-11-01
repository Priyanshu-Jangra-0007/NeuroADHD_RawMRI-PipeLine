"""Integration tests for the executable PoC V2 runner (scripts/run_poc_v2.py).

Verifies that the actual runner orchestrator completes the complete 19-stage pipeline,
produces all required export artifacts, enforces strict spatial geometry at boundaries,
validates against the provenance schema, and blocks real subject processing.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import jsonschema
import nibabel as nib
import numpy as np
import pandas as pd

from scripts.run_poc_v2 import (
    check_safety_guards,
    compute_manifest_hmac,
    compute_sha256,
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from src.features.connectivity import get_feature_names


class TestRunnerPoCV2Synthetic(unittest.TestCase):
    def setUp(self):
        os.environ["NEUROADHD_HMAC_KEY"] = "test_synthetic_integrity_key_phase_a11_internal"
        np.random.seed(42)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)
        self.output_dir = self.work_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a small, deterministic synthetic session
        self.t_vol = 55
        self.subject_id = "sub-synthetic001"
        self.session_id = "ses-1"

        self.synth_paths = generate_synthetic_phantom_session(
            self.work_dir / "inputs",
            subject_id=self.subject_id,
            session_id=self.session_id,
            t_vol=self.t_vol,
            seed=42,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_runner_poc_v2_synthetic_end_to_end(self):
        """Invoke the actual runner orchestrator and verify all 19 stages and output artifacts."""
        res = run_poc_v2_pipeline(
            bold_path=self.synth_paths["bold_nii"],
            bold_json_path=self.synth_paths["bold_json"],
            t1w_path=self.synth_paths["t1w_nii"],
            atlas_path=self.synth_paths["cc200_atlas"],
            template_path=self.synth_paths["mni_template"],
            output_dir=self.output_dir,
            subject_id=self.subject_id,
            session_id=self.session_id,
            max_memory_gb=2.5,
            allow_real_scans=False,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["subject_id"], self.subject_id)
        run_out_dir = Path(res["output_dir"])
        self.assertTrue(run_out_dir.is_dir())

        # 1. CC200 ROI Time Series CSV
        roi_csv_path = run_out_dir / "roi_CC200.csv"
        self.assertTrue(roi_csv_path.is_file())
        roi_df = pd.read_csv(roi_csv_path)
        # Format matches historical C-PAC: 200 rows x (T+1) columns
        self.assertEqual(roi_df.shape, (200, self.t_vol + 1))
        self.assertEqual(roi_df.columns[0], "node/volume")
        np.testing.assert_array_equal(roi_df["node/volume"].to_numpy(), np.arange(1, 201))

        # 2. 19,900 Fisher-z Connectivity Features
        feat_npy_path = run_out_dir / "features_19900_fisher_z.npy"
        feat_csv_path = run_out_dir / "features_19900_fisher_z.csv"
        self.assertTrue(feat_npy_path.is_file())
        self.assertTrue(feat_csv_path.is_file())

        z_npy = np.load(feat_npy_path)
        self.assertEqual(z_npy.shape, (19900,))
        self.assertTrue(np.all(np.isfinite(z_npy)))

        feat_df = pd.read_csv(feat_csv_path)
        self.assertEqual(feat_df.shape, (19900, 2))
        self.assertListEqual(list(feat_df["feature_name"]), get_feature_names())

        # 3. Motion & Confound Reports
        motion_tsv_path = run_out_dir / "motion_parameters.tsv"
        self.assertTrue(motion_tsv_path.is_file())
        motion_df = pd.read_csv(motion_tsv_path, sep="\t")
        self.assertEqual(motion_df.shape, (self.t_vol, 6))

        fd_tsv_path = run_out_dir / "framewise_displacement.tsv"
        self.assertTrue(fd_tsv_path.is_file())
        fd_df = pd.read_csv(fd_tsv_path, sep="\t")
        self.assertEqual(fd_df.shape, (self.t_vol, 1))

        confounds_tsv_path = run_out_dir / "confounds.tsv"
        self.assertTrue(confounds_tsv_path.is_file())
        confounds_df = pd.read_csv(confounds_tsv_path, sep="\t")
        self.assertEqual(confounds_df.shape, (self.t_vol, 31))

        confound_rep_path = run_out_dir / "confound_report.json"
        self.assertTrue(confound_rep_path.is_file())
        with open(confound_rep_path, "r", encoding="utf-8") as f:
            conf_rep = json.load(f)
        self.assertEqual(conf_rep["num_columns"], 31)
        self.assertEqual(conf_rep["matrix_rank"], 31)

        # 4. QC Visual Overlays
        for qc_file in [
            "qc_registration_epi_to_t1w.png",
            "qc_registration_t1w_to_mni.png",
            "qc_tissue_masks.png",
            "qc_parcellation_cc200.png",
        ]:
            qc_p = run_out_dir / qc_file
            self.assertTrue(qc_p.is_file(), f"Missing QC overlay: {qc_file}")
            self.assertGreater(qc_p.stat().st_size, 5000, f"QC overlay too small: {qc_file}")

        # 5. Output Manifest covering all 12 artifacts
        manifest_path = run_out_dir / "output_manifest.json"
        self.assertTrue(manifest_path.is_file(), "output_manifest.json is missing")
        with open(manifest_path, "r", encoding="utf-8") as mf:
            out_manifest = json.load(mf)
        self.assertEqual(len(out_manifest["artifacts"]), 12)
        for art_key, art_info in out_manifest["artifacts"].items():
            art_file = run_out_dir / art_info["filename"]
            self.assertTrue(art_file.is_file(), f"Artifact file {art_info['filename']} missing on disk")
            self.assertEqual(art_info["sha256"], compute_sha256(art_file))

        # 6. Provenance JSON Report and jsonschema Validation
        prov_path = run_out_dir / "provenance.json"
        self.assertTrue(prov_path.is_file())
        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertEqual(prov["subject_id"], self.subject_id)
        self.assertEqual(prov["cohort_split"], "synthetic")
        self.assertIn("software_versions", prov)
        self.assertIn("hardware_execution", prov)
        self.assertLess(prov["hardware_execution"]["peak_rss_gb"], 2.5)

        for gate in [
            "gate1_resources",
            "gate2_bids_input",
            "gate3_atlas_grid",
            "gate4_spatial_alignment",
            "gate5_roi_coverage",
            "gate6_feature_schema_conformance",
        ]:
            self.assertEqual(prov["validation_gates"][gate]["status"], "PASSED")

        # Verify Gate 6 schema conformance and unassessed historical concordance
        gate6 = prov["validation_gates"]["gate6_feature_schema_conformance"]
        self.assertEqual(gate6["n_features"], 19900)
        self.assertTrue(gate6["finite_features"])
        self.assertTrue(gate6["edge_names_ordered"])
        self.assertFalse(gate6["historical_concordance_assessed"])
        self.assertEqual(gate6["historical_concordance_status"], "NOT_ASSESSED")

        # Verify output checksums dictionary in provenance
        self.assertEqual(len(prov["output_checksums"]), 11)
        for ck_key, ck_val in prov["output_checksums"].items():
            self.assertEqual(len(ck_val), 64, f"Checksum for {ck_key} is not 64-char hex")

        # Validate with jsonschema draft-7 validator
        schema_file = Path("configs/provenance_schema.json")
        self.assertTrue(schema_file.is_file())
        with open(schema_file, "r", encoding="utf-8") as sf:
            schema_dict = json.load(sf)
        jsonschema.validate(instance=prov, schema=schema_dict)

        # Scratch directory must be cleaned
        scratch_p = run_out_dir / "scratch"
        self.assertFalse(scratch_p.exists(), "Scratch directory was not cleaned")

    def test_runner_safety_guard_blocks_real_subject_id(self):
        """Runner must raise RuntimeError if subject ID targets sub-2344857."""
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards("sub-2344857", self.synth_paths["bold_nii"], self.synth_paths["t1w_nii"])
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))

    def test_runner_safety_guard_blocks_peking_cohort(self):
        """Runner must raise RuntimeError if subject ID targets locked Peking cohort."""
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards("sub-Peking_2_001", self.synth_paths["bold_nii"], self.synth_paths["t1w_nii"])
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))

    def test_runner_safety_guard_blocks_generic_real_subject_ids(self):
        """Runner must raise RuntimeError on generic real participant identifiers."""
        for generic_id in ["sub-001", "sub-control01", "sub-adhd01"]:
            with self.assertRaises(RuntimeError) as ctx:
                check_safety_guards(generic_id, self.synth_paths["bold_nii"], self.synth_paths["t1w_nii"])
            self.assertIn("SAFETY VIOLATION", str(ctx.exception))

    def test_runner_safety_guard_blocks_misleading_synthetic_id_without_manifest(self):
        """Subject with 'synthetic' in ID pointing to an arbitrary unverified file must be rejected."""
        arbitrary_nii = self.work_dir / "unverified.nii.gz"
        arbitrary_nii.write_bytes(b"dummy unauthenticated nifti")
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards("sub-synthetic-hijack", arbitrary_nii, arbitrary_nii)
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))

    def test_runner_safety_guard_blocks_tampered_synthetic_file(self):
        """Tampered synthetic file with hash mismatch against manifest must be rejected."""
        tampered_session = generate_synthetic_phantom_session(
            self.work_dir / "tampered", subject_id="sub-synthetic-tamp"
        )
        # Tamper with the BOLD file
        with open(tampered_session["bold_nii"], "ab") as f:
            f.write(b"corrupt_extra_bytes")

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                "sub-synthetic-tamp",
                bold_path=tampered_session["bold_nii"],
                t1w_path=tampered_session["t1w_nii"],
                bold_json_path=tampered_session["bold_json"],
                template_path=tampered_session["mni_template"],
                atlas_path=tampered_session["cc200_atlas"],
                manifest_path=tampered_session["synthetic_manifest"],
            )
        self.assertIn("SHA-256 mismatch against synthetic manifest", str(ctx.exception))

    def test_runner_safety_guard_blocks_missing_manifest(self):
        """Missing synthetic manifest must fail closed."""
        clean_session = generate_synthetic_phantom_session(
            self.work_dir / "no_manifest", subject_id="sub-synthetic-nometa"
        )
        # Remove manifest
        clean_session["synthetic_manifest"].unlink()
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                "sub-synthetic-nometa",
                bold_path=clean_session["bold_nii"],
                t1w_path=clean_session["t1w_nii"],
                bold_json_path=clean_session["bold_json"],
                template_path=clean_session["mni_template"],
                atlas_path=clean_session["cc200_atlas"],
            )
        self.assertIn("Synthetic origin manifest 'synthetic_manifest.json' not found", str(ctx.exception))

    def test_runner_safety_guard_blocks_explicit_bypass(self):
        """Passing allow_real_scans=True must raise RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                bold_path=self.synth_paths["bold_nii"],
                t1w_path=self.synth_paths["t1w_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                template_path=self.synth_paths["mni_template"],
                atlas_path=self.synth_paths["cc200_atlas"],
                allow_real_scans=True,
            )
        self.assertIn("allow_real_scans=True' is prohibited", str(ctx.exception))

    def test_runner_safety_guard_blocks_real_path(self):
        """Runner must raise RuntimeError if path references ADHD-200 raw data."""
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                "sub-synthetic001",
                bold_path="/down/outputs/adhd200/sub-01_bold.nii.gz",
                t1w_path=self.synth_paths["t1w_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                template_path=self.synth_paths["mni_template"],
                atlas_path=self.synth_paths["cc200_atlas"],
            )
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))

    def test_runner_failure_handling_and_diagnostic_report(self):
        """Runner must handle failure gracefully, record diagnostic report, quarantine failed run, and clean scratch dir."""
        fail_paths = generate_synthetic_phantom_session(
            self.work_dir / "inputs_fail", subject_id="sub-synthetic-fail"
        )
        # Update bold_json with invalid TR: 99s > 5s and re-hash in manifest so safety check passes but Stage 1 fails
        with open(fail_paths["bold_json"], "w", encoding="utf-8") as f:
            json.dump({"RepetitionTime": 99.0, "TaskName": "rest"}, f)

        # Update manifest hash & hmac for bold_json
        with open(fail_paths["synthetic_manifest"], "r", encoding="utf-8") as mf:
            m_dict = json.load(mf)
        m_dict["files"]["bold_json"]["sha256"] = compute_sha256(fail_paths["bold_json"])
        m_dict["integrity_hmac"] = compute_manifest_hmac(m_dict)
        with open(fail_paths["synthetic_manifest"], "w", encoding="utf-8") as mf:
            json.dump(m_dict, mf, indent=2)

        with self.assertRaises(ValueError) as ctx:
            run_poc_v2_pipeline(
                bold_path=fail_paths["bold_nii"],
                bold_json_path=fail_paths["bold_json"],
                t1w_path=fail_paths["t1w_nii"],
                atlas_path=fail_paths["cc200_atlas"],
                template_path=fail_paths["mni_template"],
                output_dir=self.output_dir,
                subject_id="sub-synthetic-fail",
                session_id=self.session_id,
                manifest_path=fail_paths["synthetic_manifest"],
            )
        self.assertIn("outside valid physiological range", str(ctx.exception))

        # Check failure report was generated in quarantined failed_run_* directory
        fail_dirs = list((self.output_dir / "sub-synthetic-fail").glob("failed_run_*"))
        self.assertEqual(len(fail_dirs), 1)
        fail_dir = fail_dirs[0]
        report_path = fail_dir / "pipeline_failure_report.json"
        self.assertTrue(report_path.is_file())
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "FAILED")
        self.assertEqual(rep["error_type"], "ValueError")
        self.assertIn("Stage 1: BIDS Input Validation", rep["stage"])

        # Ensure run_manifest.json also has status FAILED
        run_manifest_path = fail_dir / "run_manifest.json"
        self.assertTrue(run_manifest_path.is_file())
        with open(run_manifest_path, "r", encoding="utf-8") as rf:
            rm = json.load(rf)
        self.assertEqual(rm["status"], "FAILED")

        # Ensure no successful run_* directory was created
        self.assertEqual(len(list((self.output_dir / "sub-synthetic-fail").glob("run_*"))), 0)

    def test_runner_invalid_provenance_schema_raises_valueerror(self):
        """Pipeline must fail closed if provenance JSON fails jsonschema validation."""
        schema_file = Path("configs/provenance_schema.json")
        with open(schema_file, "r", encoding="utf-8") as sf:
            schema_dict = json.load(sf)

        # Invalid provenance: missing required gate6 and invalid cohort_split
        invalid_prov = {
            "timestamp_utc": "invalid-date",
            "subject_id": "sub-test",
            "site": "TEST",
            "cohort_split": "illegal_split",
            "software_versions": {},
            "hardware_execution": {},
            "validation_gates": {},
            "output_artifacts": {},
            "output_checksums": {},
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=invalid_prov, schema=schema_dict)

    def test_runner_safety_guard_blocks_fabricated_manifest_altered_hmac(self):
        """Fabricated manifest with altered HMAC must be blocked by safety guards."""
        with open(self.synth_paths["synthetic_manifest"], "r", encoding="utf-8") as f:
            m_data = json.load(f)
        m_data["integrity_hmac"] = "0" * 64
        altered_manifest = self.work_dir / "altered_manifest.json"
        with open(altered_manifest, "w", encoding="utf-8") as f:
            json.dump(m_data, f, indent=2)

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                self.synth_paths["bold_nii"],
                self.synth_paths["t1w_nii"],
                self.synth_paths["bold_json"],
                self.synth_paths["mni_template"],
                self.synth_paths["cc200_atlas"],
                manifest_path=altered_manifest,
            )
        self.assertIn("integrity token mismatch", str(ctx.exception))

    def test_runner_safety_guard_blocks_missing_manifest_entry(self):
        """Manifest missing any of the 5 required inputs must fail closed."""
        with open(self.synth_paths["synthetic_manifest"], "r", encoding="utf-8") as f:
            m_data = json.load(f)
        del m_data["files"]["cc200_atlas"]
        m_data["integrity_hmac"] = compute_manifest_hmac(m_data)
        incomplete_manifest = self.work_dir / "incomplete_manifest.json"
        with open(incomplete_manifest, "w", encoding="utf-8") as f:
            json.dump(m_data, f, indent=2)

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                self.synth_paths["bold_nii"],
                self.synth_paths["t1w_nii"],
                self.synth_paths["bold_json"],
                self.synth_paths["mni_template"],
                self.synth_paths["cc200_atlas"],
                manifest_path=incomplete_manifest,
            )
        self.assertIn("missing required entry", str(ctx.exception))

    def test_runner_safety_guard_blocks_swapped_input_files(self):
        """Swapped inputs (e.g. T1w passed as BOLD) must be rejected."""
        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                bold_path=self.synth_paths["t1w_nii"],
                t1w_path=self.synth_paths["bold_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                template_path=self.synth_paths["mni_template"],
                atlas_path=self.synth_paths["cc200_atlas"],
                manifest_path=self.synth_paths["synthetic_manifest"],
            )
        self.assertIn("does not match manifest path", str(ctx.exception))

    def test_runner_safety_guard_blocks_symlink_inputs(self):
        """Symlink substitutions must be rejected unconditionally."""
        symlink_bold = self.work_dir / "symlink_bold.nii.gz"
        if symlink_bold.exists():
            symlink_bold.unlink()
        symlink_bold.symlink_to(self.synth_paths["bold_nii"])

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                bold_path=symlink_bold,
                t1w_path=self.synth_paths["t1w_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                template_path=self.synth_paths["mni_template"],
                atlas_path=self.synth_paths["cc200_atlas"],
                manifest_path=self.synth_paths["synthetic_manifest"],
            )
        self.assertIn("Symlink substitution detected", str(ctx.exception))

    def test_runner_canonical_atlas_mode_rejects_synthetic_atlas(self):
        """In canonical atlas mode, an atlas not matching canonical CC200 checksum must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            run_poc_v2_pipeline(
                bold_path=self.synth_paths["bold_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                t1w_path=self.synth_paths["t1w_nii"],
                atlas_path=self.synth_paths["cc200_atlas"],
                template_path=self.synth_paths["mni_template"],
                output_dir=self.output_dir,
                subject_id=self.subject_id,
                session_id=self.session_id,
                manifest_path=self.synth_paths["synthetic_manifest"],
                atlas_mode="canonical",
            )
        self.assertIn("SHA-256 mismatch", str(ctx.exception))

    def test_runner_missing_provenance_schema_fails_run(self):
        """Pipeline must fail closed (raise FileNotFoundError) if provenance schema is missing."""
        import unittest.mock as mock
        with mock.patch("scripts.run_poc_v2.PROJECT_ROOT", self.work_dir / "fake_root"):
            with self.assertRaises(FileNotFoundError) as ctx:
                run_poc_v2_pipeline(
                    bold_path=self.synth_paths["bold_nii"],
                    bold_json_path=self.synth_paths["bold_json"],
                    t1w_path=self.synth_paths["t1w_nii"],
                    atlas_path=self.synth_paths["cc200_atlas"],
                    template_path=self.synth_paths["mni_template"],
                    output_dir=self.output_dir,
                    subject_id=self.subject_id,
                    session_id=self.session_id,
                    manifest_path=self.synth_paths["synthetic_manifest"],
                )
            self.assertIn("Mandatory provenance schema file missing", str(ctx.exception))

    def test_runner_post_write_tampered_output_fails_verification(self):
        """Pipeline must fail closed and quarantine staging if post-write output verification fails."""
        import unittest.mock as mock
        call_count = {"count": 0}
        orig_compute_sha = compute_sha256

        def mock_sha(p):
            if "features_19900_fisher_z.csv" in str(p) and ".staging_run_" in str(p):
                call_count["count"] += 1
                if call_count["count"] > 2:  # step 1 & 4 use calls 1 & 2; step 5 verification is call 3
                    return "0" * 64
            return orig_compute_sha(p)

        with mock.patch("scripts.run_poc_v2.compute_sha256", side_effect=mock_sha):
            with self.assertRaises(RuntimeError) as ctx:
                run_poc_v2_pipeline(
                    bold_path=self.synth_paths["bold_nii"],
                    bold_json_path=self.synth_paths["bold_json"],
                    t1w_path=self.synth_paths["t1w_nii"],
                    atlas_path=self.synth_paths["cc200_atlas"],
                    template_path=self.synth_paths["mni_template"],
                    output_dir=self.output_dir,
                    subject_id=self.subject_id,
                    session_id=self.session_id,
                    manifest_path=self.synth_paths["synthetic_manifest"],
                )
            self.assertIn("Output artifact verification failed", str(ctx.exception))

        # Staging directory must have been quarantined as failed_run_*
        failed_dirs = list((self.output_dir / self.subject_id).glob("failed_run_*"))
        self.assertGreaterEqual(len(failed_dirs), 1)
        # No successful run directory
        self.assertEqual(len(list((self.output_dir / self.subject_id).glob("run_*"))), 0)

    def test_runner_safety_guard_blocks_missing_hmac_token(self):
        """Manifest missing mandatory 'integrity_hmac' token must fail closed."""
        with open(self.synth_paths["synthetic_manifest"], "r", encoding="utf-8") as f:
            m_data = json.load(f)
        del m_data["integrity_hmac"]
        no_hmac_manifest = self.work_dir / "no_hmac_manifest.json"
        with open(no_hmac_manifest, "w", encoding="utf-8") as f:
            json.dump(m_data, f, indent=2)

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                self.synth_paths["bold_nii"],
                self.synth_paths["t1w_nii"],
                self.synth_paths["bold_json"],
                self.synth_paths["mni_template"],
                self.synth_paths["cc200_atlas"],
                manifest_path=no_hmac_manifest,
            )
        self.assertIn("missing mandatory 'integrity_hmac' token", str(ctx.exception))

    def test_runner_safety_guard_blocks_empty_or_malformed_hmac_token(self):
        """Manifest with empty or malformed non-hex HMAC token must fail closed."""
        with open(self.synth_paths["synthetic_manifest"], "r", encoding="utf-8") as f:
            m_data = json.load(f)

        for bad_token in ["", "   ", "not_a_hex_token", "1234"]:
            m_data["integrity_hmac"] = bad_token
            bad_manifest = self.work_dir / f"bad_hmac_{len(bad_token)}.json"
            with open(bad_manifest, "w", encoding="utf-8") as f:
                json.dump(m_data, f, indent=2)

            with self.assertRaises(RuntimeError) as ctx:
                check_safety_guards(
                    self.subject_id,
                    self.synth_paths["bold_nii"],
                    self.synth_paths["t1w_nii"],
                    self.synth_paths["bold_json"],
                    self.synth_paths["mni_template"],
                    self.synth_paths["cc200_atlas"],
                    manifest_path=bad_manifest,
                )
            msg = str(ctx.exception).lower()
            self.assertTrue("empty" in msg or "malformed" in msg)

    def test_runner_safety_guard_blocks_modified_manifest_filename(self):
        """Manifest with modified filename in files registration must fail validation."""
        with open(self.synth_paths["synthetic_manifest"], "r", encoding="utf-8") as f:
            m_data = json.load(f)
        m_data["files"]["bold_nii"]["filename"] = "altered_filename.nii.gz"
        m_data["integrity_hmac"] = compute_manifest_hmac(m_data)
        mod_filename_manifest = self.work_dir / "mod_filename_manifest.json"
        with open(mod_filename_manifest, "w", encoding="utf-8") as f:
            json.dump(m_data, f, indent=2)

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                self.synth_paths["bold_nii"],
                self.synth_paths["t1w_nii"],
                self.synth_paths["bold_json"],
                self.synth_paths["mni_template"],
                self.synth_paths["cc200_atlas"],
                manifest_path=mod_filename_manifest,
            )
        self.assertIn("Filename mismatch detected", str(ctx.exception))

    def test_runner_safety_guard_blocks_modified_resolved_path_with_same_basename(self):
        """Input with matching basename located in an unauthorized directory must be rejected."""
        unauth_dir = self.work_dir / "unauthorized_location"
        unauth_dir.mkdir(parents=True, exist_ok=True)
        spoofed_bold = unauth_dir / self.synth_paths["bold_nii"].name
        spoofed_bold.write_bytes(self.synth_paths["bold_nii"].read_bytes())

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                bold_path=spoofed_bold,
                t1w_path=self.synth_paths["t1w_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                template_path=self.synth_paths["mni_template"],
                atlas_path=self.synth_paths["cc200_atlas"],
                manifest_path=self.synth_paths["synthetic_manifest"],
            )
        self.assertIn("does not match manifest path", str(ctx.exception))

    def test_runner_safety_guard_blocks_symlink_manifest(self):
        """Synthetic manifest accessed via symlink must be rejected."""
        symlink_manifest = self.work_dir / "symlink_manifest.json"
        if symlink_manifest.exists():
            symlink_manifest.unlink()
        symlink_manifest.symlink_to(self.synth_paths["synthetic_manifest"])

        with self.assertRaises(RuntimeError) as ctx:
            check_safety_guards(
                self.subject_id,
                bold_path=self.synth_paths["bold_nii"],
                t1w_path=self.synth_paths["t1w_nii"],
                bold_json_path=self.synth_paths["bold_json"],
                template_path=self.synth_paths["mni_template"],
                atlas_path=self.synth_paths["cc200_atlas"],
                manifest_path=symlink_manifest,
            )
        self.assertIn("Symlink substitution detected", str(ctx.exception))

    def test_runner_safety_guard_blocks_missing_integrity_key(self):
        """Pipeline must fail closed if HMAC integrity key environment variable is not set."""
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                check_safety_guards(
                    self.subject_id,
                    bold_path=self.synth_paths["bold_nii"],
                    t1w_path=self.synth_paths["t1w_nii"],
                    bold_json_path=self.synth_paths["bold_json"],
                    template_path=self.synth_paths["mni_template"],
                    atlas_path=self.synth_paths["cc200_atlas"],
                    manifest_path=self.synth_paths["synthetic_manifest"],
                )
            self.assertIn("Mandatory HMAC integrity key not set in environment variable", str(ctx.exception))

    def test_runner_safety_guard_accepts_valid_synthetic_manifest(self):
        """Genuine synthetic session with valid manifest and HMAC must pass verification."""
        res = check_safety_guards(
            self.subject_id,
            bold_path=self.synth_paths["bold_nii"],
            t1w_path=self.synth_paths["t1w_nii"],
            bold_json_path=self.synth_paths["bold_json"],
            template_path=self.synth_paths["mni_template"],
            atlas_path=self.synth_paths["cc200_atlas"],
            manifest_path=self.synth_paths["synthetic_manifest"],
        )
        self.assertEqual(res["status"], "VERIFIED_SYNTHETIC")
        self.assertEqual(len(res["verified_hashes"]), 5)
        for k in ["bold_nii", "bold_json", "t1w_nii", "mni_template", "cc200_atlas"]:
            self.assertIn(k, res["verified_hashes"])


if __name__ == "__main__":
    unittest.main()

