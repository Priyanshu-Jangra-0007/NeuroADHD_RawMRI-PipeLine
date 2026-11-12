# N=6 FEASIBILITY PILOT — QUALITY CONTROL & ACCEPTANCE CRITERIA

> **`QC_THRESHOLDS_PENDING_SUPERVISOR_APPROVAL`**
>
> **Document ID:** `DOC-PILOT-QC-001`  
> **Applies To:** NeuroADHD $N = 6$ Feasibility Pilot (`STRICTLY_SYNTHETIC_ONLY` rehearsal; future authorized real runs pending Step 8 approval)  
> **Anti-Fabrication Rule:** This document separates verified **engineering integrity gates** (already enforced in code) from **proposed research QC reporting metrics**. No clinical diagnostic thresholds or unapproved medical exclusion rules are claimed or invented.

---

## 1. Boundary Between Engineering Integrity Gates vs Research QC Thresholds

1. **Enforced Engineering Integrity Gates (Code-Verified `ValidationGates` 1–6):** Structural, numerical, memory, metadata, and checksum invariants that **must pass** (`PASSED`) or fail closed (`FAILED` / `SKIPPED_*`) on every run, synthetic or future-authorized.
2. **Proposed Research QC Evaluation Criteria (`PENDING_SUPERVISOR_APPROVAL`):** Imaging quality benchmarks recorded cleanly per subject so supervisor, neuroimaging lead, and IEC can define final real-data acceptance/exclusion cutoffs before Cohort Scaling.

---

## 2. MRI & Preprocessing Quality Metrics

| Metric / Check | Stage(s) | Automated Status / Enforcement | Engineering Requirement | Proposed Pilot Reporting / Target (`PENDING_SUPERVISOR_APPROVAL`) |
| :--- | :---: | :--- | :--- | :--- |
| **BIDS Metadata Completeness** | 1 | Enforced (Gate 2) | 4D BOLD (`t_vol >= 50`), valid `RepetitionTime > 0`, 3D T1w, matching BIDS subject ID, finite header affines | Zero unreadable headers; strict metadata mode for future real runs |
| **Head Motion (Power FD, `r=50 mm`)** | 2, 17 | Measured (`framewise_displacement.tsv`) | All `t_vol` FD values finite and non-negative; motion fitted on raw uncorrected BOLD prior to STC/SDC | Report `mean_fd_mm`, `max_fd_mm`, `% volumes FD > 0.5 mm` *(supervisor to ratify exclusion cutoff; do not invent clinical cutoff)* |
| **Slice-Timing Correction (`DEC-STC-01`)** | 3 | Enforced / Conditional | `SliceTiming` in `[0, TR]`; axis validated against header `slice_dim`; `+TR/2` temporal shift; zero guesswork | Report `status` (`APPLIED` vs `SKIPPED_*`), direction, reference time, interpolation (`cubic_spline`) |
| **Susceptibility Distortion (`DEC-SDC-02`)** | 4, 14 | Enforced / Conditional | BIDS fieldmap association unambiguous; positive Jacobian (`min J > 0`, `folding_voxel_count == 0`); post-SDC QC gate (`ncc_after >= ncc_before`) | Report `status`, `max_abs_mm` displacement, `min_jacobian`, `ncc_before -> ncc_after`, `nmi_before -> nmi_after`, `mask_dice` |
| **Anatomical Defacing & Privacy (`DEC-DEF-03`)** | 7, 23 | Enforced Privacy Gate | Original T1w SHA-256 unchanged; output shape/affine/finiteness preserved; `0` eroded brain mask voxels inside face mask; sanitized sidecars clean of identifier keys | Visual orthoslice inspection sign-off required (`manual_visual_review_required = true`) before any export |
| **Brain Extraction & Segmentation** | 8–11 | Measured + QC PNG | Non-empty eroded CSF (`>0`) and WM (`>0`) masks inside brain; bias-field finite (`qc_tissue_masks.png`) | Visual confirmation of CSF/GM/WM boundary preservation without cerebellum/cortex clipping |
| **Spatial Registration & MNI Grid** | 12–14 | Enforced (Gate 3 & 4) | Resampled BOLD matches MNI/CC200 4D target shape (`(63, 75, 61, T)` in canonical mode) with finite intensities (`qc_registration_*.png`) | Qualitative visual overlay verification + brain-outline alignment concordance |
| **Temporal Cleaning & Confound Rank** | 16–19 | Measured (`confound_report.json`) | Full column rank `== num_columns`, finite SVD condition number, `5` aCompCor + `24` Friston-24 regressors + `[0.01, 0.10] Hz` bandpass | Report condition number, residual variance check, and zero `NaN`/`Inf` voxels |

---

## 3. Functional Connectivity & Feature Acceptance Criteria

| Metric / Check | Stage(s) | Engineering Invariant (Enforced) | Pilot Verification Metric |
| :--- | :---: | :--- | :--- |
| **CC200 Parcel Coverage** | 20 | Exact `n_rois == 200`, `zero_variance_rois == 0` (`gate5_roi_coverage == PASSED`) | Report ROI count (`200`), zero-variance count (`0`), and parcellation overlay (`qc_parcellation_cc200.png`) |
| **Fisher-z Feature Vector** | 21 | Exact `19,900` unique upper-triangular edges (`1 <= i < j <= 200`); `finite_features == 19900` (`gate6_feature_schema_conformance == PASSED`) | Report `min`, `max`, `mean`, `std` Fisher-z values; inspect range symmetry and absence of saturation outliers |
| **Synthetic Physiological Recovery** | Synthetic Only | Positive & negative injected parcel pairs recover expected correlation polarity | Synthetic benchmark verification: positive pair `r > 0.70`, negative pair `r < -0.65` |
| **Reproducibility / Determinism** | 1–23 | Identical seed + identical inputs produce deterministic numeric outputs | Max absolute diff `<= 1e-7` across repeated runs for ROI timeseries and `19,900` Fisher-z vector |

---

## 4. Cohort-Level Feasibility Metrics ($N = 6$ Pilot)

Because $N = 6$ is strictly a **technical feasibility pilot**, cohort-level readiness is assessed on operational reliability rather than clinical accuracy:

1. **End-to-End Completion Rate:** Target `6 / 6` synthetic dry-run slots finish cleanly (`SUCCESS` or intentionally induced/isolated test control) with per-subject failure isolation verified (`continue_on_error=True` keeps remaining slots running on isolated bad input).
2. **Peak Memory Compliance (`<= 2.50 GB`):** Every sequential subject stays strictly below `2.50 GB` peak process RSS (`gate1_resources == PASSED`) with no cross-subject memory leak trend.
3. **Runtime Budgeting:** Measure and log per-subject wall-clock runtime (`seconds`) for synthetic phantom and canonical atlas configurations to project compute requirements cleanly before scaling.
4. **Metadata & Tool Transparency:** Explicit counts of `APPLIED` vs `SKIPPED_*` outcomes across STC, SDC, and Defacing, including missing binary warnings (`pydeface`, `topup`, `prelude`, `fugue`, `3dTshift`), never substituting silent pass states for skipped steps.
5. **Artifact Checksum & Provenance Integrity:** `100%` of completed runs emit schema-validated `provenance.json` (`configs/provenance_schema.json`) and SHA-256 `output_manifest.json` passing re-verification.

---

## 5. Explicit Non-Claims

- No diagnostic sensitivity, specificity, AUC, or ADHD predictive cutoff is defined or evaluated in the $N = 6$ feasibility pilot.
- Any downstream model inference executed on synthetic sessions carries `evaluation_mode = "SYNTHETIC_DEMONSTRATION_ONLY"` and **never** represents clinical evaluation.
- Final exclusion thresholds for real human motion (`FD`) or temporal SNR must be ratified by `[SUPERVISOR_NAME]` and documented in an approved amendment prior to scaling beyond the feasibility check.

