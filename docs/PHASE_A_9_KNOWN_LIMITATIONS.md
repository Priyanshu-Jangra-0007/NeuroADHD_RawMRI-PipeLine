# NeuroADHD RawMRI PoC V2 — Phase A.9
# Comprehensive Scientific & Methodological Known Limitations

**Document ID:** `DOC-NA-A9-KL-001`  
**Version:** 1.0.0  
**Phase:** Phase A.9 — Existing Pipeline Validation & Scientific Hardening  
**Target:** Primary Research Supervisor, Thesis Evaluation Committee, Regulatory Reviewers  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** [`4be77d03758983a65d7f9582011db8da11d20fd7`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/4be77d03758983a65d7f9582011db8da11d20fd7)  
**Execution Policy:** `STRICTLY_SYNTHETIC_ONLY`  
**Authorization Posture:** `NOT_AUTHORIZED`

---

## 1. Preamble & Scope

While the NeuroADHD RawMRI PoC V2 pipeline has achieved 100% synthetic-only verification, software stability, and memory isolation, clear scientific boundaries must be established.

This document formally records the **eight core scientific and methodological limitations** inherent to the current implementation and its synthetic validation framework.

---

## 2. Core Scientific & Technical Limitations

### 2.1 Synthetic Data vs. In Vivo Biological Brain Morphology
- **Limitation:** Synthetic phantoms utilize smooth geometrical shapes (spherical and ellipsoidal tissue compartments) and simulated Gaussian noise fields.
- **Scientific Implication:** Synthetic data cannot fully capture real in vivo anatomical complexities, such as intricate cortical folding patterns (gyri/sulci), severe local magnetic susceptibility gradients near sinuses, vascular pulsatility, or structural pathologies (e.g., ventricular enlargement, focal cortical dysplasia).
- **Boundary:** Synthetic execution proves **computational correctness, pipeline robustness, and memory bounds**; it **does not prove clinical diagnostic accuracy or biological fidelity**.

### 2.2 Spatial Grid Angular Quantization Limits
- **Limitation:** In low-resolution synthetic testing grids ($24 \times 24 \times 18$ voxels at $3.0\text{ mm}$ spacing), the discrete spatial sampling interval introduces an angular quantization noise floor of approximately $\approx 0.10^\circ\text{--}0.15^\circ$ ($0.002\text{ radians}$).
- **Scientific Implication:** Under zero-motion conditions, Mutual Information gradient optimization on low-resolution grids converges to rotational residuals within $\pm 0.15^\circ$. While clinically insignificant ($0.15^\circ$ at $50\text{ mm}$ radius corresponds to only $0.13\text{ mm}$ cortical displacement), this demonstrates the mathematical lower bound of discrete optimization without continuous sub-voxel interpolation.

### 2.3 Mean-Volume Reference Bias in Extremely Short Runs ($T < 10$)
- **Limitation:** Motion correction aligns each volume against the temporal mean volume $\bar{V} = \frac{1}{T}\sum V_t$.
- **Scientific Implication:** In very short synthetic tests ($T=4$), an abrupt translation in a single volume alters the mean volume itself by $\frac{1}{4} = 25\%$, biasing estimated single-volume parameters relative to the unshifted baseline. In production clinical scans ($T \ge 60$), this leverage is negligible ($< 1.6\%$), but investigators must be aware that parameters reflect displacement relative to the run average rather than volume 0. Pairwise inter-volume differences ($p_{t} - p_{t-1}$) are immune to this baseline offset.

### 2.4 Retrospective Susceptibility Distortion Correction (Fieldmap-Less Fallback)
- **Limitation:** Retrospective public repositories (including ADHD-200) were acquired without reverse-phase spin-echo fieldmaps (AP/PA), precluding gold-standard FSL TOPUP distortion correction.
- **Scientific Implication:** For retrospective data, the pipeline relies on rigid coregistration or synthetic deformation estimation against the structural T1w volume. Residual geometric distortions in orbitofrontal and inferior temporal cortices may lead to slight parcel misregistration in those specific regions. Prospective protocols must mandate dual-phase fieldmaps.

### 2.5 Site Effects & Scanner Inter-Vendor Generalization
- **Limitation:** Multi-site neuroimaging studies suffer from severe inter-scanner variance (Siemens vs. Philips vs. GE, 1.5T vs. 3.0T field strength, differing RF head coils).
- **Scientific Implication:** Machine learning models trained on one site frequently suffer from performance degradation when tested on unseen sites (historical drop from ~67% within-site to ~62% cross-site accuracy). The frozen Random Forest classifier is an exploratory research baseline and must not be interpreted as an invariant universal diagnostic tool.

### 2.6 Pediatric Head Motion Confounding
- **Limitation:** Pediatric ADHD participants exhibit significantly greater head movement than age-matched neurotypical controls.
- **Scientific Implication:** Despite Friston-24 nuisance regression, aCompCor, and volume censoring (scrubbing), residual sub-millimeter micro-movements can introduce distance-dependent functional connectivity biases that mimic biological neural alterations.

### 2.7 Strict Maintenance of Authorization & Safety Gates
- **Limitation:** Real participant MRI processing remains strictly disabled.
- **Governance Posture:** All 23 governance items in [`configs/research_authorization_register.json`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/configs/research_authorization_register.json) are currently `PENDING`.
- **Boundary:** No real MRI data (prospective or retrospective) will be processed, and the $N=6$ feasibility pilot will not be executed until official Institutional Ethics Committee clearance and Data Use Agreements are executed and verified.

### 2.8 Unproven Historical Feature Concordance on Real Participant Scans
- **Limitation:** The V2 preprocessing pipeline has been verified against synthetic phantoms and unit tests, but has **never been compared against historical C-PAC CC200 features on a real participant scan**.
- **Provenance Status:** Provenance records explicitly track `historical_concordance_status: NOT_ASSESSED`.
- **Scientific Implication:** Feature compatibility between the newly engineered pipeline outputs (`roi_CC200.csv` and resulting Fisher-$z$ connectivity vectors) and the historical training features used to fit the downstream frozen model remains empirically unproven. Investigators must not assume that the frozen model will produce calibrated probability scores without a formal, authorized concordance audit on real data. Execution stays strictly synthetic until governance authorization is achieved.

---

## 3. Methodological Summary

$$\mathbf{SYNTHETIC\_VERIFIED \neq CLINICALLY\_VALIDATED}$$
The NeuroADHD RawMRI pipeline is established as an engineered, reproducible, memory-isolated neuroimaging framework. It stands ready for formal supervisor review and institutional governance filings without premature real-data claims.
