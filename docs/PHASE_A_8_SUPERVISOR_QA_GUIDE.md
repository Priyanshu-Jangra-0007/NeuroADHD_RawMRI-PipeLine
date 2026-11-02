# NeuroADHD RawMRI PoC V2 — Phase A.8
# Supervisor Question-and-Answer Preparation Guide

**Document ID:** `DOC-NA-A8-SQA-001`  
**Version:** 1.0.0  
**Phase:** Phase A.8 — Supervisor Handoff & External Research Authorization Preparation  
**Audience:** Research Student (for Supervisor Consultation and Thesis Defense)  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** [`0547574a6db2eb40668046e065fe8dcf989f09b1`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/0547574a6db2eb40668046e065fe8dcf989f09b1)  
**Execution Policy:** `STRICTLY_SYNTHETIC_ONLY`

---

## 1. Overview & Instructions for the Student

This guide contains **27 comprehensive technical questions and answers** anticipated during consultations with your research supervisor, thesis committee, or external examiners.

Every answer is phrased in clear, technically precise, and defensible academic English. 

> [!IMPORTANT]
> Where an answer touches upon an unresolved methodological choice (e.g., slice-timing correction policy, prospective acquisition parameters), the answer explains the scientific trade-offs honestly and explicitly states: **"This is an open decision for which we are requesting your supervisory guidance."**

---

## 2. Scientific & Neuroimaging Foundations

### Q1: Why focus on Attention-Deficit/Hyperactivity Disorder (ADHD)?
**Answer:**  
ADHD is one of the most prevalent neurodevelopmental disorders, affecting approximately 5% to 7% of school-aged children globally. Current clinical diagnostic pathways rely exclusively on qualitative behavioral rating scales (DSM-5 / Conners 3) filled out by teachers and parents. These assessments are vulnerable to subjective reporting bias, diagnostic delay, and socioeconomic disparities. Our objective is to investigate whether resting-state functional connectivity can provide objective, quantifiable neural biomarkers to support clinical evaluation and reduce diagnostic ambiguity.

### Q2: Why use resting-state fMRI (rs-fMRI) rather than task-based fMRI?
**Answer:**  
Task-based fMRI requires active, continuous participant engagement during cognitive challenges (such as the Go/No-Go or N-back working memory tasks). In pediatric ADHD cohorts, task performance is often confounded by differences in effort, comprehension, task anxiety, or fatigue. Resting-state fMRI requires only that the child remain awake and still while fixating on a crosshair for 5–7 minutes. It captures the brain's intrinsic, spontaneous functional architecture ($0.01\text{--}0.10\text{ Hz}$ BOLD fluctuations) with significantly higher participant compliance.

### Q3: Why benchmark the project against the ADHD-200 Consortium dataset?
**Answer:**  
The ADHD-200 dataset is the recognized international gold standard in computational ADHD research. It provides nearly 800 pediatric resting-state and anatomical scans collected across eight international imaging sites (including Peking University, NYU, and Kennedy Krieger Institute). Because it encompasses multiple scanner manufacturers (Siemens, Philips, GE) and varied clinical subtypes, benchmarking against ADHD-200 allows us to test inter-site generalization and robustness against real-world scanner heterogeneity.

### Q4: Why select the Craddock CC200 parcellation atlas over other templates like AAL or Schaefer?
**Answer:**  
The Craddock 2012 CC200 atlas was derived using two-level spectral clustering directly on resting-state fMRI data, enforcing spatial contiguity constraints to optimize functional homogeneity within each parcel. In contrast, the Automated Anatomical Labeling (AAL) atlas is based on gross anatomical sulci and gyri, which frequently group functionally distinct cortical populations into a single region. While the Schaefer 400 atlas is also functionally derived, 400 regions generate 79,800 pairwise connections, exacerbating the curse of dimensionality on pediatric sample sizes. CC200 offers an optimal balance between spatial resolution and statistical tractability.

### Q5: How do 200 ROIs generate exactly 19,900 functional connectivity features?
**Answer:**  
Functional connectivity between all parcel pairs forms a symmetric $200 \times 200$ Pearson correlation matrix. Because the diagonal represents self-correlation ($r_{ii} = 1.0$) and the matrix is symmetric ($r_{ij} = r_{ji}$), only the unique upper-triangular off-diagonal elements are retained. The exact number of unique connections is computed by the combination formula:
$$\frac{N(N-1)}{2} = \frac{200 \times 199}{2} = 19,900\text{ pairwise edges.}$$

### Q6: What is the Fisher-z transformation, and why is it necessary?
**Answer:**  
Pearson correlation coefficients ($r$) are bounded within the interval $[-1.0, 1.0]$. Consequently, the sampling distribution of raw correlation coefficients is non-normal and exhibits variance that depends heavily on the true correlation magnitude, violating the normality assumptions of linear and distance-based machine learning estimators. The Fisher-z transformation applies the inverse hyperbolic tangent:
$$z = \operatorname{arctanh}(r) = \frac{1}{2} \ln \left( \frac{1 + r}{1 - r} \right)$$
This maps bounded correlations onto an unbounded, approximately normal distribution with stabilized variance: $\sigma_z^2 \approx \frac{1}{N_{\text{eff}} - 3}$.

### Q7: Why select only 100 features from the 19,900 extracted connections?
**Answer:**  
In neuroimaging machine learning, evaluating 19,900 features on a dataset of several hundred participants leads to severe overparameterization and the curse of dimensionality ($p \gg N$), causing models to fit noise. Selecting the top 100 features using ANOVA F-ranking (`SelectKBest`) isolates the functional edges exhibiting the strongest group-level variance between ADHD and neurotypical participants while reducing model complexity by over 99.5%, preventing classifier overfitting and maintaining fast inference.

### Q8: Why use a Random Forest classifier instead of deep learning (e.g., Graph Neural Networks)?
**Answer:**  
Deep learning architectures (such as CNNs, Transformers, and GNNs) require tens of thousands of training scans to avoid overfitting high-dimensional connectivity spaces. On sample sizes typical of psychiatric neuroimaging ($N \approx 500\text{--}800$), Random Forests consistently match or outperform deep models. Furthermore, Random Forests are non-parametric, handle collinear tabular features robustly, do not require GPU acceleration, offer interpretable feature importances (Gini impurity / permutation importance), and execute easily within our 2.5 GB memory budget.

### Q9: How was data leakage controlled between feature extraction and classification?
**Answer:**  
Data leakage was rigorously prevented by decoupling model training from pipeline execution:
1. Both the `SelectKBest` feature mask and the `RandomForestClassifier` are **frozen historical artifacts** fitted exclusively on benchmark training partitions.
2. During pipeline execution, the pipeline runs strictly in **inference mode**. It never fits, updates, or normalizes parameters based on incoming scans.
3. Every evaluation scan is processed independently, ensuring that no population-level statistics or test labels bleed into preprocessing or feature selection.

### Q10: Why use a site-disjoint (Leave-One-Site-Out) validation strategy?
**Answer:**  
Standard random K-fold cross-validation allows scans from the same imaging center to appear in both training and test sets. Machine learning models easily exploit scanner-specific acquisition artifacts (e.g., slice thickness, RF coil profiles, pulse sequence nuances) to "predict" diagnosis by identifying the site. Site-disjoint validation (training on 7 sites, testing on the 8th) forces the classifier to learn neurobiological biomarkers that generalize across different MRI scanners rather than site-specific noise.

### Q11: Why were certain subjects excluded in historical benchmark audits?
**Answer:**  
Historical benchmark audits excluded scans failing mandatory quality control criteria:
1. Excessive head motion ($\text{Mean FD} > 0.50\text{ mm}$ or motion spikes $> 5.0\text{ mm}$).
2. Incomplete acquisition volumes ($T < 50$ volumes).
3. Incomplete anatomical coverage or severe artifact corruption causing registration failure to MNI152 space.
4. Truncated or missing BIDS metadata preventing TR or voxel size determination.  
Retaining artifact-laden scans injects spurious noise into covariance estimation, distorting group-level statistical comparisons.

---

## 3. Preprocessing, Registration & Signal Processing

### Q12: How does the pipeline address head motion in pediatric cohorts?
**Answer:**  
Pediatric participants, particularly those with ADHD, exhibit significantly more head motion than adult populations. We employ a three-tier motion mitigation strategy:
1. **Rigid Realignment:** 6-parameter rigid body realignment registering each BOLD volume to the middle reference volume.
2. **Friston-24 Regression:** Regressing out 24 motion parameters (6 translation/rotation parameters, their temporal derivatives, and the squares of both) to capture non-linear displacement effects.
3. **Volume Censoring (Scrubbing):** Flagging volumes with $\text{FD} > 0.50\text{ mm}$ or extreme DVARS, and rejecting the scan if more than 30% of volumes require scrubbing.

### Q13: What is anatomical CompCor (aCompCor), and why avoid Global Signal Regression (GSR)?
**Answer:**  
Global Signal Regression (GSR) computes the average time-series across the entire brain and regresses it out. While effective at removing motion artifacts, GSR mathematically shifts the distribution of correlation coefficients, artificially introducing negative correlations (anti-correlations) that do not reflect true neurophysiology. In contrast, **aCompCor** segments anatomical white matter (WM) and cerebrospinal fluid (CSF), erodes these masks to prevent partial volume contamination from gray matter, and extracts the top 5 principal components from each compartment. This isolates physiological noise (cardiac and respiratory pulsation) without altering cortical functional dynamics.

### Q14: What is the pipeline's spatial registration strategy?
**Answer:**  
Spatial registration is executed in two robust stages:
1. **Functional-to-Structural (EPI $\rightarrow$ T1w):** A rigid 6-degree-of-freedom coregistration using Normalized Mutual Information (NMI) aligns the mean motion-corrected BOLD volume to the subject's skull-stripped T1w anatomical volume.
2. **Structural-to-Template (T1w $\rightarrow$ MNI152):** A non-linear Symmetric Diffeomorphic (SyN) registration computes a deformation field mapping the subject's native T1w brain onto the standard MNI152 1mm stereotaxic template.  
The two transformations are concatenated and applied to the 4D functional volume in a single interpolation step to minimize spatial blurring.

### Q15: What decisions remain unresolved regarding slice-timing and distortion correction?
**Answer:**  
*This is an open methodological decision for which we request your direction.*  
- **Slice-Timing Correction:** Prospective scans with short $\text{TR} < 0.8\text{ s}$ do not benefit from STC and risk interpolation blurring, whereas retrospective scans with $\text{TR} \approx 2.0\text{ s}$ benefit from STC if `SliceTiming` metadata is reliable. We have recommended **Option C** (dataset-adaptive with fail-closed metadata fallback) in [`docs/PHASE_A_8_SUPERVISOR_DECISION_REQUEST.md`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_8_SUPERVISOR_DECISION_REQUEST.md).
- **Susceptibility Distortion Correction:** Prospective studies can mandate reverse-phase spin-echo fieldmaps (TOPUP), but retrospective datasets lack them. We recommend allowing a synthetic B0 fallback for historical benchmark data while requiring TOPUP for prospective cohorts.

---

## 4. Software Architecture, Security & Governance

### Q16: What does the Research Authorization Gate do?
**Answer:**  
The Research Authorization Gate ([`src/validation/authorization_gate.py`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/src/validation/authorization_gate.py)) is a programmatic, fail-closed security mechanism implemented in Phase A.6. It tracks 23 distinct governance items across supervisor decisions, institutional legal approvals, and operational safeguards. Before any real participant MRI data can be ingested or processed, the gate asserts that every single item is in the `APPROVED` state with a valid physical evidence reference and verified SHA-256 hash. If even one item is pending, the pipeline raises an `AuthorizationDeniedError` and halts immediately.

### Q17: What does synthetic-only validation establish?
**Answer:**  
Synthetic-only validation proves the mathematical and computational correctness of the pipeline without risking patient privacy, ethical breaches, or unauthorized data access. Using mathematically defined synthetic BOLD and T1w volumes, we verified:
1. End-to-end execution of all 19 pipeline stages.
2. Exact spatial alignment and CC200 canonical grid conformity.
3. Generation of exactly 19,900 finite Fisher-z features.
4. Seamless ingestion by the frozen classifier.
5. Operating system memory isolation within the 2.5 GB ceiling.

### Q18: What security vulnerabilities were discovered in the Phase A.7 audit?
**Answer:**  
Our Phase A.7 independent security audit identified seven vulnerability vectors:
- `SEC-01` (High): Symlink traversal in evidence path verification.
- `SEC-02` (High): Path traversal allowing absolute paths outside repository boundaries (`/etc/hosts`).
- `SEC-03` (Critical): An alternate legacy entrypoint (`scripts/run_single_scan_poc.py`) that bypassed gate checks.
- `SEC-04` (High): Case-sensitive environment variable checks and non-standard truthy representations.
- `SEC-05` (Med): Non-mandatory SHA-256 digests on approved evidence items.
- `SEC-06` (Med): Acceptance of directories and non-document file types as evidence.
- `SEC-07` (Low): Audit log vulnerability to newline injection and clinical cohort keyword leakage.

### Q19: How were these security vulnerabilities remediated?
**Answer:**  
All seven findings were remediated in commit `0547574`:
1. Evidence path verification checks every path component for symlinks *before* resolving paths.
2. Absolute paths are explicitly prohibited; evidence paths must be relative to the repository root.
3. The authorization gate and safety guards were embedded directly into `scripts/run_single_scan_poc.py`.
4. Environment variable scanning was updated to inspect all keys case-insensitively and reject any non-falsy value.
5. Approved items require mandatory 64-character SHA-256 hex digests verified via constant-time comparison.
6. Evidence files must be regular files with allowed extensions (`.pdf`, `.md`, `.txt`, `.json`, `.png`, `.jpg`, `.jpeg`).
7. Audit log entries are sanitized against control characters (`\n`, `\r`, `\x00`) and clinical cohort identifiers.  
Remediation was validated with a dedicated 14-test adversarial security test suite.

### Q20: Why is operating system memory capped at 2.5 GB?
**Answer:**  
Standard neuroimaging tools (e.g., FreeSurfer, fMRIPrep) often require 8 to 16 GB of RAM, precluding deployment on standard clinical workstations or resource-constrained hospital hardware. By enforcing a 2.5 GB RAM ceiling via Linux `cgroup v2` memory controllers, we guarantee that the pipeline operates reliably on low-cost hardware. In CI and container deployment, exceeding 2560 MiB triggers the Linux kernel Out-Of-Memory (OOM) killer, ensuring predictable memory boundaries.

---

## 5. Governance, Ethics & Next Steps

### Q21: Why is real MRI processing currently disabled?
**Answer:**  
Real MRI processing is disabled because ethical, institutional, and legal prerequisites have not yet been granted. Under the ICMR 2017 Guidelines and the DPDP Act 2023, analyzing human neuroimaging data requires formal Institutional Ethics Committee clearance and an executed Data Use Agreement. Disabling real processing by code ensures that no premature or unauthorized data ingestion occurs prior to legal sign-off.

### Q22: What institutional ethics and legal approvals are currently outstanding?
**Answer:**  
All five institutional legal approvals are currently pending:
1. `AUTH-IEC-01`: Institutional Ethics Committee clearance letter.
2. `AUTH-DUA-02`: NITRC / INDI Data Use Agreement for ADHD-200.
3. `AUTH-DSU-03`: Secondary-use parental consent and child assent waiver.
4. `AUTH-SUA-04`: Statutory bioethics review under ICMR 2017 & DPDP Act 2023.
5. `AUTH-DPP-05`: Institutional Data Protection Officer compliance review.

### Q23: Can the supervisor's HMAC signature replace formal institutional approval?
**Answer:**  
No. An HMAC signature is an internal cryptographic mechanism that proves knowledge of a secret key (`NEUROADHD_HMAC_KEY`). While it verifies that the authorization register was approved by the research supervisor within the software environment, it **is not a legally recognized electronic signature** under Indian statutory law. It complements, but cannot replace, official written clearance letters issued by the Institutional Ethics Committee.

### Q24: What is the proposed Phase A.9 pilot, and why is sample size limited to N=6?
**Answer:**  
The proposed First Research Pilot ([`docs/PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md)) evaluates $N=6$ retrospective scans (3 ADHD, 3 Neurotypical controls). This sample size is deliberately constrained: its purpose is strictly to evaluate technical feasibility—specifically, testing whether authentic raw scans pass all 8 QC criteria, converge during non-linear SyN registration, and execute within the 2.5 GB RAM ceiling without memory fragmentation. It is not intended to measure statistical classification accuracy.

### Q25: What criteria would define a successful feasibility pilot?
**Answer:**  
A successful feasibility pilot requires:
1. 100% completion of all 19 stages across the 6 scans without uncaught runtime exceptions.
2. Strict adherence to the 2.5 GB RAM limit without triggering the Linux kernel OOM killer.
3. Execution time $\le 20\text{ minutes}$ per scan.
4. Non-linear SyN registration achieving $\text{Dice} \ge 0.85$ and rigid coregistration achieving $\text{NMI} \ge 0.55$.
5. Generation of 19,900 finite Fisher-z features with zero NaN/Inf values.
6. Execution of downstream classification without error.

### Q26: What happens if a scan fails one of the eight QC thresholds?
**Answer:**  
The pipeline fails closed for that subject. If a metric violates its threshold (e.g., Mean $\text{FD} > 0.50\text{ mm}$ or $\text{Dice} < 0.85$), the pipeline halts processing immediately, moves intermediate working files to a secure quarantine directory, logs a structured QC rejection event in the subject provenance metadata, and outputs visual overlay diagnostic plots for researcher inspection.

### Q27: How does the pipeline ensure scientific reproducibility across different computing systems?
**Answer:**  
Reproducibility is achieved across four pillars:
1. **Containerization:** The entire environment (Python 3.11, NiBabel, Scipy, Scikit-learn, ANTs) is encapsulated in an unprivileged Docker container image (`neuro_adhd_pipeline:v2`).
2. **Locked Random Seeds:** Pseudo-random number generators across NumPy, Scipy, and Scikit-learn are pinned to fixed seeds.
3. **Cryptographically Locked Assets:** Reference atlases (CC200) and ML models are hash-verified before execution.
4. **Structured Provenance:** Every execution generates a machine-readable JSON provenance record detailing software versions, hardware environment, parameter configurations, and execution timestamps.
