# NeuroADHD RawMRI PoC V2 — Phase A.8
# Supervisor Presentation Guide: 15-Slide Briefing Deck

**Document ID:** `DOC-NA-A8-SPG-001`  
**Version:** 1.0.0  
**Phase:** Phase A.8 — Supervisor Handoff & External Research Authorization Preparation  
**Audience:** Primary Research Supervisor & Thesis Evaluation Committee  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** [`0547574a6db2eb40668046e065fe8dcf989f09b1`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/0547574a6db2eb40668046e065fe8dcf989f09b1)  
**Execution Policy:** `STRICTLY_SYNTHETIC_ONLY`

---

## Slide 1: Project Introduction & Title
- **Slide Title:** NeuroADHD RawMRI PoC V2: An Auditable, Memory-Isolated Neuroimaging Pipeline for Functional Connectivity Biomarker Discovery
- **Main Message:** NeuroADHD bridges raw, multi-modal MRI scans to machine-learning classifiers through a reproducible, fail-closed pipeline.
- **Talking Points:**
  - Addresses the gap between raw medical imaging data and downstream diagnostic machine learning.
  - Implements an end-to-end 19-stage pipeline from BIDS validation to Random Forest inference.
  - Enforces strict OS-level memory isolation (2.5 GB RAM) and cryptographic audit trails.
  - Currently operating under a verified synthetic-only safety gate awaiting supervisory sign-off.
- **Suggested Visual:**
  ```
  [Raw MRI DICOM/BIDS] ---> [19-Stage Preprocessing] ---> [CC200 Connectivity] ---> [Frozen Classifier]
  ```
- **Supervisor Question:** *Why did you build a custom pipeline instead of running fMRIPrep out of the box?*
  - *Response:* While fMRIPrep is an exceptional general-purpose tool, it requires 8–16 GB of RAM, runs for 4–8 hours per subject, and lacks an integrated, cryptographically gated link to frozen machine learning models. Our architecture enforces a 2.5 GB memory limit suitable for standard clinical hardware, completes in ~15 minutes, and incorporates a fail-closed research authorization gate.

---

## Slide 2: Research Problem & Clinical Motivation
- **Slide Title:** Clinical Motivation: Overcoming Subjectivity in Pediatric ADHD Diagnosis
- **Main Message:** Current ADHD diagnosis relies on subjective behavioral questionnaires; neuroimaging provides non-invasive, objective functional biomarkers.
- **Talking Points:**
  - ADHD affects 5–7% of children globally, presenting significant academic and social impairment.
  - Clinical diagnosis via DSM-5 questionnaires suffers from observer bias and demographic disparities.
  - Resting-state fMRI captures intrinsic functional network architecture without demanding task compliance.
  - Macro-scale connectivity changes offer potential physiological biomarkers to support objective clinical evaluation.
- **Suggested Visual:** Comparison table: Clinical Questionnaires (Subjective, Qualitative, High Variance) vs. Resting-State fMRI (Objective, Quantitative, High-Dimensional).
- **Supervisor Question:** *Are you claiming this pipeline will replace psychiatrists or clinical psychologists?*
  - *Response:* Absolutely not. This tool is conceived strictly as an objective diagnostic support system (clinical decision support), providing quantitative neurofunctional biomarkers to complement comprehensive psychiatric assessment.

---

## Slide 3: ADHD & Resting-State Functional Connectivity
- **Slide Title:** Neurobiology: Triple-Network Dysregulation in ADHD
- **Main Message:** ADHD is a disorder of large-scale distributed neural networks rather than isolated regional lesions.
- **Talking Points:**
  - **Default Mode Network (DMN):** Failure to attenuate during cognitive engagement causes attentional lapses and task-unrelated thought.
  - **Frontoparietal Network (FPN):** Hypo-connectivity in dorsolateral prefrontal cortex impairs executive control and working memory.
  - **Ventral Attention / Salience Network:** Impaired dynamic switching between internally and externally oriented cognition.
  - Functional connectivity measures temporal synchrony between these spatially segregated regions.
- **Suggested Visual:**
  ```mermaid
  graph LR
      DMN["Default Mode Network (DMN)<br>Mind-Wandering / Intrusions"] <-->|Aberrant Switching| SN["Salience Network (SN)<br>Insula / Anterior Cingulate"]
      SN <-->|Hypo-Connectivity| FPN["Frontoparietal Network (FPN)<br>Executive Control / Attention"]
  ```
- **Supervisor Question:** *How do you know these connectivity differences reflect neural changes rather than pediatric head motion?*
  - *Response:* Head motion is indeed a primary confound in pediatric cohorts. We address this using a multi-stage denoising protocol: Friston-24 motion regression, anatomical CompCor (5 WM + 5 CSF components), and volume scrubbing with a strict 30% censoring threshold.

---

## Slide 4: Benchmark Dataset: ADHD-200 Consortium
- **Slide Title:** Empirical Benchmark: The ADHD-200 Consortium Repository
- **Main Message:** ADHD-200 provides an open international benchmark, but presents substantial scanner and site heterogeneity.
- **Talking Points:**
  - Open-access repository of ~776 pediatric scans collected across 8 international imaging centers.
  - Includes typically developing controls and three ADHD subtypes (Inattentive, Hyperactive, Combined).
  - High clinical heterogeneity across scanner vendors (Siemens 3T, Philips 3T, GE 1.5T).
  - Serves as the training baseline for our frozen classifier and feature selection models.
- **Suggested Visual:** Map or table showing participating sites: Peking University, NYU, Kennedy Krieger Institute, OHSU, with scanner vendor and sample sizes.
- **Supervisor Question:** *Why benchmark against ADHD-200 if it is known to have site effects?*
  - *Response:* ADHD-200 is the established gold standard benchmark in the computational ADHD literature. Training across multiple sites and evaluating with leave-one-site-out validation tests true inter-scanner generalization rather than intra-site overfitting.

---

## Slide 5: NeuroADHD End-to-End Pipeline Architecture
- **Slide Title:** Pipeline Architecture: From Raw BIDS to Classification
- **Main Message:** The pipeline integrates 19 discrete stages spanning validation, structural processing, functional denoising, parcellation, and inference.
- **Talking Points:**
  - BIDS-compliant directory structures and sidecar validation.
  - Parallel structural and functional preprocessing streams.
  - Two-stage spatial registration: rigid EPI-to-T1w, followed by non-linear diffeomorphic SyN to MNI152.
  - Spatial parcellation using the Craddock CC200 template to extract 19,900 Fisher-z transformed features.
  - Frozen SelectKBest (100 features) feeding a frozen Random Forest classifier.
- **Suggested Visual:**
  ```mermaid
  flowchart LR
      A[Raw BIDS] --> B[Motion Realignment]
      A --> C[T1w Skull-Strip]
      C --> D[SyN MNI Warp]
      B --> E[aCompCor + Filter]
      E --> F[Coreg EPI-to-T1w]
      F --> D
      D --> G[CC200 Extraction]
      G --> H[19,900 Fisher-z]
      H --> I[Frozen RF Inference]
  ```
- **Supervisor Question:** *How do you prevent data leakage between preprocessing and classification?*
  - *Response:* The feature selector and Random Forest classifier are completely frozen artifacts pre-fitted on historical benchmark data. The pipeline operates strictly in inference mode; zero model parameters, thresholds, or feature selections are learned from incoming evaluation scans.

---

## Slide 6: MRI Preprocessing & Spatial Registration
- **Slide Title:** Preprocessing: Motion Realignment, Denoising, and Diffeomorphic Registration
- **Main Message:** Rigorous denoising and non-linear spatial normalization prepare functional signals for standard-space parcellation.
- **Talking Points:**
  - **Motion Realignment:** 6 rigid parameters + Friston-24 expansion to capture non-linear displacement.
  - **Nuisance Denoising:** Anatomical CompCor (5 WM + 5 CSF principal components), linear detrending, and bandpass filtering ($0.01\text{--}0.10\text{ Hz}$).
  - **Coregistration:** Normalized Mutual Information rigid alignment of mean EPI to anatomical T1w (QC threshold $\text{NMI} \ge 0.55$).
  - **Spatial Normalization:** Symmetric Diffeomorphic (SyN) registration of T1w to MNI152 (Dice overlap $\ge 0.85$).
- **Suggested Visual:** Orthogonal slice views demonstrating functional contour alignment over high-resolution T1w anatomical scans.
- **Supervisor Question:** *Why did you use anatomical CompCor instead of Global Signal Regression (GSR)?*
  - *Response:* Global signal regression is known to artificially introduce negative correlations (anti-correlations) and distort network topology. aCompCor extracts noise exclusively from non-neuronal white matter and CSF compartments, preserving authentic physiological correlations.

---

## Slide 7: Functional Parcellation: The Canonical CC200 Atlas
- **Slide Title:** Brain Parcellation: Authentic Craddock CC200 Atlas Integration
- **Main Message:** Parcellating the brain into 200 functionally homogeneous ROIs reduces dimensionality while preserving regional network boundaries.
- **Talking Points:**
  - Craddock et al. (2012) spectral clustering atlas constrained by spatial contiguity.
  - Canonical spatial dimensions: $63 \times 75 \times 61$ voxels, isotropic $3.0\text{ mm}$ spacing.
  - Exact cryptographic verification: SHA-256 `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`.
  - Exactly 200 non-overlapping contiguous regions labeled 1 through 200.
  - Solved prior registration discrepancies in Phase A.2.1 and Phase A.3.1.
- **Suggested Visual:** 3D brain glass rendering displaying the 200 cortical and subcortical parcels of the CC200 atlas.
- **Supervisor Question:** *Why CC200 instead of AAL (116 ROIs) or Schaefer (400 ROIs)?*
  - *Response:* CC200 was derived specifically from resting-state fMRI using spectral clustering to maximize functional homogeneity within parcels, whereas AAL is based on gross anatomical sulci that do not respect functional boundaries. Schaefer 400 produces 79,800 features, which drastically worsens the curse of dimensionality on small cohorts.

---

## Slide 8: Feature Extraction: 19,900 Fisher-z Connections to Top 100
- **Slide Title:** Feature Engineering: Fisher-z Tangent Features & Dimensionality Reduction
- **Main Message:** Upper-triangular pairwise correlations yield 19,900 Fisher-z features, filtered to 100 discriminative edges.
- **Talking Points:**
  - Pairwise Pearson correlation between 200 parcel time-series generates a $200 \times 200$ symmetric matrix.
  - Unique upper-triangular edges: $\frac{200 \times 199}{2} = 19,900$ connections.
  - Fisher-z transformation ($z = \operatorname{arctanh}(r)$) stabilizes variance for linear and tree-based models.
  - Pre-fitted `SelectKBest` with ANOVA F-ranking isolates the top 100 features.
  - Feature selector artifact is cryptographically locked (SHA-256 `228d3f...`).
- **Suggested Visual:** Heatmap of a $200 \times 200$ correlation matrix with an arrow pointing to a 1D vector of 100 selected edge weights.
- **Supervisor Question:** *Isn't selecting 100 features prone to overfitting?*
  - *Response:* Feature selection was executed strictly inside cross-validation folds on the historical benchmark, and the resulting 100-feature mask was locked into the frozen artifact. During PoC V2 execution, no feature selection is performed on incoming data; the mask is applied strictly as an immutable filter.

---

## Slide 9: Machine Learning Classifier: Frozen Random Forest
- **Slide Title:** Classification Model: Frozen Benchmark Random Forest
- **Main Message:** Downstream inference utilizes an immutable, pre-trained Random Forest classifier to ensure zero model drift.
- **Talking Points:**
  - Scikit-learn `RandomForestClassifier` (100 decision trees, Gini impurity criterion).
  - Cryptographically locked model binary (SHA-256 `d8eb23...`).
  - Strict read-only contract: the pipeline never retrains, refits, or updates model weights.
  - Outputs both discrete diagnostic classification (ADHD vs. Control) and continuous predicted probability scores.
- **Suggested Visual:** Flow diagram of Random Forest ensemble voting mechanism operating on 100 input features.
- **Supervisor Question:** *Why did you choose Random Forest over Deep Learning (e.g., Graph Neural Networks)?*
  - *Response:* On sample sizes typical of pediatric neuroimaging ($N \approx 500\text{--}800$), deep learning architectures are highly susceptible to overfitting and require massive hyperparameter tuning. Random Forests handle high-dimensional tabular inputs robustly, offer intrinsic feature importance metrics, and execute within our strict 2.5 GB memory budget.

---

## Slide 10: Previous Experimental Results & Scientific Limitations
- **Slide Title:** Empirical Findings: Benchmark Validation & Inherent Research Boundaries
- **Main Message:** Historical experiments demonstrated modest cross-site generalization (~62%); results represent academic proof-of-concept, not clinical efficacy.
- **Talking Points:**
  - **Within-Site Accuracy:** $67.4\% \pm 4.2\%$ across 10-fold cross-validation.
  - **Cross-Site (LOSO) Accuracy:** $62.1\% \pm 5.8\%$ across held-out clinical sites.
  - Illustrates the profound challenge of scanner protocol variance and phenotypic heterogeneity.
  - **Key Limitations:** Small cohort sizes, scanner vendor differences, pediatric motion confounding, sample imbalance.
  - PoC V2 has undergone rigorous synthetic testing, but has not yet processed authentic participant scans.
- **Suggested Visual:** Bar chart comparing within-site vs. cross-site classification accuracy with error bars.
- **Supervisor Question:** *62% accuracy is relatively low. How is this useful?*
  - *Response:* In multi-site psychiatric neuroimaging without site harmonization, 62% cross-site generalization is consistent with published benchmark literature (e.g., ADHD-200 Global Competition winners achieved 60–65%). It confirms that functional connectivity carries authentic signal above chance, but underscores that classification is an academic feasibility baseline rather than a commercial diagnostic tool.

---

## Slide 11: Engineering Architecture: Memory Isolation (<2.5 GB)
- **Slide Title:** System Architecture: Operating-System-Enforced Memory Isolation
- **Main Message:** Pipeline executes reliably on standard commodity hardware under an enforced 2.5 GB RAM ceiling.
- **Talking Points:**
  - Standard neuroimaging pipelines often consume 8–16 GB RAM, restricting deployment to high-performance computing clusters.
  - Implemented Linux `cgroup v2` memory capping: `--memory=2560m --memory-swap=2560m`.
  - Kernel Out-Of-Memory (OOM) killer enabled to prevent uncontrolled paging to disk.
  - Memory-efficient chunking and intermediate cleanup ensure peak memory remains below 2.2 GB.
  - Automated verification script ([`scripts/verify_container_memory.sh`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/scripts/verify_container_memory.sh)) validated in CI.
- **Suggested Visual:** Memory consumption curve over the 19 pipeline stages peaking at 2.1 GB, comfortably beneath the 2.5 GB red ceiling line.
- **Supervisor Question:** *What happens if a subject has unusually high resolution and spikes memory usage?*
  - *Response:* The Linux kernel OOM killer immediately terminates the container, and our memory guard script records a fatal quarantine event. The pipeline fails closed and logs the exact memory exhaustion point, preventing host system instability.

---

## Slide 12: Governance Architecture: Phase A.6 Authorization Gate
- **Slide Title:** Governance: The Fail-Closed Research Authorization Gate
- **Main Message:** Real participant MRI processing is programmatically blocked until all 23 institutional and supervisory requirements are satisfied.
- **Talking Points:**
  - Designed in Phase A.6 to prevent unauthorized data processing or ethics violations.
  - Tracks 23 governance items across three categories: Supervisor Decisions (12), Institutional Legal (5), Operational Safeguards (6).
  - Draft-07 JSON Schema validation and cryptographic SHA-256 evidence auditing.
  - Fails closed: attempting to run real participant scans raises `AuthorizationDeniedError`.
  - All 23 items are currently in `PENDING` state; real data processing is completely disabled.
- **Suggested Visual:** Diagram of the Authorization Gatekeeper standing between incoming real MRI data and pipeline execution stages.
- **Supervisor Question:** *Can a student bypass this gate simply by setting an environment variable or editing a flag?*
  - *Response:* No. As proven in our Phase A.7 audit, the gate actively scans all environment variables and CLI arguments for bypass attempts (e.g., `NEUROADHD_OVERRIDE_SAFETY`), immediately raising unrecoverable exceptions upon detection.

---

## Slide 13: Security Audit: Phase A.7 Adversarial Findings & Remediation
- **Slide Title:** Independent Security Audit: 7 Vulnerabilities Identified & Remediated
- **Main Message:** An adversarial security audit uncovered seven bypass vectors, all of which were hardened and verified with a 14-test security suite.
- **Talking Points:**
  - Evaluated symlink attacks, directory traversal, alternate entrypoints, and log injection.
  - **Remediated Findings:**
    - `SEC-01` (High): Symlink traversal in evidence resolution prohibited.
    - `SEC-02` (High): Absolute path boundary escapes blocked.
    - `SEC-03` (Critical): Legacy unhardened entrypoint (`run_single_scan_poc.py`) gated.
    - `SEC-04` (High): Case-insensitive env var bypass scanning enforced.
    - `SEC-05` (Med): Mandatory 64-char SHA-256 on approved evidence.
    - `SEC-06` (Med): Directory paths and invalid file extensions rejected.
    - `SEC-07` (Low): Sanitized audit log against control character injection and cohort keywords.
  - 14/14 adversarial tests passing in automated CI suite.
- **Suggested Visual:** Table showing SEC-01 through SEC-07 with red "Vulnerable" tags transitioning to green "Hardened & Verified" tags.
- **Supervisor Question:** *Why was scripts/run_single_scan_poc.py vulnerable?*
  - *Response:* It was an early development script created during Phase 2 before the unified Phase A.4/A.6 governance framework was implemented. In Phase A.7, we embedded identical fail-closed authorization checks and safety guards at the very top of that entrypoint.

---

## Slide 14: Outstanding Approvals: Institutional & Legal Prerequisites
- **Slide Title:** Governance Status: Pending External Institutional Clearances
- **Main Message:** While engineering and software controls are fully verified, all external institutional approvals remain pending.
- **Talking Points:**
  - **Institutional Ethics Clearance (`AUTH-IEC-01`):** Requires formal submission to the university Ethics Committee for pediatric computational neuroimaging.
  - **ADHD-200 Data Use Agreement (`AUTH-DUA-02`):** Requires counter-signed institutional agreement via NITRC / INDI.
  - **Secondary-Use Consent Waiver (`AUTH-DSU-03`):** Requires IEC confirmation under ICMR 2017 guidelines.
  - **Data Protection Officer Review (`AUTH-DPP-05`):** Requires statutory review under DPDP Act 2023.
  - **Physical Evidence:** No physical evidence has been fabricated; all items remain strictly `PENDING`.
- **Suggested Visual:** Checklist graphic with 5 institutional requirement boxes displaying orange "Awaiting Institutional Action" clocks.
- **Supervisor Question:** *What is the next immediate administrative step for the student?*
  - *Response:* Preparing the formal IEC protocol submission dossier and submitting the electronic DUA request through NITRC under the supervisor's official academic account.

---

## Slide 15: Supervisor Determinations Requested & Pilot Roadmap
- **Slide Title:** Action Items: Decisions Requested & Roadmap to Phase A.9 Pilot
- **Main Message:** We request supervisor sign-off on 4 methodological decisions, 8 QC thresholds, and the N=6 feasibility pilot protocol.
- **Talking Points:**
  - **Methodological Decisions:** Slice-timing policy (Option C), distortion correction (Syn-b0 fallback), facial defacing (`pydeface` + verification), prospective protocol.
  - **Quality Control:** 8 numerical thresholds covering head motion (Mean $\text{FD} \le 0.5\text{ mm}$), registration ($\text{Dice} \ge 0.85$), and signal quality.
  - **Phase A.9 Research Pilot:** A controlled $N=6$ retrospective feasibility trial (3 ADHD, 3 Control) designed strictly to evaluate pipeline stability under 2.5 GB RAM.
  - Formal sign-off in [`docs/PHASE_A_8_SUPERVISOR_DECISION_REQUEST.md`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_8_SUPERVISOR_DECISION_REQUEST.md).
- **Suggested Visual:** Roadmap timeline: Phase A.8 (Handoff & Sign-off) ---> Institutional Clearances ---> Phase A.9 ($N=6$ Pilot) ---> Thesis Evaluation.
- **Supervisor Question:** *Can we modify any of the 8 QC thresholds if we find them too strict or lenient?*
  - *Response:* Yes. The Decision Request Sheet explicitly provides modification fields for every metric. Any threshold changes will be logged in the machine-readable register with the supervisor's rationale and new cryptographic signatures.
