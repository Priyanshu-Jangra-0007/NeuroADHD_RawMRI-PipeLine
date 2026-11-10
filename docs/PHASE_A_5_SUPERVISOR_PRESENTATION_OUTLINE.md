# NeuroADHD RawMRI PoC V2 — Phase A.5
# Supervisor & Committee Presentation Outline

**Document ID:** `DOC-NA-A5-SPO-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Presentation Audience:** Thesis Supervisor, Departmental Review Committee, Academic Examiners  
**Deck Length:** 12 Slides (~20-25 minutes oral presentation)  

---

## Slide 1: Title, Project Scope & Research Supervision Context

- **Slide Title:** Automated Raw MRI Preprocessing & CC200 Functional Connectivity for ADHD Classification: Proof-of-Concept V2
- **Presenter / Candidate:** Priyanshu Jangra
- **Supervisor / Committee:** Department of Computer Science & Engineering / Biomedical Engineering
- **Target Audience Takeaway:** NeuroADHD RawMRI PoC V2 is an engineering-rigorous, containerized neuroimaging platform designed to bridge the gap between raw DICOM/BIDS MRI acquisitions and frozen statistical machine learning classifiers.
- **Bulleted Content:**
  - Project Scope: End-to-end transformation of raw T1w and rs-fMRI data into standardized functional connectivity features.
  - Phase A.5 Milestone: Completion of synthetic validation, mathematical registration audits, container memory isolation, and governance readiness.
  - Operational Boundary: 100% synthetic validation to date; real participant processing strictly disabled pending institutional review.
- **Visual Recommendation:** Split slide: left side showing repository metadata and container logo; right side showing high-level pipeline flow (Raw MRI $\to$ CC200 FC $\to$ Frozen Model $\to$ QC Report).
- **Speaker Talking Points:**
  > "Respected supervisor and committee members, today I present Phase A.5 of the NeuroADHD RawMRI PoC V2 pipeline. Over the past development phases, we have resolved critical architectural, mathematical, and computational bottlenecks, culminating in an end-to-end containerized system that is fully audited and ready for your formal review."

---

## Slide 2: Research Problem: ADHD Biomarker Discovery & Preprocessing Reproducibility

- **Slide Title:** Methodological Bottlenecks in Neuroimaging-Based ADHD Research
- **Target Audience Takeaway:** Classical fMRI pipelines suffer from software fragility, unmitigated pediatric motion artifacts, and inconsistent downstream ML integration.
- **Bulleted Content:**
  - **Clinical Need:** ADHD affects 5–7% of children globally; objective, neurobiologically grounded biomarkers are urgently needed to supplement subjective behavioral scales.
  - **Toolchain Brittleness:** Conventional fMRI toolchains require complex manual installations, exhibit operating-system dependencies, and lack strict resource isolation.
  - **Pediatric Motion Sensitivity:** Children with ADHD exhibit significantly elevated head motion, creating false positive functional connections if not rigorously scrubbed and regressed.
  - **Data Leakage & Model Mismatch:** Machine learning classifiers are frequently trained on inconsistently preprocessed public features, leading to catastrophic failure when applied to raw prospective data.
- **Visual Recommendation:** Two-panel graphic: (1) Head motion trace comparing a control vs ADHD pediatric subject with motion spikes; (2) Diagram of spurious distance-dependent functional connectivity inflation caused by micro-movements.
- **Speaker Talking Points:**
  > "Resting-state fMRI provides a unique window into large-scale brain networks such as the Default Mode and Frontoparietal networks. However, in pediatric ADHD, head motion and pipeline variability can introduce severe artifacts that mimic neurobiological findings. Our objective was to engineer a robust, automated pipeline with zero tolerance for silent failure."

---

## Slide 3: Proposed Pipeline Architecture: 19-Stage BIDS-to-Feature Engineering

- **Slide Title:** End-to-End Modular 19-Stage Pipeline Architecture
- **Target Audience Takeaway:** The pipeline operates in two strictly isolated tiers: Tier I performs physics-based neuroimaging preprocessing; Tier II performs read-only machine learning inference.
- **Bulleted Content:**
  - **Tier I: Preprocessing & Feature Extraction (Stages 1–17)**
    - Inputs: BIDS T1w anatomical and 4D resting-state BOLD NIfTI files.
    - Motion realignment, Otsu skull stripping, N4 bias field correction, 3-class tissue segmentation (CSF, WM, GM), and aggressive erosion.
    - Rigid EPI-to-T1w coregistration and non-linear SyN T1w-to-MNI registration.
    - Unified single-step pullback resampling directly onto canonical CC200 space ($63 \times 75 \times 61$, 3.0 mm).
    - aCompCor (5 CSF + 5 WM components) + Friston-24 motion expansion + simultaneous bandpass filtering ($0.01\text{--}0.10\text{ Hz}$).
    - CC200 ROI mean time-series extraction and Fisher-$z$ transformed correlation matrix ($\binom{200}{2} = 19{,}900$ features).
  - **Tier II: Downstream Frozen ML Inference (Stages 18–19)**
    - Frozen scikit-learn preprocessing pipeline $\to 100$ features $\to$ Frozen Random Forest classifier.
- **Visual Recommendation:** Comprehensive horizontal flow diagram displaying all 19 stages color-coded by domain (Validation = Blue, Structural = Green, Functional = Orange, ML = Purple).
- **Speaker Talking Points:**
  > "Our 19-stage workflow establishes an immutable separation of concerns. The neuroimaging engine handles physics, coordinate frames, and signal denoising. It produces a strictly standardized 19,900-element vector that interfaces seamlessly with our frozen classifier."

---

## Slide 4: Image Normalization Rigor: SyN Diffeomorphic Registration & Canonical CC200 Atlas

- **Slide Title:** Mathematical Precision in Stereotaxic Normalization & Parcellation
- **Target Audience Takeaway:** Single-step pullback resampling preserves spatial acuity, while cryptographic atlas verification guarantees anatomical reproducibility.
- **Bulleted Content:**
  - **Transformation Composition:** $\Phi_{\text{total}} = \Phi_{\text{T1w}\to\text{MNI}} \circ T_{\text{EPI}\to\text{T1w}} \circ T_{\text{motion}}$.
  - **Single Interpolation Step:** 4th-order B-spline interpolation applied directly from raw BOLD space to canonical CC200 space, eliminating intermediate interpolation blurring.
  - **Canonical CC200 Atlas Hash Reconciliation:**
    - Dimension: Exactly $(63, 75, 61)$ voxels.
    - Resolution: Exactly $3.0\text{ mm}$ isotropic.
    - Cryptographic SHA-256: `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`.
    - Labels: Exactly 200 contiguous integer regions ($1\text{--}200$).
- **Visual Recommendation:** Visual comparison of single-step resampling vs cascading multi-step resampling; 3D glass brain rendering of the 200 Craddock clusters.
- **Speaker Talking Points:**
  > "A major breakthrough in Phase A.2 and A.3 was resolving atlas discrepancies. We proved mathematically that intermediate resamplings degrade high-frequency spatial boundaries. By composing the SyN warp and rigid affines into a unified pullback operator, we resample raw BOLD directly onto the hash-verified canonical CC200 grid."

---

## Slide 5: Machine Learning Integration Contract: Frozen ADHD-200 Benchmark Artifacts

- **Slide Title:** Immutable Downstream Classifier Integration Contract
- **Target Audience Takeaway:** Machine learning inference is strictly deterministic, using cryptographically verified frozen artifacts mounted as read-only with zero retraining.
- **Bulleted Content:**
  - **Model Heritage:** Trained on historical ADHD-200 CC200 connectivity features as a reference benchmark.
  - **Frozen Preprocessing Pipeline (`final_preprocessing_pipeline.joblib`):**
    - SHA-256: `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249`
    - Steps: `SimpleImputer` $\to$ `VarianceThreshold(0.0)` $\to$ `StandardScaler` $\to$ `SelectKBest(f_classif, k=100)`.
  - **Frozen Classifier (`final_model.joblib`):**
    - SHA-256: `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef`
    - Architecture: `RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)`.
  - **Enforcement Rules:** Mounted read-only (`:ro`); checksum verified before unpickling; feature dimension strictly validated ($1 \times 19{,}900$).
- **Visual Recommendation:** Flowchart depicting the feature dimensionality reduction ($19{,}900 \to 100 \to \text{Random Forest} \to \text{Probabilities}$).
- **Speaker Talking Points:**
  > "To ensure reproducibility and eliminate data leakage, we treat the machine learning models as locked software binaries. The pipeline verifies their SHA-256 signatures prior to execution. There is zero dynamic retraining or hyperparameter tuning during inference."

---

## Slide 6: Containerization & 2.5 GB Memory Isolation: Linux Cgroups & OOM Protection

- **Slide Title:** Hardware Isolation & Predictable Resource Guarantees
- **Target Audience Takeaway:** The pipeline is containerized with kernel-enforced 2.5 GB memory limits, ensuring execution on standard laboratory hardware without uncontrolled host swapping.
- **Bulleted Content:**
  - **Container Base:** Standalone Docker image (`neuro_adhd_pipeline:v2`) running Python 3.11 on Debian Linux.
  - **Security Profile:** Non-root execution (`USER neuro`, UID 10001); no privilege escalation.
  - **Linux Cgroup v2 Isolation:** Hard memory limit of $2{,}560\text{ MiB}$ ($2.5\text{ GiB}$) via `--memory=2560m`.
  - **OOM Protection:** In the event of uncontrolled memory allocation, the Linux kernel terminates the container with `SIGKILL` (exit code 137), preventing workstation freezes.
  - **Host Safety Guard:** The pipeline checks its environment on startup; if run on an unconstrained host, it fails closed unless explicit developer override is provided.
- **Visual Recommendation:** Diagram of Docker container inside Linux host showing cgroup memory fence (2560 MiB) and non-root execution boundary.
- **Speaker Talking Points:**
  > "Scientific software often fails unexpectedly when deployed on resource-constrained academic workstations. We engineered this pipeline to operate within a hard 2.5 GB RAM boundary. Through synthetic stress tests, we verified that kernel OOM protection triggers cleanly without destabilizing the host system."

---

## Slide 7: Verification Evidence Summary: 179 Automated Tests & CI/CD Pipeline

- **Slide Title:** Rigorous Verification Evidence & Automated Testing Harness
- **Target Audience Takeaway:** Comprehensive test suites spanning unit, integration, and container levels guarantee that every algorithmic component is validated.
- **Bulleted Content:**
  - **Automated Test Suite:** Exactly 179 tests collected across `tests/unit/` and `tests/integration/`.
  - **CI/CD Execution Status (GitHub Actions Run #41433992271):**
    - **176 Passed, 3 Skipped** (when model directory is not provisioned in public CI).
    - **179 Passed** when authentic frozen model artifacts are provisioned.
  - **Fail-Closed Reporting:** `scripts/ci_test_summary.py` upgraded to strictly audit JUnit XML reports, detecting missing files and distinguishing errors from failures.
  - **Container Isolation Suite:** 3/3 passed verifying cgroup limits, OOM-killer termination, and fail-closed host behavior.
- **Visual Recommendation:** Screenshot/table of the test summary dashboard showing 179 passed tests, 83% statement coverage, and green CI badge.
- **Speaker Talking Points:**
  > "Our verification strategy is fully automated. In CI, 176 tests execute on every push, with the 3 model-dependent tests executing locally against the authentic model artifacts. Every single pull request and commit is cryptographically audited and validated."

---

## Slide 8: Scientific & Methodological Limitations: Exploratory Benchmark vs Clinical Tool

- **Slide Title:** Critical Scientific, Methodological & Clinical Limitations
- **Target Audience Takeaway:** The pipeline is an investigational research platform, not a certified clinical diagnostic instrument.
- **Bulleted Content:**
  - **Non-Clinical Disclaimer:** Classification probabilities are research variables; they must never be used for clinical diagnosis or treatment decisions.
  - **Synthetic-Only Validation:** All testing to date has used synthetic phantoms; real clinical scans introduce biological complexities (vascular pulsatility, sinus artifacts).
  - **Historical Model Characteristics:** The frozen classifier is a shallow benchmark (50 trees, depth 3) trained on legacy ADHD-200 data. Its diagnostic sensitivity is unestablished for modern scanners.
  - **Cross-Site Domain Shift:** Scanner hardware differences (1.5T vs 3.0T, coil channel counts) can bias functional connectivity estimates.
  - **Parcellation Scale:** CC200 functional clusters may not fully capture individualized neuroanatomical reconfigurations.
- **Visual Recommendation:** Prominent alert box highlighting "NON-CLINICAL RESEARCH PLATFORM ONLY"; table of potential confounding variables in real fMRI.
- **Speaker Talking Points:**
  > "We must be completely transparent regarding scientific limitations. This system is an investigational research tool. Its downstream Random Forest classifier was trained on historical data and has not undergone prospective clinical validation. It must never be represented as a medical diagnostic device."

---

## Slide 9: Ethics, Privacy & Governance: ICMR 2017 Guidelines & DPDP Act 2023 Compliance

- **Slide Title:** Ethical Governance & Data Privacy Architecture
- **Target Audience Takeaway:** Comprehensive governance framework aligned with Indian statutory mandates (ICMR 2017, DPDP Act 2023) and international Data Use Agreements.
- **Bulleted Content:**
  - **Indian Regulatory Alignment:**
    - ICMR Ethical Guidelines (2017): Protection of vulnerable pediatric participants, essentiality of data, secondary use restrictions.
    - DPDP Act 2023: Purpose limitation, verifiable parental consent for children's data (Section 9), strict fiduciary security safeguards.
  - **Debunking Open-Data Myths:** ADHD-200 is governed by formal DUA via NITRC/INDI; access requires registration and re-identification prohibitions.
  - **De-Identification & Facial Defacing:** Mandatory scrubbing of 18 HIPAA and Indian statutory identifiers; automated defacing with verified frontal-pole preservation.
  - **Linkage Code Security:** Master participant linkage keys stored exclusively on an offline, air-gapped physical medium.
- **Visual Recommendation:** Governance triangle showing intersection of ICMR 2017, DPDP Act 2023, and NITRC DUA requirements around the central secure pipeline.
- **Speaker Talking Points:**
  > "Because our institution is in India, data governance cannot rely solely on US HIPAA regulations. We have explicitly mapped our security controls to the ICMR 2017 guidelines and the Digital Personal Data Protection Act of 2023. Real data processing remains cryptographically locked until formal ethics clearance is obtained."

---

## Slide 10: Proposed First Research Pilot Protocol: Feasibility on Retrospective Data

- **Slide Title:** Proposed First Research Pilot: Real MRI Technical Feasibility
- **Target Audience Takeaway:** A minimal, controlled feasibility pilot on $N=6$ retrospective scans to stress-test pipeline memory and registration under real data conditions.
- **Bulleted Content:**
  - **Objective:** Evaluate computational stability, memory bounds, and registration convergence on real human MRI data.
  - **Proposed Cohort:** $N=6$ retrospective de-identified scans from ADHD-200 (3 TDC, 3 ADHD, balanced age/sex).
  - **10-Stage Execution Plan:** Governance verification $\to$ HMAC intake $\to$ Offline Docker deployment $\to$ Defacing audit $\to$ Preprocessing $\to$ 8-Point QC $\to$ CC200 FC extraction $\to$ Frozen model inference $\to$ Audit report.
  - **Success Criteria:** 0% crash rate, peak RAM $< 2.3\text{ GB}$, SyN Dice $\ge 0.85$, valid $19{,}900$ FC matrix (classification accuracy is explicitly excluded from success criteria).
  - **Prerequisites:** Dependent entirely on supervisor authorization and formal DUA execution.
- **Visual Recommendation:** 10-stage milestone timeline graphic highlighting the strict gating between ethics approval, data intake, and technical execution.
- **Speaker Talking Points:**
  > "Our proposed first research pilot is intentionally modest: 6 retrospective scans from the ADHD-200 repository. The objective is not to evaluate classification accuracy, but to stress-test our memory fences, registration convergence, and QC pipelines on real biological tissue."

---

## Slide 11: Decisions Required from the Research Supervisor: 8 QC Thresholds & Preprocessing Choices

- **Slide Title:** Formal Decisions & Authorizations Required from the Supervisor
- **Target Audience Takeaway:** The engineering team requests guidance and formal sign-off on 4 technical preprocessing policies and 8 QC acceptance thresholds.
- **Bulleted Content:**
  - **Technical Preprocessing Decisions:**
    1. Slice-Timing Correction policy (Option B: metadata-driven vs Option C: adaptive fallback).
    2. Susceptibility Distortion Correction architecture (Fieldmap requirement vs Syn-b0 fallback).
    3. Facial Defacing protocol (`pydeface` + frontal-pole preservation audit).
    4. Prospective acquisition parameters (TR, TE, FA, voxel dimensions, duration).
  - **Provisional QC Acceptance Thresholds (All Currently Pending):**
    - T1w Brain Volume ($600\text{--}2200\text{ cm}^3$), EPI-T1w NMI ($\ge 0.55$), T1w-MNI Dice ($\ge 0.85$), Mean FD ($\le 0.50\text{ mm}$), Max Spike ($3.0\text{--}5.0\text{ mm}$), Scrubbing ($\le 30\%$), tSNR ($\ge 40$), DVARS outliers ($>1.5\times$).
- **Visual Recommendation:** Summary table showing the 4 decisions and 8 thresholds with radio buttons indicating `[ ] APPROVED   [ ] REVISION REQUIRED`.
- **Speaker Talking Points:**
  > "In the Supervisor Decision Register, we have documented four technical preprocessing decisions and eight quantitative QC thresholds. We present our engineering recommendations, but we submit these parameters for your formal review, adjustment, and sign-off."

---

## Slide 12: Next Steps, Milestones & Timeline

- **Slide Title:** Roadmap Toward Real-Data Research Pilot & Dissertation Milestones
- **Target Audience Takeaway:** A structured 4-phase transition plan from current supervisor review to pilot execution and academic reporting.
- **Bulleted Content:**
  - **Phase 1 (Immediate):** Supervisor review of Phase A.5 package; resolution of Decision Register items.
  - **Phase 2 (Months 1–2):** Formal submission of Institutional Ethics Committee (IEC) protocol; execution of NITRC ADHD-200 Data Use Agreement.
  - **Phase 3 (Month 3):** Authorized execution of First Research Pilot ($N=6$ retrospective scans) in offline, memory-isolated container.
  - **Phase 4 (Months 4–5):** Post-pilot technical audit, evaluation of QC distributions, and preparation of peer-reviewed manuscript / dissertation chapter.
  - **Current System Status:** `READY_FOR_SUPERVISOR_DECISION`.
- **Visual Recommendation:** Gantt chart outlining the four project phases across a 6-month academic horizon.
- **Speaker Talking Points:**
  > "To conclude, NeuroADHD RawMRI PoC V2 has achieved complete engineering readiness. With your guidance and formal authorization, we are prepared to take the next measured steps toward ethical institutional submission and retrospective feasibility testing. Thank you, and I look forward to your questions and feedback."

