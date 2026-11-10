# FEATURE COMPATIBILITY PLAN

## 1. Overview & Core Compatibility Principle

The downstream machine-learning model (`final_model.joblib`) and feature-transformation pipeline (`final_preprocessing_pipeline.joblib`) are completely frozen. They cannot be adapted, refit, or retrained to accommodate discrepancies in newly extracted features.

Therefore, the raw pipeline must achieve exact numerical, structural, and ordering compatibility with the historical CC200 feature extraction implementation.

---

## 2. Mathematical Definition & Feature Pipeline

The existing feature generation code is located at `NeuroADHD/src/features/extract_cc200_connectivity.py`. The raw pipeline must produce outputs that interface seamlessly with this logic.

### 2.1 Pearson Correlation
For a scan with 200 ROIs and $T$ timepoints ($T \ge 50$), let $X \in \mathbb{R}^{T \times 200}$ denote the zero-mean standardized ROI time series. The sample correlation between region $i$ and region $j$ is:
$$r_{ij} = \frac{\sum_{t=1}^T (X_{t, i} - \bar{X}_i)(X_{t, j} - \bar{X}_j)}{\sqrt{\sum_{t=1}^T (X_{t, i} - \bar{X}_i)^2} \sqrt{\sum_{t=1}^T (X_{t, j} - \bar{X}_j)^2}}$$

- **Numerical Symmetrization**:
  To eliminate floating-point discrepancies, the correlation matrix $R$ is explicitly symmetrized:
  $$R_{\text{sym}} = \frac{R + R^T}{2}$$
  with diagonal elements fixed: $R_{\text{sym}}[i, i] = 1.0$.

### 2.2 Boundary Clipping & Fisher $r$-to-$z$ Transformation
The Fisher transformation maps bounded correlation values $r \in (-1, 1)$ to an approximately normally distributed variable $z \in (-\infty, \infty)$:
$$z_{ij} = \operatorname{arctanh}(r_{ij}) = \frac{1}{2} \ln \left( \frac{1 + r_{ij}}{1 - r_{ij}} \right)$$

To avoid numerical singularity at $r = \pm 1.0$, values are clipped using $\varepsilon = 10^{-7}$:
$$r_{\text{clipped}} = \operatorname{clip}(r_{ij}, -1 + \varepsilon, 1 - \varepsilon)$$
$$z_{ij} = \operatorname{arctanh}(r_{\text{clipped}})$$

### 2.3 Edge Selection & Authoritative Ordering
- **Count**: Exactly $\frac{200 \times 199}{2} = 19,900$ edges.
- **Index Definition**: Upper triangle without diagonal, strictly $k=1$ (`np.triu_indices(200, k=1)`).
- **Naming & Lexicographic Order**:
  $$\text{ROI\_001\_\_ROI\_002}, \text{ROI\_001\_\_ROI\_003}, \dots, \text{ROI\_001\_\_ROI\_200}, \dots, \text{ROI\_199\_\_ROI\_200}$$
  Confirmed by `NeuroADHD/src/features/extract_cc200_connectivity.py:L57-L60`.

---

## 3. Downstream Frozen Pipeline Interface

> [!NOTE]
> **Audit Note on Subject Counts and Test Suite Metrics (November 2026)**:
> - **Subject Count Discrepancy**: The imputer medians below were learned on 114 training subjects (from a historical 114 training + 20 test split = 134 subjects), whereas other documents cite 153 subjects for cohort scaling. This subject count is **unreconciled - needs the original split manifest** rather than guessing.
> - **Current Test Suite**: 343 tests collected: 336 passed, 7 skipped (5 Docker tests, 2 requiring `NEUROADHD_MODEL_DIR`), with 80% statement coverage across `src/` as of November 2026.

When a new 19,900-element feature vector is produced, it is transformed through `final_preprocessing_pipeline.joblib`:

```
Input: (1, 19900) 2D NumPy array
                ↓
1. SimpleImputer (strategy="median", keep_empty_features=True)
   - Replaces any missing values using medians learned on the 114 training subjects
                ↓
2. VarianceThreshold (threshold=0.0)
   - Eliminates constant features based on training variance
                ↓
3. StandardScaler (with_mean=True, with_std=True)
   - Centers and scales using training means and variances
                ↓
4. SelectKBest (score_func=f_classif, k=100)
   - Projects the 19,900 dimensions down to the EXACT 100 features learned from training data
                ↓
Output: (1, 100) Selected feature vector ready for final_model.joblib
```

---

## 4. Verification Protocol for Extracted Features

During Phase 2 (One-Scan Proof of Concept) and Phase 3 (Feature-Level Compatibility):
1. **Historical Subject Dual-Processing**:
   Select a subject present in both the raw BIDS dataset and the historical curated cohort (e.g. `1017176_session_1` from Peking_1).
2. **Concordance Metrics**:
   - Compute the Pearson correlation between the new 19,900 $z$-vector and the historical 19,900 $z$-vector.
   - Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across all 19,900 edges.
   - Inspect edge-by-edge scatter plots to verify the absence of systematic sign flips or spatial inversions.
3. **Tolerance Thresholds**:
   - Because exact bitwise reproduction of the 2015 C-PAC environment is impossible, correlation between re-preprocessed and historical features is expected to be high ($r > 0.85$ across all edges), but not 1.00.
   - Differences must be documented with explicit attribution (e.g. newer spline interpolation, modern boundary-based registration vs historical FLIRT).
