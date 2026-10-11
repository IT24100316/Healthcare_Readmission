# Machine Learning Project Final Report (Binary Classification Track)
**Domain:** Healthcare  
**Dataset:** Diabetes 130-US Hospitals  
**Target Definition:** 30-Day Early Readmission Risk (Binary: `<30` = 1, `NO` / `>30` = 0)  

---

## 1. Problem Framing

* **Stakeholder:** Hospital Care-Planning Team, Discharge Navigators, and Healthcare Administrators.
* **Decision Need:** Identifying which diabetic inpatients require immediate, intensive post-discharge transitional care (e.g., home health visits, telephone triage within 48 hours, medication counseling) to prevent early acute readmission.
* **Primary Lens:** 30-day early hospital readmission risk prediction.
* **Secondary Lens:** High-risk patient profiling (identifying clinical drivers and comorbidity profiles of patients flagged as high risk).
* **Unit of Analysis:** One single hospital encounter (visit) by a diabetic patient.
* **Exact Task/Output:** **Supervised Binary Classification**:
  * **Positive Class (1):** Patient readmitted within 30 days (`<30`) (~11.6% of encounters).
  * **Negative Class (0):** Patient not readmitted within 30 days (`NO` or `>30` combined) (~88.4% of encounters).
* **Clinical Rationale:** Aligns directly with hospital operational workflows and the US Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP). Distinguishing between late readmission (>30 days) and no readmission is clinically irrelevant for acute 30-day discharge transition planning.

---

## 2. Decision Log
| Decision | Options Considered | Chosen | Reason | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Target Formulation** | 3-Class (`NO`, `>30`, `<30`) vs. **Binary (`<30` vs. `Others`)** | **Binary (`<30` = 1, `NO` + `>30` = 0)** | 3-class prediction suffers from high clinical noise between `>30` and `NO` (post-discharge outpatient adherence and social determinant data do not exist), creating the "Accuracy Paradox" (~45%–50% balanced accuracy). Binary formulation directly targets the operational 30-day transitional care window. | CMS HRRP guidelines; targeted EDA showing indistinguishable clinical profiles between late returns (>30 days) and no returns. |
| **Metric Selection** | Raw Accuracy vs. **Recall / F1-Score / PR-AUC / ROC-AUC** | **Recall + Positive F1 + PR-AUC** | With ~11.6% positive class prevalence, raw accuracy is deceptive (a dummy baseline predicting all 0 yields ~88.4% accuracy). Missing an early high-risk bounce-back is far more costly than an extra follow-up phone call. | Target distribution showing severe class imbalance (~11.6% positive). |
| **Splitting Strategy (Step 9)** | Random split vs. **Patient-level split (`patient_nbr`)** | **Patient-level split with binary stratification** | Repeated encounters exist for the same patient ID (~71k patients across ~101k visits). A random split causes severe patient memorization leakage. | Encounters per Patient chart (up to 40 visits per patient); 0 patient overlap verified. |
| **Duplicate Removal (Step 2)** | Skip check vs. Explicit `drop_duplicates()` | **Explicit `drop_duplicates()`** | Ensures exact duplicate rows do not artificially inflate model confidence. | EDA Section 4 and Step 2 confirm 0 exact duplicates. |
| **Data Type Fixing (Step 4)** | Leave as-is vs. Explicit casting | **Explicit casting** | Admin ID code columns stored as `int64` and count columns as `float64` cause downstream parsing and scaling errors. | `df.dtypes` inspection after cleaning (IDs cast to `object`, counts to `int`). |
| **Descriptor Column Strategy (Step 6)** | Keep alongside `_id` vs. Drop as redundant | **Drop `_desc` columns** | `_desc` columns are exact text equivalents of `_id` columns. Keeping both causes multicollinearity and double-counting in the feature space. | Column dictionary & Preprocessing Step 6. |
| **Resampling Strategy (Step 6A)** | Ignore imbalance vs. Algorithmic weighting vs. Resampling | **Algorithmic Weighting (`class_weight='balanced'`) + Optional Phase 6A Training-Only Resampling (SMOTE / Oversampling)** | Provides flexibility to experiment with SMOTE or RandomOverSampler while enforcing the **Golden Rule of zero data leakage**: resampling is applied strictly to training data only (`X_train_final`, `y_train`); test set remains 100% natural. | Cell 94 comparative table and leakage verification. |
| **Feature Validation (FE Step 7)** | Skip vs. Explicit validation cell | **Explicit validation cell** | Confirms 0 NaNs, correct shapes, zero target/identifier leakage, and valid numeric scaling ranges before model training. | Best practice for evaluation-ready ML pipelines. |

---

## 3. Workflow Diagram
*(To be inserted)*

---

## 4. Exploratory Data Analysis (EDA)

### 4.1. EDA Insight Log
| Analysis Area | Observation / Insight | Evidence (Real Data) | Action Taken for Modeling |
| :--- | :--- | :--- | :--- |
| **Target Variable** | The binary target exhibits severe class imbalance: Positive `<30` (~11.2%) vs. Negative `NO` + `>30` (~88.8%). | Side-by-side Target distribution plot showing `<30` as minority (11,357 vs 90,409 encounters in raw data; ~11.6% post-terminal filtering). | Stratified patient-level split based on binary label; prioritize Recall, Positive F1, and PR-AUC over raw accuracy. |

| **Distributions (Numerical)** | Count variables (e.g., `number_inpatient`, `number_emergency`) are heavily right-skewed. | Univariate numerical histograms and boxplots. | Retain clinical high-utilizers; apply `log1p` transformation in the ColumnTransformer pipeline to compress extreme tails. |
| **Distributions (Categorical)** | Dozens of medications (e.g., `chlorpropamide`, `acetohexamide`) have >99% "No" values. `diag` features have 700+ unique ICD-9 codes. | Univariate categorical frequency plots. | Drop near-zero variance medications to reduce noise. Group ICD-9 codes into 9 broad clinical categories. |
| **Feature Relationships** | Highest correlations: `num_medications` & `time_in_hospital` (0.47). No severe multicollinearity (>0.8). | Pearson correlation heatmap. | Retain numerical features as they provide distinct clinical signals without severe multicollinearity. |
| **Target Relationships** | `A1Cresult` and medication risk profiles do not increase monotonically. `number_inpatient` strongly correlates with `<30` risk. | Targeted Feature vs Readmission proportion plots. | Use One-Hot Encoding for nominals and careful Ordinal Encoding for binary/ordered categories to avoid false linear risk assumptions. |
| **Feature Association (Categorical)** | Plotting dozens of categorical features with arbitrary filters (>100 samples) clutters analysis. | Cramér's V statistical ranking. | Calculated Cramér's V for all categorical features to rank true statistical association with the target; plotted Top 9. |

### 4.2. EDA: Data Issues Log
| Data Issue | Observation / Evidence | Impact on the Task | Action Taken |
| :--- | :--- | :--- | :--- |
| **Excessive missing data in `weight`** | `weight` is missing in ~97% of records. | Adds noise and bias; imputing would fabricate ~97,000 values. | Drop the `weight` column completely before modeling. |
| **Masked Missing Values (Placeholders)** | Missing values are coded as `'?'`, `'Unknown/Invalid'`, `'NULL'`, `'Not Available'`, or `'None'`. | Pandas cannot automatically detect these, causing parsing errors. | Replace all textual placeholders with proper `np.nan` values during initial load. |
| **Hidden Nulls in Admin Codes** | `IDS_mapping.csv` reveals integer IDs (e.g., 5, 6, 8 for admission type) map to "NULL" or "Not Mapped". | Look like valid categories to models but actually represent missingness. | Parse mapping definitions and convert conceptually missing IDs to `np.nan` or `"Unknown"`. |
| **Logical Impossibility (Terminal Codes)** | Thousands of patients (~2,300 encounters) were discharged to Hospice or Expired (IDs 11, 13, 14, 19, 20, 21). | These patients physically cannot be readmitted, which artificially inflates the negative class with uninformative negatives. | Drop all rows containing terminal discharge disposition codes to prevent target leakage. |
| **Patient Memorization Leakage** | No exact duplicate rows, but up to 40 encounters exist for the same patient ID (~71k patients, ~101k rows). | A random split allows the same patient in train and test, causing severe data leakage (memorization). | Split the data strictly grouping by `patient_nbr` with binary stratification. |

---

## 5. Preprocessing & Feature Engineering Log
| Decision | Options Considered | Chosen | Reason (Why we chose/rejected) | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1: Data Structure Understanding** | 1) Proceed blind. <br>2) Inspect shapes, dtypes, and null counts explicitly. | **Inspect explicitly** | Ensures verified starting state (101,766 rows × 50 cols) before any mutation. | Step 1 verification cell output. |
| **Step 2: Duplicate Removal** | 1) Skip duplicate check. <br>2) Explicit `drop_duplicates()`. | **Explicit `drop_duplicates()`** | Ensures exact duplicates don't inflate model confidence, even if currently 0. | Step 2 code cell confirms 0 duplicates removed. |
| **Step 3 (a): Placeholder Handling** | 1) Keep strings as categories. <br>2) Convert placeholders to `np.nan`. | **Convert placeholders to `np.nan`** | Allows vectorized missing value operations in pandas and prevents fake category levels. | Step 3a code cell. |
| **Step 3 (b): Terminal Patients Drop** | 1) Keep all rows. <br>2) Drop terminal discharge codes (11, 13, 14, 19, 20, 21). | **Drop terminal codes** | *Rejected 1:* Deceased/hospice patients cannot physically be readmitted. <br>*Chose 2:* Focuses the model strictly on viable readmission candidates. | `discharge_disposition` mapping counts and Step 3b plot. |
| **Step 3 (c-e): Informative Missingness & Tests** | 1) Mode imputation. <br>2) Impute as `"Unknown"` / `"Not tested"`. | **Impute as `"Unknown"` / `"Not tested"`** | *Rejected 1:* Falsely assumes missingness is completely at random. <br>*Chose 2:* Lack of specialist/payer and unperformed A1C tests carry distinct clinical context. | Step 3d/3e distribution checks. |
| **Step 3 (f): Non-imputable Critical Flaws** | 1) Mode Impute `diag` and `gender`. <br>2) Drop flawed rows. | **Drop rows (<0.1%)** | Removes tiny fraction (<0.1%) of records missing primary diagnosis or valid gender without distorting feature distributions. | Step 3f verification (<0.1% dropped). |
| **Step 4: Data Type Fixing** | 1) Leave as-is. <br>2) Explicitly cast ID to string, counts to int. | **Explicitly cast** | Prevents numeric algorithms from treating administrative ID categories as continuous quantities and removes float upcasting. | Step 4 `df.dtypes` inspection. |
| **Step 5: Invalid Value Handling** | 1) Keep all values. <br>2) Filter invalid gender & clip negative counts to 0. | **Filter & clip** | Eliminates impossible clinical values (negative visit counts) and non-binary gender anomalies. | Step 5 consistency checks. |
| **Step 6: Redundant Columns & Text Standardization** | 1) Keep `_desc` columns and mixed case. <br>2) Drop `_desc` columns and standardize strings. | **Drop `_desc` & standardize** | Eliminates duplicate feature representations (`_id` and `_desc`) and prevents category fragmentation (`ch` vs `Ch`). | Step 6 transformation cell. |
| **Step 7: Irrelevant Columns & Zero-Variance Drugs** | 1) Impute `weight` / Keep all meds. <br>2) Drop `weight`, `encounter_id`, constants, and near-zero variance meds. | **Drop `weight`, identifiers, and near-zero variance meds** | *Rejected 1:* Imputing 97% missing `weight` creates pure bias; keeping >99% "No" meds adds useless sparsity. <br>*Chose 2:* Clean, robust feature space. | Step 7 dynamic threshold check (>99% "No" dropped). |
| **Step 8: Outlier Investigation** | 1) Drop IQR outliers. <br>2) Retain & log1p transform. | **Retain & log1p transform** | *Rejected 1:* Patients with 20+ visits are the exact high-utilizers most at risk; deleting them removes critical predictive signal. <br>*Chose 2:* Compresses extreme tails without discarding high-risk patients. | Step 8 IQR Outlier Summary table. |
| **Step 9: Dataset Splitting & Stratification** | 1) Standard random split. <br>2) Grouped split by `patient_nbr` with binary stratification. | **Grouped split by `patient_nbr` with binary stratification** | *Rejected 1:* Fatal data leakage (memorization across visits). <br>*Chose 2:* Isolates patients strictly to train or test while maintaining equal binary label proportions (80/20). | Step 9 patient overlap check (0 overlap verified). |
| **FE Step 1: Feature Creation** | 1) Rely only on individual prior visit counts. <br>2) Create `total_visits`, `num_drug_changes`, `num_active_drugs`. | **Create aggregate clinical features** | Captures overall healthcare utilization intensity and polypharmacy care complexity. | FE Step 1 summary statistics. |
| **FE Step 2: Feature Transformation** | 1) Apply raw scaling. <br>2) `log1p` transformation inside ColumnTransformer. | **`log1p` transformation** | Safely stabilizes variance and handles zero counts (`log(1+0) = 0`) without test set leakage. | FE Step 2 distribution comparison plot. |
| **FE Step 3: Diagnosis Extraction** | 1) OHE all 700+ raw codes. <br>2) Group into 9 baseline ICD-9 categories. | **Group into 9 baseline categories** | Avoids high sparsity and curses of dimensionality while retaining major disease classifications (Diabetes, Circulatory, etc.). | FE Step 3 category distribution. |
| **FE Step 4: Feature Combination (Ratios)** | 1) Raw counts only. <br>2) Rate features (`meds_per_day`, `lab_tests_per_day`). | **Create intensity rate features** | Disentangles length of stay from treatment intensity. | FE Step 4 ratio distributions. |
| **FE Step 5: Feature Selection** | 1) Keep all summarized columns. <br>2) Drop summarized drug columns and `patient_nbr`. | **Drop summarized drugs and `patient_nbr`** | Retains only `insulin` as an individual drug (strongest predictor) and prevents identifier leakage. | FE Step 5 shape verification. |
| **FE Step 6: Representation Refinement** | 1) Raw high-cardinality nominals. <br>2) Top-10 grouping for specialties/payers, clinical admin groupings, and ordinal age mapping. | **Top-10 grouping + Clinical mapping + Ordinal Age** | Avoids overfitting on rare physicians/payers with 1 encounter; maps age intervals to ordered integers [0–9]. | FE Step 6 category inspection. |
| **Step 10: ColumnTransformer Assembly & Target Encoding** | 1) Multi-class mapping (0, 1, 2). <br>2) **Binary Mapping (`<30`=1, `NO`/`>30`=0)** + Pipeline fit on train only. | **Binary Mapping + Scikit-Learn `ColumnTransformer`** | Fits strictly on training set and transforms test set, guaranteeing zero leakage during inference. Numerical → `log1p` + `RobustScaler`; Ordinal → `OrdinalEncoder`; Nominal → `OneHotEncoder`. | Step 10 pipeline output shapes (78,354 train rows, 19,589 test rows). |
| **FE Step 7: Feature Validation** | 1) Assume pipeline works. <br>2) Comprehensive automated validation report. | **Explicit validation report** | Confirms 0 NaNs, 0 leakage columns, strictly aligned train/test feature columns, and valid post-scaling value ranges. | FE Step 7 validation output (Passed all checks). |
| **Optional Step 6A: Resampling for Imbalance** | 1) Downsample majority. <br>2) Loss weighting (`class_weight='balanced'`). <br>3) **Optional Training-Only Resampling (SMOTE / RandomOverSampler / RUS)**. | **Algorithmic Weighting + Optional Training-Only Resampling** | Baseline models use `class_weight='balanced'`. Cell 94/95 provide commented resampling options with strict training-only isolation to guarantee zero test leakage. | Cell 94 class count table (Train: 69,271 Neg / 9,083 Pos). |
| **Step 6 (Save): Final Artifacts Export** | 1) In-memory only. <br>2) Export CSVs and pipeline pickle to `dataset_processed/`. | **Export CSVs & `preprocessing_pipeline.pkl`** | Guarantees self-contained, reproducible handoff for modeling scripts and real-time inference. | Step 6 export confirmation in `binary_classification/dataset_processed/`. |


