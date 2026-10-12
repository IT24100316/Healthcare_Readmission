# Exploratory Data Analysis (EDA) Report
**Domain:** Healthcare Informatics  
**Dataset:** Diabetes 130-US Hospitals (1999–2008)  
**Task:** 30-Day Multi-Class Hospital Readmission Prediction (`<30`, `>30`, `NO`)  
**Notebook Reference:** [`eda/eda_notebook.ipynb`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/eda/eda_notebook.ipynb)  
**Decision Log Reference:** [`eda/decision_log.md`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/eda/decision_log.md)

---

## 1. Problem Framing & Clinical Objectives

* **Stakeholder:** Hospital Care-Planning Team, Discharge Coordinators, and Clinical Administrators.
* **Decision Need:** Proactively identify high-risk diabetic inpatients requiring immediate post-discharge intervention (e.g., nurse follow-up calls within 48h, home health aids, medication reconciliation) to prevent costly and hazardous 30-day readmissions.
* **Primary Lens:** 30-day hospital readmission risk prediction (`<30` class).
* **Secondary Lens:** High-risk patient profiling (identifying high-utilization archetypes, polypharmacy patterns, and clinical risk drivers).
* **Unit of Analysis:** Single hospital encounter (visit) by a diabetic inpatient.
* **Target Classes:**
  1. `<30`: Patient readmitted within 30 days (High Risk, ~11.2% - primary target of clinical interest).
  2. `>30`: Patient readmitted after 30 days (Medium Risk, ~34.9%).
  3. `NO`: No recorded readmission (Low Risk, ~53.9%).

---

## 2. EDA Insight Log

The table below summarizes key statistical patterns observed across all 50 features and how they directly governed downstream preprocessing and modeling.

| Analysis Area | Observation / Insight | Evidence (Real Data) | Action Taken for Modeling |
| :--- | :--- | :--- | :--- |
| **Target Variable** | The 3 classes are heavily imbalanced: `NO` (~54%), `>30` (~35%), `<30` (~11%). | Target distribution plot showing `<30` as a severe minority class. | Enforce class-weighted losses (`class_weight='balanced'`) or SMOTE within CV; evaluate via Macro Recall and Macro F1 rather than raw accuracy. |
| **Distributions (Numerical)** | Utilization counts (e.g., `number_inpatient`, `number_emergency`) are heavily right-skewed with long tails of high-utilizers. | Univariate numerical histograms and boxplots. | Apply `log1p` transformation to skewed counts to stabilize linear/distance models while preserving the extreme high-utilizer signal for tree models. |
| **Distributions (Categorical)** | Over 20 specific medications (e.g., `chlorpropamide`, `acetohexamide`) show >99% "No" values. `diag_1`, `diag_2`, `diag_3` contain >700 unique raw ICD-9 codes. | Univariate categorical frequency counts. | Drop near-zero variance medications to eliminate noise and sparsity. Group ICD-9 codes into broader clinical categories (9 baseline diagnostic groups). |
| **Feature Relationships** | Highest pairwise correlation: `num_medications` & `time_in_hospital` ($r = 0.47$). No severe collinearity ($r > 0.8$) between raw predictors. | Pearson correlation heatmap. | Retain all numerical features as distinct clinical signals without risk of catastrophic multicollinearity. |
| **Target Relationships** | Glycemic tests (`A1Cresult`, `max_glu_serum`) and medication adjustments do not correlate monotonically with readmission risk. `number_inpatient` is the single strongest predictor of early readmission. | Bivariate feature vs. readmission proportion plots. | Use One-Hot Encoding rather than Ordinal Encoding for nominal and non-monotonic categories to prevent imposing artificial linear relationships. |
| **Feature Association (Categorical)** | Plotting 30+ categorical features with arbitrary frequency thresholds hides rare categories and obscures statistical significance. | Cramér's V and Chi-Square statistical ranking. | Calculated Cramér's V across all categorical variables to mathematically rank true association with readmission. Visualized top 9 predictive features; plotted `age` chronologically. |

---

## 3. Data Quality & Data Issues Log

During deep exploratory auditing of raw values, identifiers, and data dictionaries, critical structural risks and hidden nulls were identified.

| Data Issue | Observation / Evidence | Impact on the Task | Action Taken |
| :--- | :--- | :--- | :--- |
| **Excessive Missingness in `weight`** | `weight` is missing in ~97% of all patient records. | Imputation would introduce substantial artificial bias and synthetic noise into models. | Drop the `weight` column entirely before modeling. |
| **Masked Missing Values (Placeholders)** | Missingness is masked by string tokens: `'?'`, `'Unknown/Invalid'`, `'None'`. | Default parser reads these as valid string categories; standard `.isnull()` checks miss them entirely. | Explicitly replace textual placeholders with `np.nan` during data ingestion. |
| **Hidden Nulls in Admin Codes** | `IDS_mapping.csv` reveals integer IDs (e.g., admission type 5, 6, 8) represent "NULL" or "Not Mapped". | Algorithms treat them as legitimate administrative categories instead of missing information. | Cross-reference mappings and translate conceptually missing IDs to `np.nan` or `"Unknown"`. |
| **Logical Target Leakage (Terminal Patients)** | Thousands of patients were discharged to Hospice or Expired (e.g., disposition IDs 11, 13, 14, 19, 20, 21). | Expired patients cannot physically be readmitted. Leaving them artificially inflates the negative (`NO`) class performance. | Drop all encounters with terminal discharge disposition codes to maintain empirical hygiene. |
| **Patient Memorization Leakage** | While there are 0 exact duplicate rows, ~71k unique patients account for ~101k visits (up to 40 encounters per patient). | A naive random train/test split places the same patient across both splits, causing models to memorize individual patients rather than generalizing clinical risk. | Perform train/test splitting strictly grouped by `patient_nbr` (Option A: keep all visits, split by patient). |

---

## 4. Key Exploratory Findings & Statistical Ranking

### 4.1. Categorical Feature Association (Cramér's V)
To avoid manual visual screening of 50 columns, Cramér's V association was computed against `readmitted`:
1. `discharge_disposition_id` ($V \approx 0.13$) - Strongest administrative predictor; patients discharged to skilled nursing or rehab have significantly higher early return rates.
2. `admission_source_id` ($V \approx 0.10$) - Emergency room admissions show markedly higher readmission rates than scheduled elective admissions.
3. `number_inpatient` (binned) ($V \approx 0.14$) - Strongest overall behavioral predictor. Patients with $\ge 2$ prior inpatient admissions have over a 25% early readmission rate (vs ~8% for zero prior visits).
4. `medical_specialty` ($V \approx 0.08$) - Cardiology, Internal Medicine, and Family Practice represent the primary clinical cohorts. Rare specialties with $<100$ records introduce overfitting risk and require grouping into an `"Other"` category.

### 4.2. Non-Monotonicity in Diabetes Tests & Medications
A critical finding during targeted EDA was that higher A1C results (`>8`) did not correspond to a strictly linear increase in early readmission compared to `Norm` or `None`. In fact, patients whose tests were **not performed** (`None`) had distinct baseline profiles compared to patients who were actively tested and had medications adjusted (`Ch` = Change). This demonstrated that Ordinal Encoding would introduce false gradient penalties, validating **One-Hot Encoding** for these variables.

---

## 5. Handoff Recommendations for Preprocessing

1. **Row Filtering:** Drop 2,300+ terminal encounters immediately.
2. **Column Pruning:** Drop `encounter_id`, `weight`, zero-variance medications (`examide`, `citoglipton`), and `_desc` redundant columns.
3. **Splitting Hygiene:** Group-stratify strictly by `patient_nbr` prior to any feature transformations.
4. **Encoding Pipeline:** Fit all scalers and encoders (`ColumnTransformer`) exclusively on training folds.
