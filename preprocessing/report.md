# Preprocessing & Feature Engineering Report
**Domain:** Healthcare Informatics — Diabetic Readmission Prediction  
**Dataset:** Diabetes 130-US Hospitals (1999–2008)  
**Notebook References:** [`preprocessing/preprocessing_notebook_polished.ipynb`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/preprocessing/preprocessing_notebook_polished.ipynb) | [`preprocessing/preprocessing_notebook.ipynb`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/preprocessing/preprocessing_notebook.ipynb)  
**Decision Log Reference:** [`preprocessing/decision_log.md`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/preprocessing/decision_log.md)  
**Output Artifacts:** `dataset_processed/X_train_final.csv`, `X_test_final.csv`, `y_train_final.csv`, `y_test_final.csv`, `preprocessing_pipeline.pkl`

---

## 1. Executive Summary & Hygiene Strategy

The preprocessing and feature engineering pipeline translates the raw 101,766-encounter diabetic patient dataset into an analytically rigorous, leak-free feature matrix tailored for **Binary Classification**:
* **Positive Class (1):** Acute 30-day readmission (`<30`) — the high-risk target for clinical intervention and CMS HRRP penalty avoidance (~11.5% prevalence).
* **Negative Class (0):** Non-readmitted encounters (`NO`) combined with long-term readmissions (`>30`) (~88.5% prevalence).

### Core Hygiene Principles
1. **Strict Prevention of Data Leakage:**
   - **Zero Patient Cross-Contamination:** Encounters are split strictly at the `patient_nbr` level (Option A: all patient visits stay on the same side of the train/test barrier; 0 patient overlap).
   - **No Snooping in Transformation:** All categorical encoders, numerical scalers, and frequency aggregations are fit exclusively on the training split (`X_train`), and then applied to the test split (`X_test`).
   - **Terminal Patient Exclusion:** Deceased or hospice patients are dropped prior to feature computation to eliminate artificial negative-class inflation.
2. **Clinical Dimensionality Management:**
   - 700+ raw ICD-9 codes are condensed into 9 baseline clinical pathology clusters across primary, secondary, and tertiary diagnoses.
   - Near-zero-variance medications (>99% "No") and component drugs are pruned, and polypharmacy is captured via synthesized feature indicators (`num_active_drugs`, `num_drug_changes`).
   - Right-skewed utilization counts are scaled via outlier-resistant `RobustScaler`.

---

## 2. Step-by-Step Preprocessing Workflow

### Step 1: Ingestion & Missing Value Discovery
* Loaded raw dataset (`101,766` rows, `50` columns).
* Identified string placeholders (`'?'`, `'Unknown/Invalid'`, `'None'`, `'NULL'`, `'Not Mapped'`) which initially mask missing values from standard `.isnull()` checks.

### Step 2: Exact Duplicate Audit
* Programmatically verified that exact duplicate rows are 0 (`drop_duplicates()` explicitly retained for pipeline reproducibility).

### Step 3: Missing Value Treatment & Terminal Patient Exclusion
* **Placeholder Replacement (3a):** Replaced placeholder strings with `np.nan`.
* **Terminal Encounter Removal (3b):** Removed 2,423 encounters with discharge disposition IDs 11, 13, 14, 19, 20, 21 (Hospice / Expired). Deceased patients physically cannot experience 30-day readmission; retaining them creates false-negative bias ($101,766 \to 99,343$).
* **Administrative Code Imputation (3c & 3d):** Converted hidden missing IDs in `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` to `np.nan` and imputed high-missingness administrative/clinical features (`medical_specialty`, `payer_code`, `race`) with `"Unknown"`.
* **Laboratory Absence Imputation (3e):** Imputed `max_glu_serum` and `A1Cresult` with `"Not tested"`—untested glycemic status conveys distinct diagnostic context.
* **Critical Field Drop (3f):** Dropped 1,521 encounters (1.5%) with missing gender or diagnosis codes ($99,343 \to 97,822$).

### Step 4: Data Type Normalization
* Explicitly cast administrative ID codes (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) to string/categorical (`object`) to prevent models from treating administrative codes as numeric magnitudes.
* Verified medical count columns are `int64`.

### Step 5: Validity Checks
* Confirmed that only valid gender categories ('Male', 'Female') exist and verified that medical count features contain zero negative values.

### Step 6: Redundant, Constant, and Identifier Column Pruning
* Dropped `encounter_id` (pure identifier), `weight` (97% missing), `examide`/`citoglipton` (100% constant zero-variance), and redundant descriptive columns (`_desc`).

### Step 7: Leakage-Free Train/Test Splitting
* Grouped by `patient_nbr` using the patient's first encounter readmission status for stratification. An 80/20 split was executed:
  * **Train Set:** 78,331 patient encounters (55,108 unique patients)
  * **Holdout Test Set:** 19,491 patient encounters (13,777 unique patients)
  * **Patient Overlap:** Exactly 0.

### Step 8: Outlier Investigation & Retention Policy
* Investigated extreme right-skew across hospital stays and prior utilization (`number_inpatient`, `number_emergency`, `number_outpatient`).
* High-utilizers represent the exact clinical archetype of readmission risk; retained all records and designated `RobustScaler` to handle extreme tails gracefully.

### Step 9: Clinical Feature Engineering
1. **`total_visits`:** Sum of past outpatient, emergency, and inpatient visits (`number_outpatient + number_emergency + number_inpatient`).
2. **`meds_per_day` & `lab_tests_per_day`:** Normalizes medication and diagnostic procedure volume by `time_in_hospital`.
3. **`num_active_drugs`:** Sum of active diabetic medications prescribed to capture polypharmacy load.
4. **`num_drug_changes`:** Count of medications adjusted (`Up` or `Down`) during hospitalization to quantify therapeutic instability.
5. **ICD-9 Clinical Grouping:** Mapped primary, secondary, and tertiary diagnoses (`diag_1`, `diag_2`, `diag_3`) into 9 standardized diagnostic clusters.
6. **Administrative Code Grouping:** Consolidated discharge dispositions into (Home, Facility, Home_Health, Other) and admission sources into (Emergency_Room, Referral, Transfer, Other).
7. **Feature Selection:** Pruned 7 raw component drug columns to prevent multicollinearity with the synthesized indicators.

### Step 10: Encoding, Scaling, & Pipeline Assembly
* Assembled an end-to-end scikit-learn `ColumnTransformer` pipeline:
  * **Numerical (14 features):** Scaled via `RobustScaler`.
  * **Binary / Ordinal (6 features):** Mapped via `OrdinalEncoder` (`gender`, `change`, `diabetesMed`, `max_glu_serum`, `A1Cresult`, `insulin`).
  * **Nominal (9 features):** One-Hot Encoded (`handle_unknown='ignore'`).
* Transformer fit strictly on `X_train` and evaluated on `X_test`. Total feature count: **86 columns**.

### Step 11: Class Imbalance & Validation
* **Class Imbalance Strategy:** Applied `class_weight='balanced'` in estimators to algorithmically penalize minority misclassifications (~11.5% `<30`).
* **Validation Suite:** Confirmed 0 missing values, 0 leaked columns, correct dimensions, and valid scaling ranges across all splits.

---

## 3. Final Feature Matrix Specifications

* **Final Feature Dimensionality:** 86 features (post-transformation).
* **Training Encounters ($X_{train}$):** 78,331 records (55,108 unique patients).
* **Testing Encounters ($X_{test}$):** 19,491 records (13,777 unique patients).
* **Patient Cross-Split Leakage:** Exactly 0.
* **Missing Values in Final Output:** Exactly 0.
* **Target Classes ($y_{train}$):** Negative `0` (69,327, 88.5%) vs. Positive `1` (9,004, 11.5%).
* **Saved Outputs:** `dataset_processed/X_train_final.csv`, `y_train_final.csv`, `X_test_final.csv`, `y_test_final.csv`, `preprocessing_pipeline.pkl`.
