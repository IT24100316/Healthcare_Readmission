# Machine Learning Project Final Report
**Domain:** Healthcare
**Dataset:** Diabetes 130-US Hospitals

---

## 1. Problem Framing (5 Marks)

* **Stakeholder:** Hospital Care-Planning Team and Administrators.
* **Decision Need:** Identifying which diabetic patients require immediate extra follow-up care and resource allocation (e.g., home nursing, medication counseling) after discharge to prevent early readmission.
* **Primary Lens:** 30-day hospital readmission risk prediction.
* **Secondary Lens:** High-risk patient profiling (identifying common characteristics and risk drivers of the patients flagged as high-risk).
* **Unit of Analysis:** One single hospital encounter (visit) by a diabetic patient.
* **Exact Task/Output:** Supervised multi-class classification to predict whether a patient will be readmitted within 30 days, after 30 days, or not at all.
* **Rationale:** By predicting immediate-risk patients (<30 days) versus other cases, the hospital can efficiently allocate limited post-discharge resources to the most critical cases.

---

## 2. Decision Log
| Decision | Options Considered | Chosen | Reason | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| Target Formulation | Binary vs. 3-class (NO, <30, >30) | 3-class | We decided to stick with the original 3 categories. | Assignment specific setup. |
| Metric Selection | Accuracy / Recall / F1 | Recall + F1 | Imbalanced dataset (~11% positive). Missing a high-risk patient is more costly than a false alarm. | Target distribution chart showing severe class imbalance. |
| Splitting Strategy | Random split vs Patient-level split | Patient-level split | High number of repeated encounters per patient would cause data leakage in random split. | Distribution of Encounters per Patient chart showing max 40 encounters per patient. |

---

## 3. Workflow Diagram
*(To be inserted)*

---

## 4. Exploratory Data Analysis (EDA)

### 4.1. EDA Insight Log
| Analysis Area | Observation / Insight | Evidence (Real Data) | Action Taken for Modeling |
| :--- | :--- | :--- | :--- |
| **Target Variable** | The 3 classes are highly imbalanced: NO (~54%), >30 (~35%), <30 (~11%). | Target distribution plot showing `<30` as severe minority. | Will use class weights or SMOTE during modeling and prioritize metrics (Recall/F1) for the `<30` class. |
| **Distributions (Numerical)** | Count variables (e.g., `number_inpatient`, `number_emergency`) are heavily right-skewed. | Univariate numerical histograms and boxplots. | Will apply `log1p` transformation to heavily skewed features to stabilize linear models. |
| **Distributions (Categorical)** | Dozens of medications (e.g., `chlorpropamide`) have >99% "No" values. `diag` features have 700+ unique ICD-9 codes. | Univariate categorical frequency plots. | Drop near-zero variance medications to reduce noise. Group ICD-9 codes into broader clinical categories. |
| **Feature Relationships** | Highest correlations: `num_medications` & `time_in_hospital` (0.47). No severe multicollinearity (>0.8). | Pearson correlation heatmap. | Retain all numerical features as they provide distinct signals without risking severe multicollinearity. |
| **Target Relationships** | `A1Cresult` and medication risk profiles do not strictly increase monotonically. `number_inpatient` strongly correlates with `<30` risk. | Targeted Feature vs Readmission proportion plots. | Use One-Hot Encoding instead of Ordinal Encoding to prevent forcing false mathematical assumptions on models. |

### 4.2. EDA: Data Issues Log
| Data Issue | Observation / Evidence | Impact on the Task | Action Taken |
| :--- | :--- | :--- | :--- |
| Excessive missing data in `weight` | `weight` is missing in ~97% of records. | Adds noise and provides almost no signal; imputing would create pure bias. | Drop the `weight` column completely before modeling. |
| Masked Missing Values (Placeholders) | Missing values are coded as `'?'`, `'Unknown/Invalid'`, or `'None'`. | Pandas cannot automatically detect these as missing data, causing parsing errors and preventing standard imputation. | Replace all textual placeholders with proper `np.nan` values during initial load. |
| Hidden Nulls in Admin Codes | `IDS_mapping.csv` reveals integer IDs map to "NULL" or "Not Mapped". | These look like valid categories to the model but actually represent missing data. | Parse mapping definitions and convert conceptually missing IDs to `np.nan` or "Unknown". |
| Logical Impossibility (Terminal Codes) | Thousands of patients were discharged to Hospice or Expired (e.g., ID 11, 13, 14). | These patients physically cannot be readmitted, which artificially inflates the negative ('NO') class accuracy. | Drop all rows containing terminal discharge disposition codes to prevent target leakage. |
| Patient Memorization Leakage | No exact duplicate rows, but up to 40 encounters exist for the same patient ID (~71k patients, ~101k rows). | A standard random train/test split would place the same patient in both sets, causing severe data leakage (model memorizes the patient). | Split the data strictly grouping by `patient_nbr` to prevent patient memorization leakage. |

---

## 5. Preprocessing & Feature Engineering Log
| Decision | Options Considered | Chosen | Reason (Why we chose/rejected) | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Terminal Patients** | 1) Keep all rows. <br>2) Drop terminal discharge codes. | Drop terminal codes | *Rejected 1:* Artificially inflates the negative class because deceased patients cannot physically return. <br>*Chose 2:* Ensures the model only learns from viable candidates. | `discharge_disposition` mapping counts from EDA. |
| **Phase 1: Useless Columns** | 1) Impute `weight` / Keep all meds. <br>2) Drop `weight` & zero-variance meds. | Drop `weight` & zero-variance meds | *Rejected 1:* Imputing a 97% missing column creates pure bias/noise. Keeping 99% "No" meds causes sparsity issues. <br>*Chose 2:* Reduces noise and dimensionality safely. | Missing % bar chart; Univariate medication counts. |
| **Phase 2: Data Leakage & Splitting** | 1) Standard random split. <br>2) Split by `patient_nbr` (Option A: keep all visits). <br>3) Keep only 1st visit per patient (Option B). | Option A (Keep all visits, split by patient) | *Rejected 1:* Fatal data leakage (patient memorization). <br>*Rejected 3:* Loses ~30k valuable clinical encounters. <br>*Chose 2:* Preserves maximum data volume while strictly isolating patients to prevent leakage. | Patient encounters histogram (shows max 40 visits per patient). |
| **Phase 3: Informative Missing Values** | 1) Mode imputation. <br>2) Impute as `"Unknown"`. | Impute as `"Unknown"` | *Rejected 1:* Falsely assumes missingness is random; forces patients into dominant categories. <br>*Chose 2:* Lack of a specialist or payer code is inherently a valid signal/category. | High missing % (49% and 39%) from EDA. |
| **Phase 3: Missing Test Results** | 1) Drop rows. <br>2) Impute as `"Not tested"`. | Impute as `"Not tested"` | *Rejected 1:* Drops >80% of data. <br>*Chose 2:* The absence of an A1C test strongly signals something about the hospital visit context. | Test categorical distributions showing 'None'. |
| **Phase 3: Minor Flaws** | 1) Mode Impute `diag` and `gender`. <br>2) Drop rows. | Drop rows | *Rejected 1:* A model shouldn't guess a primary diagnosis or gender. <br>*Chose 2:* Safely removes <0.1% of flawed records without losing signal. | EDA Missing % table. |
| **Phase 4: ICD-9 Diagnosis** | 1) OHE all 700+ codes. <br>2) Group into 9 baseline categories vs. 17 granular categories. | Export both (Baseline & Granular) | *Rejected 1:* Creates massive sparsity. <br>*Chose 2:* Retains baseline 9 categories for interpretability, but additionally exports 17 granular categories experimentally to empirically test if specific high-risk conditions (e.g. Heart Failure, COPD) improve predictive performance without overfitting. | Cross-Validation experimental design. |
| **Phase 4: Count Variables** | 1) Bin into 0, 1, 2, 3+. <br>2) Preserve raw continuous counts. | Preserve raw continuous counts | *Rejected 1:* Hides massive clinical differences (e.g., 4 vs 20 prior visits) from Tree models. <br>*Chose 2:* Tree models naturally learn thresholds on raw counts without skew penalties. We apply `log1p` during Phase 5 specifically to protect linear models. | ML Best Practices for Tree Algorithms. |
| **Phase 4: Rare Categories** | 1) Keep all 70+ specialties. <br>2) Top 10 + "Other". | Top 10 + "Other" | *Rejected 1:* Rare doctors with 1 patient will heavily overfit the tree models. <br>*Chose 2:* Retains 90% of the signal while heavily reducing dimensionality. | `medical_specialty` value counts. |
| **Phase 4: Admin ID Codes** | 1) Top 3 frequencies. <br>2) Conceptual grouping (Home, Facility, Transfer, etc). | Conceptual grouping | *Rejected 1:* Purely statistical grouping loses clinical meaning. <br>*Chose 2:* Aligns with actual medical patient flow and provides actionable insights. | `IDS_mapping.csv` definitions. |
| **Phase 4: Drug Summarization** | 1) Keep all 23 drugs. <br>2) Summarize into counts (Active, Changed). | Summarize into counts | *Rejected 1:* High dimensionality and most drugs are rarely used. <br>*Chose 2:* The total drug load and stability (changes) are stronger predictors than specific rare drugs. | Clinical literature on polypharmacy. |
| **Phase 4: Lab Tests per Day** | 1) Raw count. <br>2) Ratio of labs per day in hospital. | Ratio of labs per day | *Rejected 1:* Raw counts are highly correlated with length of stay. <br>*Chose 2:* Captures the *intensity* of treatment independent of stay duration. | Feature engineering best practices. |
| **Phase 5: Meds & Tests Encoding** | 1) Ordinal Encoding (-1, 0, 1). <br>2) One-Hot Encoding. | One-Hot Encoding | *Rejected 1:* EDA proved readmission risk does *not* monotonically increase from `Norm` to `>8` or `Steady` to `Up`, meaning Ordinal Encoding would force false mathematical assumptions on linear models. <br>*Chose 2:* Allows the models to independently weight the risk of each state. | Targeted EDA Target Correlation plots. |
| **Phase 5: Numeric Transformation** | 1) Standard Scaler only. <br>2) `log1p` transformation + Standard Scaler. | `log1p` + Scaler | *Rejected 1:* Features like inpatient visits are highly right-skewed and zero-inflated. <br>*Chose 2:* `log1p` normalizes the skew safely handling zeros, improving linear model performance. | Outlier and Distribution EDA. |
| **Phase 6: Pipeline Assembly** | 1) Manual Pandas encoding. <br>2) Scikit-learn `ColumnTransformer`. | `ColumnTransformer` | *Rejected 1:* Prone to data leakage and missing column errors during inference. <br>*Chose 2:* Ensures exact same encoding logic is strictly fit on training data and perfectly applied to testing data. | ML Best Practices for avoiding data leakage. |
| **Phase 6: Class Imbalance** | 1) Downsample majority. <br>2) Use `class_weight='balanced'`. | `class_weight='balanced'` | *Rejected 1:* Discards too much valuable data from the NO class. <br>*Chose 2:* Safely penalizes minority misclassifications algorithmically. Will explore SMOTE later in CV. | Target distribution severe imbalance (<30 is 11%). |
