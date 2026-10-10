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
| Duplicate Removal | Skip check / Explicit drop_duplicates() | Explicit drop_duplicates() | Even though 0 duplicates exist, the call is explicitly kept for reproducibility, correctness and evaluation clarity. | EDA Section 4 confirms 0 exact duplicates. |
| Data Type Fixing | Leave as-is / Explicit cast | Explicit cast | ID code columns were stored as int64 and count columns upcasted to float64 due to NaN presence. Explicit casting prevents silent downstream errors. | `df.dtypes` inspection after cleaning. |
| _desc Column Strategy | Keep alongside _id / Drop as redundant | Drop | _desc columns are exact text equivalents of _id columns. Keeping both would cause double-representation of the same information in the feature matrix. | Gap analysis vs preprocessing checklist. |
| Feature Validation | Skip / Explicit validation cell | Explicit validation | A dedicated feature validation cell at end of pipeline explicitly confirms zero NaN, no leakage columns, correct shapes, and expected scaling range. | Best practice for evaluation-ready ML pipelines. |

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
| **Feature Association (Categorical)** | Plotting dozens of categorical features with arbitrary filters (>100 samples) hides data and clutters the analysis. | Cramér's V and Chi-Square statistical ranking. | Calculated Cramér's V for all categorical features to rank their true association with the target. Plotted only the top 9 most relevant features. Ensured `age` is plotted in natural chronological order rather than sorted by risk. |

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
| **Phase 2: Duplicate Removal** | 1) Skip duplicate check. <br>2) Explicit `drop_duplicates()`. | Explicit `drop_duplicates()` | *Rejected 1:* Assumes data is clean without verification. <br>*Chose 2:* Ensures exact duplicates don't inflate model confidence, even if currently 0. | EDA Section 4 confirms 0 duplicates. |
| **Phase 2: Data Leakage & Splitting** | 1) Standard random split. <br>2) Split by `patient_nbr` (Option A: keep all visits). <br>3) Keep only 1st visit per patient (Option B). | Option A (Keep all visits, split by patient) | *Rejected 1:* Fatal data leakage (patient memorization). <br>*Rejected 3:* Loses ~30k valuable clinical encounters. <br>*Chose 2:* Preserves maximum data volume while strictly isolating patients to prevent leakage. | Patient encounters histogram (shows max 40 visits per patient). |
| **Phase 3: Informative Missing Values** | 1) Mode imputation. <br>2) Impute as `"Unknown"`. | Impute as `"Unknown"` | *Rejected 1:* Falsely assumes missingness is random; forces patients into dominant categories. <br>*Chose 2:* Lack of a specialist or payer code is inherently a valid signal/category. | High missing % (49% and 39%) from EDA. |
| **Phase 3: Missing Test Results** | 1) Drop rows. <br>2) Impute as `"Not tested"`. | Impute as `"Not tested"` | *Rejected 1:* Drops >80% of data. <br>*Chose 2:* The absence of an A1C test strongly signals something about the hospital visit context. | Test categorical distributions showing 'None'. |
| **Phase 3: Minor Flaws** | 1) Mode Impute `diag` and `gender`. <br>2) Drop rows. | Drop rows | *Rejected 1:* A model shouldn't guess a primary diagnosis or gender. <br>*Chose 2:* Safely removes <0.1% of flawed records without losing signal. | EDA Missing % table. |
| **Phase 4: Data Type Fixing** | 1) Leave as-is. <br>2) Explicitly cast ID to str, counts to int. | Explicitly cast | *Rejected 1:* ID columns as int64 and count columns as float64 cause downstream errors. <br>*Chose 2:* Explicit casting prevents silent parsing and scaling errors. | `df.dtypes` inspection after cleaning. |
| **Phase 5: Invalid Value Handling** | 1) Keep all values. <br>2) Filter invalid gender & clip negative counts. | Filter & clip | *Rejected 1:* Impossible values (e.g., negative counts) distort model logic. <br>*Chose 2:* Ensures clinical and logical consistency of the data. | Code verification showing filtered/clipped values. |
| **Phase 6: Redundant Columns** | 1) Keep `_desc` columns. <br>2) Drop `_desc` columns. | Drop `_desc` columns | *Rejected 1:* Double-representation of the same information (`_id` and `_desc`). <br>*Chose 2:* Reduces redundancy and prevents leakage before splitting. | Preprocessing checklist & Gap analysis. |
| **Phase 6: String Standardization** | 1) Leave text as-is. <br>2) Standardize to lowercase. | Standardize to lowercase | *Rejected 1:* Mixed casing (`Ch` vs `ch`) can cause categories to split during encoding. <br>*Chose 2:* Standardized `change` and `diabetesMed` to human-readable lowercase formats. | Basic data hygiene. |
| **Phase 8: Outlier Handling** | 1) Drop IQR outliers. <br>2) Retain & log1p transform. | Retain & log1p transform | *Rejected 1:* High-utilizers are exactly the patients most at risk of readmission; dropping them removes critical signal. <br>*Chose 2:* Keeps high-risk patients while stabilizing linear models. | IQR Outlier Summary table. |
| **Phase 4 (FE): Feature Creation** | 1) Rely only on individual prior visit counts. <br>2) Create `total_visits` aggregate. | Create `total_visits` | *Rejected 1:* Misses the combined severity of a patient's historical healthcare utilization. <br>*Chose 2:* The total sum of past visits provides a very strong single indicator of a "high-utilizer" patient. | Domain knowledge on patient utilization. |
| **Phase 4 (FE): ICD-9 Diagnosis** | 1) OHE all 700+ codes. <br>2) Group into 9 baseline categories. | Group into 9 baseline categories | *Rejected 1:* Creates massive sparsity. <br>*Chose 2:* Retains baseline 9 categories for interpretability and significantly reduces dimensionality without severe information loss. | Cross-Validation experimental design. |
| **Phase 4 (FE): Count Variables** | 1) Bin into 0, 1, 2, 3+. <br>2) Preserve raw continuous counts. | Preserve raw continuous counts | *Rejected 1:* Hides massive clinical differences (e.g., 4 vs 20 prior visits) from Tree models. <br>*Chose 2:* Tree models naturally learn thresholds on raw counts without skew penalties. We apply `log1p` during Phase 5 specifically to protect linear models. | ML Best Practices for Tree Algorithms. |
| **Phase 4 (FE): Rare Categories** | 1) Keep all 70+ specialties. <br>2) Top 10 + "Other". | Top 10 + "Other" | *Rejected 1:* Rare doctors with 1 patient will heavily overfit the tree models. <br>*Chose 2:* Retains 90% of the signal while heavily reducing dimensionality. | `medical_specialty` value counts. |
| **Phase 4 (FE): Admin ID Codes** | 1) Top 3 frequencies. <br>2) Conceptual grouping (Home, Facility, Transfer, etc). | Conceptual grouping | *Rejected 1:* Purely statistical grouping loses clinical meaning. <br>*Chose 2:* Aligns with actual medical patient flow and provides actionable insights. | `IDS_mapping.csv` definitions. |
| **Phase 4 (FE): Drug Summarization** | 1) Keep all 23 drugs. <br>2) Summarize into counts (Active, Changed). | Summarize into counts | *Rejected 1:* High dimensionality and most drugs are rarely used. <br>*Chose 2:* The total drug load and stability (changes) are stronger predictors than specific rare drugs. | Clinical literature on polypharmacy. |
| **Phase 4 (FE): Feature Selection** | 1) Keep raw drug columns along with summaries. <br>2) Drop raw drug columns (except insulin). | Drop raw drug columns | *Rejected 1:* Keeping raw components alongside their sum creates severe multicollinearity and redundancy. <br>*Chose 2:* Keeps the feature space clean and focuses the model on the overall polypharmacy signal. | ML Best Practices for reducing redundancy. |
| **Phase 4 (FE): Lab Tests per Day** | 1) Raw count. <br>2) Ratio of labs per day in hospital. | Ratio of labs per day | *Rejected 1:* Raw counts are highly correlated with length of stay. <br>*Chose 2:* Captures the *intensity* of treatment independent of stay duration. | Feature engineering best practices. |
| **Phase 5 (FE): Meds & Tests Encoding** | 1) One-Hot Encoding all categoricals. <br>2) Ordinal Encoding binary/ordinal features + OHE nominals. | Ordinal Encoding (binary/ordinal) + OHE (nominals) | *Rejected 1:* Creates unnecessary high dimensionality (98 features). <br>*Chose 2:* Reduces dimensionality to 86 features without information loss. Binary features and naturally ordinal features (`max_glu_serum`, `insulin`) were explicitly mapped in `OrdinalEncoder`. (Tree-based models can handle their non-linear risk patterns). | Dimensionality constraints & model selection (Tree-based algorithms). |
| **Phase 5 (FE): Numeric Transformation** | 1) `log1p` + Standard Scaler. <br>2) `log1p` + Robust Scaler. | `log1p` + Robust Scaler | *Rejected 1:* Standard Scaler is heavily distorted by extreme outliers present in the data. <br>*Chose 2:* `RobustScaler` uses median and IQR, preserving the scale of normal data without being skewed by extreme outliers. | Outlier and Distribution EDA showing severe outliers. |
| **Phase 6 (FE): Pipeline Assembly** | 1) Manual Pandas encoding. <br>2) Scikit-learn `ColumnTransformer`. | `ColumnTransformer` | *Rejected 1:* Prone to data leakage and missing column errors during inference. <br>*Chose 2:* Ensures exact same encoding logic is strictly fit on training data and perfectly applied to testing data. | ML Best Practices for avoiding data leakage. |
| **Phase 6 (FE): Class Imbalance** | 1) Downsample majority. <br>2) Use `class_weight='balanced'`. | `class_weight='balanced'` | *Rejected 1:* Discards too much valuable data from the NO class. <br>*Chose 2:* Safely penalizes minority misclassifications algorithmically. Will explore SMOTE later in CV. | Target distribution severe imbalance (<30 is 11%). |
| **Phase 7 (FE): Feature Validation** | 1) Assume pipeline works. <br>2) Explicit validation cell. | Explicit validation cell | *Rejected 1:* Blind faith in the pipeline risks silent leakage. <br>*Chose 2:* Confirms zero NaNs, no leakage, correct shapes, and expected scaling ranges. | Best practice for evaluation-ready ML pipelines. |
