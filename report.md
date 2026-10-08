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

## 4. EDA Insight Log
| Chart | What you learned | What you did about it |
| :--- | :--- | :--- |
| Target Distribution Bar Chart | Class imbalance with 3 categories (<30 is ~11%). | Will use class weights or SMOTE and prioritize metrics for the minority class. |
| Encounters per Patient Histogram | Many patients visit multiple times (up to 40 times). | Confirmed necessity of splitting train/test by patient ID to prevent leakage. |
| Missing % Bar Chart | `weight` is 97% missing; `medical_specialty` ~49%. | Will drop `weight`. Will impute `medical_specialty` with "Unknown". |
| Readmission by `discharge_disposition` | Certain codes (hospice/expired) mean readmission is impossible. | Will drop rows with terminal discharge codes to prevent target leakage. |
| Correlation Heatmap | No severe multicollinearity amongst numeric features. | Will retain all numeric features for tree-based models. |
| Outlier Boxplots | Many numerical features (e.g., number_inpatient) have severe right-tail outliers. | Used the IQR method to count them dynamically. Will consider Winsorization or robust algorithms. |

---

## 5. Data Issues Log
| Issue | Impact on the task | Action taken |
| :--- | :--- | :--- |
| Excessive missing data in `weight` | Adds noise, provides no signal if 97% missing. | Drop `weight` column completely. |
| Placeholders like '?' used instead of NaN | Pandas cannot automatically detect missing data. | Replace '?', 'Unknown/Invalid', etc., with `np.nan`. |
| Terminal discharge codes (Death/Hospice) | Patients conceptually cannot be readmitted, artificially inflates negative class. | Remove all rows with these discharge codes. |
| Same patient appears in many rows | Random train/test split will cause data leakage. | Group split by `patient_nbr` / remove subsequent visits. |
| Coded IDs stored as integers | Masked hidden nulls (e.g., "NULL", "Not Mapped") from pandas missing value checkers. | Dynamically parsed `IDS_mapping.csv` to map IDs and properly flag hidden nulls as NaNs. |

---

## 6. Preprocessing & Feature Engineering Log
| Decision | Options Considered | Chosen | Reason (Why we chose/rejected) | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Terminal Patients** | 1) Keep all rows. <br>2) Drop terminal discharge codes. | Drop terminal codes | *Rejected 1:* Artificially inflates the negative class because deceased patients cannot physically return. <br>*Chose 2:* Ensures the model only learns from viable candidates. | `discharge_disposition` mapping counts from EDA. |
| **Phase 1: Useless Columns** | 1) Impute `weight` / Keep all meds. <br>2) Drop `weight` & zero-variance meds. | Drop `weight` & zero-variance meds | *Rejected 1:* Imputing a 97% missing column creates pure bias/noise. Keeping 99% "No" meds causes sparsity issues. <br>*Chose 2:* Reduces noise and dimensionality safely. | Missing % bar chart; Univariate medication counts. |
| **Phase 2: Data Leakage & Splitting** | 1) Standard random split. <br>2) Split by `patient_nbr` (Option A: keep all visits). <br>3) Keep only 1st visit per patient (Option B). | Option A (Keep all visits, split by patient) | *Rejected 1:* Fatal data leakage (patient memorization). <br>*Rejected 3:* Loses ~30k valuable clinical encounters. <br>*Chose 2:* Preserves maximum data volume while strictly isolating patients to prevent leakage. | Patient encounters histogram (shows max 40 visits per patient). |
| **Phase 3: Informative Missing Values** | 1) Mode imputation. <br>2) Impute as `"Unknown"`. | Impute as `"Unknown"` | *Rejected 1:* Falsely assumes missingness is random; forces patients into dominant categories. <br>*Chose 2:* Lack of a specialist or payer code is inherently a valid signal/category. | High missing % (49% and 39%) from EDA. |
| **Phase 3: Missing Test Results** | 1) Drop rows. <br>2) Impute as `"Not tested"`. | Impute as `"Not tested"` | *Rejected 1:* Drops >80% of data. <br>*Chose 2:* The absence of an A1C test strongly signals something about the hospital visit context. | Test categorical distributions showing 'None'. |
| **Phase 3: Minor Flaws** | 1) Mode Impute `diag` and `gender`. <br>2) Drop rows. | Drop rows | *Rejected 1:* A model shouldn't guess a primary diagnosis or gender. <br>*Chose 2:* Safely removes <0.1% of flawed records without losing signal. | EDA Missing % table. |
| **Phase 4: ICD-9 Diagnosis** | 1) OHE all 700+ codes. <br>2) Group into 9 medical categories. | Group into 9 categories | *Rejected 1:* Creates massive sparsity (curse of dimensionality) leading to overfitting. <br>*Chose 2:* Follows standard medical practice to group by broad system (Circulatory, Respiratory, etc.). | ICD-9 grouping guidelines. |
| **Phase 4: Count Variables** | 1) Leave continuous. <br>2) Bin into 0, 1, 2, 3+. | Bin into categories | *Rejected 1:* Extreme right-skew causes model instability. <br>*Chose 2:* Captures the non-linear jump between 'never admitted' and 'frequent flyer' without outlier skew. | `number_inpatient` histogram. |
| **Phase 4: Rare Categories** | 1) Keep all 70+ specialties. <br>2) Top 10 + "Other". | Top 10 + "Other" | *Rejected 1:* Rare doctors with 1 patient will heavily overfit the tree models. <br>*Chose 2:* Retains 90% of the signal while heavily reducing dimensionality. | `medical_specialty` value counts. |
