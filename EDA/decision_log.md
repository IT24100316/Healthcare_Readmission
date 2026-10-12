# Exploratory Data Analysis (EDA) Decision Log
**Domain:** Healthcare Readmission Prediction  
**Related Report:** [`eda/report.md`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/eda/report.md)  
**Related Notebook:** [`eda/eda_notebook.ipynb`](file:///d:/y3s1/ml%20project%20final/Healthcare_Readmission/eda/eda_notebook.ipynb)  

---

## 1. High-Level EDA Decisions

| Decision | Options Considered | Chosen | Reason | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Target Formulation** | Binary (`<30` [1] vs `NO`/`>30` [0]) vs 3-Class (`NO`, `>30`, `<30`) | Binary Formulation (`<30` = 1, `NO` + `>30` = 0) | Aligns directly with CMS Hospital Readmissions Reduction Program (HRRP) 30-day acute intervention threshold; eliminates non-actionable ambiguity between late return (>30) and non-return (NO) to focus model capacity strictly on acute post-discharge deterioration. | Positive class rate = ~11.5% (acute 30-day readmissions), representing the critical high-risk cohort for proactive clinical intervention. |
| **Exploratory Feature Selection** | Manual plotting of 50 features vs Statistical Ranking | Statistical Ranking (Cramér's V + Chi-Square) | Plotting 30+ categorical charts blindly hides rare groups and creates visual noise. Ranking highlights the top 9 true predictors. | Statistical tests showing top association for disposition, source, and inpatient history. |
| **Terminal Encounter Strategy** | Keep all encounters vs Identify and flag terminal codes | Flag & recommend dropping terminal discharge codes | Expired or hospice patients physically cannot be readmitted; retaining them artificially inflates negative class accuracy. | Data dictionary lookup for IDs 11, 13, 14, 19, 20, 21. |
| **Patient Leakage Audit** | Standard random row analysis vs Patient encounter clustering analysis | Patient-level clustering audit | Patients with up to 40 repeat visits create severe leakage risk if split randomly. | Value count of `patient_nbr` showing ~71k unique patients across ~101k rows. |
| **Missingness Identification** | Naive `df.isnull()` vs Multi-token parsing (`?`, `Unknown`, `None`, ID mappings) | Multi-token parsing + ID mapping parsing | Naive checks miss string placeholders (`?`) and masked integer codes (5, 6, 8, etc.). | Inspection of `IDS_mapping.csv` and categorical unique values. |

---

## 2. EDA Insight Log

| Analysis Area | Observation / Insight | Evidence (Real Data) | Action Taken for Downstream Pipeline |
| :--- | :--- | :--- | :--- |
| **Target Distribution** | Severe imbalance in binary formulation: Acute `<30` (Class 1) is ~11.5%, while Non-acute (`NO` + `>30`, Class 0) is ~88.5%. | Target distribution plot. | Recommend PR-AUC, ROC-AUC, Recall, and F1-score for Class 1 as primary metrics; enforce algorithmic class balancing (`class_weight='balanced'`) rather than raw accuracy. |
| **Numerical Distributions** | Extreme right-skew in past utilization counts (`number_inpatient`, `number_emergency`). | Histograms and boxplots showing long positive tails. | Recommend `log1p` transformation and RobustScaler to dampen leverage without dropping high-utilizer records. |
| **Categorical Distributions** | 20+ medication columns show >99% "No" values. `diag` features contain >700 distinct ICD-9 codes. | Value counts per column. | Recommend dropping zero-variance drugs and aggregating ICD-9 codes into broad clinical categories. |
| **Feature Collinearity** | Maximum correlation between predictors is $0.47$ (`num_medications` and `time_in_hospital`). | Pearson correlation heatmap. | Confirmed no severe multicollinearity ($r > 0.8$); safe to retain all core numerical predictors. |
| **Bivariate Relationships** | Multi-category nominal features show complex non-linear associations with readmission risk. | Feature vs. target percentage bar plots. | Advised One-Hot Encoding for multi-category nominal features (race, specialty, admission, discharge, diagnoses); advised ordinal/threshold representations for structured lab measurements and medication adjustments. |

---

## 3. EDA Data Issues Log

| Data Issue | Observation / Evidence | Impact on the Task | Action Taken |
| :--- | :--- | :--- | :--- |
| **`weight` Column Missingness** | Missing in ~97% of records. | Imputation creates severe synthetic bias and model noise. | Mandated dropping `weight` column completely. |
| **Masked Missing Placeholders** | String tokens (`'?'`, `'Unknown/Invalid'`). | Undetected by pandas, distorting statistical distributions. | Converted all placeholder tokens to `np.nan`. |
| **Hidden Nulls in Admin Mappings** | Integer codes in `IDS_mapping.csv` representing "Not Mapped" or "NULL". | Treated as valid administrative categories rather than null values. | Cross-referenced mapping dictionaries and mapped them to `np.nan` or `"Unknown"`. |
| **Terminal Discharge Disposition** | Codes 11, 13, 14, 19, 20, 21 indicate death or hospice transfer. | False negative signal inflating accuracy on deceased patients. | Mandated filtering out terminal rows before train/test splitting. |
| **Repeat Encounter Patient Leakage** | ~71k unique patients across ~101k encounters (up to 40 encounters per patient). | Random splitting causes patient memorization and invalid test performance. | Mandated patient-level train/test split based on `patient_nbr`. |
