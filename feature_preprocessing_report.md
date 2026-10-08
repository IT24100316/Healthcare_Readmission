# Feature Selection and Preprocessing Report

## 1. Overview
* **Total Original Features:** 50 columns
* **Final Selected/Engineered Features:** 22 core features (prior to One-Hot Encoding expansion)

---

## 2. Features Removed (And Why)

We aggressively pruned features that lacked predictive power, contained excessive noise, or risked introducing bias.

| Feature(s) Removed | Condition / Reason for Removal |
| :--- | :--- |
| `weight` | **Missing Data:** 97% of the data was missing. Imputing this would introduce pure noise. |
| `examide`, `citoglipton` | **Zero Variance:** 100% of the patients had the exact same value. Zero predictive power. |
| `encounter_id` | **Administrative Noise:** A randomly generated primary key for the hospital visit. |
| `patient_nbr` | **Data Leakage Risk:** Used exclusively to safely split train/test sets, then dropped. The model must learn clinical patterns, not memorize specific patient IDs. |
| 22 Specific Drug Columns <br>*(e.g., `acetohexamide`, `glimepiride`)* | **Near-Zero Variance & Sparsity:** Over 99% of patients had "No" for these specific medications. Instead of keeping 22 highly sparse, noisy columns, we summarized their clinical impact into two new engineered features (`num_drug_changes` and `num_active_drugs`), and dropped the individual columns. |

---

## 3. Preprocessing Steps for Selected Features

For the features we kept, specific preprocessing decisions were made based on their clinical relevance and statistical distribution.

### A. Demographics
| Feature | Preprocessing Applied | Justification |
| :--- | :--- | :--- |
| `age` | Ordinal Mapping (e.g., `[50-60)` → `5`), scaled. | Age is strictly ordinal. Converting categorical age bins to numerical values captures the linear relationship between aging and health risk. |
| `race` | Impute missing with `"Unknown"`. One-Hot Encoded (OHE). | Missing race data is often not missing at random. Treating `"Unknown"` as its own category preserves that signal. |
| `gender` | Drop rows with "Unknown/Invalid" (only 3 rows). OHE. | Model shouldn't guess gender. Dropping 3 rows out of 100,000 avoids noise at zero cost to data volume. |

### B. Administrative & Categorical Context
| Feature | Preprocessing Applied | Justification |
| :--- | :--- | :--- |
| `admission_type_id`<br>`discharge_disposition_id`<br>`admission_source_id` | 1. Mapped hidden IDs (e.g., `11`, `18`) to NaN using `IDS_mapping.csv`.<br>2. Imputed with `"Unknown"`.<br>3. Conceptually grouped (e.g., mapped codes `1` & `8` to `"Home"`, `2-5` to `"Facility"`). OHE. | Administrative codes are heavily fragmented. Grouping them by true clinical destination (Home vs Facility) drastically reduces noise and improves model generalizability. |
| `medical_specialty`<br>`payer_code` | 1. Imputed missing with `"Unknown"`.<br>2. Kept Top 10 frequencies, grouped the rest into `"Other"`. OHE. | Retaining 70+ rare specialties (some with only 1 patient) causes severe overfitting in tree models. Top 10 retains 90% of the statistical signal safely. |

### C. Clinical Metrics (Diagnoses, Tests, Meds)
| Feature | Preprocessing Applied | Justification |
| :--- | :--- | :--- |
| `diag_1`, `diag_2`, `diag_3` | **Experimental Dual-Export:**<br>1. Grouped into baseline 9 broad medical categories.<br>2. Grouped into experimental 17 granular categories (isolating Heart Failure, COPD, Diabetes, etc.). OHE for both sets. | One-Hot Encoding 700+ columns creates massive sparsity. We kept the broad 9-group representation for baseline interpretability, but added a 17-category representation to experimentally test if isolating specific high-risk conditions improves predictive performance without overfitting. |
| `max_glu_serum`, `A1Cresult` | Imputed `"None"` with `"Not tested"`. OHE. | The *decision* not to order an A1C test provides strong clinical context about the visit (e.g., it wasn't a diabetes-focused emergency). |
| `insulin` | Kept raw values (`Up`, `Down`, `Steady`, `No`). OHE. | As the primary treatment for diabetes, insulin changes possess enough independent variance and importance to remain its own feature. |

### D. Numerical Counts & Skewed Data
| Feature | Preprocessing Applied | Justification |
| :--- | :--- | :--- |
| `number_outpatient`<br>`number_emergency`<br>`number_inpatient` | Preserved as raw numerical counts. | Tree algorithms (XGBoost, Random Forest) natively learn optimal thresholds for these variables and are immune to skew. Binning them into "3+" hides massive clinical differences (e.g. 4 vs 20 prior visits) from the trees. |
| `time_in_hospital`<br>`num_lab_procedures`<br>`num_procedures`<br>`num_medications`<br>`number_diagnoses` | Applied `log1p` transformation, followed by `StandardScaler`. | These clinical counts naturally exhibit right-skew. The `log1p` transformation normalizes the distribution, safely neutralizing the leverage of severe outliers without deleting the patients. |

### E. Engineered Features
| Feature | Preprocessing Applied | Justification |
| :--- | :--- | :--- |
| `total_visits` | Sum of all prior visits. `log1p` + Scaled. | Captures the holistic historical health burden of the patient. |
| `meds_per_day` | `num_medications / time_in_hospital`. `log1p` + Scaled. | A proxy for the intensity of daily pharmaceutical intervention. |
| `lab_tests_per_day` | `num_lab_procedures / time_in_hospital`. `log1p` + Scaled. | A proxy for diagnostic uncertainty or critical care intensity during the stay. |
| `num_drug_changes` | Count of drugs shifted `Up`/`Down`. `log1p` + Scaled. | Captures medication instability. Frequent drug adjustments heavily correlate with volatile health states. |
| `num_active_drugs` | Count of drugs `Up`/`Down`/`Steady`. `log1p` + Scaled. | Captures "polypharmacy" risk. Patients actively managing multiple drugs have much higher baseline readmission risks. |

### F. Target Variable
| Feature | Preprocessing Applied | Justification |
| :--- | :--- | :--- |
| `readmitted` | Encoded: `NO` = 0, `>30` = 1, `<30` = 2. | Formats the multi-class target for scikit-learn classifiers, prioritizing the minority `<30` class as the highest integer. |
