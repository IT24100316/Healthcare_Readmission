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

---

## 5. Data Issues Log
| Issue | Impact on the task | Action taken |
| :--- | :--- | :--- |
| Excessive missing data in `weight` | Adds noise, provides no signal if 97% missing. | Drop `weight` column completely. |
| Placeholders like '?' used instead of NaN | Pandas cannot automatically detect missing data. | Replace '?', 'Unknown/Invalid', etc., with `np.nan`. |
| Terminal discharge codes (Death/Hospice) | Patients conceptually cannot be readmitted, artificially inflates negative class. | Remove all rows with these discharge codes. |
| Same patient appears in many rows | Random train/test split will cause data leakage. | Group split by `patient_nbr` / remove subsequent visits. |

---

## 6. Preprocessing & Feature Engineering Log
| Area | Decision Made | Reason | Evidence |
| :--- | :--- | :--- | :--- |
| *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* |
