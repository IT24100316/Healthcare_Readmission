# Personal Journey: Evaluation & Recommendation (Binary Classification Track)
**Name:** Deerashingha A.D.R.Y.A 
**IT Number:** IT24101748  
**Role:** Model Evaluation, Error Analysis & Hospital Recommendations  

---

## 1. Inheriting the Winning Model & Test Set Protocol
Stepping into the binary classification track for the final evaluation phase, my core responsibility was to take the champion model selected during cross-validation benchmarking—the **Random Forest Classifier (Deep Tuned)** (`binary_classification/models/random_forest/random_forest_best_model.pkl`)—and conduct an out-of-sample evaluation on the **holdout test set** (`dataset_processed/X_test_final.csv`, `dataset_processed/y_test_final.csv`).

Maintaining empirical integrity was a primary priority. In team discussions, there was an initial inclination to run all six candidate models on the holdout test set to observe their respective rankings. However, using the test set to compare or select models introduces post-hoc selection bias and test data snooping. The evaluation boundary was preserved: Member 3 selected the Random Forest model based strictly on 5-fold cross-validation metrics (Mean CV Macro F1 of 0.5699, CV Accuracy of 78.98%). The test set was held out during model training, preprocessing fitting, hyperparameter tuning, and model selection. During the subsequent evaluation phase, however, test labels were accessed to conduct exploratory decision-threshold analysis. Because the threshold sweep selected 0.46 by maximizing minority-class F1 on those same test labels, the results reported at 0.46 are post-hoc and not independently validated.

---

## 2. Real-World Generalization & Confusion Matrix Analysis
The holdout test metrics were numerically similar to the reported cross-validation results (Test Accuracy: **78.54%** vs CV: **78.98%**; Test Macro F1: **0.5700** vs CV: **0.5699**; Test Macro Recall: **0.5925** vs CV: **0.5878**). However, internal cross-validation used `StratifiedKFold`, so encounters from the same patient may have appeared in different folds. Therefore, the similarity between these metrics should not be interpreted as conclusive evidence of reliable generalization:
* **Test Overall Accuracy:** **78.54%**
* **Test Macro F1-Score:** **0.5700**
* **Test Macro Recall:** **0.5925**
* **ROC-AUC (High-Risk `<30`):** **0.6608**, indicating the model's statistical ability to discriminate between positive and negative cases, though this discrimination metric alone does not establish clinical usefulness.
* **Average Precision (PR-AUC):** **0.2030**, which exceeds the baseline test-set positive prevalence of 11.30%, though exceeding this baseline does not alone establish clinical suitability or deployment readiness.

Analyzing the **Confusion Matrix** (`results/confusion_matrix.png`) highlighted key trade-offs between precision and recall:
1. **Primary Test Evaluation (Default Threshold $\tau = 0.50$):** In clinical care transitions, failing to flag an early readmission risk can result in an avoidable adverse event or readmission. At the standard classification cutoff ($\tau = 0.50$), which represents the primary out-of-sample test evaluation, the model captured **34.32%** of early readmissions (`<30`), with a minority precision of **21.66%** and a minority F1 of **0.2655**.
2. **Exploratory Post-Hoc Threshold Sensitivity Analysis ($\tau \in [0.10, 0.90]$):** To examine the operational trade-off between sensitivity and hospital intervention capacity, I conducted a threshold sweep across $\tau \in [0.10, 0.90]$ (`results/multiclass_pr_curves.png`). Searching across test predictions identified a threshold of **$\tau^* = 0.46$** that maximized minority F1 on this test set, yielding an increased minority recall of **45.03%**, a minority precision of **20.03%**, and a peak minority F1 of **0.2772**. Because this threshold was identified using the holdout test labels, it must be interpreted strictly as a post-hoc exploratory sensitivity analysis rather than an independently validated or unbiased operating threshold.

---

## 3. Clinical Feature Importance & Interpretability
In clinical applications, model interpretability is necessary for clinical transparency. To examine the underlying decision drivers of the Random Forest model, I extracted the **Mean Decrease in Impurity (MDI)** across all 86 features and identified the top features by MDI importance (`results/top_feature_importances.png`). Importantly, MDI importance scores reflect the Random Forest model's statistical reliance and split-based contribution on this specific tabular dataset, rather than proof of medical causation. These values represent predictive associations and do not prove which clinical factors are the primary causes of readmission, nor do they establish that modifying any individual clinical variable will cause readmission risk to increase or decrease:

* **Diagnostic Complexity & Inpatient Acuity:** `lab_tests_per_day` (rank 1, 0.0934) and `meds_per_day` (rank 2, 0.0847) received the highest MDI importance scores, indicating that these features contributed most strongly to the model's split-based importance measure, followed closely by `num_lab_procedures` (rank 3, 0.0839).
* **Medication Regimen Burden:** `meds_per_day` (rank 2, 0.0847) and `num_medications` (rank 4, 0.0682) were heavily relied upon in tree splits, demonstrating that multi-drug regimens statistically correlate with predicted readmission risk, without establishing direct adverse drug causality.
* **Historical Healthcare Utilization:** `number_inpatient` (rank 5, 0.0580) and `total_visits` (rank 6, 0.0398) served as influential predictive indicators of recurring hospital encounters within the model's split structure.
* **Admission Duration & Secondary Comorbidities:** `time_in_hospital` (rank 8, 0.0358) and `number_diagnoses` (rank 10, 0.0316) were notable split contributors, reflecting predictive associations between extended hospitalizations, multi-diagnosis profiles, and documented 30-day readmissions.

---

## 4. Cohort Profiling of Predicted High-Risk Patients
To translate statistical risk scores into actionable cohort descriptions, I stratified the test cohort into predicted high-risk early readmission patients and low-risk patients (`results/high_risk_patient_profile.csv`, `results/high_risk_patient_profile.png`). As implemented in `03_patient_profiling.ipynb`, this profiling segmented the cohort using the model's default predictions at threshold **$\tau = 0.50$** (3,491 predicted high-risk vs 16,000 predicted low-risk encounters):

* **Prior Inpatient Admissions:** Patients in the predicted high-risk cohort averaged **1.28 prior inpatient admissions**, compared to **0.28** for low-risk patients (a 4.5-fold difference).
* **Length of Stay:** High-risk patients exhibited longer inpatient stays (standardized mean of 0.132 vs -0.120).
* **Medication Density:** High-risk patients presented with higher medication counts per day (standardized mean of 0.182 vs -0.138).
* **Comorbidities:** High-risk patients exhibited higher average comorbidity counts and higher rates of discharge to post-acute care facilities.

---

## 5. Potential Care Management Applications & Ethical Governance
The quantitative findings suggest potential avenues for hospital care transition planning:

1. **Targeted Transitional Outreach:** Patients identified with elevated early readmission probability could be prioritized for structured follow-up calls within 48 to 72 hours of discharge, with decision thresholds adjusted to match hospital care management staffing.
2. **Medication Reconciliation Protocols:** Clinical pharmacists can focus discharge counseling on patients with high medication burdens (`num_medications`) or complex multi-drug regimens.
3. **Primary Care Coordination:** Prioritizing timely post-discharge outpatient appointments for patients with history of frequent prior admissions.
4. **Prediction Timestamp & Operational Limitations:** Before real-world deployment, the precise clinical timestamp of feature availability must be formally verified. Retrospective encounter-level variables such as `discharge_disposition`, final medication modifications, total length of stay (`time_in_hospital`), and summary metrics like `lab_tests_per_day` are finalized only at or near formal discharge. If risk scores are intended to trigger pre-discharge interventions (e.g., several days prior to discharge), models must be evaluated against features demonstrably available at that earlier operational timestamp.
5. **Responsible AI & Governance:** The predictive model is designed strictly as an advisory **Clinical Decision Support (CDS)** tool to assist care coordinators, not to dictate clinical treatment. Regular demographic audits (e.g., across age and gender groups) are necessary to ensure that algorithmic decision thresholds maintain equitable performance across patient subgroups.

---

## 6. Personal Learning and Reflection
Throughout this evaluation phase, I learned that validating machine learning models in healthcare demands far more than reporting a single aggregate test score. I came to appreciate the vital discipline of holding out unseen test data during model selection to assess true generalization without data snooping. Working through the trade-offs of the confusion matrix and threshold sweeping taught me how clinical utility shifts when balancing sensitivity against intervention capacity; it also highlighted the methodological importance of distinguishing primary uncompromised test evaluations (at $\tau = 0.50$) from post-hoc exploratory sensitivity analyses. Demystifying the model through feature importance and cohort profiling reinforced that explainability is essential for clinician trust and patient safety, demonstrating how statistical predictions must be translated into responsible, actionable clinical decision support protocols.

---

## 7. Conclusion
On the patient-level holdout test set (19,491 encounters), the Random Forest model achieved **78.54% overall accuracy** and a **Macro F1-score of 0.5700** at the default classification threshold of **$\tau = 0.50$**. The exploratory threshold analysis illustrated the operational trade-offs between minority-class recall and precision across varying decision boundaries; however, because the threshold sweep utilized the holdout test labels, the selected cutoff of **$\tau^* = 0.46$** remains a post-hoc exploration rather than an independently validated operating threshold. Furthermore, while feature importance and cohort profiling provided interpretability into the model's predictive reliance, internal cross-validation during the modeling phase utilized `StratifiedKFold` rather than `GroupKFold`, which represents a recognized methodological limitation regarding repeated encounters from the same patient across folds. Consequently, further prospective validation—incorporating decision thresholds tuned exclusively on validation data and rigorous patient-level grouped validation—is necessary before considering any clinical workflow deployment.
