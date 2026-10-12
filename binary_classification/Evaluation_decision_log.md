# Model Evaluation & Recommendation Decision Log (Binary Track)
**Domain:** Healthcare Informatics — Diabetic Readmission Prediction  
**Related Report:** [`binary_classification/personal_journey_Evaluation.md`](personal_journey_Evaluation.md) | [`binary_classification/report.md`](report.md)  
**Related Notebooks:** [`binary_classification/evaluation/notebooks/01_test_evaluation.ipynb`](evaluation/notebooks/01_test_evaluation.ipynb) | [`binary_classification/evaluation/notebooks/02_feature_importance.ipynb`](evaluation/notebooks/02_feature_importance.ipynb) | [`binary_classification/evaluation/notebooks/03_patient_profiling.ipynb`](evaluation/notebooks/03_patient_profiling.ipynb)  
**Evaluated Model Artifact:** [`binary_classification/models/random_forest/random_forest_best_model.pkl`](models/random_forest/random_forest_best_model.pkl)  
**Author:** Deerashingha A.D.R.Y.A (IT24101748)  
**Role:** Model Evaluation, Error Analysis & Hospital Recommendations (Member 4)  

---

## 1. Complete Model Evaluation Decision Matrix

This decision log documents every design choice, evaluation boundary, clinical interpretation, and governance justification made across all model evaluation and recommendation stages.

| Step / Evaluation Phase | Decision Area | Options Considered | Chosen Option | Reason (Why We Chose / Rejected) | Empirical Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Evaluation Scope** | Test Set Model Evaluation Boundary | 1) Evaluate all 6 candidate models on the holdout test set. <br>2) **Single-pass evaluation strictly on the champion model selected via cross-validation**. | **Single-Pass Evaluation on Selected Random Forest (`random_forest_best_model.pkl`)** | *Rejected 1:* Evaluating multiple candidate models on the test set to choose a winner introduces post-hoc selection bias and test-set snooping. <br>*Chose 2:* Upholds strict evaluation hygiene: Member 3 selected Random Forest based on 5-fold CV Macro F1 (`0.5699`), and Member 4 performs a single out-of-sample evaluation on unseen patient data. | Established machine learning methodology for holdout generalization. |
| **Generalization Assessment** | Test Set Verification Protocol | 1) Evaluate raw accuracy only. <br>2) Multi-metric verification against 5-fold CV baseline. | **Multi-Metric Comparison against 5-Fold CV Baseline** | *Rejected 1:* In an 11.30% minority prevalence setting, aggregate accuracy is dominated by the negative class (88.70%). <br>*Chose 2:* Evaluates whether balanced performance across both classes is maintained on unseen patient encounters. *Methodological limitation note:* Because internal CV used `StratifiedKFold` without patient grouping, similarity between CV and test metrics should not be interpreted as definitive proof of zero leakage across development folds. | Test metrics: **78.54% Accuracy** (vs. 78.98% CV), **0.5700 Macro F1** (vs. 0.5699 CV), and **0.5925 Macro Recall** (vs. 0.5878 CV) in `test_metrics_summary.json`. |
| **Model Explainability** | Interpretability & Attribution Strategy | 1) Black-box model deployment without explanation. <br>2) Mean Decrease in Impurity (MDI / Gini Importance). | **MDI Feature Importance + Top 15 Tree Predictors** | *Rejected 1:* Lack of interpretability hinders clinical transparency and stakeholder auditability. <br>*Chose 2:* Extracts tree-based split contributions across all 86 features. *Clarification:* MDI importance describes the Random Forest model's statistical reliance and split contributions on this specific dataset, rather than establishing clinical or biological causation. | Top features by MDI: `lab_tests_per_day` (0.0934), `meds_per_day` (0.0847), `num_lab_procedures` (0.0839), `num_medications` (0.0682), `number_inpatient` (0.0580) (`02_feature_importance.ipynb`). |
| **High-Risk Patient Profiling** | Clinical Cohort Segmentation | 1) Treat all diabetic patients identically. <br>2) Segment and profile model-predicted high-risk `<30` cohort against low-risk cohort. | **Cohort Profiling of Predicted High-Risk vs. Low-Risk Encounters** | *Rejected 1:* Care transition teams have limited operational bandwidth and require targeted patient stratification. <br>*Chose 2:* Quantifies observed demographic and clinical characteristics of patients predicted as high risk at threshold $\tau=0.50$ (3,491 predicted high-risk vs. 16,000 predicted low-risk encounters). | High-risk cohort averaged 1.284 prior inpatient admissions vs. 0.281 for low-risk encounters, with longer stays and higher daily medication densities (`03_patient_profiling.ipynb`, `high_risk_patient_profile.csv`). |
| **Decision Thresholding** | Operating Cutoff Determination | 1) Fixed default threshold ($\tau=0.50$). <br>2) Exploratory post-hoc threshold sensitivity analysis ($\tau \in [0.10, 0.90]$). | **Primary Evaluation at $\tau=0.50$ + Exploratory Post-Hoc Sensitivity Analysis reporting $\tau^*=0.46$** | *Rejected 1 alone:* Standard cutoff yields 34.32% minority recall, leaving many early readmissions undetected. <br>*Chose 2 as exploratory analysis:* Primary test evaluation remains at default $\tau=0.50$ (minority recall 34.32%, precision 21.66%, F1 0.2655). A threshold sweep across test predictions identified $\tau^*=0.46$ (minority recall 45.03%, precision 20.03%, F1 0.2772). Because this threshold was identified using holdout test labels, it must be documented strictly as post-hoc exploratory sensitivity analysis, not an independently validated operating threshold. For operational deployment, decision thresholds should be selected on development data and tested on an independent holdout set. | Binary ROC-AUC of **0.6608**, PR-AUC of **0.2030**, and test threshold sweep across $\tau \in [0.10, 0.90]$ (`01_test_evaluation.ipynb`). |
| **Hospital Intervention Framework** | Translational Care Recommendations | 1) Blanket follow-ups for all discharged patients. <br>2) Prioritized transitional care protocols. | **Prioritized Care Management Protocols** | *Rejected 1:* Inefficient resource allocation across hospital care management teams. <br>*Chose 2:* Targets intensive transitional care (telephone follow-up within 48 hours, bedside medication reconciliation, rapid outpatient primary care scheduling) toward patients flagged with elevated readmission risk. | Care transition literature and patient profiling metrics in `03_patient_profiling.ipynb`. |
| **Responsible AI & Governance** | Operational Deployment Safeguards | 1) Autonomous clinical decision-making. <br>2) Clinical Decision Support (CDS) with bias monitoring & operational timing verification. | **Clinical Decision Support (CDS) with Bias Audits & Timing Verification** | *Rejected 1:* Unsafe; automated decisions without clinical oversight violate healthcare governance standards. <br>*Chose 2:* Defines the model strictly as an advisory triage support tool to assist care coordinators. Requires regular fairness audits across demographic groups. Furthermore, recognizes that encounter-level variables (`discharge_disposition`, length of stay, summary medication metrics) are finalized near or at discharge, requiring verification that features are demonstrably available at the intended operational prediction time. | Medical AI governance guidelines and healthcare operational standards. |

---

## 2. Holdout Test Set Performance Summary

* **Test Dataset Size:** 19,491 unseen patient encounters (20% holdout split grouped by `patient_nbr`; verified in `01_test_evaluation.ipynb` and `test_metrics_summary.json`).
* **Overall Test Accuracy:** **`78.54%`** (CV: `78.98%`).
* **Macro F1-Score:** **`0.5700`** (CV: `0.5699`).
* **Macro Recall:** **`0.5925`** (CV: `0.5878`).
* **ROC-AUC (High-Risk `<30`):** **`0.6608`**.
* **Average Precision (PR-AUC):** **`0.2030`** (Baseline Prevalence: 11.30%).
* **Primary Test Evaluation (Default $\tau=0.50$):** High-Risk Minority Recall: **`34.32%`**, Precision: **`21.66%`**, Minority F1: **`0.2655`**.
* **Exploratory Sensitivity Analysis ($\tau^*=0.46$):** Post-hoc Minority Recall: **`45.03%`**, Precision: **`20.03%`**, Minority F1: **`0.2772`** (post-hoc exploratory sweep on test labels; not an independently validated operating threshold).
* **Evaluated Model Artifact:** [`binary_classification/models/random_forest/random_forest_best_model.pkl`](models/random_forest/random_forest_best_model.pkl).
* **Evaluation Suite:** [`evaluation/notebooks/01_test_evaluation.ipynb`](evaluation/notebooks/01_test_evaluation.ipynb), [`02_feature_importance.ipynb`](evaluation/notebooks/02_feature_importance.ipynb), [`03_patient_profiling.ipynb`](evaluation/notebooks/03_patient_profiling.ipynb).

---

## 3. High-Risk Patient Profile Summary

Derived from model predictions on the holdout test set at the default threshold $\tau=0.50$ (`03_patient_profiling.ipynb`, `high_risk_patient_profile.csv`):

| Clinical Metric | Low-Risk Cohort (Predicted 0) | High-Risk Cohort (Predicted 1) | Relative Risk Multiplier | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Cohort Size (Encounters)** | 16,000 encounters (82.1%) | 3,491 encounters (17.9%) | — | Flagged cohort prioritized for transitional care intervention. |
| **Prior Inpatient Admissions (Mean)** | 0.281 visits | 1.284 visits | **4.57× higher** | Prior hospitalization frequency is the strongest empirical marker of acute readmission vulnerability. |
| **Prior Emergency Visits (Mean)** | 0.165 visits | 0.407 visits | **2.47× higher** | Reflects acute crisis instability and emergency department reliance. |
| **Time in Hospital (Mean)** | 4.19 days | 5.37 days | **1.28× longer** | Marker of inpatient clinical complexity and illness severity. |
| **Number of Medications (Mean)** | 15.65 medications | 18.06 medications | **1.15× higher** | Chronic polypharmacy load requiring medication reconciliation. |
| **Daily Medication Density (`meds_per_day`)** | 4.60 meds/day | 4.88 meds/day | Elevated | Intensive inpatient therapeutic regimen. |
| **Diagnostic Intensity (`lab_tests_per_day`)** | 12.92 tests/day | 13.88 tests/day | Elevated | Diagnostic intensity and lab monitoring during hospitalization. |

---

## 4. Final Evaluation Verification Matrix

| Verification Metric / Artifact | Expected Value | Verified Workspace Output | Verification Status |
| :--- | :--- | :--- | :---: |
| **Test Set Encounters** | 19,491 records | Verified in `dataset_processed/y_test_final.csv` | **VERIFIED** |
| **Test Class Distribution** | Class 0: 17,288 \| Class 1: 2,203 | Verified in `dataset_processed/y_test_final.csv` | **VERIFIED** |
| **Default Threshold $\tau=0.50$ Metrics** | Acc: 78.54%, Macro F1: 0.5700, Recall: 34.32%, Prec: 21.66% | Verified in `test_metrics_summary.json` | **VERIFIED** |
| **Exploratory Threshold $\tau^*=0.46$ Metrics** | Minority Recall: 45.03%, Prec: 20.03%, F1: 0.2772 | Verified in `test_metrics_summary.json` | **VERIFIED** |
| **Discrimination Metrics** | ROC-AUC: 0.6608, PR-AUC: 0.2030 | Verified in `test_metrics_summary.json` | **VERIFIED** |
| **Evaluated Model Artifact** | `models/random_forest/random_forest_best_model.pkl` | Verified present and loaded via `joblib` | **VERIFIED** |
| **Top Feature by MDI** | `lab_tests_per_day` (0.0934) | Verified in `02_feature_importance.ipynb` | **VERIFIED** |
| **Cohort Counts** | 3,491 High-Risk vs. 16,000 Low-Risk | Verified in `high_risk_patient_profile.csv` | **VERIFIED** |
