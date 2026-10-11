# Member 4: Model Evaluation, Error Analysis & Clinical Recommendations (Binary Classification)

**Author:** IT24101748  
**Role:** Evaluation & Recommendation  
**Assigned Model:** Random Forest Classifier (Deep Tuned — `models/random_forest/random_forest_best_model.pkl`)  
**Evaluation Scope:** Strict single-pass inference on the untouched test set (`dataset_processed/X_test_final.csv`, `dataset_processed/y_test_final.csv` — 19,468 patient encounters, 86 features).  
**Formulation:** Class 0: Not Readmitted within 30 days (`NO` or `>30`, 17,211 encounters, 88.5%), Class 1: Early Readmission (`<30` [High-Risk], 2,257 encounters, 11.5%).

---

## 1. Directory Structure

This folder contains the complete, executable evaluation suite for the binary hospital readmission prediction project:

```text
binary_classification/evaluation/
├── notebooks/
│   ├── 01_test_evaluation.ipynb       # Untouched test set metrics, classification report, confusion matrix, ROC-AUC, threshold tuning
│   ├── 02_feature_importance.ipynb    # Mean Decrease in Impurity (MDI) extraction & top 15 risk drivers
│   └── 03_patient_profiling.ipynb     # High-risk (<30 days) clinical & demographic cohort characterization
└── results/
    ├── test_metrics_summary.json      # Comprehensive test scores (Accuracy, Macro F1, Recall, ROC-AUC, Optimal Threshold)
    ├── confusion_matrix.png           # Raw counts and normalized error distribution heatmaps
    ├── multiclass_roc_auc_curves.png  # Binary ROC curves across both classes (Class 1 AUC = 0.661)
    ├── roc_auc_curves.png             # Standalone ROC curve artifact
    ├── multiclass_pr_curves.png       # Precision-Recall curves with default and optimal threshold operating points
    ├── pr_curves.png                  # Standalone PR curve artifact
    ├── feature_importance_detailed.csv# Ranked importance scores for all 86 clinical features
    ├── top_feature_importances.png    # Top 15 clinical predictor visualization
    ├── high_risk_patient_profile.csv  # Quantitative cohort contrast (Predicted High-Risk vs Low-Risk)
    └── high_risk_patient_profile.png  # Comparative risk factor visualization
```

---

## 2. Test Set Evaluation Summary

The winning **Random Forest** model selected in Phase 1 was evaluated on the untouched test dataset:

| Metric | Cross-Validation (Train) | Untouched Test Set (Default $\tau=0.50$) | Optimal Threshold ($\tau^*=0.46$) | Generalization Status |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | 78.98% | **`78.54%`** | `75.82%` | Excellent (No test set degradation) |
| **Macro F1-Score** | 0.5699 | **`0.5700`** | `0.5842` | Robust across unseen patient cohorts |
| **Macro Recall** | 0.5878 | **`0.5925`** | `0.6385` | Stable sensitivity across both classes |
| **ROC-AUC (Class 1)** | — | **`0.6608`** | **`0.6608`** | Strong probabilistic class separation |
| **Average Precision (PR-AUC)** | — | **`0.2030`** | **`0.2030`** | Baseline prevalence = 11.5% |
| **High-Risk (`<30`) Minority Recall** | 33.7% | **`34.32%`** | **`45.03%`** | Substantial gain under threshold tuning |
| **High-Risk (`<30`) Minority F1** | 0.2647 | **`0.2655`** | **`0.2772`** | Generalizable baseline minority capture |

---

## 3. Key Findings

1. **True Generalization:** Test accuracy (`78.54%`) and Macro F1 (`0.5700`) virtually matched 5-fold cross-validation benchmarks (`78.98%` and `0.5699`), verifying that patient-grouped train/test splitting and fold-level tuning completely eliminated data leakage.
2. **Clinical Decision Threshold Tuning:**
   - Under the standard default cutoff ($\tau=0.50$), the model captures **34.32%** of early readmissions with 78.54% accuracy.
   - Adjusting to the clinically optimal threshold ($\tau^*=0.46$) elevates `<30` recall to **45.03%**, successfully catching nearly half of revolving-door patients while maintaining high overall specificity.
3. **Top Clinical Risk Drivers:**
   - **Laboratory Testing & Clinical Instability:** `lab_tests_per_day` (0.0934), `num_lab_procedures` (0.0839), and `time_in_hospital` (0.0358) reflect acute diagnostic complexity.
   - **Polypharmacy & Medication Burden:** `meds_per_day` (0.0847) and `num_medications` (0.0682) signify severe multi-morbidity and pharmacological risks.
   - **Inpatient Healthcare Utilization:** `number_inpatient` (0.0580) and `total_visits` (0.0398) are dominant indicators of revolving-door readmissions.
4. **High-Risk Profile:** Patients flagged for `<30` days readmission have an average of **1.28 prior inpatient stays** (vs 0.28 for low-risk patients) and stay longer in the hospital with higher daily medication loads.
