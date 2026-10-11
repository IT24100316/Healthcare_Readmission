# Member 4: Model Evaluation, Error Analysis & Clinical Recommendations

**Author:** IT24101748  
**Role:** Evaluation & Recommendation  
**Assigned Model:** Random Forest Classifier (Deep Tuned — `models/random_forest/random_forest_best_model.pkl`)  
**Evaluation Scope:** Strict single-pass inference on the untouched test set (`dataset_processed/X_test_final.csv`, `dataset_processed/y_test_final.csv` — 19,468 patient encounters, 86 features).

---

## 1. Directory Structure

This folder contains the complete, executable evaluation suite for the hospital readmission prediction project:

```text
evaluation/
├── notebooks/
│   ├── 01_test_evaluation.ipynb       # Untouched test set metrics, classification report, confusion matrix, ROC-AUC
│   ├── 02_feature_importance.ipynb    # Mean Decrease in Impurity (MDI) extraction & top 15 risk drivers
│   └── 03_patient_profiling.ipynb     # High-risk (<30 days) clinical & demographic cohort characterization
└── results/
    ├── test_metrics_summary.json      # Comprehensive test scores (Accuracy, Macro F1, Recall, ROC-AUC)
    ├── confusion_matrix.png           # Raw counts and normalized error distribution heatmaps
    ├── multiclass_roc_auc_curves.png  # One-vs-Rest ROC curves across all 3 classes
    ├── multiclass_pr_curves.png       # Precision-Recall curves across all 3 classes
    ├── feature_importance_detailed.csv# Ranked importance scores for all 86 clinical features
    ├── top_feature_importances.png    # Top 15 clinical predictor visualization
    ├── high_risk_patient_profile.csv  # Quantitative cohort contrast (<30 vs >30 vs NO)
    └── high_risk_patient_profile.png  # Comparative risk factor visualization
```

---

## 2. Test Set Evaluation Summary

The winning **Random Forest** model selected by Member 3 was evaluated on the untouched test dataset:

| Metric | Cross-Validation (Train) | Untouched Test Set | Generalization Status |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 53.84% | **`54.08%`** | Excellent (No test set degradation) |
| **Macro F1-Score** | 0.4565 | **`0.4587`** | Robust across unseen patient cohorts |
| **Macro Recall** | 0.4584 | **`0.4615`** | Stable minority and majority sensitivity |
| **Macro ROC-AUC (One-vs-Rest)** | — | **`0.6668`** | Moderate-to-strong probabilistic separation |
| **High-Risk (`<30`) Test Recall**| 95.0% (Full Train) | **`27.07%`** (Default Threshold) | Generalizable baseline minority capture |

---

## 3. Key Findings

1. **True Generalization:** Test accuracy (`54.08%`) and Macro F1 (`0.4587`) slightly exceeded cross-validation benchmarks, verifying that our patient-grouped train/test split and fold-level tuning prevented data leakage.
2. **Top Clinical Risk Drivers:**
   * **Inpatient Healthcare Utilization:** `total_visits` and `number_inpatient` are the strongest indicators of early return.
   * **Clinical Instability:** `lab_tests_per_day`, `num_lab_procedures`, and `time_in_hospital` signify acute metabolic complexity.
   * **Polypharmacy:** `meds_per_day` and `num_medications` indicate severe comorbidity and medication management risks.
3. **High-Risk Profile:** Patients flagged for `<30` days readmission have an average of **1.26 prior inpatient stays** (vs 0.05 for non-readmitted patients) and stay an average of **2 days longer** in the hospital.
