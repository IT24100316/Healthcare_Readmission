# Healthcare Readmission Prediction Project 🏥 (Binary Classification Track)

This repository contains the codebase for predicting 30-day hospital readmission risk for diabetic patients using the **Diabetes 130-US Hospitals dataset**.

---

## 🎯 Target Formulation: Binary Readmission Risk
* **Positive Class (1):** Patient readmitted within 30 days (`<30`) — High-risk bounce-backs requiring urgent transitional intervention (11.5% minority, 2,257 test encounters).
* **Negative Class (0):** Patient not readmitted within 30 days (`NO` or `>30` combined) — Standard post-discharge outpatient flow (88.5% majority, 17,211 test encounters).

---

## ✅ What We Have Done So Far

### 1. Exploratory Data Analysis (EDA)
**Status:** Completed
* **Target Imbalance:** Discovered that the `<30` days readmission class is a severe minority (~11.5%), meaning raw Accuracy alone cannot be used as an evaluation metric.
* **Statistical Feature Ranking:** Instead of arbitrary filters, we calculated **Cramér's V** to mathematically rank the true association of all categorical features with the target, plotting only the Top 9.
* **Leakage Detection:** Identified that 101,000 encounters actually belonged to only 71,500 unique patients.
* **Data Cleansing Logic:** Identified that patients discharged to hospice or who expired (e.g., Codes 11, 13) must be dropped, as they physically cannot be readmitted (predicting for them creates target leakage).

### 2. Preprocessing & Feature Engineering
**Status:** Completed
The data has been rigorously cleaned, transformed, and split. All preprocessing logic was strictly fit on the training data only.
* **Patient-Level Split:** Data was split grouped by `patient_nbr` (not randomly) to guarantee **zero patient memorization leakage** between train and test sets.
* **Outlier Handling:** Kept extreme "high-utilizer" patients (e.g., 20+ prior visits) as they carry the highest readmission risk, but stabilized them for linear models using a `log1p` transformation.
* **Feature Creation:** Engineered new clinical features like `total_visits`, `num_drug_changes`, and grouped 700+ raw ICD-9 codes into 9 broad categories.
* **Final Matrices:** Generated perfectly clean, 0-NaN, encoded matrices using a Scikit-Learn `ColumnTransformer`.

---

### 3. Classification Modelling & Optimization
**Status:** Completed (All 6 Models Trained, Tuned, and Evaluated)
Each model is structured in its own workflow folder under `binary_classification/models/`, evaluated using **5-Fold Stratified Cross-Validation**, and tuned with **Optuna (Bayesian TPE)**:

| Model | Status | CV Macro F1 | CV Macro Recall | CV Accuracy | Key Highlight |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | **Tuned** | **0.5699** | **0.5878** | **78.98%** | **🥇 Champion Model!** High ensemble diversity, 96% training recall on `<30`. |
| **XGBoost** | **Tuned** | **0.5563** | **0.5820** | **76.50%** | **🥈 2nd Place!** Exact gradient boosting with L1/L2 penalties & `scale_pos_weight=7.7`. |
| **LightGBM** | **Tuned** | **0.5538** | **0.5662** | **78.63%** | **🥉 3rd Place.** Fast histogram leaf-wise boosting capturing comorbidity thresholds. |
| **CatBoost** | **Tuned** | **0.5468** | **0.5979** | **72.18%** | Symmetric oblivious trees; highest CV recall (59.79%) among boosters. |
| **MLP (Neural Net)** | **Tuned** | **0.5369** | **0.5331** | **85.35%** | Highest raw accuracy (85.35%) with fold-isolated SMOTE (30%) + early stopping. |
| **Logistic Regression** | **Tuned** | **0.5210** | **0.5917** | **66.94%** | Linear baseline with fold-isolated SMOTE (30%) + ElasticNet. |

---

### 4. Model Evaluation & Clinical Recommendations (Phase 2)
**Status:** Completed (Single-Pass Evaluation on Untouched Test Set)
The champion **Random Forest** model was evaluated on the 19,468 held-out patient encounters in `binary_classification/evaluation/`:

| Metric | Cross-Validation | Untouched Test Set (Default $\tau=0.50$) | Optimal Threshold ($\tau^*=0.46$) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 78.98% | **78.54%** | 75.82% |
| **Macro F1-Score** | 0.5699 | **0.5700** | **0.5842** |
| **Macro Recall** | 0.5878 | **0.5925** | **0.6385** |
| **ROC-AUC (High-Risk `<30`)** | — | **0.6608** | **0.6608** |
| **Average Precision (PR-AUC)** | — | **0.2030** | **0.2030** |
| **High-Risk `<30` Recall** | 33.7% | **34.32%** | **45.03%** |

* **Executable Notebooks:**
  - `binary_classification/evaluation/notebooks/01_test_evaluation.ipynb`
  - `binary_classification/evaluation/notebooks/02_feature_importance.ipynb`
  - `binary_classification/evaluation/notebooks/03_patient_profiling.ipynb`
* **Artifacts & Results:**
  - `binary_classification/evaluation/results/test_metrics_summary.json`
  - `binary_classification/evaluation/results/confusion_matrix.png`
  - `binary_classification/evaluation/results/multiclass_roc_auc_curves.png` & `roc_auc_curves.png`
  - `binary_classification/evaluation/results/multiclass_pr_curves.png` & `pr_curves.png`
  - `binary_classification/evaluation/results/top_feature_importances.png` & `feature_importance_detailed.csv`
  - `binary_classification/evaluation/results/high_risk_patient_profile.png` & `high_risk_patient_profile.csv`

---

## 🏆 Project Completion Status
Both Phase 1 (Modelling & Cross-Validation Benchmarking across 6 architectures) and Phase 2 (Holdout Test Evaluation, Threshold Optimization, Feature Importance & Clinical Patient Profiling) are **100% complete, fully executed, and verified** for the Binary Classification Track.
