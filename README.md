# Healthcare Readmission Prediction Project 🏥

This repository contains the codebase for predicting 30-day hospital readmission risk for diabetic patients using the **Diabetes 130-US Hospitals dataset**.

---

## ✅ What We Have Done So Far

### 1. Exploratory Data Analysis (EDA)
**Status:** Completed
* **Target Imbalance:** Discovered that the `<30` days readmission class is a severe minority (~11%), meaning Accuracy cannot be used as an evaluation metric.
* **Statistical Feature Ranking:** Instead of arbitrary filters, we calculated **Cramér's V** to mathematically rank the true association of all categorical features with the target, plotting only the Top 9.
* **Leakage Detection:** Identified that 101,000 encounters actually belonged to only 71,500 unique patients.
* **Data Cleansing Logic:** Identified that patients discharged to hospice or who expired (e.g., Codes 11, 13) must be dropped, as they physically cannot be readmitted (predicting for them creates target leakage).

### 2. Preprocessing & Feature Engineering (Fernando's Handoff)
**Status:** Completed
The data has been rigorously cleaned, transformed, and split. All preprocessing logic was strictly fit on the training data only.
* **Patient-Level Split:** Data was split grouped by `patient_nbr` (not randomly) to guarantee **zero patient memorization leakage** between train and test sets.
* **Outlier Handling:** Kept extreme "high-utilizer" patients (e.g., 20+ prior visits) as they carry the highest readmission risk, but stabilized them for linear models using a `log1p` transformation.
* **Feature Creation:** Engineered new clinical features like `total_visits`, `num_drug_changes`, and grouped 700+ raw ICD-9 codes into 9 broad categories.
* **Final Matrices:** Generated perfectly clean, 0-NaN, encoded matrices using a Scikit-Learn `ColumnTransformer`.

---

## ⏭️ Next Steps: Handoff for Pasiya (Model Training)

**Pasiya**, the data is 100% ready for you to start modeling! You do not need to do any encoding, scaling, or imputation. Just load the files and build the models.

### 📁 Where to find your data:
The ready-to-train datasets are located in the `dataset_processed/` folder:
- `X_train_final.csv` & `y_train_final.csv` (Use this for training & cross-validation)
- `X_test_final.csv` & `y_test_final.csv` (Hold out for final model evaluation)
- `preprocessing_pipeline.pkl` (The Scikit-Learn pipeline to transform new raw data later, if needed)

### 🛠️ Your To-Do List for Modeling:
1. **Handle the Class Imbalance:** The `<30` days class is only ~11% of the data. You **must** address this algorithmically. 
   * *Suggestion:* Use `class_weight='balanced'` in your model constructors (e.g., `LogisticRegression(class_weight='balanced')`), or apply **SMOTE** to the training data.
2. **Algorithmic Feature Selection:** The preprocessing removed *leaky* and *redundant* features. If you want to reduce the feature count further, do it during modeling.
   * *Suggestion:* Use L1 (Lasso) penalty to shrink weak features to 0, or use a Random Forest's `.feature_importances_` to select the top features.
3. **Train Baselines & Advanced Models:** Start with a baseline (like Logistic Regression) and then move to Tree-based models (Random Forest, XGBoost, or LightGBM).
4. **Use the Right Metrics:** **Do not use Accuracy.** You must evaluate your models based on **Recall** and **Macro F1 Score** specifically focusing on the minority `<30` days class, because missing a high-risk patient is the most costly error for the hospital.