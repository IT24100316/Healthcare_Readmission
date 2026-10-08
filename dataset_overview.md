# Healthcare Readmission Project Overview

## Project Objective
The goal of this project is to develop a machine learning model to predict the risk of hospital readmission for diabetic patients within 30 days of discharge. This aligns with the hospital's core decision need to identify and support high-risk patients to prevent immediate readmissions.

### 1. Primary Lens — Hospital Readmission Risk Prediction
* **Task Type:** Multi-Class Classification (Supervised Learning)
* **Target Variable:** `readmitted`
  * **NO:** Not readmitted
  * **>30:** Readmitted after 30 days
  * **<30:** Readmitted within 30 days

### 2. Secondary Lens — High-Risk Patient Profiling
* Analyze feature importance from the trained classifier.
* Profile predicted high-risk patients by demographic and clinical characteristics to aid in targeted intervention programs.

---

## Dataset Overview (Diabetes 130-US Hospitals)

The dataset consists of clinical care data from 130 US hospitals for the years 1999-2008.

* **Total Records:** 101,766 patient encounters
* **Total Features:** 50 attributes

### Data Dictionary

Below is the full list of features present in `diabetic_data.csv` along with their inferred datatypes. Note that `object` typically refers to strings (categorical data), while `int64` refers to integers (numerical data). 

| Feature Name | Data Type | Description (Inferred) |
| :--- | :--- | :--- |
| `encounter_id` | `int64` | Unique identifier of an encounter |
| `patient_nbr` | `int64` | Unique identifier of a patient |
| `race` | `object` | Patient's race |
| `gender` | `object` | Patient's gender |
| `age` | `object` | Patient's age grouped in 10-year intervals |
| `weight` | `object` | Patient's weight in pounds |
| `admission_type_id` | `int64` | Integer identifier corresponding to admission type |
| `discharge_disposition_id` | `int64` | Integer identifier corresponding to discharge disposition |
| `admission_source_id` | `int64` | Integer identifier corresponding to admission source |
| `time_in_hospital` | `int64` | Integer number of days between admission and discharge |
| `payer_code` | `object` | Integer identifier corresponding to payer code |
| `medical_specialty` | `object` | Integer identifier of a specialty of the admitting physician |
| `num_lab_procedures` | `int64` | Number of lab tests performed during the encounter |
| `num_procedures` | `int64` | Number of procedures (other than lab tests) performed |
| `num_medications` | `int64` | Number of distinct generic names administered |
| `number_outpatient` | `int64` | Number of outpatient visits of the patient in the year preceding the encounter |
| `number_emergency` | `int64` | Number of emergency visits of the patient in the year preceding the encounter |
| `number_inpatient` | `int64` | Number of inpatient visits of the patient in the year preceding the encounter |
| `diag_1` | `object` | Primary diagnosis (ICD9 code) |
| `diag_2` | `object` | Secondary diagnosis (ICD9 code) |
| `diag_3` | `object` | Additional secondary diagnosis (ICD9 code) |
| `number_diagnoses` | `int64` | Number of diagnoses entered to the system |
| `max_glu_serum` | `object` | Indicates the range of the result or if the test was not taken |
| `A1Cresult` | `object` | Indicates the range of the result or if the test was not taken |
| `metformin` | `object` | Indicates if medication was prescribed or dosage changed |
| `repaglinide` | `object` | Indicates if medication was prescribed or dosage changed |
| `nateglinide` | `object` | Indicates if medication was prescribed or dosage changed |
| `chlorpropamide` | `object` | Indicates if medication was prescribed or dosage changed |
| `glimepiride` | `object` | Indicates if medication was prescribed or dosage changed |
| `acetohexamide` | `object` | Indicates if medication was prescribed or dosage changed |
| `glipizide` | `object` | Indicates if medication was prescribed or dosage changed |
| `glyburide` | `object` | Indicates if medication was prescribed or dosage changed |
| `tolbutamide` | `object` | Indicates if medication was prescribed or dosage changed |
| `pioglitazone` | `object` | Indicates if medication was prescribed or dosage changed |
| `rosiglitazone` | `object` | Indicates if medication was prescribed or dosage changed |
| `acarbose` | `object` | Indicates if medication was prescribed or dosage changed |
| `miglitol` | `object` | Indicates if medication was prescribed or dosage changed |
| `troglitazone` | `object` | Indicates if medication was prescribed or dosage changed |
| `tolazamide` | `object` | Indicates if medication was prescribed or dosage changed |
| `examide` | `object` | Indicates if medication was prescribed or dosage changed |
| `citoglipton` | `object` | Indicates if medication was prescribed or dosage changed |
| `insulin` | `object` | Indicates if medication was prescribed or dosage changed |
| `glyburide-metformin` | `object` | Indicates if medication was prescribed or dosage changed |
| `glipizide-metformin` | `object` | Indicates if medication was prescribed or dosage changed |
| `glimepiride-pioglitazone` | `object` | Indicates if medication was prescribed or dosage changed |
| `metformin-rosiglitazone` | `object` | Indicates if medication was prescribed or dosage changed |
| `metformin-pioglitazone` | `object` | Indicates if medication was prescribed or dosage changed |
| `change` | `object` | Indicates if there was a change in diabetic medications |
| `diabetesMed` | `object` | Indicates if there was any diabetic medication prescribed |
| `readmitted` | `object` | Days to inpatient readmission (Target Variable) |

### ID Mappings (`IDS_mapping.csv`)
A separate data dictionary (`IDS_mapping.csv`) provides string descriptions for the numerical IDs used in the following categorical columns:
* **`admission_type_id`** (e.g., 1 = Emergency, 2 = Urgent, 3 = Elective)
* **`discharge_disposition_id`** (e.g., 1 = Discharged to home)
* **`admission_source_id`** (e.g., 7 = Emergency Room)

---

## Project Workflow & Responsibilities
The project is divided among team members focusing on:
1. **Problem Framing & Data Understanding:** EDA, data dictionary, quality issues.
2. **Preprocessing & Feature Engineering:** Handling missing values (`?`), encoding categorical features, scaling, handling class imbalance (SMOTE).
3. **Classification Modelling:** Baseline Logistic Regression, exploring Random Forest, XGBoost, SVM, hyperparameter tuning with k-fold cross-validation.
4. **Evaluation & Recommendation:** Evaluating models (Prioritizing **Recall**, then F1-score, ROC-AUC), feature importance analysis, and developing practical clinical recommendations.
