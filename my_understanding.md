# My Understanding of the Project

This document breaks down exactly what dataset we are working on, what our goal is, and the current state of our data.

## 1. What are we working on?
We are working with the **Diabetes 130-US Hospitals** dataset. It contains 10 years (1999–2008) of clinical care records from 130 US hospitals. 
The business problem we are solving is helping hospital administrators and care-planning teams predict which diabetic patients are likely to be readmitted to the hospital shortly after their discharge.

* **Total Encounters (Rows):** 101,766
* **Total Features (Columns):** 50

## 2. What is our Target?
The target variable is `readmitted` with three categories:
1. `NO` (Not readmitted)
2. `>30` (Readmitted after 30 days)
3. `<30` (Readmitted within 30 days)

We will predict exactly these three categories (Multi-class classification).

## 3. How balanced are the classes?
The dataset is imbalanced. 
* **NO:** ~54%
* **>30:** ~35%
* **<30 (Immediate Risk):** ~11%

**Implication:** The most critical class (`<30`) is the minority. We must handle this imbalance and ensure our model isn't just ignoring the rare, critical cases.

## 4. What are the columns and their meanings?
Here is a breakdown of the 50 columns in our dataset, grouped by their category:

### Identifiers
* `encounter_id`: Unique ID for the hospital visit.
* `patient_nbr`: Unique ID for the patient. *(Note: A single patient can have multiple encounters!)*

### Demographics
* `age`: Grouped into 10-year brackets (e.g., `[50-60)`).
* `gender`: Male / Female / Unknown.
* `race`: Caucasian, AfricanAmerican, Asian, etc.
* `weight`: Patient's weight. *(Note: Missing in ~97% of records)*

### Admission & Discharge Details (Codes)
* `admission_type_id`: Integer code for Emergency, Urgent, Elective, etc.
* `discharge_disposition_id`: Integer code for where they went after (Home, Hospice, SNF).
* `admission_source_id`: Integer code for where they came from (ER, Transfer).
* `payer_code`: Insurance code (e.g., Medicare, Blue Cross).
* `medical_specialty`: The specialty of the admitting doctor (e.g., Cardiology).

### Clinical & Hospital Metrics
* `time_in_hospital`: Total days between admission and discharge.
* `num_lab_procedures`: Number of lab tests performed.
* `num_procedures`: Number of procedures (non-lab) performed.
* `number_diagnoses`: Number of diagnoses entered into the system.

### Patient Medical History
* `number_outpatient`: Outpatient visits in the preceding year.
* `number_emergency`: ER visits in the preceding year.
* `number_inpatient`: Inpatient visits in the preceding year.

### Medical Test Results
* `diag_1`, `diag_2`, `diag_3`: Primary and secondary ICD-9 diagnosis codes.
* `max_glu_serum`: Glucose serum test result range.
* `A1Cresult`: A1C blood test result range.

### Medications (24 separate columns)
* Features like `metformin`, `repaglinide`, `insulin`, `glipizide`, etc. 
* Values indicate if the dosage was `"Up"`, `"Down"`, `"Steady"`, or `"No"` (not prescribed).

### Overall Medication Status
* `change`: Indicates if there was *any* change in diabetic medications ("Ch" or "No").
* `diabetesMed`: Indicates if *any* diabetic medication was prescribed ("Yes" or "No").
