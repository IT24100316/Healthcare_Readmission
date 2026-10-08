import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('dataset_processed', exist_ok=True)
os.makedirs('plots/Preprocessing', exist_ok=True)

# Load data
df = pd.read_csv('dataset_extracted/diabetic_data.csv', keep_default_na=False, na_values=['?', 'Unknown/Invalid', 'Not Available', 'NULL', 'Not Mapped'], low_memory=False)

print(f"Original dataset shape: {df.shape}")

# --- Phase 1: Clean rows and columns ---
# 1. Drop Hospice/Expired
# Save 'Before' distribution
dist_before = df['readmitted'].value_counts(normalize=True) * 100

terminal_codes = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(terminal_codes)]
print(f"Shape after dropping terminal patients: {df.shape}")

# Save 'After' distribution and plot
dist_after = df['readmitted'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=dist_before.index, y=dist_before.values, ax=axes[0], palette='Reds', order=['NO', '>30', '<30'])
axes[0].set_title("Target Distribution (Before Terminal Drop)")
axes[0].set_ylabel("Percentage (%)")
sns.barplot(x=dist_after.index, y=dist_after.values, ax=axes[1], palette='Greens', order=['NO', '>30', '<30'])
axes[1].set_title("Target Distribution (After Terminal Drop)")
plt.savefig('plots/Preprocessing/phase1_terminal_drop.png', bbox_inches='tight')
plt.close()

# 2. Drop columns
cols_to_drop = ['weight', 'examide', 'citoglipton', 'encounter_id']
df = df.drop(columns=cols_to_drop, errors='ignore')

# 3. Drop near-zero variance medications (>99% No)
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

dropped_meds = []
for m in med_cols:
    if m in df.columns:
        pct_no = (df[m] == 'No').mean() * 100
        if pct_no > 99:
            dropped_meds.append(m)

df = df.drop(columns=dropped_meds, errors='ignore')
print(f"Dropped {len(dropped_meds)} near-zero variance medications: {dropped_meds}")

# --- Phase 2: Split the data ---
# We use Option A: Keep all visits, but split by patient ID.
# To stratify properly, we use the patient's first encounter's readmission status.
first_encounters = df.sort_values('patient_nbr').groupby('patient_nbr').first()
patient_targets = first_encounters['readmitted']

train_patients, test_patients = train_test_split(
    patient_targets.index, 
    test_size=0.20, 
    stratify=patient_targets.values,
    random_state=42
)

# Assign rows to train/test based on patient_nbr
train_df = df[df['patient_nbr'].isin(train_patients)]
test_df = df[df['patient_nbr'].isin(test_patients)]

print("\n--- Split Results (Option A: Keep all visits, Split by Patient) ---")
print(f"Train set: {train_df.shape[0]} encounters ({len(train_patients)} unique patients)")
print(f"Test set:  {test_df.shape[0]} encounters ({len(test_patients)} unique patients)")
print("\nTrain Class Distribution:")
print(train_df['readmitted'].value_counts(normalize=True) * 100)
print("\nTest Class Distribution:")
print(test_df['readmitted'].value_counts(normalize=True) * 100)

train_df.to_csv('dataset_processed/train_phase1_2.csv', index=False)
test_df.to_csv('dataset_processed/test_phase1_2.csv', index=False)

# Visualizing the Split to prove stratification
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_dist = train_df['readmitted'].value_counts(normalize=True) * 100
test_dist = test_df['readmitted'].value_counts(normalize=True) * 100

sns.barplot(x=train_dist.index, y=train_dist.values, ax=axes[0], palette='Blues', order=['NO', '>30', '<30'])
axes[0].set_title("Train Set Class Distribution")
axes[0].set_ylabel("Percentage (%)")
sns.barplot(x=test_dist.index, y=test_dist.values, ax=axes[1], palette='Oranges', order=['NO', '>30', '<30'])
axes[1].set_title("Test Set Class Distribution")
plt.savefig('plots/Preprocessing/phase2_split_stratification.png', bbox_inches='tight')
plt.close()

print("\nPhase 1 and 2 completed. Intermediate files saved to dataset_processed/")

# --- Phase 3: Missing Value Imputation ---
print("\n--- Starting Phase 3 ---")
# 1. Map ID hidden nulls to NaN
null_ids = {
    'admission_type_id': [5, 6, 8],
    'discharge_disposition_id': [18, 25, 26],
    'admission_source_id': [9, 15, 17, 20, 21]
}
for col, ids in null_ids.items():
    train_df[col] = train_df[col].replace(ids, np.nan)
    test_df[col] = test_df[col].replace(ids, np.nan)

# 2. Impute informative missing values
informative_cols = ['medical_specialty', 'payer_code', 'race', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
for col in informative_cols:
    train_df[col] = train_df[col].fillna('Unknown')
    test_df[col] = test_df[col].fillna('Unknown')

# 3. Handle 'None' in lab test results
test_cols = ['max_glu_serum', 'A1Cresult']
for col in test_cols:
    train_df[col] = train_df[col].replace('None', 'Not tested').fillna('Not tested')
    test_df[col] = test_df[col].replace('None', 'Not tested').fillna('Not tested')

# 4. Drop tiny fraction of remaining NaNs
train_df = train_df.dropna(subset=['gender', 'diag_1', 'diag_2', 'diag_3'])
test_df = test_df.dropna(subset=['gender', 'diag_1', 'diag_2', 'diag_3'])

# --- Phase 4: Feature Engineering ---
print("\n--- Starting Phase 4 ---")
import re
def map_diagnosis(code):
    if code == '?' or pd.isna(code): return 'Other'
    if str(code).startswith('V') or str(code).startswith('E'): return 'Other'
    try:
        c = float(code)
        if c == 250: return 'Diabetes'
        if 390 <= c <= 459 or c == 785: return 'Circulatory'
        if 460 <= c <= 519 or c == 786: return 'Respiratory'
        if 520 <= c <= 579 or c == 787: return 'Digestive'
        if 800 <= c <= 999: return 'Injury'
        if 710 <= c <= 739: return 'Musculoskeletal'
        if 580 <= c <= 629 or c == 788: return 'Genitourinary'
        if 140 <= c <= 239: return 'Neoplasms'
        return 'Other'
    except:
        return 'Other'

def map_diagnosis_granular(code):
    if code == '?' or pd.isna(code): return 'Other'
    code_str = str(code)
    if code_str.startswith('V') or code_str.startswith('E'): return 'External/Supplemental'
    try:
        c = float(code)
        # Specific high-risk conditions
        if c == 250: return 'Diabetes'
        if 401 <= c <= 405: return 'Hypertension'
        if c == 428: return 'Heart Failure'
        if 410 <= c <= 414: return 'Ischemic Heart Disease'
        if 430 <= c <= 438: return 'Cerebrovascular Disease'
        if 490 <= c <= 496: return 'COPD/Asthma'
        if 480 <= c <= 488: return 'Pneumonia'
        if 580 <= c <= 589: return 'Kidney Disease'
        
        # Broad categories
        if 1 <= c <= 139: return 'Infectious'
        if 140 <= c <= 239: return 'Neoplasms'
        if 240 <= c <= 279: return 'Endocrine/Metabolic'
        if 280 <= c <= 289: return 'Blood'
        if 290 <= c <= 319: return 'Mental'
        if 320 <= c <= 389: return 'Nervous'
        if 390 <= c <= 459 or c == 785: return 'Circulatory_Other'
        if 460 <= c <= 519 or c == 786: return 'Respiratory_Other'
        if 520 <= c <= 579 or c == 787: return 'Digestive'
        if 580 <= c <= 629 or c == 788: return 'Genitourinary_Other'
        if 630 <= c <= 679: return 'Pregnancy'
        if 680 <= c <= 709: return 'Skin'
        if 710 <= c <= 739: return 'Musculoskeletal'
        if 740 <= c <= 759: return 'Congenital'
        if 760 <= c <= 779: return 'Perinatal'
        if 780 <= c <= 799: return 'Symptoms/Ill-defined'
        if 800 <= c <= 999: return 'Injury/Poisoning'
        return 'Other'
    except:
        return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    train_df[col] = train_df[col].apply(map_diagnosis)
    test_df[col] = test_df[col].apply(map_diagnosis)
    
    train_df[col + '_granular'] = train_df[col].apply(map_diagnosis_granular)
    test_df[col + '_granular'] = test_df[col].apply(map_diagnosis_granular)

age_map = {'[0-10)':0, '[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':4, '[50-60)':5, '[60-70)':6, '[70-80)':7, '[80-90)':8, '[90-100)':9}
train_df['age'] = train_df['age'].map(age_map)
test_df['age'] = test_df['age'].map(age_map)

top_specialties = train_df['medical_specialty'].value_counts().nlargest(10).index
train_df['medical_specialty'] = train_df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other')
test_df['medical_specialty'] = test_df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other')

top_payers = train_df['payer_code'].value_counts().nlargest(10).index
train_df['payer_code'] = train_df['payer_code'].apply(lambda x: x if x in top_payers else 'Other')
test_df['payer_code'] = test_df['payer_code'].apply(lambda x: x if x in top_payers else 'Other')

def map_discharge(id_val):
    if id_val == 'Unknown' or pd.isna(id_val): return 'Other'
    id_val = int(float(id_val))
    if id_val in [1, 8]: return 'Home'
    if id_val in [6]: return 'Home Health'
    if id_val in [2, 3, 4, 5, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30]: return 'Facility'
    return 'Other'

def map_adm_type(id_val):
    if id_val == 'Unknown' or pd.isna(id_val): return 'Other'
    id_val = int(float(id_val))
    if id_val == 1: return 'Emergency'
    if id_val == 3: return 'Elective'
    return 'Other'

def map_adm_source(id_val):
    if id_val == 'Unknown' or pd.isna(id_val): return 'Other'
    id_val = int(float(id_val))
    if id_val == 7: return 'Emergency Room'
    if id_val in [1, 2, 3]: return 'Referral'
    if id_val in [4, 5, 6, 10, 18, 19, 22, 25, 26]: return 'Transfer'
    return 'Other'

train_df['discharge_disposition_id'] = train_df['discharge_disposition_id'].apply(map_discharge)
test_df['discharge_disposition_id'] = test_df['discharge_disposition_id'].apply(map_discharge)

train_df['admission_type_id'] = train_df['admission_type_id'].apply(map_adm_type)
test_df['admission_type_id'] = test_df['admission_type_id'].apply(map_adm_type)

train_df['admission_source_id'] = train_df['admission_source_id'].apply(map_adm_source)
test_df['admission_source_id'] = test_df['admission_source_id'].apply(map_adm_source)

# (We are leaving number_outpatient, number_emergency, and number_inpatient as raw numeric counts based on experimental feedback)

train_df['total_visits'] = train_df['number_outpatient'] + train_df['number_emergency'] + train_df['number_inpatient']
test_df['total_visits'] = test_df['number_outpatient'] + test_df['number_emergency'] + test_df['number_inpatient']

train_df['meds_per_day'] = train_df['num_medications'] / np.maximum(1, train_df['time_in_hospital'])
test_df['meds_per_day'] = test_df['num_medications'] / np.maximum(1, test_df['time_in_hospital'])

train_df['lab_tests_per_day'] = train_df['num_lab_procedures'] / np.maximum(1, train_df['time_in_hospital'])
test_df['lab_tests_per_day'] = test_df['num_lab_procedures'] / np.maximum(1, test_df['time_in_hospital'])

drug_cols = [c for c in train_df.columns if c in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']]

def count_drug_changes(row):
    return sum(1 for d in drug_cols if row[d] in ['Up', 'Down'])

def count_active_drugs(row):
    return sum(1 for d in drug_cols if row[d] in ['Up', 'Down', 'Steady'])

train_df['num_drug_changes'] = train_df.apply(count_drug_changes, axis=1)
test_df['num_drug_changes'] = test_df.apply(count_drug_changes, axis=1)

train_df['num_active_drugs'] = train_df.apply(count_active_drugs, axis=1)
test_df['num_active_drugs'] = test_df.apply(count_active_drugs, axis=1)

drugs_to_drop = [d for d in drug_cols if d != 'insulin']
train_df = train_df.drop(columns=drugs_to_drop, errors='ignore')
test_df = test_df.drop(columns=drugs_to_drop, errors='ignore')

print(f"Train set shape after Phase 4: {train_df.shape}")
print(f"Test set shape after Phase 4: {test_df.shape}")

train_df.to_csv('dataset_processed/train_phase3_4.csv', index=False)
test_df.to_csv('dataset_processed/test_phase3_4.csv', index=False)
print("\nPhase 3 and 4 completed. Intermediate files saved to dataset_processed/")
