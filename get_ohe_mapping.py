import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
import json

df = pd.read_csv('dataset_extracted/diabetic_data.csv', keep_default_na=False)

placeholders = ['?', 'Unknown/Invalid', 'Not Available', 'NULL', 'Not Mapped']
df.replace(placeholders, np.nan, inplace=True)

terminal_codes = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(terminal_codes)].copy()

null_id_codes = {
    'admission_type_id': [5, 6, 8],
    'discharge_disposition_id': [18, 25, 26],
    'admission_source_id': [9, 15, 17, 20, 21]
}
for col, ids in null_id_codes.items():
    df[col] = df[col].replace(ids, np.nan)

informative_cols = ['medical_specialty', 'payer_code', 'race',
                    'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
for col in informative_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

test_cols = ['max_glu_serum', 'A1Cresult']
for col in test_cols:
    df[col] = df[col].replace('None', 'Not tested').fillna('Not tested')

df.dropna(subset=['gender', 'diag_1', 'diag_2', 'diag_3'], inplace=True)

code_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
for col in code_cols:
    df[col] = df[col].astype(str)

count_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
              'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
for col in count_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

df = df[df['gender'].isin(['Male', 'Female'])].copy()

if 'change' in df.columns:
    df['change'] = df['change'].str.lower().replace({'ch': 'changed', 'no': 'unchanged'})
if 'diabetesMed' in df.columns:
    df['diabetesMed'] = df['diabetesMed'].str.lower()

cols_to_drop = ['weight', 'examide', 'citoglipton', 'encounter_id']
df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

all_med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                'metformin-rosiglitazone', 'metformin-pioglitazone']

dropped_meds = [m for m in all_med_cols if m in df.columns and (df[m] == 'No').mean() > 0.99]
df.drop(columns=dropped_meds, errors='ignore', inplace=True)
df.drop(columns=['target'], errors='ignore', inplace=True)

df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

drug_cols_present = [c for c in df.columns if c in all_med_cols]
def count_drug_changes(row): return sum(1 for d in drug_cols_present if row[d] in ['Up', 'Down'])
def count_active_drugs(row): return sum(1 for d in drug_cols_present if row[d] in ['Up', 'Down', 'Steady'])

df['num_drug_changes'] = df.apply(count_drug_changes, axis=1)
df['num_active_drugs'] = df.apply(count_active_drugs, axis=1)

def map_diagnosis(code):
    if pd.isna(code) or str(code) in ['?', 'nan']: return 'Other'
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
    except: return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].apply(map_diagnosis)

df['meds_per_day'] = df['num_medications'] / np.maximum(1, df['time_in_hospital'])
df['lab_tests_per_day'] = df['num_lab_procedures'] / np.maximum(1, df['time_in_hospital'])

drugs_to_drop = [d for d in drug_cols_present if d != 'insulin']
df.drop(columns=drugs_to_drop, errors='ignore', inplace=True)
df.drop(columns=['patient_nbr'], errors='ignore', inplace=True)

top_specialties = df['medical_specialty'].value_counts().nlargest(10).index
df['medical_specialty'] = df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other')

top_payers = df['payer_code'].value_counts().nlargest(10).index
df['payer_code'] = df['payer_code'].apply(lambda x: x if x in top_payers else 'Other')

def map_discharge(v):
    try:
        v = int(float(v))
        if v in [1, 8]: return 'Home'
        if v in [6]: return 'Home_Health'
        if v in [2,3,4,5,10,15,16,17,22,23,24,27,28,29,30]: return 'Facility'
        return 'Other'
    except: return 'Other'

def map_adm_type(v):
    try:
        v = int(float(v))
        if v == 1: return 'Emergency'
        if v == 3: return 'Elective'
        return 'Other'
    except: return 'Other'

def map_adm_source(v):
    try:
        v = int(float(v))
        if v == 7: return 'Emergency_Room'
        if v in [1,2,3]: return 'Referral'
        if v in [4,5,6,10,18,19,22,25,26]: return 'Transfer'
        return 'Other'
    except: return 'Other'

df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(map_discharge)
df['admission_type_id']        = df['admission_type_id'].apply(map_adm_type)
df['admission_source_id']      = df['admission_source_id'].apply(map_adm_source)

age_map = {'[0-10)':0,'[10-20)':1,'[20-30)':2,'[30-40)':3,'[40-50)':4,
           '[50-60)':5,'[60-70)':6,'[70-80)':7,'[80-90)':8,'[90-100)':9}
df['age'] = df['age'].map(age_map)

X = df.drop(columns=['readmitted'])
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

ordinal_cats_dict = {
    'gender': ['Female', 'Male'],
    'change': ['unchanged', 'changed'],
    'diabetesMed': ['no', 'yes'],
    'max_glu_serum': ['Not tested', 'Norm', '>200', '>300'],
    'A1Cresult': ['Not tested', 'Norm', '>7', '>8'],
    'insulin': ['No', 'Down', 'Steady', 'Up']
}

ordinal_features = [c for c in ordinal_cats_dict.keys() if c in categorical_features]
nominal_features = [c for c in categorical_features if c not in ordinal_features]

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X[nominal_features])

mapping = {
    "ordinal_encoding": {},
    "one_hot_encoding": {}
}

for c in ordinal_features:
    # Explicitly show the string -> integer mapping for clarity
    mapping["ordinal_encoding"][c] = {category: i for i, category in enumerate(ordinal_cats_dict[c])}

for i, col in enumerate(nominal_features):
    mapping["one_hot_encoding"][col] = list(encoder.categories_[i])

with open("ohe_mapping.json", "w") as f:
    json.dump(mapping, f, indent=4)
print("Done")

numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
print(f'Numerical features ({len(numerical_features)}): {numerical_features}')
print(f'Total Final Features: {len(numerical_features) + 84}')
