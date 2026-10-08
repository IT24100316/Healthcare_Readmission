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
