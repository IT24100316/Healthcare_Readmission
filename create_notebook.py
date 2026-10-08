import json

def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

cells = []

# Problem Framing
cells.append(create_markdown_cell("""# Machine Learning Project: Hospital Readmission Prediction

## 1. Problem Framing
* **Stakeholder:** Hospital Care-Planning Team and Administrators.
* **Decision Need:** Identifying which diabetic patients require immediate extra follow-up care and resource allocation after discharge to prevent early readmission.
* **Primary Lens:** 30-day hospital readmission risk prediction.
* **Secondary Lens:** High-risk patient profiling (identifying common characteristics and risk drivers).
* **Unit of Analysis:** One single hospital encounter (visit) by a diabetic patient.
* **Exact Task/Output:** Supervised multi-class classification to predict whether a patient will be readmitted within 30 days, after 30 days, or not at all (3 categories: `<30`, `>30`, `NO`).
* **Rationale:** By predicting immediate-risk patients (<30 days) versus other cases, the hospital can efficiently allocate limited post-discharge resources to the most critical cases.
"""))

# Header
cells.append(create_markdown_cell("""## Exploratory Data Analysis (EDA)
This notebook performs a comprehensive EDA following the project requirements. We apply analysis across ALL features to ensure no potential signal is missed.
"""))

# Imports
cells.append(create_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import math
import os

warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_theme(style="whitegrid")

# Create directory for plots
os.makedirs('plots/EDA', exist_ok=True)
"""))

# 1. Dataset overview
cells.append(create_markdown_cell("""## 1. Dataset Overview
We begin by loading the dataset, checking its shape, data types, and identifying the key identifier columns vs features."""))

cells.append(create_code_cell("""# Load data
df = pd.read_csv('dataset_extracted/diabetic_data.csv')
ids = pd.read_csv('dataset_extracted/IDS_mapping.csv')

print(f"Dataset Shape: {df.shape}")
print("\\nMemory Usage:")
print(df.info(memory_usage='deep'))

# Separate columns by type
id_cols = ['encounter_id', 'patient_nbr']
target_col = 'readmitted'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()

# Remove IDs and Target from feature lists
categorical_cols = [c for c in categorical_cols if c not in id_cols + [target_col]]
numeric_cols = [c for c in numeric_cols if c not in id_cols + [target_col]]

print(f"\\nIdentifier Columns: {id_cols}")
print(f"Target Column: {target_col}")
print(f"Numerical Columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
"""))

cells.append(create_markdown_cell("""### What we learned
- The dataset has 101,766 rows and 50 columns.
- `encounter_id` and `patient_nbr` are identifiers, and `readmitted` is the target.
- Most features are stored as `object` (categorical/strings), and there are several numerical features.
- Missing values are coded as `?` instead of actual NaNs.

### What we will do about it
- We will convert the `?` placeholders into proper `NaN` values.
- We will dynamically loop through all identified numerical and categorical columns for comprehensive EDA rather than selectively picking a few.
"""))

# 2. Target analysis
cells.append(create_markdown_cell("""## 2. Target Analysis
The target variable is `readmitted` with 3 classes: `NO`, `>30`, and `<30`."""))

cells.append(create_code_cell("""# We will use the 3 categories as our target
df['target'] = df['readmitted']

# Class counts and percentages
counts = df['target'].value_counts()
pcts = df['target'].value_counts(normalize=True) * 100
target_stats = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts})
print(target_stats)

# Plot
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='target', order=['NO', '>30', '<30'])
plt.title('Distribution of Readmission (3 Classes)')
plt.xlabel('Readmitted Category')
plt.ylabel('Count')

# Annotate counts
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=10, xytext=(0, 5), textcoords='offset points')
plt.savefig('plots/EDA/target_distribution.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""### What we learned
- The dataset is distributed across 3 classes: NO (~54%), >30 (~35%), and <30 (~11%).
- The class <30 (early readmission) is the minority class.

### What we will do about it
- We need to handle this as a multi-class classification problem predicting exactly these 3 categories.
- We must handle class imbalance during modeling since the `<30` class is much smaller than `NO` and `>30`.
"""))

# 3. Missing values
cells.append(create_markdown_cell("""## 3. Missing Values
We replace all string placeholders with NaN and calculate missingness across ALL columns."""))

cells.append(create_code_cell("""# Replace placeholders with NaN
placeholders = ['?', 'Unknown/Invalid', 'Not Available', 'NULL', 'Not Mapped']
df.replace(placeholders, np.nan, inplace=True)

# Calculate missing percentages for all features
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing_counts, 'Missing_Pct': missing_pct})
missing_df = missing_df[missing_df['Missing_Pct'] > 0].sort_values(by='Missing_Pct', ascending=False)

print(missing_df)

# Plot
plt.figure(figsize=(10, 5))
if not missing_df.empty:
    ax = sns.barplot(x=missing_df.index, y='Missing_Pct', data=missing_df, palette='viridis')
    for c in ax.containers: ax.bar_label(c, fmt='%.1f%%', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.title('Percentage of Missing Values per Feature (Only features with missing data)')
    plt.ylabel('% Missing')
    plt.savefig('plots/EDA/missing_values.png', bbox_inches='tight')
    plt.show()
else:
    print("No missing values found after processing!")
"""))

cells.append(create_markdown_cell("""### What we learned
- `weight` is missing in ~97% of the records.
- `medical_specialty` is missing ~49%.
- `payer_code` is missing ~39%.
- `race` and a few others have <5% missing values.

### What we will do about it
- **Drop** `weight` entirely due to excessive missingness.
- For `medical_specialty` and `payer_code`, we will fill with an `"Unknown"` category because the fact that it is missing might itself carry predictive information (e.g., lack of specialist access).
- For low-missingness columns like `race`, we will impute with the mode (most frequent) or drop those specific rows.
"""))

# 4. Duplicates and Leakage
cells.append(create_markdown_cell("""## 4. Duplicates & Patient Leakage Risk
Checking for identical rows and repeated encounters for the same patient."""))

cells.append(create_code_cell("""# Exact duplicate rows
print(f"Exact duplicate rows: {df.duplicated().sum()}")

# Patient encounters distribution
unique_patients = df['patient_nbr'].nunique()
print(f"Unique patients: {unique_patients} across {len(df)} total encounters.")

encounter_counts = df['patient_nbr'].value_counts()
print(f"Max encounters for a single patient: {encounter_counts.max()}")

plt.figure(figsize=(8, 4))
sns.histplot(encounter_counts, bins=30, kde=False)
plt.title('Distribution of Encounters per Patient')
plt.xlabel('Number of Encounters')
plt.ylabel('Number of Patients')
plt.yscale('log')
plt.savefig('plots/EDA/encounters_per_patient.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""### What we learned
- There are no exact duplicate rows.
- However, there are ~71,518 unique patients for ~101,766 encounters, meaning tens of thousands of patients visit multiple times (up to 40 times!).

### What we will do about it
- We **must split the train/test sets by `patient_nbr`**, not randomly. Random splitting would cause data leakage. 
- We will likely extract only the first encounter per patient during preprocessing, or use GroupKFold.
"""))

# 5. Univariate Analysis (ALL Features)
cells.append(create_markdown_cell("""## 5. Univariate Analysis
Let's explore the distribution of **ALL** numerical and categorical variables dynamically. We will plot every feature."""))

cells.append(create_code_cell("""# Plot ALL Numerical features
num_cols_count = len(numeric_cols)
rows = math.ceil(num_cols_count / 3)
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='teal', bins=30)
    axes[i].set_title(f'Distribution: {col}')

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('plots/EDA/univariate_numerical.png', bbox_inches='tight')
plt.show()

# Stats
stats_df = df[numeric_cols].describe().T
stats_df['skewness'] = df[numeric_cols].skew()

# IQR Outlier Detection
outliers_list = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
    outliers_list.append(outliers_count)
    
stats_df['outliers (IQR)'] = outliers_list
print("\\nNumerical Statistics & Outliers:")
print(stats_df)
"""))

cells.append(create_markdown_cell("""### Outlier Visualization (Boxplots)
To visually inspect the outliers detected by the IQR method above, we plot boxplots for all numerical features."""))

cells.append(create_code_cell("""# Boxplots for numerical outliers
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=axes[i], color='coral')
    axes[i].set_title(f'Outliers: {col}')

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('plots/EDA/outliers_boxplots.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_code_cell("""# Plot ALL Categorical features
# Note: For high cardinality features like diagnoses (diag_1, diag_2, diag_3), we only plot the top 15 categories to keep it readable.
cat_cols_count = len(categorical_cols)
rows = math.ceil(cat_cols_count / 3)
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    # Get top 15 categories if there are too many unique values
    top_cats = df[col].value_counts().nlargest(15).index
    sns.countplot(data=df[df[col].isin(top_cats)], x=col, ax=axes[i], palette='Set2', order=top_cats)
    for c in axes[i].containers: axes[i].bar_label(c, fmt='%.0f', fontsize=8)
    axes[i].set_title(f'Count: {col} (Top 15)')
    axes[i].tick_params(axis='x', rotation=45)

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('plots/EDA/univariate_categorical.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""### What we learned
- Count-based numerical variables (`number_emergency`, `number_inpatient`) are heavily right-skewed with most values at 0.
- `time_in_hospital` and `num_lab_procedures` follow somewhat skewed normal distributions.
- Dozens of medication columns (like `chlorpropamide`, `tolbutamide`) are overwhelmingly heavily imbalanced, where >99% of patients have "No" (not prescribed).
- The ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) have hundreds of unique values.

### What we will do about it
- We will consider applying log transformations to highly skewed numerical features.
- We should drop medication features that have close to zero variance (e.g., 99.9% "No") as they offer no predictive power.
- Diagnosis codes must be grouped (e.g., grouping by disease category like Circulatory, Respiratory) rather than one-hot encoding hundreds of unique codes.
"""))

# 6. Feature vs Target (ALL Features)
cells.append(create_markdown_cell("""## 6. Feature vs Target (Readmission Risk)
We analyze how **ALL** features correlate with the 3 target classes using stacked bar charts."""))

cells.append(create_code_cell("""def plot_readmission_distribution(col, ax):
    # Cross tabulation of the feature vs target
    ct = pd.crosstab(df[col], df['target'], normalize='index') * 100
    
    # Only show categories with more than 100 samples to avoid extreme noise from small groups
    counts = df[col].value_counts()
    valid_cats = counts[counts > 100].index
    
    if len(valid_cats) == 0:
        return # Skip if no categories have enough data
        
    ct = ct.loc[valid_cats]
    
    # Sort by the critical '<30' class if it exists
    if '<30' in ct.columns:
        ct = ct.sort_values(by='<30', ascending=False)
        
    # Plot top 15 max to keep it readable
    ct = ct.head(15)
        
    ct[['NO', '>30', '<30']].plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title(f'Readmission by {col} (Top 15 cats >100 samples)')
    ax.set_ylabel('% of Patients')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Target')

# Plot for ALL categorical features
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    plot_readmission_distribution(col, axes[i])

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('plots/EDA/feature_vs_target_categorical.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_code_cell("""# Plot for ALL numerical features (by binning them)
num_rows = math.ceil(num_cols_count / 3)
fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    # Create 4 quantile bins for the numerical feature
    bin_col = f'{col}_bin'
    try:
        df[bin_col] = pd.qcut(df[col], q=4, duplicates='drop')
        plot_readmission_distribution(bin_col, axes[i])
    except ValueError:
        # If qcut fails (e.g. too many zeros), use regular cut
        df[bin_col] = pd.cut(df[col], bins=4)
        plot_readmission_distribution(bin_col, axes[i])

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('plots/EDA/feature_vs_target_numerical.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""### What we learned
- Across almost all features, the proportion of `<30` and `>30` fluctuates.
- Certain `discharge_disposition_id` codes have very high readmission rates.
- Patients with higher `number_inpatient` (previous inpatient visits) have noticeably higher proportions of early readmission (`<30`).
- Medication changes (`change` = "Ch") slightly increase the likelihood of readmission.

### What we will do about it
- `number_inpatient` will likely be a very strong predictive feature.
- We will group similar discharge disposition codes and admission source codes to strengthen the signal for the models.
"""))

# 7. Relationships (Correlation)
cells.append(create_markdown_cell("""## 7. Feature Relationships
Checking for multicollinearity amongst ALL numerical features."""))

cells.append(create_code_cell("""# Correlation heatmap
corr = df[numeric_cols].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Spearman Correlation between ALL Numerical Features')
plt.savefig('plots/EDA/correlation_heatmap.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""### What we learned
- `num_medications` and `time_in_hospital` are positively correlated (0.47).
- `num_medications` and `num_procedures` also have positive correlation (0.39).
- Overall, there is no severe multicollinearity (no correlations > 0.8).

### What we will do about it
- We don't need to aggressively drop numerical features due to multicollinearity. All of them can be retained for the tree-based models.
"""))

# 8. Data Quality & Leakage
cells.append(create_markdown_cell("""## 8. Data Quality & Conceptual Leakage
Certain patients (e.g., deceased or transferred to hospice) conceptually cannot be readmitted. Including them can bias the model."""))

cells.append(create_code_cell("""# Check discharge_disposition_id for expired/hospice patients
# Codes 11, 13, 14, 19, 20, 21 represent death or hospice
terminal_codes = [11, 13, 14, 19, 20, 21]
terminal_patients = df[df['discharge_disposition_id'].isin(terminal_codes)]

print(f"Patients discharged to hospice or expired: {len(terminal_patients)}")
"""))

cells.append(create_markdown_cell("""### What we learned
- Thousands of patients were discharged to hospice or expired.
- These patients are virtually guaranteed not to be readmitted, meaning they are "easy" negatives that artificially inflate model performance for the 'NO' class.

### What we will do about it
- We will **drop** all rows with these discharge disposition codes during preprocessing to ensure the model focuses on patients who actually *could* be readmitted.
"""))

# 9. Targeted EDA (Before Preprocessing)
cells.append(create_markdown_cell("""## 9. Targeted EDA (Before Preprocessing)
We need to explicitly analyze the ID mappings and the internal distributions of test results and medications before we can safely finalize Phase 4 and Phase 5."""))

cells.append(create_code_cell("""# A. ID Codes Distributions
id_cols_to_check = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(id_cols_to_check):
    counts = df[col].value_counts().nlargest(10)
    sns.barplot(x=counts.index, y=counts.values, ax=axes[i], order=counts.index, palette='magma')
    for c in axes[i].containers: axes[i].bar_label(c, fmt='%.0f', fontsize=9)
    axes[i].set_title(f'Top 10: {col}')
    
plt.tight_layout()
plt.savefig('plots/EDA/id_mappings_targeted.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_code_cell("""# B. Test Results & Medications Ordinality Check
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A1Cresult vs Readmission
plot_readmission_distribution('A1Cresult', axes[0])
axes[0].set_title("A1Cresult vs Target")

# Plot Insulin (since it's the most common) vs Readmission
plot_readmission_distribution('insulin', axes[1])
axes[1].set_title("Insulin vs Target")

plt.tight_layout()
plt.savefig('plots/EDA/meds_tests_targeted.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""### What we learned from Targeted EDA
- For `A1Cresult`, the proportion of early readmissions (`<30`) does **not** strictly increase from `Norm` -> `>7` -> `>8`. This means it is NOT strictly ordinal in relation to our target.
- For `insulin`, the `Up` and `Down` categories exhibit different risk profiles compared to `Steady`.
- ID columns are heavily concentrated in just 2-4 categories (e.g., ID 1, 2, 3), meaning we can safely group the long tail into "Other".

### What we will do about it
- We will group the ID columns based on their Top 3 mapped categories and map the rest to "Other" in Phase 4.
- We will use **One-Hot Encoding** (Nominal) for `A1Cresult` and medications in Phase 5 rather than Ordinal Encoding, as it prevents forcing false linear assumptions on non-linear risks.
"""))

# 10. Conclusions
cells.append(create_markdown_cell("""## 10. EDA Conclusions & Next Steps
We have a clear path forward for the Preprocessing phase.

| Finding | Implication | Action |
| :--- | :--- | :--- |
| `weight` has 97% missing data | Unusable feature | Drop column |
| Near-zero variance on many medications | No predictive power | Drop columns with >99% 'No' |
| Multi-class imbalance | Accuracy is a bad metric | Use SMOTE/class weights |
| High encounters per patient | Random split causes leakage | GroupKFold / Split by `patient_nbr` |
| Hospice/Death discharge codes | Inflates negative classes | Drop rows with terminal codes |
| Diagnosis codes have 700+ unique values | High dimensionality | Group ICD-9 codes by category |
| Tests/Meds lack strict monotonic correlation | Ordinal encoding creates false math | Use One-Hot Encoding |
"""))

# --- PREPROCESSING START ---
cells.append(create_markdown_cell("""# Preprocessing

## Phase 1: Clean Rows and Columns
Following our Preprocessing Plan, we will drop terminal patients, drop `weight` and identifier columns, and dynamically drop near-zero variance medications."""))

cells.append(create_code_cell("""# 1. Drop Hospice/Expired
# Save 'Before' distribution
dist_before = df['readmitted'].value_counts(normalize=True) * 100

terminal_codes = [11, 13, 14, 19, 20, 21]
df_clean = df[~df['discharge_disposition_id'].isin(terminal_codes)].copy()
print(f"Original shape: {df.shape} | Shape after dropping terminal patients: {df_clean.shape}")

# Save 'After' distribution and plot
dist_after = df_clean['readmitted'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=dist_before.index, y=dist_before.values, ax=axes[0], palette='Reds', order=['NO', '>30', '<30'])
for c in axes[0].containers: axes[0].bar_label(c, fmt='%.1f%%', fontsize=10)
axes[0].set_title("Target Dist (Before Terminal Drop)")
axes[0].set_ylabel("Percentage (%)")

sns.barplot(x=dist_after.index, y=dist_after.values, ax=axes[1], palette='Greens', order=['NO', '>30', '<30'])
for c in axes[1].containers: axes[1].bar_label(c, fmt='%.1f%%', fontsize=10)
axes[1].set_title("Target Dist (After Terminal Drop)")

# Make sure plots/Preprocessing folder exists
import os
os.makedirs('plots/Preprocessing', exist_ok=True)
plt.savefig('plots/Preprocessing/phase1_terminal_drop.png', bbox_inches='tight')
plt.show()

# 2. Drop columns
cols_to_drop = ['weight', 'examide', 'citoglipton', 'encounter_id']
df_clean.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# 3. Drop near-zero variance medications (>99% 'No')
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

dropped_meds = []
for m in med_cols:
    if m in df_clean.columns:
        if (df_clean[m] == 'No').mean() * 100 > 99:
            dropped_meds.append(m)

df_clean.drop(columns=dropped_meds, errors='ignore', inplace=True)
print(f"Dropped {len(dropped_meds)} zero-variance medications: {dropped_meds}")
"""))

cells.append(create_markdown_cell("""## Phase 2: Split the Data
We use **Option A**: Keep all visits but strictly split by `patient_nbr`. We use the patient's first encounter to stratify the class balance appropriately."""))

cells.append(create_code_cell("""from sklearn.model_selection import train_test_split

# Get the first visit's target for each patient to stratify safely
first_encounters = df_clean.sort_values('patient_nbr').groupby('patient_nbr').first()
patient_targets = first_encounters['readmitted']

train_patients, test_patients = train_test_split(
    patient_targets.index, 
    test_size=0.20, 
    stratify=patient_targets.values,
    random_state=42
)

train_df = df_clean[df_clean['patient_nbr'].isin(train_patients)].copy()
test_df = df_clean[df_clean['patient_nbr'].isin(test_patients)].copy()

print(f"Train set: {train_df.shape[0]} encounters ({len(train_patients)} unique patients)")
print(f"Test set:  {test_df.shape[0]} encounters ({len(test_patients)} unique patients)")

# Plotting the Stratification Success
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_dist = train_df['readmitted'].value_counts(normalize=True) * 100
test_dist = test_df['readmitted'].value_counts(normalize=True) * 100

sns.barplot(x=train_dist.index, y=train_dist.values, ax=axes[0], palette='Blues', order=['NO', '>30', '<30'])
for c in axes[0].containers: axes[0].bar_label(c, fmt='%.1f%%', fontsize=10)
axes[0].set_title("Train Set Class Distribution")
axes[0].set_ylabel("Percentage (%)")

sns.barplot(x=test_dist.index, y=test_dist.values, ax=axes[1], palette='Oranges', order=['NO', '>30', '<30'])
for c in axes[1].containers: axes[1].bar_label(c, fmt='%.1f%%', fontsize=10)
axes[1].set_title("Test Set Class Distribution")

plt.savefig('plots/Preprocessing/phase2_split_stratification.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""## Phase 3: Missing Value Imputation
We will address the missing data carefully. Certain "missing" values are highly informative (e.g., lack of a specialist or lack of an A1C test) and must be treated as independent categories rather than imputed with a mode or median."""))

cells.append(create_code_cell("""# Save missing counts before Phase 3
missing_before = train_df.isnull().sum()
missing_before = missing_before[missing_before > 0]

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

# 2. Handle 'None' in lab test results
test_cols = ['max_glu_serum', 'A1Cresult']
for col in test_cols:
    train_df[col] = train_df[col].replace('None', 'Not tested').fillna('Not tested')
    test_df[col] = test_df[col].replace('None', 'Not tested').fillna('Not tested')

# 3. Drop tiny fraction of remaining NaNs (e.g., missing gender, missing diag_1)
train_df.dropna(subset=['gender', 'diag_1', 'diag_2', 'diag_3'], inplace=True)
test_df.dropna(subset=['gender', 'diag_1', 'diag_2', 'diag_3'], inplace=True)

# Plotting the Before & After of Missing Values
missing_after = train_df.isnull().sum()
missing_after = missing_after[missing_after > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if not missing_before.empty:
    ax0 = sns.barplot(x=missing_before.index, y=missing_before.values, ax=axes[0], palette='Reds')
    for c in ax0.containers: ax0.bar_label(c, fmt='%.0f', fontsize=9)
    axes[0].set_title("Missing Values Count (Before Phase 3)")
    axes[0].tick_params(axis='x', rotation=45)

if missing_after.empty:
    axes[1].text(0.5, 0.5, '0 Missing Values Remaining!\\nDataset is Clean.', ha='center', va='center', fontsize=16, color='green', fontweight='bold')
    axes[1].set_title("Missing Values Count (After Phase 3)")
    axes[1].axis('off')

plt.tight_layout()
import os
os.makedirs('plots/Preprocessing', exist_ok=True)
plt.savefig('plots/Preprocessing/phase3_missing_imputation.png', bbox_inches='tight')
plt.show()

print(f"Train set after Phase 3: {train_df.shape}")
print(f"Test set after Phase 3:  {test_df.shape}")

print("\\nMissing values remaining in Train:\\n", train_df.isnull().sum()[train_df.isnull().sum() > 0])
"""))

cells.append(create_markdown_cell("""## Phase 4: Feature Engineering
We will group high-cardinality features (like diagnosis codes), convert ordinal variables to numbers, and engineer new aggregate features like `total_visits` and `meds_per_day`."""))

cells.append(create_code_cell("""import re

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

# Age mapping
age_map = {'[0-10)':0, '[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':4, '[50-60)':5, '[60-70)':6, '[70-80)':7, '[80-90)':8, '[90-100)':9}
train_df['age'] = train_df['age'].map(age_map)
test_df['age'] = test_df['age'].map(age_map)

# Medical Specialty and Payer Code (Top 10)
# Remember: Apply the Top 10 from TRAIN to TEST to prevent leakage!
top_specialties = train_df['medical_specialty'].value_counts().nlargest(10).index
train_df['medical_specialty'] = train_df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other')
test_df['medical_specialty'] = test_df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other')

top_payers = train_df['payer_code'].value_counts().nlargest(10).index
train_df['payer_code'] = train_df['payer_code'].apply(lambda x: x if x in top_payers else 'Other')
test_df['payer_code'] = test_df['payer_code'].apply(lambda x: x if x in top_payers else 'Other')

# Grouping ID Columns (Admission Type, Source, Discharge)
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

# New Engineered Features
train_df['total_visits'] = train_df['number_outpatient'] + train_df['number_emergency'] + train_df['number_inpatient']
test_df['total_visits'] = test_df['number_outpatient'] + test_df['number_emergency'] + test_df['number_inpatient']

train_df['meds_per_day'] = train_df['num_medications'] / np.maximum(1, train_df['time_in_hospital'])
test_df['meds_per_day'] = test_df['num_medications'] / np.maximum(1, test_df['time_in_hospital'])

train_df['lab_tests_per_day'] = train_df['num_lab_procedures'] / np.maximum(1, train_df['time_in_hospital'])
test_df['lab_tests_per_day'] = test_df['num_lab_procedures'] / np.maximum(1, test_df['time_in_hospital'])

# Drug Summarization
drug_cols = [c for c in train_df.columns if c in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']]

def count_drug_changes(row):
    return sum(1 for d in drug_cols if row[d] in ['Up', 'Down'])

def count_active_drugs(row):
    return sum(1 for d in drug_cols if row[d] in ['Up', 'Down', 'Steady'])

train_df['num_drug_changes'] = train_df.apply(count_drug_changes, axis=1)
test_df['num_drug_changes'] = test_df.apply(count_drug_changes, axis=1)

train_df['num_active_drugs'] = train_df.apply(count_active_drugs, axis=1)
test_df['num_active_drugs'] = test_df.apply(count_active_drugs, axis=1)

# Drop drug columns except insulin
drugs_to_drop = [d for d in drug_cols if d != 'insulin']
train_df = train_df.drop(columns=drugs_to_drop, errors='ignore')
test_df = test_df.drop(columns=drugs_to_drop, errors='ignore')

print("Phase 4 Feature Engineering complete!")
print("Train set shape:", train_df.shape)
"""))

cells.append(create_code_cell("""# Visualization of Phase 4 (Feature Engineering Proof)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=train_df, x='diag_1', ax=axes[0], palette='magma')
for c in axes[0].containers: axes[0].bar_label(c, fmt='%.0f', fontsize=9)
axes[0].set_title("Grouped Diagnoses (diag_1) - Reduced from 700+ to 9")
axes[0].tick_params(axis='x', rotation=45)

# We've removed the binning plot since we are keeping them as raw counts for experimentation
sns.histplot(data=train_df, x='number_inpatient', ax=axes[1], bins=20, color='teal')
axes[1].set_title("Raw Inpatient Visits (Preserved for Experimentation)")
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('plots/Preprocessing/phase4_features.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""## Phase 5: Prepare for Models
Here we will encode the target variable, apply log transformations to numerical features, and use scikit-learn's `ColumnTransformer` to One-Hot Encode categorical variables and Scale numerical variables."""))

cells.append(create_code_cell("""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Target Encoding (NO = 0, >30 = 1, <30 = 2)
target_mapping = {'NO': 0, '>30': 1, '<30': 2}
y_train = train_df['readmitted'].map(target_mapping)
y_test = test_df['readmitted'].map(target_mapping)

# Drop target and patient identifiers from feature sets
X_train = train_df.drop(columns=['readmitted', 'patient_nbr'])
X_test = test_df.drop(columns=['readmitted', 'patient_nbr'])

# Identify column types
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

# 2. Build the ColumnTransformer
# The plan specifies applying log1p to skewed counts. We'll apply it to numeric features.
log_transformer = FunctionTransformer(np.log1p, validate=True)

numeric_transformer = Pipeline(steps=[
    ('log', log_transformer),
    ('scaler', StandardScaler())
])

# For categorical features, we one-hot encode
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Fit on Train, Transform Train and Test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(cat_features_out)

print(f"Original X_train shape: {X_train.shape}")
print(f"Preprocessed X_train shape: {X_train_preprocessed.shape}")
"""))

cells.append(create_code_cell("""# Visualization of Phase 5 (Log1p & Scaling Proof)
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original highly skewed feature
sns.histplot(X_train['total_visits'], bins=30, ax=axes[0], color='orange', kde=True)
axes[0].set_title("Before: 'total_visits' (Highly Right-Skewed)")
axes[0].set_ylabel("Count")

# Extract the transformed 'total_visits' column
total_visits_idx = numerical_features.index('total_visits')
transformed_total_visits = X_train_preprocessed[:, total_visits_idx]

sns.histplot(transformed_total_visits, bins=30, ax=axes[1], color='purple', kde=True)
axes[1].set_title("After: 'total_visits' (log1p + StandardScaler)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig('plots/Preprocessing/phase5_scaling.png', bbox_inches='tight')
plt.show()
"""))

cells.append(create_markdown_cell("""## Phase 6: Handle Imbalance and Save
The dataset is imbalanced. We will use `class_weight='balanced'` in our models, and later try SMOTE during cross-validation. 
Finally, we save the preprocessed datasets and the pipeline for use in modeling."""))

cells.append(create_code_cell("""import joblib
import pandas as pd
import os

os.makedirs('dataset_processed', exist_ok=True)

# Convert preprocessed arrays back to DataFrames
X_train_final = pd.DataFrame(X_train_preprocessed, columns=all_feature_names)
X_test_final = pd.DataFrame(X_test_preprocessed, columns=all_feature_names)

# Save datasets
X_train_final.to_csv('dataset_processed/X_train_final.csv', index=False)
X_test_final.to_csv('dataset_processed/X_test_final.csv', index=False)
y_train.to_csv('dataset_processed/y_train_final.csv', index=False)
y_test.to_csv('dataset_processed/y_test_final.csv', index=False)

# Save the sklearn pipeline
joblib.dump(preprocessor, 'dataset_processed/preprocessing_pipeline.pkl')

print("All preprocessing steps completed and saved successfully!")
"""))

# Construct JSON
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "ML Project Kernel",
            "language": "python",
            "name": "ml_project_kernel"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
