# Personal Journey: From Raw Data to ML-Ready Features
**Name:** Fernando S.A.W
**IT Number:** IT24101304
**Role:** Exploratory Data Analysis & Preprocessing / Feature Engineering

## The Initial Overwhelm & Early Failures in EDA
When I first loaded the Diabetes 130-US Hospitals dataset, the sheer volume of data—over 101,000 encounters and 50 features—felt overwhelming. My initial instinct was simply to write a script that generated standard histograms and boxplots for every single variable. However, this led to an early failure: I produced dozens of charts that looked statistically correct but lacked any real clinical or business meaning. 

For instance, trying to plot the `diag_1` (primary diagnosis) column resulted in an unreadable chart with over 700 unique ICD-9 codes. It hit me that plotting data without a clear strategy was useless. I had to pivot my entire approach. I realized that true Exploratory Data Analysis isn't just about visualization; it's about interrogating the data to make concrete preprocessing decisions.

## Discovering Data Leakage and Hidden Nulls
By forcing myself to view the data through a logical, investigative lens, I uncovered three critical insights that fundamentally shifted how I approached the entire pipeline:

1. **The Patient Memorization Problem:** While checking the shape of the dataset, I noticed that although there were ~101,000 encounters, there were only about 71,500 unique patients (`patient_nbr`). Some patients had visited up to 40 times! I realized that if I performed a standard random train/test split, a single patient's records would bleed into both the training and testing sets. The model would just memorize the patient rather than learn generalizable patterns. I immediately decided that all future splitting *must* be strictly grouped by `patient_nbr`.
2. **Target Leakage in Terminal Codes:** Digging into the `IDS_mapping.csv`, I found that certain discharge codes (like 11, 13, 14, 19, 20, 21) literally meant the patient had expired or was transferred to hospice. A deceased patient physically cannot be readmitted. Leaving these rows in would artificially inflate the accuracy of predicting the "NO" class. I realized I had to drop these records entirely to maintain real-world business logic.
3. **Unmasking Hidden Missing Data:** I also discovered that many standard integer codes in the admin columns actually mapped to "NULL" or "Not Mapped". A standard `.isnull()` check in pandas completely missed these. I learned that I had to manually parse the mapping definitions and convert these conceptually missing IDs to true `NaN` values before doing any imputation.

## Refining Features with Statistics (Self-Correction)
As I continued EDA, I initially used arbitrary filters (like only plotting categories with `>100` samples) to clean up my categorical charts. But I quickly realized this was hiding valuable data on rare categories. I decided to teach myself a more rigorous approach. Instead of guessing what to plot, I implemented **Cramér's V and Chi-Square tests** to mathematically rank every categorical feature by its actual association with the target. This allowed me to confidently select the top 9 most influential features to plot, rather than relying on arbitrary visual filters.

## Building the Preprocessing Pipeline
Moving into the Preprocessing and Feature Engineering phase, the challenges shifted from understanding the data to transforming it without introducing bias:

- **Handling Outliers Counter-Intuitively:** When I looked at the numerical features, variables like `number_inpatient` (prior visits) had extreme right-tail outliers. My first thought was to drop them using the IQR method. But stepping back, I realized that patients with 20+ prior visits are the exact "high-utilizer" patients that are most at risk of readmission! Dropping them would delete my most important signal. Instead, I taught myself to apply a `log1p` transformation, which safely compressed these extreme values to stabilize linear models without losing the high-risk patients.
- **Tackling High Dimensionality:** The 700+ diagnosis codes and 23 individual medication columns were too sparse to be useful. I engineered a solution by grouping the ICD-9 codes into 9 broad clinical categories (like 'Circulatory' and 'Diabetes'). For the medications, I realized that 20 of them had near-zero variance (>99% "No"). I dropped the useless ones and summarized the rest into powerful aggregate features: `num_drug_changes` and `num_active_drugs`, capturing the intensity of polypharmacy rather than memorizing rare drug names.

## Conclusion
My biggest takeaway from this journey is that preprocessing is not just a checklist of pandas commands—it is an exercise in domain logic and preventing data leakage. From catching the non-monotonic relationships that forced me to use One-Hot Encoding instead of Ordinal Encoding, to strictly fitting my `ColumnTransformer` only on the training data, I learned that a machine learning model's predictive power is strictly bound by the rigor and clinical accuracy of the feature engineering that precedes it.
