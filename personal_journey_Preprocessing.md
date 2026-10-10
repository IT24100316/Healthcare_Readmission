# Personal Journey: Preprocessing & Feature Engineering
**Name:** Fernando S.A.W
**IT Number:** IT24101304
**Role:** Data Preprocessing & Feature Engineering

## Inheriting the Data & Finding Leakage
Taking over the dataset after the EDA phase, my immediate task was to transform the raw, messy clinical data into a clean, machine-learning-ready matrix. But right away, I spotted a massive logical flaw in how we were going to split the data.

While analyzing the dataset shape, I noticed that although there were ~101,000 encounters, there were only about 71,500 unique patients (`patient_nbr`). Some patients had visited up to 40 times! I quickly realized that if I performed a standard random train/test split, a single patient's records would bleed into both the training and testing sets. The model would just memorize the patient rather than learn generalizable patterns. I took it upon myself to completely rewrite the splitting logic to group strictly by `patient_nbr`, ensuring zero patient memorization leakage.

## Counter-Intuitive Outlier Handling
One of my biggest self-learning moments came when dealing with outliers. When I looked at the numerical features, variables like `number_inpatient` (prior visits) had extreme right-tail outliers. My first instinct, based on standard tutorials, was to just drop them using the IQR method. 

But stepping back and thinking about the hospital's actual business problem, I realized something critical: patients with 20+ prior visits are the exact "high-utilizer" patients that are most at risk of readmission! Dropping them would delete the most important signal in the dataset. Instead, I researched and applied a `log1p` transformation, which safely compressed these extreme values to stabilize our linear models without losing the high-risk patients.

## Conquering High Dimensionality
The dataset suffered from massive dimensionality and sparsity. The 700+ diagnosis codes and 23 individual medication columns were too noisy to be useful directly. 
I engineered a robust solution:
1. I grouped the ICD-9 codes into 9 broad clinical categories based on medical definitions.
2. For the medications, I noticed that 20 of them had near-zero variance (>99% "No"). I dropped the useless ones and created powerful new aggregate features: `num_drug_changes` and `total_visits`. This captured the true intensity of the patient's condition (polypharmacy) rather than forcing the model to memorize rare drug names.

## Conclusion
My biggest takeaway from this journey is that preprocessing is not just a checklist of code functions—it is an exercise in rigorous domain logic. From dropping deceased patients to strictly fitting my `ColumnTransformer` only on the training data to prevent future leakage, I learned that a machine learning model's predictive power relies entirely on the structural integrity of the preprocessing pipeline I built.
