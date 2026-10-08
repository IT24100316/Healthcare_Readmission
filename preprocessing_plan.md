# Preprocessing Steps (Simple English)

## Phase 1: Clean the rows and columns
* Remove patients who died or went to hospice (discharge codes 11, 13, 14, 19, 20, 21). They can't really be readmitted. This removes 2,423 rows.
* Drop columns we don't need:
  * `weight`, because it is 97% missing.
  * `examide` and `citoglipton`, because every row has the same value.
  * `encounter_id`, because it is just a row label.
* Keep `patient_nbr` for now. We only use it to split the data.
* Check the other drug columns. If a drug is almost always "No" and also has no link to readmission, drop it. If it shows any link, keep it.

## Phase 2: Split the data
* Split by patient, not by row. Put 80% of patients in training and 20% in testing, so the same patient is never in both. Keep the class mix (NO / >30 / <30) similar in both sets.
* Decide how to handle patients with many visits. Either:
  * **Option A**: keep all visits and split by patient (my suggestion, it keeps more data).
  * **Option B**: keep only each patient's first visit (simpler, but loses about 30k rows).
* Whichever you pick, write down why and show the row counts and class percentages.
* **From now on, learn everything from the training set only.** Averages, top categories and scaling values must come from training data and then be applied to the test data.

## Phase 3: Fix missing values
* Turn hidden nulls into real missing values. Change `?` to NaN. Use `IDS_mapping.csv` to find ID codes that mean "NULL" or "Not Mapped" and turn them into NaN too.
* Fill the gaps:
  * `medical_specialty`, `payer_code`, `race` and the ID columns → "Unknown".
  * `max_glu_serum` and `A1Cresult` → "Not tested" (a missing value here means the test wasn't done).
  * The 3 gender rows and 21 `diag_1` rows → pick one rule (drop them or "Unknown") and use it everywhere.

## Phase 4: Build better features
* Group the diagnosis codes (`diag_1`, `diag_2`, `diag_3`) into about 9 groups: circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other.
* Group the ID codes into a few simple categories:
  * Discharge: home / facility / home health / other.
  * Admission type: emergency / elective / other.
  * Admission source: emergency room / referral / transfer / other.
* Shrink long lists. Keep the top ~10 values of `medical_specialty` and `payer_code`, and put the rest into "Other".
* Turn age into a number. For example, `[50-60)` becomes 5.
* Bin the visit counts into 0, 1, 2 and 3+ for `number_outpatient`, `number_emergency` and `number_inpatient`. 
* Summarise the drug columns. Count how many drugs were changed up or down, and how many are active. Keep insulin as its own column.
* Add a few helpful features: total prior visits, medications per day in hospital, and lab tests per day.

## Phase 5: Prepare for the models
* Keep outliers. They are real patients, not mistakes. Tree models handle them fine. For linear models, apply `log1p` to skewed counts instead of cutting them off.
* One-hot encode the remaining category columns.
* Scale the numbers with StandardScaler, only for models that need it (like logistic regression).
* Encode the target: NO = 0, >30 = 1, <30 = 2.

## Phase 6: Handle imbalance and save
* Imbalance: start with class weights ("balanced"). Try SMOTE later, but only on training folds during cross-validation, never before the split.
* Put steps 3–5 into one sklearn `Pipeline` so every step is fitted on training data only.
* Save the cleaned train and test sets and the pipeline.

**Don't forget:** For each step above, add a row to your decision log: decision, options considered, what you chose, reason, evidence.
