# Personal Journey: Preprocessing the Diabetic Readmission Dataset
**Name:** Abeyrathna P.H.T.S
**IT Number:** IT24100007
**Role:** Preprocessing & Feature Engineering

## The Initial Overwhelm & Early Failures
Facing over 100,000 records and 50 features, my textbook instinct was to aggressively clean the data—drop missing values and randomly split 80/20. This led to my first major mistake. I initially used a standard random split and achieved overly optimistic baseline results, only to realize I had introduced fatal data leakage because some patients visited the hospital up to 40 times. Grouping the split strictly by `patient_nbr` was a hard-learned lesson in the difference between theoretical machine learning and real-world clinical data. 

## Clinical Context Over Blind Statistics
Relying on my teammate’s EDA findings, I learned not to apply statistical formulas blindly without understanding the domain. I focused on three core clinical representations:
* **Missing Data as Signal:** Instead of mathematically imputing `medical_specialty` or `payer_code`, I recognized that the *absence* of a specialist is a workflow reality. Imputing these as `"Unknown"` treated the missingness as a valid clinical signal.
* **Taming Rare Categories:** Retaining 70+ rare doctors would cause tree models to severely overfit. I kept only the Top 10 most frequent categories (which covered over 85% of all patient records), grouping the rest into `"Other"`.
* **Conceptual Grouping:** Purely statistical grouping of Administrative IDs (e.g., Discharge Disposition) destroyed their meaning. Manually mapping them into conceptual flows like `"Home"` or `"Facility"` preserved their real-world impact.

## Feature Engineering
Keeping 22 highly sparse columns for specific diabetes medications introduced excessive noise. Instead, I engineered two features: `num_active_drugs` and `num_drug_changes`. This captured the concepts of "polypharmacy" and "health instability" while drastically reducing dimensionality. I applied similar logic to create a `lab_tests_per_day` ratio to capture the intensity of a hospital stay.

## Embracing an Experimental Mindset
The most significant evolution in my approach came from a lecturer's feedback. I had initially binned outlier values (like a patient with 20 prior visits) into a safe `"3+"` category to protect linear models. The lecturer pointed out that tree-based algorithms are immune to skew and thrive on raw counts. 

Rather than assuming one method was better, I adopted an experimental design. I updated the pipeline to preserve raw numeric counts and created a granular 17-category diagnosis mapping to test alongside my baseline 9-category mapping. 

## Conclusion
Wrapping the final transformations securely inside a scikit-learn `ColumnTransformer` ensured no leakage occurred during encoding and scaling. Moving forward into the modeling phase, this experimental foundation allows me to objectively let cross-validation prove whether raw counts and granular diagnoses truly improve predictive performance over the safer baselines.
