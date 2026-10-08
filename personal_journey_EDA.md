# Personal Journey: Problem Framing & Exploratory Data Analysis (EDA)
**Name:** Fernando S.A.W
**IT Number:** IT24101304
**Role:** Problem Framing & Data Understanding

## The Initial Overwhelm & Early Failures
When I first received the Diabetes 130-US Hospitals dataset, the 100,000 encounters and 50 features felt overwhelming. My initial instinct was simply to write code generating standard histograms for every variable. However, this led to an early struggle: I produced dozens of charts that looked statistically correct but lacked any real clinical or business meaning. 

A lecturer’s feedback pointed out that visualizations without actionable interpretations hold zero value in machine learning. This forced me to pivot my entire approach. I created an **EDA Insight Log** where, for every single plot, I forced myself to answer: *“What does this mean for hospital readmission risk, and what exact action must the preprocessing team take?”*

## Critical Discoveries
By viewing the data through a clinical lens rather than just a mathematical one, I identified three major insights that fundamentally shaped our team's entire pipeline:

* **The Patient Leakage Discovery:** While analyzing duplicate records, I discovered that 100,000 encounters only belonged to ~71,500 unique patients (some visiting up to 40 times). If we performed a standard random split, a single patient's records would bleed into both the training and testing sets. Flagging this ensured our team used a strict patient-level split, preventing the model from simply memorizing individuals.
* **Target Leakage in Discharge Codes:** Analyzing the `discharge_disposition_id` mapping revealed codes representing "expired" (death) or "hospice care". Conceptually, these patients cannot be readmitted. Leaving them in the dataset would artificially inflate model performance on the negative class. This required dropping those records completely to maintain business logic.
* **Unmasking Hidden Nulls:** Parsing the `IDS_mapping.csv` revealed that many standard integer codes were actually masking "NULL" or "Not Mapped" values. A naive statistical imputation algorithm would have missed these entirely, treating them as valid categories.

## Targeted EDA for Feature Engineering
As we prepared to hand off to the feature engineering phase, I initially assumed we could just use Ordinal Encoding for test results like `A1Cresult` (e.g., -1, 0, 1). However, I performed a "Targeted EDA" to verify this assumption. By plotting the readmission rates against test results, I discovered the relationship was not strictly monotonic (e.g., readmission risk doesn't strictly increase linearly from `Norm` to `>7` to `>8`). 

## Conclusion
Catching that non-monotonic relationship proved the absolute necessity of Targeted EDA. If I hadn't verified the data visually, we would have fed mathematically false assumptions into our linear models, opting for Ordinal instead of One-Hot Encoding. As I hand these insights off to my teammate for Preprocessing, I have learned that the predictive power of an ML model is strictly bound by the depth of understanding gained during the EDA phase.
