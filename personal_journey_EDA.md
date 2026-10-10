# Personal Journey: Exploratory Data Analysis (EDA)
**Name:** [Friend's Name]
**IT Number:** [Friend's IT Number]
**Role:** Exploratory Data Analysis (EDA)

## The Initial Overwhelm & Early Failures
When I first received the Diabetes 130-US Hospitals dataset, the 100,000 encounters and 50 features felt overwhelming. My initial instinct was simply to write code generating standard histograms for every variable. However, this led to an early struggle: I produced dozens of charts that looked statistically correct but lacked any real clinical or business meaning. 

For instance, trying to plot the `diag_1` (primary diagnosis) column resulted in an unreadable chart with over 700 unique ICD-9 codes. It hit me that plotting data without a clear strategy was useless. I realized I had to pivot my entire approach and view the data through an investigative lens to figure out what was actually driving readmission.

## Teaching Myself Statistical Feature Ranking
Initially, to clean up my categorical charts, I just applied a basic filter to only show categories with `>100` samples. But I soon realized this was a massive mistake—it was hiding valuable long-tail data and rare categories! I knew I needed a better way to find out which categorical features actually mattered. 

I took the initiative to self-learn and implement **Cramér's V and Chi-Square tests** in my EDA code. Instead of blindly plotting 30+ categorical charts, my new code mathematically ranked every feature by its true association with the target. This breakthrough allowed me to confidently narrow down and visualize only the top 9 most influential features, which completely changed our understanding of the dataset.

## Unmasking Hidden Dangers
Digging deeper into the data dictionaries, I uncovered two critical issues that would have destroyed our models if left unchecked:
1. **Target Leakage:** Analyzing the `discharge_disposition_id` mapping revealed codes representing "expired" (death) or "hospice care". Conceptually, these patients cannot be readmitted. Leaving them in the dataset would artificially inflate model performance on the negative class. I documented this explicitly so the preprocessing team would drop them.
2. **Hidden Nulls:** Parsing the `IDS_mapping.csv` revealed that many standard integer codes were actually masking "NULL" or "Not Mapped" values. A naive pandas `.isnull()` check missed these entirely. 

## Conclusion
As I prepared my findings for the Preprocessing handoff, I performed a "Targeted EDA" on test results and medication changes. By plotting readmission rates, I proved that the relationship was not strictly monotonic (e.g., readmission risk doesn't strictly increase linearly from `Norm` to `>7` to `>8`). Catching this non-monotonic relationship was huge—it proved to the preprocessing team that they must use One-Hot Encoding instead of Ordinal Encoding. I've learned that the predictive power of our final model is entirely bound by the depth of understanding gained right here in the EDA phase.
