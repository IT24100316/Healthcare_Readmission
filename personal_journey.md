# Individual Learning Journey - Member 1
**Name:** Fernando S.A.W
**IT Number:** IT24101304
**Role:** Problem Framing & Data Understanding

---

## 1. Initial Thoughts and Problem Framing
When we first received the Diabetes 130-US Hospitals dataset, my immediate challenge was translating raw hospital data into a concrete machine learning problem. The initial dataset felt overwhelming—over 100,000 encounters and 50 features, with a target variable (`readmitted`) split into three classes: NO, <30 days, and >30 days.

I realized that to satisfy the assignment, we need to handle all three classes. The `<30` class is the most critical for hospital intervention, so we will need to prioritize metrics related to this specific class during our modeling phase, even while treating it as a multi-class problem.

## 2. Navigating the Data Understanding Phase (EDA)
My core responsibility was to lead the Exploratory Data Analysis (EDA) to understand what we were actually working with. This phase taught me the crucial difference between "writing code" and "doing data science." 

Initially, I plotted standard histograms and bar charts for every variable. However, referring back to our grading rubric, I learned that charts without interpretation hold little value. I shifted my approach to maintaining a strict **EDA Insight Log**. For every visualization, I forced myself to answer: *What does this mean for our readmission risk?* and *What action must the team take in the preprocessing stage?*

**Key Discoveries and Challenges:**
* **The "Weight" Variable Trap:** I noticed the `weight` column was missing in ~97% of the records. My initial instinct was to try and impute it based on age or gender. However, through critical reflection, I realized that imputing 97% of a feature introduces massive bias and noise. I made the hard decision to log it as a severe data quality issue and advocate for dropping it entirely.
* **The Patient Leakage Discovery:** The most critical lesson I learned during EDA was regarding data leakage. While analyzing duplicate records, I discovered that there were ~101,000 encounters but only ~71,500 unique patients. Some patients had visited the hospital up to 40 times! If we performed a standard, random train/test split, a single patient's visits could end up in both sets, allowing the model to "cheat" by memorizing patient-specific patterns. I immediately flagged this as a critical data issue, ensuring our team will use a patient-level split (GroupKFold) moving forward.
* **Target Leakage from Discharge Codes:** By analyzing the `discharge_disposition_id` mapping, I noticed codes representing "expired" (death) or "hospice care". Conceptually, these patients cannot be readmitted. Leaving them in the dataset would artificially inflate our model's performance on the negative class (target leakage). Identifying this business-logic flaw early was a proud moment for me.
* **Outlier Detection via IQR:** While writing the Univariate Analysis, I implemented dynamic outlier detection using the Interquartile Range (IQR) method and visualized them using boxplots. This highlighted that certain features, like inpatient visits, have massive right-tail outliers. We must handle these carefully (e.g., via Winsorization or robust models) to prevent them from skewing our gradient descents later on.
* **Unmasking Hidden Nulls:** I successfully parsed the stacked `IDS_mapping.csv` to map integer codes to their real string descriptions. This revealed that many integers were actually masking "NULL" or "Not Mapped" values, which standard imputation would have completely missed!

## 3. Reflection and Next Steps
Leading the EDA phase has taught me that the quality of a machine learning model is strictly bound by the quality of the data going into it. I've learned to view data not just as numbers, but as real-world events that have constraints (like terminal patients not returning). 

As I hand off the EDA insights and logs to Member 2 for Preprocessing and Feature Engineering, I feel confident that we have a clean roadmap. My next focus will be assisting the team in ensuring the preprocessing steps accurately follow the data quality recommendations I've logged, specifically ensuring the patient-level splitting is executed correctly to maintain the integrity of our final model evaluation.
