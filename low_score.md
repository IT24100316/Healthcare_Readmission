Think of it like an **exam with questions from two different students**:

---

### 1. The Normal (Naive) Random Split: "The Cheat"
Imagine **Patient John** visited the hospital **10 times**.

If you just do a normal random train/test split:
* Visits 1 to 8 of John go into the **Train set**.
* Visits 9 and 10 of John go into the **Test set**.

**What happens?**
The model sees John's age, John's specific rare disease, John's exact doctor, and John's exact medication mix during training. When testing on Visit 9, the model **already knows John personally**. It doesn't need to learn general medicine; it just remembers John. 

This artificially inflates the accuracy to **75%–80%**, but in real life, when a **brand new patient** walks into the hospital, the model fails.

---

### 2. What Your Team Did: Patient-Level Split (Option A)
Instead of splitting visits randomly, you split by **people**:

* **80% of people (e.g. John, Sarah, David)** and **ALL of their visits** go to the **Train set**.
* **20% of people (e.g. Mike, Emma)** and **ALL of their visits** go to the **Test set**.

**What happens now?**
The model is tested on **Mike and Emma**, whom it has **never seen a single time before**.

---

### 3. Why does this make your score lower (54% instead of 80%)?
Because predicting whether a **complete stranger** will return to the hospital is much harder than predicting for a patient the model already studied 8 times! 

* A score of **80% with random splitting is fake** (it’s just memorization / data leakage).
* A score of **54% with patient splitting is real-world medicine** (true generalization to new patients).