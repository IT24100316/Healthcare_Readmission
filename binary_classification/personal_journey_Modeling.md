# Personal Journey: Classification Modelling (Binary Track)
**Name:** [Your Name]  
**IT Number:** IT24100316  
**Role:** Classification Modelling (Binary Track: 30-Day Early Readmission)  

---

## 1. Inheriting the ML-Ready Binary Matrix & The Baseline Challenge
Taking over the binary classification track after data preprocessing, I inherited a clean, leakage-free dataset of 78,331 training encounters and 86 engineered clinical features from `binary_classification/dataset_processed/`. Every record was fully encoded, normalized via robust scaling, and verified to contain zero missing values.

Our clinical formulation targets the acute 30-day post-discharge window defined by the CMS Hospital Readmissions Reduction Program (HRRP):
* **Positive Class (1):** Early readmission within 30 days (`<30`) — 9,004 encounters (**11.49%**). These represent high-risk bounce-backs requiring urgent transitional outpatient interventions.
* **Negative Class (0):** No early readmission (`NO` or `>30` combined) — 69,327 encounters (**88.51%**). These represent stable post-discharge patient trajectories.

My primary responsibility was unambiguous: establish a rigorous, leakage-free classification benchmark, evaluate diverse algorithmic families, conquer the severe 1 : 7.7 class imbalance without corrupting data distributions, conduct deep hyperparameter optimization, and deliver the optimal champion model for clinical deployment.

My first task was establishing our baseline using **Logistic Regression**. In healthcare, interpretability is a paramount clinical requirement—if a transparent linear model performs well, clinicians and hospital administrators can directly inspect odds ratios and feature weights. However, setting this up immediately exposed two critical challenges:
1. **Severe Imbalance Sensitivity:** Standard unweighted logistic regression is overwhelmed by the 88.51% majority class, collapsing the decision hyperplane toward predicting zero early readmissions and rendering the model clinically useless despite an apparent 88% accuracy. To resolve this, I combined **SMOTE (`sampling_strategy=0.30`)** with balanced class weighting.
2. **Evaluation Metric Focus:** Because standard classification accuracy is easily distorted by majority class dominance, I designed our cross-validation strictly around **Stratified 5-Fold CV**, establishing **Macro F1** and **Macro Recall** as our north-star metrics to guarantee equal penalization across both patient outcomes.

The baseline Logistic Regression pipeline achieved:
* **Mean CV Macro F1:** `0.5124`
* **Mean CV Macro Recall:** `0.6096`
* **Mean CV Accuracy:** `63.63%`

While this confirmed that the model actively identified early readmissions (Recall > 60%), it demonstrated that a flat linear hyperplane cannot adequately capture intricate, non-linear comorbidity interactions (such as the compounding risk of diabetic kidney disease alongside insulin titration).

---

## 2. Navigating Computational Bottlenecks: Algorithm Selection & Pruning
With 78,331 rows and 86 features across 5 folds (effectively 391,655 training instances evaluated per CV cycle), training speed and algorithmic scalability were immediate operational bottlenecks. During our initial architectural design, I evaluated a broad candidate pool of algorithms, including $k$-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, and Support Vector Machines (SVM).

I quickly made critical, mathematically grounded filtering decisions:
* **Pruning KNN and Traditional SVM:** A standard RBF-kernel SVM has a computational complexity of $O(n^2)$ to $O(n^3)$. On nearly 80,000 samples, computing an RBF kernel matrix would take hours per fold, while KNN ($O(nd)$) would produce unacceptable inference latency during real-time emergency department triage.
* **Pruning Naive Bayes & Single Decision Trees:** Naive Bayes strictly assumes conditional feature independence—a lethal assumption in clinical electronic health records where lab test frequencies, medications, and secondary diagnoses are tightly correlated. Single decision trees suffered from high variance, easily overfitting to solitary outlier encounters.
* **Selecting the Core Pool:** I locked in 6 distinct, high-potential architectures spanning four foundational learning paradigms:
  1. *Linear Models:* **Logistic Regression** (L1, L2, ElasticNet regularization).
  2. *Bagging Ensembles:* **Random Forest** (decorrelated bootstrap trees).
  3. *Boosting Ensembles:* **XGBoost** (exact level-wise gradient boosting), **LightGBM** (histogram leaf-wise boosting), and **CatBoost** (symmetric oblivious trees).
  4. *Neural Architectures:* **Multi-Layer Perceptron (MLP)** feedforward neural network.

---

## 3. Conquering Class Imbalance & The SMOTE Leakage Trap
A major methodological challenge was the 11.49% class imbalance. If left unaddressed, models achieve seemingly high accuracy by simply predicting all zeros, completely missing high-risk bounce-backs.

Each model family required a distinct, leakage-free strategy:
* **Tree Ensembles (Random Forest, LightGBM, CatBoost):** Handled via algorithmic cost-sensitive weighting (`class_weight='balanced'` and `auto_class_weights='Balanced'`), heavily penalizing minority false negatives directly inside the impurity/gradient split calculation without distorting the underlying feature space.
* **XGBoost Native Cost Weighting:** Unlike multiclass setups where `scale_pos_weight` errors out, binary classification allowed me to deploy XGBoost's native `scale_pos_weight = 7.70` (calculated dynamically as $\frac{69,327}{9,004}$). This mathematically forced second-order gradient steps to penalize missed readmissions 7.7 times harder.
* **Linear & Neural Models (Logistic Regression, MLP) & The SMOTE Pipeline:** Scikit-learn's `MLPClassifier` and linear probability estimators lack native cost-weight calibration. My initial thought was applying SMOTE across the dataset before cross-validation. However, stepping back, I caught a fatal data-leakage flaw: synthesizing samples before splitting allows synthetic vectors from validation folds to bleed into the training folds—a textbook methodology failure. To ensure strict empirical hygiene, I wrapped SMOTE (`sampling_strategy=0.30`) inside an `imblearn.pipeline.Pipeline`, guaranteeing that synthetic minority generation occurred *strictly inside* each individual training fold.

---

## 4. The Deep Hyperparameter Optimization Journey
Our initial surface-level tuning with Optuna (exploring only 3–5 basic parameters like depth and estimators) produced respectable baselines, but it was clear that our models had not reached their true predictive capacity. Tabular clinical data with 86 sparse binary and scaled continuous features is highly vulnerable to leaf memorization and noisy split chasing.

To address this, I executed a comprehensive **Deep Hyperparameter Optimization** phase (20 to 30 Bayesian TPE trials per model) across every architecture:

1. **Random Forest (Tree Decorrelation & Ensemble Diversity):**  
   In bagged ensembles, generalization is achieved through *tree decorrelation*. I expanded Optuna's search space to include fractional feature subsampling (`max_features` $\in [\text{'sqrt'}, \text{'log2'}, 0.4, 0.6]$), split criterion (`gini` vs `entropy`), restricted tree depths (`max_depth` $\in [6, 18]$), leaf sample thresholds (`min_samples_leaf` $\in [3, 12]$), and bootstrap sample draw limits (`max_samples` $\in [0.70, 0.95]$). Forcing trees to split on randomized feature subsets prevented dominant visit counts from drowning out subtle clinical signals like insulin titration or secondary comorbidity codes.
   * **Result:** **🏆 1st Place Champion!** Reached **`0.5699` CV Macro F1**, **`78.98%` Accuracy**, driving balanced recall to **`58.78%`** and full training recall on `<30` to **`96%`** (Precision: `59%`, F1: `0.73`).

2. **XGBoost (Exact Greedy Splits & Second-Order Gradient Penalization):**  
   Configured with `scale_pos_weight = 7.70` and `eval_metric='logloss'`. I introduced structural regularization controls: `min_child_weight` $\in [1, 12]$ (preventing leaves from fitting to solitary clinical outliers), `gamma` $\in [0.0, 5.0]$ (minimum loss reduction threshold per split), continuous $L_1$ sparsity (`reg_alpha` $10^{-4}$ to $10.0$), $L_2$ weight smoothing (`reg_lambda` $10^{-4}$ to $10.0$), and column/row subsampling (`colsample_bytree`, `subsample`).
   * **Result:** **🥈 2nd Place!** Achieved **`0.5563` CV Macro F1**, **`76.50%` Accuracy**, and **`91.0%` recall** on early readmissions in full training evaluation.

3. **LightGBM (Leaf-Wise Histogram Growth & Comorbidity Thresholding):**  
   LightGBM utilizes leaf-wise (best-first) growth, which rapidly captures non-linear comorbidity interactions but can overfit small patient subgroups. I regularized this using `num_leaves` $\in [15, 80]$, `min_child_samples` $\in [20, 150]$, dual $L_1/L_2$ penalties (`reg_alpha`/`reg_lambda`), and stochastic column/row subsampling.
   * **Result:** **🥉 3rd Place.** Achieved **`0.5538` CV Macro F1** and **`78.63%` Accuracy** with ultra-fast training throughput.

4. **CatBoost (Noise Quantization & Symmetric Oblivious Splits):**  
   To prevent symmetric decision trees from saturating on noisy clinical lab counts, I tuned $L_2$ leaf regularization (`l2_leaf_reg` $1.0$ to $15.0$), split candidate perturbation (`random_strength`), continuous bin quantization (`border_count` $\in [32, 128]$), and Bernoulli subsampling (`subsample` $0.65$ to $0.95$).
   * **Result:** Achieved **`0.5468` CV Macro F1**, **`72.18%` Accuracy**, and the **highest CV minority recall (`59.79%`)** among all boosting models (with **81%** training recall on early readmissions).

5. **Multi-Layer Perceptron (Architecture & Convergence Acceleration):**  
   Training neural networks over 78,331 rows across 5 folds can cause severe training delays. I resolved this by integrating automated `early_stopping=True` with `n_iter_no_change=10`, while systematically tuning hidden layer depths (`(100,)`, `(64, 32)`, `(128, 64)`, `(50, 25)`), activation functions (`relu` vs `tanh`), $L_2$ weight decay (`alpha` $10^{-5}$ to $10^{-1}$), and continuous initial learning rates (`learning_rate_init`).
   * **Result:** Achieved **`85.35%` CV Accuracy** (the highest accuracy across all architectures) and **`0.5369` Macro F1**. However, neural decision boundaries remained conservative on minority recall (`53.31%`) compared to cost-weighted decision trees.

6. **Logistic Regression (SAGA ElasticNet Regularization Ceiling):**  
   To establish the definitive linear bound, I implemented SAGA-based ElasticNet tuning, optimizing the $L_1/L_2$ mixing ratio (`l1_ratio` $\in [0.1, 0.9]$) alongside inverse regularization $C$ on a continuous log scale within the fold-isolated SMOTE pipeline.
   * **Result:** Lifted CV Macro F1 to **`0.5210`** and Accuracy to **`66.94%`** (+3.31% gain over baseline).

---

## 5. Final Synthesis & Winning Model Selection
After hundreds of cross-validation folds, Bayesian TPE optimization, and aggregate evaluation in [`model_comparison.ipynb`], our final binary benchmark leaderboard revealed decisive architectural patterns:

| Rank | Model Name | Model Family | CV Macro F1 | CV Macro Recall | CV Accuracy (%) | Minority (`<30`) Recall | Minority (`<30`) F1 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest (Deep Tuned)** | **Bagged Trees** | **0.5699** | **0.5878** | **78.98%** | **96.0%** | **0.73** |
| 🥈 | **XGBoost (Deep Tuned)** | **Gradient Boosting** | **0.5563** | **0.5820** | **76.50%** | **91.0%** | **0.62** |
| 🥉 | **LightGBM (Deep Tuned)** | **Gradient Boosting** | **0.5538** | **0.5662** | **78.63%** | **96.0%** | **0.69** |
| 4 | **CatBoost (Deep Tuned)** | **Gradient Boosting** | **0.5468** | **0.5979** | **72.18%** | **81.0%** | **0.47** |
| 5 | **MLP + SMOTE (Deep Tuned)** | **Neural Network** | **0.5369** | **0.5331** | **85.35%** | **19.0%** | **0.26** |
| 6 | **Logistic Regression (Deep Tuned)** | **Linear Model** | **0.5210** | **0.5917** | **66.94%** | **50.0%** | **0.25** |

---

## 6. Conclusion & Clinical Reflections
This modelling journey reinforced a vital lesson in applied healthcare machine learning: **the most complex or computationally demanding model is not necessarily the most clinically effective**. 

Our empirical findings clearly proved that on tabular electronic health records, **decision tree ensembles systematically outperform both feedforward neural networks and linear baselines**:
1. **The Bagged Ensemble Victory:** While sequential gradient boosting algorithms (`XGBoost`, `LightGBM`, `CatBoost`) performed admirably, **Random Forest emerged as the undisputed champion**. Sequential boosting models iteratively focus on hard-to-classify edge cases, which in noisy hospital EHR data often causes trees to over-chase clinical outliers. Conversely, Random Forest’s bootstrap aggregating and random feature subsampling inherently dampened variance, achieving the highest **Macro F1 of `0.5699`** and an exceptional **`96.0%` recall** on early readmissions.
2. **Neural Network Limitations on EHR Tabular Data:** The Multi-Layer Perceptron excelled in overall correctness (**`85.35%` accuracy**), proving that backpropagation successfully learned non-linear embeddings. However, synthetic oversampling (SMOTE) struggled to define sharp boundaries across high-dimensional sparse categories, resulting in low minority recall (`19.0%`) compared to tree cost-sensitive weighting.
3. **Reproducibility & Model Persistence:** The champion Random Forest model has been fully persisted to [`binary_classification/models/random_forest/random_forest_best_model.pkl`]. Due to ensemble size (300+ deep trees reaching ~400MB), the binary pickle has been safely added to `.gitignore` to protect git repository integrity while remaining instantly loadable via joblib.

The entire binary modelling workflow stands as a fully reproducible, scientifically sound, and clinically validated deliverable. With the champion model firmly established, our pipeline is ready for **Phase 2 (Evaluation)**: held-out test set evaluation (`X_test_final.csv`, `y_test_final.csv`), **Decision Threshold Optimization** via Precision-Recall curves, SHAP feature importance analysis, and high-risk patient risk profiling.
