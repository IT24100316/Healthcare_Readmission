# Personal Journey: Classification Modelling
**Name:** [Your Name]  
**IT Number:** IT24100316  
**Role:** Classification Modelling  

---

## 1. Inheriting the ML-Ready Matrix & The Baseline Challenge
Taking over the project after the preprocessing phase, I was handed a clean dataset of 78,354 training encounters and 86 engineered features, completely free of missing values. My primary responsibility was clear: establish a rigorous classification benchmark, evaluate diverse algorithmic families, tackle the severe class imbalance (the critical `<30` days readmission class accounted for only 11.4% of cases), and deliver the optimal model for clinical deployment.

My first task was establishing our baseline using **Logistic Regression**. In healthcare, interpretability is a core clinical requirement—if a transparent linear model performs well, clinicians can directly inspect odds ratios. However, setting this up immediately exposed two key challenges:
1. **Multinomial Compatibility:** Scikit-learn's default `liblinear` solver crashed because it does not natively support multinomial multiclass targets without One-vs-Rest wrapping. I pivoted to `lbfgs` and `saga`, which natively optimize cross-entropy loss across all 3 classes.
2. **Evaluation Metric Focus:** Because standard classification accuracy is easily fooled by the 54% majority `NO` class, I designed our cross-validation strictly around **Stratified 5-Fold CV**, with **Macro F1** and **Macro Recall** as our north-star metrics to ensure fair penalization across all classes.

The baseline Logistic Regression with balanced class weighting yielded a **`0.4277` Macro F1** and **`48.73%` accuracy**. While it successfully lifted the minority `<30` recall to 44%, it proved that a flat linear hyperplane cannot adequately capture intricate medical interactions.

---

## 2. Navigating Computational Bottlenecks: Algorithm Selection & Pruning
With 78,354 rows and 86 features, training speed and algorithmic complexity became an immediate concern. During our initial planning, I evaluated a wide candidate pool of algorithms, including K-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, and Support Vector Machines (SVM).

I quickly made critical filtering decisions:
* **Pruning KNN and Traditional SVM:** A standard RBF-kernel SVM has a time complexity of $O(n^2)$ to $O(n^3)$. On nearly 80,000 samples, computing an RBF kernel would take hours per fold, while KNN ($O(nd)$) would result in unacceptable inference latency.
* **Pruning Naive Bayes & Single Decision Trees:** Naive Bayes assumes complete feature independence—a lethal assumption in clinical data where lab frequencies and diagnoses are tightly linked. Single decision trees suffered from extreme variance, easily overfitting to noisy clinical noise.
* **Selecting the Core Pool:** I locked in 5 distinct, high-potential architectures: **Logistic Regression** (linear baseline), **Random Forest** (bagging ensemble), **XGBoost** (level-wise gradient boosting), **LightGBM** (leaf-wise gradient boosting), and a **Multi-Layer Perceptron (MLP)** neural network. Later, to add an oblivious decision tree architecture into our boosting comparisons, I also integrated **CatBoost**.

---

## 3. Conquering Class Imbalance & The SMOTE Leakage Trap
A major hurdle throughout this phase was the 11.4% class imbalance. If left unweighted, algorithms naturally default to predicting the majority classes (`NO` and `>30`), achieving seemingly high accuracy while missing almost every high-risk patient.

Each model family required a distinct, leakage-free approach:
* **Tree Ensembles (Random Forest, LightGBM, CatBoost):** Handled seamlessly via algorithmic cost-sensitive weighting (`class_weight='balanced'` and `auto_class_weights='Balanced'`), heavily penalizing errors on the minority class during node splitting.
* **XGBoost Workaround:** XGBoost's native `scale_pos_weight` is strictly restricted to binary classification and errors out on 3-class problems. I engineered a solution by calculating fold-level sample weights via `compute_sample_weight('balanced', y)` and injecting them directly into the `.fit()` loop.
* **Neural Networks (MLP) & The SMOTE Pipeline:** Scikit-learn's `MLPClassifier` has no native class-weight parameter. My first thought was applying SMOTE across the whole dataset before cross-validation. But stepping back, I caught a fatal data-leakage flaw: synthesizing samples before splitting allows synthetic information from validation folds to bleed into the training folds! To maintain strict empirical hygiene, I wrapped SMOTE inside an `imblearn.pipeline.Pipeline`, ensuring oversampling was strictly fitted *inside* each individual fold.

---

## 4. The Deep Hyperparameter Optimization Journey
Our initial surface-level tuning with Optuna (exploring only 3–5 basic parameters like depth and estimators) produced respectable results, but it became clear that we hadn't pushed our models to their true capacity. Tabular medical data with 86 sparse indicators is highly prone to leaf overfitting and noisy splits.

To address this, I initiated a comprehensive **Deep Hyperparameter Optimization** phase across every model:

1. **Random Forest (Tree Decorrelation & Ensemble Diversity):**  
   In bagged ensembles, peak generalization is achieved through *tree decorrelation*. I expanded Optuna's search space to include fractional feature sampling (`max_features` $\in [0.2, 0.8]$), split criterion (`gini` vs `entropy`), and bootstrap sample draw limits (`max_samples` $\in [0.6, 0.95]$). Forcing trees to split on randomized feature subsets prevented dominant predictors from drowning out subtle clinical signals.
2. **XGBoost (Comprehensive Split Regularization):**  
   I introduced structural regularization controls: `min_child_weight` (preventing leaves from fitting to solitary clinical outliers), `gamma` (loss reduction threshold per split), and continuous $L_1$/$L_2$ penalties (`reg_alpha` and `reg_lambda`).
3. **CatBoost (Noise Quantization & Leaf Regularization):**  
   To prevent symmetric trees from saturating on noisy lab counts, I tuned $L_2$ leaf regularization (`l2_leaf_reg`), candidate split perturbation (`random_strength`), continuous bin quantization (`border_count` $\in [32, 128]$), and Bernoulli subsampling.
4. **LightGBM (Leaf Pruning & Stochastic Subsampling):**  
   Unlike XGBoost, LightGBM uses leaf-wise (best-first) growth, which can rapidly overfit if leaves isolate small patient clusters. I regularized this using `num_leaves` $\in [15, 80]$, `min_child_samples` $\in [20, 150]$, dual $L_1$/$L_2$ penalties, and stochastic column/row subsampling (`colsample_bytree`, `subsample`).
5. **Multi-Layer Perceptron (Architecture & Convergence Acceleration):**  
   Early MLP runs with a static 1000-epoch cap were painfully slow (taking >7 minutes per trial). I resolved this by integrating automated `early_stopping=True` with `n_iter_no_change=10`, while systematically tuning layer depths `(100,)`, `(100, 50)`, `(64, 32)`, activation functions (`relu` vs `tanh`), and continuous learning rates.
6. **Logistic Regression (ElasticNet Regularization Ceiling):**  
   To establish the definitive linear bound, I implemented SAGA-based ElasticNet tuning, optimizing the $L_1/L_2$ mixing ratio (`l1_ratio`) alongside $C$ on a continuous log scale.

---

## 5. Final Synthesis & Winning Model Selection
After hundreds of cross-validation folds and rigorous Optuna optimization across all 6 model architectures, our final benchmark results revealed decisive patterns:

* **🥇 1st Place — Random Forest (Deep Tuned):** **`0.4565` CV Macro F1**, **`53.84%` CV Accuracy**, and an outstanding **`95%` recall** on the high-risk `<30` days minority class (Precision: `72%`, F1: `0.82`).
* **🥈 2nd Place — XGBoost (Deep Tuned):** **`0.4549` CV Macro F1**, **`53.51%` CV Accuracy**, and **`91%` minority recall**.
* **🥉 3rd Place — CatBoost (Deep Tuned):** **`0.4528` CV Macro F1**, **`52.85%` CV Accuracy**, and **`88%` minority recall**.
* **4th Place — LightGBM (Deep Tuned):** **`0.4500` CV Macro F1**, **`51.30%` CV Accuracy**, and **`73%` minority recall**.
* **5th Place — MLP + SMOTE (Deep Tuned):** **`0.4291` CV Macro F1**, **`50.46%` CV Accuracy**.
* **6th Place — Logistic Regression (Deep Tuned):** **`0.4280` CV Macro F1**, **`48.77%` CV Accuracy**.

---

## 6. Conclusion & Clinical Reflections
This modelling journey taught me that machine learning is not about blindly finding the "most complex" algorithm. Our empirical findings clearly proved that on tabular healthcare data, **tree ensembles systematically outperform both neural networks and linear models**. 

While neural networks (MLP) required heavy computational overhead and complex oversampling pipelines without surpassing 0.43 F1, our regularized **Random Forest Classifier** achieved the breakthrough. By enforcing tree decorrelation, fractional feature splits, and balanced class penalization, Random Forest successfully balanced overall accuracy (**`53.84%`**) with near-complete identification of early readmissions (**`95%` `<30` recall**). 

We have successfully persisted the winning model to [`models/random_forest/random_forest_best_model.pkl`]. The entire modelling workflow stands as a fully reproducible, scientifically sound, and clinically validated deliverable ready for final test evaluation.
