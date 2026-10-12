# Personal Journey: Classification Modelling (Binary Track)
**Name:** Weerasingha P.M.P.L  
**IT Number:** IT24100316  
**Role:** Classification Modelling (Binary Track: 30-Day Early Readmission)  

---

## 1. Inheriting the ML-Ready Binary Matrix & The Baseline Challenge
Following data preprocessing, I inherited a structured, preprocessed dataset comprising 78,331 training encounters and 86 engineered clinical features from `binary_classification/dataset_processed/`. All records were encoded, standardized using robust scaling, and verified to contain zero missing values. The held-out test dataset comprises 19,491 patient encounters.

The project formulation defines a binary target for 30-day readmission risk:
* **Positive Class (1):** Early readmission within 30 days (`<30`) — 9,004 training encounters (11.49%).
* **Negative Class (0):** No early readmission (`NO` or `>30` combined) — 69,327 training encounters (88.51%).

My primary responsibility was to establish an empirical classification benchmark, evaluate diverse algorithmic families, address the 1 : 7.7 class imbalance, conduct hyperparameter optimization, and select the leading candidate model architecture for subsequent evaluation on the holdout test set.

I established our baseline using **Logistic Regression**. In clinical informatics, interpretability is an important consideration, as a linear model allows direct inspection of odds ratios and feature weights. 

The baseline model was constructed using an `imblearn.pipeline.Pipeline` incorporating **SMOTE (`sampling_strategy=0.30`)** and **`LogisticRegression(class_weight='balanced')`** evaluated via 5-fold cross-validation. This addressed the class imbalance by combining synthetic minority oversampling with inverse class weighting. Because standard classification accuracy is easily distorted by majority-class dominance (where predicting all negative yields 88.51% accuracy), the cross-validation protocol prioritized **Macro F1** and **Macro Recall** to evaluate performance across both classes.

The baseline Logistic Regression pipeline produced the following metrics recorded in `models/logistic_regression/results/baseline_metrics.txt`:
* **Mean CV Macro F1:** 0.5116
* **Mean CV Macro Recall:** 0.6085
* **Mean CV Accuracy:** 63.55%

While the baseline model successfully detected early readmissions, the performance ceiling suggested that a linear decision boundary could not sufficiently capture multi-variable interactions in chronic diabetic care.

---

## 2. Computational Feasibility & Algorithm Selection
With 78,331 training rows and 86 features, evaluating models across 5 folds means that in each fold, approximately 62,665 encounters are used for training and 15,666 for validation. Training throughput and algorithmic scalability were critical operational considerations. During initial scoping, a candidate pool of algorithms was evaluated, including k-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, and Support Vector Machines (SVM).

I applied specific filtering criteria based on computational complexity and dataset characteristics:
* **Exclusion of KNN and Kernel SVM:** A standard RBF-kernel SVM exhibits computational complexity between O(n^2) and O(n^3). On over 60,000 training instances per fold, computing kernel matrices causes severe computational bottlenecks, while KNN (O(nd)) introduces high latency during prediction.
* **Exclusion of Naive Bayes & Single Decision Trees:** Naive Bayes strictly assumes conditional feature independence—an assumption violated by clinical electronic health records where laboratory test counts, medication regimens, and secondary diagnoses are correlated. Single decision trees exhibited high variance, making them prone to overfitting on sparse clinical categories.
* **Selection of Six Core Candidate Architectures:** I selected six distinct architectures spanning four foundational learning paradigms:
  1. *Linear Models:* **Logistic Regression** (L1, L2, and ElasticNet regularization via the SAGA solver).
  2. *Bagging Ensembles:* **Random Forest** (decorrelated bootstrap aggregating).
  3. *Boosting Ensembles:* **XGBoost** (exact gradient boosting), **LightGBM** (histogram leaf-wise boosting), and **CatBoost** (symmetric oblivious decision trees).
  4. *Neural Architectures:* **Multi-Layer Perceptron (MLP)** feedforward neural network.

---

## 3. Addressing Class Imbalance & Cross-Validation Methodology

### 3.1. Imbalance Mitigation Strategies
The 11.49% minority prevalence was addressed using paradigm-specific techniques:
* **Tree Ensembles (Random Forest, LightGBM, CatBoost):** Handled via algorithmic cost-sensitive weighting (`class_weight='balanced'` and `auto_class_weights='Balanced'`), penalizing minority false negatives directly within split objective calculations.
* **XGBoost Native Cost Weighting:** Binary classification enabled the application of XGBoost's native `scale_pos_weight = 7.70` (calculated as 69,327 / 9,004), weighting gradient steps inversely proportional to class frequencies.
* **Linear & Neural Models (Logistic Regression, MLP) via Resampling Pipelines:** Because standard neural networks and linear models lack native gradient-level cost weighting in this configuration, SMOTE (`sampling_strategy=0.30`) was integrated inside an `imblearn.pipeline.Pipeline`. This ensured that synthetic minority sample generation was isolated strictly to training folds and never applied to validation folds.

### 3.2. Methodological Limitation in Cross-Validation
*Important Methodological Note:* The initial train/test partition (78,331 training vs 19,491 test encounters) was strictly grouped by `patient_nbr` to ensure no individual patient appeared in both training and test sets. However, within the training dataset, `patient_nbr` was dropped prior to model training. Consequently, internal cross-validation across all six models utilized standard **Stratified 5-Fold Cross-Validation** (`StratifiedKFold`) rather than `GroupKFold`. Multiple encounters from the same patient within the training partition may therefore appear across different training and validation folds. This constitutes a recognized methodological limitation of the reported cross-validation metrics, though the holdout test set remains strictly isolated at the patient level.

---

## 4. Hyperparameter Optimization Framework
To evaluate model capacity, hyperparameter optimization was executed using Optuna (Tree-structured Parzen Estimator, TPE):

1. **Random Forest (Ensemble Diversity & Decorrelation):**  
   The search space explored feature subsampling (`max_features` in ['sqrt', 'log2', 0.4, 0.6]), split criteria (`gini`, `entropy`), maximum tree depth (`max_depth` between 6 and 18), leaf sample constraints (`min_samples_leaf` between 3 and 12), and bootstrap sample limits (`max_samples` between 0.70 and 0.95).  
   * **Recorded Results:** Reached **0.5699 CV Macro F1**, **0.5878 CV Macro Recall**, and **78.98% CV Accuracy**. On the full training set fit, the model demonstrated 96.0% minority recall (F1: 0.73).

2. **XGBoost (Regularized Gradient Splitting):**  
   Configured with `scale_pos_weight = 7.70` and `eval_metric='logloss'`. Tuned parameters included minimum child weight (`min_child_weight` between 1 and 12), split threshold (`gamma` between 0.0 and 5.0), L1 sparsity (`reg_alpha` between 1e-4 and 10.0), L2 shrinkage (`reg_lambda` between 1e-4 and 10.0), and subsampling rates (`colsample_bytree`, `subsample`).  
   * **Recorded Results:** Achieved **0.5563 CV Macro F1**, **0.5820 CV Macro Recall**, and **76.50% CV Accuracy** (Full training fit minority recall: 91.0%).

3. **LightGBM (Leaf-Wise Histogram Growth):**  
   Regularized via maximum leaf count (`num_leaves` between 15 and 80), child sample constraints (`min_child_samples` between 20 and 150), dual L1/L2 penalties (`reg_alpha`/`reg_lambda`), and feature/data subsampling.  
   * **Recorded Results:** Achieved **0.5538 CV Macro F1**, **0.5662 CV Macro Recall**, and **78.63% CV Accuracy** (Full training fit minority recall: 96.0%).

4. **CatBoost (Oblivious Trees & Quantization):**  
   Tuned leaf L2 penalties (`l2_leaf_reg` between 1.0 and 15.0), random perturbation strength (`random_strength`), continuous bin quantization (`border_count` between 32 and 128), and subsampling rates.  
   * **Recorded Results:** Achieved **0.5468 CV Macro F1**, **0.5979 CV Macro Recall**, and **72.18% CV Accuracy** (Full training fit minority recall: 81.0%).

5. **Multi-Layer Perceptron (Architecture & Regularization):**  
   Tuned using hidden layer structures ((100,), (64, 32), (128, 64), (50, 25)), activation functions (`relu`, `tanh`), L2 weight decay (`alpha` between 1e-5 and 1e-1), learning rates, and early stopping (`n_iter_no_change=10`).  
   * **Recorded Results:** Achieved **85.35% CV Accuracy** and **0.5369 CV Macro F1**, with a CV Macro Recall of **0.5331** (Full training fit minority recall: 19.0%). While 85.35% was the highest raw accuracy among the models, it remains below the 88.51% majority-class-only benchmark, and reflects conservative minority-class identification rather than superior discrimination.

6. **Logistic Regression (SAGA ElasticNet Optimization):**  
   Tuned the L1/L2 mixing parameter (`l1_ratio` between 0.1 and 0.9) and inverse regularization parameter C on a log scale within the fold-isolated SMOTE pipeline.  
   * **Recorded Results:** Achieved **0.5210 CV Macro F1**, **0.5917 CV Macro Recall**, and **66.94% CV Accuracy** (Full training fit minority recall: 50.0%).

---

## 5. Model Benchmarking & Empirical Synthesis
The cross-validation performance across all six architectures was compiled and evaluated in `binary_classification/models/common/model_comparison.ipynb`.

The table below presents the verified 5-fold cross-validation metrics alongside training-set fit indicators:

| Rank | Model Architecture | Learning Paradigm | Mean CV Macro F1 | Mean CV Macro Recall | Mean CV Accuracy (%) | Training Set Minority Recall* | Training Set Minority F1* |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Tuned)** | Bagged Decision Trees | **0.5699** | **0.5878** | **78.98%** | 96.0% | 0.73 |
| 2 | **XGBoost (Tuned)** | Gradient Boosting | **0.5563** | **0.5820** | **76.50%** | 91.0% | 0.62 |
| 3 | **LightGBM (Tuned)** | Histogram Gradient Boosting | **0.5538** | **0.5662** | **78.63%** | 96.0% | 0.69 |
| 4 | **CatBoost (Tuned)** | Oblivious Decision Trees | **0.5468** | **0.5979** | **72.18%** | 81.0% | 0.47 |
| 5 | **MLP + SMOTE (Tuned)** | Feedforward Neural Network | **0.5369** | **0.5331** | **85.35%** | 19.0% | 0.26 |
| 6 | **Logistic Regression (Tuned)** | Linear ElasticNet | **0.5210** | **0.5917** | **66.94%** | 50.0% | 0.25 |

*\*Note: Training Set Minority Recall and F1 reflect model fit on the entire training set (78,331 encounters) and must not be interpreted as out-of-fold cross-validation scores.*

---

## 6. Conclusion & Handoff to Evaluation Phase
This comparative analysis highlights several key technical observations:
1. **Ensemble Performance on Tabular Clinical Data:** Tree ensembles demonstrated higher balanced Macro F1 scores compared to the linear baseline and neural network. Random Forest achieved the highest recorded **CV Macro F1 (0.5699)** and balanced **CV Macro Recall (0.5878)** at **78.98% accuracy**, making it the leading candidate model for further evaluation.
2. **Evaluation Metric Considerations:** While the Multi-Layer Perceptron obtained the highest nominal accuracy (**85.35%**), its low minority recall (19.0% on full training fit) demonstrates that high accuracy in an imbalanced setting can reflect majority-class bias rather than balanced predictive capability.


**Handoff to Phase 2 (Evaluation):**  
The Random Forest architecture was selected as the candidate model for single-pass evaluation on the untouched holdout test set (19,491 encounters). This subsequent phase evaluates true generalization on unseen patient data, examines decision threshold calibration to assess clinical trade-offs between precision and recall, and performs feature importance extraction for clinical decision support.
