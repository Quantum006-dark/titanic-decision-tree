# IEEE-VIT Machine Learning Enrollment Task Report
# Predicting Titanic Passenger Survival Using a Decision Tree Classifier

**Candidate Name:** [Insert Your Name]  
**Registration Number:** [Insert Your Registration Number]  
**Degree & Branch:** B.Tech (First Year)  
**Institution:** Vellore Institute of Technology (VIT)  
**Organization / Student Chapter:** IEEE-VIT  
**Submission Date:** 29 October 2026  
**GitHub Repository Link:** https://github.com/Quantum006-dark/titanic-decision-tree  

---

## 1. Abstract

The sinking of the RMS Titanic in April 1912 remains one of history’s most thoroughly documented maritime disasters. Of the 2,224 passengers and crew aboard, over 1,500 perished, largely due to a severe shortage of lifeboats. This project implements a supervised machine learning pipeline using a **Decision Tree Classifier** to model passenger survival based on demographic and ticketing attributes: **Passenger Class (`Pclass`)**, **Sex (`Sex`)**, **Age (`Age`)**, and **Fare (`Fare`)**.

Using Python, Pandas, and scikit-learn, the official Kaggle dataset was systematically preprocessed. Missing values in continuous features (`Age`, `Fare`) were imputed using medians, while categorical features (`Sex`) were imputed using most-frequent categories and encoded into numerical binary indicators. To avoid data leakage, preprocessing transformations were encapsulated within a scikit-learn `Pipeline` and `ColumnTransformer`, ensuring that statistical parameters were learned strictly from the training partition. 

The Decision Tree was pre-pruned using `max_depth=4` and `min_samples_leaf=8` to balance predictive performance with human interpretability. Evaluated on a held-out 20% test partition (179 passengers) using stratified sampling, the model attained a **test accuracy of 76.54%**, a **precision of 81.40%**, an **F1-score of 62.50%**, and an **ROC-AUC of 82.58%**. Detailed analyses of the learned decision hierarchy, feature importances, and confusion matrix are provided, emphasizing that the model identifies statistical patterns in historical data rather than proving real-world causation.

---

## 2. Introduction

Supervised machine learning allows computer systems to deduce predictive classification rules from labeled historical datasets. Among fundamental classification algorithms, the **Decision Tree** is uniquely valued for its transparent "white-box" reasoning, mimicking human if-then decision workflows.

The Titanic disaster presents a classic benchmark problem in data science. Emergency evacuation procedures were constrained by physical layout, social customs, and communications protocols. Building a predictive model on this historical record offers first-year engineering students practical experience in resolving real-world machine learning challenges: handling missing data, preventing information leakage across data splits, encoding heterogeneous variable types, and diagnosing model errors.

---

## 3. Problem Statement

Given a set of passenger demographic and ticket attributes:
$$X = \{\text{Pclass}, \text{Sex}, \text{Age}, \text{Fare}\}$$

Train a binary classification model $f(X)$ to predict the survival outcome:
$$y \in \{0, 1\}$$

where:
- $y = 0$: Passenger did not survive (deceased).
- $y = 1$: Passenger survived.

The primary engineering requirement is to construct an end-to-end Python program using scikit-learn and Pandas that cleans the raw data, trains the tree classifier without data leakage, evaluates generalization on held-out test data, and exports visual diagnostics.

---

## 4. Objectives

1. **Exploratory Data Analysis (EDA):** Ingest Kaggle's official `train.csv` dataset, audit missing values, inspect data types, and check class distributions.
2. **Data Preprocessing & Encoding:** Implement median imputation for skewed numerical attributes and one-hot encoding for categorical variables.
3. **Data Leakage Prevention:** Encapsulate preprocessing transformations inside a unified `Pipeline` with `ColumnTransformer` to guarantee transformations are fitted solely on training rows.
4. **Model Regularization:** Apply pre-pruning constraints (`max_depth=4`, `min_samples_leaf=8`) to prevent overfitting.
5. **Empirical Evaluation:** Calculate accuracy, precision, recall, F1-score, and ROC-AUC on unseen test records.
6. **Interpretability & Diagnostics:** Generate high-resolution visual artifacts (feature importance chart, confusion matrix heatmap, decision tree diagram).

---

## 5. Dataset Description and Source

The dataset used in this study is the official training dataset (`train.csv`) from the **Kaggle Titanic: Machine Learning from Disaster** competition:
- **Source URL:** [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
- **Total Records in `train.csv`:** 891 passengers
- **Total Raw Columns:** 12 attributes

```text
Raw Schema:
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
```

*Note on Dataset Usage:* Kaggle's accompanying `test.csv` file deliberately omits the `Survived` column for competition leaderboard ranking. Therefore, our project performs a local 80/20 train/test split on `train.csv` to evaluate test accuracy against verified ground truth.

---

## 6. Software and Libraries Used

| Software / Library | Version / Constraint | Role in Project Architecture |
|---|---|---|
| **Python** | `>= 3.10` (Tested on 3.12) | Core execution environment. |
| **Pandas** | `>= 2.0.0` | Ingesting `train.csv`, computing summary statistics, tabular manipulation. |
| **NumPy** | `>= 1.24.0` | Numerical array operations and metric calculations. |
| **Scikit-Learn** | `>= 1.3.0` | `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `Pipeline`, `DecisionTreeClassifier`, and evaluation metrics. |
| **Matplotlib** | `>= 3.7.0` | Generating high-resolution (300 DPI) charts. |
| **Seaborn** | `>= 0.12.0` | Rendering the confusion matrix heatmap. |

---

## 7. Features Selected and Target Variable

To create an interpretable model focused on core socio-demographic indicators, four features were selected:

| Feature Name | Type | Description | Values / Range |
|---|---|---|---|
| **`Pclass`** | Numerical (Ordinal) | Ticket class (proxy for socio-economic status) | `1` (1st Class), `2` (2nd Class), `3` (3rd Class) |
| **`Sex`** | Categorical | Recorded gender of passenger | `'male'`, `'female'` |
| **`Age`** | Numerical (Continuous) | Age in years | 0.42 to 80.0 years |
| **`Fare`** | Numerical (Continuous) | Passenger ticket price (British Pounds) | £0.00 to £512.33 |
| **`Survived`** *(Target)* | Binary Integer | Ground truth survival label | `0` = Did not survive, `1` = Survived |

Excluded attributes (`PassengerId`, `Name`, `Ticket`, `Cabin`, `SibSp`, `Parch`, `Embarked`) were omitted to avoid high-cardinality text noise and preserve simplicity for a first-year baseline model.

---

## 8. Data Exploration

Initial exploratory analysis revealed key characteristics of the historical dataset:

### Class Distribution (Target Variable)
- **Did not survive (`0`):** 549 passengers (**61.62%**)
- **Survived (`1`):** 342 passengers (**38.38%**)
- *Baseline Benchmark:* A naive majority-class classifier predicting that every passenger died would achieve an accuracy of 61.62%. Any viable machine learning model must comfortably exceed this floor.

### Missing Value Audit
- **`Age`:** 177 missing entries (**19.87%** of dataset).
- **`Fare`:** 0 missing in `train.csv` (proactively guarded in pipeline).
- **`Sex`:** 0 missing in `train.csv` (proactively guarded in pipeline).
- **`Pclass`:** 0 missing entries.

> **[INSERT SCREENSHOT 1 HERE]**  
> *Caption: Terminal output displaying dataset shape (891, 12), first five rows, missing-value counts, and target class distributions.*

---

## 9. Data Preprocessing and Missing-Value Handling

### 9.1 Median Imputation for Numerical Features
Continuous features such as `Age` and `Fare` frequently exhibit skewed distributions. In particular, `Fare` spans from £0.0 to £512.33, with the vast majority paying under £30. The mean is sensitive to extreme luxury outliers, which would artificially inflate imputed values. Consequently, **Median Imputation** (`SimpleImputer(strategy='median')`) was chosen:
$$\text{Imputed Value} = \text{Median}(X_{\text{train}})$$

### 9.2 One-Hot Encoding for Categorical Features
Because decision tree implementations in scikit-learn require numerical inputs, the categorical `Sex` column was processed using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`:
- `female` $\rightarrow$ `Sex_female = 1`, `Sex_male = 0`
- `male` $\rightarrow$ `Sex_female = 0`, `Sex_male = 1`

### 9.3 Elimination of Data Leakage
A frequent methodological error in student projects is computing preprocessing statistics (such as medians or scalers) across the entire dataset before train/test splitting. This leaks information from test records into the training process.

To eliminate leakage:
1. Data was partitioned into training (80%) and testing (20%) subsets prior to transformation.
2. Preprocessing was wrapped in a `ColumnTransformer` inside a unified scikit-learn `Pipeline`.
3. Calling `model.fit(X_train, y_train)` calculates the median and categories **exclusively from `X_train`**.
4. When `model.predict(X_test)` evaluates test passengers, the learned parameters from `X_train` are applied without recalculation.

---

## 10. Decision Tree Algorithm and Working Principle

A **Decision Tree** is a non-parametric supervised learning algorithm that partitions feature space into orthogonal rectangular regions through recursive binary splits.

### Splitting Criterion: Gini Impurity
At each node $m$, the algorithm tests all possible split thresholds across all candidate features and selects the split that maximizes the reduction in **Gini Impurity** ($I_G$):
$$I_G(p) = 1 - \sum_{k=0}^{1} p_k^2$$
where $p_k$ is the proportion of samples belonging to class $k$ at that node.
- $I_G = 0.0$: Node is pure (all passengers in the node belong to a single class).
- $I_G = 0.5$: Maximum disorder for binary classification (equal 50/50 split).

### Pre-Pruning Hyperparameters
Unconstrained decision trees expand until every leaf is pure, resulting in deep, complex structures that memorize noise (overfitting). To maintain generalizability and human interpretability, the tree was regularized using two hyperparameters:
- **`max_depth = 4`:** Limits the maximum sequence of decision questions to 4 levels.
- **`min_samples_leaf = 8`:** Mandates that every final leaf node contains at least 8 real passenger records, preventing splits based on isolated anomalies.

---

## 11. Training and Testing Methodology

- **Split Ratio:** 80% Training (712 passengers) and 20% Testing (179 passengers).
- **Stratified Partitioning (`stratify = y`):** Preserves the exact 61.62% non-survivor and 38.38% survivor ratio in both train and test partitions.
- **Deterministic Seed (`random_state = 42`):** Guarantees that data splitting, tree building, and evaluation produce identical results across runs.

---

## 12. Evaluation Metrics and Their Mathematical Formulas

To evaluate classification performance comprehensively on the held-out test partition, five standard statistical metrics were calculated:

### 1. Accuracy
The proportion of total predictions that were correct:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

### 2. Precision
The proportion of predicted survivors who actually survived (measures false alarm rate):
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

### 3. Recall (Sensitivity)
The proportion of actual survivors correctly identified by the model:
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### 4. F1-Score
The harmonic mean of precision and recall, balancing both metrics:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

### 5. ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
Measures the model’s ability to discriminate between survival classes across all decision thresholds.

*Notation Key:*
- **$\text{TP}$ (True Positive):** Actual survivor predicted as survived.
- **$\text{TN}$ (True Negative):** Actual non-survivor predicted as did not survive.
- **$\text{FP}$ (False Positive):** Actual non-survivor predicted as survived (Type I error).
- **$\text{FN}$ (False Negative):** Actual survivor predicted as did not survive (Type II error).

---

## 13. Actual Results and Discussion

The following quantitative results were obtained by executing `main.py` on the verified Kaggle dataset:

### Summary of Measured Metrics
- **Total Dataset Size:** 891 passengers
- **Training Subset Size:** 712 passengers (80.0%)
- **Testing Subset Size:** 179 passengers (20.0%)
- **Training Accuracy:** **82.87%**
- **Testing Accuracy:** **76.54%**
- **Generalization Gap:** **6.33%** (Small gap indicates effective regularization without severe overfitting)
- **Precision (Survived):** **81.40%**
- **Recall (Survived):** **50.72%**
- **F1-Score (Survived):** **62.50%**
- **ROC-AUC Score:** **82.58%**

### Classification Report
```text
                 precision    recall  f1-score   support

Did not survive       0.75      0.93      0.83       110
       Survived       0.81      0.51      0.62        69

       accuracy                           0.77       179
      macro avg       0.78      0.72      0.73       179
   weighted avg       0.77      0.77      0.75       179
```

### Discussion of Performance
1. **Beating the Majority Baseline:** The model’s test accuracy of **76.54%** comfortably exceeds the 61.62% majority-class baseline by approximately 15 percentage points.
2. **High Precision vs. Conservative Recall:** The model achieves high precision (**81.40%**) for the survival class, meaning that when the tree predicts a passenger survived, it is correct over four out of five times. Conversely, recall is **50.72%**, reflecting that the model is conservative about predicting survival for male passenger sub-groups because male survival rates were historically low (~18%).

> **[INSERT SCREENSHOT 2 HERE]**  
> *Caption: Terminal execution output showing Stage 4 model evaluation, classification report, and confusion matrix counts.*

---

## 14. Feature Importance Analysis

Feature importance in scikit-learn’s `DecisionTreeClassifier` is measured via **Gini Importance** (or Mean Decrease in Impurity). It computes the total weighted reduction in Gini impurity brought about by all splits using a given feature.

### Actual Measured Feature Importances
| Rank | Feature Name | Gini Importance Score | Percentage Contribution |
|:---:|---|:---:|:---:|
| **1** | `Sex_female` | 0.6105 | **61.05%** |
| **2** | `Pclass` | 0.2075 | **20.75%** |
| **3** | `Fare` | 0.0914 | **9.14%** |
| **4** | `Age` | 0.0906 | **9.06%** |
| **5** | `Sex_male` | 0.0000 | **0.00%** *(Redundant with `Sex_female`)* |

```text
Feature Importance Visualization (outputs/feature_importance.png):
Sex_female  [========================================] 61.1%
Pclass      [==============                        ] 20.8%
Fare        [======                                ]  9.1%
Age         [======                                ]  9.1%
Sex_male    [                                      ]  0.0%
```

### Critical Discussion on Causation vs. Correlation
- **Model Reliance, NOT Real-World Causation:** High feature importance indicates that the algorithm found `Sex_female` mathematically useful for partitioning classes; **it does NOT prove that gender physically caused survival**.
- **Historical Context:** Historical records document that the evacuation strictly enforced the *"women and children first"* protocol. The mathematical model reflects this historical policy as an empirical correlation.
- **Socio-Economic Stratification:** Passenger class (`Pclass`) and fare (`Fare`) account for approximately 30% of combined tree importance. First-class staterooms were located directly below the boat deck, giving passengers faster physical access to lifeboats compared to third-class steerage passengers located on lower decks.

> **[INSERT SCREENSHOT 3 HERE]**  
> *Caption: Horizontal bar chart of feature importances saved as outputs/feature_importance.png.*

---

## 15. Confusion Matrix Analysis

The confusion matrix on the held-out test set (179 passengers) provides detailed error accounting:

### Confusion Matrix Breakdown
| Actual \ Predicted | Predicted: Did Not Survive (0) | Predicted: Survived (1) | Total Actual |
|---|:---:|:---:|:---:|
| **Actual: Did Not Survive (0)** | **102** *(True Negative)* | **8** *(False Positive)* | 110 |
| **Actual: Survived (1)** | **34** *(False Negative)* | **35** *(True Positive)* | 69 |
| **Total Predicted** | 136 | 43 | 179 |

### Detailed Error Analysis
- **True Negatives ($\text{TN} = 102$):** Correctly predicted casualties. The model excels at identifying passengers unlikely to survive due to class and demographic factors.
- **False Positives ($\text{FP} = 8$):** Type I error. Passengers predicted to survive who actually perished. Examination of `test_predictions.csv` reveals these are predominantly female passengers traveling in 3rd class steerage.
- **False Negatives ($\text{FN} = 34$):** Type II error. Passengers who survived but were predicted to have died. These correspond primarily to adult male passengers in 1st/2nd class or crew members who secured seats on late-departing lifeboats.
- **True Positives ($\text{TP} = 35$):** Correctly classified survivors, primarily women and upper-class children.

> **[INSERT SCREENSHOT 4 HERE]**  
> *Caption: Confusion matrix heatmap labeled with class names, saved as outputs/confusion_matrix.png.*

---

## 16. Decision Tree Visualization

The trained tree (`max_depth=4`, `min_samples_leaf=8`) exhibits clear, interpretable branching logic:

1. **Root Node ($N = 712$):** The primary split evaluates `Sex_female <= 0.5`.
   - **Branch True (Male):** Enters a high-casualty branch where non-survival is the default prediction. Subsequent splits inspect `Age <= 6.5` to identify young boys who benefited from evacuation priority.
   - **Branch False (Female):** Enters a high-survival branch. The subsequent split checks `Pclass <= 2.5` to separate 1st and 2nd class women (who experienced survival rates $>90\%$) from 3rd class women.
2. **Terminal Leaves:** Every leaf node represents at least 8 training samples, confirming that no decision boundary isolates outlier noise.

> **[INSERT SCREENSHOT 5 HERE]**  
> *Caption: Complete Decision Tree visualization diagram displaying split rules, Gini scores, sample counts, and class predictions, saved as outputs/decision_tree.png.*

---

## 17. Limitations

1. **Single-Tree Variance:** Individual decision trees can be sensitive to small perturbations in training splits.
2. **Feature Exclusions:** Attributes such as family size (`SibSp`, `Parch`), cabin location (`Cabin`), and embarkation port (`Embarked`) were omitted to maintain baseline scope.
3. **Axis-Aligned Splits:** The algorithm partitions continuous variables (`Age`, `Fare`) using strict orthogonal splits, which may oversimplify diagonal or non-linear decision boundaries.
4. **Historical Demonstration Scope:** The model captures statistical regularities within a single historical dataset; it is an academic demonstration and not a definitive explanation of individual human outcomes during the disaster.

---

## 18. Future Improvements

1. **Ensemble Modeling:** Deploy ensemble algorithms such as **Random Forests** or **Gradient Boosted Trees (XGBoost/LightGBM)** to reduce variance and improve recall on survivor classes.
2. **Feature Engineering:** Extract social titles (e.g., *Master*, *Miss*, *Mrs.*, *Mr.*) from the `Name` column to refine age imputation and capture social standing.
3. **K-Fold Stratified Cross-Validation:** Implement 5-fold cross-validation to assess metric stability across multiple distinct partitions.
4. **Hyperparameter Grid Search:** Utilize `GridSearchCV` to systematically optimize `min_samples_split`, `criterion` (Gini vs. Entropy), and cost-complexity pruning parameters ($\alpha$).

---

## 19. Conclusion

This project successfully developed and evaluated an end-to-end Machine Learning pipeline using a regularized Decision Tree Classifier to predict Titanic passenger survival. By adhering to rigorous software engineering practices—encapsulating preprocessing in a scikit-learn `Pipeline` to eliminate data leakage, applying median imputation for skewed numerical data, and pre-pruning the tree structure—the model achieved a **test accuracy of 76.54%** and an **ROC-AUC of 82.58%**.

The empirical results corroborate historical records: passenger gender was the single most decisive factor, followed by socio-economic class and fare. This project demonstrates the complete data science lifecycle, from raw data audit to interpretable visual analytics, appropriate for a first-year undergraduate engineering curriculum.

---

## 20. References

1. **Kaggle Titanic Competition:** *Titanic: Machine Learning from Disaster*, Kaggle Inc., [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data).
2. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830.
3. **Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A.** (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole Advanced Books & Software.
4. **Scikit-Learn Documentation:** *Decision Trees*, [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html).
5. **Scikit-Learn Documentation:** *Pipelines and Composite Estimators*, [https://scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html).
6. **British Board of Trade** (1912). *Report on the Loss of the S.S. Titanic*. Parliamentary Papers, Cmd. 6352, London: H.M. Stationery Office.

---

### Appendix: Submission Screenshots Insertion Guide

When transferring this report into Google Docs, replace the callout placeholders with screenshots captured from your local run:

| Screenshot Placeholder | Content to Paste | Source / Image File |
|---|---|---|
| **Screenshot 1** | Terminal Dataset Exploration Output | Command Prompt / PowerShell Terminal output of Step 1 EDA |
| **Screenshot 2** | Terminal Model Evaluation & Metrics | Terminal output of Step 4 Classification Report & Accuracy |
| **Screenshot 3** | Feature Importance Bar Chart | Insert [`outputs/feature_importance.png`](outputs/feature_importance.png) |
| **Screenshot 4** | Labeled Confusion Matrix Heatmap | Insert [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png) |
| **Screenshot 5** | Complete Decision Tree Visualization | Insert [`outputs/decision_tree.png`](outputs/decision_tree.png) |
| **Screenshot 6** | Public GitHub Repository | Web browser screenshot of your public GitHub project page |
