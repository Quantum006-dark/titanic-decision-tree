


## 1. Project Overview

Welcome to the **Titanic Passenger Survival Prediction** project! This repository contains a complete, beginner-friendly machine-learning pipeline built with Python, Pandas, and scikit-learn. 

The goal is to analyze historical passenger data from the tragic 1912 sinking of the RMS Titanic and build a predictive model that determines whether an individual passenger survived or perished based on their ticket class, gender, age, and fare.

This project was built to illustrate core data science principles from scratch: exploratory data analysis, handling missing values, encoding categories, preventing data leakage with scikit-learn Pipelines, training an interpretable decision tree, and evaluating model predictions on unseen data.

---

## 2. Problem Statement & Objectives

### Problem Statement
On April 15, 1912, the RMS Titanic collided with an iceberg and sank. With an insufficient number of lifeboats, over 1,500 of the 2,224 passengers and crew lost their lives. While survival contained elements of chance, certain passenger groups (such as women, children, and first-class passengers) had markedly higher survival rates. 

Our task is to build a classification model:
$$\text{Input Features } (Pclass, Sex, Age, Fare) \longrightarrow \text{Target Prediction } (Survived: 0 \text{ or } 1)$$

### Key Objectives
1. **Explore the data:** Understand data distributions, types, and missing fields using Pandas.
2. **Clean and preprocess correctly:** Impute missing numerical data using medians and encode categorical variables without data leakage.
3. **Train an interpretable Decision Tree:** Constrain tree growth (`max_depth=4`, `min_samples_leaf=8`) to prevent overfitting.
4. **Evaluate thoroughly:** Use accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix on a held-out test set.
5. **Generate clear visual diagnostics:** Export readable charts for feature importance, confusion matrix, and tree structure.

---

## 3. Tools and Libraries Used

| Tool / Library | Purpose in this Project |
|---|---|
| **Python (>= 3.10)** | Programming language for the pipeline. |
| **Pandas (`pandas>=2.0.0`)** | Loading CSV files, inspecting rows, computing summary statistics. |
| **NumPy (`numpy>=1.24.0`)** | Numerical arrays, probability calculations, and sorting. |
| **Scikit-learn (`scikit-learn>=1.3.0`)** | Preprocessing (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`), model training (`DecisionTreeClassifier`), and metrics (`accuracy_score`, `confusion_matrix`, etc.). |
| **Matplotlib (`matplotlib>=3.7.0`)** | Plotting high-resolution charts and saving them to PNG images. |
| **Seaborn (`seaborn>=0.12.0`)** | Visualizing the confusion matrix heatmap with custom color schemes. |

All library requirements are pinned in [`requirements.txt`](requirements.txt).

---

## 4. Dataset Source & Setup Instructions

This project uses the official training dataset from the **Kaggle Titanic: Machine Learning from Disaster** competition.

### Step-by-Step Instructions to Download `train.csv`:
1. Open your web browser and navigate to:  
   👉 [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
2. Sign in to your Kaggle account (create a free account if you don't have one).
3. If prompted, click **"I Understand and Accept"** to agree to the competition rules.
4. In the **Data Files** pane, click on **`train.csv`** and download it (or click **Download All**, which downloads `titanic.zip`, and extract `train.csv`).
5. Copy or move **`train.csv`** directly into the project folder next to `main.py`:
   ```text
   titanic_decision_tree_prototype/
   ├── main.py
   ├── train.csv         <-- Place here
   ├── requirements.txt
   └── README.md
   ```

> **Why not `test.csv`?**  
> Kaggle's `test.csv` does not include the `Survived` column (because it is used for scoring competition submissions on Kaggle's leaderboard). For model training and evaluation, we split `train.csv` into training (80%) and testing (20%) subsets so we can test the model on known ground truth.

---

## 5. Selected Features & Target Variable

To keep the model interpretable and focused, we use four primary attributes:

| Column Name | Data Type | Description | Values / Range |
|---|---|---|---|
| **`Pclass`** | Numerical (Ordinal) | Ticket class / socio-economic status proxy | `1` = 1st Class, `2` = 2nd Class, `3` = 3rd Class |
| **`Sex`** | Categorical | Passenger gender | `'male'` or `'female'` |
| **`Age`** | Numerical (Continuous) | Passenger age in years | Fractional if < 1 year; up to 80 years |
| **`Fare`** | Numerical (Continuous) | Ticket fare paid (in British Pounds) | £0.00 to £512.33 |
| **`Survived`** *(Target)* | Binary Integer | Survival outcome to predict | `0` = Did not survive, `1` = Survived |

Other columns (such as `Name`, `Ticket`, and `Cabin`) are omitted in this baseline model because they contain high-cardinality text or high missingness that would complicate a beginner decision tree.

---

## 6. Preprocessing: Missing Values & One-Hot Encoding

Raw real-world datasets are rarely ready for machine learning out-of-the-box. Here is how our script prepares the data:

### Handling Missing Values
- **Numeric Features (`Age`, `Fare`, `Pclass`):**
  - In `train.csv`, `Age` has 177 missing entries (~19.9%).
  - We use **Median Imputation** (`SimpleImputer(strategy='median')`).
  - *Why median instead of mean?* Ticket prices and ages can be skewed by extreme values (e.g., luxury suites cost over £500, while most tickets cost under £30). The mean is pulled up by extreme outliers, whereas the median reflects the true middle value and is much more stable.
- **Categorical Feature (`Sex`):**
  - We use **Most-Frequent Imputation** (`SimpleImputer(strategy='most_frequent')`) as a safe fallback.

### One-Hot Encoding
Computers and machine-learning algorithms cannot perform arithmetic on text strings like `"male"` and `"female"`. 
- We apply `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to the `Sex` column.
- This creates two numeric binary indicator columns:
  - `Sex_female`: `1` if female, `0` otherwise.
  - `Sex_male`: `1` if male, `0` otherwise.

### Preventing Data Leakage with Pipelines
A classic beginner mistake is calculating medians on the *entire* dataset before splitting it into train and test sets. Doing so causes **data leakage**, meaning information from the test set leaks into the training process, leading to falsely optimistic performance.
- In `main.py`, preprocessing is placed inside a `ColumnTransformer` and bundled with the model in a scikit-learn `Pipeline`.
- When `model.fit(X_train, y_train)` runs, imputers compute medians **exclusively from `X_train`**.
- When `model.predict(X_test)` runs, it applies those training medians to the test set without learning anything new from `X_test`.

---

## 7. Train / Test Splitting

Before any model training occurs, the dataset is split:
- **80% Training Set (712 passengers):** The model studies these examples to learn the rules.
- **20% Testing Set (179 passengers):** Held out as a final exam to measure how well the model predicts new, unseen data.

Two critical settings are used:
1. **`random_state=42`**: Fixes the random number generator so that the exact same split is produced every time the program is run, making the results 100% reproducible.
2. **`stratify=y`**: In the Titanic dataset, ~61.6% of passengers died and ~38.4% survived. Stratification ensures both the 80% train set and 20% test set have the exact same 61.6% to 38.4% ratio, preventing class bias.

---

## 8. The Decision Tree Algorithm Explained Simply

A **Decision Tree** is a flowchart-like algorithm that makes decisions by asking a sequence of simple YES / NO questions about feature values.

### How Does it Work?
1. **Root Node:** The algorithm starts at the top with all training passengers.
2. **Finding the Best Split:** It tests every possible question (e.g., *"Is Sex female?"*, *"Is Pclass <= 2.5?"*, *"Is Fare <= 10.5?"*) and selects the split that best separates survivors from casualties.
3. **Gini Impurity:** To pick the best split, the algorithm measures **Gini Impurity**:
   - Gini = $0.0$: A perfectly "pure" node where everyone either survived or died.
   - Gini = $0.5$: A completely mixed node (50% survived, 50% died).
   - The tree always chooses the question that reduces Gini impurity the most.
4. **Pruning (Limiting Tree Growth):**
   - Left unconstrained, a tree will keep splitting until every single passenger is isolated in their own leaf node. This is called **overfitting** (memorizing the training examples rather than learning general patterns).
   - In `main.py`, we set `max_depth=4` (the tree can ask at most 4 questions in a row) and `min_samples_leaf=8` (each decision must represent at least 8 passengers). This keeps the model small, readable, and generalizable.

---

```

---

## 10. Running the Project

Once dependencies are installed and `train.csv` is in the folder, run:

```powershell
python main.py
```

### Specifying a Custom Path
If your `train.csv` is stored in another folder (for example, in your Downloads folder), run:

```powershell
python main.py --data "C:\Users\SIDDHANT\Downloads\train.csv"
```

### Running the Automated Test Suite
To verify that all 12 reviewer checks pass without error, run:

```powershell
python test_suite.py
```

---

##  Description of Generated Output Files

When `main.py` finishes, an `outputs/` folder is automatically created containing:

```text
outputs/
├── run_summary.txt         # Text summary of actual accuracy, sizes, and rankings
├── metrics_summary.json    # Machine-readable JSON metrics for grading
├── feature_importance.png  # Horizontal bar chart of Gini importance
├── feature_importance.csv  # Numerical table of feature importance scores
├── confusion_matrix.png    # Labeled heatmap of prediction outcomes
├── decision_tree.png       # Complete visual diagram of the decision tree
└── test_predictions.csv    # Test set with actual vs. predicted survival
```

1. **`run_summary.txt`**: Complete, human-readable terminal output and performance metrics.
2. **`feature_importance.png`**: High-resolution (300 DPI) horizontal bar chart ranking the features by Gini importance.
3. **`confusion_matrix.png`**: Heatmap displaying True Negatives, False Positives, False Negatives, and True Positives labeled with `"Did not survive"` and `"Survived"`.
4. **`decision_tree.png`**: Diagram displaying every decision node, question threshold, Gini impurity, sample count, and predicted class.
5. **`test_predictions.csv`**: A CSV file containing test passengers, their feature values, their actual survival, model predictions, and calculated probabilities.

---

##  Interpreting the Model Results

Here are the **actual results from running `main.py`** on the official 80/20 split:

### Summary of Actual Metrics
- **Dataset Size:** 891 total rows (712 train, 179 test)
- **Training Accuracy:** **82.87%**
- **Testing Accuracy:** **76.54%**
- **Precision (Survived):** **81.40%**
- **Recall (Survived):** **50.72%**
- **F1-Score (Survived):** **62.50%**
- **ROC-AUC Score:** **82.58%**

### What Each Metric Means:
1. **Accuracy (76.54%):**
   - The percentage of total test passengers classified correctly ($137 / 179 = 76.54\%$).
   - *Baseline comparison:* A naive guess of "everyone died" would yield 61.62%. Our model is ~15 percentage points more accurate.
2. **Precision (81.40%):**
   - Out of all passengers the model predicted survived, 81.40% actually did. High precision means very few false alarms.
3. **Recall (50.72%):**
   - Out of all passengers who actually survived, the model identified 50.72%. The model is cautious about predicting survival for males due to the low baseline male survival rate.
4. **F1-Score (62.50%):**
   - The harmonic mean of precision and recall: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
5. **Confusion Matrix Breakdown:**
   - **True Negatives (102):** Passengers who died and were correctly predicted as dead.
   - **False Positives (8):** Passengers who died, but the model predicted survived (Type I error).
   - **False Negatives (34):** Passengers who survived, but the model predicted died (Type II error).
   - **True Positives (35):** Passengers who survived and were correctly predicted as alive.
6. **Feature Importance Ranking:**
   ```text
   1. Sex_female : 61.05%
   2. Pclass     : 20.75%
   3. Fare       :  9.14%
   4. Age        :  9.06%
   5. Sex_male   :  0.00% (Redundant with Sex_female in binary split)
   ```
 






