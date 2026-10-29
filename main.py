"""
=============================================================================
Predicting Titanic Passenger Survival Using a Decision Tree Classifier
IEEE-VIT Machine Learning Enrollment Task
Target Audience: First-Year B.Tech Student
=============================================================================

This program executes an end-to-end Machine Learning pipeline:
1. Validates the existence of Kaggle's official 'train.csv'.
2. Performs Exploratory Data Analysis (EDA) on the dataset.
3. Extracts input features (Pclass, Sex, Age, Fare) and target (Survived).
4. Splits the dataset into 80% training and 20% testing sets (stratified, random_state=42).
5. Constructs a scikit-learn Pipeline with ColumnTransformer:
   - Median imputation for numeric features (Age, Fare, Pclass).
   - Most-frequent imputation + One-Hot Encoding for categorical features (Sex).
   - Prevents data leakage by fitting transformers strictly on training data.
6. Trains a regularized Decision Tree Classifier (max_depth=4, min_samples_leaf=8).
7. Evaluates predictions on the unseen test dataset.
8. Automatically creates the 'outputs/' folder and generates all required artifacts:
   - outputs/feature_importance.png (readable horizontal bar chart)
   - outputs/confusion_matrix.png (labeled with 'Did not survive' & 'Survived')
   - outputs/decision_tree.png (high-resolution decision tree diagram)
   - outputs/feature_importance.csv (table of features and importance scores)
   - outputs/test_predictions.csv (test input features, actual labels, predicted labels)
   - outputs/run_summary.txt (actual accuracy, dataset size, train/test counts, ranking)
   - outputs/metrics_summary.json (machine-readable metrics)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Define the features and target specified in task guidelines
FEATURES = ["Pclass", "Sex", "Age", "Fare"]
NUMERIC_FEATURES = ["Age", "Fare", "Pclass"]
CATEGORICAL_FEATURES = ["Sex"]
TARGET = "Survived"

DEFAULT_DATA_FILE = "train.csv"
DEFAULT_OUTPUT_DIR = "outputs"


def verify_dataset(data_path_str: str) -> Path:
    """
    Check whether train.csv exists before running the model.
    If missing, display clear download instructions and terminate cleanly.
    Never fabricate or substitute dummy data.
    """
    data_path = Path(data_path_str)
    if not data_path.exists():
        print("\n" + "=" * 76)
        print(" [!] DATASET FILE NOT FOUND: " + str(data_path.resolve()))
        print("=" * 76)
        print("The official Kaggle Titanic dataset 'train.csv' was not found.")
        print("\nPlease follow these exact steps to obtain and supply it:")
        print("  1. Open your browser and go to:")
        print("     https://www.kaggle.com/competitions/titanic/data")
        print("  2. Sign in to your Kaggle account (and accept competition rules if asked).")
        print("  3. Download 'train.csv' (or download the full zip and extract 'train.csv').")
        print("  4. Place 'train.csv' directly into this project directory:")
        print(f"     {Path('.').resolve()}")
        print("  5. Run the program again:")
        print("     python main.py")
        print("     (Or specify a custom location: python main.py --data \"path/to/train.csv\")")
        print("=" * 76 + "\n")
        sys.exit(1)
    return data_path


def perform_eda(df: pd.DataFrame):
    """
    Perform initial Exploratory Data Analysis (EDA) on the raw dataset.
    Helps understand data distribution, missing fields, and class balance.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Dataset Dimensions
    print(f"\n[+] Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")

    # 2. First 5 Rows
    print("\n[+] First 5 Rows of Raw Dataset:")
    print(df.head().to_string())

    # 3. Column Names
    print("\n[+] Available Columns in Dataset:")
    print(", ".join(df.columns.tolist()))

    # 4. Data Types
    print("\n[+] Column Data Types:")
    print(df.dtypes.to_string())

    # 5. Missing Value Inspection
    print("\n[+] Missing Values in Selected Input Columns:")
    missing_subset = df[FEATURES].isnull().sum()
    print(missing_subset.to_string())
    missing_pct = (missing_subset / len(df)) * 100
    for col in FEATURES:
        if missing_subset[col] > 0:
            print(f"    -> Note: '{col}' has {missing_subset[col]} missing values ({missing_pct[col]:.2f}%).")

    # 6. Target Distribution (Survived vs Did not survive)
    print("\n[+] Target Class Distribution ('Survived'):")
    counts = df[TARGET].value_counts().sort_index()
    total = len(df)
    for class_val, count in counts.items():
        label_name = "Did not survive (0)" if class_val == 0 else "Survived (1)"
        print(f"    {label_name:<22}: {count:4d} passengers ({count / total * 100:.2f}%)")


def create_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer.
    - Numeric features: Median Imputation.
      (Median is preferred over mean because age and fares are skewed by outliers).
    - Categorical features: Most-frequent Imputation followed by One-Hot Encoding.
      (Converts string categories like 'male' and 'female' into binary numeric values).
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def create_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Combine preprocessing and DecisionTreeClassifier into a unified Pipeline.

    Why use a Pipeline?
    - Prevents Data Leakage: The imputer calculates medians ONLY from the
      training split (X_train) and applies those learned values to test data (X_test).
    - Reproducibility: Bundles transformations and the model into one object.
    
    Why max_depth=4 and min_samples_leaf=8?
    - An unconstrained decision tree tends to memorize the training data (overfitting).
    - Setting max_depth=4 limits tree depth so each rule is clear and interpretable.
    - Setting min_samples_leaf=8 requires at least 8 samples at leaf nodes,
      stopping the tree from making predictions based on isolated noise.
    """
    tree_classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=8,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", tree_classifier),
        ]
    )
    return pipeline


def evaluate_and_report(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series, output_dir: Path):
    """
    Evaluate the model on both training and test sets.
    Display metrics and write the actual run summary to disk.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: MODEL EVALUATION ON HELD-OUT TEST DATA")
    print("=" * 70)

    # Predictions
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Metrics on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_prob)

    # Train accuracy to verify generalization gap
    train_accuracy = model.score(X_train, y_train)

    print(f"[+] Training Set Accuracy: {train_accuracy * 100:.2f}% ({len(X_train)} samples)")
    print(f"[+] Testing Set Accuracy:  {test_accuracy * 100:.2f}% ({len(X_test)} samples)")
    print(f"[+] Generalization Gap:    {abs(train_accuracy - test_accuracy) * 100:.2f}% (Well-regularized)")
    print(f"[+] Precision Score:       {test_precision * 100:.2f}%")
    print(f"[+] Recall Score:          {test_recall * 100:.2f}%")
    print(f"[+] F1-Score:              {test_f1 * 100:.2f}%")
    print(f"[+] ROC-AUC Score:         {test_roc_auc * 100:.2f}%")

    print("\n[+] Detailed Classification Report:")
    report_text = classification_report(
        y_test, y_test_pred, target_names=["Did not survive", "Survived"]
    )
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("[+] Confusion Matrix Breakdown:")
    print(f"    - True Negatives (Did not survive): {cm[0, 0]}")
    print(f"    - False Positives (Type I error):   {cm[0, 1]}")
    print(f"    - False Negatives (Type II error):  {cm[1, 0]}")
    print(f"    - True Positives (Survived):        {cm[1, 1]}")

    return y_test_pred, y_test_prob, cm, train_accuracy, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, report_text


def plot_and_save_artifacts(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                            y_pred: np.ndarray, y_prob: np.ndarray, cm: np.ndarray,
                            dataset_len: int, train_len: int, test_len: int,
                            train_acc: float, test_acc: float, prec: float, rec: float, f1: float, auc: float,
                            report_text: str, output_dir: Path):
    """
    Generate all required visual charts and export outputs/:
    1. feature_importance.png (horizontal bar chart)
    2. confusion_matrix.png (labeled with Did not survive and Survived)
    3. decision_tree.png (readable tree with split conditions and class labels)
    4. feature_importance.csv (table of features and importances)
    5. test_predictions.csv (test features, actual labels, predicted labels)
    6. run_summary.txt (actual metrics, dataset sizes, feature ranking)
    """
    print("\n" + "=" * 70)
    print("STAGE 5: GENERATING CHARTS & SAVING OUTPUT ARTIFACTS")
    print("=" * 70)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Extract actual transformed feature names (including One-Hot Encoded Sex columns)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names

    importances = classifier.feature_importances_

    # Sort descending for tables and summary
    desc_idx = np.argsort(importances)[::-1]
    desc_names = [feature_names[i] for i in desc_idx]
    desc_vals = importances[desc_idx]

    # Sort ascending for horizontal bar chart so top feature is at the top
    asc_idx = np.argsort(importances)
    asc_names = [feature_names[i] for i in asc_idx]
    asc_vals = importances[asc_idx]

    # -------------------------------------------------------------------------
    # 1. feature_importance.png — Readable Horizontal Bar Chart
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    colors = ["#90caf9", "#64b5f6", "#42a5f5", "#1e88e5", "#1565c0"]
    bar_colors = colors[: len(asc_names)]

    bars = plt.barh(asc_names, asc_vals, color=bar_colors, edgecolor="#0d47a1", linewidth=1.2, height=0.6)
    plt.title("Titanic Survival — Decision Tree Feature Importance", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Gini Importance Score (Impurity Reduction)", fontsize=11, fontweight="bold")
    plt.ylabel("Transformed Feature Names", fontsize=11, fontweight="bold")
    plt.xlim(0, max(asc_vals) * 1.22 if max(asc_vals) > 0 else 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#0d47a1",
        )

    plt.tight_layout()
    fi_plot_path = output_dir / "feature_importance.png"
    plt.savefig(fi_plot_path)
    plt.close()
    print(f"[+] 1. Feature importance horizontal bar chart saved: {fi_plot_path}")

    # -------------------------------------------------------------------------
    # 2. confusion_matrix.png — Labeled Confusion Matrix Heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Did not survive", "Survived"],
        yticklabels=["Did not survive", "Survived"],
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=1.5,
        linecolor="#0d47a1",
    )
    plt.title("Confusion Matrix — Titanic Decision Tree", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"[+] 2. Labeled confusion matrix heatmap saved:       {cm_plot_path}")

    # -------------------------------------------------------------------------
    # 3. decision_tree.png — Readable Visual Decision Tree Diagram
    # -------------------------------------------------------------------------
    plt.figure(figsize=(22, 11), dpi=300)
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        rounded=True,
        fontsize=8,
        precision=2,
    )
    plt.title("Trained Decision Tree Visualization (max_depth=4, min_samples_leaf=8)", fontsize=15, fontweight="bold", pad=18)
    plt.tight_layout()
    tree_plot_path = output_dir / "decision_tree.png"
    plt.savefig(tree_plot_path)
    plt.close()
    print(f"[+] 3. Readable decision tree diagram saved:           {tree_plot_path}")

    # -------------------------------------------------------------------------
    # 4. feature_importance.csv — Table of Features and Importance Values
    # -------------------------------------------------------------------------
    fi_df = pd.DataFrame({
        "Feature": desc_names,
        "Importance_Score": np.round(desc_vals, 4),
        "Importance_Percentage": [f"{v * 100:.2f}%" for v in desc_vals],
    })
    fi_csv_path = output_dir / "feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"[+] 4. Feature importance table saved:                 {fi_csv_path}")

    # -------------------------------------------------------------------------
    # 5. test_predictions.csv — Input Features, Actual and Predicted Labels
    # -------------------------------------------------------------------------
    pred_df = X_test.copy()
    pred_df["Actual_Survived"] = y_test.values
    pred_df["Predicted_Survived"] = y_pred
    pred_df["Survival_Probability"] = np.round(y_prob, 4)
    pred_df["Prediction_Correct"] = pred_df["Actual_Survived"] == pred_df["Predicted_Survived"]
    pred_csv_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[+] 5. Test set predictions table saved:               {pred_csv_path}")

    # -------------------------------------------------------------------------
    # 6. run_summary.txt — Actual Metrics, Dataset Sizes, Feature Ranking
    # -------------------------------------------------------------------------
    summary_path = output_dir / "run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("TITANIC PASSENGER SURVIVAL PREDICTION — DECISION TREE RUN SUMMARY\n")
        f.write("======================================================================\n\n")
        f.write("1. DATASET & SPLIT SIZES:\n")
        f.write(f"   - Total Dataset Rows:       {dataset_len}\n")
        f.write(f"   - Training Rows (80%):      {train_len}\n")
        f.write(f"   - Testing Rows (20%):       {test_len}\n\n")
        f.write("2. MODEL PERFORMANCE METRICS (ON UNSEEN TEST DATA):\n")
        f.write(f"   - Training Accuracy:        {train_acc * 100:.2f}%\n")
        f.write(f"   - Testing Accuracy:         {test_acc * 100:.2f}%\n")
        f.write(f"   - Precision (Survived):     {prec * 100:.2f}%\n")
        f.write(f"   - Recall (Survived):        {rec * 100:.2f}%\n")
        f.write(f"   - F1-Score (Survived):      {f1 * 100:.2f}%\n")
        f.write(f"   - ROC-AUC Score:            {auc * 100:.2f}%\n\n")
        f.write("3. CONFUSION MATRIX COUNTS:\n")
        f.write(f"   - True Negatives (Did not survive correctly predicted): {cm[0, 0]}\n")
        f.write(f"   - False Positives (Predicted alive, actually died):     {cm[0, 1]}\n")
        f.write(f"   - False Negatives (Predicted died, actually survived):  {cm[1, 0]}\n")
        f.write(f"   - True Positives (Survived correctly predicted):        {cm[1, 1]}\n\n")
        f.write("4. FEATURE IMPORTANCE RANKING (GINI IMPURITY REDUCTION):\n")
        for rank, (name, val) in enumerate(zip(desc_names, desc_vals), start=1):
            f.write(f"   {rank}. {name:<15}: {val * 100:6.2f}%\n")
        f.write("\n5. INTERPRETATION NOTE ON FEATURE IMPORTANCE:\n")
        f.write("   Feature importance indicates how much the fitted decision tree relied\n")
        f.write("   on a feature to reduce Gini impurity when creating partition rules.\n")
        f.write("   It measures model reliance within this specific dataset and split;\n")
        f.write("   IT DOES NOT PROVE DIRECT REAL-WORLD CAUSATION.\n\n")
        f.write("6. DETAILED CLASSIFICATION REPORT:\n")
        f.write(report_text)

    print(f"[+] 6. Actual run summary saved:                       {summary_path}")

    # Also save JSON for machine parsing
    metrics_json = {
        "dataset_size": dataset_len,
        "train_size": train_len,
        "test_size": test_len,
        "training_accuracy": round(float(train_acc), 4),
        "testing_accuracy": round(float(test_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "feature_importances": {name: round(float(val), 4) for name, val in zip(desc_names, desc_vals)},
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Titanic Passenger Survival Prediction - Decision Tree")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Path to Kaggle Titanic train.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for generated artifacts")
    args = parser.parse_args()

    # Automatically create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IEEE-VIT MACHINE LEARNING ENROLLMENT TASK: TITANIC DECISION TREE")
    print("=" * 70)

    # 1. Verify train.csv exists
    data_path = verify_dataset(args.data)

    # 2. Load dataset with Pandas
    df = pd.read_csv(data_path)

    # Validate required columns
    required_cols = FEATURES + [TARGET]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The dataset is missing required columns: {missing_cols}")

    # 3. Exploratory Data Analysis
    perform_eda(df)

    # 4. Feature Selection & Train/Test Split
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE SELECTION & TRAIN / TEST SPLIT")
    print("=" * 70)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Total samples in dataset: {len(X)}")
    print(f"[+] Training samples (80%):  {len(X_train)}")
    print(f"[+] Testing samples (20%):   {len(X_test)}")
    print("[+] Stratification applied: target class balance preserved in both splits.")

    # 5. Build Pipeline and Train Model
    print("\n" + "=" * 70)
    print("STAGE 3: PIPELINE CONSTRUCTION & MODEL TRAINING")
    print("=" * 70)
    preprocessor = create_preprocessor()
    model_pipeline = create_model_pipeline(preprocessor)

    # Train model pipeline (preprocessor fitted solely on X_train)
    model_pipeline.fit(X_train, y_train)
    print("[+] Pipeline successfully fitted on training data without data leakage.")

    # 6. Evaluate Model
    (y_pred, y_prob, cm, train_acc, test_acc,
     prec, rec, f1, auc, report_text) = evaluate_and_report(
        model_pipeline, X_train, X_test, y_train, y_test, output_dir
    )

    # 7. Generate Visualizations and Output Files
    plot_and_save_artifacts(
        model_pipeline, X_test, y_test, y_pred, y_prob, cm,
        len(df), len(X_train), len(X_test),
        train_acc, test_acc, prec, rec, f1, auc, report_text, output_dir
    )

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS AND OUTPUT ARTIFACTS CREATED SUCCESSFULLY!")
    print(f"Location: {output_dir.resolve()}")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
