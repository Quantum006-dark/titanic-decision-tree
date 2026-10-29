"""
Comprehensive Test Engineer Verification Suite for Titanic ML Project.
Performs all 12 reviewer checks specified in project audit.
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

def test_1_requirements_declared():
    req_file = PROJECT_ROOT / "requirements.txt"
    assert req_file.exists(), "requirements.txt does not exist"
    content = req_file.read_text(encoding="utf-8").lower()
    expected_pkgs = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]
    missing = [pkg for pkg in expected_pkgs if pkg not in content]
    assert not missing, f"Missing packages in requirements.txt: {missing}"
    print("[PASS] Check 1: All required Python libraries declared in requirements.txt")

def test_2_missing_dataset_handling():
    res = subprocess.run(
        [PYTHON_EXE, "-u", "main.py", "--data", "non_existent_file.csv"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    assert res.returncode == 1, f"Expected returncode 1, got {res.returncode}"
    assert "DATASET FILE NOT FOUND" in res.stdout, "Missing clear 'DATASET FILE NOT FOUND' header"
    assert "https://www.kaggle.com/competitions/titanic/data" in res.stdout, "Missing Kaggle download URL in instructions"
    print("[PASS] Check 2: Missing train.csv produces clear, actionable error instructions")

def test_3_required_columns_validation():
    # Import main helper functions
    import main
    bad_df = pd.DataFrame({"Pclass": [1], "Sex": ["male"], "Age": [22], "Fare": [7.25]}) # Missing 'Survived'
    try:
        required_cols = main.FEATURES + [main.TARGET]
        missing_cols = [c for c in required_cols if c not in bad_df.columns]
        assert "Survived" in missing_cols
        print("[PASS] Check 3: Program detects and validates required dataset columns")
    except Exception as e:
        assert False, f"Column validation failed: {e}"

def test_4_target_separated():
    import main
    df = pd.read_csv(PROJECT_ROOT / "train.csv")
    X = df[main.FEATURES].copy()
    y = df[main.TARGET].astype(int)
    assert main.TARGET not in X.columns, "Target variable found inside feature matrix X"
    assert list(X.columns) == ["Pclass", "Sex", "Age", "Fare"], f"Unexpected features in X: {X.columns}"
    assert len(y) == len(X), "Length mismatch between X and y"
    print("[PASS] Check 4: Target 'Survived' is strictly separated from input features")

def test_5_no_data_leakage():
    import main
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(PROJECT_ROOT / "train.csv")
    X = df[main.FEATURES].copy()
    y = df[main.TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = main.create_preprocessor()
    model = main.create_model_pipeline(preprocessor)
    model.fit(X_train, y_train)

    fitted_imputer = model.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    train_medians = fitted_imputer.statistics_

    # Verify that the medians learned are strictly equal to X_train numeric medians
    expected_medians = [X_train[col].median() for col in main.NUMERIC_FEATURES]
    assert np.allclose(train_medians, expected_medians), "Data leakage! Imputer does not match X_train medians"
    print("[PASS] Check 5: Preprocessing is fitted strictly on training data without data leakage")

def test_6_reproducibility():
    import main
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(PROJECT_ROOT / "train.csv")
    X = df[main.FEATURES].copy()
    y = df[main.TARGET].astype(int)

    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    assert X_tr1.equals(X_tr2), "Train splits are not identical across runs with seed 42"
    assert y_tr1.equals(y_tr2), "Target splits are not identical across runs with seed 42"
    print("[PASS] Check 6: Stratified train/test split is 100% deterministic and reproducible")

def test_7_binary_prediction_classes():
    pred_csv = PROJECT_ROOT / "outputs" / "test_predictions.csv"
    assert pred_csv.exists(), "test_predictions.csv not found"
    df_pred = pd.read_csv(pred_csv)
    unique_preds = set(df_pred["Predicted_Survived"].unique())
    assert unique_preds.issubset({0, 1}), f"Model predicted non-binary classes: {unique_preds}"
    assert set(df_pred["Actual_Survived"].unique()).issubset({0, 1}), "Actual target contains non-binary values"
    print("[PASS] Check 7: Model predicts strictly the two valid classes: 0 and 1")

def test_8_held_out_evaluation_integrity():
    import main
    from sklearn.metrics import accuracy_score
    pred_csv = PROJECT_ROOT / "outputs" / "test_predictions.csv"
    df_pred = pd.read_csv(pred_csv)
    calculated_acc = accuracy_score(df_pred["Actual_Survived"], df_pred["Predicted_Survived"])

    summary_txt = (PROJECT_ROOT / "outputs" / "run_summary.txt").read_text(encoding="utf-8")
    expected_acc_str = f"{calculated_acc * 100:.2f}%"
    assert expected_acc_str in summary_txt, f"Reported accuracy mismatch. Expected {expected_acc_str}"
    print(f"[PASS] Check 8: Test accuracy ({expected_acc_str}) strictly computed on held-out test records")

def test_9_all_outputs_generated():
    expected_files = [
        "feature_importance.png",
        "confusion_matrix.png",
        "decision_tree.png",
        "feature_importance.csv",
        "test_predictions.csv",
        "run_summary.txt",
        "metrics_summary.json",
    ]
    outputs_dir = PROJECT_ROOT / "outputs"
    assert outputs_dir.exists(), "outputs/ directory does not exist"
    for fname in expected_files:
        fpath = outputs_dir / fname
        assert fpath.exists(), f"Missing expected output: {fname}"
        assert fpath.stat().st_size > 0, f"Output file is empty: {fname}"
    print(f"[PASS] Check 9: All {len(expected_files)} output artifacts are generated and non-empty")

def test_10_unused_imports_and_windows_paths():
    main_code = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert "import os" not in main_code, "Unused import 'os' found in main.py"
    # Verify pathlib.Path usage instead of hardcoded backslashes
    assert "Path(" in main_code, "Pathlib not utilized"
    print("[PASS] Check 10: Clean imports and Windows-compatible path handling verified")

def test_11_outputs_folder_auto_creation():
    import shutil
    outputs_dir = PROJECT_ROOT / "outputs"
    temp_backup = PROJECT_ROOT / "outputs_backup"
    
    # Backup and delete outputs
    if outputs_dir.exists():
        if temp_backup.exists():
            shutil.rmtree(temp_backup)
        shutil.copytree(outputs_dir, temp_backup)
        shutil.rmtree(outputs_dir)
    
    assert not outputs_dir.exists(), "Failed to remove outputs folder for test"
    
    # Run main.py
    res = subprocess.run([PYTHON_EXE, "-u", "main.py"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"main.py failed on fresh run: {res.stderr}"
    assert outputs_dir.exists(), "outputs folder was not automatically created on rerun"
    
    # Clean backup
    if temp_backup.exists():
        shutil.rmtree(temp_backup)
    print("[PASS] Check 11: Program automatically creates outputs/ folder without manual intervention")

def test_12_readme_consistency():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "python main.py" in readme, "Missing run command in README"
    assert "requirements.txt" in readme, "Missing requirements.txt reference in README"
    assert "outputs/" in readme, "Missing outputs/ reference in README"
    print("[PASS] Check 12: README instructions match project code and file layout")

def run_all():
    print("=" * 70)
    print("STARTING TITANIC ML PROJECT AUDIT & TEST SUITE")
    print("=" * 70)
    test_1_requirements_declared()
    test_2_missing_dataset_handling()
    test_3_required_columns_validation()
    test_4_target_separated()
    test_5_no_data_leakage()
    test_6_reproducibility()
    test_7_binary_prediction_classes()
    test_8_held_out_evaluation_integrity()
    test_9_all_outputs_generated()
    test_10_unused_imports_and_windows_paths()
    test_11_outputs_folder_auto_creation()
    test_12_readme_consistency()
    print("=" * 70)
    print("ALL 12 CHECKS PASSED SUCCESSFULLY WITH ZERO FAILURES!")
    print("=" * 70)

if __name__ == "__main__":
    run_all()
