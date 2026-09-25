"""
AssureX Claims Classification Model Training & Evaluation Pipeline
==================================================================

Trains and benchmarks three supervised classifiers on train_features.csv:
1. Random Forest (RandomForestClassifier)
2. XGBoost (XGBClassifier)
3. Support Vector Machine (SVC with StandardScaler)

Methodology:
- Stratified 5-Fold Cross Validation tuning macro-F1 (equal weight across classes).
- Comprehensive evaluation on val_features.csv:
  * Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Weighted), Macro-F1.
  * Confusion Matrix and per-class reports.
- Comparison table across all 3 candidate models.
- Final evaluation of best model on test_features.csv.
- Per-class recall inspection (explicitly monitoring and flagging Manual Review recall).
- Persistence:
  * Winning model saved to model/claim_classifier.pkl with joblib.
  * Fitted preprocessing transformers & feature schema saved to model/preprocessing.pkl with joblib.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Optional XGBoost import with graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

SEED = 42
TARGET_CLASSES = ['Valid Claim', 'Invalid Claim', 'Manual Review']


def find_features_file(filename: str, dataset_dir: str = "dataset") -> str:
    """Locate feature CSV across common project paths."""
    candidates = [
        filename,
        os.path.join(dataset_dir, filename),
        os.path.join("..", filename),
        os.path.join("..", dataset_dir, filename),
        os.path.join("/content", filename),
        os.path.join("/content", dataset_dir, filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find feature matrix: {filename} in {candidates}")


def load_datasets(dataset_dir: str = "dataset"):
    """Load train, val, and test feature matrices."""
    train_path = find_features_file("train_features.csv", dataset_dir)
    val_path = find_features_file("val_features.csv", dataset_dir)
    test_path = find_features_file("test_features.csv", dataset_dir)

    print(f"[*] Loading training features:   {train_path}")
    print(f"[*] Loading validation features: {val_path}")
    print(f"[*] Loading testing features:    {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Exclude non-feature columns
    drop_cols = ['claim_id', 'class_label']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df['class_label'].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df['class_label'].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df['class_label'].copy()

    print(f"[+] Feature count: {len(feature_cols)}")
    print(f"[+] Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate_predictions(y_true, y_pred, model_name: str) -> dict:
    """Computes comprehensive multi-class metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    report = classification_report(y_true, y_pred, labels=TARGET_CLASSES, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "report": report
    }


def train_and_compare_classifiers(
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols,
    model_output_dir: str = "model",
    dataset_dir: str = "dataset"
):
    """
    Trains Random Forest, XGBoost, and SVM with 5-fold CV hyperparameter tuning.
    Compares on validation set, evaluates best model on test set, and saves artifacts.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    trained_models = {}
    val_results = []

    # Encode target labels for XGBoost compatibility (0, 1, 2)
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_CLASSES)
    y_train_idx = label_encoder.transform(y_train)
    y_val_idx = label_encoder.transform(y_val)
    y_test_idx = label_encoder.transform(y_test)

    # =========================================================================
    # 1. Random Forest Classifier
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. Tuning Random Forest Classifier (GridSearchCV with 5-Fold CV)...")
    print("=" * 70)
    rf = RandomForestClassifier(random_state=SEED)
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 12, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    grid_rf = GridSearchCV(
        rf, param_grid_rf, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
    )
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    trained_models['Random Forest'] = {
        'model': best_rf,
        'cv_score': grid_rf.best_score_,
        'params': grid_rf.best_params_,
        'uses_label_encoder': False
    }
    print(f"[+] Random Forest Best CV Macro-F1: {grid_rf.best_score_:.4f}")
    print(f"[+] Best Hyperparameters: {grid_rf.best_params_}")

    # Validation Evaluation
    rf_val_preds = best_rf.predict(X_val)
    rf_val_eval = evaluate_predictions(y_val, rf_val_preds, "Random Forest")
    rf_val_eval['cv_macro_f1'] = grid_rf.best_score_
    val_results.append(rf_val_eval)

    # =========================================================================
    # 2. XGBoost Classifier
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. Tuning XGBoost Classifier (GridSearchCV with 5-Fold CV)...")
    print("=" * 70)
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=SEED
        )
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid_xgb = GridSearchCV(
            xgb, param_grid_xgb, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid_xgb.fit(X_train, y_train_idx)

        best_xgb = grid_xgb.best_estimator_
        trained_models['XGBoost'] = {
            'model': best_xgb,
            'cv_score': grid_xgb.best_score_,
            'params': grid_xgb.best_params_,
            'uses_label_encoder': True
        }
        print(f"[+] XGBoost Best CV Macro-F1: {grid_xgb.best_score_:.4f}")
        print(f"[+] Best Hyperparameters: {grid_xgb.best_params_}")

        # Validation Evaluation
        xgb_val_idx = best_xgb.predict(X_val)
        xgb_val_preds = label_encoder.inverse_transform(xgb_val_idx)
        xgb_val_eval = evaluate_predictions(y_val, xgb_val_preds, "XGBoost")
        xgb_val_eval['cv_macro_f1'] = grid_xgb.best_score_
        val_results.append(xgb_val_eval)
    else:
        print("[!] XGBoost not available in local environment; skipping XGBoost in local run.")
        print("[!] Note: XGBoost code is fully embedded in notebooks/train_and_evaluate_classifiers.ipynb for Colab execution.")

    # =========================================================================
    # 3. Support Vector Machine (with StandardScaler)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. Tuning Support Vector Machine (StandardScaler + SVC Pipeline)...")
    print("=" * 70)
    scaler = StandardScaler()
    svm_pipe = Pipeline([
        ('scaler', scaler),
        ('svm', SVC(probability=True, random_state=SEED))
    ])
    param_grid_svm = {
        'svm__C': [0.5, 1.0, 5.0, 10.0],
        'svm__gamma': ['scale', 'auto', 0.05],
        'svm__kernel': ['rbf'],
        'svm__class_weight': ['balanced', None]
    }
    grid_svm = GridSearchCV(
        svm_pipe, param_grid_svm, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
    )
    grid_svm.fit(X_train, y_train)

    best_svm = grid_svm.best_estimator_
    trained_models['SVM'] = {
        'model': best_svm,
        'cv_score': grid_svm.best_score_,
        'params': grid_svm.best_params_,
        'uses_label_encoder': False
    }
    print(f"[+] SVM Best CV Macro-F1: {grid_svm.best_score_:.4f}")
    print(f"[+] Best Hyperparameters: {grid_svm.best_params_}")

    # Validation Evaluation
    svm_val_preds = best_svm.predict(X_val)
    svm_val_eval = evaluate_predictions(y_val, svm_val_preds, "Support Vector Machine")
    svm_val_eval['cv_macro_f1'] = grid_svm.best_score_
    val_results.append(svm_val_eval)

    # =========================================================================
    # Comparison Table of Validation Metrics
    # =========================================================================
    print("\n" + "=" * 90)
    print("MODEL VALIDATION COMPARISON TABLE (val_features.csv, N=225)")
    print("=" * 90)

    comparison_data = []
    for r in val_results:
        comparison_data.append({
            "Classifier": r["model_name"],
            "5-Fold CV F1": f"{r['cv_macro_f1']:.4f}",
            "Val Accuracy": f"{r['accuracy']:.4f}",
            "Val Prec (Macro)": f"{r['precision_macro']:.4f}",
            "Val Rec (Macro)": f"{r['recall_macro']:.4f}",
            "Val F1 (Weighted)": f"{r['f1_weighted']:.4f}",
            "Val Macro-F1": f"{r['f1_macro']:.4f}",
        })

    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))

    # Print Validation Confusion Matrices
    print("\n" + "-" * 70)
    print("VALIDATION CONFUSION MATRICES (Labels: Valid, Invalid, Manual Review)")
    print("-" * 70)
    for r in val_results:
        print(f"\n--- {r['model_name']} ---")
        cm_df = pd.DataFrame(r['confusion_matrix'], index=[f"True: {c}" for c in TARGET_CLASSES],
                             columns=[f"Pred: {c}" for c in TARGET_CLASSES])
        print(cm_df.to_string())

    # =========================================================================
    # Select Best Model & Final Evaluation on Test Set
    # =========================================================================
    # Ranked primarily by Val Macro-F1, then Val Accuracy
    best_eval = max(val_results, key=lambda x: (x['f1_macro'], x['accuracy']))
    winning_model_name = best_eval['model_name']
    winner_info = trained_models[winning_model_name]
    winning_model = winner_info['model']

    print("\n" + "=" * 90)
    print(f"WINNING MODEL SELECTED: {winning_model_name} (Val Macro-F1: {best_eval['f1_macro']:.4f})")
    print("=" * 90)

    # Evaluate on test set
    if winner_info['uses_label_encoder']:
        test_preds_raw = winning_model.predict(X_test)
        test_preds = label_encoder.inverse_transform(test_preds_raw)
    else:
        test_preds = winning_model.predict(X_test)

    test_eval = evaluate_predictions(y_test, test_preds, winning_model_name)

    print("\n" + "=" * 90)
    print(f"FINAL TEST SET EVALUATION ({winning_model_name} on test_features.csv, N=225)")
    print("=" * 90)
    print(f"Test Accuracy:           {test_eval['accuracy']:.4f} ({test_eval['accuracy']*100:.2f}%)")
    print(f"Test Precision (Macro):  {test_eval['precision_macro']:.4f}")
    print(f"Test Recall (Macro):     {test_eval['recall_macro']:.4f}")
    print(f"Test F1 (Weighted):      {test_eval['f1_weighted']:.4f}")
    print(f"Test Macro-F1:           {test_eval['f1_macro']:.4f}")

    print("\nPER-CLASS PERFORMANCE BREAKDOWN:")
    print("-" * 65)
    print(f"{'Class':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)

    manual_review_recall = 0.0
    for cls in TARGET_CLASSES:
        metrics_dict = test_eval['report'][cls]
        prec = metrics_dict['precision']
        rec = metrics_dict['recall']
        f1 = metrics_dict['f1-score']
        supp = int(metrics_dict['support'])
        print(f"{cls:<18} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {supp:<8}")
        if cls == 'Manual Review':
            manual_review_recall = rec

    print("-" * 65)

    # Manual Review Recall Guardrail / Alert
    print("\n[OPERATIONAL GUARDRAIL AUDIT: MANUAL REVIEW RECALL]")
    if manual_review_recall >= 0.85:
        print(f" [PASS - HEALTHY] Manual Review Recall is {manual_review_recall:.4f} (>= 0.85).")
        print("                 Borderline/ambiguous claims are safely routed to claims adjusters.")
    elif manual_review_recall >= 0.75:
        print(f" [WARNING - MODERATE] Manual Review Recall is {manual_review_recall:.4f} (< 0.85).")
        print("                     Adjuster triage threshold adjustment recommended.")
    else:
        print(f" [CRITICAL FLAG - LOW RECALL] Manual Review Recall is {manual_review_recall:.4f} (< 0.75)!")
        print("                             High risk of misclassifying borderline or fraudulent claims.")

    print("\nTEST SET CONFUSION MATRIX:")
    cm_test_df = pd.DataFrame(test_eval['confusion_matrix'],
                              index=[f"True: {c}" for c in TARGET_CLASSES],
                              columns=[f"Pred: {c}" for c in TARGET_CLASSES])
    print(cm_test_df.to_string())

    # =========================================================================
    # Persist Winning Model & Preprocessing Pipeline
    # =========================================================================
    os.makedirs(model_output_dir, exist_ok=True)
    model_pkl_path = os.path.join(model_output_dir, "claim_classifier.pkl")
    prep_pkl_path = os.path.join(model_output_dir, "preprocessing.pkl")
    # Fit a standard scaler on the full training features for standalone inference
    full_scaler = StandardScaler()
    full_scaler.fit(X_train)

    # Load raw train.csv if available to extract categorical frequency and risk maps for inference
    try:
        raw_train_path = find_features_file("train.csv", dataset_dir)
        raw_train = pd.read_csv(raw_train_path)
        dmg_freq = (raw_train['damage_type'].value_counts() / len(raw_train)).to_dict()
        global_prior = float((raw_train['class_label'] == 'Invalid Claim').mean())
        dmg_risk = (
            raw_train.groupby('damage_type')
            .apply(lambda g: float((g['class_label'] == 'Invalid Claim').sum() / len(g)))
            .to_dict()
        )
        ret_freq = (raw_train['retailer'].value_counts(dropna=True) / len(raw_train)).to_dict()
    except Exception:
        dmg_freq = {}
        dmg_risk = {}
        ret_freq = {}
        global_prior = 0.3333

    preprocessing_bundle = {
        "feature_names": feature_cols,
        "scaler": full_scaler,
        "label_encoder": label_encoder,
        "target_classes": TARGET_CLASSES,
        "uses_label_encoder": winner_info['uses_label_encoder'],
        "model_architecture": winning_model_name,
        "damage_type_freq_map": dmg_freq,
        "damage_type_invalid_risk_map": dmg_risk,
        "retailer_freq_map": ret_freq,
        "global_invalid_prior": global_prior,
        "training_metadata": {
            "train_samples": len(X_train),
            "val_macro_f1": best_eval['f1_macro'],
            "test_macro_f1": test_eval['f1_macro'],
            "test_accuracy": test_eval['accuracy'],
            "manual_review_recall": manual_review_recall
        }
    }

    joblib.dump(winning_model, model_pkl_path)
    joblib.dump(preprocessing_bundle, prep_pkl_path)

    print("\n" + "=" * 90)
    print(f"[SUCCESS] Saved winning model:           {model_pkl_path}")
    print(f"[SUCCESS] Saved preprocessing artifacts: {prep_pkl_path}")
    print("=" * 90)

    return winning_model, preprocessing_bundle, comp_df, test_eval


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate AssureX Claim Classifiers")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing feature matrices")
    parser.add_argument("--model-dir", type=str, default="model", help="Directory to save model artifacts")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_datasets(args.dataset_dir)
    train_and_compare_classifiers(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols,
        model_output_dir=args.model_dir,
        dataset_dir=args.dataset_dir
    )


if __name__ == "__main__":
    main()
