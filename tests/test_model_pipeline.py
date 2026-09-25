"""
Unit & Validation Tests for AssureX Model Training Pipeline
===========================================================
Verifies:
1. Presence and validity of notebooks/train_and_evaluate_classifiers.ipynb.
2. Syntax and structural completeness of train_classifiers.py.
3. Correct feature dimensions matching train_features.csv (22 predictor features).
4. Target class compliance (['Valid Claim', 'Invalid Claim', 'Manual Review']).
"""

import os
import json
import py_compile
import pandas as pd
from pathlib import Path


def test_notebook_structure():
    nb_path = Path("notebooks/train_and_evaluate_classifiers.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "Notebook missing 'cells' key"
    assert nb["nbformat"] == 4, f"Unexpected nbformat: {nb.get('nbformat')}"

    # Verify key sections exist in notebook source
    all_source = " ".join([" ".join(c.get("source", [])) for c in nb["cells"]])
    assert "RandomForestClassifier" in all_source, "RandomForestClassifier missing from notebook"
    assert "XGBClassifier" in all_source, "XGBClassifier missing from notebook"
    assert "SVC" in all_source, "SVC missing from notebook"
    assert "StandardScaler" in all_source, "StandardScaler missing from notebook"
    assert "GridSearchCV" in all_source, "GridSearchCV missing from notebook"
    assert "f1_macro" in all_source, "f1_macro scoring missing from notebook"
    assert "Manual Review" in all_source, "Manual Review class missing from notebook"
    assert "claim_classifier.pkl" in all_source, "claim_classifier.pkl persistence missing"
    assert "preprocessing.pkl" in all_source, "preprocessing.pkl persistence missing"


def test_train_script_syntax():
    script_path = Path("train_classifiers.py")
    assert script_path.exists(), f"Script {script_path} missing"
    # Verify compilation without syntax error
    py_compile.compile(str(script_path), doraise=True)


def test_features_matrix_compatibility():
    train_path = Path("dataset/train_features.csv") if Path("dataset/train_features.csv").exists() else Path("train_features.csv")
    assert train_path.exists(), f"Training features file {train_path} not found"

    df = pd.read_csv(train_path)
    assert "claim_id" in df.columns, "Missing claim_id"
    assert "class_label" in df.columns, "Missing class_label"
    assert len(df.columns) == 24, f"Expected 24 columns, got {len(df.columns)}"

    target_classes = set(df["class_label"].unique())
    assert target_classes == {"Valid Claim", "Invalid Claim", "Manual Review"}, f"Unexpected targets: {target_classes}"


if __name__ == "__main__":
    print("Running model pipeline validation tests...")
    test_notebook_structure()
    test_train_script_syntax()
    test_features_matrix_compatibility()
    print("[PASS] All model pipeline validation tests passed successfully!")
