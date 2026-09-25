"""
Unit & Validation Tests for AssureX Feature Engineering Pipeline
================================================================
Verifies:
1. Generation and integrity of train_features.csv, val_features.csv, test_features.csv.
2. Exact row counts matching splits (1050 train, 225 val, 225 test).
3. Presence of all 6 derived features and encoded categorical columns.
4. Absence of null values.
5. Verification of Colab notebook notebooks/claims_feature_engineering_eda.ipynb.
"""

import os
import json
import pandas as pd
from pathlib import Path


def test_feature_files_exist():
    for fname in ["train_features.csv", "val_features.csv", "test_features.csv"]:
        assert os.path.exists(fname) or os.path.exists(os.path.join("dataset", fname)), f"Missing {fname}"


def test_feature_shapes_and_columns():
    expected_shapes = {
        "train_features.csv": (1050, 24),
        "val_features.csv": (225, 24),
        "test_features.csv": (225, 24),
    }

    expected_derived = [
        "product_age_days",
        "remaining_warranty_days",
        "missing_document_count",
        "days_to_reporting_deadline",
        "repair_count",
        "has_any_contradiction",
        "damage_type_freq",
        "damage_type_invalid_risk",
        "retailer_freq",
        "retailer_is_missing",
    ]

    for fname, target_shape in expected_shapes.items():
        fpath = fname if os.path.exists(fname) else os.path.join("dataset", fname)
        df = pd.read_csv(fpath)
        assert df.shape == target_shape, f"{fname} shape {df.shape} != {target_shape}"
        assert df.isnull().sum().sum() == 0, f"{fname} contains unexpected nulls"

        for col in expected_derived:
            assert col in df.columns, f"Missing derived column {col} in {fname}"

        assert "class_label" in df.columns, f"Missing target class_label in {fname}"
        assert set(df["class_label"].unique()) == {"Valid Claim", "Invalid Claim", "Manual Review"}


def test_notebook_json_validity():
    nb_path = Path("notebooks/claims_feature_engineering_eda.ipynb")
    assert nb_path.exists(), "Notebook notebooks/claims_feature_engineering_eda.ipynb does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert nb["nbformat"] == 4
    cell_types = [c["cell_type"] for c in nb["cells"]]
    assert "markdown" in cell_types
    assert "code" in cell_types


if __name__ == "__main__":
    print("Running feature engineering validation tests...")
    test_feature_files_exist()
    test_feature_shapes_and_columns()
    test_notebook_json_validity()
    print("[PASS] All feature engineering validation tests passed successfully!")
