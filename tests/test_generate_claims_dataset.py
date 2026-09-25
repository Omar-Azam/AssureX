"""
Unit & Validation Tests for AssureX Synthetic Claims Dataset Generator
======================================================================
Verifies:
1. Dataset shape (1,500 total records: 1050 train, 225 val, 225 test).
2. Exact class balance in every split (1:1:1 across Valid / Invalid / Manual Review).
3. Completeness of all 26 specified schema fields.
4. Data integrity and validity of statistics in dataset_stats.json.
"""

import os
import json
import csv
from pathlib import Path


EXPECTED_FIELDS = [
    "claim_id",
    "user_id",
    "product_id",
    "product_category",
    "product_name",
    "brand",
    "model_number",
    "serial_number",
    "purchase_date",
    "purchase_price",
    "retailer",
    "warranty_duration_months",
    "warranty_start_date",
    "warranty_expiry_date",
    "fault_occurrence_date",
    "fault_description",
    "damage_type",
    "claim_submission_date",
    "repair_history",
    "has_receipt",
    "has_warranty_card",
    "has_product_image",
    "has_serial_evidence",
    "serial_number_on_receipt",
    "prior_replacement",
    "class_label",
]

EXPECTED_CLASSES = ["Valid Claim", "Invalid Claim", "Manual Review"]


def test_generated_files_exist():
    for filename in ["train.csv", "val.csv", "test.csv", "dataset_stats.json"]:
        assert os.path.exists(filename) or os.path.exists(os.path.join("dataset", filename)), f"Missing {filename}"


def test_csv_structure_and_stratification():
    split_targets = {
        "train.csv": {"total": 1050, "per_class": 350},
        "val.csv": {"total": 225, "per_class": 75},
        "test.csv": {"total": 225, "per_class": 75},
    }

    for fname, target in split_targets.items():
        fpath = fname if os.path.exists(fname) else os.path.join("dataset", fname)
        assert os.path.exists(fpath), f"File {fpath} does not exist"

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # Check fieldnames
        assert fieldnames == EXPECTED_FIELDS, f"Fieldnames mismatch in {fname}. Got: {fieldnames}"
        assert len(rows) == target["total"], f"Row count mismatch in {fname}: expected {target['total']}, got {len(rows)}"

        # Check stratification balance
        class_counts = {c: 0 for c in EXPECTED_CLASSES}
        for r in rows:
            lbl = r["class_label"]
            assert lbl in class_counts, f"Unexpected class {lbl} in {fname}"
            class_counts[lbl] += 1

        for c, count in class_counts.items():
            assert count == target["per_class"], f"Class {c} in {fname} has {count} rows, expected {target['per_class']}"


def test_dataset_stats_json():
    stats_path = "dataset_stats.json" if os.path.exists("dataset_stats.json") else os.path.join("dataset", "dataset_stats.json")
    assert os.path.exists(stats_path), f"Stats file {stats_path} missing"

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    assert stats["dataset_metadata"]["total_records"] == 1500
    assert stats["dataset_metadata"]["fields_count"] == 26
    assert stats["dataset_metadata"]["overall_class_balance"] == {
        "Valid Claim": 500,
        "Invalid Claim": 500,
        "Manual Review": 500
    }
    assert "splits" in stats
    assert "train" in stats["splits"]
    assert "val" in stats["splits"]
    assert "test" in stats["splits"]
    assert "missing_value_counts" in stats["splits"]["train"]


if __name__ == "__main__":
    print("Running dataset validation tests...")
    test_generated_files_exist()
    test_csv_structure_and_stratification()
    test_dataset_stats_json()
    print("[PASS] All dataset verification tests passed successfully!")
