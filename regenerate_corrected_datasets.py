"""
AssureX Clean Dataset Regeneration Script
==========================================

Regenerates `claims_dataset.csv` and split datasets (`train.csv`, `val.csv`, `test.csv`)
using the unified `claim_metrics.compute_claim_metrics()` engine.

Maintains all non-date fields intact:
- product (category, name, brand, model, serial, price, retailer)
- fault (fault_occurrence_date, fault_description, damage_type)
- documents (has_receipt, has_warranty_card, has_product_image, has_serial_evidence, serial_on_receipt)
- contract & adjudication (repair_history, prior_replacement, class_label)

Only updates date-derived fields:
- `warranty_expiry_date` set to `compute_claim_metrics().warranty_expiry_date`
"""

import os
import shutil
import pandas as pd
from claim_metrics import compute_claim_metrics, parse_date


def regenerate_all_datasets():
    csv_main = "claims_dataset.csv"
    if not os.path.exists(csv_main):
        raise FileNotFoundError(f"{csv_main} not found.")

    df_main = pd.read_csv(csv_main)
    total_records = len(df_main)
    print(f"Loaded {total_records} records from {csv_main}")

    # Track how many expiry dates actually changed
    changed_count = 0
    new_expiry_dates = []

    for idx, row in df_main.iterrows():
        p_date = row["purchase_date"]
        c_date = row["claim_submission_date"]
        w_months = row["warranty_duration_months"]
        old_expiry = str(row["warranty_expiry_date"]).strip()

        metrics = compute_claim_metrics(p_date, c_date, w_months)
        new_exp = metrics.warranty_expiry_date
        new_expiry_dates.append(new_exp)

        if old_expiry != new_exp:
            changed_count += 1

    df_main["warranty_expiry_date"] = new_expiry_dates
    print(f"Updated {changed_count} / {total_records} records with corrected calendar-month expiry dates.")

    # Save to claims_dataset.csv
    df_main.to_csv(csv_main, index=False)
    print(f"Saved corrected {csv_main}")

    # Also update dataset/claims_dataset.csv if exists
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    df_main.to_csv(os.path.join(dataset_dir, "claims_dataset.csv"), index=False)
    print(f"Saved corrected dataset/claims_dataset.csv")

    # Map claim_id to new expiry date
    expiry_map = dict(zip(df_main["claim_id"], df_main["warranty_expiry_date"]))

    # Synchronize split files: train.csv, val.csv, test.csv in root and dataset/
    splits = ["train.csv", "val.csv", "test.csv"]
    for s in splits:
        for folder in [".", dataset_dir]:
            sp_path = os.path.normpath(os.path.join(folder, s))
            if os.path.exists(sp_path):
                sp_df = pd.read_csv(sp_path)
                sp_df["warranty_expiry_date"] = sp_df["claim_id"].map(expiry_map)
                sp_df.to_csv(sp_path, index=False)
                print(f"Synchronized split {sp_path} ({len(sp_df)} rows)")

    print("\n[SUCCESS] Dataset regeneration complete.")
    return total_records, changed_count


if __name__ == "__main__":
    regenerate_all_datasets()
