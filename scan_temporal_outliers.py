"""
AssureX Temporal Outlier Scanner (Pre-Regeneration Audit)
=========================================================

Scans the existing `claims_dataset.csv` for records with:
- product_age_days mismatch > 15 days vs fresh recompute, OR
- remaining_warranty_days mismatch > 15 days vs fresh recompute.

Prints full raw field values of all identified outliers so the underlying
data discrepancy can be inspected before clean regeneration.
"""

import os
import json
from datetime import datetime, date
from typing import Dict, Any, List
import pandas as pd
from claim_metrics import compute_claim_metrics, parse_date


def scan_outliers(
    csv_path: str = "claims_dataset.csv",
    threshold_days: int = 15
) -> List[Dict[str, Any]]:
    """Identifies records where age or remaining days differ by > threshold_days."""
    if not os.path.exists(csv_path):
        csv_path = "dataset/claims_dataset.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Could not find claims_dataset.csv")

    df = pd.read_csv(csv_path)
    outliers = []

    for idx, row in df.iterrows():
        p_str = row.get("purchase_date")
        c_str = row.get("claim_submission_date") or row.get("claim_filing_date")
        f_str = row.get("fault_occurrence_date")
        e_str = row.get("warranty_expiry_date")
        dur = row.get("warranty_duration_months")

        # Fresh recompute using canonical claim_metrics engine
        metrics = compute_claim_metrics(p_str, c_str, dur)
        recomp_age = metrics.product_age_days
        recomp_exp = metrics.warranty_expiry_date
        recomp_rem = metrics.remaining_warranty_days

        # Old rendered values from card / dataset
        p_date = parse_date(p_str)
        f_date = parse_date(f_str)
        e_date = parse_date(e_str)

        rendered_age = (f_date - p_date).days
        rendered_rem = (e_date - f_date).days

        diff_age = abs(recomp_age - rendered_age)
        diff_rem = abs(recomp_rem - rendered_rem)
        diff_exp = abs((parse_date(recomp_exp) - e_date).days)

        if diff_age > threshold_days or diff_rem > threshold_days:
            outliers.append({
                "index": int(idx),
                "claim_id": str(row["claim_id"]),
                "diff_age": int(diff_age),
                "diff_rem": int(diff_rem),
                "diff_exp": int(diff_exp),
                "max_diff": max(int(diff_age), int(diff_rem)),
                "recomputed_values": {
                    "product_age_days": recomp_age,
                    "warranty_expiry_date": recomp_exp,
                    "remaining_warranty_days": recomp_rem,
                    "warranty_status": metrics.warranty_status
                },
                "rendered_values": {
                    "product_age_days": rendered_age,
                    "warranty_expiry_date": str(e_str),
                    "remaining_warranty_days": rendered_rem,
                    "warranty_status": "Active" if rendered_rem >= 0 else "Expired"
                },
                "raw_record": row.to_dict()
            })

    # Sort descending by severity of discrepancy
    outliers.sort(key=lambda x: x["max_diff"], reverse=True)
    return outliers


def print_outlier_report(outliers: List[Dict[str, Any]]) -> None:
    """Formats and prints the full raw field values of identified outlier records."""
    print("=" * 85)
    print(f"ASSUREX OUTLIER AUDIT: RECORDS WITH > 15 DAYS TEMPORAL DISCREPANCY")
    print(f"Total Outlier Records Found: {len(outliers)}")
    print("=" * 85)

    if not outliers:
        print("No outlier records found exceeding 15 days discrepancy.")
        return

    # Print summary table of worst outliers
    print("\nTOP OUTLIER OVERVIEW (WORST 15 RECORDS):")
    print(f"{'Claim ID':<16} | {'Class Label':<15} | {'Diff Age':<9} | {'Diff Rem':<9} | {'Diff Exp':<9} | {'Max Diff':<9}")
    print("-" * 75)
    for o in outliers[:15]:
        cls = str(o["raw_record"].get("class_label", "Unknown"))
        print(f"{o['claim_id']:<16} | {cls:<15} | {o['diff_age']:<9} | {o['diff_rem']:<9} | {o['diff_exp']:<9} | {o['max_diff']:<9}")

    print("\n" + "=" * 85)
    print("FULL RAW FIELD VALUES OF WORST OUTLIER RECORDS:")
    print("=" * 85)

    # Print full raw field values for top outliers
    for i, o in enumerate(outliers[:10], start=1):
        print(f"\n--- OUTLIER #{i}: {o['claim_id']} (Max Discrepancy: {o['max_diff']} days) ---")
        print("  [DISCREPANCY BREAKDOWN]")
        print(f"    * Product Age:      Recomputed={o['recomputed_values']['product_age_days']}d  vs  Rendered={o['rendered_values']['product_age_days']}d  (Diff: {o['diff_age']}d)")
        print(f"    * Expiry Date:      Recomputed={o['recomputed_values']['warranty_expiry_date']}  vs  Rendered={o['rendered_values']['warranty_expiry_date']}  (Diff: {o['diff_exp']}d)")
        print(f"    * Remaining Days:   Recomputed={o['recomputed_values']['remaining_warranty_days']}d  vs  Rendered={o['rendered_values']['remaining_warranty_days']}d  (Diff: {o['diff_rem']}d)")
        print(f"    * Status:           Recomputed={o['recomputed_values']['warranty_status']}  vs  Rendered={o['rendered_values']['warranty_status']}")
        print("  [FULL RAW RECORD FIELDS]")
        for k, v in o["raw_record"].items():
            print(f"    {k:<28}: {v}")


if __name__ == "__main__":
    outliers = scan_outliers("claims_dataset.csv", threshold_days=15)
    print_outlier_report(outliers)
