"""
Unit Tests for audit_card_metrics_recomputation.py
==================================================

Verifies:
1. Recomputation functions execute cleanly.
2. Correct match/mismatch count calculations across metrics.
3. Accurate side-by-side comparison dataframe generation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_card_metrics_recomputation import run_audit


def test_audit_execution_and_summary():
    """Verify run_audit completes and returns expected summary counts."""
    audit_df, summary = run_audit("claims_dataset.csv")

    assert summary["total_records"] == 1500
    assert summary["product_age_exact_matches"] == 1500
    assert summary["product_age_mismatches"] == 0

    assert summary["expiry_date_exact_matches"] == 1500
    assert summary["expiry_date_mismatches"] == 0

    assert summary["remaining_days_exact_matches"] == 1500
    assert summary["remaining_days_mismatches"] == 0

    assert len(audit_df) == 1500


def test_dataframe_columns():
    """Verify expected columns are present in the resulting dataframe."""
    audit_df, _ = run_audit("claims_dataset.csv")
    expected_cols = [
        "claim_id", "purchase_date", "claim_filing_date", "fault_occurrence_date",
        "warranty_duration_months", "recomputed_product_age", "rendered_product_age",
        "match_product_age", "diff_product_age", "recomputed_expiry_date",
        "rendered_expiry_date", "match_expiry_date", "diff_expiry_days",
        "recomputed_remaining_days", "rendered_remaining_days", "match_remaining_days",
        "diff_remaining_days", "has_any_mismatch"
    ]
    for col in expected_cols:
        assert col in audit_df.columns, f"Missing column: {col}"


if __name__ == "__main__":
    print("Running audit_card_metrics_recomputation unit tests...")
    test_audit_execution_and_summary()
    test_dataframe_columns()
    print("[PASS] All audit recomputation unit tests passed successfully!")
