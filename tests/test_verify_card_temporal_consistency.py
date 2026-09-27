"""
Unit Tests for verify_card_temporal_consistency.py
==================================================

Verifies:
1. Recomputed expiry_date, product_age, and warranty_remaining_days calculation.
2. Across all 1,500 dataset records, maximum discrepancy <= 1 day.
3. Zero records are flagged exceeding the 1-day threshold.
4. Correct identification of the 1-day retail activation window (purchase_date vs warranty_start_date).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify_card_temporal_consistency import (
    run_temporal_audit,
    audit_claim_record,
    add_duration_standard,
    add_calendar_months
)


def test_recomputation_logic():
    """Verify single record recomputation logic."""
    sample_rec = {
        "claim_id": "CLM-TEST-AUDIT-001",
        "purchase_date": "2025-01-10",
        "warranty_start_date": "2025-01-11",
        "warranty_duration_months": 12,
        "warranty_expiry_date": "2026-01-11",
        "fault_occurrence_date": "2025-06-10",
        "claim_submission_date": "2025-06-13",
    }
    res = audit_claim_record(sample_rec, method="standard")

    assert res["recomputed_product_age_days"] == 151
    assert res["rendered_product_age_days"] == 151
    assert res["diff_product_age_days"] == 0

    # 1-day lag from activation window
    assert res["diff_expiry_days"] == 1
    assert res["diff_remaining_days"] == 1
    assert res["max_diff_days"] == 1
    assert res["is_flagged"] is False


def test_full_dataset_temporal_integrity():
    """Verify all 1,500 records satisfy the <= 1 day threshold."""
    audit_df, summary = run_temporal_audit("dataset", method="standard")

    assert summary["total_records"] == 1500
    assert summary["flagged_count"] == 0
    assert summary["max_discrepancy_observed_days"] <= 1

    # Verify per-class and per-split zero flags
    assert (audit_df["is_flagged"] == False).all()


def test_calendar_method_integrity():
    """Verify calendar month addition method also yields zero flagged records."""
    audit_df, summary = run_temporal_audit("dataset", method="calendar")

    assert summary["total_records"] == 1500
    assert summary["flagged_count"] == 0
    assert summary["max_discrepancy_observed_days"] <= 1


if __name__ == "__main__":
    print("Running verify_card_temporal_consistency unit tests...")
    test_recomputation_logic()
    test_full_dataset_temporal_integrity()
    test_calendar_method_integrity()
    print("[PASS] All card temporal consistency verification tests passed successfully!")
