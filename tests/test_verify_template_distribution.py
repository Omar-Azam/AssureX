"""
Unit & Validation Tests for verify_template_distribution.py
===========================================================
Verifies:
1. Scanning claim_cards/train/ identifies 2,100 total PNG cards.
2. Cross-tabulation generates exact 350 counts per template per class.
3. Both Template 1 (Light Executive) and Template 2 (Dark Telemetry)
   have exact 50.0% / 50.0% parity in Valid Claim, Invalid Claim, and Manual Review.
4. Statistical audit confirms zero disproportion or visual template bias.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify_template_distribution import (
    scan_train_claim_cards,
    generate_cross_tab,
    audit_distribution_balance,
    TARGET_CLASSES,
    TEMPLATE_NAMES
)


def test_scan_and_counts():
    """Verify scanning returns exactly 2,100 records."""
    df = scan_train_claim_cards("claim_cards/train")
    assert len(df) == 2100, f"Expected 2100 records, got {len(df)}"
    assert set(df["class_label"].unique()) == set(TARGET_CLASSES)
    assert set(df["variation_code"].unique()) == {"v1", "v2"}


def test_cross_tab_parity():
    """Verify cross-tab has exactly 350 cards in every cell."""
    df = scan_train_claim_cards("claim_cards/train")
    count_crosstab, pct_crosstab = generate_cross_tab(df)

    # Check cell counts
    for tmpl in ["Template 1 (Light Executive)", "Template 2 (Dark Telemetry)"]:
        for cls in TARGET_CLASSES:
            cnt = count_crosstab.loc[tmpl, cls]
            assert cnt == 350, f"Expected 350 for ({tmpl}, {cls}), got {cnt}"

    # Check row and column totals
    assert count_crosstab.loc["Total", "Valid Claim"] == 700
    assert count_crosstab.loc["Total", "Invalid Claim"] == 700
    assert count_crosstab.loc["Total", "Manual Review"] == 700
    assert count_crosstab.loc["Template 1 (Light Executive)", "Total"] == 1050
    assert count_crosstab.loc["Template 2 (Dark Telemetry)", "Total"] == 1050
    assert count_crosstab.loc["Total", "Total"] == 2100

    # Check percentages are exactly 50.0%
    for tmpl in ["Template 1 (Light Executive)", "Template 2 (Dark Telemetry)"]:
        for cls in TARGET_CLASSES:
            pct = pct_crosstab.loc[tmpl, cls]
            assert round(pct, 2) == 50.00, f"Expected 50.00% for ({tmpl}, {cls}), got {pct}"


def test_audit_balance():
    """Verify audit_distribution_balance flags perfect 1:1 balance."""
    df = scan_train_claim_cards("claim_cards/train")
    count_crosstab, _ = generate_cross_tab(df)
    audit = audit_distribution_balance(count_crosstab, tolerance_pct=5.0)

    assert audit["is_perfectly_balanced"] is True
    assert audit["is_roughly_equal"] is True
    assert audit["max_deviation_from_50"] == 0.0
    assert len(audit["alerts"]) == 0


if __name__ == "__main__":
    print("Running verify_template_distribution unit tests...")
    test_scan_and_counts()
    test_cross_tab_parity()
    test_audit_balance()
    print("[PASS] All template distribution verification tests passed successfully!")
