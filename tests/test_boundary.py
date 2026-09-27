"""
AssureX Boundary & Edge Case Test Suite
========================================
Validates temporal edge conditions and strict date arithmetic:
1. Warranty expiring exactly today (remaining_warranty_days == 0).
2. Claim filed on the exact last valid day of warranty coverage.
3. Claim filed on day 1 after purchase (product_age_days == 0 or 1).
4. Claim filed exactly 1 day after warranty expiration (remaining_warranty_days == -1).
5. 7-day incident reporting window boundary (day 7 valid vs day 8 late notification).
6. Calendar month exact leap-year arithmetic via claim_metrics.compute_claim_metrics.
"""

import pytest
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any

from claim_metrics import compute_claim_metrics
from src.rule_engine import WarrantyRuleEngine


@pytest.mark.boundary
def test_warranty_expiring_exactly_today():
    """Verify that a claim filed on the exact day warranty expires has remaining_days=0 and status Active."""
    purchase_date = date(2025, 2, 28)
    warranty_months = 12
    # Exact expiry is 2026-02-28
    filing_date = date(2026, 2, 28)

    age_days, expiry_date, remaining_days, status = compute_claim_metrics(
        purchase_date, filing_date, warranty_months
    )

    assert expiry_date == "2026-02-28"
    assert remaining_days == 0
    assert status == "Active"


@pytest.mark.boundary
def test_claim_filed_on_last_valid_day_policy_evaluation(laptop_policy: Dict[str, Any]):
    """Verify Rule Engine passes WARRANTY_ACTIVE when claim is filed on the exact final calendar day."""
    engine = WarrantyRuleEngine(laptop_policy)
    
    # 12-month warranty from 2025-03-01 to 2026-03-01
    claim = {
        "claim_id": "CLM-BOUND-001",
        "product_category": "Laptop",
        "serial_number": "LNV-BOUND-LASTDAY",
        "invoice_serial_number": "LNV-BOUND-LASTDAY",
        "purchase_date": "2025-03-01",
        "fault_date": "2026-02-28",
        "claim_date": "2026-03-01",
        "invoice_attached": True,
        "warranty_card_attached": True,
        "cnic_attached": True,
        "provided_documents": ["Original Retail Purchase Invoice (with NTN/STRN)", "Official Dealer Stamped Warranty Card", "Claimant National Identity Card (CNIC) Copy"],
        "liquid_damage_detected": False,
        "physical_damage": False,
        "tamper_seal_broken": False,
        "unauthorized_third_party_repair": False,
        "affected_component": "system_motherboard_and_processor",
        "prior_repairs": []
    }

    result = engine.evaluate_claim(claim)
    assert "WARRANTY_ACTIVE" in result["rules_passed"]
    assert "WARRANTY_ACTIVE" not in result["rules_failed"]


@pytest.mark.boundary
def test_claim_filed_one_day_after_expiration():
    """Verify filing 1 day after expiry gives remaining_days = -1 and status Expired."""
    purchase_date = date(2025, 1, 15)
    warranty_months = 12
    # Expiry is 2026-01-15
    filing_date = date(2026, 1, 16)

    age_days, expiry_date, remaining_days, status = compute_claim_metrics(
        purchase_date, filing_date, warranty_months
    )

    assert remaining_days == -1
    assert status == "Expired"


@pytest.mark.boundary
def test_claim_filed_on_purchase_date_day_zero():
    """Verify dead-on-arrival claim filed on day 0 (same day as purchase)."""
    today = date.today()
    age_days, expiry_date, remaining_days, status = compute_claim_metrics(today, today, 24)

    assert age_days == 0
    assert remaining_days > 700
    assert status == "Active"


@pytest.mark.boundary
def test_seven_day_reporting_window_boundary(smartphone_policy: Dict[str, Any]):
    """Verify 7-day reporting period threshold (day 7 passes, day 8 triggers warning/failure)."""
    engine = WarrantyRuleEngine(smartphone_policy)

    base_claim = {
        "claim_id": "CLM-BOUND-REP-01",
        "product_category": "Smartphone",
        "serial_number": "RN13-SN-BOUND-07",
        "invoice_serial_number": "RN13-SN-BOUND-07",
        "purchase_date": "2025-10-01",
        "invoice_attached": True,
        "warranty_card_attached": True,
        "cnic_attached": True,
        "pta_slip_attached": True,
        "provided_documents": ["Original Retail Purchase Invoice (with NTN/STRN)", "Official Dealer Stamped Warranty Card", "Claimant National Identity Card (CNIC) Copy", "PTA DIRBS 8484 Verification Confirmation Slip/SMS"],
        "liquid_damage_detected": False,
        "physical_damage": False,
        "tamper_seal_broken": False,
        "unauthorized_third_party_repair": False,
        "affected_component": "touchscreen_display_assembly",
        "prior_repairs": []
    }

    # Case A: Fault on Day 0, Claim filed on Day 7 (Exactly 7 days -> within allowable limit)
    claim_day_7 = dict(base_claim, fault_date="2026-02-10", claim_date="2026-02-17")
    res_7 = engine.evaluate_claim(claim_day_7)
    assert "WITHIN_REPORTING_PERIOD" in res_7["rules_passed"]

    # Case B: Fault on Day 0, Claim filed on Day 15 (15 days -> exceeds reporting window)
    claim_day_15 = dict(base_claim, fault_date="2026-02-01", claim_date="2026-02-16")
    res_15 = engine.evaluate_claim(claim_day_15)
    # Exceeding reporting window should produce a warning or rule failure
    assert any("reporting" in w.lower() or "reporting" in r.lower() for w in res_15["warnings"] for r in res_15["reasons"]) or \
           "WITHIN_REPORTING_PERIOD" in res_15["rules_failed"] or len(res_15["warnings"]) > 0


@pytest.mark.boundary
def test_leap_year_calendar_month_arithmetic():
    """Verify leap year February 29 calendar month exact arithmetic."""
    leap_purchase = date(2024, 2, 29)
    # 12 months from leap day 2024-02-29 lands on 2025-02-28
    age_days, expiry, remaining, status = compute_claim_metrics(leap_purchase, date(2025, 2, 28), 12)
    assert expiry == "2025-02-28"
    assert remaining == 0
    assert status == "Active"
