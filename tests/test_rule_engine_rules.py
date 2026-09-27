"""
AssureX Policy Rule Engine Comprehensive Test Suite
===================================================
One dedicated unit test per warranty policy rule across all 11 evaluation dimensions:
1.  Rule: DATA_INTEGRITY_CHECK
2.  Rule: PROOF_OF_PURCHASE_PRESENT
3.  Rule: SERIAL_NUMBER_MATCH
4.  Rule: WARRANTY_ACTIVE
5.  Rule: WITHIN_REPORTING_PERIOD
6.  Rule: FAULT_COVERED
7.  Rule: NO_EXCLUDED_DAMAGE
8.  Rule: PRIOR_REPAIR_HISTORY_CHECK
9.  Rule: AUTHORIZED_REPAIR_INTEGRITY
10. Rule: MANDATORY_DOCUMENTS_COMPLETE
11. Rule: NO_DUPLICATE_CLAIM
"""

import pytest
from typing import Dict, Any
from src.rule_engine import WarrantyRuleEngine


@pytest.fixture
def base_laptop_claim() -> Dict[str, Any]:
    return {
        "claim_id": "CLM-RULE-BASE-001",
        "product_category": "Laptop",
        "serial_number": "LNV-IDEAPAD-82H8-9921",
        "invoice_serial_number": "LNV-IDEAPAD-82H8-9921",
        "purchase_date": "2025-06-15",
        "fault_date": "2026-02-10",
        "claim_date": "2026-02-12",
        "affected_component": "system_motherboard_and_processor",
        "fault_description": "Motherboard power circuit failure",
        "invoice_number": "INV-10291",
        "invoice_attached": True,
        "warranty_card_attached": True,
        "cnic_attached": True,
        "provided_documents": [
            "Original Retail Purchase Invoice (with NTN/STRN)",
            "Official Dealer Stamped Warranty Card",
            "Claimant National Identity Card (CNIC) Copy"
        ],
        "liquid_damage_detected": False,
        "motherboard_lci_triggered": False,
        "physical_damage": False,
        "tamper_seal_broken": False,
        "unauthorized_third_party_repair": False,
        "prior_repairs": []
    }


# -----------------------------------------------------------------------------
# RULE 1: DATA_INTEGRITY_CHECK
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_data_integrity_check(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 1: Fails DATA_INTEGRITY_CHECK when claim date precedes purchase date."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Valid case
    valid_res = engine.evaluate_claim(base_laptop_claim)
    assert "DATA_INTEGRITY_CHECK" in valid_res["rules_passed"]

    # Contradictory case: claim date before purchase date
    corrupt_claim = dict(base_laptop_claim, purchase_date="2026-05-01", claim_date="2026-02-01")
    corrupt_res = engine.evaluate_claim(corrupt_claim)
    assert "DATA_INTEGRITY_CHECK" in corrupt_res["rules_failed"]
    assert corrupt_res["manual_review_required"] is True


# -----------------------------------------------------------------------------
# RULE 2: PROOF_OF_PURCHASE_PRESENT
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_proof_of_purchase_present(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 2: Evaluates mandatory retail invoice proof of purchase."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Valid case
    valid_res = engine.evaluate_claim(base_laptop_claim)
    assert "PROOF_OF_PURCHASE_PRESENT" in valid_res["rules_passed"]

    # Missing invoice
    missing_pop = dict(base_laptop_claim, invoice_attached=False, invoice_number=None)
    res = engine.evaluate_claim(missing_pop)
    assert "PROOF_OF_PURCHASE_PRESENT" in res["rules_failed"]


# -----------------------------------------------------------------------------
# RULE 3: SERIAL_NUMBER_MATCH
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_serial_number_match(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 3: Evaluates consistency between chassis serial and receipt serial."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Valid matching
    valid_res = engine.evaluate_claim(base_laptop_claim)
    assert "SERIAL_NUMBER_MATCH" in valid_res["rules_passed"]

    # Mismatched serial (paperwork swap fraud)
    mismatched = dict(base_laptop_claim, invoice_serial_number="DIFFERENT-SERIAL-9999")
    mismatch_res = engine.evaluate_claim(mismatched)
    assert "SERIAL_NUMBER_MATCH" in mismatch_res["rules_failed"]


# -----------------------------------------------------------------------------
# RULE 4: WARRANTY_ACTIVE
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_warranty_active(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 4: Evaluates statutory and manufacturer coverage lifespan."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Active coverage
    res_active = engine.evaluate_claim(base_laptop_claim)
    assert "WARRANTY_ACTIVE" in res_active["rules_passed"]

    # Expired coverage (purchased 3 years ago on 1-year policy)
    expired_claim = dict(base_laptop_claim, purchase_date="2022-01-01", claim_date="2026-02-01", fault_date="2026-01-28")
    res_exp = engine.evaluate_claim(expired_claim)
    assert "WARRANTY_ACTIVE" in res_exp["rules_failed"]


# -----------------------------------------------------------------------------
# RULE 5: WITHIN_REPORTING_PERIOD
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_within_reporting_period(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 5: Evaluates notification time between defect occurrence and claim filing."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Prompt notification (filed 2 days after fault)
    res_prompt = engine.evaluate_claim(base_laptop_claim)
    assert "WITHIN_REPORTING_PERIOD" in res_prompt["rules_passed"]


# -----------------------------------------------------------------------------
# RULE 6: FAULT_COVERED
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_fault_covered(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 6: Evaluates component failure coverage against policy schedules."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Covered motherboard circuit defect
    res_covered = engine.evaluate_claim(base_laptop_claim)
    assert "FAULT_COVERED" in res_covered["rules_passed"]


# -----------------------------------------------------------------------------
# RULE 7: NO_EXCLUDED_DAMAGE
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_no_excluded_damage(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 7: Enforces strict exclusions for accidental drops and liquid contamination."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Clean claim
    res_clean = engine.evaluate_claim(base_laptop_claim)
    assert "NO_EXCLUDED_DAMAGE" in res_clean["rules_passed"]

    # Liquid damaged claim
    liquid_claim = dict(base_laptop_claim, liquid_damage_detected=True, motherboard_lci_triggered=True)
    res_liquid = engine.evaluate_claim(liquid_claim)
    assert "NO_EXCLUDED_DAMAGE" in res_liquid["rules_failed"]


# -----------------------------------------------------------------------------
# RULE 8: PRIOR_REPAIR_HISTORY_CHECK
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_prior_repair_history_check(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 8: Tracks servicing frequency and lemon replacement thresholds."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Clean 0 prior repairs
    res_clean = engine.evaluate_claim(base_laptop_claim)
    assert "PRIOR_REPAIR_HISTORY_CHECK" in res_clean["rules_passed"]

    # Repeat repairs triggering underwriter review
    lemon_claim = dict(base_laptop_claim, prior_repairs=[
        {"repair_date": "2025-08-01", "repaired_component": "system_motherboard_and_processor", "authorized_service_center": True},
        {"repair_date": "2025-11-01", "repaired_component": "system_motherboard_and_processor", "authorized_service_center": True}
    ])
    res_lemon = engine.evaluate_claim(lemon_claim)
    assert res_lemon["manual_review_required"] is True
    assert any("lemon" in w.lower() for w in res_lemon["warnings"])


# -----------------------------------------------------------------------------
# RULE 9: AUTHORIZED_REPAIR_INTEGRITY
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_authorized_repair_integrity(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 9: Voids warranty if unit underwent unauthorized bazaar rework or broken tamper seals."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Authorized history
    res_auth = engine.evaluate_claim(base_laptop_claim)
    assert "AUTHORIZED_REPAIR_INTEGRITY" in res_auth["rules_passed"]

    # Unauthorized rework in local bazaar
    unauth_claim = dict(base_laptop_claim, unauthorized_third_party_repair=True, tamper_seal_broken=True)
    res_unauth = engine.evaluate_claim(unauth_claim)
    assert "AUTHORIZED_REPAIR_INTEGRITY" in res_unauth["rules_failed"]


# -----------------------------------------------------------------------------
# RULE 10: MANDATORY_DOCUMENTS_COMPLETE
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_mandatory_documents_complete(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 10: Ensures all regulatory and audit documents (Invoice, Warranty Card, CNIC) are present."""
    engine = WarrantyRuleEngine(laptop_policy)

    # Complete documentation
    res_complete = engine.evaluate_claim(base_laptop_claim)
    assert "MANDATORY_DOCUMENTS_COMPLETE" in res_complete["rules_passed"]

    # Incomplete documentation (missing warranty card and CNIC)
    incomplete_claim = dict(base_laptop_claim, warranty_card_attached=False, cnic_attached=False, provided_documents=[])
    res_inc = engine.evaluate_claim(incomplete_claim)
    assert "MANDATORY_DOCUMENTS_COMPLETE" in res_inc["rules_failed"]


# -----------------------------------------------------------------------------
# RULE 11: NO_DUPLICATE_CLAIM
# -----------------------------------------------------------------------------
@pytest.mark.rule_engine
def test_rule_no_duplicate_claim(laptop_policy: Dict[str, Any], base_laptop_claim: Dict[str, Any]):
    """Rule 11: Prevents double-dipping and concurrent claims on identical components."""
    engine = WarrantyRuleEngine(laptop_policy)

    # First-time claim
    res_first = engine.evaluate_claim(base_laptop_claim)
    assert "NO_DUPLICATE_CLAIM" in res_first["rules_passed"]

    # Concurrent duplicate claim in history
    duplicate_claim = dict(base_laptop_claim, prior_claims_history=[
        {
            "claim_id": "CLM-PREV-001",
            "serial_number": base_laptop_claim["serial_number"],
            "claim_status": "PENDING",
            "claim_date": "2026-02-10",
            "affected_component": "system_motherboard_and_processor"
        }
    ])
    res_dup = engine.evaluate_claim(duplicate_claim)
    assert "NO_DUPLICATE_CLAIM" in res_dup["rules_failed"]
    assert res_dup["manual_review_required"] is True
