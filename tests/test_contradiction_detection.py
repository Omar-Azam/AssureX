"""
AssureX Contradiction & Logical Integrity Test Suite
=====================================================
Validates cross-field contradiction detection in rule engine and decision engine:
1. Temporal contradiction: Fault date prior to purchase date.
2. Temporal contradiction: Claim filing date prior to purchase date.
3. Temporal contradiction: Fault occurrence date after claim filing date.
4. Logical contradiction: Claiming 0 prior repairs while repair history records exist.
5. Decision engine extraction of detected contradictions into decision explanation.
"""

import pytest
from typing import Dict, Any
from src.rule_engine import WarrantyRuleEngine
from decision_engine import extract_and_evaluate_contradictions, final_claim_decision


@pytest.mark.contradictions
def test_fault_date_prior_to_purchase_date(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify contradiction detected when defect occurs before hardware was bought."""
    engine = WarrantyRuleEngine(laptop_policy)
    corrupted = dict(valid_claim, purchase_date="2025-10-01", fault_date="2025-08-15")
    
    res = engine.evaluate_claim(corrupted)
    assert "DATA_INTEGRITY_CHECK" in res["rules_failed"]
    assert any("prior to" in r.lower() or "purchase date" in r.lower() for r in res["reasons"])


@pytest.mark.contradictions
def test_claim_filing_date_prior_to_purchase_date(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify contradiction detected when claim is filed before purchase date."""
    engine = WarrantyRuleEngine(laptop_policy)
    corrupted = dict(valid_claim, purchase_date="2026-03-01", claim_date="2026-01-15")
    
    res = engine.evaluate_claim(corrupted)
    assert "DATA_INTEGRITY_CHECK" in res["rules_failed"]


@pytest.mark.contradictions
def test_fault_date_after_claim_date(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify contradiction when incident date is listed after claim submission date."""
    engine = WarrantyRuleEngine(laptop_policy)
    corrupted = dict(valid_claim, fault_date="2026-02-25", claim_date="2026-02-10")
    
    res = engine.evaluate_claim(corrupted)
    assert "DATA_INTEGRITY_CHECK" in res["rules_failed"]
    assert any("before the fault" in r.lower() or "fault incident date" in r.lower() or "after" in r.lower() for r in res["reasons"])


@pytest.mark.contradictions
def test_decision_engine_captures_contradictions():
    """Verify decision_engine extracts contradictions and incorporates them into explanations."""
    claim_record = {
        "claim_id": "CLM-CONTRA-001",
        "purchase_date": "2026-04-01",
        "fault_date": "2026-01-10",
        "claim_date": "2026-01-15",
        "repair_history": "0 repairs",
        "prior_repairs": [{"repair_date": "2025-12-01", "repaired_component": "battery"}]
    }
    
    contradictions = extract_and_evaluate_contradictions(claim_record)
    assert len(contradictions) >= 1
    assert any("earlier than" in c.lower() or "purchase date" in c.lower() for c in contradictions)

    # Decision engine integration
    python_result = {"predicted_class": "Likely Valid", "confidence": 0.95}
    gtm_result = {"predicted_class": "Likely Valid", "confidence": 0.92}
    rule_result = {"rules_failed": ["DATA_INTEGRITY_CHECK"], "manual_review_required": True, "contradictions": contradictions}

    decision = final_claim_decision(claim_record, python_result, gtm_result, rule_result)
    assert decision["final_decision"] == "Manual Review Required"
    assert len(decision["decision_explanation"]["contradictions"]) >= 1
