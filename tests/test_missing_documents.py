"""
AssureX Missing Document & Paperwork Audit Test Suite
=====================================================
Validates mandatory document verification and escalation when paperwork is missing:
1. Missing sales invoice proof of purchase.
2. Missing dealer stamped warranty card.
3. Missing claimant national ID (CNIC) copy.
4. Auto-escalation to Manual Review when mandatory paperwork is incomplete.
5. Reviewer action workflow for requesting missing evidence.
"""

import pytest
from typing import Dict, Any
from src.rule_engine import WarrantyRuleEngine
from decision_engine import final_claim_decision


@pytest.mark.documents
def test_missing_sales_invoice_blocks_auto_approval(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify missing sales invoice fails proof of purchase and mandates manual review."""
    engine = WarrantyRuleEngine(laptop_policy)
    claim_no_invoice = dict(valid_claim, invoice_attached=False, invoice_number=None)

    rule_result = engine.evaluate_claim(claim_no_invoice)
    assert "PROOF_OF_PURCHASE_PRESENT" in rule_result["rules_failed"]
    assert rule_result["manual_review_required"] is True

    # Decision engine test
    python_result = {"predicted_class": "Likely Valid", "confidence": 0.95}
    gtm_result = {"predicted_class": "Likely Valid", "confidence": 0.93}
    decision = final_claim_decision(claim_no_invoice, python_result, gtm_result, rule_result)

    assert decision["final_decision"] == "Manual Review Required"
    assert "PROOF_OF_PURCHASE_PRESENT" in decision["decision_explanation"]["rules_failed"]


@pytest.mark.documents
def test_missing_multiple_mandatory_documents(smartphone_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify missing warranty card and CNIC flags MANDATORY_DOCUMENTS_COMPLETE failure."""
    engine = WarrantyRuleEngine(smartphone_policy)
    claim_missing_docs = dict(
        valid_claim,
        product_category="Smartphone",
        warranty_card_attached=False,
        cnic_attached=False,
        provided_documents=["Photo of Device"]
    )

    result = engine.evaluate_claim(claim_missing_docs)
    assert "MANDATORY_DOCUMENTS_COMPLETE" in result["rules_failed"]
    assert any("mandatory documents" in r.lower() for r in result["reasons"])


@pytest.mark.documents
def test_evidence_needed_listed_in_decision_explanation(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify decision explanation explicitly itemizes missing evidence needed from claimant."""
    engine = WarrantyRuleEngine(laptop_policy)
    claim_no_invoice = dict(valid_claim, invoice_attached=False, invoice_number=None)
    rule_res = engine.evaluate_claim(claim_no_invoice)

    py_res = {"predicted_class": "Likely Valid", "confidence": 0.90}
    gtm_res = {"predicted_class": "Likely Valid", "confidence": 0.88}
    dec = final_claim_decision(claim_no_invoice, py_res, gtm_res, rule_res)

    evidence_needed = dec["decision_explanation"].get("evidence_needed", [])
    assert len(evidence_needed) > 0
    assert any("invoice" in e.lower() or "proof" in e.lower() for e in evidence_needed)
