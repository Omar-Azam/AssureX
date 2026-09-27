"""
AssureX Serial Number & Hardware Identity Test Suite
====================================================
Validates chassis serial verification, anti-paperwork swapping, and dual-IMEI integrity:
1. Invoice serial number differs from chassis barcode serial.
2. Rule engine fails SERIAL_NUMBER_MATCH on discrepancy.
3. Smartphone dual-IMEI matching (matching either IMEI 1 or IMEI 2).
4. Receipt OCR cross-verification flags serial number mismatches.
"""

import pytest
from typing import Dict, Any
from src.rule_engine import WarrantyRuleEngine
from receipt_ocr import cross_verify_receipt_with_claim
from decision_engine import final_claim_decision


@pytest.mark.serials
def test_chassis_serial_mismatch_fails_rule(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify rule engine fails SERIAL_NUMBER_MATCH when receipt serial differs from device serial."""
    engine = WarrantyRuleEngine(laptop_policy)
    swapped_claim = dict(
        valid_claim,
        serial_number="LNV-CHASSIS-ORIGINAL-01",
        invoice_serial_number="LNV-RECEIPT-SWAPPED-99"
    )

    res = engine.evaluate_claim(swapped_claim)
    assert "SERIAL_NUMBER_MATCH" in res["rules_failed"]
    assert any("mismatch" in r.lower() or "serial" in r.lower() for r in res["reasons"])


@pytest.mark.serials
def test_smartphone_dual_imei_matching(smartphone_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify smartphone matches if invoice references either IMEI 1 or IMEI 2."""
    engine = WarrantyRuleEngine(smartphone_policy)

    # Invoice references IMEI 2
    claim_imei_2 = dict(
        valid_claim,
        product_category="Smartphone",
        serial_number="860492058291048",
        imei_1="860492058291048",
        imei_2="860492058291055",
        invoice_serial_number="860492058291055"
    )
    res = engine.evaluate_claim(claim_imei_2)
    assert "SERIAL_NUMBER_MATCH" in res["rules_passed"]


@pytest.mark.serials
def test_cross_verify_receipt_flags_serial_swapping():
    """Verify cross_verify_receipt_with_claim flags serial discrepancy as unverified."""
    extracted_ocr = {
        "fields": {
            "serial_number": {"value": "RN13-STORE-STOCK-001", "confidence": "HIGH"},
            "purchase_date": {"value": "2025-11-20", "confidence": "HIGH"},
            "retailer": {"value": "Airlink Store", "confidence": "HIGH"}
        }
    }
    claim_data = {
        "serial_number": "RN13-USER-DIFFERENT-002",
        "purchase_date": "2025-11-20",
        "retailer": "Airlink Store"
    }

    result = cross_verify_receipt_with_claim(extracted_ocr, claim_data)
    assert result["is_verified"] is False
    assert any("Serial Number Mismatch" in d for d in result["discrepancies"])


@pytest.mark.serials
def test_decision_engine_rejects_or_escalates_on_serial_mismatch(valid_claim: Dict[str, Any]):
    """Verify decision engine does not auto-approve a claim with serial mismatch."""
    rule_res = {
        "rules_failed": ["SERIAL_NUMBER_MATCH"],
        "manual_review_required": True,
        "contradictions": []
    }
    py_res = {"predicted_class": "Likely Valid", "confidence": 0.90}
    gtm_res = {"predicted_class": "Likely Valid", "confidence": 0.90}

    decision = final_claim_decision(valid_claim, py_res, gtm_res, rule_res)
    assert decision["final_decision"] in ["Manual Review Required", "Likely Invalid"]
    assert "SERIAL_NUMBER_MATCH" in decision["decision_explanation"]["rules_failed"]
