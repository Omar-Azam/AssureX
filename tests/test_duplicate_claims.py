"""
AssureX Duplicate Claim & Fraud Detection Test Suite
=====================================================
Validates anti-fraud duplicate detection across file hashes and claim history:
1. Document upload SHA-256 hash collision detection flags is_duplicate=True.
2. Concurrent claim on same component and serial flags NO_DUPLICATE_CLAIM failure.
3. Decision engine escalation to Manual Review Required on duplicate flags.
"""

import io
import pytest
from datetime import date, timedelta
from typing import Dict, Any
from starlette.testclient import TestClient
from src.rule_engine import WarrantyRuleEngine
from decision_engine import final_claim_decision


@pytest.mark.duplicates
def test_sha256_document_hash_duplicate_detection(client: TestClient, customer_headers: Dict[str, str]):
    """Verify uploading the exact same document bytes triggers is_duplicate=True."""
    # Register product
    prod_resp = client.post("/api/products", json={
        "product_category": "Laptop",
        "brand": "Lenovo",
        "product_name": "IdeaPad 3",
        "model_number": "82H800",
        "serial_number": "LNV-DUP-DOC-001",
        "purchase_price": 130000.0,
        "purchase_date": str(date.today() - timedelta(days=60)),
        "retailer": "Hafeez Center"
    }, headers=customer_headers)
    product_id = prod_resp.json()["id"]

    # File Claim A
    claim_a_resp = client.post("/api/claims", json={
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=5)),
        "fault_description": "Display cracked",
        "damage_type": "Display Panel Defect",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }, headers=customer_headers)
    claim_a_id = claim_a_resp.json()["claim_id"]

    # Upload Document to Claim A
    duplicate_bytes = b"IDENTICAL SALES INVOICE TEXT - 100% BYTE FOR BYTE REPEAT"
    files_a = {"file": ("invoice.png", io.BytesIO(duplicate_bytes), "image/png")}
    doc_a_resp = client.post(f"/api/claims/{claim_a_id}/documents", files=files_a, data={"file_type": "receipt"}, headers=customer_headers)
    assert doc_a_resp.status_code == 201
    assert doc_a_resp.json()["is_duplicate"] is False

    # File Claim B
    claim_b_resp = client.post("/api/claims", json={
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=1)),
        "fault_description": "Motherboard failed",
        "damage_type": "Motherboard Power Circuit Failure",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }, headers=customer_headers)
    claim_b_id = claim_b_resp.json()["claim_id"]

    # Upload exact same document to Claim B -> Should be flagged as duplicate!
    files_b = {"file": ("invoice.png", io.BytesIO(duplicate_bytes), "image/png")}
    doc_b_resp = client.post(f"/api/claims/{claim_b_id}/documents", files=files_b, data={"file_type": "receipt"}, headers=customer_headers)
    assert doc_b_resp.status_code == 201
    doc_b_data = doc_b_resp.json()
    assert doc_b_data["is_duplicate"] is True
    assert doc_b_data["duplicate_of_document_id"] is not None


@pytest.mark.duplicates
def test_rule_engine_flags_duplicate_claim_history(laptop_policy: Dict[str, Any], valid_claim: Dict[str, Any]):
    """Verify rule engine fails NO_DUPLICATE_CLAIM when historical claim has identical component in review."""
    engine = WarrantyRuleEngine(laptop_policy)
    
    dup_claim = dict(
        valid_claim,
        prior_claims_history=[
            {
                "claim_id": "CLM-PREV-8819",
                "serial_number": valid_claim["serial_number"],
                "affected_component": "system_motherboard_and_processor",
                "claim_status": "PENDING",
                "claim_date": "2026-02-10"
            }
        ]
    )

    res = engine.evaluate_claim(dup_claim)
    assert "NO_DUPLICATE_CLAIM" in res["rules_failed"]
    assert res["manual_review_required"] is True


@pytest.mark.duplicates
def test_decision_engine_escalates_on_duplicate_claim(valid_claim: Dict[str, Any]):
    """Verify decision engine refuses auto-approval when duplicate claim flag is set."""
    rule_result = {
        "rules_failed": ["NO_DUPLICATE_CLAIM"],
        "manual_review_required": True,
        "contradictions": []
    }
    py_res = {"predicted_class": "Likely Valid", "confidence": 0.96}
    gtm_res = {"predicted_class": "Likely Valid", "confidence": 0.94}

    decision = final_claim_decision(valid_claim, py_res, gtm_res, rule_result)
    assert decision["final_decision"] == "Manual Review Required"
    assert "NO_DUPLICATE_CLAIM" in decision["decision_explanation"]["rules_failed"]
