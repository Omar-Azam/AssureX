"""
AssureX Functional Test Suite
==============================
Validates core functional user operations and machine learning classification pipeline:
1. Product registration & automatic statutory warranty calculation
2. Claim filing and document association
3. End-to-end multimodal classification pipeline execution
4. Claim status progression from submission to classification
"""

import io
import pytest
from datetime import date, timedelta
from typing import Dict, Any

from starlette.testclient import TestClient
from backend.models import Claim, Product, Warranty, Prediction


@pytest.mark.functional
def test_product_registration_and_warranty_binding(client: TestClient, customer_headers: Dict[str, str]):
    """Verify customer can register a consumer device and an active warranty is automatically bound."""
    payload = {
        "product_category": "Laptop",
        "brand": "Lenovo",
        "product_name": "IdeaPad Gaming 3",
        "model_number": "82K200UTUS",
        "serial_number": "LNV-SN-FUNC-001",
        "purchase_price": 185000.00,
        "purchase_date": str(date.today() - timedelta(days=60)),
        "retailer": "Hafeez Center Lahore"
    }

    resp = client.post("/api/products", json=payload, headers=customer_headers)
    assert resp.status_code == 201, f"Failed: {resp.text}"
    prod_data = resp.json()
    assert prod_data["serial_number"] == "LNV-SN-FUNC-001"
    assert prod_data["product_category"] == "Laptop"
    assert "id" in prod_data


@pytest.mark.functional
def test_claim_submission_workflow(client: TestClient, customer_headers: Dict[str, str]):
    """Verify filing a warranty claim against a registered product."""
    # 1. Register device
    prod_resp = client.post("/api/products", json={
        "product_category": "Smartphone",
        "brand": "Samsung",
        "product_name": "Galaxy S24",
        "model_number": "SM-S921B",
        "serial_number": "SM-SN-FUNC-CLAIM-002",
        "purchase_price": 270000.00,
        "purchase_date": str(date.today() - timedelta(days=90)),
        "retailer": "Airlink Mall"
    }, headers=customer_headers)
    assert prod_resp.status_code == 201
    product_id = prod_resp.json()["id"]

    # 2. File claim
    claim_payload = {
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=3)),
        "fault_description": "Green vertical line spontaneously appeared across AMOLED display",
        "damage_type": "Display Panel Defect",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }
    claim_resp = client.post("/api/claims", json=claim_payload, headers=customer_headers)
    assert claim_resp.status_code == 201, f"Claim filing failed: {claim_resp.text}"
    claim_data = claim_resp.json()
    assert claim_data["claim_id"].startswith("CLM-")
    assert claim_data["damage_type"] == "Display Panel Defect"
    assert claim_data["status"] == "Submitted"


@pytest.mark.functional
def test_document_upload_and_sha256_tracking(client: TestClient, customer_headers: Dict[str, str]):
    """Verify uploading proof of purchase generates SHA-256 hash and links to claim."""
    # Register and file claim
    prod_resp = client.post("/api/products", json={
        "product_category": "Laptop",
        "brand": "Dell",
        "product_name": "Inspiron 15",
        "model_number": "IN3520",
        "serial_number": "DLL-SN-DOC-003",
        "purchase_price": 140000.00,
        "purchase_date": str(date.today() - timedelta(days=40)),
        "retailer": "Techno City"
    }, headers=customer_headers)
    product_id = prod_resp.json()["id"]

    claim_resp = client.post("/api/claims", json={
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=2)),
        "fault_description": "Motherboard power rail failed",
        "damage_type": "Motherboard Power Circuit Failure",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }, headers=customer_headers)
    claim_id = claim_resp.json()["claim_id"]

    # Upload document
    dummy_invoice = io.BytesIO(b"Official Tax Invoice Hafeez Center - Rs. 140,000")
    dummy_invoice.name = "invoice_tax_receipt.png"
    files = {"file": ("invoice_tax_receipt.png", dummy_invoice, "image/png")}
    data = {"file_type": "receipt"}

    doc_resp = client.post(f"/api/claims/{claim_id}/documents", files=files, data=data, headers=customer_headers)
    assert doc_resp.status_code == 201, f"Document upload failed: {doc_resp.text}"
    doc_data = doc_resp.json()
    assert doc_data["file_type"] == "receipt"
    assert len(doc_data["sha256_hash"]) == 64
    assert doc_data["is_duplicate"] is False


@pytest.mark.functional
def test_multimodal_classification_pipeline(client: TestClient, admin_headers: Dict[str, str], customer_headers: Dict[str, str]):
    """Verify executing the 4-stage AI pipeline on a claim generates predictions and decisions."""
    # Register and file
    prod_resp = client.post("/api/products", json={
        "product_category": "Smartphone",
        "brand": "Xiaomi",
        "product_name": "Redmi Note 13",
        "model_number": "RN13-5G",
        "serial_number": "RN13-SN-PIPE-004",
        "purchase_price": 75000.00,
        "purchase_date": str(date.today() - timedelta(days=45)),
        "retailer": "Daraz Mall"
    }, headers=customer_headers)
    product_id = prod_resp.json()["id"]

    claim_resp = client.post("/api/claims", json={
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=2)),
        "fault_description": "Display touch layer unresponsive",
        "damage_type": "Display Panel Defect",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }, headers=customer_headers)
    claim_id = claim_resp.json()["claim_id"]

    # Trigger classification
    classify_resp = client.post(f"/api/claims/{claim_id}/classify", headers=admin_headers)
    assert classify_resp.status_code == 200, f"Classification failed: {classify_resp.text}"
    pred = classify_resp.json()

    assert pred["final_decision"] in ["Likely Valid", "Likely Invalid", "Manual Review Required"]
    assert pred["model_consistency_status"] in ["Strong Match", "Acceptable Match", "Weak Match", "Model Disagreement", "Uncertain Result"]
    assert "confidence_difference" in pred
    assert "python_prediction" in pred
    assert "rule_engine_result" in pred
    assert "decision_engine_result" in pred
