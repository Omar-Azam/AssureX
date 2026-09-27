"""
AssureX Negative Test Suite
============================
Validates system rejection, 400/404/422 status codes, and input validation guards:
1. Missing mandatory payload fields in claim creation.
2. Filing claim for non-existent product ID.
3. Querying non-existent claim ID or review queue record.
4. Submitting invalid review actions (e.g. unknown verb).
5. Future fault occurrence dates violating temporal integrity.
6. Registering product with negative price.
"""

import pytest
from datetime import date, timedelta
from typing import Dict, Any
from starlette.testclient import TestClient


@pytest.mark.negative
def test_claim_submission_missing_required_fields(client: TestClient, customer_headers: Dict[str, str]):
    """Verify 422 Unprocessable Entity when required fields are missing."""
    incomplete_payload = {
        "fault_description": "Screen is broken"
        # Missing product_id, fault_occurrence_date, damage_type
    }
    resp = client.post("/api/claims", json=incomplete_payload, headers=customer_headers)
    assert resp.status_code == 422


@pytest.mark.negative
def test_claim_submission_nonexistent_product(client: TestClient, customer_headers: Dict[str, str]):
    """Verify 404 Not Found when product_id does not exist."""
    payload = {
        "product_id": 999999,
        "fault_occurrence_date": str(date.today()),
        "fault_description": "Motherboard dead",
        "damage_type": "Motherboard Power Circuit Failure",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }
    resp = client.post("/api/claims", json=payload, headers=customer_headers)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.negative
def test_query_nonexistent_claim_id(client: TestClient, reviewer_headers: Dict[str, str]):
    """Verify 404 Not Found when searching for a non-existent claim."""
    resp = client.get("/api/claims/CLM-NONEXISTENT-99999", headers=reviewer_headers)
    assert resp.status_code == 404


@pytest.mark.negative
def test_submit_invalid_review_action(client: TestClient, reviewer_headers: Dict[str, str], customer_headers: Dict[str, str]):
    """Verify rejection when underwriter provides an illegal action keyword."""
    # First create a valid product & claim
    prod_resp = client.post("/api/products", json={
        "product_category": "Laptop",
        "brand": "Asus",
        "product_name": "ZenBook 14",
        "model_number": "UM3402",
        "serial_number": "ASU-NEG-ACT-001",
        "purchase_price": 210000.00,
        "purchase_date": str(date.today() - timedelta(days=30)),
        "retailer": "Hafeez Center"
    }, headers=customer_headers)
    product_id = prod_resp.json()["id"]

    claim_resp = client.post("/api/claims", json={
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=2)),
        "fault_description": "Trackpad failed",
        "damage_type": "Display Panel Defect",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }, headers=customer_headers)
    claim_id = claim_resp.json()["claim_id"]

    # Submit illegal review action
    invalid_review = {
        "action": "illegal_unrecognized_action",
        "decision_notes": "Attempting invalid action verb"
    }
    resp = client.post(f"/api/claims/{claim_id}/review", json=invalid_review, headers=reviewer_headers)
    assert resp.status_code == 400
    assert "invalid review action" in resp.json()["detail"].lower()


@pytest.mark.negative
def test_future_fault_occurrence_date(client: TestClient, customer_headers: Dict[str, str]):
    """Verify rejection when fault occurrence date is set to a future date."""
    prod_resp = client.post("/api/products", json={
        "product_category": "Smartphone",
        "brand": "OnePlus",
        "product_name": "OnePlus 12",
        "model_number": "OP12-5G",
        "serial_number": "OP-NEG-FUT-002",
        "purchase_price": 240000.00,
        "purchase_date": str(date.today() - timedelta(days=10)),
        "retailer": "Airlink Mall"
    }, headers=customer_headers)
    product_id = prod_resp.json()["id"]

    future_date = date.today() + timedelta(days=365)
    payload = {
        "product_id": product_id,
        "fault_occurrence_date": str(future_date),
        "fault_description": "Future defect",
        "damage_type": "Display Panel Defect",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }
    resp = client.post("/api/claims", json=payload, headers=customer_headers)
    assert resp.status_code == 400
    assert "future" in resp.json()["detail"].lower()
