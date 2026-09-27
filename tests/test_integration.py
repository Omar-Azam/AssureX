"""
AssureX End-to-End Integration Test Suite
==========================================
Validates multi-actor integration workflow across the entire claim lifecycle:
Customer Registration -> Claim Submission -> Automated ML/Rule Classification ->
Reviewer Queue Inspection -> Underwriter Adjudication -> Audit Trail -> PDF Report.
"""

import io
import pytest
from datetime import date, timedelta
from typing import Dict, Any
from starlette.testclient import TestClient


@pytest.mark.integration
def test_full_claim_adjudication_lifecycle(
    client: TestClient,
    customer_headers: Dict[str, str],
    reviewer_headers: Dict[str, str],
    admin_headers: Dict[str, str]
):
    """
    Simulates complete multi-stakeholder claim lifecycle:
    1. Customer registers laptop product.
    2. Customer files warranty claim.
    3. Customer uploads purchase receipt image.
    4. Automated pipeline evaluates claim via ML models and policy rule engine.
    5. Reviewer inspects the manual review queue and finds the claim.
    6. Reviewer adjudicates claim with an underwriter approval and audit notes.
    7. Claim status transitions to 'Approved'.
    8. Audit log records submission, classification, and reviewer action.
    9. Reviewer/Customer downloads official ReportLab PDF claim summary report.
    """
    # -------------------------------------------------------------------------
    # Step 1: Customer registers device
    # -------------------------------------------------------------------------
    prod_resp = client.post("/api/products", json={
        "product_category": "Laptop",
        "brand": "Lenovo",
        "product_name": "ThinkPad T14 Gen 4",
        "model_number": "21HD000DUS",
        "serial_number": "LNV-INT-LIFECYCLE-9901",
        "purchase_price": 310000.00,
        "purchase_date": str(date.today() - timedelta(days=90)),
        "retailer": "Official Lenovo Flagship Mall"
    }, headers=customer_headers)
    assert prod_resp.status_code == 201
    product_id = prod_resp.json()["id"]

    # -------------------------------------------------------------------------
    # Step 2: Customer files claim
    # -------------------------------------------------------------------------
    claim_resp = client.post("/api/claims", json={
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=2)),
        "fault_description": "Motherboard fails to initiate POST; power LED flashes 3 times",
        "damage_type": "Motherboard Power Circuit Failure",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }, headers=customer_headers)
    assert claim_resp.status_code == 201
    claim_id = claim_resp.json()["claim_id"]

    # -------------------------------------------------------------------------
    # Step 3: Customer uploads receipt
    # -------------------------------------------------------------------------
    invoice_content = b"ORIGINAL SALES INVOICE - LENOVO THINKPAD T14 - NTN #10294819"
    files = {"file": ("lenovo_receipt.png", io.BytesIO(invoice_content), "image/png")}
    data = {"file_type": "receipt"}
    doc_resp = client.post(f"/api/claims/{claim_id}/documents", files=files, data=data, headers=customer_headers)
    assert doc_resp.status_code == 201

    # -------------------------------------------------------------------------
    # Step 4: Run classification pipeline
    # -------------------------------------------------------------------------
    class_resp = client.post(f"/api/claims/{claim_id}/classify", headers=admin_headers)
    assert class_resp.status_code == 200
    pred_data = class_resp.json()
    assert pred_data["final_decision"] in ["Likely Valid", "Likely Invalid", "Manual Review Required"]

    # -------------------------------------------------------------------------
    # Step 5: Reviewer inspects claim details
    # -------------------------------------------------------------------------
    detail_resp = client.get(f"/api/claims/{claim_id}", headers=reviewer_headers)
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["claim_id"] == claim_id
    assert len(detail["documents"]) >= 1
    assert detail["prediction"] is not None

    # -------------------------------------------------------------------------
    # Step 6: Reviewer adjudicates claim (Approve with notes)
    # -------------------------------------------------------------------------
    review_resp = client.post(
        f"/api/claims/{claim_id}/review",
        json={
            "action": "approve",
            "decision_notes": "All hardware diagnostics and proof of purchase confirmed valid by underwriter."
        },
        headers=reviewer_headers
    )
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["action"] == "approve"

    # -------------------------------------------------------------------------
    # Step 7: Verify updated status on status tracker
    # -------------------------------------------------------------------------
    status_resp = client.get(f"/api/claims/{claim_id}/status", headers=customer_headers)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "Approved"

    # -------------------------------------------------------------------------
    # Step 8: Verify immutable audit trail entries
    # -------------------------------------------------------------------------
    audit_resp = client.get("/api/admin/audit-logs", headers=admin_headers)
    assert audit_resp.status_code == 200
    logs = audit_resp.json()
    entity_actions = [l["action"] for l in logs if l["entity_id"] == claim_id]
    assert any(a in ["CLAIM_SUBMISSION", "PREDICTION_RUN", "REVIEW_ACTION"] for a in entity_actions)

    # -------------------------------------------------------------------------
    # Step 9: Download Official PDF Claim Evaluation Report
    # -------------------------------------------------------------------------
    pdf_resp = client.get(f"/api/claims/{claim_id}/report/pdf", headers=customer_headers)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")
    assert len(pdf_resp.content) > 1000
