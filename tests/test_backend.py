"""
AssureX Backend End-to-End Test Suite
=====================================
Validates all backend REST endpoints and workflows:
1. User registration & JWT authentication
2. Role-based access control (RBAC)
3. Product & warranty registration (with calendar-month temporal metrics)
4. Claim creation & status tracking
5. Document upload with SHA-256 duplicate detection
6. Automated claim classification (Rule Engine + ML + GTM + Decision Engine)
7. Manual review queue and reviewer actions (approve, reject, override)
8. Multi-parameter claim searching and filtering
9. CSV claims export
10. Official PDF evaluation report generation
11. Audit trail logging and admin dashboard KPIs
"""

import sys
import io
from pathlib import Path
from datetime import date, timedelta

# Add workspace to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)


def get_auth_token(username: str, password: str) -> str:
    """Helper to login and obtain JWT access token."""
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.text}"
    return resp.json()["access_token"]


def test_health_check():
    """Verify system health check endpoint."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["documentation"] == "/docs"


def test_user_registration_and_login():
    """Verify user registration, hashing, and JWT token issuance."""
    import uuid
    test_username = f"user_{uuid.uuid4().hex[:8]}"
    reg_payload = {
        "username": test_username,
        "email": f"{test_username}@example.com",
        "password": "Password@123",
        "full_name": "Test Claimant",
        "role": "customer"
    }
    # Register
    reg_resp = client.post("/api/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201

    # Login
    login_resp = client.post("/api/auth/login", json={"username": test_username, "password": "Password@123"})
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # Verify /api/auth/me
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == test_username


def test_role_based_access_control():
    """Verify role-based access restrictions (customer vs admin)."""
    admin_token = get_auth_token("admin", "Admin@12345")
    cust_token = get_auth_token("customer", "Customer@12345")

    # Admin access to dashboard should succeed
    admin_dash = client.get("/api/admin/dashboard", headers={"Authorization": f"Bearer {admin_token}"})
    assert admin_dash.status_code == 200

    # Customer access to admin dashboard should be forbidden (403)
    cust_dash = client.get("/api/admin/dashboard", headers={"Authorization": f"Bearer {cust_token}"})
    assert cust_dash.status_code == 403


def test_product_and_warranty_registration():
    """Verify registering a product and associating an active warranty."""
    import uuid
    cust_token = get_auth_token("customer", "Customer@12345")
    unique_serial = f"TEST-SN-{uuid.uuid4().hex[:10].upper()}"

    prod_payload = {
        "product_category": "Smartphone",
        "product_name": "OnePlus 12R",
        "brand": "OnePlus",
        "model_number": "CPH2609",
        "serial_number": unique_serial,
        "purchase_price": 145000.0,
        "purchase_date": str(date.today() - timedelta(days=120)),
        "retailer": "Official Tech Store"
    }

    # Register product
    p_resp = client.post("/api/products", json=prod_payload, headers={"Authorization": f"Bearer {cust_token}"})
    assert p_resp.status_code == 201
    prod = p_resp.json()
    product_id = prod["id"]

    # Register warranty
    w_resp = client.post("/api/warranties", json={"product_id": product_id, "warranty_duration_months": 12}, headers={"Authorization": f"Bearer {cust_token}"})
    assert w_resp.status_code == 201
    w_data = w_resp.json()
    assert w_data["product_id"] == product_id
    assert w_data["warranty_status"] == "Active"


def test_claim_submission_and_status():
    """Verify warranty claim creation and status tracking."""
    cust_token = get_auth_token("customer", "Customer@12345")

    # Fetch existing products
    prods = client.get("/api/products", headers={"Authorization": f"Bearer {cust_token}"}).json()
    product_id = prods[0]["id"]

    claim_payload = {
        "product_id": product_id,
        "fault_occurrence_date": str(date.today() - timedelta(days=5)),
        "fault_description": "Touchscreen unresponsive along top notification bar.",
        "damage_type": "Display Panel Defect",
        "prior_replacement": False,
        "repair_history": "0 repairs"
    }

    create_resp = client.post("/api/claims", json=claim_payload, headers={"Authorization": f"Bearer {cust_token}"})
    assert create_resp.status_code == 201
    claim = create_resp.json()
    claim_id = claim["claim_id"]

    # Track status
    stat_resp = client.get(f"/api/claims/{claim_id}/status", headers={"Authorization": f"Bearer {cust_token}"})
    assert stat_resp.status_code == 200
    assert stat_resp.json()["status"] == "Submitted"


def test_document_upload_and_sha256_duplicate_detection():
    """Verify document upload with SHA-256 duplicate detection flag."""
    cust_token = get_auth_token("customer", "Customer@12345")
    claims = client.get("/api/claims", headers={"Authorization": f"Bearer {cust_token}"}).json()
    claim_id = claims[0]["claim_id"]

    # Create dummy receipt file
    file_bytes = b"ORIGINAL SALES INVOICE - ASSUREX NTN # 4829103-8"

    # First upload (clean document)
    resp1 = client.post(
        f"/api/claims/{claim_id}/documents",
        files={"file": ("invoice_sample.pdf", io.BytesIO(file_bytes), "application/pdf")},
        data={"file_type": "receipt"},
        headers={"Authorization": f"Bearer {cust_token}"}
    )
    assert resp1.status_code == 201
    doc1 = resp1.json()
    assert "sha256_hash" in doc1

    # Second upload with identical file content (Duplicate Detection Test)
    resp2 = client.post(
        f"/api/claims/{claim_id}/documents",
        files={"file": ("duplicate_invoice.pdf", io.BytesIO(file_bytes), "application/pdf")},
        data={"file_type": "receipt"},
        headers={"Authorization": f"Bearer {cust_token}"}
    )
    assert resp2.status_code == 201
    doc2 = resp2.json()
    assert doc2["is_duplicate"] is True
    assert doc2["sha256_hash"] == doc1["sha256_hash"]


def test_claim_classification_pipeline():
    """Verify execution of the full multi-model classification pipeline."""
    admin_token = get_auth_token("admin", "Admin@12345")
    claims = client.get("/api/claims", headers={"Authorization": f"Bearer {admin_token}"}).json()
    claim_id = claims[0]["claim_id"]

    classify_resp = client.post(
        f"/api/claims/{claim_id}/classify",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert classify_resp.status_code == 200
    pred = classify_resp.json()
    assert pred["final_decision"] in ["Likely Valid", "Likely Invalid", "Manual Review Required"]
    assert pred["model_consistency_status"] in [
        "Strong Match", "Acceptable Match", "Weak Match", "Model Disagreement", "Uncertain Result"
    ]
    assert "python_prediction" in pred
    assert "rule_engine_result" in pred
    assert "decision_engine_result" in pred


def test_manual_review_queue_and_action():
    """Verify inspecting review queue and submitting an underwriter approval/rejection."""
    rev_token = get_auth_token("reviewer", "Reviewer@12345")

    # Inspect queue
    queue_resp = client.get("/api/review-queue", headers={"Authorization": f"Bearer {rev_token}"})
    assert queue_resp.status_code == 200
    queue = queue_resp.json()
    assert isinstance(queue, list)
    assert len(queue) > 0

    target_claim_id = queue[0]["claim_id"]

    # Submit review action
    review_resp = client.post(
        f"/api/claims/{target_claim_id}/review",
        json={"action": "approve", "decision_notes": "All required proof of purchase verified by assessor."},
        headers={"Authorization": f"Bearer {rev_token}"}
    )
    assert review_resp.status_code == 200
    review_data = review_resp.json()
    assert review_data["action"] == "approve"


def test_search_and_filter_claims():
    """Verify multi-criteria search and filter queries."""
    admin_token = get_auth_token("admin", "Admin@12345")

    # Filter by category
    resp = client.get("/api/claims?category=Smartphone", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    claims = resp.json()
    assert isinstance(claims, list)


def test_csv_export():
    """Verify streaming CSV export of claims dataset."""
    admin_token = get_auth_token("admin", "Admin@12345")
    resp = client.get("/api/claims/export/csv", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    content = resp.text
    assert "Claim ID,Customer Username,Product Category" in content


def test_pdf_report_download():
    """Verify streaming download of the official evaluation PDF report."""
    admin_token = get_auth_token("admin", "Admin@12345")
    claims = client.get("/api/claims", headers={"Authorization": f"Bearer {admin_token}"}).json()
    claim_id = claims[0]["claim_id"]

    resp = client.get(f"/api/claims/{claim_id}/report/pdf", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")


def test_admin_dashboard_and_audit_trail():
    """Verify administrative dashboard metrics and immutable audit log queries."""
    admin_token = get_auth_token("admin", "Admin@12345")

    # Dashboard metrics
    dash_resp = client.get("/api/admin/dashboard", headers={"Authorization": f"Bearer {admin_token}"})
    assert dash_resp.status_code == 200
    metrics = dash_resp.json()
    assert "total_claims" in metrics
    assert "status_breakdown" in metrics
    assert "duplicate_documents_detected" in metrics
    assert "recent_activity" in metrics

    # Audit logs
    audit_resp = client.get("/api/admin/audit-logs", headers={"Authorization": f"Bearer {admin_token}"})
    assert audit_resp.status_code == 200
    logs = audit_resp.json()
    assert isinstance(logs, list)
    assert len(logs) > 0
    assert any(log["action"] in ["CLAIM_SUBMISSION", "LOGIN", "PREDICTION_RUN", "DOCUMENT_UPLOAD"] for log in logs)


if __name__ == "__main__":
    print("Running AssureX Backend End-to-End Test Suite...")
    test_health_check()
    test_user_registration_and_login()
    test_role_based_access_control()
    test_product_and_warranty_registration()
    test_claim_submission_and_status()
    test_document_upload_and_sha256_duplicate_detection()
    test_claim_classification_pipeline()
    test_manual_review_queue_and_action()
    test_search_and_filter_claims()
    test_csv_export()
    test_pdf_report_download()
    test_admin_dashboard_and_audit_trail()
    print("\n[PASS] All 12 Backend End-to-End Tests Passed Successfully!")
