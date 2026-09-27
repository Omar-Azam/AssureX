"""
AssureX Security & Access Control Test Suite
=============================================
Validates authentication enforcement, role-based authorization (RBAC),
SQL injection resistance on search filters, and path traversal defense:
1. Access without Authorization header (401 Unauthorized).
2. Access with malformed or tampered JWT token (401 Unauthorized).
3. RBAC escalation prevention: Customer accessing Admin Dashboard (403 Forbidden).
4. RBAC escalation prevention: Customer attempting underwriter claim review (403 Forbidden).
5. SQL Injection payloads on claim search filters.
6. Path traversal attempts on file and report endpoints.
"""

import pytest
from typing import Dict, Any
from starlette.testclient import TestClient


@pytest.mark.security
def test_unauthenticated_access_blocked(client: TestClient):
    """Verify protected endpoints reject requests lacking valid credentials."""
    endpoints = [
        "/api/auth/me",
        "/api/products",
        "/api/claims",
        "/api/review-queue",
        "/api/admin/dashboard",
        "/api/admin/audit-logs"
    ]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 401, f"Expected 401 on {ep}, got {resp.status_code}"


@pytest.mark.security
def test_malformed_token_rejected(client: TestClient):
    """Verify tampered or invalid JWT tokens are rejected."""
    bad_headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.tampered.token"}
    resp = client.get("/api/auth/me", headers=bad_headers)
    assert resp.status_code == 401


@pytest.mark.security
def test_customer_cannot_access_admin_dashboard(client: TestClient, customer_headers: Dict[str, str]):
    """Verify customers cannot access executive analytics or audit trail."""
    resp = client.get("/api/admin/dashboard", headers=customer_headers)
    assert resp.status_code == 403

    audit_resp = client.get("/api/admin/audit-logs", headers=customer_headers)
    assert audit_resp.status_code == 403


@pytest.mark.security
def test_customer_cannot_adjudicate_claims(client: TestClient, customer_headers: Dict[str, str]):
    """Verify customer role cannot approve or reject claims in the review queue."""
    resp = client.post("/api/claims/CLM-2026-00001/review", json={
        "action": "approve",
        "decision_notes": "Attempting unauthorized approval"
    }, headers=customer_headers)
    assert resp.status_code == 403


@pytest.mark.security
def test_sql_injection_defense_on_search_filters(client: TestClient, admin_headers: Dict[str, str]):
    """Verify SQL injection payloads in search parameters are safely escaped by SQLAlchemy."""
    sqli_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE claims; --",
        "1 UNION SELECT null, null, null, null--",
        "' OR 1=1 --",
        "admin'--"
    ]
    for payload in sqli_payloads:
        # Search claims
        resp = client.get(f"/api/claims?q={payload}", headers=admin_headers)
        assert resp.status_code == 200, f"SQLi probe triggered unexpected error: {resp.status_code}"
        assert isinstance(resp.json(), list)

        # Audit logs filter
        audit_resp = client.get(f"/api/admin/audit-logs?action={payload}", headers=admin_headers)
        assert audit_resp.status_code == 200
        assert isinstance(audit_resp.json(), list)


@pytest.mark.security
def test_path_traversal_defense(client: TestClient, customer_headers: Dict[str, str]):
    """Verify path traversal in claim IDs or file paths is properly trapped."""
    traversal_ids = [
        "../../../../etc/passwd",
        "..\\..\\windows\\system32\\cmd.exe",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd"
    ]
    for tid in traversal_ids:
        resp = client.get(f"/api/claims/{tid}", headers=customer_headers)
        assert resp.status_code in [404, 400]
