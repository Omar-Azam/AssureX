"""
Comprehensive Role-Based Access Control (RBAC) and Role Permissibility Verification
===================================================================================
Simulates all 4 user roles, hits every authorized endpoint, and verifies that
unauthorized endpoints are correctly denied with HTTP 403 Forbidden.
"""

from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_rbac_and_endpoints():
    print("=" * 75)
    print(" AssureX Final Self-Check: 4 Roles RBAC and Endpoint Verification")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # 1. Customer Role
    # --------------------------------------------------------------------------
    print("\n[+] 1. Testing Customer Role (customer / Customer@12345)")
    c_login = client.post("/api/auth/login", json={"username": "customer", "password": "Customer@12345"})
    assert c_login.status_code == 200, f"Customer login failed: {c_login.text}"
    c_token = c_login.json()["access_token"]
    c_headers = {"Authorization": f"Bearer {c_token}"}

    # Authorized Customer Endpoints
    cust_endpoints = [
        ("/api/auth/me", "GET"),
        ("/api/products", "GET"),
        ("/api/claims", "GET"),
        ("/api/warranties", "GET"),
        ("/api/warranties/alerts", "GET")
    ]
    for url, method in cust_endpoints:
        res = client.request(method, url, headers=c_headers)
        assert res.status_code == 200, f"Customer failed to access {url}: {res.status_code}"
        print(f"  [PASS] Authorized: {method} {url} -> 200 OK")

    # Customer RBAC Denials
    cust_forbidden = [
        ("/api/admin/dashboard", "GET"),
        ("/api/admin/export/claims", "GET"),
        ("/api/admin/model-versions", "GET"),
        ("/api/admin/anomalies", "GET"),
        ("/api/admin/audit-logs", "GET"),
        ("/api/review-queue", "GET")
    ]
    for url, method in cust_forbidden:
        res = client.request(method, url, headers=c_headers)
        assert res.status_code == 403, f"Expected 403 for Customer on {url}, got {res.status_code}"
        print(f"  [PASS] Correctly Denied (RBAC): {method} {url} -> 403 Forbidden")

    # --------------------------------------------------------------------------
    # 2. Service Center Role
    # --------------------------------------------------------------------------
    print("\n[+] 2. Testing Service Center Role (service_center / Service@12345)")
    sc_login = client.post("/api/auth/login", json={"username": "service_center", "password": "Service@12345"})
    assert sc_login.status_code == 200, f"Service Center login failed: {sc_login.text}"
    sc_token = sc_login.json()["access_token"]
    sc_headers = {"Authorization": f"Bearer {sc_token}"}

    sc_endpoints = [
        ("/api/auth/me", "GET"),
        ("/api/products", "GET"),
        ("/api/claims", "GET"),
        ("/api/repairs/product/1", "GET")
    ]
    for url, method in sc_endpoints:
        res = client.request(method, url, headers=sc_headers)
        assert res.status_code == 200, f"Service Center failed on {url}: {res.status_code}"
        print(f"  [PASS] Authorized: {method} {url} -> 200 OK")

    # Post repair record
    repair_post = client.post("/api/repairs", json={
        "product_id": 1,
        "repair_date": "2026-03-01",
        "repair_center": "TechnoCity Authorized Service Hub",
        "is_authorized": True,
        "repair_cost": 4500.0,
        "fault_repaired": "Display cable realignment",
        "notes": "Diagnostic check passed."
    }, headers=sc_headers)
    assert repair_post.status_code == 201, f"Post repair failed: {repair_post.text}"
    print("  [PASS] Authorized: POST /api/repairs -> 201 Created")

    # Service Center RBAC Denials
    sc_forbidden = [
        ("/api/admin/dashboard", "GET"),
        ("/api/admin/export/claims", "GET"),
        ("/api/admin/model-versions", "GET"),
        ("/api/review-queue", "GET")
    ]
    for url, method in sc_forbidden:
        res = client.request(method, url, headers=sc_headers)
        assert res.status_code == 403, f"Expected 403 for Service Center on {url}, got {res.status_code}"
        print(f"  [PASS] Correctly Denied (RBAC): {method} {url} -> 403 Forbidden")

    # --------------------------------------------------------------------------
    # 3. Claim Reviewer Role
    # --------------------------------------------------------------------------
    print("\n[+] 3. Testing Claim Reviewer Role (reviewer / Reviewer@12345)")
    rev_login = client.post("/api/auth/login", json={"username": "reviewer", "password": "Reviewer@12345"})
    assert rev_login.status_code == 200, f"Reviewer login failed: {rev_login.text}"
    rev_token = rev_login.json()["access_token"]
    rev_headers = {"Authorization": f"Bearer {rev_token}"}

    rev_endpoints = [
        ("/api/auth/me", "GET"),
        ("/api/review-queue", "GET"),
        ("/api/claims", "GET"),
        ("/api/claims/CLM-2026-DEMO-001", "GET"),
        ("/api/claims/CLM-2026-DEMO-001/report/pdf", "GET")
    ]
    for url, method in rev_endpoints:
        res = client.request(method, url, headers=rev_headers)
        assert res.status_code == 200, f"Reviewer failed on {url}: {res.status_code}"
        print(f"  [PASS] Authorized: {method} {url} -> 200 OK")

    # Reviewer decision submission
    rev_action = client.post("/api/claims/CLM-2026-DEMO-003/review", json={
        "action": "override",
        "decision_notes": "Supervisor reviewed Lemon Law trigger: Approved one-time courtesy motor replacement.",
        "overridden_decision": "Approved"
    }, headers=rev_headers)
    assert rev_action.status_code == 200, f"Review action failed: {rev_action.text}"
    print("  [PASS] Authorized: POST /api/claims/{id}/review -> 200 OK")

    # Reviewer RBAC Denials
    rev_forbidden = [
        ("/api/admin/dashboard", "GET"),
        ("/api/admin/export/claims", "GET"),
        ("/api/admin/export/warranties", "GET"),
        ("/api/admin/export/analytics", "GET"),
        ("/api/admin/audit-logs", "GET")
    ]
    for url, method in rev_forbidden:
        res = client.request(method, url, headers=rev_headers)
        assert res.status_code == 403, f"Expected 403 for Reviewer on {url}, got {res.status_code}"
        print(f"  [PASS] Correctly Denied (RBAC): {method} {url} -> 403 Forbidden")

    # --------------------------------------------------------------------------
    # 4. Administrator Role
    # --------------------------------------------------------------------------
    print("\n[+] 4. Testing Administrator Role (admin / Admin@12345)")
    adm_login = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@12345"})
    assert adm_login.status_code == 200, f"Admin login failed: {adm_login.text}"
    adm_token = adm_login.json()["access_token"]
    adm_headers = {"Authorization": f"Bearer {adm_token}"}

    adm_endpoints = [
        ("/api/auth/me", "GET"),
        ("/api/admin/dashboard", "GET"),
        ("/api/admin/audit-logs", "GET"),
        ("/api/admin/model-versions", "GET"),
        ("/api/admin/anomalies", "GET"),
        ("/api/admin/export/claims?format=csv", "GET"),
        ("/api/admin/export/claims?format=excel", "GET"),
        ("/api/admin/export/products?format=csv", "GET"),
        ("/api/admin/export/warranties?format=csv", "GET"),
        ("/api/admin/export/analytics?format=csv", "GET"),
        ("/api/review-queue", "GET"),
        ("/api/claims", "GET"),
        ("/api/products", "GET"),
        ("/api/claims/CLM-2026-DEMO-001/report/pdf", "GET")
    ]
    for url, method in adm_endpoints:
        res = client.request(method, url, headers=adm_headers)
        assert res.status_code == 200, f"Admin failed on {url}: {res.status_code}"
        print(f"  [PASS] Authorized: {method} {url} -> 200 OK")

    print("\n" + "=" * 75)
    print(" ALL 4 ROLES PASSED ENDPOINT AND RBAC ENFORCEMENT VERIFICATION!")
    print("=" * 75)

if __name__ == "__main__":
    test_rbac_and_endpoints()
