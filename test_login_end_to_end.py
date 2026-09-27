"""
End-to-End Authentication and Session Validation Test
=====================================================
Verifies login via username, email, /api/auth/login, and /api/auth/token,
and confirms that subsequent authenticated requests succeed without session expiration.
"""

import sys
from starlette.testclient import TestClient
from backend.main import app
from backend.auth import decode_access_token

client = TestClient(app)

USERS_TO_TEST = [
    ("admin", "admin@assurex.com", "Admin@12345", "admin", "/api/admin/dashboard"),
    ("reviewer", "reviewer@assurex.com", "Reviewer@12345", "claim_reviewer", "/api/review-queue"),
    ("service_center", "service@assurex.com", "Service@12345", "service_center", "/api/repairs/product/1"),
    ("customer", "customer@assurex.com", "Customer@12345", "customer", "/api/products")
]

def run_tests():
    print("=" * 70)
    print("Running AssureX End-to-End Login & Session Verification")
    print("=" * 70)

    for username, email, password, role, protected_endpoint in USERS_TO_TEST:
        print(f"\n--- Testing User: {username} ({role}) ---")

        # 1. Login with Username via /api/auth/login
        res_user = client.post("/api/auth/login", json={"username": username, "password": password})
        assert res_user.status_code == 200, f"Login with username failed for {username}: {res_user.text}"
        data_user = res_user.json()
        token_user = data_user["access_token"]
        assert token_user, "Access token missing"
        assert data_user["user"]["role"] == role
        print(f"  [PASS] Login with username '{username}' -> 200 OK")

        # 2. Verify token can be decoded without Expiration error
        payload = decode_access_token(token_user)
        assert payload["role"] == role
        print(f"  [PASS] JWT token decode & expiry valid (exp={payload['exp']})")

        # 3. Subsequent Authenticated Request via Bearer header
        sub_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token_user}"})
        assert sub_res.status_code == 200, f"/api/auth/me failed for {username}: {sub_res.text}"
        assert sub_res.json()["username"] == username
        print(f"  [PASS] Subsequent request /api/auth/me -> 200 OK (User ID={sub_res.json()['id']})")

        # 4. Protected role-appropriate endpoint
        role_res = client.get(protected_endpoint, headers={"Authorization": f"Bearer {token_user}"})
        assert role_res.status_code == 200, f"{protected_endpoint} failed for {username}: {role_res.text}"
        print(f"  [PASS] Protected endpoint {protected_endpoint} -> 200 OK")

        # 5. Login with Email via /api/auth/login
        res_email = client.post("/api/auth/login", json={"username": email, "password": password})
        assert res_email.status_code == 200, f"Login with email failed for {email}: {res_email.text}"
        token_email = res_email.json()["access_token"]
        assert token_email, "Email login access token missing"
        print(f"  [PASS] Login with email '{email}' -> 200 OK")

        # 6. OAuth2 /api/auth/token endpoint with form-data
        res_form = client.post("/api/auth/token", data={"username": username, "password": password})
        assert res_form.status_code == 200, f"Form token login failed for {username}: {res_form.text}"
        assert res_form.json()["access_token"]
        print(f"  [PASS] OAuth2 form token endpoint /api/auth/token -> 200 OK")

    # 7. Test invalid credentials
    print("\n--- Testing Invalid Credentials ---")
    bad_res = client.post("/api/auth/login", json={"username": "admin", "password": "WrongPassword123!"})
    assert bad_res.status_code == 401, f"Expected 401, got {bad_res.status_code}"
    assert bad_res.json()["detail"] == "Incorrect username or password."
    print("  [PASS] Bad password rejected with 401 and descriptive message")

    bad_user_res = client.post("/api/auth/login", json={"username": "nonexistent_user_999", "password": "AnyPassword"})
    assert bad_user_res.status_code == 401, f"Expected 401, got {bad_user_res.status_code}"
    print("  [PASS] Nonexistent user rejected with 401")

    print("\n" + "=" * 70)
    print("ALL AUTHENTICATION & SESSION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
