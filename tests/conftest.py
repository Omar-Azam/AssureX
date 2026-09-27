"""
AssureX Pytest Configuration and Global Fixtures
=================================================
Provides centralized test fixtures, isolated test database sessions,
Starlette test clients with pre-authenticated role headers, policy loaders,
mock OCR payloads, and the 11 official demo claim fixtures from sample_claims/.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, Generator

# Add workspace root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from starlette.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import Base, get_db
from backend.models import User, Product, Warranty, Claim, Document, Prediction, Review, AuditLog
from backend.auth import get_password_hash, create_access_token
from backend.config import ROLE_ADMIN, ROLE_CLAIM_REVIEWER, ROLE_SERVICE_CENTER, ROLE_CUSTOMER


# ==============================================================================
# SAMPLE CLAIMS & POLICY PATH FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_claims_dir(project_root: Path) -> Path:
    return project_root / "sample_claims"


@pytest.fixture(scope="session")
def policies_dir(project_root: Path) -> Path:
    return project_root / "policies"


def _load_json_claim(sample_claims_dir: Path, filename: str) -> Dict[str, Any]:
    path = sample_claims_dir / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def valid_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "valid_claim.json")


@pytest.fixture
def invalid_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "invalid_claim.json")


@pytest.fixture
def manual_review_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "manual_review_claim.json")


@pytest.fixture
def expired_warranty_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "expired_warranty_claim.json")


@pytest.fixture
def missing_document_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "missing_document_claim.json")


@pytest.fixture
def duplicate_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "duplicate_claim.json")


@pytest.fixture
def contradictory_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "contradictory_claim.json")


@pytest.fixture
def serial_mismatch_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "serial_mismatch_claim.json")


@pytest.fixture
def unauthorized_repair_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "unauthorized_repair_claim.json")


@pytest.fixture
def boundary_date_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "boundary_date_claim.json")


@pytest.fixture
def model_disagreement_claim(sample_claims_dir: Path) -> Dict[str, Any]:
    return _load_json_claim(sample_claims_dir, "model_disagreement_claim.json")


# ==============================================================================
# POLICY FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def laptop_policy(policies_dir: Path) -> Dict[str, Any]:
    with open(policies_dir / "laptop_policy.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def smartphone_policy(policies_dir: Path) -> Dict[str, Any]:
    with open(policies_dir / "smartphone_policy.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def washing_machine_policy(policies_dir: Path) -> Dict[str, Any]:
    with open(policies_dir / "washing_machine_policy.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# IN-MEMORY TEST DATABASE & CLIENT FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def test_db_engine():
    """Isolated SQLite database in memory for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="session")
def TestingSessionLocal(test_db_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)


@pytest.fixture
def db_session(test_db_engine, TestingSessionLocal) -> Generator[Session, None, None]:
    """Provides a fresh transaction rollback session per test."""
    connection = test_db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    # Seed baseline roles if not present
    if session.query(User).count() == 0:
        users = [
            User(username="test_admin", email="admin@test.com", hashed_password=get_password_hash("Pass@123"), full_name="Test Admin", role=ROLE_ADMIN),
            User(username="test_reviewer", email="rev@test.com", hashed_password=get_password_hash("Pass@123"), full_name="Test Reviewer", role=ROLE_CLAIM_REVIEWER),
            User(username="test_service", email="svc@test.com", hashed_password=get_password_hash("Pass@123"), full_name="Test Service", role=ROLE_SERVICE_CENTER),
            User(username="test_customer", email="cust@test.com", hashed_password=get_password_hash("Pass@123"), full_name="Test Customer", role=ROLE_CUSTOMER),
        ]
        session.add_all(users)
        session.commit()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Starlette TestClient with database dependency override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==============================================================================
# AUTH TOKEN FIXTURES
# ==============================================================================

@pytest.fixture
def admin_headers(db_session: Session) -> Dict[str, str]:
    token = create_access_token(data={"sub": "test_admin", "role": ROLE_ADMIN})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def reviewer_headers(db_session: Session) -> Dict[str, str]:
    token = create_access_token(data={"sub": "test_reviewer", "role": ROLE_CLAIM_REVIEWER})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def customer_headers(db_session: Session) -> Dict[str, str]:
    token = create_access_token(data={"sub": "test_customer", "role": ROLE_CUSTOMER})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def service_headers(db_session: Session) -> Dict[str, str]:
    token = create_access_token(data={"sub": "test_service", "role": ROLE_SERVICE_CENTER})
    return {"Authorization": f"Bearer {token}"}


# ==============================================================================
# MOCK OCR & PREDICTION FIXTURES
# ==============================================================================

@pytest.fixture
def mock_complete_ocr_text() -> str:
    return """
    ============================================================
              HAFEEZ CENTER ELECTRONICS MEGASTORE
                 Main Boulevard, Gulberg III, Lahore
                 NTN: 4192084-2   STRN: 03-09-8419-002
    ============================================================
    CASH SALE RECEIPT / TAX INVOICE
    Invoice No: INV-2025-99412
    Date: 2025-11-14

    CUSTOMER DETAILS:
    Name: Omar Azam
    CNIC: 35202-9418291-3

    ITEM DETAILS:
    Description: Lenovo Legion Pro 5 16IRX8 Gaming Laptop
    Model: 82WK0046US
    Serial No: LEN-LA-2025-BT7UFBW
    Warranty: 24 Months Official Manufacturer Warranty

    Amount (PKR): Rs. 385,000.00
    ============================================================
    """


@pytest.fixture
def mock_partial_ocr_text() -> str:
    return """
    Airlink Communication Store
    Sold To: Customer
    Item: Smartphone
    Date: 2025-08-10
    Total: Rs. 95000
    """


@pytest.fixture
def mock_predictions_strong_match() -> Dict[str, Any]:
    return {
        "python_result": {"predicted_class": "Likely Valid", "confidence": 0.94},
        "gtm_result": {"predicted_class": "Likely Valid", "confidence": 0.91},
        "rule_result": {"rules_failed": [], "manual_review_required": False, "contradictions": []}
    }


@pytest.fixture
def mock_predictions_disagreement() -> Dict[str, Any]:
    return {
        "python_result": {"predicted_class": "Likely Valid", "confidence": 0.95},
        "gtm_result": {"predicted_class": "Likely Invalid", "confidence": 0.89},
        "rule_result": {"rules_failed": [], "manual_review_required": False, "contradictions": []}
    }
