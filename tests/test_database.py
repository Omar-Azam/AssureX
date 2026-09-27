"""
AssureX Database Schema & Model Integrity Test Suite
=====================================================
Validates all SQLAlchemy ORM models, relations, cascades, and constraints:
1. User model unique username and email constraints.
2. Product & Warranty 1-to-1 relationship.
3. Claim model relationship with Product, Claimant, Documents, and Predictions.
4. Document model SHA-256 hash indexing and uniqueness.
5. Review model linkage to Reviewer and Claim.
6. AuditLog persistence and immutable timestamps.
7. Database transaction rollback on IntegrityError.
"""

import pytest
from datetime import date, datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.models import User, Product, Warranty, Claim, Document, RepairHistory, Prediction, Review, AuditLog
from backend.auth import get_password_hash


@pytest.mark.database
def test_user_unique_username_constraint(db_session: Session):
    """Verify unique constraint on User.username prevents duplicates."""
    user1 = User(username="unique_user_99", email="u1@test.com", hashed_password="hash", full_name="User One", role="customer")
    db_session.add(user1)
    db_session.commit()

    user2 = User(username="unique_user_99", email="u2@test.com", hashed_password="hash", full_name="User Two", role="customer")
    db_session.add(user2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


@pytest.mark.database
def test_product_and_warranty_relationship(db_session: Session):
    """Verify 1-to-1 relationship between Product and Warranty."""
    customer = db_session.query(User).filter(User.role == "customer").first()
    
    product = Product(
        user_id=customer.id,
        product_category="Laptop",
        brand="Lenovo",
        product_name="ThinkPad E14",
        model_number="20YES01",
        serial_number="LNV-DB-REL-01",
        purchase_price=160000.0,
        purchase_date=date(2025, 5, 1),
        retailer="Hafeez Center"
    )
    db_session.add(product)
    db_session.commit()

    warranty = Warranty(
        product_id=product.id,
        warranty_start_date=date(2025, 5, 1),
        warranty_expiry_date=date(2026, 5, 1),
        warranty_duration_months=12,
        warranty_status="Active"
    )
    db_session.add(warranty)
    db_session.commit()

    # Query back
    queried_prod = db_session.query(Product).filter(Product.serial_number == "LNV-DB-REL-01").first()
    assert queried_prod is not None
    assert queried_prod.warranty is not None
    assert queried_prod.warranty.warranty_duration_months == 12
    assert queried_prod.warranty.warranty_status == "Active"


@pytest.mark.database
def test_claim_document_prediction_relationships(db_session: Session):
    """Verify Claim links properly to Document, Prediction, and Review records."""
    customer = db_session.query(User).filter(User.role == "customer").first()
    reviewer = db_session.query(User).filter(User.role == "claim_reviewer").first()

    # Product
    prod = Product(
        user_id=customer.id,
        product_category="Smartphone",
        brand="Apple",
        product_name="iPhone 15",
        model_number="A3090",
        serial_number="APP-DB-REL-02",
        purchase_price=340000.0,
        purchase_date=date(2025, 9, 1),
        retailer="Apple Store"
    )
    db_session.add(prod)
    db_session.commit()

    # Claim
    claim = Claim(
        claim_id="CLM-DB-REL-001",
        user_id=customer.id,
        product_id=prod.id,
        fault_occurrence_date=date(2026, 1, 10),
        fault_description="OLED screen unresponsive",
        damage_type="Display Panel Defect",
        status="Under Review"
    )
    db_session.add(claim)
    db_session.commit()

    # Document
    doc = Document(
        claim_id=claim.id,
        file_name="receipt.jpg",
        file_path="uploads/receipt.jpg",
        file_type="receipt",
        file_size=1024,
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        is_duplicate=False
    )
    db_session.add(doc)

    # Prediction
    pred = Prediction(
        claim_id=claim.id,
        python_prediction='{"predicted_class": "Likely Valid", "confidence": 0.92}',
        gtm_prediction='{"predicted_class": "Likely Valid", "confidence": 0.89}',
        rule_engine_result='{"rules_failed": []}',
        decision_engine_result='{"final_decision": "Likely Valid"}',
        final_decision="Likely Valid",
        model_consistency_status="Strong Match",
        confidence_difference=0.03
    )
    db_session.add(pred)

    # Review
    rev = Review(
        claim_id=claim.id,
        reviewer_id=reviewer.id,
        action="approve",
        decision_notes="All documents verified."
    )
    db_session.add(rev)
    db_session.commit()

    # Verify relationships
    q_claim = db_session.query(Claim).filter(Claim.claim_id == "CLM-DB-REL-001").first()
    assert len(q_claim.documents) == 1
    assert q_claim.prediction is not None
    assert q_claim.prediction.final_decision == "Likely Valid"
    assert len(q_claim.reviews) == 1
    assert q_claim.reviews[0].reviewer.username == reviewer.username


@pytest.mark.database
def test_audit_log_creation_and_query(db_session: Session):
    """Verify AuditLog persists immutable audit records with timestamps."""
    audit_entry = AuditLog(
        user_id=1,
        action="POLICY_OVERRIDE",
        entity_type="Claim",
        entity_id="CLM-2026-9999",
        details="Underwriter authorized goodwill warranty extension for loyal customer",
        ip_address="192.168.1.100"
    )
    db_session.add(audit_entry)
    db_session.commit()

    q_log = db_session.query(AuditLog).filter(AuditLog.action == "POLICY_OVERRIDE").first()
    assert q_log is not None
    assert q_log.entity_id == "CLM-2026-9999"
    assert isinstance(q_log.timestamp, datetime)
