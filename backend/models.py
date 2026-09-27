"""
AssureX SQLAlchemy Database Models
==================================
Defines the 9 required domain entities:
1. User
2. Product
3. Warranty
4. Claim
5. Document
6. RepairHistory
7. Prediction
8. Review
9. AuditLog
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, DateTime,
    ForeignKey, Text, Index
)
from sqlalchemy.orm import relationship
from backend.database import Base


class User(Base):
    """System user with role-based access control."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    full_name = Column(String(128), nullable=False)
    role = Column(String(32), nullable=False, default="customer")  # customer, service_center, claim_reviewer, admin
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    products = relationship("Product", back_populates="owner", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="claimant", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="reviewer")
    audit_logs = relationship("AuditLog", back_populates="user")


class Product(Base):
    """Consumer electronics product registered by a customer or service center."""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_category = Column(String(64), nullable=False)  # Smartphone, Laptop, Washing Machine
    product_name = Column(String(128), nullable=False)
    brand = Column(String(64), nullable=False)
    model_number = Column(String(64), nullable=False)
    serial_number = Column(String(64), unique=True, nullable=False, index=True)
    purchase_price = Column(Float, nullable=False)
    purchase_date = Column(Date, nullable=False)
    retailer = Column(String(128), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    owner = relationship("User", back_populates="products")
    warranty = relationship("Warranty", back_populates="product", uselist=False, cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="product", cascade="all, delete-orphan")
    repairs = relationship("RepairHistory", back_populates="product", cascade="all, delete-orphan")


class Warranty(Base):
    """Product warranty policy coverage and temporal parameters."""
    __tablename__ = "warranties"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), unique=True, nullable=False)
    warranty_duration_months = Column(Integer, nullable=False, default=12)
    warranty_start_date = Column(Date, nullable=False)
    warranty_expiry_date = Column(Date, nullable=False)
    warranty_status = Column(String(32), nullable=False, default="Active")  # Active, Expired, Void
    terms_conditions = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    product = relationship("Product", back_populates="warranty")


class Claim(Base):
    """Warranty Claim record submitted by a customer or service center."""
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(32), unique=True, nullable=False, index=True)  # e.g. CLM-2026-00001
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    claim_submission_date = Column(Date, nullable=False, default=date.today)
    fault_occurrence_date = Column(Date, nullable=False)
    fault_description = Column(Text, nullable=False)
    damage_type = Column(String(64), nullable=False)
    status = Column(
        String(32),
        nullable=False,
        default="Submitted"
    )  # Submitted, Under Evaluation, Manual Review, Approved, Rejected, Overridden
    prior_replacement = Column(Boolean, default=False, nullable=False)
    repair_history = Column(String(128), default="0 repairs", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    claimant = relationship("User", back_populates="claims")
    product = relationship("Product", back_populates="claims")
    documents = relationship("Document", back_populates="claim", cascade="all, delete-orphan")
    repairs = relationship("RepairHistory", back_populates="claim", cascade="all, delete-orphan")
    prediction = relationship("Prediction", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="claim", cascade="all, delete-orphan")


class Document(Base):
    """Uploaded claim supporting evidence file with SHA-256 hash duplicate tracking."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    file_name = Column(String(256), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(64), nullable=False)  # receipt, warranty_card, product_image, serial_evidence, other
    file_size = Column(Integer, nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    is_duplicate = Column(Boolean, default=False, nullable=False)
    duplicate_of_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="documents")


class RepairHistory(Base):
    """Historical servicing and maintenance record for a product or claim."""
    __tablename__ = "repair_histories"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    repair_date = Column(Date, nullable=False)
    repair_center = Column(String(128), nullable=False)
    is_authorized = Column(Boolean, default=True, nullable=False)
    repair_cost = Column(Float, default=0.0, nullable=False)
    fault_repaired = Column(String(128), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    product = relationship("Product", back_populates="repairs")
    claim = relationship("Claim", back_populates="repairs")


class Prediction(Base):
    """Automated multimodal ML and rule engine prediction result for a claim."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), unique=True, nullable=False)
    python_prediction = Column(Text, nullable=False)       # JSON string from predict_with_confidence
    gtm_prediction = Column(Text, nullable=False)          # JSON string from gtm_classifier
    rule_engine_result = Column(Text, nullable=False)      # JSON string from rule_engine
    decision_engine_result = Column(Text, nullable=False)  # JSON string from decision_engine
    final_decision = Column(String(64), nullable=False)    # Likely Valid, Likely Invalid, Manual Review Required
    model_consistency_status = Column(String(64), nullable=False)  # Strong Match, Acceptable Match, etc.
    confidence_difference = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="prediction")


class Review(Base):
    """Human claims reviewer decision, manual review action, or override audit."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(32), nullable=False)  # approve, reject, override, request_evidence
    decision_notes = Column(Text, nullable=False)
    overridden_decision = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="reviews")
    reviewer = relationship("User", back_populates="reviews")


class AuditLog(Base):
    """Immutable audit trail logging all lifecycle actions, predictions, status changes, and overrides."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(64), nullable=False, index=True)  # CLAIM_SUBMISSION, PREDICTION_RUN, STATUS_CHANGE, etc.
    entity_type = Column(String(64), nullable=False, index=True)  # Claim, User, Review, Document, etc.
    entity_id = Column(String(64), nullable=False, index=True)
    details = Column(Text, nullable=False)  # JSON or text description
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(64), nullable=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")
