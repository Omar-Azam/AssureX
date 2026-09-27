"""
AssureX Pydantic Validation & Serialization Schemas
==================================================
Defines request and response schemas for all REST API endpoints.
"""

import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator


# ==============================================================================
# AUTH SCHEMAS
# ==============================================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2, max_length=128)
    role: str = Field("customer", description="customer, service_center, claim_reviewer, admin")


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=128)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ==============================================================================
# PRODUCT SCHEMAS
# ==============================================================================

class ProductCreate(BaseModel):
    product_category: str = Field(..., description="Smartphone, Laptop, Washing Machine")
    product_name: str
    brand: str
    model_number: str
    serial_number: str
    purchase_price: float
    purchase_date: date
    retailer: str


class ProductResponse(BaseModel):
    id: int
    product_category: str
    product_name: str
    brand: str
    model_number: str
    serial_number: str
    purchase_price: float
    purchase_date: date
    retailer: str
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# WARRANTY SCHEMAS
# ==============================================================================

class WarrantyCreate(BaseModel):
    product_id: int
    warranty_duration_months: int = 12
    warranty_start_date: Optional[date] = None
    terms_conditions: Optional[str] = None


class WarrantyResponse(BaseModel):
    id: int
    product_id: int
    warranty_duration_months: int
    warranty_start_date: date
    warranty_expiry_date: date
    warranty_status: str
    terms_conditions: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# DOCUMENT SCHEMAS
# ==============================================================================

class DocumentResponse(BaseModel):
    id: int
    claim_id: int
    file_name: str
    file_type: str
    file_size: int
    sha256_hash: str
    is_duplicate: bool
    duplicate_of_document_id: Optional[int]
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# REPAIR HISTORY SCHEMAS
# ==============================================================================

class RepairHistoryCreate(BaseModel):
    product_id: int
    claim_id: Optional[int] = None
    repair_date: date
    repair_center: str
    is_authorized: bool = True
    repair_cost: float = 0.0
    fault_repaired: str
    notes: Optional[str] = None


class RepairHistoryResponse(BaseModel):
    id: int
    product_id: int
    claim_id: Optional[int]
    repair_date: date
    repair_center: str
    is_authorized: bool
    repair_cost: float
    fault_repaired: str
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# PREDICTION & DECISION SCHEMAS
# ==============================================================================

class PredictionResponse(BaseModel):
    id: int
    claim_id: int
    final_decision: str
    model_consistency_status: str
    confidence_difference: float
    python_prediction: Optional[Dict[str, Any]] = None
    gtm_prediction: Optional[Dict[str, Any]] = None
    rule_engine_result: Optional[Dict[str, Any]] = None
    decision_engine_result: Optional[Dict[str, Any]] = None
    created_at: datetime

    @field_validator("python_prediction", "gtm_prediction", "rule_engine_result", "decision_engine_result", mode="before")
    @classmethod
    def parse_json_dict_fields(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {"raw": v}
        return v or {}

    class Config:
        from_attributes = True


# ==============================================================================
# REVIEW SCHEMAS
# ==============================================================================

class ReviewCreate(BaseModel):
    action: str = Field(..., description="approve, reject, override, request_evidence")
    decision_notes: str
    overridden_decision: Optional[str] = None


class ReviewResponse(BaseModel):
    id: int
    claim_id: int
    reviewer_id: int
    action: str
    decision_notes: str
    overridden_decision: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# CLAIM SCHEMAS
# ==============================================================================

class ClaimCreate(BaseModel):
    product_id: int
    fault_occurrence_date: date
    fault_description: str
    damage_type: str
    prior_replacement: bool = False
    repair_history: str = "0 repairs"


class ClaimResponse(BaseModel):
    id: int
    claim_id: str
    user_id: int
    product_id: int
    claim_submission_date: date
    fault_occurrence_date: date
    fault_description: str
    damage_type: str
    status: str
    prior_replacement: bool
    repair_history: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ClaimDetailResponse(ClaimResponse):
    product: Optional[ProductResponse] = None
    documents: List[DocumentResponse] = []
    prediction: Optional[PredictionResponse] = None
    reviews: List[ReviewResponse] = []


class ClaimStatusResponse(BaseModel):
    claim_id: str
    status: str
    submission_date: date
    updated_at: datetime
    final_decision: Optional[str] = None
    model_consistency_status: Optional[str] = None
    documents_count: int
    has_duplicates: bool
    manual_review_required: bool


# ==============================================================================
# AUDIT LOG SCHEMAS
# ==============================================================================

class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    entity_type: str
    entity_id: str
    details: str
    timestamp: datetime
    ip_address: Optional[str]

    class Config:
        from_attributes = True


class DashboardMetricsResponse(BaseModel):
    total_claims: int
    status_breakdown: Dict[str, int]
    decision_breakdown: Dict[str, int]
    consistency_breakdown: Dict[str, int]
    total_products: int
    total_users: int
    duplicate_documents_detected: int
    valid_count: int = 0
    invalid_count: int = 0
    manual_review_count: int = 0
    model_disagreement_rate: float = 0.0
    average_confidence: float = 0.0
    claim_trends: Dict[str, int] = {}
    recent_activity: List[AuditLogResponse]
