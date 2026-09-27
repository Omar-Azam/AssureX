"""
AssureX Repair History Management Endpoints
===========================================
Repair event logging, service center tracking, and lemon law history inspection.
"""

from typing import List, Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import RepairHistory, Product, Claim, User, AuditLog
from backend.schemas import RepairHistoryCreate, RepairHistoryResponse
from backend.auth import get_current_user, require_roles
from backend.config import ROLE_ADMIN, ROLE_CLAIM_REVIEWER, ROLE_SERVICE_CENTER, ROLE_CUSTOMER

router = APIRouter(prefix="/api/repairs", tags=["Repair History"])


@router.post("", response_model=RepairHistoryResponse, status_code=status.HTTP_201_CREATED)
def log_repair_record(
    repair_in: RepairHistoryCreate,
    request: Request,
    current_user: User = Depends(require_roles([ROLE_ADMIN, ROLE_SERVICE_CENTER, ROLE_CLAIM_REVIEWER])),
    db: Session = Depends(get_db)
):
    """
    Log an authorized or unauthorized repair history record for a product.
    Restricted to Service Center technicians, Reviewers, and Administrators.
    """
    product = db.query(Product).filter(Product.id == repair_in.product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target product not found.")

    if repair_in.claim_id:
        claim = db.query(Claim).filter(Claim.id == repair_in.claim_id).first()
        if not claim:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated claim not found.")

    repair = RepairHistory(
        product_id=repair_in.product_id,
        claim_id=repair_in.claim_id,
        repair_date=repair_in.repair_date,
        repair_center=repair_in.repair_center,
        is_authorized=repair_in.is_authorized,
        repair_cost=repair_in.repair_cost,
        fault_repaired=repair_in.fault_repaired,
        notes=repair_in.notes
    )
    db.add(repair)
    db.commit()
    db.refresh(repair)

    # Log to audit trail
    audit = AuditLog(
        user_id=current_user.id,
        action="REPAIR_LOGGED",
        entity_type="RepairHistory",
        entity_id=str(repair.id),
        details=f"Repair logged for product {product.id} at {repair.repair_center} (Authorized: {repair.is_authorized})",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return repair


@router.get("/product/{product_id}", response_model=List[RepairHistoryResponse])
def get_product_repair_history(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve complete chronological service history for a product."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")

    if current_user.role == ROLE_CUSTOMER and product.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return db.query(RepairHistory).filter(RepairHistory.product_id == product_id).order_by(RepairHistory.repair_date.desc()).all()


@router.get("/claim/{claim_id}", response_model=List[RepairHistoryResponse])
def get_claim_repair_history(
    claim_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve repair records associated with a specific claim."""
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    if current_user.role == ROLE_CUSTOMER and claim.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return db.query(RepairHistory).filter(RepairHistory.claim_id == claim.id).order_by(RepairHistory.repair_date.desc()).all()
