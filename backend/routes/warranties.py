"""
AssureX Warranty Registration Endpoints
=======================================
Warranty registration, activation, and exact expiry date computation.
"""

from datetime import date
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Product, Warranty, AuditLog, User
from backend.schemas import WarrantyCreate, WarrantyResponse
from backend.auth import get_current_user
from claim_metrics import compute_claim_metrics, parse_date

router = APIRouter(prefix="/api/warranties", tags=["Warranties"])


@router.post("", response_model=WarrantyResponse, status_code=status.HTTP_201_CREATED)
def register_warranty(
    warranty_in: WarrantyCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Register or activate a warranty policy for a product.
    Calculates exact calendar-month expiry via claim_metrics.compute_claim_metrics().
    """
    product = db.query(Product).filter(Product.id == warranty_in.product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target product not found.")

    if current_user.role == "customer" and product.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    existing_warranty = db.query(Warranty).filter(Warranty.product_id == warranty_in.product_id).first()
    if existing_warranty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A warranty is already registered for this product."
        )

    start_dt = warranty_in.warranty_start_date or product.purchase_date
    metrics = compute_claim_metrics(
        purchase_date=product.purchase_date,
        claim_filing_date=date.today(),
        warranty_duration_months=warranty_in.warranty_duration_months
    )
    expiry_dt = parse_date(metrics["warranty_expiry_date"])

    terms = getattr(warranty_in, "terms_conditions", None) or getattr(warranty_in, "coverage_details", None)
    w_type = getattr(warranty_in, "warranty_type", "Standard Manufacturer")

    warranty = Warranty(
        product_id=product.id,
        warranty_duration_months=warranty_in.warranty_duration_months,
        warranty_start_date=start_dt,
        warranty_expiry_date=expiry_dt,
        warranty_status=metrics["warranty_status"],
        terms_conditions=terms
    )
    db.add(warranty)
    db.commit()
    db.refresh(warranty)

    # Log to audit trail
    audit = AuditLog(
        user_id=current_user.id,
        action="WARRANTY_REGISTRATION",
        entity_type="Warranty",
        entity_id=str(warranty.id),
        details=f"Warranty registered for product {product.id} (Expires: {expiry_dt}, Status: {warranty.warranty_status})",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return warranty


@router.get("", response_model=List[WarrantyResponse])
def list_warranties(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List warranty records. Customers see their registered warranties; staff see all."""
    query = db.query(Warranty).join(Product, Warranty.product_id == Product.id)
    if current_user.role == "customer":
        query = query.filter(Product.user_id == current_user.id)
    return query.order_by(Warranty.warranty_expiry_date.asc()).all()


@router.get("/alerts")
def get_warranty_expiry_alerts(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve real-time warranty tracking and expiry alerts.
    Identifies warranties expiring within specified threshold days or recently expired.
    """
    query = db.query(Warranty).join(Product, Warranty.product_id == Product.id)
    if current_user.role == "customer":
        query = query.filter(Product.user_id == current_user.id)

    warranties = query.all()
    today = date.today()
    alerts = []

    for w in warranties:
        p = w.product
        rem_days = (w.warranty_expiry_date - today).days

        if rem_days < 0:
            severity = "EXPIRED"
            message = f"Warranty for {p.product_name} expired {-rem_days} days ago on {w.warranty_expiry_date}."
        elif rem_days <= 7:
            severity = "EXPIRING_CRITICAL"
            message = f"URGENT: Warranty for {p.product_name} expires in {rem_days} days ({w.warranty_expiry_date})."
        elif rem_days <= days:
            severity = "EXPIRING_WARNING"
            message = f"Warranty for {p.product_name} expires in {rem_days} days ({w.warranty_expiry_date})."
        else:
            continue

        alerts.append({
            "warranty_id": w.id,
            "product_id": p.id,
            "product_name": p.product_name,
            "serial_number": p.serial_number,
            "warranty_expiry_date": str(w.warranty_expiry_date),
            "remaining_days": rem_days,
            "severity": severity,
            "message": message
        })

    return {
        "total_alerts": len(alerts),
        "threshold_days": days,
        "alerts": sorted(alerts, key=lambda a: a["remaining_days"])
    }


@router.get("/{product_id}", response_model=WarrantyResponse)
def get_warranty_by_product(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve warranty details for a given product."""
    warranty = db.query(Warranty).filter(Warranty.product_id == product_id).first()
    if not warranty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Warranty not found for product.")

    return warranty
