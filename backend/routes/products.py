"""
AssureX Product Registration Endpoints
======================================
Product catalog management, serial number registration, and customer linking.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Product, User, AuditLog
from backend.schemas import ProductCreate, ProductResponse
from backend.auth import get_current_user, require_roles
from backend.config import ROLE_ADMIN, ROLE_CLAIM_REVIEWER, ROLE_SERVICE_CENTER

router = APIRouter(prefix="/api/products", tags=["Products"])


@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def register_product(
    product_in: ProductCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Register a new consumer electronics product with unique serial number."""
    # Check if serial number already registered
    existing = db.query(Product).filter(Product.serial_number == product_in.serial_number).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A product with serial number '{product_in.serial_number}' is already registered."
        )

    product = Product(
        product_category=product_in.product_category,
        product_name=product_in.product_name,
        brand=product_in.brand,
        model_number=product_in.model_number,
        serial_number=product_in.serial_number,
        purchase_price=product_in.purchase_price,
        purchase_date=product_in.purchase_date,
        retailer=product_in.retailer,
        user_id=current_user.id
    )
    db.add(product)
    db.commit()
    db.refresh(product)

    # Log to audit trail
    audit = AuditLog(
        user_id=current_user.id,
        action="PRODUCT_REGISTRATION",
        entity_type="Product",
        entity_id=str(product.id),
        details=f"Product '{product.product_name}' (SN: {product.serial_number}) registered",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return product


@router.get("", response_model=List[ProductResponse])
def list_products(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List registered products.
    Customers see only their own products; reviewers and admins see all.
    """
    query = db.query(Product)
    if current_user.role == "customer":
        query = query.filter(Product.user_id == current_user.id)

    if category:
        query = query.filter(Product.product_category.ilike(f"%{category}%"))

    return query.order_by(Product.created_at.desc()).all()


@router.get("/{product_id}", response_model=ProductResponse)
def get_product_by_id(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve single product details."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")

    if current_user.role == "customer" and product.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return product
