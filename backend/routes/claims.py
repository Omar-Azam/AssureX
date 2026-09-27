"""
AssureX Claims Processing & Classification Endpoints
===================================================
Claim submission, document upload with SHA-256 duplicate detection,
multimodal ML classification pipeline, status tracking, CSV export, and PDF reporting.
"""

import io
import csv
import hashlib
import json
from datetime import datetime, date
from typing import List, Optional
from pathlib import Path

from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile,
    File, Form, Query, Request, Response
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Claim, Product, Document, Prediction, AuditLog, User
from backend.schemas import (
    ClaimCreate, ClaimResponse, ClaimDetailResponse, ClaimStatusResponse,
    DocumentResponse, PredictionResponse
)
from backend.auth import get_current_user
from backend.config import UPLOADS_DIR, ROLE_CUSTOMER, ROLE_ADMIN, ROLE_CLAIM_REVIEWER
from backend.pipeline import execute_claim_classification_pipeline
from backend.pdf_generator import generate_claim_pdf_report

router = APIRouter(prefix="/api/claims", tags=["Claims"])


def _generate_claim_id(db: Session) -> str:
    """Generate sequential unique claim identifier (e.g. CLM-2026-00001)."""
    count = db.query(Claim).count() + 1
    return f"CLM-{datetime.utcnow().year}-{count:05d}"


# ==============================================================================
# CLAIM CREATION & RETRIEVAL
# ==============================================================================

@router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
def create_claim(
    claim_in: ClaimCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit a new warranty claim against a registered product."""
    product = db.query(Product).filter(Product.id == claim_in.product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target product not found.")

    if current_user.role == ROLE_CUSTOMER and product.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    if claim_in.fault_occurrence_date > date.today():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fault occurrence date cannot be in the future."
        )

    claim_id_str = _generate_claim_id(db)

    claim = Claim(
        claim_id=claim_id_str,
        user_id=current_user.id,
        product_id=product.id,
        claim_submission_date=date.today(),
        fault_occurrence_date=claim_in.fault_occurrence_date,
        fault_description=claim_in.fault_description,
        damage_type=claim_in.damage_type,
        status="Submitted",
        prior_replacement=claim_in.prior_replacement,
        repair_history=claim_in.repair_history
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)

    # Audit Trail Entry
    audit = AuditLog(
        user_id=current_user.id,
        action="CLAIM_SUBMISSION",
        entity_type="Claim",
        entity_id=claim.claim_id,
        details=f"Claim {claim.claim_id} submitted for product {product.product_name} (Damage: {claim.damage_type})",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return claim


@router.get("", response_model=List[ClaimResponse])
def search_and_filter_claims(
    status: Optional[str] = Query(None, description="Filter by status (e.g. Submitted, Approved, Rejected, Manual Review)"),
    category: Optional[str] = Query(None, description="Filter by product category"),
    search: Optional[str] = Query(None, description="Search claim ID, product name, or serial"),
    start_date: Optional[date] = Query(None, description="Filter submission date >= start_date"),
    end_date: Optional[date] = Query(None, description="Filter submission date <= end_date"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search and filter claims with multi-parameter querying:
    - Status filtering
    - Category filtering
    - Text search across claim ID, product name, and serial
    - Submission date range filtering
    """
    query = db.query(Claim).join(Product, Claim.product_id == Product.id)

    # Customer role only sees their own claims
    if current_user.role == ROLE_CUSTOMER:
        query = query.filter(Claim.user_id == current_user.id)

    if status:
        query = query.filter(Claim.status.ilike(f"%{status}%"))

    if category:
        query = query.filter(Product.product_category.ilike(f"%{category}%"))

    if start_date:
        query = query.filter(Claim.claim_submission_date >= start_date)

    if end_date:
        query = query.filter(Claim.claim_submission_date <= end_date)

    if search:
        term = f"%{search}%"
        query = query.filter(
            (Claim.claim_id.ilike(term)) |
            (Product.product_name.ilike(term)) |
            (Product.serial_number.ilike(term)) |
            (Claim.fault_description.ilike(term))
        )

    return query.order_by(Claim.created_at.desc()).all()


# ==============================================================================
# CSV EXPORT (Must precede dynamic {claim_id} path)
# ==============================================================================

@router.get("/export/csv")
def export_claims_to_csv(
    status: Optional[str] = None,
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export filtered claims dataset to downloadable CSV format."""
    query = db.query(Claim).join(Product, Claim.product_id == Product.id)

    if current_user.role == ROLE_CUSTOMER:
        query = query.filter(Claim.user_id == current_user.id)

    if status:
        query = query.filter(Claim.status.ilike(f"%{status}%"))

    if category:
        query = query.filter(Product.product_category.ilike(f"%{category}%"))

    claims = query.order_by(Claim.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Claim ID", "Customer Username", "Product Category", "Product Name",
        "Brand", "Model Number", "Serial Number", "Purchase Date",
        "Submission Date", "Damage Type", "Status", "Final Decision", "Model Consistency"
    ])

    for c in claims:
        p = c.product
        pred = c.prediction
        writer.writerow([
            c.claim_id,
            c.claimant.username if c.claimant else "N/A",
            p.product_category if p else "N/A",
            p.product_name if p else "N/A",
            p.brand if p else "N/A",
            p.model_number if p else "N/A",
            p.serial_number if p else "N/A",
            p.purchase_date if p else "N/A",
            c.claim_submission_date,
            c.damage_type,
            c.status,
            pred.final_decision if pred else "N/A",
            pred.model_consistency_status if pred else "N/A"
        ])

    output.seek(0)
    filename = f"assurex_claims_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ==============================================================================
# INDIVIDUAL CLAIM DETAIL & STATUS
# ==============================================================================

@router.get("/{claim_id}", response_model=ClaimDetailResponse)
def get_claim_details(
    claim_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve full claim record with product, documents, predictions, and reviewer logs."""
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    if current_user.role == ROLE_CUSTOMER and claim.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return claim


@router.get("/{claim_id}/status", response_model=ClaimStatusResponse)
def track_claim_status(
    claim_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Track the lifecycle status and verification indicators of a claim."""
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    if current_user.role == ROLE_CUSTOMER and claim.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    has_dupes = any(d.is_duplicate for d in claim.documents)
    pred = claim.prediction

    return {
        "claim_id": claim.claim_id,
        "status": claim.status,
        "submission_date": claim.claim_submission_date,
        "updated_at": claim.updated_at,
        "final_decision": pred.final_decision if pred else None,
        "model_consistency_status": pred.model_consistency_status if pred else None,
        "documents_count": len(claim.documents),
        "has_duplicates": has_dupes,
        "manual_review_required": (claim.status == "Manual Review") or has_dupes
    }


# ==============================================================================
# DOCUMENT UPLOAD WITH SHA-256 DUPLICATE DETECTION
# ==============================================================================

@router.post("/{claim_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_claim_document(
    claim_id: str,
    request: Request,
    file: UploadFile = File(...),
    file_type: str = Form("receipt", description="receipt, warranty_card, product_image, serial_evidence, other"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a claim document with automated SHA-256 hash computation
    and cross-system duplicate detection.
    """
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target claim not found.")

    if current_user.role == ROLE_CUSTOMER and claim.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    # Read file content and compute SHA-256 hash
    content = await file.read()
    file_size = len(content)
    sha256_hash = hashlib.sha256(content).hexdigest()

    # DUPLICATE DETECTION: Check if identical hash exists in Document table
    existing_doc = db.query(Document).filter(Document.sha256_hash == sha256_hash).first()
    is_duplicate = existing_doc is not None
    duplicate_of_id = existing_doc.id if existing_doc else None

    # Save file to disk
    save_filename = f"{claim.claim_id}_{sha256_hash[:12]}_{file.filename}"
    save_path = UPLOADS_DIR / save_filename
    with open(save_path, "wb") as f:
        f.write(content)

    doc_record = Document(
        claim_id=claim.id,
        file_name=file.filename or "uploaded_file",
        file_path=str(save_path),
        file_type=file_type,
        file_size=file_size,
        sha256_hash=sha256_hash,
        is_duplicate=is_duplicate,
        duplicate_of_document_id=duplicate_of_id
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    # Log document upload and duplicate warning if found
    audit_msg = f"Uploaded '{doc_record.file_name}' ({doc_record.file_type}, SHA256: {sha256_hash[:12]}...)"
    if is_duplicate:
        audit_msg += f" [DUPLICATE DETECTED matching document #{duplicate_of_id}]"

    audit = AuditLog(
        user_id=current_user.id,
        action="DOCUMENT_UPLOAD_DUPLICATE" if is_duplicate else "DOCUMENT_UPLOAD",
        entity_type="Document",
        entity_id=str(doc_record.id),
        details=audit_msg,
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return doc_record


# ==============================================================================
# CLAIM CLASSIFICATION (AI + RULES + DECISION ENGINE)
# ==============================================================================

@router.post("/{claim_id}/classify", response_model=PredictionResponse)
def classify_claim_endpoint(
    claim_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Executes the multi-model classification pipeline:
    1. rule_engine (policy rules)
    2. predict_with_confidence (tabular ML model)
    3. gtm_classifier (visual card classifier)
    4. decision_engine (arbitration & explanation)

    Updates Claim status and logs prediction to AuditLog.
    """
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target claim not found.")

    # Execute pipeline
    result = execute_claim_classification_pipeline(claim)
    dec_res = result["decision_result"]

    final_decision = dec_res.get("final_decision", "Manual Review Required")
    consistency_status = dec_res.get("model_consistency_status", "Uncertain Result")
    confidence_diff = dec_res.get("confidence_difference", 0.0)

    # Save or update Prediction table record
    pred = db.query(Prediction).filter(Prediction.claim_id == claim.id).first()
    if not pred:
        pred = Prediction(claim_id=claim.id)
        db.add(pred)

    pred.python_prediction = json.dumps(result["python_result"])
    pred.gtm_prediction = json.dumps(result["gtm_result"])
    pred.rule_engine_result = json.dumps(result["rule_result"])
    pred.decision_engine_result = json.dumps(dec_res)
    pred.final_decision = final_decision
    pred.model_consistency_status = consistency_status
    pred.confidence_difference = confidence_diff

    # Update claim lifecycle status
    if final_decision == "Likely Valid":
        claim.status = "Approved"
    elif final_decision == "Likely Invalid":
        claim.status = "Rejected"
    else:
        claim.status = "Manual Review"

    db.commit()
    db.refresh(pred)

    # Audit Trail Logging
    audit = AuditLog(
        user_id=current_user.id,
        action="PREDICTION_RUN",
        entity_type="Claim",
        entity_id=claim.claim_id,
        details=f"Classification completed: Decision '{final_decision}' ({consistency_status}, Gap: {confidence_diff:.4f}). Claim status updated to '{claim.status}'",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return {
        "id": pred.id,
        "claim_id": pred.claim_id,
        "final_decision": pred.final_decision,
        "model_consistency_status": pred.model_consistency_status,
        "confidence_difference": pred.confidence_difference,
        "python_prediction": result["python_result"],
        "gtm_prediction": result["gtm_result"],
        "rule_engine_result": result["rule_result"],
        "decision_engine_result": dec_res,
        "created_at": pred.created_at
    }


# ==============================================================================
# DOWNLOAD CLAIM REPORT (PDF)
# ==============================================================================

@router.get("/{claim_id}/report/pdf")
def download_claim_pdf_report(
    claim_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate and stream download official PDF evaluation report."""
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target claim not found.")

    if current_user.role == ROLE_CUSTOMER and claim.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    pdf_buffer = generate_claim_pdf_report(claim)
    filename = f"AssureX_Evaluation_Report_{claim.claim_id}.pdf"

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
