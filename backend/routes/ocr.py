"""
AssureX Optical Character Recognition (OCR) Endpoints
=====================================================
Receipt invoice parsing, OCR field extraction, and data cross-verification.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Product
from backend.auth import get_current_user
from backend.config import UPLOADS_DIR
from receipt_ocr import (
    extract_receipt_data_from_image,
    parse_receipt_text,
    verify_extracted_receipt_data
)

router = APIRouter(prefix="/api/ocr", tags=["Receipt OCR & Extraction"])


@router.post("/extract")
async def extract_receipt_fields(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """
    Extract structured receipt/invoice telemetry:
    - Purchase Date
    - Invoice Number
    - Product Name
    - Model Number
    - Serial Number
    - Retailer
    - Purchase Amount
    Returns extracted fields with confidence levels (HIGH, MEDIUM, LOW, UNVERIFIED).
    """
    if not file and not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an image file or raw OCR text must be provided."
        )

    if file:
        allowed_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".pdf"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(allowed_exts)}"
            )

        ocr_dir = UPLOADS_DIR / "ocr_temp"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        temp_dest = ocr_dir / f"ocr_{os.urandom(8).hex()}{file_ext}"

        try:
            with open(temp_dest, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)

            extracted_data = extract_receipt_data_from_image(str(temp_dest))
        finally:
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except Exception:
                    pass
    else:
        extracted_data = parse_receipt_text(raw_text)

    return {
        "status": "success",
        "extracted_data": extracted_data,
        "overall_confidence": extracted_data.get("overall_confidence", "MEDIUM")
    }


@router.post("/verify")
def verify_receipt_data(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Cross-verifies extracted OCR receipt data against registered product and claim record.
    Flags serial number matches/mismatches, retailer authorization, and amount discrepancies.
    """
    extracted_data = payload.get("extracted_data")
    if not extracted_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing extracted_data in request.")

    product_id = payload.get("product_id")
    claim_dict = payload.get("claim_record", {})

    if product_id:
        prod = db.query(Product).filter(Product.id == product_id).first()
        if prod:
            claim_dict.update({
                "purchase_date": str(prod.purchase_date),
                "serial_number": prod.serial_number,
                "model_number": prod.model_number,
                "product_name": prod.product_name,
                "retailer": prod.retailer,
                "purchase_price": prod.purchase_price
            })

    verification_result = verify_extracted_receipt_data(extracted_data, claim_dict)
    return {
        "status": "success",
        "verification": verification_result
    }
