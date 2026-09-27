"""
AssureX Receipt & Invoice Optical Character Recognition (OCR) Engine
=====================================================================

This module extracts and validates key proof-of-purchase fields from uploaded
receipt, invoice, and bill images using EasyOCR / Tesseract with tailored
regular expression parsing and per-field confidence evaluation:

Extracted Fields:
1. purchase_date       - Transaction date (normalized to YYYY-MM-DD)
2. invoice_number      - Invoice / Bill / Receipt tracking identifier
3. product_name        - Item description / device title
4. model_number        - Manufacturer model code (e.g. SM-S928B, 82WK0046US)
5. serial_number       - Serial Number / IMEI / Chassis barcode
6. retailer            - Merchant / Store / Authorized dealer name
7. purchase_amount     - Total transaction price (float & currency)

Output Contract:
----------------
Returns a structured dictionary containing extracted values, normalized data,
raw OCR snippets, and per-field confidence flags:
- 'confidence': 'HIGH' | 'MEDIUM' | 'LOW' | 'UNVERIFIED'
- 'confidence_score': float in [0.0, 1.0]
- 'needs_verification': bool (True if manual user review is advised)
- 'validation_notes': human-readable explanation
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, Union, Tuple, List

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger("AssureX.ReceiptOCR")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] AssureX.ReceiptOCR: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Optional Tesseract / EasyOCR Engine Discovery
TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    # Test if tesseract executable is accessible
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


# Known authorized retailers in Pakistani consumer electronics domain
KNOWN_RETAILERS: List[str] = [
    "Daraz Mall Authorized Brand Store",
    "Daraz Mall",
    "Airlink Communications Official Flagship",
    "Airlink Communications",
    "TechnoCity Prime Electronics",
    "TechnoCity",
    "MegaCity HyperMarket",
    "Hafeez Center Electronics",
    "Hafeez Center",
    "Hyperstar / Carrefour Pakistan",
    "Carrefour Pakistan",
    "Carrefour",
    "Hyperstar",
    "Cnergyico Tech Mart",
    "Airlink Official",
    "Samsung Official Brand Store",
    "Apple Authorized Reseller",
    "Shershah Electronics Market"
]

# Known electronics brands and models for high-confidence matching
KNOWN_BRANDS: List[str] = ["Samsung", "Apple", "Lenovo", "Dell", "HP", "OnePlus", "Xiaomi", "Sony", "LG", "Dawlance", "Haier"]


# ==============================================================================
# IMAGE PREPROCESSING FOR OCR ENHANCEMENT
# ==============================================================================

def preprocess_receipt_image(
    image_input: Union[str, Path, Image.Image]
) -> Image.Image:
    """
    Applies image preprocessing to optimize character edges for OCR:
    1. Grayscale conversion.
    2. Rescaling up to optimal resolution (minimum 1200px width).
    3. Contrast stretching (enhances faded thermal ink).
    4. Subtle sharpening.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    # 1. Convert to grayscale
    gray = img.convert("L")

    # 2. Upscale if image is low resolution (typical thermal mobile photo)
    w, h = gray.size
    if w < 1200:
        factor = 1200.0 / float(w)
        new_size = (int(w * factor), int(h * factor))
        resample_method = getattr(Image, "Resampling", Image).BICUBIC
        gray = gray.resize(new_size, resample_method)

    # 3. Enhance Contrast (boost faint receipt text)
    enhancer = ImageEnhance.Contrast(gray)
    contrast_img = enhancer.enhance(1.8)

    # 4. Sharpen character boundaries
    sharpened = contrast_img.filter(ImageFilter.SHARPEN)

    return sharpened


# ==============================================================================
# OCR TEXT EXTRACTION
# ==============================================================================

def extract_raw_ocr_text(
    image_input: Union[str, Path, Image.Image],
    engine: str = "auto"
) -> Tuple[str, str]:
    """
    Extracts raw text from receipt image using available OCR engine:
    - 'easyocr': Deep learning-based EasyOCR reader
    - 'tesseract': Tesseract OCR engine via pytesseract
    - 'auto': Selects the best available engine

    Returns:
        Tuple of (raw_ocr_text, engine_used)
    """
    processed_img = preprocess_receipt_image(image_input)

    selected_engine = engine.lower()
    if selected_engine == "auto":
        if EASYOCR_AVAILABLE:
            selected_engine = "easyocr"
        elif TESSERACT_AVAILABLE:
            selected_engine = "tesseract"
        else:
            selected_engine = "none"

    if selected_engine == "easyocr":
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR is not installed or failed to load PyTorch.")
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(processed_img)
        raw_text = "\n".join([item[1] for item in results])
        return (raw_text, "EasyOCR")

    elif selected_engine == "tesseract":
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract OCR executable is not installed or not in system PATH.")
        raw_text = pytesseract.image_to_string(processed_img, config="--psm 6")
        return (raw_text, "Tesseract")

    else:
        # Graceful notice if no OCR engine executable is configured in the environment
        raise RuntimeError(
            "No functional OCR engine (EasyOCR or Tesseract) is currently available. "
            "Please ensure Tesseract is installed and added to PATH, or provide raw text directly."
        )


# ==============================================================================
# REGEX PARSING AND VALUE NORMALIZATION
# ==============================================================================

def _parse_normalized_date(date_str: str) -> Optional[str]:
    """Attempts to parse varied date string formats into standard ISO 'YYYY-MM-DD'."""
    cleaned = date_str.strip().split("T")[0]
    cleaned = re.sub(r'\s+\d{1,2}:\d{2}(?::\d{2})?.*$', '', cleaned)
    cleaned = cleaned.replace(",", "")
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y",
        "%b %d %Y", "%B %d %Y", "%b-%d-%Y", "%B-%d-%Y"
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
            return parsed.isoformat()
        except ValueError:
            continue
    try:
        from dateutil import parser
        parsed = parser.parse(cleaned, fuzzy=True).date()
        return parsed.isoformat()
    except Exception:
        pass
    return None


def parse_receipt_text(raw_text: str) -> Dict[str, Any]:
    """
    Parses raw OCR text from a receipt/invoice using specialized regular expressions
    and evaluates per-field confidence ratings for manual verification.

    Parameters:
        raw_text: The string content extracted via OCR.

    Returns:
        Structured dictionary containing:
        - fields: Dictionary of the 7 extracted fields with confidence metadata
        - overall_confidence: float in [0.0, 1.0]
        - requires_manual_review: bool
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # --------------------------------------------------------------------------
    # 1. PURCHASE DATE EXTRACTION
    # --------------------------------------------------------------------------
    purchase_date_val = None
    date_conf = "UNVERIFIED"
    date_score = 0.0
    date_raw = ""
    date_notes = "Date not identified in OCR text."

    # Pattern A: Explicit keyword anchor (Date:, Date of Purchase:, Invoice Date:, Dated:, Txn Date:)
    date_anchor_pattern = re.compile(
        r'(?:Date\s*of\s*Purchase|Purchase\s*Date|Invoice\s*Date|Date|Dated|Txn\s*Date|Billing\s*Date)\s*[:#-]?\s*'
        r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/\s]+[A-Za-z]{3,9}[-/\s]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2}[,\s]+\d{2,4})',
        re.IGNORECASE
    )
    match_date = date_anchor_pattern.search(raw_text)

    if match_date:
        candidate = match_date.group(1).strip()
        iso_dt = _parse_normalized_date(candidate)
        if iso_dt:
            purchase_date_val = iso_dt
            date_raw = match_date.group(0)
            date_conf = "HIGH"
            date_score = 0.95
            date_notes = f"Matched explicit date anchor '{match_date.group(0).split(':')[0]}' with ISO-valid date."
    else:
        # Pattern B: Unanchored standalone date in standard formats
        standalone_date_pattern = re.compile(
            r'\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b'
        )
        match_dt_standalone = standalone_date_pattern.search(raw_text)
        if match_dt_standalone:
            candidate = match_dt_standalone.group(1).strip()
            iso_dt = _parse_normalized_date(candidate)
            if iso_dt:
                purchase_date_val = iso_dt
                date_raw = match_dt_standalone.group(0)
                date_conf = "MEDIUM"
                date_score = 0.65
                date_notes = "Extracted standalone date format without explicit anchor keyword."

    # Validate temporal plausibility
    if purchase_date_val:
        try:
            d_obj = datetime.strptime(purchase_date_val, "%Y-%m-%d").date()
            if d_obj > date.today():
                date_conf = "LOW"
                date_score = 0.25
                date_notes = f"Warning: Extracted date ({purchase_date_val}) is in the future. Requires manual verification."
            elif (date.today() - d_obj).days > 365 * 10:
                date_conf = "LOW"
                date_score = 0.30
                date_notes = f"Warning: Date ({purchase_date_val}) is older than 10 years."
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 2. INVOICE NUMBER EXTRACTION
    # --------------------------------------------------------------------------
    invoice_val = None
    inv_conf = "UNVERIFIED"
    inv_score = 0.0
    inv_raw = ""
    inv_notes = "Invoice / Bill number not detected."

    # Pattern A: Anchor keyword (Invoice No:, Tax Invoice:, Bill #:, Receipt:, Order #:)
    inv_anchor_pattern = re.compile(
        r'(?:Invoice\s*No|Invoice\s*#|Invoice\s*ID|Invoice|Receipt\s*No|Receipt\s*#|Tax\s*Invoice|Bill\s*No|Order\s*No|Order\s*#)[^\S\r\n]*[:#-][^\S\r\n]*([A-Za-z0-9_/-]{4,30})',
        re.IGNORECASE
    )
    for m in inv_anchor_pattern.finditer(raw_text):
        candidate = m.group(1).strip()
        if candidate.lower() not in ["invoice", "receipt", "bill", "order", "details", "customer"] and re.search(r'\d', candidate):
            invoice_val = candidate
            inv_raw = m.group(0)
            inv_conf = "HIGH"
            inv_score = 0.92
            inv_notes = f"Found explicit invoice label: '{inv_raw}'."
            break

    if not invoice_val:
        # Pattern B: Standard alphanumeric invoice code (e.g. INV-2025-12345, TAX-8921)
        inv_code_pattern = re.compile(r'\b(?:INV|REC|TAX|BILL|ORD|SLIP)[-_/]?\d{4,10}\b', re.IGNORECASE)
        match_code = inv_code_pattern.search(raw_text)
        if match_code:
            invoice_val = match_code.group(0).strip()
            inv_raw = match_code.group(0)
            inv_conf = "MEDIUM"
            inv_score = 0.70
            inv_notes = f"Matched standard invoice prefix pattern: '{invoice_val}'."

    # --------------------------------------------------------------------------
    # 3. SERIAL NUMBER EXTRACTION
    # --------------------------------------------------------------------------
    serial_val = None
    serial_conf = "UNVERIFIED"
    serial_score = 0.0
    serial_raw = ""
    serial_notes = "Serial number / IMEI not identified."

    # Pattern A: Explicit label (Serial No:, S/N:, SN:, IMEI:, Chassis:)
    serial_anchor_pattern = re.compile(
        r'(?:Serial\s*No|Serial\s*Number|Serial\s*#|Serial|S/N|SN|IMEI\s*1|IMEI|Chassis\s*No|Chassis)\s*[:#-]?\s*([A-Za-z0-9\-_]{6,25})',
        re.IGNORECASE
    )
    match_sn = serial_anchor_pattern.search(raw_text)

    if match_sn:
        candidate_sn = match_sn.group(1).strip().upper()
        # Filter out common false positives like 'NUMBER', 'DATE'
        if candidate_sn not in {"NUMBER", "NO", "DATE", "MODEL", "ITEM"}:
            serial_val = candidate_sn
            serial_raw = match_sn.group(0)
            # High confidence if matches standard IMEI (14-16 digits) or brand serial
            if re.match(r'^\d{14,16}$', serial_val):
                serial_conf = "HIGH"
                serial_score = 0.96
                serial_notes = f"Verified 14-16 digit standard IMEI format: '{serial_val}'."
            elif len(serial_val) >= 8:
                serial_conf = "HIGH"
                serial_score = 0.90
                serial_notes = f"Matched explicit serial label: '{serial_val}'."
            else:
                serial_conf = "MEDIUM"
                serial_score = 0.65
                serial_notes = f"Short serial number length ({len(serial_val)} chars)."
    else:
        # Pattern B: Standalone 14-16 digit IMEI or brand serial prefix (LEN-, SAM-, etc.)
        imei_pattern = re.compile(r'\b(\d{14,16})\b')
        match_imei = imei_pattern.search(raw_text)
        if match_imei:
            serial_val = match_imei.group(1)
            serial_raw = match_imei.group(0)
            serial_conf = "MEDIUM"
            serial_score = 0.75
            serial_notes = "Extracted unanchored 14-16 digit IMEI candidate."
        else:
            brand_sn_pattern = re.compile(r'\b((?:LEN|SAM|HP|DELL)[-_][A-Za-z0-9\-_]{7,20})\b', re.IGNORECASE)
            match_bsn = brand_sn_pattern.search(raw_text)
            if match_bsn:
                serial_val = match_bsn.group(1).upper()
                serial_raw = match_bsn.group(0)
                serial_conf = "MEDIUM"
                serial_score = 0.70
                serial_notes = "Extracted manufacturer brand serial prefix pattern."

    # --------------------------------------------------------------------------
    # 4. MODEL NUMBER EXTRACTION
    # --------------------------------------------------------------------------
    model_val = None
    model_conf = "UNVERIFIED"
    model_score = 0.0
    model_raw = ""
    model_notes = "Model number / code not detected."

    # Pattern A: Explicit label (Model:, Model No:, M/N:, Item Code:, SKU:)
    model_anchor_pattern = re.compile(
        r'(?:Model\s*No|Model\s*Number|Model\s*#|Model|M/N|SKU|Item\s*Code|Part\s*No)\s*[:#-]?\s*([A-Za-z0-9\-_/]{4,20})',
        re.IGNORECASE
    )
    match_model = model_anchor_pattern.search(raw_text)

    if match_model:
        candidate_model = match_model.group(1).strip().upper()
        if candidate_model not in {"NO", "NUMBER", "NAME", "CODE", "TYPE"}:
            model_val = candidate_model
            model_raw = match_model.group(0)
            model_conf = "HIGH"
            model_score = 0.91
            model_notes = f"Matched explicit model anchor: '{model_raw}'."
    else:
        # Pattern B: Recognizable OEM model structures (e.g. SM-S928B, 82WK0046US, WA13CG5441BY, CPH2609)
        oem_pattern = re.compile(
            r'\b(SM-[A-Z0-9]{4,6}|WA\d{2}[A-Z0-9]{5,8}|82[A-Z0-9]{6,10}|CPH\d{4}|A\d{4}|[A-Z]{2}\d{4}[A-Z0-9]{2,6})\b',
            re.IGNORECASE
        )
        match_oem = oem_pattern.search(raw_text)
        if match_oem:
            model_val = match_oem.group(1).upper()
            model_raw = match_oem.group(0)
            model_conf = "MEDIUM"
            model_score = 0.75
            model_notes = f"Detected standard OEM model numbering format: '{model_val}'."

    # --------------------------------------------------------------------------
    # 5. PRODUCT NAME EXTRACTION
    # --------------------------------------------------------------------------
    product_val = None
    prod_conf = "UNVERIFIED"
    prod_score = 0.0
    prod_raw = ""
    prod_notes = "Product title / item description not detected."

    # Pattern A: Explicit item label (Item:, Product:, Description:, Desc:)
    prod_anchor_pattern = re.compile(
        r'(?:Item\s*Description|Product\s*Name|Description|Item\s*Desc|Product|Item|Desc)\s*[:#-]\s*([^\n]+)',
        re.IGNORECASE
    )
    for match_prod in prod_anchor_pattern.finditer(raw_text):
        candidate_prod = match_prod.group(1).strip()
        # Clean trailing quantity or price if OCR merged lines
        candidate_prod = re.sub(r'\s+(?:Qty|Price|PKR|Rs).*$', '', candidate_prod, flags=re.IGNORECASE).strip()
        if (
            len(candidate_prod) >= 3
            and not candidate_prod.lower().startswith(("qty", "price", "amount", "detail", "total", "subtotal"))
            and not candidate_prod.endswith(":")
        ):
            product_val = candidate_prod
            prod_raw = match_prod.group(0)
            prod_conf = "HIGH"
            prod_score = 0.88
            prod_notes = f"Extracted via product item label: '{product_val}'."
            break

    # Pattern B: Search for known brand occurrences
    if not product_val:
        for brand in KNOWN_BRANDS:
            b_pattern = re.compile(rf'({brand}\s+[A-Za-z0-9\s\-+]+)', re.IGNORECASE)
            b_match = b_pattern.search(raw_text)
            if b_match:
                p_text = b_match.group(1).split("\n")[0].strip()
                if len(p_text) >= 5:
                    product_val = p_text[:60].strip()
                    prod_raw = p_text
                    prod_conf = "MEDIUM"
                    prod_score = 0.70
                    prod_notes = f"Matched known consumer electronics brand '{brand}' in text."
                    break

    # --------------------------------------------------------------------------
    # 6. RETAILER EXTRACTION
    # --------------------------------------------------------------------------
    retailer_val = None
    ret_conf = "UNVERIFIED"
    ret_score = 0.0
    ret_raw = ""
    ret_notes = "Retailer / store name not detected."

    # Check 1: Known authorized retailer lookup (highest fidelity)
    for known_ret in KNOWN_RETAILERS:
        if known_ret.lower() in raw_text.lower():
            retailer_val = known_ret
            ret_raw = known_ret
            ret_conf = "HIGH"
            ret_score = 0.95
            ret_notes = f"Verified AssureX authorized retailer: '{known_ret}'."
            break

    # Check 2: Explicit label (Retailer:, Store:, Sold By:, Merchant:, Shop:)
    if not retailer_val:
        ret_anchor_pattern = re.compile(
            r'(?:Sold\s*By|Retailer|Store|Merchant|Vendor|Shop|Dealer)\s*[:#-]?\s*([^\n]+)',
            re.IGNORECASE
        )
        match_ret = ret_anchor_pattern.search(raw_text)
        if match_ret:
            candidate_ret = match_ret.group(1).strip()
            if len(candidate_ret) >= 3:
                retailer_val = candidate_ret
                ret_raw = match_ret.group(0)
                ret_conf = "HIGH"
                ret_score = 0.85
                ret_notes = f"Matched explicit retailer tag: '{retailer_val}'."

    # Check 3: Header inspection (first 1-3 non-empty lines typically contain store header)
    if not retailer_val and lines:
        for header_line in lines[:3]:
            cleaned_header = re.sub(r'[^A-Za-z0-9\s]', '', header_line).strip()
            # If line looks like a business title (contains Store, Electronics, Tech, Mart, Mall)
            if any(term in cleaned_header.lower() for term in ["store", "electronics", "tech", "mart", "mall", "market", "shop", "enterprise"]):
                retailer_val = header_line.strip()
                ret_raw = header_line
                ret_conf = "MEDIUM"
                ret_score = 0.65
                ret_notes = f"Extracted store name from receipt header line: '{retailer_val}'."
                break

    # --------------------------------------------------------------------------
    # 7. PURCHASE AMOUNT EXTRACTION
    # --------------------------------------------------------------------------
    amount_val = None
    amt_conf = "UNVERIFIED"
    amt_score = 0.0
    amt_raw = ""
    amt_notes = "Purchase amount / total not detected."

    # Pattern A: Anchor keyword (Grand Total:, Total Amount:, Total Due:, Net Amount:, Total:)
    amount_anchor_pattern = re.compile(
        r'(?:Grand\s*Total|Total\s*Amount|Net\s*Amount|Amount\s*Paid|Total\s*Due|Total|Amount|Price)\s*[:#-]?\s*'
        r'(?:PKR|Rs\.?|USD|\$)?\s*([\d,]+(?:\.\d{1,2})?)',
        re.IGNORECASE
    )
    # Search in reverse lines or all matches to prioritize Grand Total near the bottom
    amt_matches = list(amount_anchor_pattern.finditer(raw_text))
    if amt_matches:
        # Choose the match associated with 'Grand Total' or 'Total' with highest value
        best_match = amt_matches[-1]  # Bottom-most total is typically Grand Total
        for m in amt_matches:
            if "grand" in m.group(0).lower():
                best_match = m
                break

        num_str = best_match.group(1).replace(",", "").strip()
        try:
            val_float = float(num_str)
            if val_float > 0:
                amount_val = val_float
                amt_raw = best_match.group(0)
                amt_conf = "HIGH"
                amt_score = 0.94
                amt_notes = f"Extracted via total anchor '{best_match.group(0).split(':')[0]}': PKR {amount_val:,.2f}."
        except ValueError:
            pass

    # Pattern B: Explicit currency prefix (PKR 245,500.00 or Rs. 99,500)
    if amount_val is None:
        currency_pattern = re.compile(r'(?:PKR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE)
        cur_matches = list(currency_pattern.finditer(raw_text))
        if cur_matches:
            # Pick highest amount found with currency symbol
            max_amt = 0.0
            best_cur_match = None
            for cm in cur_matches:
                try:
                    num_val = float(cm.group(1).replace(",", ""))
                    if num_val > max_amt:
                        max_amt = num_val
                        best_cur_match = cm
                except ValueError:
                    continue
            if max_amt > 0:
                amount_val = max_amt
                amt_raw = best_cur_match.group(0)
                amt_conf = "MEDIUM"
                amt_score = 0.70
                amt_notes = f"Extracted highest currency-prefixed amount: PKR {amount_val:,.2f}."

    # --------------------------------------------------------------------------
    # ASSEMBLE OUTPUT SCHEMA & COMPUTE OVERALL CONFIDENCE
    # --------------------------------------------------------------------------
    fields = {
        "purchase_date": {
            "value": purchase_date_val,
            "confidence": date_conf,
            "confidence_score": round(date_score, 2),
            "needs_verification": date_conf != "HIGH",
            "raw_match": date_raw,
            "notes": date_notes
        },
        "invoice_number": {
            "value": invoice_val,
            "confidence": inv_conf,
            "confidence_score": round(inv_score, 2),
            "needs_verification": inv_conf != "HIGH",
            "raw_match": inv_raw,
            "notes": inv_notes
        },
        "product_name": {
            "value": product_val,
            "confidence": prod_conf,
            "confidence_score": round(prod_score, 2),
            "needs_verification": prod_conf != "HIGH",
            "raw_match": prod_raw,
            "notes": prod_notes
        },
        "model_number": {
            "value": model_val,
            "confidence": model_conf,
            "confidence_score": round(model_score, 2),
            "needs_verification": model_conf != "HIGH",
            "raw_match": model_raw,
            "notes": model_notes
        },
        "serial_number": {
            "value": serial_val,
            "confidence": serial_conf,
            "confidence_score": round(serial_score, 2),
            "needs_verification": serial_conf != "HIGH",
            "raw_match": serial_raw,
            "notes": serial_notes
        },
        "retailer": {
            "value": retailer_val,
            "confidence": ret_conf,
            "confidence_score": round(ret_score, 2),
            "needs_verification": ret_conf != "HIGH",
            "raw_match": ret_raw,
            "notes": ret_notes
        },
        "purchase_amount": {
            "value": amount_val,
            "confidence": amt_conf,
            "confidence_score": round(amt_score, 2),
            "needs_verification": amt_conf != "HIGH",
            "raw_match": amt_raw,
            "notes": amt_notes
        }
    }

    # Overall Confidence Calculation
    scores = [f["confidence_score"] for f in fields.values()]
    overall_confidence = round(sum(scores) / len(scores), 2)

    # Flag for manual verification if any critical field is not HIGH confidence
    critical_fields = ["purchase_date", "serial_number", "purchase_amount"]
    requires_manual_review = any(fields[k]["needs_verification"] for k in critical_fields)

    return {
        "raw_text": raw_text,
        "fields": fields,
        "overall_confidence": overall_confidence,
        "requires_manual_review": requires_manual_review
    }


# ==============================================================================
# MAIN ENTRYPOINT FUNCTION
# ==============================================================================

def extract_receipt_data(
    image_input: Union[str, Path, Image.Image],
    ocr_engine: str = "auto",
    custom_raw_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level entrypoint that processes an uploaded receipt image,
    runs OCR, applies regex parsing, and returns extracted fields with confidence flags.

    Parameters:
        image_input: Path to image file, Path object, or PIL Image.
        ocr_engine: 'auto', 'easyocr', or 'tesseract'.
        custom_raw_text: Optional pre-extracted raw text for testing or manual override.

    Returns:
        Structured dictionary with extracted fields, confidence scores, and manual review flags.
    """
    engine_used = "Manual Input"

    if custom_raw_text is not None:
        raw_text = custom_raw_text
    else:
        try:
            raw_text, engine_used = extract_raw_ocr_text(image_input, engine=ocr_engine)
        except Exception as exc:
            logger.error(f"OCR extraction failed: {exc}")
            raise RuntimeError(f"Failed to extract text from receipt image: {exc}") from exc

    result = parse_receipt_text(raw_text)
    result["engine_used"] = engine_used
    return result


def cross_verify_receipt_with_claim(
    extracted_data: Dict[str, Any],
    claim_record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares OCR extracted receipt data against submitted claim metadata.
    Identifies discrepancies in serial number, purchase date, or retailer.
    """
    fields = extracted_data.get("fields", {})
    discrepancies: List[str] = []

    # 1. Serial Number Match
    extracted_sn = fields.get("serial_number", {}).get("value")
    claim_sn = claim_record.get("serial_number")
    if extracted_sn and claim_sn:
        if str(extracted_sn).strip().upper() != str(claim_sn).strip().upper():
            discrepancies.append(
                f"Serial Number Mismatch: Receipt OCR shows '{extracted_sn}' but claim specifies '{claim_sn}'."
            )

    # 2. Purchase Date Match
    extracted_date = fields.get("purchase_date", {}).get("value")
    claim_date = str(claim_record.get("purchase_date") or "").strip()
    if extracted_date and claim_date:
        if extracted_date != claim_date:
            discrepancies.append(
                f"Purchase Date Mismatch: Receipt OCR shows '{extracted_date}' but claim specifies '{claim_date}'."
            )

    # 3. Retailer Match
    extracted_retailer = fields.get("retailer", {}).get("value")
    claim_retailer = str(claim_record.get("retailer") or "").strip()
    if extracted_retailer and claim_retailer:
        if extracted_retailer.lower() not in claim_retailer.lower() and claim_retailer.lower() not in extracted_retailer.lower():
            discrepancies.append(
                f"Retailer Discrepancy: Receipt OCR shows '{extracted_retailer}' vs claim '{claim_retailer}'."
            )

    return {
        "is_verified": len(discrepancies) == 0,
        "discrepancies": discrepancies,
        "match_count": 3 - len(discrepancies)
    }


# Backwards compatibility and route aliases
extract_receipt_data_from_image = extract_receipt_data
verify_extracted_receipt_data = cross_verify_receipt_with_claim



# ==============================================================================
# COMMAND-LINE DEMO & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print(" AssureX Receipt & Invoice OCR Parsing Engine - Demo")
    print("=" * 70)

    # Sample realistic Pakistani retail receipt text
    sample_receipt_text = """
    ============================================================
                DARAZ MALL AUTHORIZED BRAND STORE
            STRN: 3277876123456 | NTN: 489210-9 | POS-04
    ============================================================
    TAX INVOICE: INV-2025-94812
    Date: 2025-10-25 15:45:00
    Customer: Ali Khan (Lahore)
    ------------------------------------------------------------
    Item: Samsung Galaxy S24 Ultra 512GB Titanium Gray
    Model No: SM-S928B
    Serial No: 35874247858703
    Quantity: 1
    Unit Price: PKR 245,500.00
    ------------------------------------------------------------
    Subtotal: PKR 245,500.00
    GST (18%): Inclusive
    GRAND TOTAL: PKR 245,500.00
    Payment: Debit Card (Ending 8841)
    ============================================================
    Thank you for purchasing with official warranty coverage!
    """

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Processing uploaded receipt image: {img_path}...")
        try:
            parsed = extract_receipt_data(img_path)
        except Exception as err:
            print(f"[ERROR] {err}")
            sys.exit(1)
    else:
        print("Running demonstration with realistic retail receipt text:\n")
        parsed = extract_receipt_data(None, custom_raw_text=sample_receipt_text)

    print(f"Engine Used         : {parsed.get('engine_used', 'N/A')}")
    print(f"Overall Confidence  : {parsed['overall_confidence'] * 100:.1f}%")
    print(f"Requires User Review: {'YES (Verification Needed)' if parsed['requires_manual_review'] else 'NO (High Confidence)'}")
    print("-" * 70)
    print(f"{'FIELD':<18} | {'EXTRACTED VALUE':<32} | {'CONFIDENCE':<10} | {'VERIFY?'}")
    print("-" * 70)

    for field_name, meta in parsed["fields"].items():
        val_str = str(meta["value"]) if meta["value"] is not None else "[NOT DETECTED]"
        if len(val_str) > 30:
            val_str = val_str[:27] + "..."
        conf = meta["confidence"]
        needs_v = "YES" if meta["needs_verification"] else "NO"
        print(f"{field_name:<18} | {val_str:<32} | {conf:<10} | {needs_v}")

    print("=" * 70)
