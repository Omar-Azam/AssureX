"""
Unit Tests for AssureX Receipt & Invoice OCR Engine
===================================================

Validates:
1. High-accuracy regex extraction across all 7 proof-of-purchase fields:
   - purchase_date
   - invoice_number
   - product_name
   - model_number
   - serial_number
   - retailer
   - purchase_amount
2. Per-field confidence scoring and manual verification flags
3. Date parsing and normalization to standard ISO YYYY-MM-DD
4. Multi-format serial number and IMEI parsing
5. Currency and purchase amount extraction
6. Image preprocessing pipeline (grayscale, contrast, sharpening)
7. Cross-system verification against submitted claim records
"""

import sys
from pathlib import Path
from PIL import Image

# Add workspace to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from receipt_ocr import (
    parse_receipt_text,
    extract_receipt_data,
    preprocess_receipt_image,
    cross_verify_receipt_with_claim
)


SAMPLE_COMPLETE_RECEIPT = """
============================================================
            HAFEEZ CENTER ELECTRONICS MEGASTORE
          STRN: 3277876123456 | NTN: 489210-9 | POS-01
============================================================
TAX INVOICE: INV-2025-48192
Date: 2025-11-14 11:20:00
Customer Name: Muhammad Usman
------------------------------------------------------------
Item: Lenovo Legion Pro 5i Gaming Laptop 16GB
Model No: 82WK0046US
Serial No: LEN-LA-2025-BT7UFBW
Quantity: 1
Price: PKR 417,500.00
------------------------------------------------------------
Subtotal: PKR 417,500.00
Sales Tax: Inclusive
GRAND TOTAL: PKR 417,500.00
Payment Method: Online Bank Transfer
============================================================
Authorized Warranty Document. Retain for claim filing.
"""


def test_complete_receipt_extraction_and_confidence():
    """Verify clean extraction of all 7 fields with HIGH confidence."""
    result = parse_receipt_text(SAMPLE_COMPLETE_RECEIPT)
    fields = result["fields"]

    # 1. Purchase Date
    assert fields["purchase_date"]["value"] == "2025-11-14"
    assert fields["purchase_date"]["confidence"] == "HIGH"
    assert fields["purchase_date"]["needs_verification"] is False

    # 2. Invoice Number
    assert fields["invoice_number"]["value"] == "INV-2025-48192"
    assert fields["invoice_number"]["confidence"] == "HIGH"
    assert fields["invoice_number"]["needs_verification"] is False

    # 3. Product Name
    assert "Lenovo Legion Pro 5i" in fields["product_name"]["value"]
    assert fields["product_name"]["confidence"] == "HIGH"

    # 4. Model Number
    assert fields["model_number"]["value"] == "82WK0046US"
    assert fields["model_number"]["confidence"] == "HIGH"

    # 5. Serial Number
    assert fields["serial_number"]["value"] == "LEN-LA-2025-BT7UFBW"
    assert fields["serial_number"]["confidence"] == "HIGH"
    assert fields["serial_number"]["needs_verification"] is False

    # 6. Retailer
    assert "Hafeez Center" in fields["retailer"]["value"]
    assert fields["retailer"]["confidence"] == "HIGH"

    # 7. Purchase Amount
    assert fields["purchase_amount"]["value"] == 417500.0
    assert fields["purchase_amount"]["confidence"] == "HIGH"

    # Overall Confidence
    assert result["overall_confidence"] >= 0.85
    assert result["requires_manual_review"] is False


def test_imei_and_smartphone_receipt():
    """Verify standard 14-16 digit IMEI and Samsung smartphone receipt parsing."""
    phone_receipt = """
    Airlink Communications Official Flagship
    Bill No: BILL-8921-2025
    Invoice Date: 25/10/2025
    Description: Samsung Galaxy S24 Ultra
    Model: SM-S928B
    IMEI: 35874247858703
    Total: Rs. 245,500.00
    """
    result = parse_receipt_text(phone_receipt)
    fields = result["fields"]

    assert fields["purchase_date"]["value"] == "2025-10-25"
    assert fields["invoice_number"]["value"] == "BILL-8921-2025"
    assert fields["serial_number"]["value"] == "35874247858703"
    assert fields["serial_number"]["confidence"] == "HIGH"
    assert fields["model_number"]["value"] == "SM-S928B"
    assert fields["purchase_amount"]["value"] == 245500.0


def test_date_formatting_variations():
    """Verify varied date string formats convert to standardized ISO YYYY-MM-DD."""
    dates_to_test = [
        ("Date: 2025-06-15", "2025-06-15"),
        ("Invoice Date: 15/06/2025", "2025-06-15"),
        ("Dated: 15-Jun-2025", "2025-06-15"),
        ("Txn Date: June 15, 2025", "2025-06-15")
    ]
    for raw_snippet, expected_iso in dates_to_test:
        res = parse_receipt_text(raw_snippet)
        assert res["fields"]["purchase_date"]["value"] == expected_iso
        assert res["fields"]["purchase_date"]["confidence"] in ["HIGH", "MEDIUM"]


def test_future_date_flags_manual_review():
    """Verify receipt with future date is flagged with LOW confidence for user verification."""
    future_receipt = "Date: 2035-01-01\nTotal: PKR 100,000.00"
    res = parse_receipt_text(future_receipt)
    assert res["fields"]["purchase_date"]["confidence"] == "LOW"
    assert res["fields"]["purchase_date"]["needs_verification"] is True
    assert "future" in res["fields"]["purchase_date"]["notes"].lower()


def test_incomplete_receipt_flags_manual_verification():
    """Verify missing critical fields (serial, amount) trigger manual review flag."""
    sparse_receipt = """
    Hafeez Center
    Cash Receipt
    Date: 2025-04-10
    """
    res = parse_receipt_text(sparse_receipt)
    assert res["fields"]["serial_number"]["value"] is None
    assert res["fields"]["serial_number"]["confidence"] == "UNVERIFIED"
    assert res["fields"]["serial_number"]["needs_verification"] is True
    assert res["requires_manual_review"] is True


def test_image_preprocessing_pipeline():
    """Verify PIL image preprocessing handles dimensions, grayscale, and contrast."""
    test_img = Image.new("RGB", (600, 800), color=(240, 240, 240))
    processed = preprocess_receipt_image(test_img)

    assert isinstance(processed, Image.Image)
    assert processed.mode == "L"
    assert processed.size[0] >= 1200


def test_cross_verify_receipt_with_claim():
    """Verify comparing extracted receipt fields against claim metadata."""
    receipt_data = parse_receipt_text(SAMPLE_COMPLETE_RECEIPT)

    # 1. Matching Claim
    valid_claim = {
        "serial_number": "LEN-LA-2025-BT7UFBW",
        "purchase_date": "2025-11-14",
        "retailer": "Hafeez Center Electronics Megastore"
    }
    v_result = cross_verify_receipt_with_claim(receipt_data, valid_claim)
    assert v_result["is_verified"] is True
    assert len(v_result["discrepancies"]) == 0
    assert v_result["match_count"] == 3

    # 2. Mismatching Claim (Fraud / swapped paperwork detection)
    tampered_claim = {
        "serial_number": "DIFFERENT-SERIAL-9999",
        "purchase_date": "2025-01-01",
        "retailer": "Daraz Mall"
    }
    v_mismatch = cross_verify_receipt_with_claim(receipt_data, tampered_claim)
    assert v_mismatch["is_verified"] is False
    assert len(v_mismatch["discrepancies"]) == 3
    assert any("Serial Number Mismatch" in d for d in v_mismatch["discrepancies"])


if __name__ == "__main__":
    print("Running AssureX Receipt OCR Unit Tests...")
    test_complete_receipt_extraction_and_confidence()
    test_imei_and_smartphone_receipt()
    test_date_formatting_variations()
    test_future_date_flags_manual_review()
    test_incomplete_receipt_flags_manual_verification()
    test_image_preprocessing_pipeline()
    test_cross_verify_receipt_with_claim()
    print("\n[PASS] All 7 Receipt OCR unit tests passed successfully!")
