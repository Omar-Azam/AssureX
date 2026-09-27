"""
AssureX Mock OCR Parsing & Extraction Test Suite
=================================================
Validates receipt OCR text parsing with mock OCR engine responses:
1. Complete receipt parsing: extracts all 7 proof-of-purchase fields.
2. Per-field confidence scoring and manual verification flags.
3. Partial/noisy receipt handling (missing serial or invoice number).
4. Date normalization from various receipt formats to ISO YYYY-MM-DD.
5. Smartphone IMEI extraction from receipt text.
6. Cross-verification against claim metadata.
"""

import pytest
from receipt_ocr import parse_receipt_text, cross_verify_receipt_with_claim


@pytest.mark.ocr
def test_mock_complete_receipt_extraction(mock_complete_ocr_text: str):
    """Verify high-accuracy extraction of all 7 fields from complete receipt text."""
    data = parse_receipt_text(mock_complete_ocr_text)
    f = data["fields"]

    # 1. Purchase Date
    assert f["purchase_date"]["value"] == "2025-11-14"
    assert f["purchase_date"]["confidence_score"] > 0.8
    assert f["purchase_date"]["needs_verification"] is False

    # 2. Invoice Number
    assert f["invoice_number"]["value"] == "INV-2025-99412"
    assert f["invoice_number"]["confidence_score"] > 0.8
    assert f["invoice_number"]["needs_verification"] is False

    # 3. Product Name
    assert "Lenovo Legion" in f["product_name"]["value"]

    # 4. Model Number
    assert f["model_number"]["value"] == "82WK0046US"

    # 5. Serial Number
    assert f["serial_number"]["value"] == "LEN-LA-2025-BT7UFBW"
    assert f["serial_number"]["confidence_score"] > 0.8
    assert f["serial_number"]["needs_verification"] is False

    # 6. Retailer
    assert "Hafeez Center" in f["retailer"]["value"]

    # 7. Purchase Amount
    assert f["purchase_amount"]["value"] == 385000.0


@pytest.mark.ocr
def test_mock_partial_receipt_triggers_manual_verification(mock_partial_ocr_text: str):
    """Verify that receipts missing key fields (like serial or invoice number) trigger verification flags."""
    data = parse_receipt_text(mock_partial_ocr_text)
    f = data["fields"]

    assert f["purchase_date"]["value"] == "2025-08-10"
    assert f["purchase_amount"]["value"] == 95000.0

    # Missing serial number and invoice number must flag manual verification
    assert f["serial_number"]["value"] is None
    assert f["serial_number"]["needs_verification"] is True
    assert f["invoice_number"]["value"] is None
    assert f["invoice_number"]["needs_verification"] is True


@pytest.mark.ocr
def test_date_formatting_variations():
    """Verify OCR normalizes DD/MM/YYYY, DD-Mon-YYYY, and other formats to ISO YYYY-MM-DD."""
    variations = [
        ("Invoice Date: 15/08/2025\nTotal: 1000", "2025-08-15"),
        ("Date of Purchase: 12-Nov-2025\nTotal: 5000", "2025-11-12"),
        ("Dated: 2025.10.05\nTotal: 25000", "2025-10-05"),
        ("Purchased on 04-03-2025\nTotal: 12000", "2025-03-04")
    ]
    for text, expected_iso in variations:
        res = parse_receipt_text(text)
        assert res["fields"]["purchase_date"]["value"] == expected_iso, f"Failed parsing '{text}'"


@pytest.mark.ocr
def test_smartphone_imei_extraction():
    """Verify extracting 15-digit IMEI numbers from mobile receipt text."""
    receipt = """
    AIRLINK COMMUNICATIONS OFFICIAL STORE
    Invoice: INV-AL-88219
    Date: 2025-09-20
    Device: Xiaomi Redmi Note 13 Pro 5G
    IMEI: 860492058291048
    Price: PKR 88,000.00
    """
    data = parse_receipt_text(receipt)
    assert data["fields"]["serial_number"]["value"] == "860492058291048"
    assert data["fields"]["purchase_amount"]["value"] == 88000.0


@pytest.mark.ocr
def test_cross_verify_receipt_with_claim_match_and_mismatch(mock_complete_ocr_text: str):
    """Verify comparing extracted receipt fields against claim metadata."""
    receipt_data = parse_receipt_text(mock_complete_ocr_text)

    # Case A: Perfect match
    claim_match = {
        "serial_number": "LEN-LA-2025-BT7UFBW",
        "purchase_date": "2025-11-14",
        "retailer": "Hafeez Center Electronics Megastore"
    }
    res_match = cross_verify_receipt_with_claim(receipt_data, claim_match)
    assert res_match["is_verified"] is True
    assert len(res_match["discrepancies"]) == 0

    # Case B: Fraudulent paper swapping (serial mismatch)
    claim_swapped = {
        "serial_number": "LNV-99999-DIFFERENT",
        "purchase_date": "2025-11-14",
        "retailer": "Hafeez Center Electronics Megastore"
    }
    res_swapped = cross_verify_receipt_with_claim(receipt_data, claim_swapped)
    assert res_swapped["is_verified"] is False
    assert any("Serial Number Mismatch" in d for d in res_swapped["discrepancies"])
