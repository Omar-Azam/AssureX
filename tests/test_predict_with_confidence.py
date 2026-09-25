"""
Unit Tests for AssureX predict_with_confidence Inference Function
=================================================================
Validates:
1. Output dictionary schema:
   {predicted_class, confidence_valid, confidence_invalid, confidence_manual_review, top3_predictions}
2. Normalization of probability values (sum == 1.0)
3. Confidence range constraints (0.0 <= p <= 1.0)
4. Sort order of top3_predictions (strictly descending by confidence)
5. Robustness against missing fields, None serials, and unformatted dates
6. Correct multi-class predictions across realistic claim records
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datetime import date, datetime
from predict_with_confidence import predict_with_confidence, load_artifacts, transform_raw_claim_to_features


SAMPLE_VALID_CLAIM = {
    "claim_id": "CLM-TEST-VAL-001",
    "product_category": "Smartphone",
    "product_name": "Samsung Galaxy S24 Ultra",
    "brand": "Samsung",
    "model_number": "SM-S928B",
    "serial_number": "35874247858703",
    "purchase_date": "2025-10-25",
    "purchase_price": 245500.0,
    "retailer": "Daraz Mall Authorized Brand Store",
    "warranty_duration_months": 12,
    "warranty_start_date": "2025-10-26",
    "warranty_expiry_date": "2026-10-26",
    "fault_occurrence_date": "2026-04-01",
    "fault_description": "Spontaneous vertical green line on AMOLED panel post official OTA update",
    "damage_type": "Display Panel Defect",
    "claim_submission_date": "2026-04-04",
    "repair_history": "0 repairs",
    "has_receipt": True,
    "has_warranty_card": True,
    "has_product_image": True,
    "has_serial_evidence": True,
    "serial_number_on_receipt": "35874247858703",
    "prior_replacement": False
}

SAMPLE_INVALID_CLAIM = {
    "claim_id": "CLM-TEST-INV-002",
    "product_category": "Laptop",
    "product_name": "Lenovo Legion Pro 5i",
    "brand": "Lenovo",
    "model_number": "82WK0046US",
    "serial_number": "LEN-LA-2025-BT7UFBW",
    "purchase_date": "2024-01-05",
    "purchase_price": 417500.0,
    "retailer": "Airlink Communications Official Flagship",
    "warranty_duration_months": 12,
    "warranty_start_date": "2024-01-06",
    "warranty_expiry_date": "2025-01-06",
    "fault_occurrence_date": "2025-08-03",
    "fault_description": "Tea spill across keyboard assembly; red liquid contact indicators observed on motherboard",
    "damage_type": "Liquid / Moisture Damage",
    "claim_submission_date": "2025-08-10",
    "repair_history": "1 repair (Unauthorized Third-Party)",
    "has_receipt": False,
    "has_warranty_card": True,
    "has_product_image": True,
    "has_serial_evidence": True,
    "serial_number_on_receipt": None,
    "prior_replacement": False
}

SAMPLE_MANUAL_REVIEW_CLAIM = {
    "claim_id": "CLM-TEST-REV-003",
    "product_category": "Washing Machine",
    "product_name": "Samsung Wobble Technology Top-Load 13kg",
    "brand": "Samsung",
    "model_number": "WA13CG5441BY",
    "serial_number": "SAM-WA-2025-31YKCWB",
    "purchase_date": "2025-12-25",
    "purchase_price": 99500.0,
    "retailer": "TechnoCity Prime Electronics",
    "warranty_duration_months": 12,
    "warranty_start_date": "2025-12-26",
    "warranty_expiry_date": "2026-12-26",
    "fault_occurrence_date": "2026-09-03",
    "fault_description": "Water inlet dual solenoid valve coil burnout; machine fails to fill water",
    "damage_type": "Water Valve Failure",
    "claim_submission_date": "2026-09-06",
    "repair_history": "3 repairs (Authorized Service Center)",
    "has_receipt": True,
    "has_warranty_card": False,
    "has_product_image": True,
    "has_serial_evidence": False,
    "serial_number_on_receipt": "SAM-WA-2025-31YKCWB",
    "prior_replacement": False
}


def test_return_schema():
    """Verify exact dictionary return keys and types."""
    result = predict_with_confidence(SAMPLE_VALID_CLAIM)
    
    assert isinstance(result, dict), "Result must be a dictionary"
    
    required_keys = {
        "predicted_class",
        "confidence_valid",
        "confidence_invalid",
        "confidence_manual_review",
        "top3_predictions"
    }
    assert required_keys == set(result.keys()), f"Keys mismatch: {result.keys()}"
    
    assert result["predicted_class"] in {"Valid Claim", "Invalid Claim", "Manual Review"}
    assert isinstance(result["confidence_valid"], float)
    assert isinstance(result["confidence_invalid"], float)
    assert isinstance(result["confidence_manual_review"], float)
    assert isinstance(result["top3_predictions"], list)
    assert len(result["top3_predictions"]) == 3


def test_confidence_normalization():
    """Verify confidences sum to 1.0 within floating point rounding."""
    for claim in [SAMPLE_VALID_CLAIM, SAMPLE_INVALID_CLAIM, SAMPLE_MANUAL_REVIEW_CLAIM]:
        res = predict_with_confidence(claim)
        
        # Individual bounds [0.0, 1.0]
        assert 0.0 <= res["confidence_valid"] <= 1.0
        assert 0.0 <= res["confidence_invalid"] <= 1.0
        assert 0.0 <= res["confidence_manual_review"] <= 1.0
        
        # Sum approximately 1.0
        total = res["confidence_valid"] + res["confidence_invalid"] + res["confidence_manual_review"]
        assert round(total, 2) == 1.0, f"Probabilities do not sum to 1.0: {total}"


def test_top3_predictions_structure_and_sorting():
    """Verify top3_predictions contains all 3 classes sorted descending by confidence."""
    result = predict_with_confidence(SAMPLE_VALID_CLAIM)
    top3 = result["top3_predictions"]
    
    assert len(top3) == 3
    classes_in_top3 = {item["class"] for item in top3}
    assert classes_in_top3 == {"Valid Claim", "Invalid Claim", "Manual Review"}
    
    # Check descending order
    confidences = [item["confidence"] for item in top3]
    assert confidences == sorted(confidences, reverse=True), "top3_predictions is not sorted descending"
    
    # Top item must match predicted_class
    assert top3[0]["class"] == result["predicted_class"]


def test_valid_claim_prediction():
    """Verify valid claim is classified as Valid Claim with highest confidence."""
    res = predict_with_confidence(SAMPLE_VALID_CLAIM)
    assert res["predicted_class"] == "Valid Claim"
    assert res["confidence_valid"] > res["confidence_invalid"]
    assert res["confidence_valid"] > res["confidence_manual_review"]


def test_invalid_claim_prediction():
    """Verify liquid damage and expired claim is classified as Invalid Claim."""
    res = predict_with_confidence(SAMPLE_INVALID_CLAIM)
    assert res["predicted_class"] == "Invalid Claim"
    assert res["confidence_invalid"] > res["confidence_valid"]
    assert res["confidence_invalid"] > res["confidence_manual_review"]


def test_manual_review_prediction():
    """Verify lemon threshold / multiple repairs claim is flagged as Manual Review."""
    res = predict_with_confidence(SAMPLE_MANUAL_REVIEW_CLAIM)
    assert res["predicted_class"] == "Manual Review"
    assert res["confidence_manual_review"] > res["confidence_valid"]
    assert res["confidence_manual_review"] > res["confidence_invalid"]


def test_edge_case_missing_fields_and_types():
    """Verify robust handling of missing fields, None values, datetime objects, and empty strings."""
    sparse_claim = {
        "product_category": "Laptop",
        "damage_type": "Battery Failure",
        "purchase_date": datetime(2025, 5, 10),
        "warranty_expiry_date": date(2026, 5, 10),
        "fault_occurrence_date": "2025-11-20",
        "claim_submission_date": "2025-11-22",
        "serial_number_on_receipt": None,
        "has_receipt": False
    }
    
    res = predict_with_confidence(sparse_claim)
    assert res["predicted_class"] in {"Valid Claim", "Invalid Claim", "Manual Review"}
    assert round(res["confidence_valid"] + res["confidence_invalid"] + res["confidence_manual_review"], 2) == 1.0


def test_transform_raw_claim_to_features_dimensions():
    """Verify the feature vector generated matches the expected 22 columns."""
    _, preprocessing = load_artifacts()
    X = transform_raw_claim_to_features(SAMPLE_VALID_CLAIM, preprocessing)
    assert X.shape == (1, len(preprocessing["feature_names"]))
    assert X.shape[1] == 22


if __name__ == "__main__":
    print("Running predict_with_confidence unit tests...")
    test_return_schema()
    test_confidence_normalization()
    test_top3_predictions_structure_and_sorting()
    test_valid_claim_prediction()
    test_invalid_claim_prediction()
    test_manual_review_prediction()
    test_edge_case_missing_fields_and_types()
    test_transform_raw_claim_to_features_dimensions()
    print("[PASS] All 8 predict_with_confidence unit tests passed successfully!")
