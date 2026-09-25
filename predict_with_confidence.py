"""
AssureX Claims Classification Inference Module
==============================================

Provides the `predict_with_confidence(claim_record)` inference function:
- Loads claim_classifier.pkl and preprocessing.pkl (with memory caching).
- Applies the exact feature extraction, derived calculations, and categorical
  encodings used during model training.
- Evaluates class probabilities.
- Returns a structured dictionary containing predicted_class, per-class
  confidences, and sorted top3_predictions.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Global cache for loaded model and preprocessing artifacts
_MODEL_CACHE = {
    "model": None,
    "preprocessing": None,
    "model_path": None,
    "preprocessing_path": None,
}


def _locate_artifact_file(filename: str, preferred_dir: str = "model") -> str:
    """Finds model/preprocessing pickle files across project paths."""
    candidates = [
        os.path.join(preferred_dir, filename),
        filename,
        os.path.join("..", preferred_dir, filename),
        os.path.join("..", filename),
        os.path.join("/content", preferred_dir, filename),
        os.path.join("/content", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate required artifact '{filename}' in {candidates}")


def load_artifacts(
    model_filename: str = "claim_classifier.pkl",
    preprocessing_filename: str = "preprocessing.pkl",
    force_reload: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads and caches the model and preprocessing artifacts using joblib.
    """
    global _MODEL_CACHE

    if (
        not force_reload
        and _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["preprocessing"] is not None
    ):
        return _MODEL_CACHE["model"], _MODEL_CACHE["preprocessing"]

    model_path = _locate_artifact_file(model_filename)
    preprocessing_path = _locate_artifact_file(preprocessing_filename)

    model = joblib.load(model_path)
    preprocessing = joblib.load(preprocessing_path)

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["preprocessing"] = preprocessing
    _MODEL_CACHE["model_path"] = model_path
    _MODEL_CACHE["preprocessing_path"] = preprocessing_path

    return model, preprocessing


def _parse_date_safe(date_val: Any) -> Optional[date]:
    """Robustly parses date strings, dates, or datetimes."""
    if date_val is None or pd.isna(date_val):
        return None
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, str):
        clean_str = date_val.strip().split("T")[0]
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(clean_str, fmt).date()
            except ValueError:
                continue
    return None


def transform_raw_claim_to_features(
    claim_record: Dict[str, Any],
    preprocessing: Dict[str, Any]
) -> np.ndarray:
    """
    Applies the exact feature engineering and encoding pipeline to a single raw claim dict.
    Returns a 1D numpy array aligned with preprocessing['feature_names'].
    """
    feature_names = preprocessing["feature_names"]

    # 1. Parse dates
    p_date = _parse_date_safe(claim_record.get("purchase_date"))
    f_date = _parse_date_safe(claim_record.get("fault_occurrence_date") or claim_record.get("fault_date") or claim_record.get("incident_date"))
    e_date = _parse_date_safe(claim_record.get("warranty_expiry_date"))
    c_date = _parse_date_safe(claim_record.get("claim_submission_date") or claim_record.get("claim_date")) or date.today()

    # 2. Derived Feature: product_age_days
    if p_date and c_date:
        product_age_days = float((c_date - p_date).days)
    elif p_date and f_date:
        product_age_days = float((f_date - p_date).days)
    else:
        product_age_days = 180.0  # sensible domain default

    # 3. Derived Feature: remaining_warranty_days
    if e_date and f_date:
        remaining_warranty_days = float((e_date - f_date).days)
    elif e_date and c_date:
        remaining_warranty_days = float((e_date - c_date).days)
    else:
        remaining_warranty_days = 0.0

    # 4. Derived Feature: missing_document_count
    doc_keys = ["has_receipt", "has_warranty_card", "has_product_image", "has_serial_evidence"]
    missing_document_count = 0
    for dk in doc_keys:
        val = claim_record.get(dk)
        # Check explicit False or missing
        if val is None or not bool(val):
            missing_document_count += 1

    # 5. Derived Feature: days_to_reporting_deadline
    if f_date and c_date:
        deadline_date = f_date + timedelta(days=7)
        days_to_reporting_deadline = float((deadline_date - c_date).days)
    else:
        days_to_reporting_deadline = 0.0

    # 6. Derived Feature: repair_count
    repair_hist_str = str(claim_record.get("repair_history", "0 repairs"))
    digit_match = re.search(r"\d+", repair_hist_str)
    repair_count = int(digit_match.group(0)) if digit_match else 0

    # 7. Derived Feature: has_any_contradiction
    has_receipt = bool(claim_record.get("has_receipt", False))
    unit_sn = str(claim_record.get("serial_number", "")).strip().upper()
    rcpt_sn_raw = claim_record.get("serial_number_on_receipt")
    prior_replacement = bool(claim_record.get("prior_replacement", False))
    desc_str = str(claim_record.get("fault_description", "")).lower()
    dmg_type_str = str(claim_record.get("damage_type", "")).lower()

    # Chronology mismatch
    chronology_mismatch = False
    if p_date and f_date and f_date < p_date:
        chronology_mismatch = True
    if p_date and c_date and c_date < p_date:
        chronology_mismatch = True
    if f_date and c_date and c_date < f_date:
        chronology_mismatch = True

    # Serial mismatch when receipt is present
    serial_mismatch = False
    if has_receipt:
        if rcpt_sn_raw is None or pd.isna(rcpt_sn_raw):
            serial_mismatch = True
        else:
            rcpt_sn = str(rcpt_sn_raw).strip().upper()
            if rcpt_sn == "" or rcpt_sn == "NAN" or rcpt_sn != unit_sn:
                serial_mismatch = True

    # Narrative contradiction
    narrative_mismatch = False
    if any(k in dmg_type_str for k in ["glitch", "unspecified", "wear"]):
        if any(w in desc_str for w in ["water", "pool", "spill", "tea", "drop", "shatter", "cracked"]):
            narrative_mismatch = True

    has_any_contradiction = int(
        chronology_mismatch or serial_mismatch or prior_replacement or narrative_mismatch
    )

    # 8. Categorical Encodings
    # damage_type encoding
    raw_damage_type = str(claim_record.get("damage_type", "")).strip()
    dmg_freq_map = preprocessing.get("damage_type_freq_map", {})
    dmg_risk_map = preprocessing.get("damage_type_invalid_risk_map", {})
    global_prior = preprocessing.get("global_invalid_prior", 0.3333)

    damage_type_freq = float(dmg_freq_map.get(raw_damage_type, 0.0))
    damage_type_invalid_risk = float(dmg_risk_map.get(raw_damage_type, global_prior))

    # retailer encoding
    raw_retailer = claim_record.get("retailer")
    ret_freq_map = preprocessing.get("retailer_freq_map", {})
    if raw_retailer is None or pd.isna(raw_retailer) or str(raw_retailer).strip() == "":
        retailer_is_missing = 1
        retailer_freq = 0.0
    else:
        retailer_is_missing = 0
        retailer_freq = float(ret_freq_map.get(str(raw_retailer).strip(), 0.0))

    # product_category one-hot encoding
    raw_category = str(claim_record.get("product_category", "")).strip()

    # 9. Base financial / policy metrics
    purchase_price = float(claim_record.get("purchase_price", 0.0) or 0.0)
    warranty_duration_months = int(claim_record.get("warranty_duration_months", 12) or 12)

    # Construct feature dictionary
    features_dict = {
        "purchase_price": purchase_price,
        "warranty_duration_months": float(warranty_duration_months),
        "product_age_days": product_age_days,
        "remaining_warranty_days": remaining_warranty_days,
        "missing_document_count": float(missing_document_count),
        "days_to_reporting_deadline": days_to_reporting_deadline,
        "repair_count": float(repair_count),
        "has_receipt": 1.0 if has_receipt else 0.0,
        "has_warranty_card": 1.0 if bool(claim_record.get("has_warranty_card", False)) else 0.0,
        "has_product_image": 1.0 if bool(claim_record.get("has_product_image", False)) else 0.0,
        "has_serial_evidence": 1.0 if bool(claim_record.get("has_serial_evidence", False)) else 0.0,
        "prior_replacement": 1.0 if prior_replacement else 0.0,
        "has_any_contradiction": float(has_any_contradiction),
        "damage_type_freq": damage_type_freq,
        "damage_type_invalid_risk": damage_type_invalid_risk,
        "retailer_freq": retailer_freq,
        "retailer_is_missing": float(retailer_is_missing),
        "category_Audio_Soundbar": 1.0 if raw_category == "Audio / Soundbar" else 0.0,
        "category_Laptop": 1.0 if raw_category == "Laptop" else 0.0,
        "category_Smart_TV": 1.0 if raw_category == "Smart TV" else 0.0,
        "category_Smartphone": 1.0 if raw_category == "Smartphone" else 0.0,
        "category_Washing_Machine": 1.0 if raw_category == "Washing Machine" else 0.0,
    }

    # Vectorize strictly aligned with feature_names
    vector = [features_dict.get(fn, 0.0) for fn in feature_names]
    return np.array(vector, dtype=float).reshape(1, -1)


def predict_with_confidence(claim_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a raw claim record against the trained AssureX claims classifier.

    Parameters:
    -----------
    claim_record : Dict[str, Any]
        Dictionary of raw claim attributes (e.g. product_category, serial_number,
        purchase_date, warranty dates, fault description, damage type, document flags).

    Returns:
    --------
    Dict[str, Any]
        {
            "predicted_class": str,            # "Valid Claim" | "Invalid Claim" | "Manual Review"
            "confidence_valid": float,         # Probability [0.0 - 1.0]
            "confidence_invalid": float,       # Probability [0.0 - 1.0]
            "confidence_manual_review": float, # Probability [0.0 - 1.0]
            "top3_predictions": [              # Sorted descending by confidence
                {"class": "...", "confidence": ...},
                ...
            ]
        }
    """
    # 1. Load cached model and preprocessing bundle
    model, preprocessing = load_artifacts()

    # 2. Transform raw claim record into aligned feature matrix
    X_features = transform_raw_claim_to_features(claim_record, preprocessing)
    feature_names = preprocessing.get("feature_names", [])
    X_df = pd.DataFrame(X_features, columns=feature_names) if feature_names else pd.DataFrame(X_features)

    # 3. Predict probabilities
    # Winning models (Random Forest or SVM Pipeline) were trained directly on unscaled X_train
    # (Pipelines handle internal scaling automatically, and tree models require raw features).
    # Standalone unscaled models or fallbacks are supported gracefully.
    try:
        probabilities = model.predict_proba(X_df)[0]
    except Exception:
        scaler = preprocessing.get("scaler")
        if scaler is not None and hasattr(scaler, "transform"):
            try:
                X_scaled = scaler.transform(X_df)
                probabilities = model.predict_proba(X_scaled)[0]
            except Exception:
                probabilities = model.predict_proba(X_features)[0]
        else:
            probabilities = model.predict_proba(X_features)[0]

    # Map probabilities to classes
    classes = getattr(model, "classes_", np.array(['Valid Claim', 'Invalid Claim', 'Manual Review']))
    # Handle label encoder mapping if classes are integer indices
    label_encoder = preprocessing.get("label_encoder")
    if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
        if isinstance(classes[0], (int, np.integer)):
            classes = label_encoder.inverse_transform(classes)

    class_prob_map = {}
    for cls_name, prob in zip(classes, probabilities):
        class_prob_map[str(cls_name)] = float(prob)

    # Extract individual confidences
    conf_valid = class_prob_map.get("Valid Claim", 0.0)
    conf_invalid = class_prob_map.get("Invalid Claim", 0.0)
    conf_manual_review = class_prob_map.get("Manual Review", 0.0)

    # Normalize to ensure sum == 1.0
    total_prob = conf_valid + conf_invalid + conf_manual_review
    if total_prob > 0:
        conf_valid /= total_prob
        conf_invalid /= total_prob
        conf_manual_review /= total_prob

    # Top prediction
    prob_dict = {
        "Valid Claim": conf_valid,
        "Invalid Claim": conf_invalid,
        "Manual Review": conf_manual_review,
    }
    predicted_class = max(prob_dict, key=prob_dict.get)

    # Top-3 predictions sorted descending
    top3_predictions = [
        {"class": k, "confidence": round(v, 4)}
        for k, v in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "predicted_class": predicted_class,
        "confidence_valid": round(conf_valid, 4),
        "confidence_invalid": round(conf_invalid, 4),
        "confidence_manual_review": round(conf_manual_review, 4),
        "top3_predictions": top3_predictions,
    }


if __name__ == "__main__":
    import json

    print("Running AssureX predict_with_confidence demo...")

    # Sample 1: Clear Valid Claim
    valid_claim = {
        "claim_id": "CLM-DEMO-VALID-01",
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

    # Sample 2: Clear Invalid Claim (Liquid Damage & Expired)
    invalid_claim = {
        "claim_id": "CLM-DEMO-INVALID-02",
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

    # Sample 3: Borderline Manual Review (Missing 2 Docs, Lemon threshold)
    manual_review_claim = {
        "claim_id": "CLM-DEMO-REVIEW-03",
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
        "fault_description": "Water inlet dual solenoid valve coil burnout; machine fails to fill water (Customer notes prior minor servicing at local shop)",
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

    print("\n--- TEST 1: Expected Valid Claim ---")
    res1 = predict_with_confidence(valid_claim)
    print(json.dumps(res1, indent=2))

    print("\n--- TEST 2: Expected Invalid Claim ---")
    res2 = predict_with_confidence(invalid_claim)
    print(json.dumps(res2, indent=2))

    print("\n--- TEST 3: Expected Manual Review ---")
    res3 = predict_with_confidence(manual_review_claim)
    print(json.dumps(res3, indent=2))
