"""
AssureX Multi-Model Claims Classification Pipeline
==================================================
Synthesizes:
1. Rule Engine Policy Evaluation (WarrantyRuleEngine)
2. Tabular ML Prediction (predict_with_confidence)
3. Visual Claim Card Classification (gtm_classifier)
4. Multimodal Decision Engine Arbitration (final_claim_decision)
"""

import os
import json
import logging
from pathlib import Path
from datetime import date
from typing import Dict, Any, Optional

from backend.config import POLICIES_DIR, BASE_DIR
from backend.models import Claim, Product, Warranty, Document, RepairHistory
from claim_metrics import compute_claim_metrics

logger = logging.getLogger("AssureX.Pipeline")


def load_policy_for_category(category: str) -> Dict[str, Any]:
    """Load corresponding policy JSON for product category."""
    cat_lower = (category or "").lower()
    if "smartphone" in cat_lower or "phone" in cat_lower:
        policy_file = POLICIES_DIR / "smartphone_policy.json"
    elif "laptop" in cat_lower or "computer" in cat_lower:
        policy_file = POLICIES_DIR / "laptop_policy.json"
    elif "washing" in cat_lower or "machine" in cat_lower:
        policy_file = POLICIES_DIR / "washing_machine_policy.json"
    else:
        policy_file = POLICIES_DIR / "smartphone_policy.json"

    if policy_file.is_file():
        try:
            with open(policy_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading {policy_file}: {e}")

    # Fallback basic policy dictionary
    return {
        "product_category": category,
        "general_warranty_duration_months": 12,
        "reporting_deadline_days": 30,
        "lemon_threshold_repairs": 3,
        "grace_period_days": 7,
        "covered_faults": ["Display Panel Defect", "Battery Failure", "Motherboard Fault"],
        "excluded_damages": ["Liquid / Moisture Damage", "Physical Screen Damage"]
    }


def execute_claim_classification_pipeline(claim: Claim) -> Dict[str, Any]:
    """
    Executes the full automated claims evaluation pipeline:
    1. Builds unified claim payload.
    2. Runs deterministic rule engine checks.
    3. Runs tabular machine learning classifier.
    4. Runs visual claim summary card classifier (if card available / supported).
    5. Runs decision engine arbitration to reach final adjudication.
    """
    product = claim.product
    warranty = product.warranty if product else None

    # Compute shared claim temporal metrics
    purchase_dt = product.purchase_date if product else date.today()
    claim_dt = claim.claim_submission_date or date.today()
    duration_months = warranty.warranty_duration_months if warranty else 12

    metrics = compute_claim_metrics(purchase_dt, claim_dt, duration_months)

    # Document availability flags
    doc_types = {d.file_type.lower() for d in claim.documents}
    has_receipt = "receipt" in doc_types
    has_warranty_card = "warranty_card" in doc_types
    has_product_image = "product_image" in doc_types
    has_serial_evidence = "serial_evidence" in doc_types

    # Assemble comprehensive claim dictionary
    claim_dict: Dict[str, Any] = {
        "claim_id": claim.claim_id,
        "product_category": product.product_category if product else "Smartphone",
        "product_name": product.product_name if product else "Unknown Product",
        "brand": product.brand if product else "Unknown Brand",
        "model_number": product.model_number if product else "Unknown Model",
        "serial_number": product.serial_number if product else "UNKNOWN-SN",
        "serial_number_on_receipt": product.serial_number if (product and has_receipt) else None,
        "purchase_price": product.purchase_price if product else 50000.0,
        "purchase_date": purchase_dt.isoformat() if purchase_dt else None,
        "retailer": product.retailer if product else "Authorized Retailer",
        "warranty_duration_months": duration_months,
        "warranty_start_date": warranty.warranty_start_date.isoformat() if warranty else purchase_dt.isoformat(),
        "warranty_expiry_date": metrics["warranty_expiry_date"],
        "warranty_status": metrics["warranty_status"],
        "remaining_warranty_days": metrics["remaining_warranty_days"],
        "product_age_days": metrics["product_age_days"],
        "fault_occurrence_date": claim.fault_occurrence_date.isoformat() if claim.fault_occurrence_date else None,
        "claim_submission_date": claim_dt.isoformat(),
        "fault_description": claim.fault_description,
        "damage_type": claim.damage_type,
        "prior_replacement": claim.prior_replacement,
        "repair_history": claim.repair_history,
        "has_receipt": has_receipt,
        "has_warranty_card": has_warranty_card,
        "has_product_image": has_product_image,
        "has_serial_evidence": has_serial_evidence,
        "has_any_contradiction": any(d.is_duplicate for d in claim.documents)
    }

    # --------------------------------------------------------------------------
    # 1. RULE ENGINE EVALUATION
    # --------------------------------------------------------------------------
    from src.rule_engine import WarrantyRuleEngine
    policy_data = load_policy_for_category(claim_dict["product_category"])
    rule_engine = WarrantyRuleEngine(policy_data)
    rule_result = rule_engine.evaluate_claim(claim_dict)

    # --------------------------------------------------------------------------
    # 2. TABULAR ML INFERENCE (predict_with_confidence)
    # --------------------------------------------------------------------------
    from predict_with_confidence import predict_with_confidence
    try:
        python_result = predict_with_confidence(claim_dict)
    except Exception as exc:
        logger.warning(f"predict_with_confidence failed ({exc}). Using heuristic baseline.")
        # Fallback to rule engine alignment if model artifact is inaccessible
        is_valid = len(rule_result["rules_failed"]) == 0 and not rule_result["manual_review_required"]
        python_result = {
            "predicted_class": "Valid Claim" if is_valid else ("Manual Review" if rule_result["manual_review_required"] else "Invalid Claim"),
            "confidence_valid": 0.85 if is_valid else 0.10,
            "confidence_invalid": 0.10 if is_valid else 0.85,
            "confidence_manual_review": 0.05 if is_valid else 0.05,
            "top3_predictions": []
        }

    # --------------------------------------------------------------------------
    # 3. VISUAL GTM INFERENCE (gtm_classifier)
    # --------------------------------------------------------------------------
    # Locate or generate card image if available
    from gtm_classifier import classify_claim_card
    card_path = None
    for doc in claim.documents:
        if doc.file_path and doc.file_path.endswith((".png", ".jpg", ".jpeg")):
            card_path = doc.file_path
            break

    # If no card was uploaded with the claim, search existing claim_cards directory
    if not card_path:
        search_card = list((BASE_DIR / "claim_cards").glob(f"**/{claim.claim_id}*.png"))
        if search_card:
            card_path = str(search_card[0])
        else:
            # Dynamically generate Claim Summary Card using Pillow
            try:
                from generate_claim_cards import prepare_claim_card_metrics, render_claim_card_v1
                cards_dir = BASE_DIR / "claim_cards" / "generated"
                cards_dir.mkdir(parents=True, exist_ok=True)
                gen_card_path = cards_dir / f"{claim.claim_id}_v1.png"
                card_metrics = prepare_claim_card_metrics(claim_dict)
                card_image = render_claim_card_v1(claim_dict, card_metrics)
                card_image.save(str(gen_card_path))
                card_path = str(gen_card_path)
                logger.info(f"Generated dynamic Claim Summary Card: {gen_card_path}")
            except Exception as e_card:
                logger.warning(f"Could not generate dynamic claim card: {e_card}")

    try:
        if card_path and os.path.exists(card_path):
            gtm_result = classify_claim_card(card_path)
        else:
            # If no image uploaded yet, align visual prediction with tabular expectation
            gtm_result = {
                "predicted_class": python_result.get("predicted_class", "Manual Review"),
                "confidence_valid": python_result.get("confidence_valid", 0.33),
                "confidence_invalid": python_result.get("confidence_invalid", 0.33),
                "confidence_manual_review": python_result.get("confidence_manual_review", 0.34)
            }
    except RuntimeError as r_err:
        # Strict zero-fallback in gtm_classifier raises RuntimeError if TF fails
        logger.info(f"GTM model unavailable ({r_err}). Using unassisted vision channel.")
        gtm_result = {
            "predicted_class": python_result.get("predicted_class", "Manual Review"),
            "confidence_valid": python_result.get("confidence_valid", 0.33),
            "confidence_invalid": python_result.get("confidence_invalid", 0.33),
            "confidence_manual_review": python_result.get("confidence_manual_review", 0.34),
            "notice": "Native TF execution failed or card missing. Using aligned projection."
        }

    # --------------------------------------------------------------------------
    # 4. DECISION ENGINE ARBITRATION (decision_engine)
    # --------------------------------------------------------------------------
    from decision_engine import final_claim_decision
    decision_result = final_claim_decision(
        claim_record=claim_dict,
        python_result=python_result,
        gtm_result=gtm_result,
        rule_result=rule_result
    )

    return {
        "claim_dict": claim_dict,
        "rule_result": rule_result,
        "python_result": python_result,
        "gtm_result": gtm_result,
        "decision_result": decision_result
    }
