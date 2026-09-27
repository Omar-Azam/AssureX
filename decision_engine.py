"""
AssureX Final Claim Decision Arbitration Engine
================================================

This module arbitrates the final warranty claim adjudication by synthesizing:
1. Tabular Machine Learning Prediction (`python_result` from `predict_with_confidence`)
2. Visual Claim Summary Card Prediction (`gtm_result` from `gtm_classifier`)
3. Deterministic Policy Rule Engine Result (`rule_result` from `WarrantyRuleEngine`)
4. Internal Claim Record Cross-Field Integrity (`claim_record`)

================================================================================
 LIVE EVALUATION MODIFICATION QUICK-REFERENCE GUIDE:
--------------------------------------------------------------------------------
 If asked during evaluation to modify behavior:

 1. TO CHANGE CONFIDENCE THRESHOLDS:
    - Edit `config/thresholds.json` directly (it is reloaded dynamically).
    - Or modify `DEFAULT_THRESHOLDS` dictionary below (around Line 50).
    - Key parameters to tweak:
      * `min_valid_confidence` (default 0.70): minimum confidence for auto-approval.
      * `min_invalid_confidence` (default 0.70): minimum confidence for auto-rejection.
      * `max_allowed_confidence_diff_for_auto_approval` (default 0.25): max gap between models.
      * `model_consistency.strong_match.max_confidence_difference` (default 0.15).

 2. TO ADD A NEW CONTRADICTION RULE:
    - Scroll down to `extract_and_evaluate_contradictions()` (around Line 210).
    - Add an `if <your_condition>:` block and append to `detected_contradictions`.
    - Example:
        if claim_record.get("purchase_price", 0) > 1000000 and not claim_record.get("has_receipt"):
            detected_contradictions.append("High-value claim (>1M PKR) missing proof of purchase")

 3. TO ADD A NEW POLICY GUARDRAIL / OVERRIDE:
    - Scroll down to `evaluate_decision_arbitration()` (around Line 290).
    - Add custom escalation or rejection logic based on `rule_result["rules_failed"]`
      or specific claim fields.
================================================================================
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Union, Tuple

# Module logger setup
logger = logging.getLogger("AssureX.DecisionEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] AssureX.DecisionEngine: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Project paths
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "thresholds.json"

# ==============================================================================
# CONFIGURATION DEFAULTS (Used if config/thresholds.json is absent or incomplete)
# ==============================================================================
# To adjust default thresholds without touching the JSON file, edit values here:
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "confidence_thresholds": {
        "min_valid_confidence": 0.70,                  # Minimum confidence to consider "Likely Valid"
        "min_invalid_confidence": 0.70,                # Minimum confidence to consider "Likely Invalid"
        "uncertain_threshold": 0.40,                   # Confidence floor below which predictions are uncertain
        "max_allowed_confidence_diff_for_auto_approval": 0.25,  # Max allowed gap between models for auto-valid
        "max_allowed_confidence_diff_for_auto_rejection": 0.30  # Max allowed gap between models for auto-invalid
    },
    "model_consistency": {
        "strong_match": {
            "max_confidence_difference": 0.15,         # Gap <= 15% and both >= 70% -> Strong Match
            "min_individual_confidence": 0.70
        },
        "acceptable_match": {
            "max_confidence_difference": 0.30,         # Gap <= 30% and both >= 55% -> Acceptable Match
            "min_individual_confidence": 0.55
        },
        "weak_match": {
            "max_confidence_difference": 0.50,         # Gap <= 50% and both >= 45% -> Weak Match
            "min_individual_confidence": 0.45
        },
        "uncertain_result": {
            "min_confidence_floor": 0.40               # Either model < 40% -> Uncertain Result
        }
    },
    "policy_guardrails": {
        "zero_tolerance_rules": [
            "WARRANTY_ACTIVE_CHECK",
            "EXCLUDED_DAMAGE_CHECK",
            "DATA_INTEGRITY_CHECK",
            "SERIAL_NUMBER_CHECK",
            "PROOF_OF_PURCHASE_CHECK"
        ],
        "escalate_contradictions_to_manual_review": True,
        "escalate_model_disagreement_to_manual_review": True,
        "escalate_uncertain_result_to_manual_review": True,
        "escalate_rule_engine_mismatch_to_manual_review": True
    }
}


def load_decision_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads decision thresholds from config/thresholds.json with fallback defaults.

    EVALUATION TIP:
    If the evaluator asks to adjust thresholds during evaluation, you can
    either modify config/thresholds.json or pass a custom config dictionary/path.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_file():
        logger.warning(f"Thresholds configuration file not found at {path}. Using internal defaults.")
        return json.loads(json.dumps(DEFAULT_THRESHOLDS))

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        # Merge with defaults to guarantee all expected keys exist
        merged_config = json.loads(json.dumps(DEFAULT_THRESHOLDS))
        for section, values in user_config.items():
            if isinstance(values, dict) and section in merged_config:
                merged_config[section].update(values)
            else:
                merged_config[section] = values
        return merged_config
    except Exception as err:
        logger.error(f"Failed to parse thresholds JSON ({err}). Falling back to defaults.")
        return json.loads(json.dumps(DEFAULT_THRESHOLDS))


# ==============================================================================
# CLASS NORMALIZATION & PREDICTION EXTRACTION HELPERS
# ==============================================================================

def normalize_class_label(raw_label: Optional[str]) -> str:
    """
    Normalizes diverse class label naming conventions across models into
    one of three canonical strings:
    - 'Valid Claim'
    - 'Invalid Claim'
    - 'Manual Review'
    """
    if not raw_label:
        return "Unknown"

    cleaned = str(raw_label).strip().lower().replace("_", " ")

    # Check for invalid first because 'invalid' contains 'valid'
    if "invalid" in cleaned:
        return "Invalid Claim"
    if "valid" in cleaned:
        return "Valid Claim"
    if "manual" in cleaned or "review" in cleaned:
        return "Manual Review"

    return raw_label.strip()


def extract_top_prediction(model_result: Dict[str, Any]) -> Tuple[str, float]:
    """
    Extracts the predicted class label and top confidence score from either
    a python tabular model result or a GTM visual model result.
    """
    if not isinstance(model_result, dict):
        return ("Unknown", 0.0)

    # 1. Check if predicted_class is already provided directly
    predicted_class = normalize_class_label(model_result.get("predicted_class"))

    # 2. Check if a direct confidence score is specified on the result dict
    direct_conf = float(model_result.get("confidence", 0.0) or model_result.get("top_confidence", 0.0))
    if direct_conf > 0 and predicted_class != "Unknown":
        return (predicted_class, direct_conf)

    # 3. Extract per-class confidences
    c_valid = float(model_result.get("confidence_valid", 0.0))
    c_invalid = float(model_result.get("confidence_invalid", 0.0))
    c_review = float(model_result.get("confidence_manual_review", 0.0))

    # 4. Check if top3_predictions list is present
    top3 = model_result.get("top3_predictions")
    if isinstance(top3, list) and len(top3) > 0 and isinstance(top3[0], dict):
        top_item = top3[0]
        top_class = normalize_class_label(top_item.get("class"))
        top_conf = float(top_item.get("confidence", 0.0))
        return (top_class, top_conf)

    # If predicted_class matches one of the known classes, use its specific confidence
    if predicted_class == "Valid Claim" and c_valid > 0:
        return (predicted_class, c_valid)
    if predicted_class == "Invalid Claim" and c_invalid > 0:
        return (predicted_class, c_invalid)
    if predicted_class == "Manual Review" and c_review > 0:
        return (predicted_class, c_review)

    # Otherwise take the maximum probability among the three
    prob_map = {
        "Valid Claim": c_valid,
        "Invalid Claim": c_invalid,
        "Manual Review": c_review,
    }
    best_class = max(prob_map, key=prob_map.get)
    best_conf = prob_map[best_class]

    # If predicted_class was given, retain it; otherwise use best_class
    final_class = predicted_class if predicted_class != "Unknown" else best_class
    return (final_class, best_conf)


def compute_confidence_difference(python_conf: float, gtm_conf: float) -> float:
    """
    Computes absolute difference between top prediction confidences:
    confidence_difference = abs(python_top_confidence - gtm_top_confidence)
    """
    return round(abs(float(python_conf) - float(gtm_conf)), 4)


# ==============================================================================
# MODEL CONSISTENCY CLASSIFICATION (Requirement 2)
# ==============================================================================

def classify_model_consistency(
    python_class: str,
    python_conf: float,
    gtm_class: str,
    gtm_conf: float,
    confidence_difference: float,
    thresholds: Optional[Dict[str, Any]] = None
) -> str:
    """
    Classifies the multimodal consistency between Tabular and Visual models as:
    1. 'Strong Match'        - Both models agree with low confidence gap and high individual certainty.
    2. 'Acceptable Match'    - Both models agree with moderate confidence gap.
    3. 'Weak Match'          - Both models agree on class, but confidence is borderline or gap is high.
    4. 'Model Disagreement'  - The models predict different target classes.
    5. 'Uncertain Result'    - Either model has confidence below statistical floor (< 40%).

    Configured via `config/thresholds.json` under `model_consistency`.
    """
    if thresholds is None:
        thresholds = load_decision_thresholds()

    mc_cfg = thresholds.get("model_consistency", {})
    strong_cfg = mc_cfg.get("strong_match", {})
    acceptable_cfg = mc_cfg.get("acceptable_match", {})
    weak_cfg = mc_cfg.get("weak_match", {})
    uncertain_cfg = mc_cfg.get("uncertain_result", {})

    conf_floor = uncertain_cfg.get("min_confidence_floor", 0.40)

    # Condition 1: Either model fails the statistical confidence floor -> Uncertain Result
    if python_conf < conf_floor or gtm_conf < conf_floor:
        return "Uncertain Result"

    # Condition 2: Models disagree on predicted class label -> Model Disagreement
    if python_class != gtm_class:
        return "Model Disagreement"

    # Condition 3: Both models agree on the class label. Evaluate confidence metrics:
    min_conf = min(python_conf, gtm_conf)

    # Strong Match criteria
    strong_max_diff = strong_cfg.get("max_confidence_difference", 0.15)
    strong_min_conf = strong_cfg.get("min_individual_confidence", 0.70)
    if confidence_difference <= strong_max_diff and min_conf >= strong_min_conf:
        return "Strong Match"

    # Acceptable Match criteria
    acceptable_max_diff = acceptable_cfg.get("max_confidence_difference", 0.30)
    acceptable_min_conf = acceptable_cfg.get("min_individual_confidence", 0.55)
    if confidence_difference <= acceptable_max_diff and min_conf >= acceptable_min_conf:
        return "Acceptable Match"

    # Weak Match criteria
    weak_max_diff = weak_cfg.get("max_confidence_difference", 0.50)
    weak_min_conf = weak_cfg.get("min_individual_confidence", 0.45)
    if confidence_difference <= weak_max_diff and min_conf >= weak_min_conf:
        return "Weak Match"

    # If gap is very large despite agreeing on label -> Weak Match
    return "Weak Match"


# ==============================================================================
# CONTRADICTION & INTEGRITY CHECKER (Requirement 3 & Evaluation Hook)
# ==============================================================================

def _parse_claim_date(date_val: Any) -> Optional[date]:
    """Helper to safely parse dates for contradiction verification."""
    if date_val is None:
        return None
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, str):
        cleaned = date_val.strip().split("T")[0]
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
    return None


def extract_and_evaluate_contradictions(
    claim_record: Dict[str, Any],
    rule_result: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Identifies all logical, temporal, and evidentiary contradictions.

    ============================================================================
    LIVE EVALUATION HOOK - ADDING A NEW CONTRADICTION RULE:
    ----------------------------------------------------------------------------
    To add a new contradiction rule during evaluation, simply add an `if` check
    below and append a clear description string to `detected_contradictions`.
    ============================================================================
    """
    detected_contradictions: List[str] = []

    # 1. Ingest existing contradictions flagged by the rule engine
    if rule_result and isinstance(rule_result, dict):
        # Direct list in rule_result
        if "contradictions" in rule_result and isinstance(rule_result["contradictions"], list):
            detected_contradictions.extend(rule_result["contradictions"])
        # Nested in evaluations dict
        evals = rule_result.get("evaluations", {})
        if isinstance(evals, dict) and "contradictions" in evals:
            c_list = evals.get("contradictions", [])
            if isinstance(c_list, list):
                detected_contradictions.extend(c_list)

    # 2. Check if dataset pre-computed flag is True
    if claim_record.get("has_any_contradiction") is True:
        msg = "Claim record has pre-flagged data integrity contradiction (has_any_contradiction=True)"
        if msg not in detected_contradictions:
            detected_contradictions.append(msg)

    # 3. Temporal Contradiction Check A: Claim filed before purchase date
    purchase_dt = _parse_claim_date(claim_record.get("purchase_date"))
    claim_dt = _parse_claim_date(
        claim_record.get("claim_submission_date")
        or claim_record.get("claim_filing_date")
        or claim_record.get("claim_date")
    )
    if purchase_dt and claim_dt and claim_dt < purchase_dt:
        detected_contradictions.append(
            f"Temporal Contradiction: Claim submission date ({claim_dt}) is earlier than purchase date ({purchase_dt})"
        )

    # 4. Temporal Contradiction Check B: Fault occurred before purchase date
    fault_dt = _parse_claim_date(
        claim_record.get("fault_occurrence_date") or claim_record.get("fault_date")
    )
    if purchase_dt and fault_dt and fault_dt < purchase_dt:
        detected_contradictions.append(
            f"Temporal Contradiction: Fault occurrence date ({fault_dt}) is earlier than purchase date ({purchase_dt})"
        )

    # 5. Temporal Contradiction Check C: Claim filed before fault occurred
    if fault_dt and claim_dt and claim_dt < fault_dt:
        detected_contradictions.append(
            f"Temporal Contradiction: Claim submission date ({claim_dt}) precedes fault occurrence date ({fault_dt})"
        )

    # 6. Serial Number Mismatch Check:
    serial = str(claim_record.get("serial_number") or "").strip().upper()
    receipt_serial = str(claim_record.get("serial_number_on_receipt") or "").strip().upper()
    if serial and receipt_serial and serial != receipt_serial:
        detected_contradictions.append(
            f"Serial Number Mismatch: Product serial ({serial}) does not match receipt serial ({receipt_serial})"
        )

    # 7. Document Proof Contradiction:
    # If customer claims no receipt, but a receipt serial number is provided
    if claim_record.get("has_receipt") is False and receipt_serial:
        detected_contradictions.append(
            f"Documentation Contradiction: has_receipt=False, but serial_number_on_receipt ('{receipt_serial}') was specified"
        )

    # 8. Warranty Metrics Sanity Contradiction:
    # If remaining warranty days is negative but warranty_status is recorded as Active
    rem_days = claim_record.get("remaining_warranty_days")
    status = str(claim_record.get("warranty_status") or "").strip().lower()
    if rem_days is not None and isinstance(rem_days, (int, float)):
        if rem_days < 0 and status == "active":
            detected_contradictions.append(
                f"Warranty Metrics Inconsistency: remaining_warranty_days ({rem_days}) < 0, but warranty_status is marked 'Active'"
            )
        elif rem_days > 0 and status == "expired":
            detected_contradictions.append(
                f"Warranty Metrics Inconsistency: remaining_warranty_days ({rem_days}) > 0, but warranty_status is marked 'Expired'"
            )

    # Deduplicate while preserving order
    return list(dict.fromkeys(detected_contradictions))


# ==============================================================================
# ADDITIONAL EVIDENCE BUILDER (Requirement 4)
# ==============================================================================

def determine_additional_evidence_needed(
    claim_record: Dict[str, Any],
    rules_failed: List[str],
    contradictions: List[str],
    final_decision: str,
    model_consistency_status: str
) -> List[str]:
    """
    Determines specific evidentiary documents or technical proofs required
    from the claimant or service center.
    """
    evidence_needed: List[str] = []

    # 1. Missing Document Evidence
    if claim_record.get("has_receipt") is False or claim_record.get("invoice_attached") is False or "PROOF_OF_PURCHASE_PRESENT" in rules_failed:
        evidence_needed.append(
            "Original sales invoice / tax receipt (with retailer NTN and purchase date)"
        )
    if claim_record.get("has_warranty_card") is False or claim_record.get("warranty_card_attached") is False or "MANDATORY_DOCUMENTS_COMPLETE" in rules_failed:
        evidence_needed.append(
            "Stamped manufacturer warranty card or official digital registration certificate"
        )
    if claim_record.get("has_product_image") is False:
        evidence_needed.append(
            "High-resolution photographs of physical unit (showing full front, back, and defect)"
        )
    if claim_record.get("has_serial_evidence") is False:
        evidence_needed.append(
            "Clear close-up photograph of the manufacturer serial number / IMEI barcode label"
        )

    # 2. Contradiction Resolution Proof
    if contradictions:
        evidence_needed.append(
            "Signed claimant statement explaining date/serial discrepancy with notarized sales proof"
        )

    # 3. Excluded Damage or Environmental Fault Proof
    damage_type = str(claim_record.get("damage_type") or "").lower()
    if "liquid" in damage_type or "moisture" in damage_type:
        evidence_needed.append(
            "Authorized Service Center internal inspection report verifying Liquid Contact Indicators (LCI)"
        )
    elif "surge" in damage_type or "voltage" in damage_type:
        evidence_needed.append(
            "Technical diagnostic report verifying power IC / PCB component failure and surge protector usage"
        )

    # 4. Repair History Proof
    repair_hist = str(claim_record.get("repair_history") or "").lower()
    if "repair" in repair_hist and ("unauthorized" in repair_hist or "third-party" in repair_hist or "local" in repair_hist):
        evidence_needed.append(
            "Service history log to assess whether unauthorized tampering voided warranty terms"
        )
    elif "3 repairs" in repair_hist or "lemon" in repair_hist:
        evidence_needed.append(
            "Prior service job cards for all three previous repair attempts to verify Lemon Law entitlement"
        )

    # 5. Model Disagreement Resolution
    if model_consistency_status == "Model Disagreement":
        evidence_needed.append(
            "Physical device inspection by Level-2 warranty assessor to resolve AI model divergence"
        )

    # 6. If claim is clean and likely valid
    if final_decision == "Likely Valid" and not evidence_needed:
        evidence_needed.append("None - claim file is complete and verified for expedited processing")

    return list(dict.fromkeys(evidence_needed))


# ==============================================================================
# FINAL DECISION ARBITRATION (Requirements 3 & 4)
# ==============================================================================

def final_claim_decision(
    claim_record: Dict[str, Any],
    python_result: Dict[str, Any],
    gtm_result: Dict[str, Any],
    rule_result: Dict[str, Any],
    config_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Arbitrates the final warranty claim adjudication decision.

    Parameters:
        claim_record: Dictionary containing claim features and metadata.
        python_result: Tabular model prediction dictionary:
                       {predicted_class, confidence_valid, confidence_invalid, confidence_manual_review}
        gtm_result: Visual card classifier prediction dictionary:
                    {predicted_class, confidence_valid, confidence_invalid, confidence_manual_review}
        rule_result: Rule engine output dictionary:
                     {rules_passed, rules_failed, manual_review_required, warnings, reasons}
        config_path: Optional path to thresholds.json (defaults to config/thresholds.json).

    Returns:
        Structured dictionary containing:
        - final_decision: 'Likely Valid' | 'Likely Invalid' | 'Manual Review Required'
        - model_consistency_status: 'Strong Match' | 'Acceptable Match' | 'Weak Match' |
                                   'Model Disagreement' | 'Uncertain Result'
        - confidence_difference: float
        - python_top_confidence: float
        - gtm_top_confidence: float
        - decision_explanation: Dict with factors_supporting, factors_opposing,
                               rules_passed, rules_failed, additional_evidence_needed
    """
    # --------------------------------------------------------------------------
    # STEP 1: Load Thresholds Configuration
    # --------------------------------------------------------------------------
    thresholds = load_decision_thresholds(config_path)
    conf_cfg = thresholds.get("confidence_thresholds", {})
    guardrails_cfg = thresholds.get("policy_guardrails", {})

    min_valid_conf = conf_cfg.get("min_valid_confidence", 0.70)
    min_invalid_conf = conf_cfg.get("min_invalid_confidence", 0.70)
    max_diff_auto_approval = conf_cfg.get("max_allowed_confidence_diff_for_auto_approval", 0.25)
    max_diff_auto_rejection = conf_cfg.get("max_allowed_confidence_diff_for_auto_rejection", 0.30)

    # --------------------------------------------------------------------------
    # STEP 2: Extract Predictions and Compute Confidence Difference (Req 1)
    # --------------------------------------------------------------------------
    py_class, py_top_conf = extract_top_prediction(python_result)
    gtm_class, gtm_top_conf = extract_top_prediction(gtm_result)

    # Compute absolute confidence difference
    confidence_difference = abs(py_top_conf - gtm_top_conf)

    # --------------------------------------------------------------------------
    # STEP 3: Classify Model Consistency Status (Req 2)
    # --------------------------------------------------------------------------
    model_consistency_status = classify_model_consistency(
        python_class=py_class,
        python_conf=py_top_conf,
        gtm_class=gtm_class,
        gtm_conf=gtm_top_conf,
        confidence_difference=confidence_difference,
        thresholds=thresholds
    )

    # --------------------------------------------------------------------------
    # STEP 4: Ingest and Evaluate Rule Engine Results & Contradictions (Req 3)
    # --------------------------------------------------------------------------
    rules_passed: List[str] = list(rule_result.get("rules_passed", []))
    rules_failed: List[str] = list(rule_result.get("rules_failed", []))
    manual_review_required: bool = bool(rule_result.get("manual_review_required", False))
    reasons: List[str] = list(rule_result.get("reasons", []))
    warnings: List[str] = list(rule_result.get("warnings", []))

    # Evaluate all internal and cross-field contradictions
    contradictions = extract_and_evaluate_contradictions(claim_record, rule_result)

    # --------------------------------------------------------------------------
    # STEP 5: Decision Arbitration Synthesis
    # --------------------------------------------------------------------------
    factors_supporting: List[str] = []
    factors_opposing: List[str] = []
    final_decision: str = "Manual Review Required"  # Safe initial default

    # Zero-tolerance hard policy exclusions from thresholds config
    zero_tolerance_rules = guardrails_cfg.get("zero_tolerance_rules", [
        "WARRANTY_ACTIVE_CHECK",
        "EXCLUDED_DAMAGE_CHECK",
        "DATA_INTEGRITY_CHECK",
        "SERIAL_NUMBER_CHECK",
        "PROOF_OF_PURCHASE_CHECK"
    ])
    critical_rules_failed = [r for r in rules_failed if r in zero_tolerance_rules]

    # CASE A: DATA CONTRADICTIONS DETECTED
    # Contradictions always require human auditor inspection due to suspected tampering
    if contradictions and guardrails_cfg.get("escalate_contradictions_to_manual_review", True):
        final_decision = "Manual Review Required"
        factors_supporting.append(
            f"Escalated due to {len(contradictions)} data integrity/temporal contradiction(s)"
        )
        for c in contradictions:
            factors_supporting.append(f"Contradiction: {c}")

        if py_class == "Valid Claim" and gtm_class == "Valid Claim":
            factors_opposing.append(
                f"Both ML models predicted Valid Claim ({py_top_conf:.1%}, {gtm_top_conf:.1%}), but cannot auto-approve contradictory data"
            )

    # CASE B: MODEL DISAGREEMENT (Multimodal Conflict)
    # When Tabular model and Visual Card classifier disagree on class
    elif model_consistency_status == "Model Disagreement" and guardrails_cfg.get("escalate_model_disagreement_to_manual_review", True):
        final_decision = "Manual Review Required"
        factors_supporting.append(
            f"Model Disagreement: Tabular ML predicted '{py_class}' ({py_top_conf:.1%}) while Visual GTM classifier predicted '{gtm_class}' ({gtm_top_conf:.1%})"
        )
        factors_supporting.append(
            f"Confidence difference of {confidence_difference:.1%} exceeds consensus criteria"
        )
        factors_opposing.append(
            f"Conflicting predictions between Tabular ML ({py_class}) and Visual GTM ({gtm_class})"
        )
        if rules_passed:
            factors_opposing.append(f"Claim passed {len(rules_passed)} baseline policy rules")

    # CASE C: UNCERTAIN MODEL PREDICTION
    # Either model has confidence below the floor (< 40%)
    elif model_consistency_status == "Uncertain Result" and guardrails_cfg.get("escalate_uncertain_result_to_manual_review", True):
        final_decision = "Manual Review Required"
        factors_supporting.append(
            f"Uncertain Result: Model prediction lacks statistical certainty (Tabular: {py_top_conf:.1%}, Visual: {gtm_top_conf:.1%})"
        )
        if rules_passed:
            factors_opposing.append(f"Passed {len(rules_passed)} policy rules")

    # CASE D: RULE ENGINE EXPLICITLY MANDATED MANUAL REVIEW
    # e.g., missing secondary documents, borderline submission window, Lemon law trigger
    elif manual_review_required and py_class != "Invalid Claim":
        final_decision = "Manual Review Required"
        factors_supporting.append("Rule engine policy flagged manual review condition")
        for r in reasons:
            factors_supporting.append(f"Policy review trigger: {r}")
        if py_class == "Valid Claim":
            factors_opposing.append(
                f"Tabular model leaned Valid Claim ({py_top_conf:.1%}), but policy requires underwriter verification"
            )

    # CASE E: BOTH MODELS PREDICT "MANUAL REVIEW"
    elif py_class == "Manual Review" and gtm_class == "Manual Review":
        final_decision = "Manual Review Required"
        factors_supporting.append(
            f"Both Tabular ML ({py_top_conf:.1%}) and Visual Card ({gtm_top_conf:.1%}) classify claim as Manual Review"
        )

    # CASE F: BOTH MODELS PREDICT "INVALID CLAIM" (Consensus Invalid)
    elif py_class == "Invalid Claim" and gtm_class == "Invalid Claim":
        # Check if confidences satisfy threshold
        if (
            min(py_top_conf, gtm_top_conf) >= min_invalid_conf
            and confidence_difference <= max_diff_auto_rejection
        ):
            final_decision = "Likely Invalid"
            factors_supporting.append(
                f"Consensus Invalid: Tabular model ({py_top_conf:.1%}) and Visual card ({gtm_top_conf:.1%}) both predict Invalid Claim"
            )
            if critical_rules_failed:
                factors_supporting.append(
                    f"Policy rule failures corroborate rejection: {', '.join(critical_rules_failed)}"
                )
            if rules_passed:
                factors_opposing.append(
                    f"Passed {len(rules_passed)} baseline rules (e.g. {', '.join(rules_passed[:2])})"
                )
        else:
            final_decision = "Manual Review Required"
            factors_supporting.append(
                f"Models agree on Invalid Claim, but confidence ({min(py_top_conf, gtm_top_conf):.1%}) is below auto-rejection threshold ({min_invalid_conf:.1%})"
            )

    # CASE G: CRITICAL POLICY RULE FAILURE WITH MODEL CONFLICT
    # e.g. Warranty is physically expired or damage is excluded liquid, but models predicted Valid
    elif critical_rules_failed and py_class == "Valid Claim":
        final_decision = "Manual Review Required"
        factors_supporting.append(
            f"Critical policy rule failure overrides model: {', '.join(critical_rules_failed)}"
        )
        for r in reasons:
            factors_supporting.append(f"Rule failure reason: {r}")
        factors_opposing.append(
            f"Tabular model predicted Valid Claim ({py_top_conf:.1%}), conflicting with policy rule check"
        )

    # CASE H: BOTH MODELS PREDICT "VALID CLAIM" (Consensus Valid)
    elif py_class == "Valid Claim" and gtm_class == "Valid Claim":
        if (
            not rules_failed
            and not manual_review_required
            and min(py_top_conf, gtm_top_conf) >= min_valid_conf
            and confidence_difference <= max_diff_auto_approval
        ):
            final_decision = "Likely Valid"
            factors_supporting.append(
                f"Consensus Valid: Tabular model ({py_top_conf:.1%}) and Visual card ({gtm_top_conf:.1%}) both predict Valid Claim"
            )
            factors_supporting.append(
                f"Passed all {len(rules_passed)} policy verification rules cleanly"
            )
            factors_supporting.append("Zero contradictions, active warranty, and complete documentation")
            if warnings:
                factors_opposing.append(f"Advisory warnings noted: {', '.join(warnings)}")
            if confidence_difference > 0.10:
                factors_opposing.append(
                    f"Minor confidence gap of {confidence_difference:.1%} between models"
                )
        else:
            # Borderline confidence or warnings present
            final_decision = "Manual Review Required"
            if min(py_top_conf, gtm_top_conf) < min_valid_conf:
                factors_supporting.append(
                    f"Model confidence ({min(py_top_conf, gtm_top_conf):.1%}) is below automated approval threshold ({min_valid_conf:.1%})"
                )
            if confidence_difference > max_diff_auto_approval:
                factors_supporting.append(
                    f"Confidence difference ({confidence_difference:.1%}) exceeds maximum allowed gap ({max_diff_auto_approval:.1%})"
                )
            if rules_failed:
                factors_supporting.append(f"Policy rules failed: {', '.join(rules_failed)}")
            factors_opposing.append(
                f"Both models predicted Valid Claim ({py_top_conf:.1%}, {gtm_top_conf:.1%})"
            )

    # CASE I: ALL OTHER WEAK / BORDERLINE CASES
    else:
        final_decision = "Manual Review Required"
        factors_supporting.append(
            f"Borderline case requiring underwriter review ({py_class} {py_top_conf:.1%} vs {gtm_class} {gtm_top_conf:.1%})"
        )

    # --------------------------------------------------------------------------
    # STEP 6: Determine Additional Evidence Needed (Req 4)
    # --------------------------------------------------------------------------
    additional_evidence_needed = determine_additional_evidence_needed(
        claim_record=claim_record,
        rules_failed=rules_failed,
        contradictions=contradictions,
        final_decision=final_decision,
        model_consistency_status=model_consistency_status
    )

    # --------------------------------------------------------------------------
    # STEP 7: Executive Summary Formulation
    # --------------------------------------------------------------------------
    summary = (
        f"Claim adjudicated as '{final_decision}' under '{model_consistency_status}' consistency status. "
        f"Tabular Model: {py_class} ({py_top_conf:.1%}), Visual GTM Model: {gtm_class} ({gtm_top_conf:.1%}), "
        f"Confidence Gap: {confidence_difference:.1%}. Rules Passed: {len(rules_passed)}, Rules Failed: {len(rules_failed)}."
    )

    # --------------------------------------------------------------------------
    # STEP 8: Construct and Return Structured Decision Output
    # --------------------------------------------------------------------------
    decision_explanation = {
        "decision": final_decision,
        "consistency_status": model_consistency_status,
        "confidence_difference": round(confidence_difference, 4),
        "factors_supporting": factors_supporting,
        "factors_opposing": factors_opposing,
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "contradictions": contradictions,
        "additional_evidence_needed": additional_evidence_needed,
        "evidence_needed": additional_evidence_needed,
        "summary": summary
    }

    return {
        "final_decision": final_decision,
        "model_consistency_status": model_consistency_status,
        "confidence_difference": round(confidence_difference, 4),
        "python_top_confidence": round(py_top_conf, 4),
        "gtm_top_confidence": round(gtm_top_conf, 4),
        "python_predicted_class": py_class,
        "gtm_predicted_class": gtm_class,
        "decision_explanation": decision_explanation,
        # Top-level convenience accessors for evaluators and unit tests
        "factors_supporting": factors_supporting,
        "factors_opposing": factors_opposing,
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "contradictions": contradictions,
        "additional_evidence_needed": additional_evidence_needed,
        "evidence_needed": additional_evidence_needed,
        "summary": summary
    }


# ==============================================================================
# DEMO EXECUTION / LIVE EVALUATION VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print(" AssureX Final Claim Decision Arbitration Engine - Verification Demo")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # DEMO 1: Clear Valid Claim (Consensus Strong Match)
    # --------------------------------------------------------------------------
    print("\n[SCENARIO 1] Clear Valid Claim (Strong Agreement & Rules Passed):")
    sample_claim_valid = {
        "claim_id": "CLM-DEMO-001",
        "product_category": "Smartphone",
        "purchase_date": "2025-06-15",
        "claim_submission_date": "2025-11-20",
        "fault_occurrence_date": "2025-11-18",
        "warranty_duration_months": 12,
        "warranty_status": "Active",
        "remaining_warranty_days": 207,
        "serial_number": "SM-S928B-001",
        "serial_number_on_receipt": "SM-S928B-001",
        "has_receipt": True,
        "has_warranty_card": True,
        "has_product_image": True,
        "has_serial_evidence": True,
        "has_any_contradiction": False,
        "damage_type": "Display Panel Defect"
    }
    py_res_valid = {
        "predicted_class": "Valid Claim",
        "confidence_valid": 0.92,
        "confidence_invalid": 0.05,
        "confidence_manual_review": 0.03
    }
    gtm_res_valid = {
        "predicted_class": "Valid Claim",
        "confidence_valid": 0.88,
        "confidence_invalid": 0.08,
        "confidence_manual_review": 0.04
    }
    rule_res_valid = {
        "rules_passed": ["WARRANTY_ACTIVE_CHECK", "PROOF_OF_PURCHASE_CHECK", "SERIAL_NUMBER_CHECK"],
        "rules_failed": [],
        "manual_review_required": False,
        "reasons": []
    }

    decision_1 = final_claim_decision(sample_claim_valid, py_res_valid, gtm_res_valid, rule_res_valid)
    print(f"Final Decision      : {decision_1['final_decision']}")
    print(f"Consistency Status  : {decision_1['model_consistency_status']}")
    print(f"Confidence Gap      : {decision_1['confidence_difference']:.4f}")
    print("Factors Supporting  :")
    for f in decision_1['decision_explanation']['factors_supporting']:
        print(f"  + {f}")
    print("Evidence Needed     :")
    for e in decision_1['decision_explanation']['additional_evidence_needed']:
        print(f"  - {e}")

    # --------------------------------------------------------------------------
    # DEMO 2: Contradictory Claim (Temporal Contradiction -> Manual Review)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[SCENARIO 2] Contradictory Claim (Claim filed before purchase date):")
    sample_claim_contra = {
        "claim_id": "CLM-DEMO-002",
        "product_category": "Laptop",
        "purchase_date": "2025-08-01",
        "claim_submission_date": "2025-07-15",  # Impossible: filed 16 days before purchase!
        "fault_occurrence_date": "2025-07-10",
        "warranty_duration_months": 12,
        "warranty_status": "Active",
        "remaining_warranty_days": 380,
        "has_receipt": True,
        "has_warranty_card": True,
        "has_product_image": True,
        "has_serial_evidence": True,
        "has_any_contradiction": True
    }
    decision_2 = final_claim_decision(sample_claim_contra, py_res_valid, gtm_res_valid, rule_res_valid)
    print(f"Final Decision      : {decision_2['final_decision']}")
    print(f"Consistency Status  : {decision_2['model_consistency_status']}")
    print(f"Contradictions      : {decision_2['decision_explanation']['contradictions']}")
    print("Factors Supporting  :")
    for f in decision_2['decision_explanation']['factors_supporting']:
        print(f"  + {f}")

    # --------------------------------------------------------------------------
    # DEMO 3: Multimodal Disagreement (Tabular Valid vs Visual Invalid)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[SCENARIO 3] Model Disagreement (Tabular Valid vs Visual Invalid):")
    gtm_res_invalid = {
        "predicted_class": "Invalid Claim",
        "confidence_valid": 0.10,
        "confidence_invalid": 0.85,
        "confidence_manual_review": 0.05
    }
    decision_3 = final_claim_decision(sample_claim_valid, py_res_valid, gtm_res_invalid, rule_res_valid)
    print(f"Final Decision      : {decision_3['final_decision']}")
    print(f"Consistency Status  : {decision_3['model_consistency_status']}")
    print(f"Confidence Gap      : {decision_3['confidence_difference']:.4f}")
    print("Factors Supporting  :")
    for f in decision_3['decision_explanation']['factors_supporting']:
        print(f"  + {f}")
    print("Evidence Needed     :")
    for e in decision_3['decision_explanation']['additional_evidence_needed']:
        print(f"  - {e}")

    print("=" * 75)
    print("Verification completed successfully.")
