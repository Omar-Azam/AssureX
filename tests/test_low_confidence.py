"""
AssureX Low Confidence & Uncertainty Test Suite
================================================
Validates behavior when AI model predictions fall below confidence guardrails:
1. Tabular model confidence below min_valid_confidence threshold (< 0.70).
2. Visual GTM model confidence below min_valid_confidence threshold (< 0.70).
3. Predictions falling below uncertain_threshold (< 0.40) classified as Uncertain Result.
4. Mandatory escalation to Manual Review Required under low AI confidence.
"""

import pytest
from typing import Dict, Any
from decision_engine import final_claim_decision, classify_model_consistency


@pytest.mark.confidence
def test_low_tabular_confidence_forces_manual_review(valid_claim: Dict[str, Any]):
    """Verify that tabular confidence of 0.52 (< 0.70 threshold) escalates to Manual Review."""
    py_res = {"predicted_class": "Likely Valid", "confidence": 0.52}
    gtm_res = {"predicted_class": "Likely Valid", "confidence": 0.88}
    rule_res = {"rules_failed": [], "manual_review_required": False, "contradictions": []}

    decision = final_claim_decision(valid_claim, py_res, gtm_res, rule_res)
    assert decision["final_decision"] == "Manual Review Required"
    assert any("confidence" in f.lower() for f in decision["decision_explanation"]["factors_supporting"])


@pytest.mark.confidence
def test_low_visual_confidence_forces_manual_review(valid_claim: Dict[str, Any]):
    """Verify that visual card confidence of 0.48 (< 0.70 threshold) escalates to Manual Review."""
    py_res = {"predicted_class": "Likely Valid", "confidence": 0.92}
    gtm_res = {"predicted_class": "Likely Valid", "confidence": 0.48}
    rule_res = {"rules_failed": [], "manual_review_required": False, "contradictions": []}

    decision = final_claim_decision(valid_claim, py_res, gtm_res, rule_res)
    assert decision["final_decision"] == "Manual Review Required"


@pytest.mark.confidence
def test_uncertain_result_classification():
    """Verify predictions below 0.40 are classified as Uncertain Result."""
    status = classify_model_consistency(
        python_class="Likely Valid",
        python_conf=0.35,
        gtm_class="Likely Valid",
        gtm_conf=0.38,
        confidence_difference=0.03
    )
    assert status == "Uncertain Result"


@pytest.mark.confidence
def test_both_models_low_confidence(valid_claim: Dict[str, Any]):
    """Verify that when both models exhibit low confidence, auto-approval is blocked."""
    py_res = {"predicted_class": "Likely Valid", "confidence": 0.45}
    gtm_res = {"predicted_class": "Likely Valid", "confidence": 0.42}
    rule_res = {"rules_failed": [], "manual_review_required": False, "contradictions": []}

    decision = final_claim_decision(valid_claim, py_res, gtm_res, rule_res)
    assert decision["final_decision"] == "Manual Review Required"
    assert decision["model_consistency_status"] in ["Weak Match", "Uncertain Result"]
