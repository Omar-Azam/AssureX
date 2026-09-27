"""
AssureX Multimodal Model Disagreement Test Suite
=================================================
Validates conflict resolution when Tabular ML and Visual GTM models produce opposing predictions:
1. Disagreement classification: Python (Likely Valid) vs GTM (Likely Invalid).
2. Disagreement classification: Python (Likely Invalid) vs GTM (Likely Valid).
3. Confidence gap calculation between divergent predictions.
4. Mandatory escalation to underwriter queue under Model Disagreement.
5. Verification of factors supporting and opposing in decision rationale.
"""

import pytest
from typing import Dict, Any
from decision_engine import (
    final_claim_decision,
    classify_model_consistency,
    compute_confidence_difference
)


@pytest.mark.confidence
def test_model_disagreement_classification_valid_vs_invalid():
    """Verify opposing predictions with high confidence are classified as Model Disagreement."""
    status = classify_model_consistency(
        python_class="Likely Valid",
        python_conf=0.94,
        gtm_class="Likely Invalid",
        gtm_conf=0.89,
        confidence_difference=0.05
    )
    assert status == "Model Disagreement"


@pytest.mark.confidence
def test_model_disagreement_classification_invalid_vs_valid():
    """Verify opposite direction (Invalid vs Valid) is also Model Disagreement."""
    status = classify_model_consistency(
        python_class="Likely Invalid",
        python_conf=0.91,
        gtm_class="Likely Valid",
        gtm_conf=0.88,
        confidence_difference=0.03
    )
    assert status == "Model Disagreement"


@pytest.mark.confidence
def test_confidence_difference_calculation():
    """Verify compute_confidence_difference returns absolute difference rounded properly."""
    diff = compute_confidence_difference(0.92, 0.77)
    assert abs(diff - 0.15) < 1e-4

    diff_reversed = compute_confidence_difference(0.77, 0.92)
    assert abs(diff_reversed - 0.15) < 1e-4


@pytest.mark.confidence
def test_decision_engine_mandates_manual_review_on_disagreement(model_disagreement_claim: Dict[str, Any]):
    """Verify decision engine always sets Manual Review Required when models disagree."""
    preds = model_disagreement_claim["model_predictions"]
    py_res = preds["python_result"]
    gtm_res = preds["gtm_result"]
    rule_res = {"rules_failed": [], "manual_review_required": False, "contradictions": []}

    decision = final_claim_decision(model_disagreement_claim, py_res, gtm_res, rule_res)

    assert decision["final_decision"] == "Manual Review Required"
    assert decision["model_consistency_status"] == "Model Disagreement"
    
    explanation = decision["decision_explanation"]
    assert len(explanation["factors_supporting"]) > 0
    assert len(explanation["factors_opposing"]) > 0
    assert any("opposing" in f.lower() or "disagree" in f.lower() or "conflict" in f.lower() 
               for f in explanation["factors_opposing"] + explanation.get("evidence_needed", []))
