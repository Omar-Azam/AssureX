"""
Unit Tests for AssureX Final Claim Decision Arbitration Engine
==============================================================

Validates:
1. Confidence difference calculation: abs(python_top_confidence - gtm_top_confidence)
2. All 5 model_consistency_status classifications:
   - Strong Match
   - Acceptable Match
   - Weak Match
   - Model Disagreement
   - Uncertain Result
3. Dynamic threshold loading from config/thresholds.json
4. Contradiction detection (temporal, serial number, documentation, pre-computed flags)
5. Decision arbitration outcomes:
   - Likely Valid (unanimous agreement + passed rules + high confidence)
   - Likely Invalid (consensus invalid + failed rules)
   - Manual Review Required (disagreements, contradictions, rule triggers, uncertainty)
6. Decision explanation dictionary schema:
   - factors_supporting
   - factors_opposing
   - rules_passed
   - rules_failed
   - additional_evidence_needed
   - summary
"""

import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_engine import (
    final_claim_decision,
    classify_model_consistency,
    extract_top_prediction,
    extract_and_evaluate_contradictions,
    load_decision_thresholds,
    determine_additional_evidence_needed,
    DEFAULT_THRESHOLDS
)


def test_confidence_difference_computation():
    """Verify confidence_difference = abs(python_top_confidence - gtm_top_confidence)."""
    claim = {"claim_id": "CLM-001"}
    py_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.85}
    gtm_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.70}
    rule_res = {"rules_passed": ["WARRANTY_ACTIVE_CHECK"], "rules_failed": []}

    res = final_claim_decision(claim, py_res, gtm_res, rule_res)
    expected_diff = round(abs(0.85 - 0.70), 4)
    assert res["confidence_difference"] == expected_diff
    assert res["python_top_confidence"] == 0.85
    assert res["gtm_top_confidence"] == 0.70


def test_model_consistency_all_five_categories():
    """Verify all 5 consistency status values are generated according to thresholds."""
    cfg = load_decision_thresholds()

    # 1. Strong Match (Same class, diff=0.05 <= 0.15, min_conf=0.85 >= 0.70)
    status_strong = classify_model_consistency(
        "Valid Claim", 0.90, "Valid Claim", 0.85, 0.05, cfg
    )
    assert status_strong == "Strong Match"

    # 2. Acceptable Match (Same class, diff=0.20 <= 0.30, min_conf=0.60 >= 0.55)
    status_acc = classify_model_consistency(
        "Valid Claim", 0.80, "Valid Claim", 0.60, 0.20, cfg
    )
    assert status_acc == "Acceptable Match"

    # 3. Weak Match (Same class, diff=0.35 <= 0.50, min_conf=0.48 >= 0.45)
    status_weak = classify_model_consistency(
        "Valid Claim", 0.83, "Valid Claim", 0.48, 0.35, cfg
    )
    assert status_weak == "Weak Match"

    # 4. Model Disagreement (Different classes)
    status_disagree = classify_model_consistency(
        "Valid Claim", 0.85, "Invalid Claim", 0.80, 0.05, cfg
    )
    assert status_disagree == "Model Disagreement"

    # 5. Uncertain Result (Either confidence < 0.40)
    status_uncertain = classify_model_consistency(
        "Valid Claim", 0.35, "Valid Claim", 0.38, 0.03, cfg
    )
    assert status_uncertain == "Uncertain Result"


def test_thresholds_json_loading():
    """Verify config/thresholds.json is loaded and properly structured."""
    cfg = load_decision_thresholds()
    assert isinstance(cfg, dict)
    assert "confidence_thresholds" in cfg
    assert "model_consistency" in cfg
    assert "policy_guardrails" in cfg
    assert cfg["confidence_thresholds"]["min_valid_confidence"] == 0.70


def test_contradiction_detection():
    """Verify temporal, serial, and dataset contradiction detection."""
    # Scenario A: Submission date before purchase date
    bad_temporal_claim = {
        "purchase_date": "2025-06-01",
        "claim_submission_date": "2025-05-15",
        "serial_number": "SN-100",
        "serial_number_on_receipt": "SN-100"
    }
    c_list = extract_and_evaluate_contradictions(bad_temporal_claim)
    assert len(c_list) > 0
    assert any("earlier than purchase date" in c for c in c_list)

    # Scenario B: Serial number mismatch
    bad_serial_claim = {
        "purchase_date": "2025-01-01",
        "claim_submission_date": "2025-03-01",
        "serial_number": "SN-ALPHA-99",
        "serial_number_on_receipt": "SN-BETA-88"
    }
    c_list2 = extract_and_evaluate_contradictions(bad_serial_claim)
    assert any("Serial Number Mismatch" in c for c in c_list2)


def test_decision_likely_valid_when_unanimous():
    """Verify clean consensus claim yields Likely Valid."""
    claim = {
        "claim_id": "CLM-VALID-001",
        "purchase_date": "2025-05-01",
        "claim_submission_date": "2025-08-01",
        "fault_occurrence_date": "2025-07-28",
        "has_receipt": True,
        "has_warranty_card": True,
        "has_product_image": True,
        "has_serial_evidence": True,
        "has_any_contradiction": False
    }
    py_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.88, "confidence_invalid": 0.06, "confidence_manual_review": 0.06}
    gtm_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.82, "confidence_invalid": 0.10, "confidence_manual_review": 0.08}
    rule_res = {
        "rules_passed": ["WARRANTY_ACTIVE_CHECK", "PROOF_OF_PURCHASE_CHECK"],
        "rules_failed": [],
        "manual_review_required": False
    }

    res = final_claim_decision(claim, py_res, gtm_res, rule_res)
    assert res["final_decision"] == "Likely Valid"
    assert res["model_consistency_status"] == "Strong Match"
    assert "None - claim file is complete and verified for expedited processing" in res["additional_evidence_needed"]


def test_decision_likely_invalid_when_consensus():
    """Verify consensus invalid claim yields Likely Invalid."""
    claim = {
        "claim_id": "CLM-INVALID-001",
        "purchase_date": "2024-01-01",
        "claim_submission_date": "2025-08-01",
        "damage_type": "Liquid / Moisture Damage",
        "has_receipt": True,
        "has_warranty_card": True,
        "has_product_image": True,
        "has_serial_evidence": True,
        "has_any_contradiction": False
    }
    py_res = {"predicted_class": "Invalid Claim", "confidence_valid": 0.05, "confidence_invalid": 0.90, "confidence_manual_review": 0.05}
    gtm_res = {"predicted_class": "Invalid Claim", "confidence_valid": 0.08, "confidence_invalid": 0.86, "confidence_manual_review": 0.06}
    rule_res = {
        "rules_passed": [],
        "rules_failed": ["WARRANTY_ACTIVE_CHECK", "EXCLUDED_DAMAGE_CHECK"],
        "manual_review_required": False
    }

    res = final_claim_decision(claim, py_res, gtm_res, rule_res)
    assert res["final_decision"] == "Likely Invalid"
    assert res["model_consistency_status"] == "Strong Match"


def test_decision_manual_review_on_model_disagreement():
    """Verify differing model predictions triggers Manual Review Required."""
    claim = {"claim_id": "CLM-DISAGREE-001", "has_any_contradiction": False}
    py_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.85}
    gtm_res = {"predicted_class": "Invalid Claim", "confidence_invalid": 0.80}
    rule_res = {"rules_passed": ["WARRANTY_ACTIVE_CHECK"], "rules_failed": []}

    res = final_claim_decision(claim, py_res, gtm_res, rule_res)
    assert res["final_decision"] == "Manual Review Required"
    assert res["model_consistency_status"] == "Model Disagreement"


def test_decision_manual_review_on_contradiction():
    """Verify contradictions force Manual Review Required even if models predict Valid."""
    claim = {
        "claim_id": "CLM-CONTRA-001",
        "purchase_date": "2025-10-01",
        "claim_submission_date": "2025-09-01",  # Impossible date
        "has_any_contradiction": True
    }
    py_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.95}
    gtm_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.90}
    rule_res = {"rules_passed": ["WARRANTY_ACTIVE_CHECK"], "rules_failed": []}

    res = final_claim_decision(claim, py_res, gtm_res, rule_res)
    assert res["final_decision"] == "Manual Review Required"
    assert len(res["decision_explanation"]["contradictions"]) > 0


def test_decision_explanation_dictionary_schema():
    """Verify decision_explanation contains all required factors and fields."""
    claim = {
        "claim_id": "CLM-EXPLANATION-001",
        "has_receipt": False,
        "has_warranty_card": True,
        "has_product_image": True,
        "has_serial_evidence": True,
        "has_any_contradiction": False
    }
    py_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.75}
    gtm_res = {"predicted_class": "Valid Claim", "confidence_valid": 0.72}
    rule_res = {
        "rules_passed": ["WARRANTY_ACTIVE_CHECK"],
        "rules_failed": ["PROOF_OF_PURCHASE_CHECK"],
        "manual_review_required": True,
        "reasons": ["Missing original sales receipt"]
    }

    res = final_claim_decision(claim, py_res, gtm_res, rule_res)
    expl = res["decision_explanation"]

    assert isinstance(expl, dict)
    assert "factors_supporting" in expl
    assert "factors_opposing" in expl
    assert "rules_passed" in expl
    assert "rules_failed" in expl
    assert "additional_evidence_needed" in expl
    assert "summary" in expl

    assert isinstance(expl["factors_supporting"], list)
    assert isinstance(expl["factors_opposing"], list)
    assert isinstance(expl["rules_passed"], list)
    assert isinstance(expl["rules_failed"], list)
    assert isinstance(expl["additional_evidence_needed"], list)
    assert isinstance(expl["summary"], str)

    # Verify missing receipt triggers specific evidentiary requirement
    assert any("invoice" in e.lower() or "receipt" in e.lower() for e in expl["additional_evidence_needed"])


if __name__ == "__main__":
    print("Running AssureX Decision Engine Unit Tests...")
    test_confidence_difference_computation()
    test_model_consistency_all_five_categories()
    test_thresholds_json_loading()
    test_contradiction_detection()
    test_decision_likely_valid_when_unanimous()
    test_decision_likely_invalid_when_consensus()
    test_decision_manual_review_on_model_disagreement()
    test_decision_manual_review_on_contradiction()
    test_decision_explanation_dictionary_schema()
    print("[PASS] All 9 Decision Engine unit tests passed successfully!")
