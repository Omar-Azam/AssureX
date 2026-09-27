"""
AssureX Decision Engine Source Bridge
=====================================
Re-exports `final_claim_decision` and arbitration helpers for package imports.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_engine import (
    final_claim_decision,
    classify_model_consistency,
    extract_top_prediction,
    extract_and_evaluate_contradictions,
    load_decision_thresholds,
    determine_additional_evidence_needed
)

__all__ = [
    "final_claim_decision",
    "classify_model_consistency",
    "extract_top_prediction",
    "extract_and_evaluate_contradictions",
    "load_decision_thresholds",
    "determine_additional_evidence_needed"
]
