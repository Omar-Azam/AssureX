"""
AssureX GTM Classifier Source Bridge
====================================
Re-exports `classify_claim_card` and `GTMClaimClassifier` for package imports.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gtm_classifier import (
    GTMClaimClassifier,
    classify_claim_card,
    predict_claim_card,
    get_gtm_classifier,
    get_model_status
)

__all__ = [
    "GTMClaimClassifier",
    "classify_claim_card",
    "predict_claim_card",
    "get_gtm_classifier",
    "get_model_status"
]
