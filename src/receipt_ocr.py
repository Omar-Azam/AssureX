"""
AssureX Receipt OCR Source Bridge
=================================
Re-exports `extract_receipt_data` and parsing helpers for package imports.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from receipt_ocr import (
    extract_receipt_data,
    parse_receipt_text,
    preprocess_receipt_image,
    cross_verify_receipt_with_claim
)

__all__ = [
    "extract_receipt_data",
    "parse_receipt_text",
    "preprocess_receipt_image",
    "cross_verify_receipt_with_claim"
]
