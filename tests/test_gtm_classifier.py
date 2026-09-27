"""
Unit Tests for AssureX Teachable Machine (GTM) Claim Card Classifier
====================================================================

Validates:
1. Model and labels file discovery and status checking
2. Strict Zero-Fallback Policy: Missing model or labels strictly raises RuntimeError
3. Image Preprocessing:
   - Resizing to (224, 224) RGB
   - Normalization range [-1.0, 1.0]
   - Multi-format input support (str, Path, PIL Image, np.ndarray)
4. Inference Output Contract:
   - Output schema matching {predicted_class, confidence_valid, confidence_invalid, confidence_manual_review}
   - Correct argmax class prediction and probability mapping
5. API Aliases & Object-Oriented Wrapper (GTMClaimClassifier, predict_claim_card)
"""

import sys
import glob
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from PIL import Image

from gtm_classifier import (
    load_labels,
    load_gtm_model,
    preprocess_image,
    classify_claim_card,
    predict_claim_card,
    get_model_status,
    GTMClaimClassifier,
    DEFAULT_MODEL_PATH,
    DEFAULT_LABELS_PATH,
)


def get_sample_card() -> str:
    """Finds an actual generated claim card image path in workspace."""
    pattern = "claim_cards/*/*/*.png"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No cards found matching: {pattern}")
    return matches[0]


def test_labels_loading():
    """Verify labels.txt loads correctly with index stripped."""
    labels = load_labels(DEFAULT_LABELS_PATH)
    assert isinstance(labels, list)
    assert labels == ["Valid Claim", "Invalid Claim", "Manual Review"]


def test_missing_labels_raises_runtime_error():
    """Verify missing labels.txt raises RuntimeError immediately (no fallback)."""
    try:
        load_labels("non_existent_labels_file.txt")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as err:
        assert "not found" in str(err).lower()


def test_missing_model_raises_runtime_error():
    """Verify missing model file raises RuntimeError immediately (no fallback)."""
    try:
        load_gtm_model("non_existent_model_file.h5")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as err:
        assert "missing at" in str(err).lower()


def test_model_status():
    """Verify get_model_status() returns required status fields."""
    status = get_model_status()
    assert isinstance(status, dict)
    assert status["model_file_exists"] is True
    assert status["labels_file_exists"] is True
    assert status["labels"] == ["Valid Claim", "Invalid Claim", "Manual Review"]


def test_image_preprocessing():
    """Verify image preprocessing produces normalized (1, 224, 224, 3) float32 batch."""
    card_path = get_sample_card()
    batch = preprocess_image(card_path)

    assert isinstance(batch, np.ndarray)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert -1.0 <= batch.min() <= 1.0
    assert -1.0 <= batch.max() <= 1.0


def test_in_memory_pil_and_numpy_input():
    """Verify preprocessing handles in-memory PIL and NumPy images."""
    # PIL image
    pil_img = Image.new("RGBA", (300, 300), color=(120, 80, 200, 255))
    batch_pil = preprocess_image(pil_img)
    assert batch_pil.shape == (1, 224, 224, 3)
    assert batch_pil.dtype == np.float32

    # NumPy array
    arr = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    batch_arr = preprocess_image(arr)
    assert batch_arr.shape == (1, 224, 224, 3)
    assert batch_arr.dtype == np.float32


def test_nonexistent_image_raises_runtime_error():
    """Verify missing image file raises RuntimeError or FileNotFoundError."""
    try:
        preprocess_image("non_existent_image_file_12345.png")
        assert False, "Should have raised RuntimeError"
    except (RuntimeError, FileNotFoundError):
        pass


def test_inference_output_schema_and_scores():
    """Verify classification output schema, class determination, and confidence extraction."""
    card_path = get_sample_card()
    mock_model = MagicMock()

    # Scenario 1: Valid Claim is highest
    mock_model.predict.return_value = np.array([[0.82, 0.11, 0.07]], dtype=np.float32)
    with patch("gtm_classifier.load_gtm_model", return_value=mock_model):
        res = classify_claim_card(card_path)

    expected_keys = {
        "predicted_class",
        "confidence_valid",
        "confidence_invalid",
        "confidence_manual_review",
    }
    assert set(res.keys()) == expected_keys
    assert res["predicted_class"] == "Valid Claim"
    assert round(res["confidence_valid"], 2) == 0.82
    assert round(res["confidence_invalid"], 2) == 0.11
    assert round(res["confidence_manual_review"], 2) == 0.07

    # Scenario 2: Invalid Claim is highest
    mock_model.predict.return_value = np.array([[0.04, 0.92, 0.04]], dtype=np.float32)
    with patch("gtm_classifier.load_gtm_model", return_value=mock_model):
        res = classify_claim_card(card_path)
    assert res["predicted_class"] == "Invalid Claim"
    assert round(res["confidence_invalid"], 2) == 0.92

    # Scenario 3: Manual Review is highest
    mock_model.predict.return_value = np.array([[0.05, 0.15, 0.80]], dtype=np.float32)
    with patch("gtm_classifier.load_gtm_model", return_value=mock_model):
        res = classify_claim_card(card_path)
    assert res["predicted_class"] == "Manual Review"
    assert round(res["confidence_manual_review"], 2) == 0.80


def test_predict_claim_card_alias():
    """Verify predict_claim_card alias is identical to classify_claim_card."""
    assert predict_claim_card is classify_claim_card


def test_gtm_claim_classifier_class():
    """Verify GTMClaimClassifier class wraps model and labels correctly."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.10, 0.85, 0.05]], dtype=np.float32)

    with patch("gtm_classifier.load_gtm_model", return_value=mock_model):
        classifier = GTMClaimClassifier()
        card_path = get_sample_card()
        res1 = classifier.classify(card_path)
        res2 = classifier.predict(card_path)

    assert res1 == res2
    assert res1["predicted_class"] == "Invalid Claim"


def test_strict_zero_fallback_on_inference_failure():
    """Verify that any inference failure raises RuntimeError and NEVER falls back."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("GPU out of memory")

    with patch("gtm_classifier.load_gtm_model", return_value=mock_model):
        try:
            classify_claim_card(get_sample_card())
            assert False, "Should have raised RuntimeError"
        except RuntimeError as err:
            assert "error during model.predict()" in str(err).lower()


if __name__ == "__main__":
    print("Running AssureX GTM Classifier unit tests...")
    test_labels_loading()
    test_missing_labels_raises_runtime_error()
    test_missing_model_raises_runtime_error()
    test_model_status()
    test_image_preprocessing()
    test_in_memory_pil_and_numpy_input()
    test_nonexistent_image_raises_runtime_error()
    test_inference_output_schema_and_scores()
    test_predict_claim_card_alias()
    test_gtm_claim_classifier_class()
    test_strict_zero_fallback_on_inference_failure()
    print("[PASS] All 11 GTM Classifier tests passed successfully!")
