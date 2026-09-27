"""
AssureX Teachable Machine (GTM) Claim Card Classifier
=====================================================

This module provides the image-based claims classification pipeline for
Claim Summary Cards (PNG) trained with and exported from Google Teachable Machine (GTM).

The classifier:
1. Loads the exported Keras model (`model/gtm_model/keras_model.h5`) using
   `tensorflow.keras.models.load_model(path, compile=False)`.
2. Reads the class labels from `model/gtm_model/labels.txt`.
3. Preprocesses input images to 224x224 RGB and normalizes pixel values to match
   Teachable Machine's expected input range [-1.0, 1.0]: `(image / 127.5) - 1.0`.
4. Runs `model.predict()` to return real confidence scores for all three classes
   plus the predicted class.

Strict Zero-Fallback Policy:
---------------------------
There is NO fallback logic of any kind — no dummy predictions, no "intelligent
fallback engine," and no invented confidence values. If TensorFlow is not
installed, the model file is missing, or loading/inference fails for any reason,
a clear RuntimeError is raised immediately. It will never return a prediction
without executing the real model.

Output Schema:
--------------
{
    "predicted_class": str,            # "Valid Claim" | "Invalid Claim" | "Manual Review"
    "confidence_valid": float,         # Real model probability [0.0 - 1.0]
    "confidence_invalid": float,       # Real model probability [0.0 - 1.0]
    "confidence_manual_review": float  # Real model probability [0.0 - 1.0]
}
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps

# Project root path resolution
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "model" / "gtm_model" / "keras_model.h5"
DEFAULT_LABELS_PATH = _PROJECT_ROOT / "model" / "gtm_model" / "labels.txt"
INPUT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Global model cache to avoid reloading weights repeatedly across predictions
_MODEL_CACHE: Dict[str, Any] = {}


def load_labels(labels_path: Union[str, Path] = DEFAULT_LABELS_PATH) -> List[str]:
    """
    Read class labels from Teachable Machine labels.txt file.

    Parameters:
        labels_path: Path to the labels.txt file.

    Returns:
        List of cleaned class label strings in index order.

    Raises:
        RuntimeError: If the labels file is missing or empty.
    """
    path = Path(labels_path)
    if not path.is_file():
        raise RuntimeError(
            f"GTM labels file not found at: '{path.resolve()}'. "
            "A valid labels.txt file is strictly required."
        )

    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned:
                continue
            # Teachable Machine formats lines as: '<index> <label_name>'
            parts = cleaned.split(" ", 1)
            if len(parts) > 1 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(cleaned)

    if not labels:
        raise RuntimeError(
            f"GTM labels file at '{path.resolve()}' is empty. "
            "Expected class labels (e.g. Valid Claim, Invalid Claim, Manual Review)."
        )

    return labels


def load_gtm_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    use_cache: bool = True
) -> Any:
    """
    Load the Keras model using tensorflow.keras.models.load_model(path, compile=False).

    Parameters:
        model_path: Path to the keras_model.h5 file.
        use_cache: Whether to return a cached model instance if already loaded.

    Returns:
        Loaded TensorFlow / Keras Model instance.

    Raises:
        RuntimeError: If TensorFlow is missing/incompatible, model file is missing,
                      or loading fails for any reason. NO fallback logic is executed.
    """
    path = Path(model_path)
    resolved_path_str = str(path.resolve())

    if use_cache and resolved_path_str in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_path_str]

    if not path.is_file():
        raise RuntimeError(
            f"GTM model file missing at '{path.resolve()}'. "
            "A valid Keras .h5 model file is strictly required. Fallbacks are disabled."
        )

    # Attempt to import TensorFlow and load_model
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError as err:
        raise RuntimeError(
            f"TensorFlow is not installed or cannot be imported ({err}). "
            "TensorFlow is strictly required to run GTM model inference. "
            "No fallback predictions are permitted."
        ) from err
    except Exception as err:
        raise RuntimeError(
            f"TensorFlow runtime failed to initialize ({err}). "
            "Cannot run GTM model without a functional native TensorFlow runtime. "
            "No fallback predictions are permitted."
        ) from err

    try:
        model = load_model(str(path), compile=False)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load Keras model from '{path.resolve()}': {err}. "
            "The model file may be corrupt or incompatible with the installed Keras version. "
            "No fallback predictions are permitted."
        ) from err

    if use_cache:
        _MODEL_CACHE[resolved_path_str] = model

    return model


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = INPUT_IMAGE_SIZE
) -> np.ndarray:
    """
    Preprocess an input Claim Summary Card image to match Teachable Machine specifications:
    1. Convert to RGB.
    2. Resize to 224x224 and crop from center using Lanczos interpolation.
    3. Normalize pixel values to [-1.0, 1.0] range: `(arr / 127.5) - 1.0`.
    4. Reshape to batch array of shape (1, 224, 224, 3).

    Parameters:
        image_input: File path, Path, PIL Image, or NumPy array.
        target_size: Target dimensions (width, height), default (224, 224).

    Returns:
        NumPy float32 ndarray with shape (1, 224, 224, 3).

    Raises:
        RuntimeError: If image file does not exist or image cannot be processed.
    """
    pil_img: Image.Image

    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise RuntimeError(f"Image file does not exist: '{img_path.resolve()}'")
        try:
            pil_img = Image.open(img_path)
        except Exception as err:
            raise RuntimeError(f"Failed to open image file '{img_path}': {err}") from err
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        try:
            if image_input.dtype != np.uint8:
                # If float array in [0, 1], scale to uint8
                if image_input.max() <= 1.0 and image_input.min() >= 0.0:
                    image_input = (image_input * 255.0).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)
            pil_img = Image.fromarray(image_input)
        except Exception as err:
            raise RuntimeError(f"Failed to convert NumPy array to PIL Image: {err}") from err
    else:
        raise RuntimeError(
            f"Unsupported image input type: {type(image_input)}. "
            "Expected file path (str/Path), PIL.Image.Image, or np.ndarray."
        )

    # Convert to RGB color mode
    pil_img = pil_img.convert("RGB")

    # Resize and center-crop to 224x224
    resample_method = getattr(Image, "Resampling", Image).LANCZOS
    resized_img = ImageOps.fit(pil_img, target_size, resample_method)

    # Convert to float32 NumPy array
    img_array = np.asarray(resized_img, dtype=np.float32)

    # Normalize pixel values to Teachable Machine's expected range [-1.0, 1.0]
    normalized_array = (img_array / 127.5) - 1.0

    # Expand batch dimension to (1, 224, 224, 3)
    batch_array = np.expand_dims(normalized_array, axis=0)

    return batch_array


def classify_claim_card(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    labels_path: Union[str, Path] = DEFAULT_LABELS_PATH
) -> Dict[str, Any]:
    """
    Classify a Claim Summary Card PNG using the real Teachable Machine Keras model.

    This function:
    1. Loads the class labels from `labels_path`.
    2. Loads the model from `model_path` via `load_model(..., compile=False)`.
    3. Preprocesses the image into a normalized (1, 224, 224, 3) batch.
    4. Runs `model.predict()` to obtain real model confidence scores.
    5. Returns the predicted class and confidence scores for all three classes.

    STRICT GUARANTEE:
    There is no fallback logic. If any step fails, a RuntimeError is raised.
    Never returns synthetic or invented predictions.

    Parameters:
        image_input: Path to image file, PIL Image, or NumPy image array.
        model_path: Path to keras_model.h5 (default: model/gtm_model/keras_model.h5).
        labels_path: Path to labels.txt (default: model/gtm_model/labels.txt).

    Returns:
        Dict[str, Any] with keys:
            - predicted_class: str ("Valid Claim" | "Invalid Claim" | "Manual Review")
            - confidence_valid: float (probability in [0.0, 1.0])
            - confidence_invalid: float (probability in [0.0, 1.0])
            - confidence_manual_review: float (probability in [0.0, 1.0])
    """
    # 1. Load labels
    labels = load_labels(labels_path)

    # 2. Load model (raises RuntimeError if TF is unavailable or loading fails)
    model = load_gtm_model(model_path)

    # 3. Preprocess image
    batch = preprocess_image(image_input, target_size=INPUT_IMAGE_SIZE)

    # 4. Run real model inference
    try:
        raw_predictions = model.predict(batch, verbose=0)
    except Exception as err:
        raise RuntimeError(f"Error during model.predict() execution: {err}") from err

    if raw_predictions is None or len(raw_predictions) == 0:
        raise RuntimeError("Model returned empty prediction output.")

    scores = raw_predictions[0]
    best_index = int(np.argmax(scores))

    if best_index < len(labels):
        predicted_class = labels[best_index]
    else:
        predicted_class = f"Class_{best_index}"

    # 5. Extract confidence scores for all 3 classes
    conf_valid = 0.0
    conf_invalid = 0.0
    conf_manual_review = 0.0

    for idx, label in enumerate(labels):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        normalized_label = label.lower().replace("_", " ").strip()

        if "valid" in normalized_label and "invalid" not in normalized_label:
            conf_valid = score
        elif "invalid" in normalized_label:
            conf_invalid = score
        elif "manual" in normalized_label or "review" in normalized_label:
            conf_manual_review = score

    return {
        "predicted_class": predicted_class,
        "confidence_valid": conf_valid,
        "confidence_invalid": conf_invalid,
        "confidence_manual_review": conf_manual_review,
    }


# Convenience alias matching project conventions
predict_claim_card = classify_claim_card


class GTMClaimClassifier:
    """
    Object-oriented interface for the Teachable Machine (GTM) Claim Card Classifier.
    Eagerly validates and loads the Keras model and labels on initialization.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        labels_path: Union[str, Path] = DEFAULT_LABELS_PATH
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.labels = load_labels(self.labels_path)
        self.model = load_gtm_model(self.model_path)

    def classify(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """Classify a single claim card image."""
        return classify_claim_card(
            image_input=image_input,
            model_path=self.model_path,
            labels_path=self.labels_path
        )

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """Alias for classify."""
        return self.classify(image_input)


_GTM_CLASSIFIER_INSTANCE: Optional[GTMClaimClassifier] = None


def get_gtm_classifier(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    labels_path: Union[str, Path] = DEFAULT_LABELS_PATH
) -> GTMClaimClassifier:
    """Return a shared singleton instance of GTMClaimClassifier."""
    global _GTM_CLASSIFIER_INSTANCE
    if _GTM_CLASSIFIER_INSTANCE is None:
        _GTM_CLASSIFIER_INSTANCE = GTMClaimClassifier(
            model_path=model_path,
            labels_path=labels_path
        )
    return _GTM_CLASSIFIER_INSTANCE


def get_model_status(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    labels_path: Union[str, Path] = DEFAULT_LABELS_PATH
) -> Dict[str, Any]:
    """Inspect status of the GTM model and labels files."""
    m_path = Path(model_path)
    l_path = Path(labels_path)
    labels: List[str] = []
    if l_path.is_file():
        try:
            labels = load_labels(l_path)
        except Exception:
            labels = []

    return {
        "model_file_exists": m_path.is_file(),
        "model_path": str(m_path.resolve()),
        "labels_file_exists": l_path.is_file(),
        "labels_path": str(l_path.resolve()),
        "labels": labels,
        "is_model_loaded": str(m_path.resolve()) in _MODEL_CACHE,
    }



# ==============================================================================
# COMMAND-LINE DEMO
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AssureX Teachable Machine (GTM) Claim Card Classifier CLI Demo"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the Claim Summary Card image (PNG/JPG) to classify"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to Keras .h5 model file (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(DEFAULT_LABELS_PATH),
        help=f"Path to labels.txt file (default: {DEFAULT_LABELS_PATH})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" AssureX GTM Claim Card Classifier - Real Model Inference")
    print("=" * 60)
    print(f"Target Image : {args.image_path}")
    print(f"Model File   : {args.model}")
    print(f"Labels File  : {args.labels}")
    print("-" * 60)

    try:
        result = classify_claim_card(
            image_input=args.image_path,
            model_path=args.model,
            labels_path=args.labels
        )
        print("\nReal Model Prediction:")
        print(f"  Predicted Class          : {result['predicted_class']}")
        print(f"  Confidence (Valid)       : {result['confidence_valid'] * 100:.2f}% ({result['confidence_valid']:.4f})")
        print(f"  Confidence (Invalid)     : {result['confidence_invalid'] * 100:.2f}% ({result['confidence_invalid']:.4f})")
        print(f"  Confidence (Manual Review): {result['confidence_manual_review'] * 100:.2f}% ({result['confidence_manual_review']:.4f})")
        print("\n[SUCCESS] Model execution completed with real inference scores.")
        print("=" * 60)
    except RuntimeError as r_err:
        print(f"\n[EXECUTION ERROR] {r_err}", file=sys.stderr)
        print("No fallback logic was run.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
