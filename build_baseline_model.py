"""
Baseline Model Builder & Serialization for AssureX Claim Classifier
===================================================================
Builds a high-accuracy, calibrated multi-class classifier on train_features.csv,
computes metrics on val_features.csv and test_features.csv, and serializes:
- model/claim_classifier.pkl
- model/preprocessing.pkl
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

SEED = 42
TARGET_CLASSES = ['Valid Claim', 'Invalid Claim', 'Manual Review']


class StandardFeatureScaler:
    """Numpy-based standard scaler with identical API to sklearn.preprocessing.StandardScaler."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        # Prevent division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SoftmaxClaimClassifier:
    """
    Calibrated Multi-Class Softmax Logistic Model with L2 Regularization.
    Trained via Adam optimizer on the engineered features.
    Provides identical interface to scikit-learn classifiers:
    - classes_
    - predict_proba(X)
    - predict(X)
    """
    def __init__(self, lr=0.01, l2=0.001, epochs=1500, random_state=42):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.random_state = random_state
        self.classes_ = np.array(TARGET_CLASSES)
        self.W = None
        self.b = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        N, D = X_arr.shape
        K = len(self.classes_)

        # One-hot encode targets
        Y = np.zeros((N, K))
        for k, c in enumerate(self.classes_):
            Y[y == c, k] = 1.0

        # Initialize weights
        self.W = np.random.randn(D, K) * 0.01
        self.b = np.zeros((1, K))

        # Adam optimizer state
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            logits = X_arr @ self.W + self.b
            probs = self._softmax(logits)

            # Gradients with L2 regularization
            grad_W = (X_arr.T @ (probs - Y)) / N + self.l2 * self.W
            grad_b = np.sum(probs - Y, axis=0, keepdims=True) / N

            # Adam step
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
            m_hat_W = mW / (1 - beta1 ** t)
            v_hat_W = vW / (1 - beta2 ** t)
            self.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + eps)

            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
            m_hat_b = mb / (1 - beta1 ** t)
            v_hat_b = vb / (1 - beta2 ** t)
            self.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        logits = X_arr @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def build_and_save_artifacts():
    print("[*] Reading feature matrices...")
    train_df = pd.read_csv("dataset/train_features.csv" if os.path.exists("dataset/train_features.csv") else "train_features.csv")
    val_df = pd.read_csv("dataset/val_features.csv" if os.path.exists("dataset/val_features.csv") else "val_features.csv")
    test_df = pd.read_csv("dataset/test_features.csv" if os.path.exists("dataset/test_features.csv") else "test_features.csv")

    drop_cols = ['claim_id', 'class_label']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df['class_label'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['class_label'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['class_label'].values

    # Fit Scaler
    scaler = StandardFeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train Calibrated Model
    print("[*] Training calibrated classifier...")
    clf = SoftmaxClaimClassifier(lr=0.05, l2=0.0005, epochs=2500, random_state=SEED)
    clf.fit(X_train_scaled, y_train)

    # Evaluate Validation Performance
    val_preds = clf.predict(X_val_scaled)
    val_acc = np.mean(val_preds == y_val)

    # Evaluate Test Performance
    test_preds = clf.predict(X_test_scaled)
    test_acc = np.mean(test_preds == y_test)

    # Per-class recall on test
    test_recalls = {}
    for c in TARGET_CLASSES:
        mask = (y_test == c)
        rec = np.mean(test_preds[mask] == c) if np.sum(mask) > 0 else 0.0
        test_recalls[c] = rec

    print(f"[+] Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"[+] Test Accuracy:       {test_acc:.4f} ({test_acc*100:.1f}%)")
    for c, r in test_recalls.items():
        print(f"    - Recall [{c}]: {r:.4f}")

    # Build Preprocessing Metadata Bundle (fitted from raw train.csv)
    raw_train = pd.read_csv("dataset/train.csv" if os.path.exists("dataset/train.csv") else "train.csv")
    known_cats = sorted(raw_train['product_category'].dropna().unique())
    dmg_freq = (raw_train['damage_type'].value_counts() / len(raw_train)).to_dict()
    global_prior = float((raw_train['class_label'] == 'Invalid Claim').mean())
    dmg_risk = (
        raw_train.groupby('damage_type')
        .apply(lambda g: float((g['class_label'] == 'Invalid Claim').sum() / len(g)))
        .to_dict()
    )
    ret_freq = (raw_train['retailer'].value_counts(dropna=True) / len(raw_train)).to_dict()

    preprocessing_bundle = {
        "feature_names": feature_cols,
        "scaler": scaler,
        "target_classes": TARGET_CLASSES,
        "known_categories": known_cats,
        "damage_type_freq_map": dmg_freq,
        "damage_type_invalid_risk_map": dmg_risk,
        "retailer_freq_map": ret_freq,
        "global_invalid_prior": global_prior,
        "model_architecture": "Calibrated Softmax Classifier (L2 Regularized)",
        "metrics": {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_recalls": test_recalls
        }
    }

    # Save artifacts to model/
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "claim_classifier.pkl")
    prep_path = os.path.join("model", "preprocessing.pkl")

    joblib.dump(clf, model_path)
    joblib.dump(preprocessing_bundle, prep_path)

    print(f"[+] Serialized model:         {model_path}")
    print(f"[+] Serialized preprocessing: {prep_path}")
    print("[SUCCESS] Model artifacts successfully generated!")


if __name__ == "__main__":
    build_and_save_artifacts()
