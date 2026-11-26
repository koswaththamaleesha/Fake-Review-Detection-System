"""
train.py
Trains Logistic Regression, Random Forest, and Linear SVM classifiers
on the Op-Spam training split.  Saves:
  - models/<model_name>_pipeline.pkl
  - models/feature_extractor.pkl     (shared fitted extractor)
  - docs/test_reports/metrics_<timestamp>.json

Usage
-----
  python -m ml.src.train            # from project root
  python -m ml.src.train --models lr  # only logistic regression
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import pickle
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .data_loader import load_opspam
from .features import build_feature_extractor

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path("models")
REPORTS_DIR = pathlib.Path("docs/test_reports")

CLASSIFIER_CONFIGS: dict[str, object] = {
    # LR: MaxAbsScaler preserves sparsity; lbfgs converges well on scaled TF-IDF
    "lr": Pipeline([
        ("scaler", MaxAbsScaler()),
        ("clf", LogisticRegression(
            max_iter=10_000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            C=1.0,
            tol=1e-4,
        )),
    ]),
    # SVM: MaxAbsScaler + much higher max_iter
    "svm": Pipeline([
        ("scaler", MaxAbsScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=20_000,
                tol=1e-4,
            )
        )),
    ]),
    # RF: tree-based, no scaling needed
    "rf": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def train(selected_models: list[str] | None = None) -> dict:
    """
    Fit feature extractor + each classifier on the Op-Spam train split.
    Returns a dict of {model_name: pipeline} pairs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info("Loading Op-Spam dataset …")
    df = load_opspam()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    logger.info("Training rows: %d  (fake=%d, genuine=%d)",
                len(train_df),
                train_df.label.sum(),
                (train_df.label == 0).sum())

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].values

    # --- Fit feature extractor ONCE on training data ---
    logger.info("Fitting feature extractor (TF-IDF + linguistic) …")
    extractor = build_feature_extractor(
        tfidf_max_features=20_000,
        tfidf_ngram_range=(1, 2),
        return_dense=False,
    )
    t0 = time.time()
    X_train_feat = extractor.fit_transform(X_train, y_train)
    logger.info("Feature extraction done in %.1fs, shape=%s", time.time() - t0, X_train_feat.shape)

    # Save fitted extractor
    extractor_path = MODELS_DIR / "feature_extractor.pkl"
    with open(extractor_path, "wb") as f:
        pickle.dump(extractor, f)
    logger.info("Saved feature extractor → %s", extractor_path)

    # --- Train each classifier ---
    to_train = selected_models or list(CLASSIFIER_CONFIGS)
    fitted = {}
    for name in to_train:
        clf = CLASSIFIER_CONFIGS[name]
        logger.info("Training %s …", name.upper())
        t0 = time.time()
        clf.fit(X_train_feat, y_train)
        elapsed = time.time() - t0
        logger.info("  %s trained in %.1fs", name.upper(), elapsed)

        path = MODELS_DIR / f"{name}_pipeline.pkl"
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("  Saved → %s", path)
        fitted[name] = clf

    return fitted, extractor


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train fake-review detection models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CLASSIFIER_CONFIGS),
        default=None,
        help="Which models to train (default: all)",
    )
    args = parser.parse_args()
    train(args.models)


if __name__ == "__main__":
    main()
