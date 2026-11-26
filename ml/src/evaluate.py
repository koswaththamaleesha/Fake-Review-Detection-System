"""
evaluate.py
Evaluates trained models on:
  - in-domain test split (Op-Spam fold5)
  - cross-dataset split  (Yelp filtered reviews, held out entirely)

Metrics: Precision, Recall, F1, False-Positive Rate, ROC-AUC.
Saves a JSON report to docs/test_reports/ and prints a summary table.

Usage
-----
  python -m ml.src.evaluate               # all models
  python -m ml.src.evaluate --models lr   # only LR
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .data_loader import load_all

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path("models")
REPORTS_DIR = pathlib.Path("docs/test_reports")


def _fpr(y_true, y_pred) -> float:
    """False-Positive Rate = FP / (FP + TN)"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float("nan")


def evaluate_model(
    name: str,
    clf,
    X_feat,
    y_test: np.ndarray,
    split_label: str,
) -> dict:
    """Compute all metrics for one model on one pre-extracted split."""
    y_pred = clf.predict(X_feat)

    # Probability scores for AUC (available on calibrated estimators)
    try:
        y_proba = clf.predict_proba(X_feat)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except AttributeError:
        auc = float("nan")

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    fpr = _fpr(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=["genuine", "deceptive"])

    result = {
        "model": name,
        "split": split_label,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "n_samples": len(y_test),
        "class_report": report_str,
    }
    return result


def evaluate(selected_models: list[str] | None = None) -> list[dict]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load extractor ---
    extractor_path = MODELS_DIR / "feature_extractor.pkl"
    if not extractor_path.exists():
        raise FileNotFoundError(
            f"Feature extractor not found at {extractor_path}. Run train.py first."
        )
    with open(extractor_path, "rb") as f:
        extractor = pickle.load(f)
    logger.info("Loaded feature extractor from %s", extractor_path)

    # --- Load data ---
    df = load_all()
    indomain = df[df["split"] == "test_indomain"]
    cross = df[df["split"] == "test_cross"]
    logger.info("In-domain test: %d rows | Cross-dataset test: %d rows",
                len(indomain), len(cross))

    splits = {
        "test_indomain": (extractor.transform(indomain["text"].tolist()), indomain["label"].values),
        "test_cross":    (extractor.transform(cross["text"].tolist()),    cross["label"].values),
    }

    # --- Discover models ---
    model_files = list(MODELS_DIR.glob("*_pipeline.pkl"))
    if not model_files:
        raise FileNotFoundError("No *_pipeline.pkl files in models/. Run train.py first.")

    all_results: list[dict] = []
    for pkl_path in sorted(model_files):
        name = pkl_path.stem.replace("_pipeline", "")
        if selected_models and name not in selected_models:
            continue
        with open(pkl_path, "rb") as f:
            clf = pickle.load(f)
        logger.info("Evaluating %s …", name.upper())
        for split_label, (X_feat, y) in splits.items():
            result = evaluate_model(name, clf, X_feat, y, split_label)
            all_results.append(result)
            logger.info(
                "  %s | %s → P=%.3f R=%.3f F1=%.3f FPR=%.3f AUC=%s",
                name, split_label,
                result["precision"], result["recall"],
                result["f1"], result["fpr"],
                result["roc_auc"],
            )
            print(result["class_report"])

    # --- Save JSON report ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"metrics_{timestamp}.json"
    with open(report_path, "w") as f:
        # Exclude class_report strings from JSON for cleanliness
        json.dump(
            [{k: v for k, v in r.items() if k != "class_report"} for r in all_results],
            f, indent=2
        )
    logger.info("Report saved → %s", report_path)

    # --- Summary table ---
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Model": r["model"].upper(),
            "Split": r["split"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "FPR": r["fpr"],
            "ROC-AUC": r["roc_auc"],
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

    return all_results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate fake-review detection models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to evaluate (default: all found in models/)",
    )
    args = parser.parse_args()
    evaluate(args.models)


if __name__ == "__main__":
    main()
