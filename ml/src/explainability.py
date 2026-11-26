"""
explainability.py
Human-readable explanations for why a review is flagged as suspicious.

For each prediction, returns:
  - top_words          : list of (word, tfidf_weight) sorted by |weight|
  - linguistic_signals : dict of flagged stylistic indicators
  - suspicion_tier     : "Low" | "Medium" | "High"
  - confidence         : float 0-1 (probability of being deceptive)

Works best with linear models (Logistic Regression / LinearSVC).
For Random Forest, falls back to TF-IDF coefficients via a companion LR.
"""
from __future__ import annotations

import pickle
import pathlib
import logging

import numpy as np

from .preprocessing import clean_text, preprocess_for_tfidf
from .features import LinguisticFeatureExtractor

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path("models")


def _suspicion_tier(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    elif probability >= 0.45:
        return "Medium"
    return "Low"


def _get_top_tfidf_words(
    text: str,
    extractor,
    clf,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Extract the most influential TF-IDF terms from a linear model's
    coefficient vector.  Returns [(word, weight), …] sorted by |weight|.
    Handles both raw estimators and sklearn Pipelines.
    """
    # Unwrap Pipeline to get the actual classifier
    from sklearn.pipeline import Pipeline
    estimator = clf
    if isinstance(clf, Pipeline):
        estimator = clf.named_steps.get("clf", clf[-1])

    # Only meaningful for models with coef_ (LR, LinearSVC)
    try:
        # CalibratedClassifierCV wraps LinearSVC
        base = getattr(estimator, "calibrated_classifiers_", None)
        if base:
            coef = base[0].estimator.coef_[0]  # (n_features,)
        else:
            coef = estimator.coef_[0]
    except AttributeError:
        # Random Forest – no coef_; return empty list
        return []

    # TF-IDF feature names (first N features)
    tfidf_feature_names = extractor._tfidf.get_feature_names_out()
    n_tfidf = len(tfidf_feature_names)
    tfidf_coef = coef[:n_tfidf]

    # Transform the input text to get per-term TF-IDF weights
    text_tfidf = extractor._tfidf.transform([text])
    # Element-wise product: contribution = coef * tfidf_weight
    text_arr = text_tfidf.toarray()[0]
    contributions = tfidf_coef * text_arr

    top_indices = np.argsort(np.abs(contributions))[::-1][:top_n]
    result = [
        (tfidf_feature_names[i], round(float(contributions[i]), 4))
        for i in top_indices
        if text_arr[i] > 0  # only words actually in this text
    ]
    return result


def _linguistic_signals(text: str) -> dict[str, object]:
    """
    Return a dict of human-readable stylistic signals with flag booleans.
    """
    extractor = LinguisticFeatureExtractor()
    vals = extractor._extract(text)
    names = extractor.FEATURE_NAMES
    feat = dict(zip(names, vals))

    signals = {}

    if feat["exclamation_count"] >= 3:
        signals["high_exclamation_use"] = {
            "value": int(feat["exclamation_count"]),
            "note": "Unusually high number of exclamation marks – common in promotional fake reviews.",
        }

    if feat["uppercase_ratio"] > 0.08:
        signals["high_uppercase_ratio"] = {
            "value": round(feat["uppercase_ratio"], 3),
            "note": "High proportion of capital letters – may signal excessive emphasis.",
        }

    if feat["word_count"] < 50:
        signals["very_short_review"] = {
            "value": int(feat["word_count"]),
            "note": "Very short review with low detail – genuine reviews tend to be longer.",
        }

    if feat["type_token_ratio"] < 0.4:
        signals["low_lexical_diversity"] = {
            "value": round(feat["type_token_ratio"], 3),
            "note": "Low vocabulary variety – may indicate formulaic/template text.",
        }

    if feat["superlative_count"] >= 3:
        signals["heavy_superlative_use"] = {
            "value": int(feat["superlative_count"]),
            "note": "Many superlatives ('best', 'greatest') – promotional writing pattern.",
        }

    if feat["sentiment_compound"] > 0.85:
        signals["extreme_positive_sentiment"] = {
            "value": round(feat["sentiment_compound"], 3),
            "note": "Extremely positive sentiment – overly effusive reviews are more often fake.",
        }
    elif feat["sentiment_compound"] < -0.70:
        signals["extreme_negative_sentiment"] = {
            "value": round(feat["sentiment_compound"], 3),
            "note": "Extremely negative sentiment – may indicate a competitor attack.",
        }

    if feat["specificity_score"] < 2:
        signals["low_specificity"] = {
            "value": int(feat["specificity_score"]),
            "note": "Very few specific details (names, numbers, locations) – genuine reviews tend to be more concrete.",
        }

    return signals


def explain(
    text: str,
    model_name: str = "lr",
    top_words: int = 10,
) -> dict:
    """
    Produce a full explanation for a single review.

    Parameters
    ----------
    text        : raw review text
    model_name  : 'lr' | 'svm' | 'rf'
    top_words   : how many top TF-IDF terms to return

    Returns
    -------
    dict with keys: confidence, suspicion_tier, label,
                    top_words, linguistic_signals
    """
    extractor_path = MODELS_DIR / "feature_extractor.pkl"
    model_path = MODELS_DIR / f"{model_name}_pipeline.pkl"

    if not extractor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Models not found. Run `python -m ml.src.train` first."
        )

    with open(extractor_path, "rb") as f:
        extractor = pickle.load(f)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    cleaned = clean_text(text)
    X_feat = extractor.transform([cleaned])

    # Confidence
    try:
        confidence = float(clf.predict_proba(X_feat)[0, 1])
    except AttributeError:
        # Fallback: decision function normalised
        score = clf.decision_function(X_feat)[0]
        confidence = float(1 / (1 + np.exp(-score)))

    prediction = int(confidence >= 0.5)
    tier = _suspicion_tier(confidence)

    top = _get_top_tfidf_words(cleaned, extractor, clf, top_n=top_words)
    signals = _linguistic_signals(cleaned)

    return {
        "confidence": round(confidence, 4),
        "suspicion_tier": tier,
        "label": prediction,       # 1 = deceptive, 0 = genuine
        "top_words": [{"word": w, "weight": wt} for w, wt in top],
        "linguistic_signals": signals,
    }


if __name__ == "__main__":
    import sys
    sample = " ".join(sys.argv[1:]) or (
        "Best hotel ever!! Absolutely amazing, the BEST in the whole city!!! "
        "Perfect in every way, I cannot recommend it enough!!!"
    )
    result = explain(sample)
    import json
    print(json.dumps(result, indent=2))
