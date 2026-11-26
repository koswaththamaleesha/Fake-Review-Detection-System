"""
features.py
Builds the combined feature matrix:
  1. TF-IDF features (word uni/bigrams) – fitted ONLY on training data
  2. Linguistic/stylistic hand-crafted features

Linguistic features (all computed on the *lightly-cleaned* text):
  - word_count
  - char_count
  - sentence_count
  - avg_word_length
  - exclamation_count
  - question_count
  - uppercase_ratio
  - sentiment_compound  (VADER)
  - type_token_ratio    (lexical diversity)
  - superlative_count   (spaCy POS tags – RBS)
  - specificity_score   (numbers / concrete nouns as a proxy)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from scipy.sparse import hstack, issparse

from .preprocessing import clean_text, preprocess_for_tfidf, preprocess_for_tfidf_bigrams

# Ensure project-local NLTK data (venv/nltk_data) is found first,
# so training works inside the sandbox without needing AppData\Roaming.
import pathlib as _pathlib
import nltk as _nltk
_local_nltk = str(_pathlib.Path(__file__).resolve().parents[2] / "venv" / "nltk_data")
if _local_nltk not in _nltk.data.path:
    _nltk.data.path.insert(0, _local_nltk)

# VADER sentiment – loaded lazily to avoid import-time network calls
_vader = None

def _get_vader():
    global _vader
    if _vader is None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as _SIA
            _vader = _SIA()
        except LookupError:
            import nltk
            nltk.download("vader_lexicon", quiet=True, download_dir=None)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as _SIA
            _vader = _SIA()
    return _vader

# spaCy for POS-based features
import spacy
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


# ---------------------------------------------------------------------------
# Linguistic feature extractor
# ---------------------------------------------------------------------------

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer.
    Accepts a list/Series of raw text strings, returns a 2D float array.
    """
    FEATURE_NAMES = [
        "word_count",
        "char_count",
        "sentence_count",
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
        "sentiment_compound",
        "type_token_ratio",
        "superlative_count",
        "specificity_score",
    ]

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X, y=None):
        return np.array([self._extract(text) for text in X], dtype=float)

    def _extract(self, raw_text: str) -> list[float]:
        text = clean_text(raw_text)

        # Basic counts
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # Sentence count (split on .!? followed by space/end)
        sentences = re.split(r"[.!?]+(?:\s|$)", text.strip())
        sentence_count = max(1, len([s for s in sentences if s.strip()]))

        avg_word_length = (
            np.mean([len(w) for w in words]) if words else 0.0
        )

        exclamation_count = text.count("!")
        question_count = text.count("?")

        # Uppercase ratio (letters only)
        letters = [c for c in text if c.isalpha()]
        uppercase_ratio = (
            sum(1 for c in letters if c.isupper()) / len(letters)
            if letters else 0.0
        )

        # VADER sentiment
        sentiment_compound = _get_vader().polarity_scores(text)["compound"]

        # Lexical diversity
        lower_words = [w.lower() for w in words]
        type_token_ratio = (
            len(set(lower_words)) / len(lower_words) if lower_words else 0.0
        )

        # spaCy POS for superlatives and specificity
        nlp = _get_nlp()
        doc = nlp(text[:1000])  # cap length for speed
        superlative_count = sum(1 for tok in doc if tok.tag_ == "JJS")

        # Specificity: cardinal numbers + proper nouns as concrete-detail proxy
        specificity_score = sum(
            1 for tok in doc if tok.pos_ in {"NUM"} or tok.tag_ == "NNP"
        )

        return [
            word_count,
            char_count,
            sentence_count,
            avg_word_length,
            exclamation_count,
            question_count,
            uppercase_ratio,
            sentiment_compound,
            type_token_ratio,
            superlative_count,
            specificity_score,
        ]

    def get_feature_names_out(self):
        return np.array(self.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# TF-IDF transformer wrapper (makes feature-name access clean)
# ---------------------------------------------------------------------------
class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps TfidfVectorizer so it accepts raw text (cleaning is applied inside).
    """
    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self._vec: TfidfVectorizer | None = None

    def fit(self, X, y=None):
        self._vec = TfidfVectorizer(
            analyzer=preprocess_for_tfidf_bigrams,  # returns list of unigram+bigram tokens
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            # ngram_range is intentionally omitted: bigrams are built inside the callable
        )
        self._vec.fit(X)
        return self

    def transform(self, X, y=None):
        return self._vec.transform(X)

    def get_feature_names_out(self):
        return self._vec.get_feature_names_out()


# ---------------------------------------------------------------------------
# Combined feature pipeline
# ---------------------------------------------------------------------------

class CombinedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Concatenates TF-IDF (sparse) with linguistic features (dense).
    Returns a dense numpy array for tree models, or keeps sparse for linear.
    """
    def __init__(
        self,
        tfidf_max_features: int = 20_000,
        tfidf_ngram_range: tuple = (1, 2),
        return_dense: bool = False,
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.return_dense = return_dense
        self._tfidf = TfidfTransformer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
        )
        self._ling = LinguisticFeatureExtractor()

    def fit(self, X, y=None):
        self._tfidf.fit(X, y)
        self._ling.fit(X, y)
        return self

    def transform(self, X, y=None):
        from scipy.sparse import csr_matrix
        tfidf_mat = self._tfidf.transform(X)
        ling_mat = self._ling.transform(X)
        ling_sparse = csr_matrix(ling_mat)
        combined = hstack([tfidf_mat, ling_sparse])
        return combined.toarray() if self.return_dense else combined

    def get_feature_names_out(self):
        tfidf_names = self._tfidf.get_feature_names_out()
        ling_names = self._ling.get_feature_names_out()
        return np.concatenate([tfidf_names, ling_names])


def build_feature_extractor(
    tfidf_max_features: int = 20_000,
    tfidf_ngram_range: tuple = (1, 2),
    return_dense: bool = False,
) -> CombinedFeatureExtractor:
    """Factory: returns an *unfitted* CombinedFeatureExtractor."""
    return CombinedFeatureExtractor(
        tfidf_max_features=tfidf_max_features,
        tfidf_ngram_range=tfidf_ngram_range,
        return_dense=return_dense,
    )
