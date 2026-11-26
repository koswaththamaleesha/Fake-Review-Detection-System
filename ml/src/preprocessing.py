"""
preprocessing.py
Text normalisation utilities for the fake-review detection pipeline.

Design decisions
----------------
* Lowercase, but preserve sentence structure for feature extraction.
* Remove HTML artefacts and excess whitespace.
* Optionally strip punctuation (disabled by default) because exclamation
  marks, question marks, and capitalisation carry stylistic signal.
* Expand common English contractions so the tokeniser sees full words.
* Lemmatise with spaCy for clean bag-of-words features, but keep the
  original (lightly cleaned) text for linguistic/stylistic feature extraction.
"""
from __future__ import annotations

import re
import unicodedata

import spacy

# ---------------------------------------------------------------------------
# spaCy model – loaded lazily once
# ---------------------------------------------------------------------------
_nlp = None

def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _nlp


# ---------------------------------------------------------------------------
# Contraction expansion map (common American English)
# ---------------------------------------------------------------------------
_CONTRACTIONS: dict[str, str] = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdon't\b": "do not",
    r"\bdidn't\b": "did not",
    r"\bdoesn't\b": "does not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhadn't\b": "had not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
    r"\bI'm\b": "I am",
    r"\bI've\b": "I have",
    r"\bI'll\b": "I will",
    r"\bI'd\b": "I would",
    r"\bthey're\b": "they are",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bwe're\b": "we are",
    r"\byou're\b": "you are",
    r"\bwe've\b": "we have",
    r"\bthey've\b": "they have",
    r"\bwe'll\b": "we will",
    r"\bthey'll\b": "they will",
    r"\bthere's\b": "there is",
    r"\bwhat's\b": "what is",
}

_CONTRACTION_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _CONTRACTIONS.items()
]


def _expand_contractions(text: str) -> str:
    for pattern, repl in _CONTRACTION_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _remove_html(text: str) -> str:
    """Strip basic HTML tags and decode common HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    return text


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalise_unicode(text: str) -> str:
    """Convert fancy Unicode quotes/dashes to ASCII equivalents."""
    text = unicodedata.normalize("NFKC", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str, expand_contractions: bool = True) -> str:
    """
    Light cleaning: unicode normalisation, HTML removal, contraction
    expansion, and whitespace normalisation.

    Preserves case, punctuation (!, ?, …) intentionally so that
    linguistic features can still be computed on the returned string.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _normalise_unicode(text)
    text = _remove_html(text)
    if expand_contractions:
        text = _expand_contractions(text)
    text = _normalise_whitespace(text)
    return text


def lemmatise(text: str, remove_stopwords: bool = True) -> str:
    """
    Lemmatise and optionally remove stopwords.
    Returns a space-joined string of lemmas (lowercase) suitable for TF-IDF.
    """
    nlp = _get_nlp()
    doc = nlp(text.lower())
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if remove_stopwords and tok.is_stop:
            continue
        tokens.append(tok.lemma_)
    return " ".join(tokens)


def preprocess_for_tfidf(text: str) -> list[str]:
    """
    Full pipeline: clean → lemmatise → return token list.
    Used as the TF-IDF analyser callable (must return an iterable of tokens,
    NOT a string — otherwise sklearn iterates over characters).
    """
    return lemmatise(clean_text(text)).split()


def preprocess_for_tfidf_bigrams(text: str) -> list[str]:
    """
    Same as preprocess_for_tfidf but also appends bigram pseudo-tokens
    so that TfidfVectorizer with analyzer=callable still captures bigrams.
    """
    tokens = preprocess_for_tfidf(text)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def preprocess_series(series, n_jobs: int = 1):
    """Apply preprocess_for_tfidf to a pandas Series, optionally in parallel."""
    import pandas as pd
    if n_jobs == 1:
        return series.map(preprocess_for_tfidf)
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_for_tfidf)(t) for t in series
    )
    return pd.Series(results, index=series.index)
