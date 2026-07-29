"""Shared constants for SpamShield AI.

This module stores immutable identifiers used across preprocessing,
feature extraction, training, prediction, evaluation, and the Streamlit UI.
"""

from __future__ import annotations

from typing import Final

FEATURE_BOW: Final[str] = "bow"
FEATURE_TFIDF: Final[str] = "tfidf"
FEATURE_WORD2VEC: Final[str] = "word2vec"
FEATURE_AVG_WORD2VEC: Final[str] = "avg_word2vec"

FEATURE_DISPLAY_NAMES: Final[dict[str, str]] = {
    FEATURE_BOW: "Bag of Words",
    FEATURE_TFIDF: "TF-IDF",
    FEATURE_WORD2VEC: "Word2Vec",
    FEATURE_AVG_WORD2VEC: "Average Word2Vec",
}

MODEL_FILENAMES: Final[dict[str, str]] = {
    FEATURE_BOW: "bow_nb.pkl",
    FEATURE_TFIDF: "tfidf_lr.pkl",
    FEATURE_WORD2VEC: "word2vec_rf.pkl",
    FEATURE_AVG_WORD2VEC: "avg_word2vec_xgb.pkl",
}

VECTORIZER_FILENAMES: Final[dict[str, str]] = {
    FEATURE_BOW: "bow.pkl",
    FEATURE_TFIDF: "tfidf.pkl",
    FEATURE_WORD2VEC: "word2vec.model",
    FEATURE_AVG_WORD2VEC: "avg_word2vec.model",
}

SUPPORTED_FEATURES: Final[tuple[str, ...]] = (
    FEATURE_BOW,
    FEATURE_TFIDF,
    FEATURE_WORD2VEC,
    FEATURE_AVG_WORD2VEC,
)

TEXT_COLUMNS: Final[tuple[str, ...]] = ("message",)
TARGET_COLUMN: Final[str] = "label"
PREDICTION_COLUMN: Final[str] = "prediction"
CONFIDENCE_COLUMN: Final[str] = "confidence"
SPAM_PROBABILITY_COLUMN: Final[str] = "spam_probability"
HAM_PROBABILITY_COLUMN: Final[str] = "ham_probability"

CLASS_SPAM: Final[int] = 1
CLASS_HAM: Final[int] = 0

PLOTLY_TEMPLATE: Final[str] = "plotly_dark"
DEFAULT_BATCH_PREDICTION_FILENAME: Final[str] = "sample_predictions.csv"
