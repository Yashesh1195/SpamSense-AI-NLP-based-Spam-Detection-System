"""Feature extraction utilities for SpamShield AI.

This module is responsible only for loading and applying vectorizers and
embedding models for Bag of Words, TF-IDF, Word2Vec, and Average Word2Vec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.constants import (
    FEATURE_AVG_WORD2VEC,
    FEATURE_BOW,
    FEATURE_DISPLAY_NAMES,
    FEATURE_TFIDF,
    FEATURE_WORD2VEC,
    MODEL_FILENAMES,
    SUPPORTED_FEATURES,
    VECTORIZER_FILENAMES,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureArtifacts:
    """Container for a feature extractor and its associated model artifact."""

    feature_name: str
    vectorizer: Any
    model: Any | None = None


class FeatureExtractionError(RuntimeError):
    """Raised when a feature extraction artifact cannot be loaded or used."""


def is_supported_feature(feature_name: str) -> bool:
    """Return ``True`` when the requested feature extractor is supported."""
    return feature_name in SUPPORTED_FEATURES


def get_feature_display_name(feature_name: str) -> str:
    """Return a human-friendly display name for a feature extractor."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


def get_model_filename(feature_name: str) -> str:
    """Return the persisted model filename for a feature extractor."""
    try:
        return MODEL_FILENAMES[feature_name]
    except KeyError as exc:
        raise FeatureExtractionError(f"Unsupported feature extractor: {feature_name}") from exc


def get_vectorizer_filename(feature_name: str) -> str:
    """Return the persisted vectorizer filename for a feature extractor."""
    try:
        return VECTORIZER_FILENAMES[feature_name]
    except KeyError as exc:
        raise FeatureExtractionError(f"Unsupported feature extractor: {feature_name}") from exc


def load_vectorizer(vectorizer_path: str | Path) -> Any:
    """Load a persisted vectorizer or embedding model from disk."""
    resolved_path = Path(vectorizer_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found: {resolved_path}")

    LOGGER.info("Loading vectorizer artifact from %s", resolved_path)
    try:
        if resolved_path.suffix == ".pkl":
            return joblib.load(resolved_path)
        if resolved_path.suffix == ".model":
            from gensim.models import Word2Vec

            return Word2Vec.load(str(resolved_path))
        return joblib.load(resolved_path)
    except Exception as exc:  # pragma: no cover - depends on persisted artifact format
        raise FeatureExtractionError(f"Unable to load vectorizer artifact: {resolved_path}") from exc


def load_feature_artifacts(
    feature_name: str,
    *,
    vectorizer_path: str | Path,
    model_path: str | Path | None = None,
) -> FeatureArtifacts:
    """Load the vectorizer and optional model for a selected feature extractor."""
    if not is_supported_feature(feature_name):
        raise FeatureExtractionError(f"Unsupported feature extractor: {feature_name}")

    vectorizer = load_vectorizer(vectorizer_path)
    model = None
    if model_path is not None:
        model = load_model_artifact(model_path)
    return FeatureArtifacts(feature_name=feature_name, vectorizer=vectorizer, model=model)


def load_model_artifact(model_path: str | Path) -> Any:
    """Load a persisted estimator from disk."""
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {resolved_path}")

    LOGGER.info("Loading model artifact from %s", resolved_path)
    try:
        return joblib.load(resolved_path)
    except Exception as exc:  # pragma: no cover - depends on persisted artifact format
        raise FeatureExtractionError(f"Unable to load model artifact: {resolved_path}") from exc


def vectorize_texts(
    texts: list[str],
    vectorizer: CountVectorizer | TfidfVectorizer | Any,
) -> Any:
    """Transform text documents into numerical features."""
    if not texts:
        return np.empty((0, 0))

    if hasattr(vectorizer, "transform"):
        return vectorizer.transform(texts)

    raise FeatureExtractionError(f"Vectorizer does not support transformation: {type(vectorizer)!r}")


def fit_count_vectorizer(
    texts: list[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int],
) -> CountVectorizer:
    """Fit a Bag of Words vectorizer on the provided corpus."""
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(texts)
    return vectorizer


def fit_tfidf_vectorizer(
    texts: list[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int],
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on the provided corpus."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(texts)
    return vectorizer


def transform_with_word2vec(
    texts: list[str],
    word2vec_model: Any,
    *,
    strategy: str = FEATURE_WORD2VEC,
) -> np.ndarray:
    """Convert texts to sentence vectors using a trained Word2Vec model.

    Args:
        texts: Preprocessed documents.
        word2vec_model: A loaded ``gensim.models.Word2Vec`` model.
        strategy: Either ``word2vec`` for summed token vectors or
            ``avg_word2vec`` for mean pooled token vectors.

    Returns:
        A 2D NumPy array of sentence embeddings.
    """
    if strategy not in {FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC}:
        raise FeatureExtractionError(f"Unknown Word2Vec strategy: {strategy}")

    if not texts:
        return np.empty((0, 0))

    if not hasattr(word2vec_model, "wv"):
        raise FeatureExtractionError("Word2Vec model must expose a `wv` attribute")

    vector_size = int(getattr(word2vec_model.wv, "vector_size", 0))
    embeddings: list[np.ndarray] = []

    for text in texts:
        tokens = [token for token in text.split() if token]
        vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
        if not vectors:
            embeddings.append(np.zeros(vector_size, dtype=float))
            continue

        stacked = np.vstack(vectors)
        if strategy == FEATURE_AVG_WORD2VEC:
            embeddings.append(stacked.mean(axis=0))
        else:
            embeddings.append(stacked.sum(axis=0))

    return np.vstack(embeddings)


def resolve_artifact_paths(
    feature_name: str,
    vectorizers_dir: str | Path,
    models_dir: str | Path,
) -> tuple[Path, Path]:
    """Resolve persisted paths for a selected feature extractor."""
    vectorizer_path = Path(vectorizers_dir) / get_vectorizer_filename(feature_name)
    model_path = Path(models_dir) / get_model_filename(feature_name)
    return vectorizer_path, model_path
