"""Prediction pipeline for SpamShield AI.

This module is responsible only for loading trained artifacts and generating
single-message or batch predictions with probabilities and confidence scores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config
from src.constants import (
    CLASS_HAM,
    CLASS_SPAM,
    CONFIDENCE_COLUMN,
    HAM_PROBABILITY_COLUMN,
    MODEL_FILENAMES,
    PREDICTION_COLUMN,
    SPAM_PROBABILITY_COLUMN,
)
from src.feature_extraction import FeatureArtifacts, FeatureExtractionError, load_feature_artifacts, transform_with_word2vec
from src.model import ModelConfigurationError, get_default_model_for_feature, get_model_filename
from src.preprocessing import clean_text
from src.train import ensure_model_artifacts

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictionResult:
    """Structured output for a single spam prediction."""

    input_text: str
    cleaned_text: str
    feature_name: str
    model_name: str
    predicted_label: str
    predicted_class: int
    confidence: float
    spam_probability: float
    ham_probability: float
    prediction_time_seconds: float


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced."""



def _resolve_artifacts(feature_name: str, *, config: AppConfig) -> FeatureArtifacts:
    """Load the persisted feature and model artifacts for the given feature."""
    vectorizer_path, model_path = ensure_model_artifacts(feature_name, config=config)
    return load_feature_artifacts(feature_name, vectorizer_path=vectorizer_path, model_path=model_path)



def _vectorize_input(cleaned_text: str, feature_name: str, artifacts: FeatureArtifacts) -> Any:
    """Transform a single preprocessed message into model-ready features."""
    if feature_name in {"bow", "tfidf"}:
        return artifacts.vectorizer.transform([cleaned_text])

    if feature_name in {"word2vec", "avg_word2vec"}:
        strategy = feature_name
        return transform_with_word2vec([cleaned_text], artifacts.vectorizer, strategy=strategy)

    raise PredictionError(f"Unsupported feature extractor: {feature_name}")



def _extract_probabilities(model: Any, features: Any) -> tuple[float, float]:
    """Return ham/spam probabilities for a binary classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        if len(proba) == 1:
            ham_probability = float(1.0 - proba[0])
            spam_probability = float(proba[0])
        else:
            ham_probability = float(proba[0])
            spam_probability = float(proba[1])
        return ham_probability, spam_probability

    predicted_class = int(model.predict(features)[0])
    return (1.0, 0.0) if predicted_class == CLASS_HAM else (0.0, 1.0)



def predict_message(
    message: str,
    feature_name: str,
    *,
    config: AppConfig | None = None,
) -> PredictionResult:
    """Predict whether a single SMS message is spam or ham."""
    runtime_config = config or get_config()
    cleaned_text = clean_text(message)
    model_name = get_default_model_for_feature(feature_name)
    artifacts = _resolve_artifacts(feature_name, config=runtime_config)

    start_time = time.perf_counter()
    features = _vectorize_input(cleaned_text, feature_name, artifacts)
    model = artifacts.model
    if model is None:
        raise PredictionError(f"Model artifact missing for {feature_name}")

    predicted_class = int(model.predict(features)[0])
    ham_probability, spam_probability = _extract_probabilities(model, features)
    elapsed = time.perf_counter() - start_time

    if predicted_class == CLASS_SPAM:
        predicted_label = "Spam"
        confidence = spam_probability
    else:
        predicted_label = "Ham"
        confidence = ham_probability

    return PredictionResult(
        input_text=message,
        cleaned_text=cleaned_text,
        feature_name=feature_name,
        model_name=model_name,
        predicted_label=predicted_label,
        predicted_class=predicted_class,
        confidence=float(confidence),
        spam_probability=float(spam_probability),
        ham_probability=float(ham_probability),
        prediction_time_seconds=elapsed,
    )



def predict_messages(
    messages: list[str],
    feature_name: str,
    *,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Predict labels for a batch of SMS messages."""
    results = [predict_message(message, feature_name, config=config) for message in messages]
    return pd.DataFrame(
        [
            {
                "input_text": result.input_text,
                "cleaned_text": result.cleaned_text,
                PREDICTION_COLUMN: result.predicted_label,
                CONFIDENCE_COLUMN: result.confidence,
                SPAM_PROBABILITY_COLUMN: result.spam_probability,
                HAM_PROBABILITY_COLUMN: result.ham_probability,
                "prediction_time_seconds": result.prediction_time_seconds,
                "model_name": result.model_name,
                "feature_name": result.feature_name,
            }
            for result in results
        ]
    )



def predict_from_artifacts(
    message: str,
    feature_name: str,
    *,
    config: AppConfig | None = None,
) -> PredictionResult:
    """Convenience wrapper for a single prediction.

    The wrapper is intentionally separate so the Streamlit app can depend on a
    stable public API even if the internal implementation evolves.
    """
    return predict_message(message, feature_name, config=config)



def get_prediction_summary(result: PredictionResult) -> str:
    """Return a concise textual summary for UI display."""
    return (
        f"{result.predicted_label} | confidence={result.confidence:.2%} | "
        f"spam={result.spam_probability:.2%} | ham={result.ham_probability:.2%}"
    )
