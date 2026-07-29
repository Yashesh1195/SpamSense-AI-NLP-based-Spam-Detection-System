"""Model factory and registry for SpamShield AI.

This module is responsible for creating, describing, and resolving supported
machine learning estimators. Training and evaluation logic live elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.constants import (
    FEATURE_AVG_WORD2VEC,
    FEATURE_BOW,
    FEATURE_TFIDF,
    FEATURE_WORD2VEC,
    MODEL_FILENAMES,
)

LOGGER = logging.getLogger(__name__)


class ModelConfigurationError(RuntimeError):
    """Raised when a model name or artifact cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Description of a trainable model configuration."""

    name: str
    display_name: str
    factory: Callable[[], Any]
    supports_probabilities: bool = True
    preferred_features: tuple[str, ...] = ()


def _build_naive_bayes() -> MultinomialNB:
    return MultinomialNB()


def _build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, solver="lbfgs")


def _build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=42)


def _build_xgboost_classifier() -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ModelConfigurationError(
            "xgboost is not installed. Add it to requirements.txt to enable the XGBoost model."
        ) from exc

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "bow_nb": ModelSpec(
        name="bow_nb",
        display_name="Naive Bayes",
        factory=_build_naive_bayes,
        supports_probabilities=True,
        preferred_features=(FEATURE_BOW,),
    ),
    "tfidf_lr": ModelSpec(
        name="tfidf_lr",
        display_name="Logistic Regression",
        factory=_build_logistic_regression,
        supports_probabilities=True,
        preferred_features=(FEATURE_TFIDF,),
    ),
    "word2vec_rf": ModelSpec(
        name="word2vec_rf",
        display_name="Random Forest",
        factory=_build_random_forest,
        supports_probabilities=True,
        preferred_features=(FEATURE_WORD2VEC,),
    ),
    "avg_word2vec_xgb": ModelSpec(
        name="avg_word2vec_xgb",
        display_name="XGBoost",
        factory=_build_xgboost_classifier,
        supports_probabilities=True,
        preferred_features=(FEATURE_AVG_WORD2VEC,),
    ),
}

FEATURE_TO_MODEL: dict[str, str] = {
    FEATURE_BOW: "bow_nb",
    FEATURE_TFIDF: "tfidf_lr",
    FEATURE_WORD2VEC: "word2vec_rf",
    FEATURE_AVG_WORD2VEC: "avg_word2vec_xgb",
}


def list_model_names() -> tuple[str, ...]:
    """Return the registered model names in declaration order."""
    return tuple(MODEL_REGISTRY.keys())


def get_model_spec(model_name: str) -> ModelSpec:
    """Resolve a model specification from the registry."""
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ModelConfigurationError(f"Unknown model: {model_name}") from exc


def get_model_display_name(model_name: str) -> str:
    """Return the user-facing display name for a model."""
    return get_model_spec(model_name).display_name


def get_default_model_for_feature(feature_name: str) -> str:
    """Return the recommended model name for a feature extractor."""
    try:
        return FEATURE_TO_MODEL[feature_name]
    except KeyError as exc:
        raise ModelConfigurationError(f"Unsupported feature extractor: {feature_name}") from exc


def build_model(model_name: str) -> Any:
    """Instantiate a supported model."""
    spec = get_model_spec(model_name)
    return spec.factory()


def load_model(model_path: str) -> Any:
    """Load a persisted model artifact from disk."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model artifact not found: {model_path}") from exc
    except Exception as exc:  # pragma: no cover - depends on external artifact format
        raise ModelConfigurationError(f"Unable to load model artifact: {model_path}") from exc


def save_model(model: Any, model_path: str) -> None:
    """Persist a trained model artifact to disk."""
    joblib.dump(model, model_path)
    LOGGER.info("Model saved to %s", model_path)


def get_model_filename(model_name: str) -> str:
    """Return the canonical filename for a model artifact."""
    if model_name in MODEL_FILENAMES.values():
        return model_name

    for filename in MODEL_FILENAMES.values():
        if filename.startswith(model_name + "."):
            return filename

    if model_name not in MODEL_REGISTRY:
        raise ModelConfigurationError(f"Unknown model: {model_name}")

    return f"{model_name}.pkl"


def infer_feature_for_model(model_name: str) -> str:
    """Return the preferred feature extractor for a model."""
    spec = get_model_spec(model_name)
    if not spec.preferred_features:
        raise ModelConfigurationError(f"No preferred feature extractor configured for {model_name}")
    return spec.preferred_features[0]
