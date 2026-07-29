"""Training workflow for SpamShield AI.

This module handles dataset loading, preprocessing, feature extraction,
model fitting, artifact persistence, and lightweight training summaries.
Evaluation-specific reporting belongs in ``src.evaluation``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import AppConfig, get_config
from src.constants import (
    CLASS_HAM,
    CLASS_SPAM,
    FEATURE_AVG_WORD2VEC,
    FEATURE_BOW,
    FEATURE_TFIDF,
    FEATURE_WORD2VEC,
    HAM_PROBABILITY_COLUMN,
    PREDICTION_COLUMN,
    SPAM_PROBABILITY_COLUMN,
    TARGET_COLUMN,
)
from src.feature_extraction import (
    FeatureArtifacts,
    FeatureExtractionError,
    fit_count_vectorizer,
    fit_tfidf_vectorizer,
    transform_with_word2vec,
)
from src.model import (
    FEATURE_TO_MODEL,
    ModelConfigurationError,
    build_model,
    get_default_model_for_feature,
    get_model_filename,
    infer_feature_for_model,
    load_model,
    save_model,
)
from src.preprocessing import build_corpus

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingResult:
    """Structured result returned after a successful training run."""

    feature_name: str
    model_name: str
    model_path: Path
    vectorizer_path: Path
    training_time_seconds: float
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    metrics: dict[str, float] | None = None


class TrainingError(RuntimeError):
    """Raised when the training workflow cannot be completed."""


@dataclass(slots=True)
class DatasetSplit:
    """Prepared train/test split for the spam classification task."""

    X_train_text: list[str]
    X_test_text: list[str]
    y_train: np.ndarray
    y_test: np.ndarray



def load_dataset(dataset_path: str | Path, *, text_column: str = "message", target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Load the SMS spam dataset from disk."""
    resolved_path = Path(dataset_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_path}")

    try:
        if resolved_path.suffix.lower() == ".csv":
            frame = pd.read_csv(resolved_path)
        else:
            frame = pd.read_csv(resolved_path, sep="\t", names=[target_column, text_column])
    except Exception as exc:  # pragma: no cover - dependent on external dataset format
        raise TrainingError(f"Unable to load dataset: {resolved_path}") from exc

    if text_column not in frame.columns or target_column not in frame.columns:
        raise TrainingError(
            f"Dataset must contain '{text_column}' and '{target_column}' columns. Found {list(frame.columns)}"
        )

    return frame[[target_column, text_column]].copy()



def preprocess_dataset(
    frame: pd.DataFrame,
    *,
    text_column: str = "message",
    target_column: str = TARGET_COLUMN,
    use_lemmatization: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Convert raw text and labels into a cleaned corpus and binary targets."""
    cleaned_text = build_corpus(frame[text_column].astype(str).tolist(), use_lemmatization=use_lemmatization)
    labels = frame[target_column].astype(str).str.lower().str.strip().map({"spam": CLASS_SPAM, "ham": CLASS_HAM})

    if labels.isna().any():
        invalid_labels = sorted(set(frame[target_column].astype(str)) - {"spam", "ham"})
        raise TrainingError(f"Unexpected label values found: {invalid_labels}")

    return cleaned_text, labels.astype(int).to_numpy()



def split_dataset(
    corpus: list[str],
    labels: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    """Split the corpus into training and test sets."""
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        corpus,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return DatasetSplit(
        X_train_text=list(X_train_text),
        X_test_text=list(X_test_text),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
    )



def _ensure_output_dirs(config: AppConfig) -> None:
    """Create artifact directories when they do not already exist."""
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    config.paths.vectorizers_dir.mkdir(parents=True, exist_ok=True)



def _fit_word2vec_model(tokenized_texts: list[list[str]], *, vector_size: int = 100, window: int = 5, min_count: int = 1) -> Any:
    """Train a Word2Vec model for sentence embedding generation."""
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TrainingError(
            "gensim is required for Word2Vec training. Add it to requirements.txt to enable this feature."
        ) from exc

    return Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
        seed=42,
    )



def _vectorize_training_data(
    feature_name: str,
    X_train_text: list[str],
    X_test_text: list[str],
    config: AppConfig,
) -> tuple[Any, Any, Any]:
    """Fit the appropriate feature extractor and transform train/test text."""
    if feature_name == FEATURE_BOW:
        vectorizer = fit_count_vectorizer(
            X_train_text,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
        )
        return vectorizer, vectorizer.transform(X_train_text).toarray(), vectorizer.transform(X_test_text).toarray()

    if feature_name == FEATURE_TFIDF:
        vectorizer = fit_tfidf_vectorizer(
            X_train_text,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
        )
        return vectorizer, vectorizer.transform(X_train_text).toarray(), vectorizer.transform(X_test_text).toarray()

    if feature_name in {FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC}:
        tokenized_train = [text.split() for text in X_train_text]
        tokenized_test = [text.split() for text in X_test_text]
        vectorizer = _fit_word2vec_model(tokenized_train)
        train_matrix = transform_with_word2vec(X_train_text, vectorizer, strategy=feature_name)
        test_matrix = transform_with_word2vec(X_test_text, vectorizer, strategy=feature_name)
        return vectorizer, train_matrix, test_matrix

    raise FeatureExtractionError(f"Unsupported feature extractor: {feature_name}")



def _evaluate_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    """Compute a compact set of classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics



def train_model(
    feature_name: str,
    dataset_path: str | Path | None = None,
    *,
    config: AppConfig | None = None,
    use_lemmatization: bool = False,
) -> TrainingResult:
    """Train and persist a SpamShield AI model for the selected feature extractor."""
    runtime_config = config or get_config()
    _ensure_output_dirs(runtime_config)

    selected_dataset_path = Path(dataset_path) if dataset_path is not None else runtime_config.dataset_path
    frame = load_dataset(selected_dataset_path)
    corpus, labels = preprocess_dataset(frame, use_lemmatization=use_lemmatization)
    split = split_dataset(
        corpus,
        labels,
        test_size=runtime_config.test_size,
        random_state=runtime_config.random_state,
    )

    vectorizer, X_train, X_test = _vectorize_training_data(feature_name, split.X_train_text, split.X_test_text, runtime_config)

    model_name = get_default_model_for_feature(feature_name)
    model = build_model(model_name)

    start_time = time.perf_counter()
    model.fit(X_train, split.y_train)
    training_time_seconds = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:  # pragma: no cover - estimator-specific behavior
            y_proba = None

    metrics = _evaluate_binary_predictions(split.y_test, y_pred, y_proba)

    vectorizer_filename = "word2vec.model" if feature_name in {FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC} else {
        FEATURE_BOW: "bow.pkl",
        FEATURE_TFIDF: "tfidf.pkl",
    }[feature_name]
    vectorizer_path = runtime_config.vectorizer_path(vectorizer_filename)
    model_path = runtime_config.model_path(get_model_filename(model_name))

    if feature_name in {FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC}:
        vectorizer.save(str(vectorizer_path))
    else:
        joblib.dump(vectorizer, vectorizer_path)

    save_model(model, str(model_path))

    LOGGER.info(
        "Trained %s with %s in %.2fs (accuracy=%.4f)",
        feature_name,
        model_name,
        training_time_seconds,
        metrics.get("accuracy", float("nan")),
    )

    return TrainingResult(
        feature_name=feature_name,
        model_name=model_name,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        training_time_seconds=training_time_seconds,
        train_shape=tuple(X_train.shape),
        test_shape=tuple(X_test.shape),
        metrics=metrics,
    )



def train_all_models(dataset_path: str | Path | None = None, *, config: AppConfig | None = None) -> list[TrainingResult]:
    """Train all supported feature/model combinations."""
    runtime_config = config or get_config()
    results: list[TrainingResult] = []
    for feature_name in (FEATURE_BOW, FEATURE_TFIDF, FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC):
        results.append(train_model(feature_name, dataset_path=dataset_path, config=runtime_config))
    return results



def load_trained_artifacts(feature_name: str, *, config: AppConfig | None = None) -> FeatureArtifacts:
    """Load the persisted vectorizer and model for a selected feature extractor."""
    runtime_config = config or get_config()
    model_name = get_default_model_for_feature(feature_name)
    vectorizer_filename = "word2vec.model" if feature_name in {FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC} else {
        FEATURE_BOW: "bow.pkl",
        FEATURE_TFIDF: "tfidf.pkl",
    }[feature_name]

    vectorizer_path = runtime_config.vectorizer_path(vectorizer_filename)
    model_path = runtime_config.model_path(get_model_filename(model_name))

    from src.feature_extraction import load_feature_artifacts

    return load_feature_artifacts(
        feature_name,
        vectorizer_path=vectorizer_path,
        model_path=model_path,
    )



def ensure_model_artifacts(feature_name: str, *, config: AppConfig | None = None) -> tuple[Path, Path]:
    """Return the expected artifact paths for a selected feature extractor."""
    runtime_config = config or get_config()
    model_name = get_default_model_for_feature(feature_name)
    vectorizer_filename = "word2vec.model" if feature_name in {FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC} else {
        FEATURE_BOW: "bow.pkl",
        FEATURE_TFIDF: "tfidf.pkl",
    }[feature_name]
    return runtime_config.vectorizer_path(vectorizer_filename), runtime_config.model_path(get_model_filename(model_name))
