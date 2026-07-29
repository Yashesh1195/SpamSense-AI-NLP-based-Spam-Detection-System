"""Model evaluation utilities for SpamShield AI.

This module is responsible for computing classification metrics, confusion
matrices, ROC data, and report tables for trained spam detection models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluation output for a binary classifier."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: np.ndarray
    classification_report: dict[str, Any]
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    thresholds: np.ndarray | None = None


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot be completed."""



def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> EvaluationResult:
    """Compute standard binary classification metrics."""
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    roc_auc = None
    fpr = None
    tpr = None
    thresholds = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        except ValueError:
            LOGGER.exception("ROC computation failed")

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        classification_report=report,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )



def evaluation_to_dataframe(result: EvaluationResult) -> pd.DataFrame:
    """Convert a single evaluation result into a table-friendly dataframe."""
    return pd.DataFrame(
        [
            {
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "roc_auc": result.roc_auc,
            }
        ]
    )



def build_classification_report_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Convert a sklearn classification report dictionary into a dataframe."""
    frame = pd.DataFrame(report).transpose()
    frame.index.name = "label"
    return frame.reset_index()



def confusion_matrix_to_dataframe(matrix: np.ndarray, *, labels: tuple[str, str] = ("Ham", "Spam")) -> pd.DataFrame:
    """Convert a confusion matrix to a labeled dataframe."""
    if matrix.shape != (2, 2):
        raise EvaluationError(f"Expected a 2x2 confusion matrix, received shape {matrix.shape}")

    return pd.DataFrame(
        matrix,
        index=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        columns=[f"Predicted {labels[0]}", f"Predicted {labels[1]}"],
    )



def compare_models(evaluation_results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Create a comparison table across multiple trained models."""
    rows: list[dict[str, Any]] = []
    for name, result in evaluation_results.items():
        rows.append(
            {
                "model": name,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "roc_auc": result.roc_auc,
            }
        )
    return pd.DataFrame(rows).sort_values(by="f1_score", ascending=False).reset_index(drop=True)



def summarize_evaluation(result: EvaluationResult) -> str:
    """Return a compact textual summary of the metrics."""
    roc_auc_text = "n/a" if result.roc_auc is None else f"{result.roc_auc:.4f}"
    return (
        f"accuracy={result.accuracy:.4f}, precision={result.precision:.4f}, recall={result.recall:.4f}, "
        f"f1={result.f1_score:.4f}, roc_auc={roc_auc_text}"
    )
