"""Shared utility helpers for SpamShield AI.

This module contains cross-cutting helpers that do not belong to a single
pipeline stage: logging, file I/O, dataframe formatting, and Streamlit-safe
presentation helpers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent application-wide logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_timestamp() -> str:
    """Return a compact UTC timestamp suitable for filenames or logs."""
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def read_text_file(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file from disk."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")
    return resolved_path.read_text(encoding=encoding)


def write_text_file(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write text content to disk, creating the parent directory if required."""
    resolved_path = Path(path)
    ensure_directory(resolved_path.parent)
    resolved_path.write_text(content, encoding=encoding)
    return resolved_path


def load_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a tabular file into a dataframe using pandas."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Data file not found: {resolved_path}")
    return pd.read_csv(resolved_path, **kwargs)


def save_dataframe(dataframe: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Persist a dataframe to CSV."""
    resolved_path = Path(path)
    ensure_directory(resolved_path.parent)
    dataframe.to_csv(resolved_path, index=index)
    return resolved_path


def dataframe_preview(dataframe: pd.DataFrame, *, rows: int = 5) -> pd.DataFrame:
    """Return a small preview of a dataframe for UI rendering."""
    return dataframe.head(rows).copy()


def dataframe_shape_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return a compact summary of dataframe dimensions."""
    return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}


def safe_percentage(value: float | int | None, *, precision: int = 2) -> str:
    """Format a decimal probability as a percentage string."""
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{precision}f}%"


def to_serializable_object(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable equivalents."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_serializable_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable_object(item) for item in value]
    return value


def export_json_ready(data: Any) -> Any:
    """Return a JSON-friendly representation of arbitrary structured data."""
    return to_serializable_object(data)


def chunk_text(text: str, *, chunk_size: int = 200) -> list[str]:
    """Split text into roughly equal-sized chunks for UI display."""
    if not text:
        return []
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def truncate_text(text: str, *, max_length: int = 120) -> str:
    """Truncate text to a friendly display length."""
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def percentage_bar(value: float, *, width: int = 100) -> str:
    """Render a lightweight ASCII progress bar for non-Streamlit contexts."""
    clamped = max(0.0, min(1.0, float(value)))
    filled = int(round(clamped * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {clamped:.0%}"


def log_dataframe_info(dataframe: pd.DataFrame, *, name: str = "dataframe") -> None:
    """Log summary information about a dataframe."""
    LOGGER.info("%s shape=%s columns=%s", name, dataframe.shape, list(dataframe.columns))


def apply_custom_css() -> None:
    """Read the custom stylesheet and inject it into the Streamlit application shell."""
    import streamlit as st
    css_path = Path(__file__).resolve().parents[1] / "assets" / "styles.css"
    if css_path.exists():
        try:
            css_content = css_path.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as exc:
            LOGGER.warning("Could not apply custom CSS stylesheet: %s", exc)


def render_sidebar() -> str:
    """Render a consistent, professional, state-synchronized sidebar across all pages."""
    import streamlit as st
    from src.constants import (
        FEATURE_DISPLAY_NAMES,
        FEATURE_BOW,
        FEATURE_TFIDF,
        FEATURE_WORD2VEC,
        FEATURE_AVG_WORD2VEC
    )
    
    # Initialize session state for active extraction technique if missing
    if "active_feature" not in st.session_state:
        st.session_state.active_feature = "tfidf"

    # Project Branding
    st.sidebar.markdown(
        """
        <div style='text-align: center; margin-bottom: 1.25rem;'>
            <h2 style='color: #60a5fa; margin-bottom: 0; font-family: sans-serif; font-weight: 700; font-size: 1.6rem;'>SpamShield AI</h2>
            <div style='color: #94a3b8; font-size: 0.82rem; margin-top: 0.15rem;'>Intelligent Spam Filter Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.divider()
    
    # Feature Extractor Radio Selection
    st.sidebar.markdown("### Extraction Configuration")
    feature_options = [FEATURE_BOW, FEATURE_TFIDF, FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC]
    active_idx = feature_options.index(st.session_state.active_feature)
    
    selected_feature = st.sidebar.radio(
        "Active Extractor Option:",
        feature_options,
        format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x),
        index=active_idx,
        help="Choose the feature extraction strategy. The application resolves the model automatically.",
        label_visibility="collapsed"
    )
    
    # Rerun on changes to maintain strict state consistency
    if selected_feature != st.session_state.active_feature:
        st.session_state.active_feature = selected_feature
        st.rerun()
        
    st.sidebar.divider()
    
    # Model Metadata Resolution
    model_mapping = {
        FEATURE_BOW: ("Naive Bayes", "bow_nb.pkl", "MultinomialNB"),
        FEATURE_TFIDF: ("Logistic Regression", "tfidf_lr.pkl", "LogisticRegression"),
        FEATURE_WORD2VEC: ("Random Forest", "word2vec_rf.pkl", "RandomForestClassifier"),
        FEATURE_AVG_WORD2VEC: ("XGBoost", "avg_word2vec_xgb.pkl", "XGBClassifier"),
    }
    display_model, filename, class_name = model_mapping[selected_feature]
    
    st.sidebar.markdown("### Pipeline Metadata")
    st.sidebar.markdown(
        f"""
        <div style='background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 12px; padding: 0.8rem;'>
            <div style='font-size: 0.78rem; text-transform: uppercase; color: #60a5fa;'>Resolved Model</div>
            <div style='font-weight: 600; font-size: 0.95rem; color: #e5e7eb; margin-top: 0.1rem;'>{display_model}</div>
            <div style='font-size: 0.78rem; color: #94a3b8; margin-top: 0.35rem;'>Class: <code>{class_name}</code></div>
            <div style='font-size: 0.78rem; color: #94a3b8;'>File: <code>{filename}</code></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.divider()
    
    # Sidebar Footer
    st.sidebar.markdown(
        """
        <div style='font-size: 0.8rem; color: #94a3b8;'>
            <div><strong>Environment:</strong> Production (Cloud-Ready)</div>
            <div><strong>Theme:</strong> Slate Dark (AI/ML)</div>
            <div><strong>Version:</strong> v1.0.0</div>
            <div><strong>Developer:</strong> Yashesh Mehta</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return selected_feature
