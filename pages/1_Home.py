"""Home page for the SpamSense AI Streamlit application."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.config import get_config
from src.constants import FEATURE_AVG_WORD2VEC, FEATURE_BOW, FEATURE_TFIDF, FEATURE_WORD2VEC
from src.utils import apply_custom_css, render_sidebar, safe_percentage

LOGGER = logging.getLogger(__name__)

APP_CONFIG = get_config()
PROJECT_ROOT = APP_CONFIG.paths.root
ASSETS_DIR = APP_CONFIG.paths.assets_dir
BANNER_PATH = ASSETS_DIR / "banner.png"

st.markdown("<div class='panel'>", unsafe_allow_html=True)
hero_left, hero_right = st.columns([1.1, 1.4], vertical_alignment="center")
with hero_left:
    st.markdown("<div style='font-size: 1.75rem; text-transform: uppercase; letter-spacing: 0.12em; color: #60a5fa; font-weight: 700;'>SpamSense AI</div>", unsafe_allow_html=True)
    st.title("Intelligent Spam Detection Workspace")
    st.write(
        "A production-ready NLP application designed to detect spam SMS messages using count-based vectorization, TF-IDF weighting, and Word2Vec semantic embeddings mapped to multiple machine learning algorithms."
    )
    st.markdown(
        "".join(
            f"<span class='badge'>{label}</span>"
            for label in ["BoW", "TF-IDF", "Word2Vec", "Average Word2Vec", "Streamlit Cloud Ready"]
        ),
        unsafe_allow_html=True,
    )
with hero_right:
    if BANNER_PATH.exists():
        st.image(str(BANNER_PATH), use_container_width=True)
    else:
        st.markdown(
            """
            <div style="background: rgba(2, 6, 23, 0.45); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 12px; padding: 2rem; text-align: center;">
                <h3 style="color: #60a5fa; margin-bottom: 0.5rem;">Spam Classification Platform</h3>
                <p class="muted">Decoupled ML pipeline for preprocessing, feature modeling, and inference.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

metric_cols = st.columns(4)
metric_cols[0].metric("Feature Techniques", "4", help="BoW, TF-IDF, Word2Vec, Average Word2Vec")
metric_cols[1].metric("Model Families", "4", help="NB, Logistic Regression, Random Forest, XGBoost")
metric_cols[2].metric("Pipeline Stages", "7", help="Input to confidence score")
metric_cols[3].metric("Deployment Ready", "Yes", help="Designed for Streamlit Community Cloud")

st.divider()

workflow_tab, architecture_tab, stack_tab = st.tabs(["Pipeline Workflow", "Project Architecture", "Technology Stack"])

with workflow_tab:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### NLP Pipeline Diagram")
        st.code(
            """Raw SMS Text
  ↓
Clean Text (Remove URLs, HTML tags, special chars)
  ↓
Tokenize (Split into word tokens)
  ↓
Stopword Removal (Filter NLTK English stopwords)
  ↓
Stemming / Lemmatization (Root form mapping)
  ↓
Vectorization (BoW / TF-IDF / Word2Vec embedding)
  ↓
Estimator Inference (Predict label & probabilities)
  ↓
Output Class & Confidence Badge""",
            language="text",
        )
    with col_b:
        st.markdown("### Feature Extraction Strategies")
        for feature_name, description in [
            (FEATURE_BOW, "Count-based frequency matrix representing local word presence."),
            (FEATURE_TFIDF, "Term frequency-inverse document frequency weighting to emphasize rare terms."),
            (FEATURE_WORD2VEC, "Continuous vector representations capturing word similarity and relations."),
            (FEATURE_AVG_WORD2VEC, "Mean pooling token embeddings into static sentence-level vectors."),
        ]:
            st.markdown(f"**{feature_name.upper()}**")
            st.caption(description)

with architecture_tab:
    arch_left, arch_right = st.columns(2)
    with arch_left:
        st.markdown("### Component Specifications")
        st.markdown(
            """
            <div class='panel'>
                <ul>
                    <li><strong>app.py:</strong> Entry point for Streamlit rendering.</li>
                    <li><strong>src/preprocessing.py:</strong> Thread-safe text cleaning and normalization.</li>
                    <li><strong>src/feature_extraction.py:</strong> Loads and fits text representation vectorizers.</li>
                    <li><strong>src/train.py:</strong> Executes estimators training workflows.</li>
                    <li><strong>src/predict.py:</strong> Serves real-time inference.</li>
                    <li><strong>src/evaluation.py:</strong> Evaluates metrics on held-out splits.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch_right:
        st.markdown("### Repository Layout")
        st.code(
            """SpamSense-AI/
├── app.py
├── pages/
├── src/
├── assets/
├── data/
├── models/
├── vectorizers/
├── notebooks/
├── README.md
└── requirements.txt""",
            language="text",
        )

with stack_tab:
    st.markdown("### Core Stack")
    technologies = ["Python", "Streamlit", "Pandas", "NumPy", "scikit-learn", "NLTK", "Plotly", "Joblib", "Gensim", "XGBoost"]
    st.markdown("".join(f"<span class='badge'>{item}</span>" for item in technologies), unsafe_allow_html=True)
    st.markdown("### Dataset Details")
    st.info(
        "Trained on the public SMS Spam Collection dataset containing 5,574 messages. "
        "The corpus features an imbalanced class share (86.6% Ham / 13.4% Spam)."
    )

st.divider()

roadmap_left, roadmap_right = st.columns(2)
with roadmap_left:
    st.markdown("### Future Roadmap")
    roadmap_items = [
        "Optimize Word2Vec semantic word embeddings",
        "Experiment with CBOW and Skip-Gram configurations",
        "Introduce grid-search hyperparameter tuning",
        "Integrate Explainable AI (LIME or SHAP charts)",
        "Build cloud deployment pipeline with auto-scaling",
    ]
    for item in roadmap_items:
        st.checkbox(item, value=False, disabled=True)

with roadmap_right:
    st.markdown("### Stratification Constants")
    st.metric("Spam Target Class", "1 (Spam)")
    st.metric("Ham Target Class", "0 (Ham)")
    st.metric("Train/Test Test Size", safe_percentage(APP_CONFIG.test_size))

st.caption(
    "SpamSense AI is structured as a portfolio-ready ML project demonstrating clean code practices."
)
