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

# System Operational Status Banner
st.markdown(
    """
    <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(96, 165, 250, 0.25); border-radius: 12px; padding: 0.65rem 1.25rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <div>
            <span style="color: #4ade80; font-size: 0.85rem; font-weight: 600;">● SYSTEM OPERATIONAL</span>
            <span style="color: #64748b; margin: 0 0.5rem;">|</span>
            <span style="color: #cbd5e1; font-size: 0.85rem; font-weight: 500;">Production NLP Spam Detection Engine</span>
        </div>
        <div>
            <span style="background: rgba(96, 165, 250, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); border-radius: 20px; padding: 0.25rem 0.75rem; font-size: 0.78rem; font-weight: 600;">v1.0.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.25, 1.0], vertical_alignment="center")
with hero_left:
    st.markdown(
        """
        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em; color: #60a5fa; font-weight: 700; margin-bottom: 0.25rem;">
            SpamSense AI • Machine Learning Platform
        </div>
        <h1 style="color: #f8fafc; font-size: 2.1rem; font-weight: 800; line-height: 1.25; margin: 0 0 1rem 0; border: none; padding: 0;">
            Intelligent Spam Detection Workspace
        </h1>
        <p style="color: #94a3b8; font-size: 0.98rem; line-height: 1.6; margin-bottom: 1.25rem;">
            A production-ready NLP application designed to detect spam SMS messages using count-based vectorization, TF-IDF weighting, and Word2Vec semantic embeddings mapped to multiple machine learning algorithms.
        </p>
        """,
        unsafe_allow_html=True,
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
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(96, 165, 250, 0.2); border-radius: 16px; padding: 1.4rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(148, 163, 184, 0.12); padding-bottom: 0.75rem;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Pipeline Specifications</div>
                    <span style="background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 12px; padding: 0.15rem 0.6rem; font-size: 0.75rem; font-weight: 600;">● Active System</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.85rem; margin-bottom: 1rem;">
                    <div style="background: rgba(30, 41, 59, 0.55); border-radius: 10px; padding: 0.75rem; border: 1px solid rgba(148, 163, 184, 0.08);">
                        <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Dataset Size</div>
                        <div style="color: #60a5fa; font-size: 1.2rem; font-weight: 700; margin-top: 0.15rem;">5,574 <span style="font-size: 0.75rem; color: #cbd5e1; font-weight: 400;">SMS</span></div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.55); border-radius: 10px; padding: 0.75rem; border: 1px solid rgba(148, 163, 184, 0.08);">
                        <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Max Vocabulary</div>
                        <div style="color: #c084fc; font-size: 1.2rem; font-weight: 700; margin-top: 0.15rem;">2,500 <span style="font-size: 0.75rem; color: #cbd5e1; font-weight: 400;">Terms</span></div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.55); border-radius: 10px; padding: 0.75rem; border: 1px solid rgba(148, 163, 184, 0.08);">
                        <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Estimator Models</div>
                        <div style="color: #38bdf8; font-size: 1.2rem; font-weight: 700; margin-top: 0.15rem;">4 <span style="font-size: 0.75rem; color: #cbd5e1; font-weight: 400;">Algorithms</span></div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.55); border-radius: 10px; padding: 0.75rem; border: 1px solid rgba(148, 163, 184, 0.08);">
                        <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Word2Vec Dim</div>
                        <div style="color: #f472b6; font-size: 1.2rem; font-weight: 700; margin-top: 0.15rem;">100 <span style="font-size: 0.75rem; color: #cbd5e1; font-weight: 400;">Vector Dim</span></div>
                    </div>
                </div>
                <div style="background: rgba(96, 165, 250, 0.08); border-radius: 8px; padding: 0.55rem 0.8rem; border: 1px solid rgba(96, 165, 250, 0.15); font-size: 0.78rem; color: #cbd5e1;">
                    ⚡ <strong>Pipeline:</strong> Normalization → Tokenization → Stopwords → Stemming → Vectorization
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

metric_cols = st.columns(4)
metric_cols[0].metric("Feature Techniques", "4", help="BoW, TF-IDF, Word2Vec, Average Word2Vec")
metric_cols[1].metric("Model Families", "4", help="NB, Logistic Regression, Random Forest, XGBoost")
metric_cols[2].metric("Pipeline Stages", "7", help="Input to confidence score")
metric_cols[3].metric("Deployment Ready", "Yes", help="Designed for Streamlit Community Cloud")

st.divider()

st.markdown("### Dataset & Stratification Settings")
strat_cols = st.columns(3)
strat_cols[0].metric("Spam Target Label", "1 (Spam)", help="Positive class for spam identification")
strat_cols[1].metric("Ham Target Label", "0 (Ham)", help="Negative class for legitimate messages")
strat_cols[2].metric("Test Split Ratio", safe_percentage(APP_CONFIG.test_size), help="Held-out evaluation split percentage")

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

# Future Roadmap (Commented out)
# st.markdown("### Future Roadmap")
# roadmap_items = [
#     "Optimize Word2Vec semantic word embeddings",
#     "Experiment with CBOW and Skip-Gram configurations",
#     "Introduce grid-search hyperparameter tuning",
#     "Integrate Explainable AI (LIME or SHAP charts)",
#     "Build cloud deployment pipeline with auto-scaling",
# ]
# for item in roadmap_items:
#     st.checkbox(item, value=False, disabled=True)

st.caption(
    "SpamSense AI — An end-to-end Machine Learning suite for real-time text classification, semantic embeddings, and interactive NLP analytics."
)
