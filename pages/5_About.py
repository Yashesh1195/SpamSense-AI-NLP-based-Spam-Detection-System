"""About page for the SpamSense AI Streamlit application."""

from __future__ import annotations

import streamlit as st

from src.config import get_config
from src.utils import apply_custom_css, render_sidebar

APP_CONFIG = get_config()
ASSETS_DIR = APP_CONFIG.paths.assets_dir

# About page content follows

st.title("About SpamSense AI")
st.write(
    "SpamSense AI is a modular spam detection application that turns the original notebook workflow into a reusable, maintainable Streamlit product."
)

top_cols = st.columns(3)
top_cols[0].metric("Supported Pipelines", "4")
top_cols[1].metric("UI Mode", "Multipage")
top_cols[2].metric("Deployment Target", "Streamlit Community Cloud")

st.divider()

overview_tab, architecture_tab, notes_tab, developer_tab = st.tabs(["Overview", "Architecture", "Notes", "Developer"])

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            """
            <div class='panel'>
                <h3>What this project does</h3>
                <p>It classifies short text messages as spam or ham and supports multiple feature extraction strategies:</p>
                <ul>
                    <li>Bag of Words with Naive Bayes</li>
                    <li>TF-IDF with Logistic Regression</li>
                    <li>Word2Vec with Random Forest</li>
                    <li>Average Word2Vec with XGBoost</li>
                </ul>
                <p>The Streamlit app exposes prediction, comparison, and EDA workflows through separate pages.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class='panel'>
                <h3>Why it is structured this way</h3>
                <p>The notebook logic was split into small modules so feature extraction, model training, evaluation, and prediction can evolve independently.</p>
                <p>This keeps the app testable, easier to deploy, and much simpler to extend with new models or datasets later.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Included runtime modules")
    module_rows = [
        {"Module Path": "src/config.py", "Responsibility": "Centralized filesystem and app configuration."},
        {"Module Path": "src/constants.py", "Responsibility": "Shared feature names, labels, and artifact paths."},
        {"Module Path": "src/preprocessing.py", "Responsibility": "Reusable cleaning and tokenization pipeline."},
        {"Module Path": "src/feature_extraction.py", "Responsibility": "Vectorizer and embedding helpers."},
        {"Module Path": "src/model.py", "Responsibility": "Model registry and serialization helpers."},
        {"Module Path": "src/train.py", "Responsibility": "Dataset loading, split, training, and artifact persistence."},
        {"Module Path": "src/predict.py", "Responsibility": "Single and batch inference helpers."},
        {"Module Path": "src/evaluation.py", "Responsibility": "Metrics, reports, and comparison utilities."},
        {"Module Path": "src/utils.py", "Responsibility": "Shared UI and data utilities."},
    ]
    st.dataframe(module_rows, use_container_width=True, hide_index=True)

with architecture_tab:
    st.markdown("### Architecture at a glance")
    st.code(
        """
Notebook data -> preprocessing -> feature extraction -> model training
                         |                      |
                         v                      v
                   model artifacts        prediction helpers
                         |                      |
                         v                      v
                    Streamlit pages <------ evaluation layer
        """.strip(),
        language="text",
    )
    st.markdown(
        """
        <div class='panel'>
            <h3>Design goals</h3>
            <ul>
                <li>Keep page files thin and presentation-focused.</li>
                <li>Centralize artifact naming and filesystem paths.</li>
                <li>Allow each pipeline to be trained, saved, and reused independently.</li>
                <li>Prefer safe fallbacks when artifacts are missing so the UI still renders.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with notes_tab:
    st.markdown("### Operational notes")
    st.markdown(
        """
        - The app expects trained model and vectorizer artifacts in the configured project folders.
        - If the real dataset is unavailable, the EDA page falls back to sample data to stay interactive.
        - The comparison page dynamically loads metrics from `models/evaluation_results.json` for real-time portfolio scoring.
        - Streamlit Community Cloud deployment will need a clean requirements file, a README, and the processed data file committed or generated during startup.
        """
    )

    st.markdown("### Future additions")
    future_items = [
        "Persist evaluation outputs to disk and load them on the comparison page.",
        "Add explicit training controls for retraining the pipelines from the UI.",
        "Introduce a model registry view for artifact selection and promotion.",
        "Capture lightweight telemetry for prediction latency and usage patterns.",
    ]
    for item in future_items:
        st.markdown(f"- {item}")

with developer_tab:
    st.markdown("### Developer Profile")
    dev_left, dev_right = st.columns([1.15, 1])
    with dev_left:
        st.markdown(
            """
            <div class='panel'>
                <h3>Yashesh Mehta</h3>
                <p><strong>Senior Machine Learning Engineer & Full-Stack Streamlit Developer</strong></p>
                <p>Specializing in Natural Language Processing (NLP), modular ML systems, and interactive data products.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dev_right:
        st.markdown(
            """
            <div class='panel'>
                <h3>Project Portals</h3>
                <ul>
                    <li><strong>GitHub Repository:</strong> <a href="https://github.com/Yashesh1195/SpamSense-AI-NLP-based-Spam-Detection-System" target="_blank">github.com/Yashesh1195/SpamSense-AI</a></li>
                    <li><strong>LinkedIn Profile:</strong> <a href="https://www.linkedin.com/in/yashesh-mehta/" target="_blank">linkedin.com/in/yashesh-mehta</a></li>
                    <li><strong>Professional Portfolio:</strong> <a href="https://yasheshmehta.me/" target="_blank">yasheshmehta.me</a></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<div class='panel'><strong>Project summary:</strong> SpamSense AI turns a notebook-based text classification demo into a structured application with reusable runtime code and separate UI surfaces for prediction, analysis, and comparison.</div>",
    unsafe_allow_html=True,
)