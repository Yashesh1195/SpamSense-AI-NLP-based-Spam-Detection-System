"""Future work page for the SpamSense AI Streamlit application."""

from __future__ import annotations

import streamlit as st

from src.config import get_config
from src.utils import apply_custom_css, render_sidebar

APP_CONFIG = get_config()
ASSETS_DIR = APP_CONFIG.paths.assets_dir

# Future Work page content follows

st.title("Future Work & Roadmap")
st.write(
    "Track the implemented pipeline features and the upcoming enhancements planned for the SpamSense AI dashboard."
)

top_cols = st.columns(3)
top_cols[0].metric("Roadmap Themes", "4")
top_cols[1].metric("Current Focus", "Explainability")
top_cols[2].metric("Milestone Release", "v1.1.0")

st.divider()

roadmap_tab, details_tab = st.tabs(["Interactive Roadmap Dashboard", "Enhancement Logs"])

with roadmap_tab:
    st.markdown("### Feature Enhancements Overview")
    
    # 2-column grid layout for attractive feature cards
    card_cols = st.columns(2)
    
    with card_cols[0]:
        st.markdown(
            """
            <div class='panel'>
                <span class='badge' style='background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.25);'>COMPLETED</span>
                <span class='badge'>v1.0.0</span>
                <h4 style='margin-top: 0.5rem; margin-bottom: 0.25rem; color: #fff;'>Word2Vec Embeddings</h4>
                <p class="muted">Fit continuous Skip-gram architectures on the cleaned SMS corpus to generate 100-dimensional word representations capturing local contextual similarities.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class='panel'>
                <span class='badge' style='background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.25);'>COMPLETED</span>
                <span class='badge'>v1.0.0</span>
                <h4 style='margin-top: 0.5rem; margin-bottom: 0.25rem; color: #fff;'>Interactive Visualizations</h4>
                <p class="muted">Implemented Plotly-based character and word count histograms, class shares, bigram/trigram charts, and custom HTML/CSS responsive cloud containers.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class='panel'>
                <span class='badge' style='background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.25);'>IN PROGRESS</span>
                <span class='badge'>v1.1.0</span>
                <h4 style='margin-top: 0.5rem; margin-bottom: 0.25rem; color: #fff;'>Hyperparameter Optimization</h4>
                <p class="muted">Add Optuna-driven grid-search loops during training to automate regularization checks and find the best weights for Naive Bayes, LR, and RF models.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_cols[1]:
        st.markdown(
            """
            <div class='panel'>
                <span class='badge' style='background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.25);'>COMPLETED</span>
                <span class='badge'>v1.0.0</span>
                <h4 style='margin-top: 0.5rem; margin-bottom: 0.25rem; color: #fff;'>Average Word2Vec XGBoost</h4>
                <p class="muted">Aggregate word vector inputs into mean sentence embeddings, passed to non-linear XGBoost ensembles to achieve premium boundary separation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        

        st.markdown(
            """
            <div class='panel'>
                <span class='badge' style='background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.25);'>COMPLETED</span>
                <span class='badge'>v1.0.0</span>
                <h4 style='margin-top: 0.5rem; margin-bottom: 0.25rem; color: #fff;'>Cloud Deployment</h4>
                <p class="muted">Fully deployable structure on Streamlit Community Cloud with automated NLTK package downloads, and precompiled wheels setup.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class='panel'>
                <span class='badge' style='background: rgba(148, 163, 184, 0.1); color: #94a3b8; border-color: rgba(148, 163, 184, 0.2);'>PLANNED</span>
                <span class='badge'>v1.2.0</span>
                <h4 style='margin-top: 0.5rem; margin-bottom: 0.25rem; color: #fff;'>Explainability (LIME/SHAP)</h4>
                <p class="muted">Integrate Local Interpretable Model-agnostic Explanations (LIME) to highlight token-level contributions directly in the prediction interface.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with details_tab:
    st.markdown("### Operational Details and Enhancements")
    st.markdown(
        """
        - **Hyperparameter Optimization:** Planned cross-validation loops will write optimized parameter configurations directly into `models/` parameters metadata.
        - **Explainability Integrations:** The LIME explainer will color-code text input (blue for ham-indicative words, red for spam-indicative words) inside a custom prediction panel.
        - **Pipeline Scaling:** The pipeline is pre-configured to swap estimators dynamically as long as they adhere to the standard `fit`/`predict` API.
        """
    )

st.divider()
st.caption(
    "Roadmap statuses are updated automatically according to milestone completion criteria."
)