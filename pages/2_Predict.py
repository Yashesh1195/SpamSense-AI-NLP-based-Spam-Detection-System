"""Prediction page for the SpamSense AI Streamlit application."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import pandas as pd
import streamlit as st

from src.config import get_config
from src.constants import (
    FEATURE_AVG_WORD2VEC,
    FEATURE_BOW,
    FEATURE_DISPLAY_NAMES,
    FEATURE_TFIDF,
    FEATURE_WORD2VEC,
    HAM_PROBABILITY_COLUMN,
    PREDICTION_COLUMN,
    SPAM_PROBABILITY_COLUMN,
)
from src.predict import PredictionError, predict_message, predict_messages
from src.utils import apply_custom_css, configure_logging, render_sidebar, safe_percentage, truncate_text

configure_logging()
LOGGER = logging.getLogger(__name__)

APP_CONFIG = get_config()
ASSETS_DIR = APP_CONFIG.paths.assets_dir

st.title("Predict Spam Message")
st.write(
    "Enter a message manually or upload a CSV file to score multiple messages with the selected feature extraction technique."
)

# Extraction Configuration & Pipeline Metadata (Symmetrical Card Layout)
feature_options = [FEATURE_BOW, FEATURE_TFIDF, FEATURE_WORD2VEC, FEATURE_AVG_WORD2VEC]
active_feature = st.session_state.get("active_feature", FEATURE_TFIDF)
active_idx = feature_options.index(active_feature) if active_feature in feature_options else 1

ctrl_col1, ctrl_col2 = st.columns([1.1, 1.0], vertical_alignment="top")

with ctrl_col1:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(148, 163, 184, 0.12); padding-bottom: 0.65rem; margin-bottom: 0.85rem;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">Extraction Configuration</div>
            <span style="background: rgba(96, 165, 250, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); border-radius: 12px; padding: 0.15rem 0.65rem; font-size: 0.75rem; font-weight: 600;">Active Strategy</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    rad_left, rad_right = st.columns([1.1, 1.4], vertical_alignment="top")
    
    with rad_left:
        selected_feature = st.radio(
            "Select Feature Extractor Strategy:",
            options=feature_options,
            format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x),
            index=active_idx,
            help="Choose the feature extraction strategy. The application resolves the model automatically.",
            label_visibility="collapsed",
        )
        if selected_feature != st.session_state.get("active_feature"):
            st.session_state.active_feature = selected_feature
            st.rerun()

    feature_info_map = {
        FEATURE_BOW: {
            "title": "Bag of Words (BoW)",
            "desc": "Count-based frequency matrix representing local word occurrences across the corpus.",
            "tag": "Sparse Frequency Matrix",
        },
        FEATURE_TFIDF: {
            "title": "TF-IDF Weighting",
            "desc": "Term frequency-inverse document frequency weighting to emphasize rare, distinctive tokens.",
            "tag": "Weighted Sparse Representation",
        },
        FEATURE_WORD2VEC: {
            "title": "Word2Vec Embeddings",
            "desc": "Continuous 100-dimensional neural word representations capturing semantic similarities.",
            "tag": "Semantic Vector Clustering",
        },
        FEATURE_AVG_WORD2VEC: {
            "title": "Average Word2Vec",
            "desc": "Mean-pooled token vector embeddings forming static, dense sentence representations.",
            "tag": "Dense Vector Pooling",
        },
    }
    info = feature_info_map.get(selected_feature, feature_info_map[FEATURE_TFIDF])

    with rad_right:
        st.markdown(
            f"""
            <div style="margin-top: 0.25rem;">
                <div style="color: #60a5fa; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.35rem;">{info['title']}</div>
                <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.45; margin-bottom: 0.65rem;">{info['desc']}</div>
                <span style="background: rgba(96, 165, 250, 0.12); color: #38bdf8; border: 1px solid rgba(96, 165, 250, 0.2); border-radius: 8px; padding: 0.2rem 0.6rem; font-size: 0.85rem; font-weight: 600;">{info['tag']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

feature_name = selected_feature
feature_display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)

model_mapping = {
    FEATURE_BOW: ("Naive Bayes", "bow_nb.pkl", "MultinomialNB", "Count Frequency Matrix"),
    FEATURE_TFIDF: ("Logistic Regression", "tfidf_lr.pkl", "LogisticRegression", "Term Frequency-Inverse Document Frequency"),
    FEATURE_WORD2VEC: ("Random Forest", "word2vec_rf.pkl", "RandomForestClassifier", "100-Dim Continuous Vector Space"),
    FEATURE_AVG_WORD2VEC: ("XGBoost", "avg_word2vec_xgb.pkl", "XGBClassifier", "Mean Pooled Dense Embeddings"),
}
model_title, model_file, model_class, model_desc = model_mapping.get(feature_name, ("Estimator", "model.pkl", "Classifier", "Vector Strategy"))

with ctrl_col2:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem; border-bottom: 1px solid rgba(148, 163, 184, 0.12); padding-bottom: 0.65rem;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">Pipeline Metadata</div>
            <span style="background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 12px; padding: 0.15rem 0.65rem; font-size: 0.75rem; font-weight: 600;">● Active Model</span>
        </div>
        <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; margin-bottom: 0.15rem;">Resolved Estimator</div>
        <div style="font-weight: 700; font-size: 1.3rem; color: #f8fafc; margin-bottom: 0.5rem;">{model_title}</div>
        <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 0.35rem;">Class: <code style="color: #38bdf8;">{model_class}</code></div>
        <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 0.35rem;">Artifact: <code style="color: #c084fc;">{model_file}</code></div>
        <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem; border-top: 1px solid rgba(148, 163, 184, 0.08); padding-top: 0.5rem;">Representation: <strong>{model_desc}</strong></div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

single_tab, batch_tab = st.tabs(["Single Message", "Batch CSV"])

with single_tab:
    left, right = st.columns([1.25, 1])
    with left:
        message = st.text_area(
            "Message Content",
            height=200,
            placeholder="Type or paste an SMS message here...",
            help="The pipeline preprocesses the text (lowercase, cleaning, stopwords, stemming/lemmatization) before scoring.",
        )
        predict_button = st.button("Predict Message", type="primary", use_container_width=True)

    with right:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 14px; padding: 1.25rem;">
                <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 0.75rem;">Model Output Specifications</h4>
                <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 1.2rem; margin-bottom: 0.5rem;">
                    <li style="margin-bottom: 0.4rem;"><strong>Classification Badge:</strong> Large color-coded SPAM or HAM label.</li>
                    <li style="margin-bottom: 0.4rem;"><strong>Confidence Score:</strong> Calibrated probability metric threshold.</li>
                    <li style="margin-bottom: 0.4rem;"><strong>Class Distribution:</strong> Visual Spam vs Ham probability calibration.</li>
                    <li><strong>Pipeline Latency:</strong> Real-time inference execution speed in seconds.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_button:
        if not message.strip():
            st.error("Please enter a message before predicting.")
        else:
            try:
                with st.spinner("Classifying message..."):
                    result = predict_message(message, feature_name, config=APP_CONFIG)
                
                st.toast("Prediction completed successfully")
                
                # Append to history in session state
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Input Text": truncate_text(message, max_length=60),
                    "Feature": FEATURE_DISPLAY_NAMES.get(result.feature_name, result.feature_name),
                    "Model": result.model_name,
                    "Prediction": result.predicted_label.upper(),
                    "Confidence": safe_percentage(result.confidence),
                    "Latency": f"{result.prediction_time_seconds:.4f}s"
                })
                
                st.divider()
                st.markdown("### Prediction Results")
                
                result_cols = st.columns(4)
                badge = "result-spam" if result.predicted_label.lower() == "spam" else "result-ham"
                result_cols[0].markdown(f"<span class='{badge}'>{result.predicted_label.upper()}</span>", unsafe_allow_html=True)
                result_cols[1].metric("Confidence", safe_percentage(result.confidence))
                result_cols[2].metric("Spam Probability", safe_percentage(result.spam_probability))
                result_cols[3].metric("Ham Probability", safe_percentage(result.ham_probability))

                # Display progress bars for visual representation of probability
                st.markdown("#### Class Probability Distribution")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.write(f"**Spam Probability:** {safe_percentage(result.spam_probability)}")
                    st.progress(float(result.spam_probability))
                with p_col2:
                    st.write(f"**Ham Probability:** {safe_percentage(result.ham_probability)}")
                    st.progress(float(result.ham_probability))

                latency_col, cleaned_col = st.columns([1, 2.5])
                latency_col.metric("Prediction Latency", f"{result.prediction_time_seconds:.4f} s")
                cleaned_col.info(f"**Cleaned Text:** {result.cleaned_text or '(empty after preprocessing)'}")

                with st.expander("Detailed Pipeline Metadata", expanded=False):
                    st.json({
                        "feature_name": result.feature_name,
                        "model_name": result.model_name,
                        "predicted_class": result.predicted_class,
                        "cleaned_text": result.cleaned_text,
                        "raw_input_text": message,
                    })
            except PredictionError as exc:
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI surface only
                LOGGER.exception("Prediction failed")
                st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.markdown("Upload a CSV file containing a text column. The pipeline will score each row and return a downloadable results file.")
    uploaded_file = st.file_uploader("CSV file upload:", type=["csv"])
    text_column_name = st.text_input("Text column header name:", value="message")
    run_batch = st.button("Predict Batch", type="primary", use_container_width=True)

    if run_batch:
        if uploaded_file is None:
            st.error("Upload a CSV file to run batch prediction.")
        else:
            try:
                batch_frame = pd.read_csv(uploaded_file)
                if text_column_name not in batch_frame.columns:
                    st.error(f"Column '{text_column_name}' was not found in the uploaded file.")
                else:
                    with st.spinner("Running batch predictions..."):
                        predictions = predict_messages(batch_frame[text_column_name].astype(str).tolist(), feature_name, config=APP_CONFIG)
                    
                    output_frame = pd.concat([batch_frame.reset_index(drop=True), predictions], axis=1)
                    st.success(f"Processed {len(output_frame)} rows")
                    st.dataframe(output_frame, use_container_width=True)

                    # Append to history in session state
                    if "prediction_history" not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        "Timestamp": time.strftime("%H:%M:%S"),
                        "Input Text": f"CSV Batch: {uploaded_file.name}",
                        "Feature": FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
                        "Model": "Multiple (Batch)",
                        "Prediction": f"{len(output_frame)} rows scored",
                        "Confidence": "N/A",
                        "Latency": "N/A"
                    })

                    csv_data = output_frame.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions CSV",
                        data=csv_data,
                        file_name=f"spamsense_predictions_{feature_name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except PredictionError as exc:
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI surface only
                LOGGER.exception("Batch prediction failed")
                st.error(f"Batch prediction failed: {exc}")

st.divider()

st.markdown("### Prediction History")
history_placeholder = st.container()

with history_placeholder:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.prediction_history:
        # Reverse to show newest first
        history_df = pd.DataFrame(st.session_state.prediction_history[::-1])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No prediction requests in this session yet.")

st.caption(
    "Tip: use the TF-IDF or BoW models for fast baseline scoring, and Word2Vec variants for semantic experiments."
)
