"""Prediction page for the SpamShield AI Streamlit application."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import pandas as pd
import streamlit as st

from src.config import get_config
from src.constants import (
    FEATURE_DISPLAY_NAMES,
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

feature_name = st.session_state.get("active_feature", "tfidf")

st.title("Predict SMS Spam")
st.write(
    "Enter a message manually or upload a CSV file to score multiple messages with the selected feature extraction technique."
)

feature_col, info_col = st.columns([1.2, 1])
with feature_col:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(f"### Active Extraction: {FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)}")
    st.caption("Change the extraction option in the sidebar to automatically resolve and hot-reload matching vectorizers and classifiers.")
    st.markdown("</div>", unsafe_allow_html=True)
with info_col:
    st.metric("Supported Techniques", "4")
    st.metric("Default Feature", "TF-IDF")

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
        st.markdown("### Model Outputs")
        st.markdown(
            "- Spam / Ham label (large color-coded badge)\n"
            "- Confidence score (probability threshold metrics)\n"
            "- Class probabilities (Ham vs Spam distribution)\n"
            "- Processing latency (inference computation speed)"
        )
        st.caption("Results appear dynamically below once you click Predict Message.")

    if predict_button:
        if not message.strip():
            st.error("Please enter a message before predicting.")
        else:
            try:
                with st.spinner("Classifying message..."):
                    result = predict_message(message, feature_name, config=APP_CONFIG)
                
                st.toast("Prediction completed")
                
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
                
                result_cols = st.columns(4)
                badge = "result-spam" if result.predicted_label.lower() == "spam" else "result-ham"
                result_cols[0].markdown(f"<span class='{badge}'>{result.predicted_label.upper()}</span>", unsafe_allow_html=True)
                result_cols[1].metric("Confidence", safe_percentage(result.confidence))
                result_cols[2].metric("Spam Probability", safe_percentage(result.spam_probability))
                result_cols[3].metric("Ham Probability", safe_percentage(result.ham_probability))

                # Display progress bars for visual representation of probability
                st.markdown("#### Probability Calibration")
                st.write("Spam Probability")
                st.progress(float(result.spam_probability))
                st.write("Ham Probability")
                st.progress(float(result.ham_probability))

                latency_col, cleaned_col = st.columns([1, 2])
                latency_col.metric("Prediction Time", f"{result.prediction_time_seconds:.4f} s")
                cleaned_col.info(f"Cleaned message: {result.cleaned_text or '(empty after preprocessing)'}")

                with st.expander("Pipeline metadata details", expanded=False):
                    st.write({
                        "feature_name": result.feature_name,
                        "model_name": result.model_name,
                        "predicted_class": result.predicted_class,
                        "cleaned_text": result.cleaned_text,
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
                        file_name=f"spamshield_predictions_{feature_name}.csv",
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
