"""Streamlit entrypoint for SpamSense AI.

This module acts as the central router for the multi-page Streamlit application
using st.navigation.
"""

from __future__ import annotations

import logging
from pathlib import Path
import streamlit as st

from src.config import get_config
from src.utils import apply_custom_css, render_sidebar, configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

APP_CONFIG = get_config()
ASSETS_DIR = APP_CONFIG.paths.assets_dir

# Resolve custom favicon image path
FAVICON_PATH = ASSETS_DIR / "favicon.png" if (ASSETS_DIR / "favicon.png").exists() else (ASSETS_DIR / "favicon.ico")

# Set global page config (must be called first and only once)
st.set_page_config(
    page_title="SpamSense AI",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS stylesheet
apply_custom_css()

# Define the pages in the navigation menu with professional Material Icons
home_page = st.Page("pages/1_Home.py", title="Home", icon=":material/home:", default=True)
predict_page = st.Page("pages/2_Predict.py", title="Predict", icon=":material/mail:")
comparison_page = st.Page("pages/3_Model_Comparison.py", title="Model Comparison", icon=":material/bar_chart:")
eda_page = st.Page("pages/4_EDA.py", title="EDA", icon=":material/insights:")
about_page = st.Page("pages/5_About.py", title="About", icon=":material/info:")
future_page = st.Page("pages/6_Future_Work.py", title="Future Work", icon=":material/rocket:")

# Register navigation
pg = st.navigation([home_page, predict_page, comparison_page, eda_page, about_page, future_page])

# Render the consistent sidebar elements
render_sidebar()

# Run the active page
try:
    pg.run()
except Exception as exc:
    LOGGER.exception("Routing failed")
    st.error(f"Application routing failed: {exc}")
