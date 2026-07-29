"""Exploratory Data Analysis (EDA) page for SpamShield AI."""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import get_config
from src.utils import apply_custom_css, render_sidebar

LOGGER = logging.getLogger(__name__)

APP_CONFIG = get_config()
ASSETS_DIR = APP_CONFIG.paths.assets_dir
DATASET_PATH = APP_CONFIG.paths.processed_data_dir / "SMSSpamCollection.txt"

# EDA content follows

def load_dataset() -> pd.DataFrame:
    """Load the raw text dataset, falling back to a structured mock if missing."""
    if DATASET_PATH.exists():
        try:
            return pd.read_csv(DATASET_PATH, sep="\t", names=["label", "message"])
        except Exception as exc:
            LOGGER.warning("Could not read real dataset at %s: %s", DATASET_PATH, exc)
            
    # Premium mockup fallback dataset matching real distribution style
    return pd.DataFrame(
        [
            {"label": "ham", "message": "Hey, are we still meeting for lunch today?"},
            {"label": "ham", "message": "Ok, I will text you when I get home."},
            {"label": "spam", "message": "URGENT! Your mobile was awarded a £2000 prize. Call 09066364589 now."},
            {"label": "ham", "message": "No worries, take your time!"},
            {"label": "spam", "message": "FREE ringtone! Reply YES to join. Txt STOP to exit."},
            {"label": "ham", "message": "Are you free this weekend? Let me know."},
            {"label": "spam", "message": "Private Account Statement. Call 08718738001. Show ID 9081."},
            {"label": "ham", "message": "Sounds good, see you later."},
            {"label": "spam", "message": "Congratulations! You won a free flight ticket. Text CLAIM."},
            {"label": "ham", "message": "Can you pick up some milk on your way back?"},
        ]
    )

def preprocess_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper length and count columns for plotting."""
    analysis = df.copy()
    analysis["message_length"] = analysis["message"].astype(str).str.len()
    analysis["word_count"] = analysis["message"].astype(str).apply(lambda x: len(x.split()))
    analysis["capital_ratio"] = analysis["message"].astype(str).apply(
        lambda text: (sum(1 for char in text if char.isupper()) / max(len(text), 1))
    )
    return analysis

# Load and prepare data
raw_data = load_dataset()
analysis_df = preprocess_for_eda(raw_data)

st.title("Exploratory Data Analysis")
st.write(
    "Analyze the textual characteristics and class distributions of the SMS Spam corpus."
)

st.caption(f"Loaded {len(analysis_df):,} rows from {DATASET_PATH.name if DATASET_PATH.exists() else 'mock fallback dataset'}.")

# Stat metrics cards
metrics = st.columns(4)
metrics[0].metric("Total Messages", f"{len(analysis_df):,}")
metrics[1].metric("Unique Words", f"{len(set(' '.join(analysis_df['message'].astype(str)).lower().split())):,}")
spam_pct = (analysis_df["label"].eq("spam").mean() * 100)
metrics[2].metric("Spam Percentage", f"{spam_pct:.1f}%")
metrics[3].metric("Avg Msg Length", f"{analysis_df['message_length'].mean():.1f} chars")

st.divider()

# Organize dashboard charts inside the 6 requested tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Overview", "Spam Distribution", "Word Clouds", "Message Statistics", "Top Words", "N-Grams"
])

# Process token frequencies for Word Cloud, Top Words, and N-Grams
all_tokens = []
all_bigrams = []
all_trigrams = []
stopwords = {
    "to", "i", "the", "a", "you", "is", "and", "in", "it", "for", "of", "your", "my", "me", "have", "on", 
    "that", "are", "at", "be", "with", "but", "ur", "im", "get", "so", "go", "can", "not", "or", "we", 
    "do", "was", "if", "will", "just", "no", "this", "up", "am", "how", "when", "out", "from", "up", "what"
}

for msg in analysis_df["message"].astype(str):
    cleaned = re.sub(r"[^a-zA-Z\s]", "", msg.lower())
    tokens = [w for w in cleaned.split() if w not in stopwords and len(w) > 1]
    all_tokens.extend(tokens)
    if len(tokens) >= 2:
        all_bigrams.extend([" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)])
    if len(tokens) >= 3:
        all_trigrams.extend([" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)])

token_counts = dict(Counter(all_tokens).most_common(50))

# Tab 1: Dataset Overview
with tab1:
    st.markdown("### Dataset Overview")
    st.write("Browse a sample of the processed corpus data below:")
    st.dataframe(
        analysis_df[["label", "message", "message_length", "word_count", "capital_ratio"]].head(15),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("### Feature Summary Description")
    st.dataframe(
        analysis_df[["message_length", "word_count", "capital_ratio"]].describe().T,
        use_container_width=True
    )

# Tab 2: Spam Distribution
with tab2:
    st.markdown("### Class Label Share")
    class_counts = analysis_df["label"].value_counts().reset_index()
    class_counts.columns = ["label", "count"]
    
    pie_cols = st.columns([1, 1.2])
    with pie_cols[0]:
        st.markdown("<br /><br />", unsafe_allow_html=True)
        st.dataframe(class_counts, use_container_width=True, hide_index=True)
        
        ham_count = class_counts.loc[class_counts["label"] == "ham", "count"].values[0] if "ham" in class_counts["label"].values else 0
        spam_count = class_counts.loc[class_counts["label"] == "spam", "count"].values[0] if "spam" in class_counts["label"].values else 0
        st.markdown(f"""
        - **Legitimate (Ham):** {ham_count:,} messages
        - **Junk/Spam:** {spam_count:,} messages
        - **Imbalance Ratio:** {ham_count/max(spam_count, 1):.1f}:1
        """)
    with pie_cols[1]:
        pie_fig = px.pie(
            class_counts,
            names="label",
            values="count",
            hole=0.4,
            template="plotly_dark",
            color="label",
            color_discrete_map={"ham": "#10b981", "spam": "#ef4444"},
            title="Distribution Share of Ham vs Spam Messages"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

# Tab 3: Word Clouds
with tab3:
    st.markdown("### Interactive Word Cloud")
    st.write("Visualizing the most frequent clean vocabulary words. Larger sizes indicate higher frequency:")
    
    # Render custom HTML/CSS word cloud matching dark mode theme
    def render_html_word_cloud(frequencies: dict[str, int]) -> str:
        if not frequencies:
            return "<div style='color:#94a3b8;'>No words available</div>"
        max_freq = max(frequencies.values())
        min_freq = min(frequencies.values())
        freq_range = max_freq - min_freq or 1
        
        colors = ["#60a5fa", "#3b82f6", "#10b981", "#34d399", "#f59e0b", "#fb7185", "#c084fc", "#a7f3d0", "#a5b4fc"]
        words_html = []
        import random
        random.seed(42)
        
        for word, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:50]:
            size = 12 + int(((freq - min_freq) / freq_range) * 26)
            color = random.choice(colors)
            words_html.append(f"<span style='font-size: {size}px; color: {color}; margin: 8px; display: inline-block; font-weight: 600; font-family: sans-serif;'>{word}</span>")
        return f"<div style='background: rgba(15, 23, 42, 0.5); padding: 2.5rem; border-radius: 16px; text-align: center; border: 1px solid rgba(148, 163, 184, 0.1);'>{ ''.join(words_html) }</div>"

    st.markdown(render_html_word_cloud(token_counts), unsafe_allow_html=True)

# Tab 4: Message Statistics
with tab4:
    st.markdown("### Character and Word Length Boxplots")
    stat_left, stat_right = st.columns(2)
    with stat_left:
        length_fig = px.histogram(
            analysis_df,
            x="message_length",
            color="label",
            nbins=40,
            marginal="box",
            template="plotly_dark",
            color_discrete_map={"ham": "#10b981", "spam": "#ef4444"},
            labels={"message_length": "Message Character Length"},
            title="Character Length Density by Class",
        )
        st.plotly_chart(length_fig, use_container_width=True)
    with stat_right:
        word_fig = px.histogram(
            analysis_df,
            x="word_count",
            color="label",
            nbins=40,
            marginal="box",
            template="plotly_dark",
            color_discrete_map={"ham": "#10b981", "spam": "#ef4444"},
            labels={"word_count": "Message Word Count"},
            title="Word Count Density by Class",
        )
        st.plotly_chart(word_fig, use_container_width=True)

# Tab 5: Top Words
with tab5:
    st.markdown("### Most Frequent Words")
    top_word_data = pd.DataFrame(Counter(all_tokens).most_common(20), columns=["word", "count"])
    
    top_words_fig = px.bar(
        top_word_data,
        x="count",
        y="word",
        orientation="h",
        template="plotly_dark",
        color="count",
        color_continuous_scale="Blues",
        labels={"word": "Vocabulary Word", "count": "Frequency"},
        title="Top 20 Most Frequent Clean Words",
    )
    top_words_fig.update_layout(yaxis_categoryorder="total ascending")
    st.plotly_chart(top_words_fig, use_container_width=True)

# Tab 6: N-Grams
with tab6:
    ngram_left, ngram_right = st.columns(2)
    with ngram_left:
        st.markdown("### Top Bigrams (2-grams)")
        bigram_data = pd.DataFrame(Counter(all_bigrams).most_common(15), columns=["bigram", "count"])
        bigram_fig = px.bar(
            bigram_data,
            x="count",
            y="bigram",
            orientation="h",
            template="plotly_dark",
            color="count",
            color_continuous_scale="Purples",
            labels={"bigram": "Bigram Sequence", "count": "Frequency"},
            title="Top 15 Most Common Bigrams",
        )
        bigram_fig.update_layout(yaxis_categoryorder="total ascending")
        st.plotly_chart(bigram_fig, use_container_width=True)
    with ngram_right:
        st.markdown("### Top Trigrams (3-grams)")
        trigram_data = pd.DataFrame(Counter(all_trigrams).most_common(15), columns=["trigram", "count"])
        trigram_fig = px.bar(
            trigram_data,
            x="count",
            y="trigram",
            orientation="h",
            template="plotly_dark",
            color="count",
            color_continuous_scale="Teals",
            labels={"trigram": "Trigram Sequence", "count": "Frequency"},
            title="Top 15 Most Common Trigrams",
        )
        trigram_fig.update_layout(yaxis_categoryorder="total ascending")
        st.plotly_chart(trigram_fig, use_container_width=True)

st.divider()
st.caption(
    "Visualizations are computed dynamically on load using Plotly templates suited for dark dashboard environments."
)