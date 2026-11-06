"""
app.py
------
SentimentAI — Text Sentiment Analyzer
A professional, modern Streamlit dashboard for Sentiment Analysis using NLP,
TF-IDF Vectorization, and Logistic Regression.
"""

import os
import io
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import clean_text, clean_dataset
from src.train_model import train_sentiment_model, DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH
from src.predict import load_model, predict_text, explain_prediction, predict_batch
from src.evaluation import (
    plot_confusion_matrix,
    plot_classification_metrics,
    plot_sentiment_distribution,
    plot_probability_bars,
    get_metric_explanations,
    SENTIMENT_COLORS,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SentimentAI — Text Sentiment Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Dark Navy & Purple/Blue Accents)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Styles & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Brand Accent */
    .brand-gradient {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 82, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }

    /* Metric Cards */
    .metric-card {
        background: #131b2e;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: left;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(168, 85, 247, 0.4);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-subtitle {
        color: #64748b;
        font-size: 12px;
    }

    /* Sentiment Badges */
    .badge-positive {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 15px;
        display: inline-block;
    }
    .badge-negative {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 15px;
        display: inline-block;
    }
    .badge-neutral {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 15px;
        display: inline-block;
    }

    /* Prediction Result Box */
    .result-container {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 22px;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    /* Word Attribution Pills */
    .pill-positive {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px;
        display: inline-block;
        font-weight: 500;
    }
    .pill-negative {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px;
        display: inline-block;
        font-weight: 500;
    }
    .pill-neutral {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px;
        display: inline-block;
        font-weight: 500;
    }

    /* Disclaimer Callout */
    .callout-info {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #cbd5e1;
        margin-top: 14px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Load model bundle with Streamlit resource caching to prevent retraining on refresh."""
    if not os.path.exists(model_path):
        train_sentiment_model(save_path=model_path)
    return load_model(model_path)


@st.cache_data(show_spinner=False)
def get_cached_sample_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load sample dataset with caching."""
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame(columns=["text", "sentiment"])


# ---------------------------------------------------------
# Sidebar Navigation & Model Info
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 16px 0;">
            <h2 style="margin: 0; font-size: 22px;">⚡ <span class="brand-gradient">SentimentAI</span></h2>
            <p style="margin: 4px 0 0 0; font-size: 12px; color: #94a3b8;">NLP & Machine Learning System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize or verify session state for navigation
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home Dashboard"

    # Radio options for pages
    nav_options = [
        "Home Dashboard",
        "Single Text Analysis",
        "Batch Sentiment Analysis",
        "Model Performance",
        "Dataset Explorer",
    ]
    
    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state.nav_page)
        if st.session_state.nav_page in nav_options
        else 0,
        label_visibility="collapsed",
    )
    st.session_state.nav_page = selected_page

    st.markdown("---")

    # Load active model bundle
    try:
        model_bundle = get_cached_model()
        metadata = model_bundle.get("metadata", {})
        metrics = model_bundle.get("metrics", {})
        classes = model_bundle.get("classes", [])

        st.markdown(
            f"""
            <div style="background: #131b2e; border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 14px;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Active Model</div>
                <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-top: 2px;">Logistic Regression</div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">TF-IDF (Unigram + Bigram)</div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06);">
                    <span>Classes</span>
                    <span style="font-weight: 600; color: #a855f7;">{len(classes)} ({', '.join(classes)})</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1; padding-top: 4px;">
                    <span>Test Accuracy</span>
                    <span style="font-weight: 600; color: #10b981;">{metrics.get('accuracy', 0.0) * 100:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1; padding-top: 4px;">
                    <span>Vocabulary</span>
                    <span style="font-weight: 600; color: #38bdf8;">{metadata.get('vocabulary_size', 0):,} terms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_bundle = None

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 11px; color: #64748b; line-height: 1.5;">
            <b>Project:</b> Sentiment Analysis System<br>
            <b>Stack:</b> Streamlit • Scikit-Learn • TF-IDF<br>
            <b>Environment:</b> Python 3.11+
        </div>
        """,
        unsafe_allow_html=True,
    )


# Helper function to switch tabs via button
def go_to_page(page_name: str):
    st.session_state.nav_page = page_name
    st.rerun()


# ---------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# ---------------------------------------------------------
if st.session_state.nav_page == "Home Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 style="margin: 0 0 8px 0; font-size: 32px;">
                Welcome to <span class="brand-gradient">SentimentAI</span>
            </h1>
            <p style="margin: 0; font-size: 15px; color: #cbd5e1; max-width: 850px; line-height: 1.6;">
                An end-to-end Natural Language Processing system that automatically extracts, vectorizes, 
                and classifies user feedback into <b>Positive</b>, <b>Negative</b>, or <b>Neutral</b> sentiment.
                Powered by a scikit-learn TF-IDF pipeline and a multi-class Logistic Regression classifier.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_bundle:
        meta = model_bundle.get("metadata", {})
        met = model_bundle.get("metrics", {})

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Model Algorithm</div>
                    <div class="metric-value" style="font-size: 20px; color: #818cf8;">Logistic Regression</div>
                    <div class="metric-subtitle">L-BFGS • Balanced Weights</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Samples</div>
                    <div class="metric-value">{meta.get('total_samples', 0):,}</div>
                    <div class="metric-subtitle">Train: {meta.get('train_samples', 0)} | Test: {meta.get('test_samples', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            acc = met.get("accuracy", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Actual Test Accuracy</div>
                    <div class="metric-value" style="color: #10b981;">{acc:.1f}%</div>
                    <div class="metric-subtitle">Evaluated on {met.get('test_sample_count', 0)} unseen test reviews</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            f1 = met.get("f1_weighted", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Weighted F1-Score</div>
                    <div class="metric-value" style="color: #38bdf8;">{f1:.1f}%</div>
                    <div class="metric-subtitle">Harmonic mean of precision & recall</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # Overview Grid
        left_col, right_col = st.columns([3, 2])
        with left_col:
            st.markdown("### 🧠 How This System Works")
            st.markdown(
                """
                1. **Text Preprocessing:** Raw input text is stripped of HTML, hyperlinks, and special symbols. Standard English contractions (e.g. *don't* → *do not*) are expanded to rigorously preserve sentiment negation.
                2. **TF-IDF Vectorization:** Words and bi-grams (two-word sequences like *"not good"* or *"fast shipping"*) are weighted based on their frequency in the text relative to the overall dataset.
                3. **Logistic Regression Classification:** The model calculates class log-odds and applies the Softmax function to produce real, calibrated probabilities across **Positive**, **Negative**, and **Neutral** classes.
                4. **Zero Data Leakage:** The TF-IDF vectorizer is fitted strictly on the training partition. Unseen test reviews are never accessed during feature learning.
                """
            )

            # Quick Action CTAs
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔍 Try Single Text Analysis", use_container_width=True, type="primary"):
                    go_to_page("Single Text Analysis")
            with btn_col2:
                if st.button("📁 Run Batch CSV Analysis", use_container_width=True):
                    go_to_page("Batch Sentiment Analysis")

        with right_col:
            st.markdown("### 📊 Training Class Distribution")
            class_dist = meta.get("class_distribution", {})
            if class_dist:
                fig_dist = plot_sentiment_distribution(class_dist, "Current Training Balance")
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("Class distribution details are not available.")


# ---------------------------------------------------------
# PAGE 2: SINGLE TEXT ANALYSIS
# ---------------------------------------------------------
elif st.session_state.nav_page == "Single Text Analysis":
    st.markdown("## 🔍 Single Text Sentiment Analysis")
    st.markdown(
        "Analyze customer feedback, product reviews, tweets, or support tickets in real-time."
    )

    # Example Quick Buttons
    st.markdown("##### 💡 Try Sample Preset Reviews:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    preset_text = None
    with p_col1:
        if st.button("🌟 Positive Review", use_container_width=True):
            preset_text = "The battery life is exceptional and the display quality completely blew me away. Highly recommended!"
    with p_col2:
        if st.button("⚠️ Negative Review", use_container_width=True):
            preset_text = "The product arrived broken and customer service refused to issue a refund. Complete waste of money!"
    with p_col3:
        if st.button("📦 Neutral Statement", use_container_width=True):
            preset_text = "The package arrived on Tuesday via standard ground courier and includes a USB-C charging cord."
    with p_col4:
        if st.button("⚖️ Mixed / Nuanced Review", use_container_width=True):
            preset_text = "The camera takes great pictures during the day, but the battery drains fast and the device runs warm."

    # Text Input State Management
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if preset_text is not None:
        st.session_state.input_text = preset_text

    user_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.input_text,
        height=140,
        placeholder="Type or paste customer review text here (e.g. 'The shipping was super fast and the packaging was great!')...",
        help="Input can be any customer review, product feedback, or social media post.",
    )

    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Input"):
            st.session_state.input_text = ""
            st.rerun()

    if analyze_clicked or (user_input and preset_text is not None):
        if not user_input or len(user_input.strip()) == 0:
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing text sentiment..."):
                prediction = predict_text(user_input, model_bundle)

            if prediction["is_empty"]:
                st.warning("⚠️ No valid words detected after cleaning. Please enter descriptive text.")
            else:
                label = prediction["predicted_label"]
                conf = prediction["confidence"] * 100
                probs = prediction["probabilities"]
                cleaned = prediction["cleaned_text"]

                # Badge Class
                badge_class = f"badge-{label.lower()}"

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.markdown(
                        f"""
                        <div class="result-container">
                            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">
                                Predicted Sentiment
                            </div>
                            <div style="margin: 10px 0 14px 0;">
                                <span class="{badge_class}">{label.upper()}</span>
                            </div>
                            <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 4px;">
                                <b>Model Probability:</b> <span style="color: #f8fafc; font-weight: 700;">{conf:.1f}%</span>
                            </div>
                            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                                {prediction['explanation']}
                            </div>
                            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 12px; color: #64748b;">
                                <b>Cleaned NLP Input:</b> <i>"{cleaned}"</i>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    fig_bars = plot_probability_bars(probs)
                    st.plotly_chart(fig_bars, use_container_width=True)

                # Feature Attribution & Explanation Section
                st.markdown("### 🔬 Word Attribution & Feature Analysis")
                explanation = explain_prediction(user_input, model_bundle, top_k=6)
                influential_terms = explanation.get("influential_terms", [])

                if influential_terms:
                    st.markdown(
                        "The table below shows detected n-grams in your input and how they shifted the Logistic Regression odds:"
                    )
                    term_rows = []
                    for t in influential_terms:
                        term_rows.append(
                            {
                                "Detected Term": t["term"],
                                "TF-IDF Weight": t["tfidf_score"],
                                "Negative Influence": t["class_weights"].get("Negative", 0.0),
                                "Neutral Influence": t["class_weights"].get("Neutral", 0.0),
                                "Positive Influence": t["class_weights"].get("Positive", 0.0),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(term_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "No specific high-weighted vocabulary terms from the training set were matched. "
                        "The prediction was determined by baseline class priors and sublinear features."
                    )

                # Educational Disclaimer
                st.markdown(
                    """
                    <div class="callout-info">
                        <b>Note on Confidence Scores:</b> Displayed percentages represent model class probabilities 
                        derived from the Softmax transform of Logistic Regression logits. They reflect statistical 
                        correlation with the training data, not absolute certainty.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------
# PAGE 3: BATCH SENTIMENT ANALYSIS
# ---------------------------------------------------------
elif st.session_state.nav_page == "Batch Sentiment Analysis":
    st.markdown("## 📁 Batch Sentiment Analysis")
    st.markdown(
        "Upload a CSV file containing multiple reviews to classify them in bulk, view aggregate statistics, and export predictions."
    )

    batch_source = st.radio(
        "Choose Data Source:",
        ["Upload Custom CSV File", "Load Demo Batch Reviews (20 Pre-packaged Reviews)"],
        horizontal=True,
    )

    batch_df: Optional[pd.DataFrame] = None

    if batch_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload a CSV file containing customer feedback or reviews:",
            type=["csv"],
            help="The CSV must contain at least one column with text strings.",
        )
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if batch_df.empty:
                    st.error("The uploaded CSV file is empty. Please upload a file containing rows of text.")
                    batch_df = None
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                batch_df = None
    else:
        demo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "batch_test_reviews.csv"
        )
        if os.path.exists(demo_path):
            batch_df = pd.read_csv(demo_path)
            st.success("Loaded pre-packaged demo dataset with 20 varied customer reviews across 6 industries.")
        else:
            st.warning("Demo batch file not found on disk.")

    if batch_df is not None:
        st.markdown(f"**Preview of Loaded Data ({len(batch_df)} rows):**")
        st.dataframe(batch_df.head(5), use_container_width=True)

        # Select text column
        text_columns = list(batch_df.columns)
        default_index = 0
        for i, col in enumerate(text_columns):
            if "review" in col.lower() or "text" in col.lower() or "comment" in col.lower():
                default_index = i
                break

        selected_col = st.selectbox(
            "Select the text column to analyze:",
            text_columns,
            index=default_index,
        )

        if st.button("🚀 Run Batch Analysis", type="primary"):
            with st.spinner("Processing batch predictions with NLP pipeline..."):
                results_df = predict_batch(batch_df, selected_col, model_bundle)

            st.success(f"Successfully processed {len(results_df)} records!")

            # Summary Metrics
            pred_counts = results_df["Predicted Sentiment"].value_counts().to_dict()
            total_preds = len(results_df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Records</div>
                        <div class="metric-value">{total_preds:,}</div>
                        <div class="metric-subtitle">Processed Rows</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                pos_count = pred_counts.get("Positive", 0)
                pos_pct = (pos_count / total_preds) * 100 if total_preds > 0 else 0
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Positive Reviews</div>
                        <div class="metric-value" style="color: #10b981;">{pos_count} <span style="font-size: 16px;">({pos_pct:.1f}%)</span></div>
                        <div class="metric-subtitle">Satisfied sentiments</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                neu_count = pred_counts.get("Neutral", 0)
                neu_pct = (neu_count / total_preds) * 100 if total_preds > 0 else 0
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Neutral Reviews</div>
                        <div class="metric-value" style="color: #3b82f6;">{neu_count} <span style="font-size: 16px;">({neu_pct:.1f}%)</span></div>
                        <div class="metric-subtitle">Factual / balanced</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col4:
                neg_count = pred_counts.get("Negative", 0)
                neg_pct = (neg_count / total_preds) * 100 if total_preds > 0 else 0
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Negative Reviews</div>
                        <div class="metric-value" style="color: #ef4444;">{neg_count} <span style="font-size: 16px;">({neg_pct:.1f}%)</span></div>
                        <div class="metric-subtitle">Dissatisfied / critical</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # Distribution Visualizer
            chart_col1, chart_col2 = st.columns([1, 1])
            with chart_col1:
                fig_batch_dist = plot_sentiment_distribution(pred_counts, "Predicted Sentiment Breakdown")
                st.plotly_chart(fig_batch_dist, use_container_width=True)
            with chart_col2:
                # Bar Chart
                labels = list(pred_counts.keys())
                vals = list(pred_counts.values())
                colors = [SENTIMENT_COLORS.get(l, "#8b5cf6") for l in labels]
                import plotly.graph_objects as go
                fig_bar = go.Figure(
                    go.Bar(
                        x=labels,
                        y=vals,
                        marker_color=colors,
                        text=[f"{v} ({v/total_preds*100:.1f}%)" for v in vals],
                        textposition="auto",
                    )
                )
                fig_bar.update_layout(
                    title="Predicted Volume by Class",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f8fafc"},
                    yaxis=dict(title="Review Count", gridcolor="#1e293b"),
                    xaxis=dict(gridcolor="#1e293b"),
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Results Table & Download Button
            st.markdown("### 📋 Prediction Results Table")
            st.dataframe(results_df, use_container_width=True)

            # Convert to CSV for download
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_data,
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                type="primary",
            )


# ---------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------
elif st.session_state.nav_page == "Model Performance":
    st.markdown("## 📈 Model Performance & Evaluation Metrics")
    st.markdown(
        "All metrics below were computed on an **unseen, stratified test holdout set** that the model was never trained on."
    )

    if model_bundle:
        metrics = model_bundle.get("metrics", {})
        classes = model_bundle.get("classes", [])
        report_dict = metrics.get("classification_report_dict", {})
        cm = metrics.get("confusion_matrix", [])

        # Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            acc = metrics.get("accuracy", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Test Accuracy</div>
                    <div class="metric-value" style="color: #10b981;">{acc:.1f}%</div>
                    <div class="metric-subtitle">Overall test correctness</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            prec = metrics.get("precision_weighted", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Precision (Weighted)</div>
                    <div class="metric-value" style="color: #06b6d4;">{prec:.1f}%</div>
                    <div class="metric-subtitle">False alarm minimization</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            rec = metrics.get("recall_weighted", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Recall (Weighted)</div>
                    <div class="metric-value" style="color: #8b5cf6;">{rec:.1f}%</div>
                    <div class="metric-subtitle">Target capture rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            f1_w = metrics.get("f1_weighted", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">F1-Score (Weighted)</div>
                    <div class="metric-value" style="color: #ec4899;">{f1_w:.1f}%</div>
                    <div class="metric-subtitle">Class-weighted balance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col5:
            f1_m = metrics.get("f1_macro", 0.0) * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Macro F1-Score</div>
                    <div class="metric-value" style="color: #f59e0b;">{f1_m:.1f}%</div>
                    <div class="metric-subtitle">Equal-weight per class</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Charts: Confusion Matrix & Classification Metrics
        g_col1, g_col2 = st.columns([1, 1])
        with g_col1:
            if cm:
                fig_cm = plot_confusion_matrix(cm, classes)
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix data unavailable.")

        with g_col2:
            if report_dict:
                fig_metrics = plot_classification_metrics(report_dict, classes)
                st.plotly_chart(fig_metrics, use_container_width=True)
            else:
                st.info("Per-class metrics unavailable.")

        # Detailed Classification Report Table
        st.markdown("### 📋 Detailed Classification Report")
        if report_dict:
            report_rows = []
            for c in classes:
                if c in report_dict:
                    report_rows.append(
                        {
                            "Sentiment Class": c,
                            "Precision": f"{report_dict[c]['precision'] * 100:.1f}%",
                            "Recall": f"{report_dict[c]['recall'] * 100:.1f}%",
                            "F1-Score": f"{report_dict[c]['f1-score'] * 100:.1f}%",
                            "Test Support": int(report_dict[c]["support"]),
                        }
                    )
            # Add summary rows
            if "weighted avg" in report_dict:
                report_rows.append(
                    {
                        "Sentiment Class": "Weighted Average",
                        "Precision": f"{report_dict['weighted avg']['precision'] * 100:.1f}%",
                        "Recall": f"{report_dict['weighted avg']['recall'] * 100:.1f}%",
                        "F1-Score": f"{report_dict['weighted avg']['f1-score'] * 100:.1f}%",
                        "Test Support": int(report_dict['weighted avg']['support']),
                    }
                )
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        # Educational Metric Explanations
        st.markdown("### 🎓 Beginner-Friendly Guide to ML Evaluation")
        explanations = get_metric_explanations()
        with st.expander("📖 What do Accuracy, Precision, Recall, and F1 really mean?", expanded=False):
            for term, explanation in explanations.items():
                st.markdown(f"**{term}:** {explanation}")

        # Dataset Size Limitations Explanation
        st.markdown(
            """
            <div class="callout-info">
                <b>Realistic Sample Size Discussion:</b>
                The sample dataset included in this project contains ~300 curated reviews. 
                In classical NLP, Logistic Regression models trained on hundreds of samples 
                achieve realistic test accuracies around 60%–70% for 3-class sentiment because human language 
                contains vast vocabulary, sarcasm, and domain nuance. 
                When scaling to larger datasets (e.g., 50,000+ samples from IMDB or Yelp), models typically 
                approach 85%+ accuracy. You can upload and train on your own large dataset in the <b>Dataset Explorer</b> tab!
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE 5: DATASET EXPLORER & RETRAINING
# ---------------------------------------------------------
elif st.session_state.nav_page == "Dataset Explorer":
    st.markdown("## 🗂️ Dataset Explorer & Custom Model Retraining")
    st.markdown(
        "Inspect the training data, filter sentiment classes, or train a new model using your own custom labeled dataset."
    )

    tab_explore, tab_retrain = st.tabs(["📊 Explore Current Dataset", "🔄 Retrain with Custom Dataset"])

    with tab_explore:
        df_sample = get_cached_sample_data()

        if not df_sample.empty:
            total_rows = len(df_sample)
            null_count = df_sample.isnull().sum().sum()
            dup_count = df_sample.duplicated(subset=["text"]).sum() if "text" in df_sample.columns else 0
            counts = df_sample["sentiment"].value_counts().to_dict() if "sentiment" in df_sample.columns else {}

            # Stats cards
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Rows</div>
                        <div class="metric-value">{total_rows:,}</div>
                        <div class="metric-subtitle">Sample reviews in CSV</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Duplicate Texts</div>
                        <div class="metric-value" style="color: {'#10b981' if dup_count == 0 else '#f59e0b'};">{dup_count}</div>
                        <div class="metric-subtitle">Filtered during training</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Missing Values</div>
                        <div class="metric-value" style="color: {'#10b981' if null_count == 0 else '#ef4444'};">{null_count}</div>
                        <div class="metric-subtitle">Null entries in CSV</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Classes</div>
                        <div class="metric-value" style="color: #8b5cf6;">{len(counts)}</div>
                        <div class="metric-subtitle">{', '.join(counts.keys())}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # Filter & Search Table
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                filter_choice = st.selectbox(
                    "Filter by Sentiment Class:",
                    ["All Classes"] + list(counts.keys()),
                )
            with f_col2:
                search_query = st.text_input("Search reviews by keyword:", placeholder="e.g. battery, food, movie...")

            filtered_df = df_sample.copy()
            if filter_choice != "All Classes":
                filtered_df = filtered_df[filtered_df["sentiment"] == filter_choice]
            if search_query:
                filtered_df = filtered_df[
                    filtered_df["text"].str.contains(search_query, case=False, na=False)
                ]

            st.dataframe(filtered_df, use_container_width=True)
            st.caption(f"Showing {len(filtered_df)} of {total_rows} records.")

        else:
            st.warning("Sample dataset not found.")

    with tab_retrain:
        st.markdown("### 🛠️ Train a New Sentiment Model")
        st.markdown(
            """
            Upload a CSV containing your own labeled data. The file must contain:
            1. A column with text reviews
            2. A column with sentiment labels (**Positive**, **Negative**, and optionally **Neutral**)
            """
        )

        custom_file = st.file_uploader(
            "Upload labeled training dataset (CSV):",
            type=["csv"],
            key="retrain_uploader",
        )

        if custom_file is not None:
            try:
                custom_df = pd.read_csv(custom_file)
                st.write("**Dataset Preview:**")
                st.dataframe(custom_df.head(4), use_container_width=True)

                cols = list(custom_df.columns)
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    text_col_select = st.selectbox("Select Text Review Column:", cols, index=0)
                with r_col2:
                    label_col_select = st.selectbox(
                        "Select Sentiment Label Column:",
                        cols,
                        index=1 if len(cols) > 1 else 0,
                    )

                # Validation checks
                cleaned_preview = None
                valid_for_training = False

                try:
                    cleaned_preview = clean_dataset(
                        custom_df, text_col=text_col_select, label_col=label_col_select
                    )
                    unique_classes = cleaned_preview[label_col_select].unique()

                    if len(unique_classes) < 2:
                        st.error(
                            f"❌ Dataset must have at least 2 distinct classes. Found: {list(unique_classes)}"
                        )
                    elif len(cleaned_preview) < 15:
                        st.error(
                            f"❌ Dataset too small ({len(cleaned_preview)} valid records). At least 15 valid samples required."
                        )
                    else:
                        valid_for_training = True
                        st.success(
                            f"✅ Dataset validated! Contains {len(cleaned_preview)} valid records across classes: {list(unique_classes)}"
                        )
                except Exception as val_err:
                    st.error(f"Validation error: {val_err}")

                if valid_for_training:
                    st.warning(
                        "⚠️ Clicking 'Train New Model' will train on your dataset and safely replace the active model."
                    )
                    if st.button("🚀 Train New Model", type="primary"):
                        with st.spinner("Training new model with scikit-learn pipeline..."):
                            try:
                                result = train_sentiment_model(
                                    data=custom_df,
                                    text_col=text_col_select,
                                    label_col=label_col_select,
                                )
                                # Clear cache so new model is immediately active
                                st.cache_resource.clear()
                                st.success("🎉 New model successfully trained, evaluated, and saved!")

                                new_metrics = result["metrics"]
                                st.markdown(
                                    f"""
                                    - **New Test Accuracy:** {new_metrics['accuracy'] * 100:.2f}%
                                    - **New Weighted F1:** {new_metrics['f1_weighted'] * 100:.2f}%
                                    - **Training Samples:** {new_metrics['train_sample_count']} (Test: {new_metrics['test_sample_count']})
                                    - **Active Classes:** {new_metrics['classes']}
                                    """
                                )
                                if st.button("Refresh Dashboard"):
                                    st.rerun()

                            except Exception as train_err:
                                st.error(f"Training failed: {train_err}. The previous model was preserved.")

            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
