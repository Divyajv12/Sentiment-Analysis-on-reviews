"""
evaluation.py
-------------
Visualization and evaluation utilities for Sentiment Analysis.
Generates interactive Plotly figures styled with dark navy & glowing accent themes.
"""

from typing import Dict, List, Any
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Theme Color Palette
COLOR_POSITIVE = "#10b981"  # Emerald green
COLOR_NEGATIVE = "#ef4444"  # Coral red
COLOR_NEUTRAL = "#3b82f6"   # Electric blue
COLOR_PURPLE = "#8b5cf6"    # Vibrant violet
COLOR_BG = "#0b0f19"        # Deep navy
COLOR_CARD = "#131b2e"      # Card navy
COLOR_TEXT = "#f8fafc"      # Light text
COLOR_GRID = "#1e293b"      # Subtle gridline

SENTIMENT_COLORS = {
    "Positive": COLOR_POSITIVE,
    "Negative": COLOR_NEGATIVE,
    "Neutral": COLOR_NEUTRAL,
}


def _apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent sleek dark navy styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 16, "color": COLOR_TEXT, "family": "Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_TEXT, "family": "Inter, sans-serif"},
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color="#94a3b8"),
        ),
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """
    Generate an interactive Confusion Matrix heatmap with counts and percentages.
    """
    cm_arr = np.array(cm)
    total_samples = cm_arr.sum()
    
    # Text annotations combining count and percentage
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm_arr[i, j]
            pct = (count / total_samples) * 100 if total_samples > 0 else 0
            annotations.append(
                f"<b>{count}</b><br><span style='font-size:11px; color:#cbd5e1;'>({pct:.1f}%)</span>"
            )
    annotation_text = np.array(annotations).reshape(len(classes), len(classes))

    # Custom color scale: dark navy to luminous violet
    colorscale = [
        [0.0, "#0b1329"],
        [0.3, "#1e2952"],
        [0.6, "#4f46e5"],
        [1.0, "#9333ea"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=[f"Pred: {c}" for c in classes],
            y=[f"True: {c}" for c in classes],
            text=annotation_text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "#ffffff"},
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text="Count", font=dict(color=COLOR_TEXT, size=12)),
                tickfont=dict(color="#94a3b8"),
                len=0.8,
            ),
            hoverongaps=False,
        )
    )

    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return _apply_dark_theme(fig, "Confusion Matrix (Test Evaluation)")


def plot_classification_metrics(report_dict: Dict[str, Any], classes: List[str]) -> go.Figure:
    """
    Generate a grouped bar chart comparing Precision, Recall, and F1 per class.
    """
    class_labels = [c for c in classes if c in report_dict]
    precisions = [round(report_dict[c]["precision"] * 100, 1) for c in class_labels]
    recalls = [round(report_dict[c]["recall"] * 100, 1) for c in class_labels]
    f1s = [round(report_dict[c]["f1-score"] * 100, 1) for c in class_labels]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Precision",
            x=class_labels,
            y=precisions,
            marker_color="#06b6d4",
            text=[f"{v}%" for v in precisions],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Recall",
            x=class_labels,
            y=recalls,
            marker_color="#8b5cf6",
            text=[f"{v}%" for v in recalls],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="F1-Score",
            x=class_labels,
            y=f1s,
            marker_color="#10b981",
            text=[f"{v}%" for v in f1s],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 105], title="Score (%)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLOR_TEXT),
        ),
    )
    return _apply_dark_theme(fig, "Per-Class Performance Metrics")


def plot_sentiment_distribution(
    counts: Dict[str, int],
    title: str = "Sentiment Class Distribution",
) -> go.Figure:
    """
    Generate an interactive donut chart showing sentiment distribution.
    """
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(label, "#94a3b8") for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13, color="#ffffff"),
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color=COLOR_TEXT),
        ),
        annotations=[
            dict(
                text=f"<b>{sum(values)}</b><br><span style='font-size:11px; color:#94a3b8'>Samples</span>",
                x=0.5,
                y=0.5,
                font=dict(size=18, color=COLOR_TEXT),
                showarrow=False,
            )
        ],
    )
    return _apply_dark_theme(fig, title)


def plot_probability_bars(probabilities: Dict[str, float]) -> go.Figure:
    """
    Horizontal probability bar chart for single text analysis.
    """
    labels = list(probabilities.keys())
    # Sort for consistent display: Negative, Neutral, Positive if present
    preferred_order = ["Negative", "Neutral", "Positive"]
    sorted_labels = [l for l in preferred_order if l in labels] + [l for l in labels if l not in preferred_order]
    
    values = [probabilities[l] * 100 for l in sorted_labels]
    colors = [SENTIMENT_COLORS.get(l, COLOR_PURPLE) for l in sorted_labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=sorted_labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=[f"<b>{v:.1f}%</b>" for v in values],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=13),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Probability (%)", ticksuffix="%"),
        yaxis=dict(title=""),
        height=220,
        margin=dict(l=80, r=20, t=30, b=30),
    )
    return _apply_dark_theme(fig, "Predicted Sentiment Probabilities")


def get_metric_explanations() -> Dict[str, str]:
    """Educational dictionary explaining core ML evaluation concepts."""
    return {
        "Accuracy": (
            "The proportion of correct predictions out of all unseen test samples. "
            "While intuitive, accuracy alone can be misleading if a dataset has unbalanced classes."
        ),
        "Precision": (
            "Out of all samples the model predicted as positive/negative/neutral, how many were truly correct? "
            "High precision means fewer false alarms (False Positives)."
        ),
        "Recall": (
            "Out of all actual positive/negative/neutral samples, how many did the model capture? "
            "High recall means few actual examples were missed (False Negatives)."
        ),
        "F1-Score": (
            "The harmonic mean of Precision and Recall: 2 * (P * R) / (P + R). "
            "It gives a balanced assessment when you need to avoid both false positives and false negatives."
        ),
        "Confusion Matrix": (
            "A grid showing exact counts of True vs. Predicted classes. "
            "Diagonal cells represent correct predictions; off-diagonal cells highlight where the model got confused."
        ),
        "Stratified Train/Test Split": (
            "Splitting data such that the training and testing sets maintain the exact same proportion of positive, "
            "negative, and neutral reviews as the parent dataset, ensuring fair and unbiased evaluation."
        ),
    }
