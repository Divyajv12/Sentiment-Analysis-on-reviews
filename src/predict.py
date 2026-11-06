"""
predict.py
----------
Inference engine for Sentiment Analysis:
- Loads persistent model artifacts (Pipeline + Metadata)
- Single text sentiment prediction with genuine probabilities
- Feature attribution and explanation (top contributing n-grams)
- Batch dataset prediction with formatted output
"""

import os
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sentiment_model.joblib"
)


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Load saved model bundle containing scikit-learn pipeline,
    classes list, test metrics, and metadata.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please train the model first by running src/train_model.py or clicking 'Train Model' in the app."
        )

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid model bundle format. Expected dictionary with 'pipeline' key.")

    return bundle


def predict_text(text: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict sentiment and class probabilities for a single text input.

    Parameters
    ----------
    text : str
        Raw input string.
    bundle : dict
        Loaded model bundle containing pipeline and metadata.

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_label: str
        - confidence: float (0.0 to 1.0)
        - probabilities: dict of class -> float
        - cleaned_text: str
        - is_empty: bool
        - explanation: str
    """
    cleaned = clean_text(text)
    classes = bundle.get("classes", ["Negative", "Neutral", "Positive"])
    pipeline: Pipeline = bundle["pipeline"]

    if not cleaned or len(cleaned.strip()) == 0:
        return {
            "predicted_label": "Unknown",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in classes},
            "cleaned_text": "",
            "is_empty": True,
            "explanation": "No valid text detected. Please provide input containing words.",
        }

    # Model inference
    probas = pipeline.predict_proba([cleaned])[0]
    predicted_label = pipeline.predict([cleaned])[0]
    prob_dict = {str(cls): float(p) for cls, p in zip(pipeline.classes_, probas)}
    confidence = float(prob_dict[str(predicted_label)])

    # Human-friendly confidence interpretation
    if confidence >= 0.70:
        desc = f"Strong model probability ({confidence * 100:.1f}%) favoring {predicted_label}."
    elif confidence >= 0.50:
        desc = f"Moderate model probability ({confidence * 100:.1f}%) favoring {predicted_label}; text contains nuanced or balanced cues."
    else:
        desc = f"Low model probability ({confidence * 100:.1f}%) favoring {predicted_label}; language is ambiguous or evenly spread across classes."

    return {
        "predicted_label": str(predicted_label),
        "confidence": confidence,
        "probabilities": prob_dict,
        "cleaned_text": cleaned,
        "is_empty": False,
        "explanation": desc,
    }


def explain_prediction(
    text: str,
    bundle: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Inspect TF-IDF features and Logistic Regression weights for the input text.
    Identifies which words/n-grams contributed most strongly to the prediction.

    Parameters
    ----------
    text : str
        Raw input string.
    bundle : dict
        Loaded model bundle.
    top_k : int
        Number of top features to return.

    Returns
    -------
    dict
        Detailed word attributions per class and top positive/negative influencers.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"features_detected": [], "influential_terms": []}

    pipeline: Pipeline = bundle["pipeline"]
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Transform input text to tfidf vector
    vec = tfidf.transform([cleaned])
    feature_indices = vec.nonzero()[1]
    feature_names = tfidf.get_feature_names_out()

    if len(feature_indices) == 0:
        return {"features_detected": [], "influential_terms": []}

    detected_terms = [str(feature_names[i]) for i in feature_indices]
    tfidf_values = [float(vec[0, i]) for i in feature_indices]

    classes = [str(c) for c in clf.classes_]
    terms_info = []

    is_multiclass = len(classes) > 2

    for term, idx, val in zip(detected_terms, feature_indices, tfidf_values):
        term_data = {
            "term": str(term),
            "tfidf_score": round(float(val), 4),
            "class_weights": {},
        }
        for class_idx, class_name in enumerate(classes):
            if is_multiclass:
                weight = float(clf.coef_[class_idx, idx])
            else:
                weight = float(clf.coef_[0, idx]) if class_idx == 1 else -float(clf.coef_[0, idx])
            
            term_data["class_weights"][str(class_name)] = round(float(weight * val), 4)

        terms_info.append(term_data)

    # Sort terms by absolute maximum contribution across classes
    terms_info.sort(
        key=lambda x: max(abs(v) for v in x["class_weights"].values()),
        reverse=True,
    )

    return {
        "features_detected": detected_terms,
        "influential_terms": terms_info[:top_k],
    }


def predict_batch(
    df: pd.DataFrame,
    text_col: str,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    """
    Batch sentiment prediction over a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    text_col : str
        Column containing review text.
    bundle : dict
        Loaded model bundle.

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with predictions, probabilities, and confidence scores.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not present in provided DataFrame.")

    pipeline: Pipeline = bundle["pipeline"]
    classes = list(pipeline.classes_)

    results_df = df.copy()

    # Preprocess all rows
    cleaned_series = results_df[text_col].fillna("").astype(str).apply(clean_text)
    results_df["Cleaned Text"] = cleaned_series

    # Identify non-empty rows for vectorized prediction
    valid_mask = cleaned_series.str.len() > 0
    predictions = ["Unknown"] * len(results_df)
    confidences = [0.0] * len(results_df)

    prob_cols = {f"{c} Prob (%)": [0.0] * len(results_df) for c in classes}

    if valid_mask.any():
        valid_texts = cleaned_series[valid_mask].tolist()
        preds = pipeline.predict(valid_texts)
        probas = pipeline.predict_proba(valid_texts)

        valid_indices = results_df[valid_mask].index.tolist()
        for idx, pred, proba in zip(valid_indices, preds, probas):
            predictions[idx] = str(pred)
            max_prob = float(np.max(proba))
            confidences[idx] = round(max_prob * 100, 1)
            for c_idx, c_name in enumerate(classes):
                prob_cols[f"{c_name} Prob (%)"][idx] = round(float(proba[c_idx]) * 100, 1)

    results_df["Predicted Sentiment"] = predictions
    results_df["Confidence (%)"] = confidences
    for col_name, col_values in prob_cols.items():
        results_df[col_name] = col_values

    return results_df
