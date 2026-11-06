"""
train_model.py
--------------
Machine Learning pipeline for Sentiment Analysis:
- TF-IDF Vectorization (unigrams + bigrams)
- Logistic Regression Classifier
- Stratified Train/Test split
- Rigorous metrics calculation on unseen test data
- Artifact persistence via Joblib
"""

import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_dataset, clean_text


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_sentiment.csv"
)


def build_pipeline(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    c_param: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline with TF-IDF Vectorizer and Logistic Regression.
    Fitting the pipeline on training data guarantees zero data leakage.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    classifier = LogisticRegression(
        C=c_param,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the model on unseen test data.
    Computes genuine accuracy, weighted & macro precision, recall, F1,
    confusion matrix, and detailed per-class classification report.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, labels=classes, zero_division=0)

    return {
        "accuracy": acc,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classes": classes.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_str": report_str,
        "test_sample_count": len(y_test),
    }


def train_sentiment_model(
    data: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
    text_col: str = "text",
    label_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Full training workflow:
    1. Loads dataset from DataFrame or CSV.
    2. Cleans text and standardizes labels.
    3. Checks class balance and validates >= 2 classes.
    4. Performs stratified train/test split.
    5. Fits TF-IDF and Logistic Regression on training split.
    6. Evaluates rigorously on test split.
    7. Saves model bundle to disk via Joblib.

    Returns dictionary with metrics, sample sizes, and pipeline metadata.
    """
    if data is None:
        if csv_path is None:
            csv_path = DEFAULT_DATA_PATH
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = data.copy()

    # Clean dataset
    cleaned_df = clean_dataset(df, text_col=text_col, label_col=label_col)

    if len(cleaned_df) < 15:
        raise ValueError(
            f"Dataset too small ({len(cleaned_df)} valid samples). Minimum 15 samples required."
        )

    # Class balance check
    class_counts = cleaned_df[label_col].value_counts().to_dict()
    classes = np.array(sorted(class_counts.keys()))

    if len(classes) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 distinct classes. Found only: {list(classes)}"
        )

    # Check minimum samples per class for stratification
    min_class_count = min(class_counts.values())
    can_stratify = min_class_count >= 2

    X = cleaned_df["cleaned_text"].tolist()
    y = cleaned_df[label_col].tolist()

    stratify_target = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Convert to standard arrays for pipeline processing
    X_train = list(X_train)
    X_test = list(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Build and fit pipeline
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # Evaluate on unseen test data
    test_metrics = evaluate_model(pipeline, X_test, y_test, pipeline.classes_)

    # Compute train accuracy for diagnostics
    y_train_pred = pipeline.predict(X_train)
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_metrics["train_accuracy"] = train_acc
    test_metrics["train_sample_count"] = len(y_train)
    test_metrics["total_samples"] = len(cleaned_df)
    test_metrics["class_distribution"] = class_counts
    test_metrics["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Vocabulary size
    vocab = pipeline.named_steps["tfidf"].vocabulary_
    test_metrics["vocabulary_size"] = len(vocab)

    # Package model bundle
    bundle = {
        "pipeline": pipeline,
        "classes": pipeline.classes_.tolist(),
        "metrics": test_metrics,
        "metadata": {
            "model_name": "Logistic Regression + TF-IDF (N-grams 1-2)",
            "vectorizer_type": "TF-IDF",
            "classifier_type": "Logistic Regression (lbfgs, balanced)",
            "trained_at": test_metrics["trained_at"],
            "total_samples": len(cleaned_df),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "class_distribution": class_counts,
            "vocabulary_size": len(vocab),
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_weighted": test_metrics["f1_weighted"],
        },
    }

    # Ensure models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to disk
    joblib.dump(bundle, save_path)

    return {
        "bundle": bundle,
        "metrics": test_metrics,
        "save_path": save_path,
        "pipeline": pipeline,
    }


if __name__ == "__main__":
    print("Training Sentiment Analysis model with sample dataset...")
    result = train_sentiment_model()
    metrics = result["metrics"]
    print(f"\n Model successfully trained and saved to: {result['save_path']}")
    print(f"Total Samples: {metrics['total_samples']} (Train: {metrics['train_sample_count']}, Test: {metrics['test_sample_count']})")
    print(f"Classes: {metrics['classes']}")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test F1 (Weighted): {metrics['f1_weighted'] * 100:.2f}%")
    print("\nClassification Report:\n", metrics["classification_report_str"])
    print("Confusion Matrix:\n", np.array(metrics["confusion_matrix"]))
