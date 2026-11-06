"""
test_prediction.py
------------------
Unit tests for model loading, single prediction, probability bounds,
explanation generation, and batch predictions.
"""

import os
import pandas as pd
import pytest
from src.predict import load_model, predict_text, explain_prediction, predict_batch
from src.train_model import DEFAULT_MODEL_PATH


@pytest.fixture(scope="module")
def loaded_bundle():
    """Ensure model artifact exists and return loaded bundle."""
    if not os.path.exists(DEFAULT_MODEL_PATH):
        from src.train_model import train_sentiment_model
        train_sentiment_model()
    return load_model(DEFAULT_MODEL_PATH)


class TestPrediction:
    def test_load_model_structure(self, loaded_bundle):
        assert "pipeline" in loaded_bundle
        assert "classes" in loaded_bundle
        assert "metrics" in loaded_bundle
        assert "metadata" in loaded_bundle
        assert len(loaded_bundle["classes"]) >= 2

    def test_predict_text_valid(self, loaded_bundle):
        text = "This is a great product with outstanding quality!"
        res = predict_text(text, loaded_bundle)

        assert res["is_empty"] is False
        assert res["predicted_label"] in loaded_bundle["classes"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert isinstance(res["probabilities"], dict)

        # Probabilities sum to approximately 1.0
        prob_sum = sum(res["probabilities"].values())
        assert pytest.approx(prob_sum, abs=1e-3) == 1.0

    def test_predict_text_empty_input(self, loaded_bundle):
        res = predict_text("   ", loaded_bundle)
        assert res["is_empty"] is True
        assert res["predicted_label"] == "Unknown"
        assert res["confidence"] == 0.0

    def test_explain_prediction_structure(self, loaded_bundle):
        text = "The food arrived cold and customer service was rude."
        explanation = explain_prediction(text, loaded_bundle)

        assert "features_detected" in explanation
        assert "influential_terms" in explanation
        assert isinstance(explanation["features_detected"], list)
        assert isinstance(explanation["influential_terms"], list)

    def test_predict_batch_dataframe(self, loaded_bundle):
        sample_data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "review_text": [
                    "Super fast shipping and great communication!",
                    "Terrible quality, broke on the first day.",
                    "The package contains a charging cable and manual.",
                ],
            }
        )

        results = predict_batch(sample_data, text_col="review_text", bundle=loaded_bundle)

        assert len(results) == 3
        assert "Predicted Sentiment" in results.columns
        assert "Confidence (%)" in results.columns
        assert "Cleaned Text" in results.columns

        for label in results["Predicted Sentiment"]:
            assert label in loaded_bundle["classes"]

    def test_predict_batch_missing_column_raises_error(self, loaded_bundle):
        df = pd.DataFrame({"wrong_column": ["sample text"]})
        with pytest.raises(ValueError):
            predict_batch(df, text_col="review_text", bundle=loaded_bundle)
