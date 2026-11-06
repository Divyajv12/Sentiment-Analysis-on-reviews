"""
test_preprocessing.py
---------------------
Unit tests for text preprocessing and dataset cleaning functions.
"""

import pandas as pd
import pytest
from src.preprocessing import (
    clean_text,
    expand_contractions,
    standardize_sentiment_label,
    clean_dataset,
)


class TestTextPreprocessing:
    def test_clean_text_basic(self):
        raw = "This is a GREAT product!"
        cleaned = clean_text(raw)
        assert cleaned == "this is a great product"

    def test_clean_text_none_and_empty(self):
        assert clean_text(None) == ""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(123) == ""

    def test_clean_text_removes_urls(self):
        raw = "Check this review at https://example.com/item and www.shop.com for details!"
        cleaned = clean_text(raw)
        assert "http" not in cleaned
        assert "www" not in cleaned
        assert "check this review at" in cleaned

    def test_clean_text_removes_html(self):
        raw = "<b>Super fast</b> delivery &amp; great <i>packaging</i>."
        cleaned = clean_text(raw)
        assert "<b>" not in cleaned
        assert "</b>" not in cleaned
        assert "&amp;" not in cleaned
        assert "super fast delivery" in cleaned

    def test_clean_text_removes_mentions(self):
        raw = "Thanks @seller for prompt delivery! #happy"
        cleaned = clean_text(raw)
        assert "@seller" not in cleaned
        assert "thanks" in cleaned
        assert "happy" in cleaned

    def test_contraction_expansion(self):
        assert "do not" in expand_contractions("don't")
        assert "cannot" in expand_contractions("can't")
        assert "will not" in expand_contractions("won't")
        assert "it is" in expand_contractions("it's")

    def test_negation_preserved_in_clean_text(self):
        raw = "I didn't like the product, it wasn't good."
        cleaned = clean_text(raw)
        assert "did not" in cleaned
        assert "was not" in cleaned

    def test_whitespace_normalization(self):
        raw = "   Multiple    spaces   between    words   \n\t  "
        cleaned = clean_text(raw)
        assert cleaned == "multiple spaces between words"


class TestLabelStandardization:
    def test_positive_aliases(self):
        assert standardize_sentiment_label("positive") == "Positive"
        assert standardize_sentiment_label("Pos") == "Positive"
        assert standardize_sentiment_label("1") == "Positive"
        assert standardize_sentiment_label(1) == "Positive"

    def test_negative_aliases(self):
        assert standardize_sentiment_label("negative") == "Negative"
        assert standardize_sentiment_label("Neg") == "Negative"
        assert standardize_sentiment_label("-1") == "Negative"
        assert standardize_sentiment_label(-1) == "Negative"

    def test_neutral_aliases(self):
        assert standardize_sentiment_label("neutral") == "Neutral"
        assert standardize_sentiment_label("Neu") == "Neutral"
        assert standardize_sentiment_label("0") == "Neutral"
        assert standardize_sentiment_label(0) == "Neutral"

    def test_none_label(self):
        assert standardize_sentiment_label(None) is None


class TestDatasetCleaning:
    def test_clean_dataset_removes_nulls_and_duplicates(self):
        data = {
            "text": [
                "Excellent battery life!",
                "Excellent battery life!",  # duplicate
                None,                       # null
                "Terrible service.",
                "Factual statement with standard packaging.",
                "   ",                     # empty
            ],
            "sentiment": ["Positive", "Positive", "Positive", "Negative", "Neutral", "Neutral"],
        }
        df = pd.DataFrame(data)
        cleaned_df = clean_dataset(df, text_col="text", label_col="sentiment")

        assert len(cleaned_df) == 3
        assert set(cleaned_df["sentiment"].unique()) == {"Positive", "Negative", "Neutral"}
        assert "cleaned_text" in cleaned_df.columns

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({"review": ["Good"], "score": [5]})
        with pytest.raises(ValueError):
            clean_dataset(df, text_col="text", label_col="sentiment")
