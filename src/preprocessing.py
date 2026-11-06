"""
preprocessing.py
----------------
Text cleaning and normalization utilities for Sentiment Analysis.
Consistent preprocessing applied to both training data and live inference.
"""

import re
import html
from typing import Optional, Dict
import pandas as pd

# Common English contraction mapping to preserve negation semantics
CONTRACTIONS_DICT: Dict[str, str] = {
    "can't": "cannot",
    "can not": "cannot",
    "won't": "will not",
    "n't": " not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "ain't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mustn't": "must not",
    "needn't": "need not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "wouldn't": "would not",
}

# Regex pattern for contraction expansion
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS_DICT.keys()) + r")\b",
    re.IGNORECASE,
)

# Common URL and HTML patterns
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<.*?>")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_HASH_RE = re.compile(r"#(\w+)")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def expand_contractions(text: str) -> str:
    """Expand standard English contractions into full words."""
    def _replace(match: re.Match) -> str:
        token = match.group(0).lower()
        return CONTRACTIONS_DICT.get(token, token)
    return _CONTRACTION_RE.sub(_replace, text)


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize raw text for sentiment analysis.

    Steps performed:
    1. Handle None / non-string gracefully.
    2. Decode HTML entities (&amp; -> &).
    3. Remove URLs and hyperlinks.
    4. Strip HTML tags.
    5. Strip social mentions (@username).
    6. Keep hashtag text but remove the leading '#' (#superb -> superb).
    7. Expand contractions (don't -> do not) to preserve sentiment negations.
    8. Convert to lowercase.
    9. Remove unwanted special characters and punctuation while keeping words.
    10. Normalize multiple whitespace and trim.

    Parameters
    ----------
    text : Optional[str]
        Raw input string.

    Returns
    -------
    str
        Cleaned, normalized text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Decode HTML entities
    cleaned = html.unescape(text)

    # Remove URLs
    cleaned = _URL_RE.sub(" ", cleaned)

    # Remove HTML tags
    cleaned = _HTML_RE.sub(" ", cleaned)

    # Remove user mentions
    cleaned = _MENTION_RE.sub(" ", cleaned)

    # Convert hashtags to plain text
    cleaned = _HASHTAG_HASH_RE.sub(r"\1", cleaned)

    # Expand contractions before stripping punctuation
    cleaned = expand_contractions(cleaned)

    # Lowercase
    cleaned = cleaned.lower()

    # Remove punctuation and non-word characters (leaving spaces & alphanumerics)
    cleaned = _PUNCT_RE.sub(" ", cleaned)

    # Normalize whitespace
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def standardize_sentiment_label(label: str) -> Optional[str]:
    """
    Standardize varied sentiment label representations into
    canonical: 'Positive', 'Negative', 'Neutral'.
    Supports integers: 1 (Pos), 0 (Neu), -1 (Neg) or 2/1/0.
    """
    if label is None:
        return None
    s = str(label).strip().lower()
    
    if s in {"positive", "pos", "1", "1.0", "good"}:
        return "Positive"
    elif s in {"negative", "neg", "-1", "-1.0", "bad"}:
        return "Negative"
    elif s in {"neutral", "neu", "0", "0.0", "average"}:
        return "Neutral"
    
    # Capitalize capitalized standard strings if already matched
    if s in {"positive", "negative", "neutral"}:
        return s.capitalize()
    
    # Fallback to Title case if alphanumeric
    return s.capitalize() if s else None


def clean_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "sentiment",
) -> pd.DataFrame:
    """
    Clean and validate a sentiment dataset DataFrame.

    - Strips whitespace from column names.
    - Checks required columns exist.
    - Drops missing values in text and label columns.
    - Drops duplicate rows.
    - Preprocesses text using `clean_text`.
    - Drops rows where cleaned text is empty or fewer than 2 characters.
    - Standardizes labels.
    - Resets index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    text_col : str
        Name of the column containing review text.
    label_col : str
        Name of the column containing sentiment labels.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated dataset ready for training.
    """
    df_clean = df.copy()
    df_clean.columns = [str(c).strip() for c in df_clean.columns]

    if text_col not in df_clean.columns:
        raise ValueError(f"Required text column '{text_col}' not found in DataFrame.")
    if label_col not in df_clean.columns:
        raise ValueError(f"Required label column '{label_col}' not found in DataFrame.")

    # Drop nulls in required columns
    df_clean = df_clean.dropna(subset=[text_col, label_col])

    # Convert text and label to string
    df_clean[text_col] = df_clean[text_col].astype(str)
    df_clean[label_col] = df_clean[label_col].astype(str)

    # Drop duplicate raw texts
    df_clean = df_clean.drop_duplicates(subset=[text_col])

    # Apply text cleaning
    df_clean["cleaned_text"] = df_clean[text_col].apply(clean_text)

    # Standardize labels
    df_clean[label_col] = df_clean[label_col].apply(standardize_sentiment_label)
    df_clean = df_clean.dropna(subset=[label_col])

    # Filter out empty or trivially short cleaned texts (< 2 chars)
    df_clean = df_clean[df_clean["cleaned_text"].str.strip().str.len() >= 2]

    # Drop duplicates in cleaned text
    df_clean = df_clean.drop_duplicates(subset=["cleaned_text"])

    return df_clean.reset_index(drop=True)
