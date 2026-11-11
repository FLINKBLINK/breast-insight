import html
import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()
_UNIVERSE = np.linspace(0.0, 100.0, 1001)

def clean_review(text):
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _triangle(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

def _trapezoid(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0

def _output_membership(label):
    if label == "negative":
        return np.array([_trapezoid(v, 0, 0, 22, 48) for v in _UNIVERSE])
    if label == "neutral":
        return np.array([_triangle(v, 30, 50, 70) for v in _UNIVERSE])
    return np.array([_trapezoid(v, 52, 78, 100, 100) for v in _UNIVERSE])

_OUTPUT_SETS = {name: _output_membership(name)
                for name in ("negative", "neutral", "positive")}

def fuzzy_inference(polarity, intensity):
    """Mamdani inference: polarity [-1,1], intensity [0,1], output [0,100]."""
    polarity = float(np.clip(polarity, -1.0, 1.0))
    intensity = float(np.clip(intensity, 0.0, 1.0))
    p_neg = _trapezoid(polarity, -1.0, -1.0, -0.35, -0.05)
    p_neu = _triangle(polarity, -0.30, 0.0, 0.30)
    p_pos = _trapezoid(polarity, 0.05, 0.35, 1.0, 1.0)
    i_low = _trapezoid(intensity, 0.0, 0.0, 0.12, 0.35)
    i_mid = _triangle(intensity, 0.15, 0.45, 0.75)
    i_high = _trapezoid(intensity, 0.55, 0.80, 1.0, 1.0)
    rules = [
        (min(p_neg, i_low), "neutral"),
        (min(p_neg, i_mid), "negative"),
        (min(p_neg, i_high), "negative"),
        (min(p_neu, max(i_low, i_mid, i_high)), "neutral"),
        (min(p_pos, i_low), "neutral"),
        (min(p_pos, i_mid), "positive"),
        (min(p_pos, i_high), "positive"),
    ]
    aggregated = np.zeros_like(_UNIVERSE)
    for strength, consequent in rules:
        aggregated = np.maximum(aggregated,
                                np.minimum(strength, _OUTPUT_SETS[consequent]))
    if aggregated.sum() == 0:
        return 50.0
    return float(np.sum(_UNIVERSE * aggregated) / aggregated.sum())

def analyze_review(review):
    cleaned = clean_review(review)
    scores = _ANALYZER.polarity_scores(cleaned)
    intensity = min(1.0, scores["pos"] + scores["neg"])
    score = fuzzy_inference(scores["compound"], intensity)
    return {
        "polarity": scores["compound"],
        "positive_lexicon": scores["pos"],
        "negative_lexicon": scores["neg"],
        "intensity": intensity,
        "fuzzy_score": score,
        "fuzzy_class": "positive" if score >= 55 else "negative" if score <= 45 else "neutral",
    }

def analyze_dataset(df):
    results = df.copy()
    features = results["review"].map(analyze_review).apply(pd.Series)
    return pd.concat([results.reset_index(drop=True), features], axis=1)
