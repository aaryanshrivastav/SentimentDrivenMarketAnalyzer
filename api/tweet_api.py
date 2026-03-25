"""
Tweet scoring API.

Run directly:
  uvicorn api.tweet_api:app --host 0.0.0.0 --port 8011 --reload
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.lstm import ALL_FEATURES, build_interaction_features
from src.sentiment.finbert import FinBERTEngine
from src.utils.text_cleaner import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

V4_MODEL_PATH = MODELS_DIR / "v4_full_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

SENTIMENT_FEATURES = [
    "avg_sentiment",
    "sentiment_std",
    "pos_count",
    "neg_count",
    "neu_count",
    "bull_bear_ratio",
    "mention_volume",
    "weighted_volume",
    "high_confidence_ratio",
    "sentiment_momentum",
    "sentiment_acceleration",
    "sentiment_available",
    "sentiment_imputed",
]


def _safe_float(value: float | int | np.floating | np.integer | None) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v


@lru_cache(maxsize=1)
def get_finbert_engine() -> FinBERTEngine:
    return FinBERTEngine()


@lru_cache(maxsize=1)
def load_v4_artifacts():
    if not V4_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {V4_MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = joblib.load(V4_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def _read_template_frame() -> pd.DataFrame:
    candidates = [
        OUTPUT_DIR / "fused_features.csv",
        OUTPUT_DIR / "feature_matrix.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if len(df) > 0:
                if "timestamp_utc" in df.columns:
                    df = df.sort_values("timestamp_utc")
                return df

    raise FileNotFoundError(
        "No non-empty template found. Expected output/fused_features.csv or output/feature_matrix.csv"
    )


def _sentiment_payload_from_text(text: str, user_credibility: float) -> Dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")

    engine = get_finbert_engine()
    result = engine.predict([cleaned])[0]

    label = result["label"]
    confidence = _safe_float(result["confidence"])
    numeric = int(result["numeric"])

    uncertain_multiplier = 0.30 if result["is_uncertain"] else 1.0
    weighted_sentiment = numeric * confidence * user_credibility * uncertain_multiplier

    pos_count = 1 if label == "positive" else 0
    neg_count = 1 if label == "negative" else 0
    neu_count = 1 if label == "neutral" else 0

    return {
        "clean_text": cleaned,
        "sentiment_label": label,
        "sentiment_numeric": numeric,
        "sentiment_confidence": confidence,
        "is_uncertain": bool(result["is_uncertain"]),
        "finbert_scores": result["scores"],
        "weighted_sentiment_score": weighted_sentiment,
        "avg_sentiment": weighted_sentiment,
        "sentiment_std": 0.0,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count,
        "bull_bear_ratio": 1.0 if (pos_count + neg_count) == 0 else pos_count / (pos_count + neg_count),
        "mention_volume": 1.0,
        "weighted_volume": user_credibility,
        "high_confidence_ratio": 1.0 if confidence > 0.80 else 0.0,
        "sentiment_momentum": 0.0,
        "sentiment_acceleration": 0.0,
        "sentiment_available": 1.0,
        "sentiment_imputed": 0.0,
    }


def _predict_v4_probability(ticker: str, sentiment_features: Dict) -> Dict:
    model, scaler = load_v4_artifacts()
    template_df = _read_template_frame()

    if "ticker" in template_df.columns:
        by_ticker = template_df[template_df["ticker"].astype(str).str.upper() == ticker.upper()]
        base_row = by_ticker.tail(1) if len(by_ticker) > 0 else template_df.tail(1)
    else:
        base_row = template_df.tail(1)

    row = base_row.copy()
    for key, value in sentiment_features.items():
        if key in SENTIMENT_FEATURES:
            row[key] = _safe_float(value)

    row = build_interaction_features(row)

    feature_names: List[str]
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = [f for f in ALL_FEATURES if f in row.columns]

    if not feature_names:
        raise ValueError("Could not determine model feature columns for v4_full_model.pkl")

    X = pd.DataFrame(index=row.index)
    for feature in feature_names:
        X[feature] = pd.to_numeric(row.get(feature, 0.0), errors="coerce").fillna(0.0)

    Xs = scaler.transform(X.values)
    proba = model.predict_proba(Xs)[0]

    down_score = float(proba[0])
    up_score = float(proba[1])
    predicted_direction = "UP" if up_score >= 0.5 else "DOWN"
    main_score = up_score

    return {
        "main_score": main_score,
        "predicted_direction": predicted_direction,
        "v4_up_probability": up_score,
        "v4_down_probability": down_score,
        "model_features_used": feature_names,
    }


class TweetScoreRequest(BaseModel):
    tweet: str = Field(..., min_length=1)
    ticker: str = Field(default="AAPL", min_length=1)
    user_credibility: float = Field(default=1.0, ge=0.1, le=1.0)


app = FastAPI(title="Tweet Scoring API", version="1.0.0")


@app.get("/")
def health() -> dict:
    return {
        "service": "tweet_api",
        "status": "running",
        "model_path": str(V4_MODEL_PATH),
    }


@app.post("/api/tweet/score")
def score_tweet(payload: TweetScoreRequest) -> dict:
    try:
        sentiment = _sentiment_payload_from_text(
            text=payload.tweet,
            user_credibility=payload.user_credibility,
        )
        model_scores = _predict_v4_probability(payload.ticker, sentiment)

        return {
            "ticker": payload.ticker.upper(),
            "input_tweet": payload.tweet,
            "clean_text": sentiment["clean_text"],
            "main_score": model_scores["main_score"],
            "predicted_direction": model_scores["predicted_direction"],
            "scores": {
                "v4_up_probability": model_scores["v4_up_probability"],
                "v4_down_probability": model_scores["v4_down_probability"],
                "finbert_label": sentiment["sentiment_label"],
                "finbert_numeric": sentiment["sentiment_numeric"],
                "finbert_confidence": sentiment["sentiment_confidence"],
                "finbert_scores": sentiment["finbert_scores"],
                "weighted_sentiment_score": sentiment["weighted_sentiment_score"],
            },
            "metadata": {
                "model_file": str(V4_MODEL_PATH),
                "scaler_file": str(SCALER_PATH),
                "feature_count": len(model_scores["model_features_used"]),
            },
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tweet scoring failed: {exc}") from exc
