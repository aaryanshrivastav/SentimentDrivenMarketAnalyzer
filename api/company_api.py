"""
Company scoring API.

Flow:
1) Resolve company -> ticker
2) Collect Reddit posts via ticker_reddit_collector
3) Score posts with FinBERT
4) Build sentiment aggregate
5) Predict final score using models/v4_full_model.pkl

Run directly:
  uvicorn api.company_api:app --host 0.0.0.0 --port 8012 --reload
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.tweet_api import _predict_v4_probability, _safe_float, get_finbert_engine
from src.data_collection.ticker_reddit_collector import collect_for_ticker, resolve_ticker
from src.utils.text_cleaner import clean_text


def _aggregate_company_sentiment(post_results: List[Dict]) -> Dict:
    if not post_results:
        raise ValueError("No scored posts available for aggregation.")

    labels = [r["label"] for r in post_results]
    confidences = [_safe_float(r["confidence"]) for r in post_results]
    numerics = [int(r["numeric"]) for r in post_results]
    creds = [_safe_float(r.get("user_credibility", 1.0)) for r in post_results]
    uncertain_flags = [bool(r.get("is_uncertain", False)) for r in post_results]

    weighted_scores = []
    for num, conf, cred, uncertain in zip(numerics, confidences, creds, uncertain_flags):
        multiplier = 0.30 if uncertain else 1.0
        weighted_scores.append(num * conf * cred * multiplier)

    n = len(weighted_scores)
    pos_count = int(sum(1 for x in labels if x == "positive"))
    neg_count = int(sum(1 for x in labels if x == "negative"))
    neu_count = int(sum(1 for x in labels if x == "neutral"))
    denominator = pos_count + neg_count
    bull_bear_ratio = (pos_count / denominator) if denominator > 0 else 0.5

    avg_sentiment = float(sum(weighted_scores) / n)
    sentiment_std = 0.0
    if n > 1:
        mean = avg_sentiment
        sentiment_std = float((sum((x - mean) ** 2 for x in weighted_scores) / n) ** 0.5)

    high_conf_ratio = float(sum(1 for c in confidences if c > 0.80) / n)

    return {
        "avg_sentiment": avg_sentiment,
        "sentiment_std": sentiment_std,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count,
        "bull_bear_ratio": bull_bear_ratio,
        "mention_volume": float(n),
        "weighted_volume": float(sum(creds)),
        "high_confidence_ratio": high_conf_ratio,
        "sentiment_momentum": 0.0,
        "sentiment_acceleration": 0.0,
        "sentiment_available": 1.0,
        "sentiment_imputed": 0.0,
        "weighted_sentiment_score": avg_sentiment,
    }


class CompanyScoreRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    posts_per_query: int = Field(default=30, ge=5, le=200)
    fetch_comments: bool = True


app = FastAPI(title="Company Scoring API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health() -> dict:
    return {
        "service": "company_api",
        "status": "running",
    }


@app.post("/api/company/score")
def score_company(payload: CompanyScoreRequest) -> dict:
    try:
        ticker, market = resolve_ticker(payload.company_name)

        collected = collect_for_ticker(
            ticker=ticker,
            market=market,
            posts_per_query=payload.posts_per_query,
            fetch_comments=payload.fetch_comments,
        )

        if collected.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No Reddit data found for {payload.company_name} ({ticker}).",
            )

        text_list = [clean_text(str(t)) for t in collected["text"].fillna("").tolist()]
        text_list = [t for t in text_list if t]

        if not text_list:
            raise HTTPException(
                status_code=400,
                detail="Collected posts are empty after cleaning.",
            )

        engine = get_finbert_engine()
        finbert_results = engine.predict(text_list)

        post_scores: List[Dict] = []
        for result in finbert_results:
            post_scores.append(
                {
                    "label": result["label"],
                    "numeric": int(result["numeric"]),
                    "confidence": _safe_float(result["confidence"]),
                    "is_uncertain": bool(result["is_uncertain"]),
                    "user_credibility": 1.0,
                }
            )

        sentiment_agg = _aggregate_company_sentiment(post_scores)
        model_scores = _predict_v4_probability(ticker, sentiment_agg)

        label_distribution = Counter([x["label"] for x in post_scores])
        avg_finbert_conf = float(
            sum(x["confidence"] for x in post_scores) / len(post_scores)
        )

        sample_posts = text_list[:5]

        return {
            "company_name": payload.company_name,
            "ticker": ticker,
            "market": market,
            "main_score": model_scores["main_score"],
            "predicted_direction": model_scores["predicted_direction"],
            "scores": {
                "v4_up_probability": model_scores["v4_up_probability"],
                "v4_down_probability": model_scores["v4_down_probability"],
                "avg_sentiment": sentiment_agg["avg_sentiment"],
                "sentiment_std": sentiment_agg["sentiment_std"],
                "weighted_sentiment_score": sentiment_agg["weighted_sentiment_score"],
                "avg_finbert_confidence": avg_finbert_conf,
                "label_distribution": dict(label_distribution),
                "mention_volume": sentiment_agg["mention_volume"],
            },
            "metadata": {
                "collected_rows": int(len(collected)),
                "scored_rows": int(len(post_scores)),
                "sample_posts": sample_posts,
            },
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Company scoring failed: {exc}") from exc
