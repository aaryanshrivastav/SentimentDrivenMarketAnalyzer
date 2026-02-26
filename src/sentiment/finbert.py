"""
STAGE 1B — SENTIMENT ANALYSIS ENGINE (FinBERT)
==============================================
Model: ProsusAI/finbert (or your fine-tuned version)
Purpose: Run inference on UNLABELED data (news, reddit)

Input:  Cleaned, sarcasm-checked, ticker-attributed DataFrame
Output: Original DataFrame enriched with sentiment columns

Usage:
  1. Train model: python fintrain.py
  2. For batch inference on unlabeled CSVs: python finbert.py
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_NAME        = "ProsusAI/finbert"
MAX_TOKENS        = 512
BATCH_SIZE        = 32          # tune down if OOM
CONFIDENCE_THRESH = 0.70        # below this → uncertain
UNCERTAIN_WEIGHT  = 0.30        # uncertain posts count at 30% weight

LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}


# ─────────────────────────────────────────────
# Model loader (singleton-style)
# ─────────────────────────────────────────────
class FinBERTEngine:
    """
    Wraps tokeniser + model.  Call `.predict(texts)` to score a list of strings.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading FinBERT on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # HuggingFace finbert labels: positive / negative / neutral (order matters)
        self.id2label = self.model.config.id2label          # {0: 'positive', 1: 'negative', 2: 'neutral'}
        logger.info(f"Label map from model config: {self.id2label}")

    # ------------------------------------------------------------------
    def predict(self, texts: list[str]) -> list[dict]:
        """
        Returns a list of dicts, one per input text:
        {
            "label":        "positive" | "negative" | "neutral",
            "numeric":      +1 | -1 | 0,
            "confidence":   float  (score of the top label),
            "scores":       {"positive": float, "negative": float, "neutral": float},
            "is_uncertain": bool
        }
        """
        all_results = []

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,          # truncate from the END — key info at the start
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits           # (B, 3)
            probs = softmax(logits, dim=-1).cpu().numpy()       # (B, 3)

            for row in probs:
                scores = {self.id2label[i].lower(): float(row[i]) for i in range(len(row))}
                top_label      = max(scores, key=scores.get)
                top_confidence = scores[top_label]

                all_results.append({
                    "label":        top_label,
                    "numeric":      LABEL_MAP[top_label],
                    "confidence":   top_confidence,
                    "scores":       scores,
                    "is_uncertain": top_confidence < CONFIDENCE_THRESH,
                })

        return all_results


# ─────────────────────────────────────────────
# Main enrichment function
# ─────────────────────────────────────────────
def run_finbert_stage(
    df: pd.DataFrame,
    text_col:        str = "clean_text",
    credibility_col: str = "user_credibility",
    engine: Optional[FinBERTEngine] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df               : DataFrame produced by Stage 1A.
                       Must have `text_col` and `credibility_col`.
    text_col         : Column with pre-cleaned post text.
    credibility_col  : Float 0-1 credibility weight per author.
    engine           : Optional pre-loaded FinBERTEngine (avoids reloading).

    Returns
    -------
    df with additional columns:
        sentiment_label, sentiment_numeric, sentiment_confidence,
        is_uncertain, weighted_sentiment_score
    """
    if engine is None:
        engine = FinBERTEngine()

    texts   = df[text_col].fillna("").tolist()
    results = engine.predict(texts)

    labels        = [r["label"]        for r in results]
    numerics      = [r["numeric"]      for r in results]
    confidences   = [r["confidence"]   for r in results]
    is_uncertains = [r["is_uncertain"] for r in results]

    df = df.copy()
    df["sentiment_label"]      = labels
    df["sentiment_numeric"]    = numerics
    df["sentiment_confidence"] = confidences
    df["is_uncertain"]         = is_uncertains

    # ── Weighted sentiment score ─────────────────────────────────────
    # score = numeric × confidence × credibility_weight
    # If uncertain, apply additional 70% reduction (i.e. ×0.30)
    credibility    = df[credibility_col].fillna(1.0).values
    uncertainty_mf = np.where(df["is_uncertain"].values, UNCERTAIN_WEIGHT, 1.0)

    df["weighted_sentiment_score"] = (
        df["sentiment_numeric"].values
        * df["sentiment_confidence"].values
        * credibility
        * uncertainty_mf
    )

    logger.info(
        f"Stage 1B complete. "
        f"Uncertain posts: {df['is_uncertain'].sum()}/{len(df)} "
        f"({df['is_uncertain'].mean()*100:.1f}%)"
    )
    return df


# ─────────────────────────────────────────────
# BATCH INFERENCE ON UNLABELED DATA
# ─────────────────────────────────────────────
def batch_inference_on_csv(
    input_csv: str,
    output_csv: str,
    text_col: str = "text_clean",
    model_name: str = MODEL_NAME,
    use_finetuned: bool = False,
) -> pd.DataFrame:
    """
    Run sentiment inference on an UNLABELED CSV file.
    
    Parameters
    ----------
    input_csv : Path to unlabeled CSV (e.g., news_clean_*.csv, reddit_clean_*.csv)
    output_csv : Path to save predictions
    text_col : Column containing text to analyze
    model_name : FinBERT model to use ('ProsusAI/finbert' or path to fine-tuned)
    use_finetuned : If True, looks for fine-tuned model in ../finbert_finetuned/
    
    Returns
    -------
    DataFrame with sentiment predictions added
    """
    from pathlib import Path
    
    logger.info(f"Running batch inference on: {input_csv}")
    
    # Load data
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        # Try to find text column
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        if text_col is None:
            raise ValueError(f"No text column found in {input_csv}")
        logger.info(f"Using text column: {text_col}")
    
    # Use fine-tuned model if requested and available
    if use_finetuned:
        finetuned_path = Path(__file__).parent.parent / "finbert_finetuned" / "best_model"
        if finetuned_path.exists():
            model_name = str(finetuned_path)
            logger.info(f"Using fine-tuned model: {model_name}")
        else:
            logger.warning(f"Fine-tuned model not found at {finetuned_path}, using base model")
    
    # Create engine
    engine = FinBERTEngine(model_name=model_name)
    
    # Add credibility column if doesn't exist (default to 1.0)
    if "user_credibility" not in df.columns:
        df["user_credibility"] = 1.0
    
    # Run inference
    df_with_sentiment = run_finbert_stage(
        df,
        text_col=text_col,
        credibility_col="user_credibility",
        engine=engine
    )
    
    # Save results
    df_with_sentiment.to_csv(output_csv, index=False)
    logger.info(f"Saved predictions to: {output_csv}")
    logger.info(f"\nSentiment distribution:")
    logger.info(df_with_sentiment["sentiment_label"].value_counts().to_string())
    
    return df_with_sentiment


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)

    # If CSV path provided as argument, run batch inference
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".csv", "_sentiment.csv")
        use_finetuned = "--finetuned" in sys.argv or "-f" in sys.argv
        
        batch_inference_on_csv(
            input_csv=input_file,
            output_csv=output_file,
            use_finetuned=use_finetuned
        )
    else:
        # Run smoke test
        sample = pd.DataFrame({
            "clean_text": [
                "Tesla just smashed earnings, this stock is going to the moon!",
                "I'm very worried about Apple's supply chain issues in Q4.",
                "NVDA trading sideways today, nothing special happening.",
                "Honestly not sure if this is good or bad news for AMD.",   # uncertain example
            ],
            "user_credibility": [0.9, 0.7, 0.5, 0.6],
            "ticker": ["TSLA", "AAPL", "NVDA", "AMD"],
        })

        out = run_finbert_stage(sample)
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS")
        print("="*70)
        print(out[[
            "ticker", "sentiment_label", "sentiment_numeric",
            "sentiment_confidence", "is_uncertain", "weighted_sentiment_score"
        ]].to_string(index=False))
        
        print("\n" + "="*70)
        print("BATCH INFERENCE USAGE")
        print("="*70)
        print("Process unlabeled CSV:")
        print("  python finbert.py <input.csv> [output.csv] [--finetuned]")
        print("\nExample:")
        print("  python finbert.py data/processed/news_clean_20240218.csv --finetuned")
        print("="*70)