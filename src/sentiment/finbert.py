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
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from typing import Optional
import logging
import os
from tqdm import tqdm

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

        if self.device == "cpu":
            self._configure_cpu_runtime()

        self.batch_size = self._effective_batch_size()
        logger.info(f"Loading FinBERT on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.device == "cpu":
            self._maybe_quantize_model()
        self.model.to(self.device)
        self.model.eval()

        # HuggingFace finbert labels: positive / negative / neutral (order matters)
        self.id2label = self.model.config.id2label          # {0: 'positive', 1: 'negative', 2: 'neutral'}
        logger.info(f"Label map from model config: {self.id2label}")
        logger.info(f"FinBERT batch size: {self.batch_size}")

    def _configure_cpu_runtime(self):
        """Tune PyTorch CPU threading; overridable with FINBERT_CPU_THREADS."""
        cpu_count = os.cpu_count() or 4
        default_threads = max(2, min(8, cpu_count - 2))
        env_threads = os.getenv("FINBERT_CPU_THREADS", "")
        try:
            threads = int(env_threads) if env_threads else default_threads
            threads = max(1, threads)
            torch.set_num_threads(threads)
            logger.info("FinBERT CPU threads set to %d", threads)
        except ValueError:
            logger.warning("Invalid FINBERT_CPU_THREADS='%s'. Using defaults.", env_threads)

    def _maybe_quantize_model(self):
        """Optional dynamic quantization on CPU for faster linear layers."""
        quantize = os.getenv("FINBERT_CPU_QUANTIZE", "1") == "1"
        if not quantize:
            return
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            logger.info("Enabled dynamic quantization for FinBERT (CPU)")
        except Exception as exc:
            logger.warning("Could not quantize FinBERT model; continuing without quantization: %s", exc)

    def _effective_batch_size(self) -> int:
        env_bs = os.getenv("FINBERT_BATCH_SIZE", "")
        if env_bs:
            try:
                parsed = int(env_bs)
                if parsed > 0:
                    return parsed
            except ValueError:
                logger.warning("Invalid FINBERT_BATCH_SIZE='%s'. Using defaults.", env_bs)
        return 64 if self.device == "cpu" else 128

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

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size if texts else 0
        batch_iter = range(0, len(texts), self.batch_size)

        for batch_start in tqdm(
            batch_iter,
            total=total_batches,
            desc=f"FinBERT inference ({self.device}, bs={self.batch_size})",
            unit="batch",
        ):
            batch = texts[batch_start : batch_start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,          # truncate from the END — key info at the start
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
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

    score_only_unlabeled = os.getenv("FINBERT_ONLY_UNLABELED", "0") == "1"

    df = df.copy()

    if score_only_unlabeled and "original_label_available" in df.columns:
        to_score_mask = ~df["original_label_available"].fillna(False)
        score_idx = df.index[to_score_mask].tolist()
        passthrough_idx = df.index[~to_score_mask].tolist()

        texts = df.loc[score_idx, text_col].fillna("").tolist()
        logger.info(
            "FinBERT scoring scope: %d/%d rows (FINBERT_ONLY_UNLABELED=1)",
            len(texts), len(df)
        )
        scored_results = engine.predict(texts)

        # Initialize arrays for whole DataFrame and fill from scored + passthrough rows.
        labels = [None] * len(df)
        numerics = [0] * len(df)
        confidences = [0.0] * len(df)
        is_uncertains = [True] * len(df)

        pos_by_idx = {idx: pos for pos, idx in enumerate(df.index.tolist())}

        # Fill model outputs for unlabeled rows.
        for idx, r in zip(score_idx, scored_results):
            p = pos_by_idx[idx]
            labels[p] = r["label"]
            numerics[p] = r["numeric"]
            confidences[p] = r["confidence"]
            is_uncertains[p] = r["is_uncertain"]

        # Preserve existing labels for originally labeled rows without model call.
        label_norm_map = {
            "bullish": ("positive", 1),
            "bearish": ("negative", -1),
            "neutral": ("neutral", 0),
            "positive": ("positive", 1),
            "negative": ("negative", -1),
        }
        for idx in passthrough_idx:
            p = pos_by_idx[idx]
            raw = str(df.at[idx, "sentiment_label"]).strip().lower()
            mapped_label, mapped_num = label_norm_map.get(raw, ("neutral", 0))
            labels[p] = mapped_label
            numerics[p] = mapped_num
            confidences[p] = 1.0
            is_uncertains[p] = False
    else:
        texts = df[text_col].fillna("").tolist()
        logger.info("FinBERT scoring scope: %d/%d rows", len(texts), len(df))
        results = engine.predict(texts)

        labels        = [r["label"]        for r in results]
        numerics      = [r["numeric"]      for r in results]
        confidences   = [r["confidence"]   for r in results]
        is_uncertains = [r["is_uncertain"] for r in results]

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