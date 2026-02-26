"""
FINBERT FINE-TUNING PIPELINE — TRAINING ON LABELED DATA ONLY
==============================================================
Base model: ProsusAI/finbert

Training data (LABELED ONLY):
  - clean_phrasebank.csv   → Financial PhraseBank (labeled sentences)
  - clean_stocknews.csv    → Stock news headlines (labeled)

Unlabeled data (news_clean_*.csv, reddit_clean_*.csv):
  → Use finbert.py for INFERENCE after training

Run:
    python fintrain.py
"""

import os
import re
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — change paths to match your folder structure
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"  # folder with your CSVs
OUTPUT_DIR = Path(__file__).parent.parent / "finbert_finetuned_2"   # where the model is saved
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN    = 128
BATCH_SIZE = 4                         # lower to 8 if GPU OOM
EPOCHS     = 2
LR         = 2e-5
SEED       = 42

# Canonical label mapping — everything gets normalised to these three
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Training uses LABELED data only


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & NORMALISE EACH DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_label(raw) -> str | None:
    """Map any label variant to positive / negative / neutral, or None to drop."""
    if isinstance(raw, (int, float)):
        raw_int = int(raw)
        if raw_int == 2:
            return "positive"
        if raw_int == 1:
            return "neutral"
        if raw_int == 0:
            return "negative"
        if raw == -1.0:
            return "negative"
        return None
    
    # Handle string labels
    if not isinstance(raw, str):
        return None
    raw = raw.strip().lower()
    if raw in {"positive", "pos", "bullish", "2", "buy"}:
        return "positive"
    if raw in {"negative", "neg", "bearish", "0", "-1", "sell"}:
        return "negative"
    if raw in {"neutral", "neu", "hold", "1"}:
        return "neutral"
    return None


def load_phrasebank(path: Path) -> pd.DataFrame:
    """
    Financial PhraseBank standard format:
        sentence,label
        "Profit rose 20%...",positive
    Some versions use '@' as separator — handled below.
    """
    try:
        df = pd.read_csv(path)
        # Try standard column names first
        if "sentence" in df.columns and "label" in df.columns:
            df = df[["sentence", "label"]].rename(columns={"sentence": "text"})
        elif "text" in df.columns and "label" in df.columns:
            df = df[["text", "label"]]
        else:
            # Fall back: assume first col = text, second col = label
            df.columns = ["text", "label"] + list(df.columns[2:])
            df = df[["text", "label"]]
    except Exception:
        # Some phrasebank files use '@' separator: "sentence @ label"
        rows = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "@" in line:
                    parts = line.rsplit("@", 1)
                    rows.append({"text": parts[0].strip(), "label": parts[1].strip()})
        df = pd.DataFrame(rows)

    df["label"] = df["label"].apply(_normalise_label)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"Phrasebank: {len(df)} rows  |  {path.name}")
    return df


def load_stocknews(path: Path) -> pd.DataFrame:
    """
    StockNews format varies. Common variants:
        headline, sentiment
        text, label
        title, sentiment_label
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Find text column
    text_col  = next((c for c in df.columns if c in {"headline","title","text","sentence"}), None)
    label_col = next((c for c in df.columns if c in {"sentiment","label","sentiment_label","target"}), None)

    if text_col is None or label_col is None:
        logger.warning(f"Couldn't identify columns in {path.name} — skipping.  Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["text","label"])

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].apply(_normalise_label)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"StockNews: {len(df)} rows  |  {path.name}")
    return df


def load_news_clean(path: Path) -> pd.DataFrame:
    """
    Load news file ONLY if it has labels.
    Unlabeled news files should be used for inference, not training.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    text_col  = next((c for c in df.columns if "text" in c), None)
    label_col = next((c for c in df.columns if c in {"sentiment","label","sentiment_label","target"}), None)

    if text_col is None:
        logger.warning(f"No text column in {path.name} — skipping.")
        return pd.DataFrame(columns=["text","label"])
    
    if label_col is None:
        logger.info(f"No labels in {path.name} — skipping (use finbert.py for inference).")
        return pd.DataFrame(columns=["text","label"])
    
    # Has labels - use for training
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].apply(_normalise_label)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"News clean: {len(df)} labeled rows  |  {path.name}")
    return df





def build_dataset() -> pd.DataFrame:
    """Load LABELED training data only."""
    logger.info("\n" + "="*70)
    logger.info("  LOADING LABELED TRAINING DATA")
    logger.info("="*70)
    
    frames = []

    # Load phrasebank
    pb = DATA_DIR / "clean_phrasebank.csv"
    if pb.exists():
        frames.append(load_phrasebank(pb))
    else:
        logger.warning("⚠ clean_phrasebank.csv not found")

    # Load stocknews
    sn = DATA_DIR / "clean_stocknews.csv"
    if sn.exists():
        frames.append(load_stocknews(sn))
    else:
        logger.warning("⚠ clean_stocknews.csv not found")

    # Check for labeled news files (unlikely, but possible)
    for p in DATA_DIR.glob("news_clean_*.csv"):
        news_df = load_news_clean(p)
        if len(news_df) > 0:
            frames.append(news_df)

    if not frames:
        raise FileNotFoundError(
            f"No labeled training data found in {DATA_DIR.resolve()}\n"
            f"Make sure clean_phrasebank.csv and/or clean_stocknews.csv exist."
        )

    # Combine and clean
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby("label").sample(n=5000, random_state=SEED)
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 10]
    combined = combined.drop_duplicates(subset=["text"])

    # Map labels to integer IDs
    combined["label_id"] = combined["label"].map(LABEL2ID)
    combined = combined.dropna(subset=["label_id"])

    logger.info(f"\n\u2713 Final training dataset: {len(combined)} rows")
    logger.info(f"\nLabel distribution:")
    logger.info(combined["label"].value_counts().to_string())
    logger.info("\n\u2192 For inference on unlabeled data, use: finbert.py")
    logger.info("="*70 + "\n")
    
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":  f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)

    # ── Load data ────────────────────────────────────────────────────
    df = build_dataset()

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["label_id"]
    )
    logger.info(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    # ── Tokeniser ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset(
        train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, MAX_LEN
    )
    val_dataset = SentimentDataset(
        val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, MAX_LEN
    )

    # ── Model ────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # safe when reusing pretrained head
    )

    # ── Training arguments ───────────────────────────────────────────
    args = TrainingArguments(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        learning_rate               = LR,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,          # 10% of steps for warmup
        lr_scheduler_type           = "linear",
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        logging_steps               = 50,
        fp16                        = torch.cuda.is_available(),
        seed                        = SEED,
        report_to                   = "none",       # swap to "wandb" if you use it
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    trainer.save_model(str(OUTPUT_DIR / "best_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))
    logger.info(f"Model saved to {OUTPUT_DIR / 'best_model'}")

    # ── Final evaluation report ──────────────────────────────────────
    preds_output = trainer.predict(val_dataset)
    preds        = np.argmax(preds_output.predictions, axis=-1)
    true         = val_df["label_id"].tolist()
    print("\n── Validation Classification Report ──")
    print(f"Unique classes in true labels: {np.unique(true)}")
    print(f"Unique classes in predictions: {np.unique(preds)}")
    print(classification_report(true, preds, labels=[0, 1, 2], target_names=["positive","negative","neutral"], zero_division=0))
    
    # ── Next steps ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\nFine-tuned model saved to: {OUTPUT_DIR / 'best_model'}")
    print("\nNext: Run inference on UNLABELED data (news, reddit):")
    print(f"  python finbert.py data/processed/news_clean_*.csv --finetuned")
    print(f"  python finbert.py data/processed/reddit_clean_*.csv --finetuned")
    print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — LOAD YOUR FINE-TUNED MODEL FOR INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned(model_dir: str = str(OUTPUT_DIR / "best_model")):
    """
    Drop-in replacement for ProsusAI/finbert in your Stage 1B pipeline.
    Just swap MODEL_NAME → model_dir in FinBERTEngine().
    """
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model     = model_dir,
        tokenizer = model_dir,
        device    = 0 if torch.cuda.is_available() else -1,
    )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()