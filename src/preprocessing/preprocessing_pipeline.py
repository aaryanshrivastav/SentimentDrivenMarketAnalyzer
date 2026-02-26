"""
Stage 1A — Manual Preprocessing Pipeline

Usage:
    python src/preprocessing_pipeline.py
"""

import os
import glob
import logging
import pandas as pd

from preprocessing.bot_detection import BotDetection
from preprocessing.spam_filter import SpamFilter
from preprocessing.sarcasm_detection import SarcasmDetection
from preprocessing.ner_linking import NERLinking
from preprocessing.credibility_scoring import CredibilityScoring


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage1a")


# ------------------------------------------------------------
# Helper: Load latest file matching pattern
# ------------------------------------------------------------
def load_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    return pd.read_csv(latest)


# ------------------------------------------------------------
# Load social data
# ------------------------------------------------------------
def load_social_data():

    reddit_df = load_latest_file("data/processed/reddit_clean_*.csv")
    news_df = load_latest_file("data/processed/news_clean_*.csv")

    frames = []

    if reddit_df is not None:
        reddit_df["source"] = "reddit"
        frames.append(reddit_df)
        logger.info(f"Loaded Reddit: {len(reddit_df)} rows")

    if news_df is not None:
        news_df["source"] = "news"
        frames.append(news_df)
        logger.info(f"Loaded News: {len(news_df)} rows")

    if not frames:
        raise ValueError("No cleaned social data found.")

    return frames


# ------------------------------------------------------------
# Main preprocessing logic
# ------------------------------------------------------------
def run_preprocessing():

    frames = load_social_data()

    reddit_df = None
    news_df = None

    for df in frames:
        if df["source"].iloc[0] == "reddit":
            reddit_df = df
        else:
            news_df = df

    # ---------------- Reddit branch ----------------
    if reddit_df is not None:
        logger.info("Running bot detection on Reddit...")
        reddit_df = BotDetection(reddit_df).run()
        logger.info(f"After bot detection: {len(reddit_df)}")

        logger.info("Running spam filtering on Reddit...")
        reddit_df = SpamFilter(reddit_df).run()
        logger.info(f"After spam filtering: {len(reddit_df)}")

    # ---------------- Merge ----------------
    combined = []

    if reddit_df is not None:
        combined.append(reddit_df)

    if news_df is not None:
        combined.append(news_df)

    df = pd.concat(combined, ignore_index=True)
    logger.info(f"Combined rows: {len(df)}")

    # ---------------- Sarcasm ----------------
    logger.info("Running sarcasm detection...")
    df = SarcasmDetection(df).run()

    # ---------------- NER Linking ----------------
    logger.info("Running ticker attribution...")
    df = NERLinking(df).run()
    logger.info(f"After ticker linking: {len(df)}")

    # ---------------- Credibility Scoring ----------------
    logger.info("Running credibility scoring...")
    df = CredibilityScoring(df).run()

    # ---------------- Save ----------------
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/social_stage1a_final.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Stage 1A complete. Saved to {output_path}")


if __name__ == "__main__":
    run_preprocessing()
