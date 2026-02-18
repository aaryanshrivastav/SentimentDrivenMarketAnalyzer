"""
Stage 0 — Pipeline Orchestrator
Runs all data collectors in sequence, on a fixed hourly schedule.

Usage:
    python src/pipeline.py                  # run once immediately (all sources)
    python src/pipeline.py --schedule       # run every hour on the hour (production)
    python src/pipeline.py --market         # market data only
    python src/pipeline.py --social         # reddit + news only (no market)
    python src/pipeline.py --static         # load static datasets (PhraseBank + Kaggle)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

from pandas import DataFrame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    COLLECTION_INTERVAL_MINUTES, TIMEZONE, LOG_FORMAT, LOG_LEVEL
)
from reddit_collector import run_reddit_collection
from news_collector import run_news_collection
from static_dataset_loader import run_static_dataset_loading

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("orchestrator")


def run_all(run_social: bool = True, run_market: bool = True, run_static: bool = False) -> dict:
    run_ts = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info(f"═══ Pipeline run started at {run_ts} ═══")
    summary = {}

    # ── Static datasets (run once manually, not on schedule) ─────────────
    if run_static:
        try:
            logger.info("--- Loading Static Datasets (PhraseBank + Kaggle) ---")
            static_data: DataFrame = run_static_dataset_loading()
            summary["phrasebank_records"] = len(static_data.get("phrasebank", []))
            summary["stocknews_records"] = len(static_data.get("stocknews", []))
        except Exception as e:
            logger.error(f"Static dataset loading failed: {e}", exc_info=True)
            summary["static_records"] = 0

    # ── Social / News data ───────────────────────────────────────────────
    if run_social:
        try:
            logger.info("--- Collecting Reddit (30-day historical) ---")
            reddit_df = run_reddit_collection()
            summary["reddit"] = len(reddit_df)
        except Exception as e:
            logger.error(f"Reddit collection failed: {e}", exc_info=True)
            summary["reddit"] = 0

        try:
            logger.info("--- Collecting Financial News (RSS) ---")
            news_df = run_news_collection()
            summary["news"] = len(news_df)
        except Exception as e:
            logger.error(f"News collection failed: {e}", exc_info=True)
            summary["news"] = 0



    finish_ts = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info(f"═══ Pipeline run complete at {finish_ts} ═══")
    logger.info(f"    Summary: {summary}")
    return summary


def seconds_until_next_hour() -> float:
    now = datetime.now(tz=TIMEZONE)
    seconds_past_hour = now.minute * 60 + now.second + now.microsecond / 1_000_000
    return 3600 - seconds_past_hour


def run_scheduled(run_social: bool = True, run_market: bool = True):
    wait = seconds_until_next_hour()
    logger.info(f"Scheduled mode: waiting {wait:.0f}s until next hour mark ...")
    time.sleep(wait)
    while True:
        cycle_start = time.time()
        run_all(run_social=run_social, run_market=run_market)
        elapsed = time.time() - cycle_start
        sleep_time = max(0, COLLECTION_INTERVAL_MINUTES * 60 - elapsed)
        logger.info(f"Next run in {sleep_time:.0f}s")
        time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0 Pipeline Orchestrator")
    parser.add_argument("--schedule", action="store_true",
                        help="Run on fixed hourly schedule (production mode)")
    parser.add_argument("--market",   action="store_true",
                        help="Market data only")
    parser.add_argument("--social",   action="store_true",
                        help="Reddit + News only")
    parser.add_argument("--static",   action="store_true",
                        help="Load static datasets (PhraseBank + Kaggle) only")
    args = parser.parse_args()

    # Determine what to run
    only_static = args.static and not args.market and not args.social
    run_social  = (not args.market and not only_static)
    run_market  = (not args.social and not only_static)
    run_static  = args.static

    if args.schedule:
        run_scheduled(run_social=run_social, run_market=run_market)
    else:
        run_all(run_social=run_social, run_market=run_market, run_static=run_static)