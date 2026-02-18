"""
Stage 0 — Pipeline Orchestrator
Runs all data collectors in sequence, on a fixed hourly schedule.

Usage:
    python src/run_pipeline.py              # run once immediately
    python src/run_pipeline.py --schedule   # run every hour on the hour (production mode)
    python src/run_pipeline.py --market     # market data only (no social)
    python src/run_pipeline.py --social     # social data only (no market)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    COLLECTION_INTERVAL_MINUTES, TIMEZONE, LOG_FORMAT, LOG_LEVEL
)
from src.reddit_collector import run_reddit_collection
from src.stockwits_collector import run_stocktwits_collection
from src.marketdata_collector import run_market_data_collection# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("orchestrator")


def run_all(run_social: bool = True, run_market: bool = True) -> dict:
    """
    Execute one full collection cycle.
    Returns a summary dict with row counts per source.
    """
    run_ts = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info(f"═══ Pipeline run started at {run_ts} ═══")

    summary = {}

    # ── Social data ──────────────────────────────────────────────────────
    if run_social:
        try:
            logger.info("--- Collecting Reddit ---")
            reddit_df = run_reddit_collection()
            summary["reddit"] = len(reddit_df)
        except Exception as e:
            logger.error(f"Reddit collection failed: {e}", exc_info=True)
            summary["reddit"] = 0

        try:
            logger.info("--- Collecting StockTwits ---")
            st_df = run_stocktwits_collection()
            summary["stocktwits"] = len(st_df)
        except Exception as e:
            logger.error(f"StockTwits collection failed: {e}", exc_info=True)
            summary["stocktwits"] = 0

    # ── Market data ──────────────────────────────────────────────────────
    if run_market:
        try:
            logger.info("--- Collecting Market Data ---")
            market_results = run_market_data_collection()
            summary["market_tickers"] = sum(
                len(v) for k, v in market_results.items() if k.startswith("prices_") and "_all" not in k
            )
            summary["vix_rows"] = len(market_results.get("vix", []))
        except Exception as e:
            logger.error(f"Market data collection failed: {e}", exc_info=True)
            summary["market"] = 0

    finish_ts = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info(f"═══ Pipeline run complete at {finish_ts} ═══")
    logger.info(f"    Summary: {summary}")
    return summary


def seconds_until_next_hour() -> float:
    """How many seconds until the top of the next hour (00:00 of next hour)."""
    now = datetime.now(tz=TIMEZONE)
    seconds_past_hour = now.minute * 60 + now.second + now.microsecond / 1_000_000
    return 3600 - seconds_past_hour


def run_scheduled(run_social: bool = True, run_market: bool = True):
    """
    Production scheduler: waits until the top of the hour, then runs every hour.
    Aligned collection windows = aligned timestamps = no lag problems later.
    """
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
    parser.add_argument("--market", action="store_true",
                        help="Run market data collection only")
    parser.add_argument("--social", action="store_true",
                        help="Run social data collection only")
    args = parser.parse_args()

    run_social = not args.market   # if --market flag, skip social
    run_market = not args.social   # if --social flag, skip market

    if args.schedule:
        run_scheduled(run_social=run_social, run_market=run_market)
    else:
        run_all(run_social=run_social, run_market=run_market)