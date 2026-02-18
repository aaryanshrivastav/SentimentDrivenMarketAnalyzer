"""
Central configuration for the Sentiment-Driven Financial Market Analysis pipeline.
All parameters live here — change nothing else to reconfigure the project.
"""

import os
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# STOCKS IN SCOPE
# ─────────────────────────────────────────────

# US tickers
US_TICKERS = ["TSLA", "GME", "AAPL", "NVDA", "AMZN"]

# Indian tickers (NSE — .NS suffix required by yfinance)
IN_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "WIPRO.NS",
    "SBIN.NS",
    "TATAMOTORS.NS",
    "ADANIENT.NS",
    "BAJFINANCE.NS",
]

# Combined — used by default in the pipeline
TICKERS = US_TICKERS + IN_TICKERS

# ─────────────────────────────────────────────
# REDDIT API CREDENTIALS
# ─────────────────────────────────────────────

REDDIT_SUBREDDITS = [
    # US
    "wallstreetbets", "stocks", "investing",
    # India
    "IndiaInvestments", "DalalStreetTalks", "IndianStockMarket",
]
REDDIT_POST_LIMIT    = 500   # posts per subreddit per pull
REDDIT_COMMENT_LIMIT = 100   # top-level comments per post

# ─────────────────────────────────────────────
# STOCKTWITS
# ─────────────────────────────────────────────
STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
STOCKTWITS_LIMIT    = 30   # max per request (API hard cap)
# StockTwits uses plain ticker names; NSE tickers need the exchange prefix dropped
# e.g. RELIANCE.NS → search as "RELIANCE" on StockTwits (handled in collector)

# ─────────────────────────────────────────────
# MARKET DATA (yfinance)
# ─────────────────────────────────────────────
PRICE_INTERVAL = "1h"    # hourly candles
PRICE_PERIOD   = "90d"   # how far back to pull on first run

# Volatility indices
VIX_TICKER      = "^VIX"     # CBOE VIX (US)
INDIA_VIX_TICKER = "^INDIAVIX"  # NSE India VIX

# Broad indices (for macro context — not used in signal, just stored)
INDEX_TICKERS = {
    "nifty50":  "^NSEI",
    "sensex":   "^BSESN",
    "sp500":    "^GSPC",
    "nasdaq":   "^IXIC",
}

# ─────────────────────────────────────────────
# PIPELINE TIMING
# ─────────────────────────────────────────────
COLLECTION_INTERVAL_MINUTES = 60
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMEZONE = timezone.utc

# ─────────────────────────────────────────────
# BOT / SPAM FILTER THRESHOLDS (Stage 1A)
# ─────────────────────────────────────────────
BOT_MIN_ACCOUNT_AGE_DAYS  = 30
BOT_MIN_KARMA             = 100
BOT_MAX_POSTS_PER_HOUR    = 15
SPAM_MIN_UPVOTES          = 5
SPAM_MIN_WORD_COUNT       = 10
DUPLICATE_WINDOW_HOURS    = 24

# ─────────────────────────────────────────────
# CREDIBILITY SCORING WEIGHTS (Stage 1A)
# ─────────────────────────────────────────────
SOURCE_WEIGHTS = {
    "news":       1.0,
    "stocktwits": 0.7,
    "reddit":     0.5,
}

ACCOUNT_AGE_WEIGHTS = {
    "under_30_days":  0.0,
    "30_to_90_days":  0.3,
    "90_to_365_days": 0.6,
    "over_1_year":    1.0,
}

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
MARKET_DATA_DIR = os.path.join(BASE_DIR, "data", "market")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"