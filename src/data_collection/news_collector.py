"""
Stage 0 — Financial News RSS Collector
Replaces StockTwits (blocked in India) with free RSS feeds from:
  - Yahoo Finance (US stocks)
  - Moneycontrol (Indian stocks/markets)
  - Economic Times Markets
  - LiveMint

No API key required. Output schema matches reddit_collector for seamless merging.

Output:
    data/raw/news_raw_{timestamp}.csv
"""

import requests
import pandas as pd
import logging
import time
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    US_TICKERS, IN_TICKERS, RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE
)

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_DELAY_SECONDS = 2.0

def _clean_ticker(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


# ── RSS Feed definitions ─────────────────────────────────────────────────────
def _build_feed_list() -> list[dict]:
    feeds = []

    # Yahoo Finance RSS — one feed per US ticker
    for ticker in US_TICKERS:
        feeds.append({
            "url":    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            "source": "yahoo_finance",
            "market": "US",
            "ticker": ticker,
        })

    # Yahoo Finance RSS — Indian tickers
    for ticker in IN_TICKERS:
        feeds.append({
            "url":    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN",
            "source": "yahoo_finance",
            "market": "IN",
            "ticker": ticker,
        })

    # Moneycontrol — Indian market news
    feeds.append({
        "url":    "https://www.moneycontrol.com/rss/marketreports.xml",
        "source": "moneycontrol",
        "market": "IN",
        "ticker": None,
    })
    feeds.append({
        "url":    "https://www.moneycontrol.com/rss/buzzingstocks.xml",
        "source": "moneycontrol",
        "market": "IN",
        "ticker": None,
    })

    # Economic Times Markets
    feeds.append({
        "url":    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "source": "economic_times",
        "market": "IN",
        "ticker": None,
    })
    feeds.append({
        "url":    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "source": "economic_times",
        "market": "IN",
        "ticker": None,
    })

    # LiveMint Markets
    feeds.append({
        "url":    "https://www.livemint.com/rss/markets",
        "source": "livemint",
        "market": "IN",
        "ticker": None,
    })

    return feeds


def _parse_rss_date(date_str: str) -> str:
    """Parse RSS date string to UTC formatted string."""
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)
    except Exception:
        return datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT)


def fetch_rss_feed(feed: dict, pull_time: str) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns list of news record dicts."""
    try:
        response = requests.get(feed["url"], headers=HEADERS, timeout=15)
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {feed['url']}")
            return []

        root = ET.fromstring(response.content)
        items = root.findall(".//item")
        records = []

        for item in items:
            title       = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            pub_date    = item.findtext("pubDate") or ""
            link        = item.findtext("link") or ""

            # Combine title + description as the text field
            text = f"{title}. {description}".strip(". ") if description else title
            if not text:
                continue

            records.append({
                "id":                 link or f"{feed['source']}_{hash(text)}",
                "source":             feed["source"],
                "subreddit":          None,
                "type":               "news",
                "author":             feed["source"],
                "text":               text,
                "upvotes":            None,
                "downvotes":          None,
                "upvote_ratio":       None,
                "account_age_days":   None,
                "account_karma":      None,
                "post_flair":         feed.get("ticker"),
                "market":             feed["market"],
                "timestamp_utc":      _parse_rss_date(pub_date) if pub_date else pull_time,
                "pull_timestamp_utc": pull_time,
            })

        logger.info(f"{feed['source']} ({feed.get('ticker', 'general')}): {len(records)} articles")
        return records

    except ET.ParseError as e:
        logger.warning(f"XML parse error for {feed['url']}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch {feed['url']}: {e}")
        return []


def run_news_collection() -> pd.DataFrame:
    """Entry point: fetch all RSS feeds, deduplicate, save CSV."""
    pull_time = datetime.now(tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)
    feeds = _build_feed_list()
    all_records = []

    logger.info(f"Fetching {len(feeds)} RSS feeds...")

    for feed in feeds:
        records = fetch_rss_feed(feed, pull_time)
        all_records.extend(records)
        time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(all_records)

    if df.empty:
        logger.warning("No news data collected.")
        return df

    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    logger.info(f"Deduplicated: {before} to {len(df)} news articles")

    timestamp_tag = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_DATA_DIR, f"news_raw_{timestamp_tag}.csv")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    df = run_news_collection()
    print(f"\nTotal articles: {len(df)}")
    print(df[["source", "market", "timestamp_utc"]].value_counts().head(20))