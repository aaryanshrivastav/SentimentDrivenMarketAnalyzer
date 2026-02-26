"""
Stage 0 — Reddit Data Collector (No-Auth JSON API version)
Fetches posts + comments using Reddit's public search endpoint to get
up to 30 days of historical data — no API key required.

Output schema (one row per post or comment):
    id, source, subreddit, type, author, text, upvotes, downvotes,
    upvote_ratio, account_age_days, account_karma, post_flair,
    timestamp_utc, pull_timestamp_utc
"""

import requests
import pandas as pd
import logging
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    REDDIT_SUBREDDITS, REDDIT_POST_LIMIT, REDDIT_COMMENT_LIMIT,
    US_TICKERS, IN_TICKERS, RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_DELAY_SECONDS = 3.0

def _clean_ticker(ticker: str) -> str:
    """Strip exchange suffix for Reddit search queries."""
    return ticker.replace(".NS", "").replace(".BO", "")

# US tickers searched as $TSLA, Indian as plain RELIANCE
SEARCH_QUERIES = (
    [f"${t}" for t in US_TICKERS] +
    [_clean_ticker(t) for t in IN_TICKERS]
)


def _get_json(url: str, params: dict = None, retries: int = 3) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 15 * attempt
                logger.warning(f"Rate limited (429). Waiting {wait}s before retry {attempt}/{retries}...")
                time.sleep(wait)
            elif response.status_code == 403:
                logger.error(f"403 Forbidden: {url} — subreddit may be private.")
                return None
            else:
                logger.warning(f"HTTP {response.status_code} for {url} (attempt {attempt}/{retries})")
                time.sleep(REQUEST_DELAY_SECONDS * attempt)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error attempt {attempt}/{retries}: {e}")
            time.sleep(REQUEST_DELAY_SECONDS * attempt)
    return None


def _parse_timestamp(created_utc: float) -> str:
    return datetime.fromtimestamp(created_utc, tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)


def fetch_post_comments(subreddit: str, post_id: str, pull_time: str) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = _get_json(url, params={"limit": REDDIT_COMMENT_LIMIT, "depth": 1})
    time.sleep(REQUEST_DELAY_SECONDS)
    if not data or len(data) < 2:
        return []
    records = []
    for item in data[1]["data"]["children"][:REDDIT_COMMENT_LIMIT]:
        if item["kind"] != "t1":
            continue
        c = item["data"]
        body = c.get("body", "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        records.append({
            "id":                 c.get("id", ""),
            "source":             "reddit",
            "subreddit":          subreddit,
            "type":               "comment",
            "author":             c.get("author", "[deleted]"),
            "text":               body,
            "upvotes":            c.get("score", 0),
            "downvotes":          None,
            "upvote_ratio":       None,
            "account_age_days":   -1,
            "account_karma":      -1,
            "post_flair":         None,
            "timestamp_utc":      _parse_timestamp(c["created_utc"]),
            "pull_timestamp_utc": pull_time,
        })
    return records


def search_subreddit(subreddit: str, query: str, pull_time: str, limit: int = 100) -> list[dict]:
    """Search a subreddit for a query going back ~30 days via /search.json."""
    records = []
    collected = 0
    after = None

    while collected < limit:
        batch = min(100, limit - collected)
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q":           query,
            "sort":        "new",
            "t":           "month",
            "limit":       batch,
            "restrict_sr": "true",
        }
        if after:
            params["after"] = after

        data = _get_json(url, params=params)
        time.sleep(REQUEST_DELAY_SECONDS)
        if not data:
            break

        children = data["data"]["children"]
        if not children:
            break

        for item in children:
            if item["kind"] != "t3":
                continue
            p = item["data"]
            if p.get("removed_by_category") or p.get("author") in ("[deleted]", None):
                continue
            post_text = (p.get("title", "") + " " + p.get("selftext", "")).strip()
            records.append({
                "id":                 p["id"],
                "source":             "reddit",
                "subreddit":          subreddit,
                "type":               "post",
                "author":             p.get("author", "[deleted]"),
                "text":               post_text,
                "upvotes":            p.get("score", 0),
                "downvotes":          None,
                "upvote_ratio":       p.get("upvote_ratio"),
                "account_age_days":   -1,
                "account_karma":      -1,
                "post_flair":         p.get("link_flair_text"),
                "timestamp_utc":      _parse_timestamp(p["created_utc"]),
                "pull_timestamp_utc": pull_time,
            })
            comments = fetch_post_comments(subreddit, p["id"], pull_time)
            records.extend(comments)

        collected += len(children)
        after = data["data"].get("after")
        if not after:
            break

    return records


def collect_subreddit_historical(subreddit_name: str) -> list[dict]:
    """For each ticker query, search the subreddit for the past 30 days."""
    pull_time = datetime.now(tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)
    all_records = []
    posts_per_query = max(10, REDDIT_POST_LIMIT // len(SEARCH_QUERIES))

    logger.info(f"Collecting r/{subreddit_name} — {len(SEARCH_QUERIES)} queries x {posts_per_query} posts each")

    for query in SEARCH_QUERIES:
        logger.info(f"  Searching r/{subreddit_name} for '{query}'")
        records = search_subreddit(subreddit_name, query, pull_time, limit=posts_per_query)
        all_records.extend(records)
        logger.info(f"  '{query}': {len(records)} records")
        time.sleep(REQUEST_DELAY_SECONDS * 2)

    logger.info(f"r/{subreddit_name}: {len(all_records)} total records")
    return all_records


def run_reddit_collection() -> pd.DataFrame:
    all_records = []
    for sub in REDDIT_SUBREDDITS:
        try:
            records = collect_subreddit_historical(sub)
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Failed to collect r/{sub}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS * 3)

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No Reddit data collected.")
        return df

    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    logger.info(f"Deduplicated: {before} to {len(df)} records")

    timestamp_tag = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_DATA_DIR, f"reddit_raw_{timestamp_tag}.csv")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path} ({len(df)} records)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    df = run_reddit_collection()
    print(f"\nTotal records: {len(df)}")
    print(df[["subreddit", "type", "timestamp_utc"]].head(10))