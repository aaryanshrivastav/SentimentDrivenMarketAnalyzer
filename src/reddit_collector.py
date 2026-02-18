"""
Stage 0 — Reddit Data Collector (No-Auth JSON API version)
Scrapes posts + comments from r/wallstreetbets, r/stocks, r/investing
using Reddit's public .json endpoint — no API key required.

Output schema (one row per post or comment):
    id, source, subreddit, type (post/comment), author,
    text, upvotes, downvotes, upvote_ratio,
    account_age_days, account_karma, post_flair,
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
    RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Reddit will block requests without a User-Agent header
HEADERS = {"User-Agent": "Mozilla/5.0 (sentiment_finance_scraper/1.0)"}

# Be polite — unauthenticated requests are rate-limited more aggressively
REQUEST_DELAY_SECONDS = 1.5


def _get_json(url: str, params: dict = None, retries: int = 3) -> dict | None:
    """
    GET a Reddit .json URL with retry logic.
    Returns parsed JSON dict, or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 10 * attempt
                logger.warning(f"Rate limited (429). Waiting {wait}s before retry {attempt}/{retries}...")
                time.sleep(wait)
            elif response.status_code == 403:
                logger.error(f"403 Forbidden for URL: {url}. Subreddit may be private.")
                return None
            else:
                logger.warning(f"HTTP {response.status_code} for {url} (attempt {attempt}/{retries})")
                time.sleep(REQUEST_DELAY_SECONDS * attempt)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt}/{retries}: {e}")
            time.sleep(REQUEST_DELAY_SECONDS * attempt)
    return None


def _parse_timestamp(created_utc: float) -> str:
    """Convert a Unix UTC timestamp to a formatted string."""
    return datetime.fromtimestamp(created_utc, tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)


def fetch_post_comments(subreddit: str, post_id: str, pull_time: str) -> list[dict]:
    """
    Fetch top-level comments for a single post using the .json endpoint.
    Returns a list of comment record dicts.
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = _get_json(url, params={"limit": REDDIT_COMMENT_LIMIT, "depth": 1})
    time.sleep(REQUEST_DELAY_SECONDS)

    if not data or len(data) < 2:
        return []

    records = []
    comment_listing = data[1]["data"]["children"]

    for item in comment_listing[:REDDIT_COMMENT_LIMIT]:
        if item["kind"] != "t1":        # t1 = comment; skip MoreComments etc.
            continue
        c = item["data"]
        body = c.get("body", "").strip()
        if not body or body == "[deleted]" or body == "[removed]":
            continue

        records.append({
            "id":                 c.get("id", ""),
            "source":             "reddit",
            "subreddit":          subreddit,
            "type":               "comment",
            "author":             c.get("author", "[deleted]"),
            "text":               body,
            "upvotes":            c.get("score", 0),
            "downvotes":          None,          # Reddit no longer exposes this
            "upvote_ratio":       None,           # not available for comments
            "account_age_days":   -1,             # not in public listing JSON
            "account_karma":      -1,             # not in public listing JSON
            "post_flair":         None,
            "timestamp_utc":      _parse_timestamp(c["created_utc"]),
            "pull_timestamp_utc": pull_time,
        })

    return records


def collect_subreddit_posts(subreddit_name: str) -> list[dict]:
    """
    Scrape the latest posts (up to REDDIT_POST_LIMIT) from a subreddit
    using Reddit's public JSON API, then fetch their comments.
    Returns a flat list of post + comment record dicts.
    """
    pull_time = datetime.now(tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)
    records = []
    collected = 0
    after = None   # pagination cursor

    logger.info(f"Fetching posts from r/{subreddit_name} ...")

    while collected < REDDIT_POST_LIMIT:
        batch_size = min(100, REDDIT_POST_LIMIT - collected)   # Reddit max = 100 per page
        url = f"https://www.reddit.com/r/{subreddit_name}/new.json"
        params = {"limit": batch_size}
        if after:
            params["after"] = after

        data = _get_json(url, params=params)
        time.sleep(REQUEST_DELAY_SECONDS)

        if not data:
            logger.error(f"Failed to fetch posts from r/{subreddit_name}. Stopping.")
            break

        children = data["data"]["children"]
        if not children:
            break   # nothing left to page through

        for item in children:
            if item["kind"] != "t3":    # t3 = link/post
                continue
            p = item["data"]

            # Skip deleted / removed posts
            if p.get("removed_by_category") or p.get("author") in ("[deleted]", None):
                continue

            post_text = (p.get("title", "") + " " + p.get("selftext", "")).strip()

            post_record = {
                "id":                 p["id"],
                "source":             "reddit",
                "subreddit":          subreddit_name,
                "type":               "post",
                "author":             p.get("author", "[deleted]"),
                "text":               post_text,
                "upvotes":            p.get("score", 0),
                "downvotes":          None,
                "upvote_ratio":       p.get("upvote_ratio"),
                "account_age_days":   -1,   # requires extra API call; skipped
                "account_karma":      -1,   # requires extra API call; skipped
                "post_flair":         p.get("link_flair_text"),
                "timestamp_utc":      _parse_timestamp(p["created_utc"]),
                "pull_timestamp_utc": pull_time,
            }
            records.append(post_record)

            # Fetch comments for this post
            comments = fetch_post_comments(subreddit_name, p["id"], pull_time)
            records.extend(comments)
            logger.debug(f"Post {p['id']}: +{len(comments)} comments")

        collected += len(children)
        after = data["data"].get("after")

        if not after:
            break   # no more pages

    logger.info(f"r/{subreddit_name}: collected {len(records)} records (posts + comments)")
    return records


def run_reddit_collection() -> pd.DataFrame:
    """
    Entry point: scrape all configured subreddits, deduplicate, save raw CSV.
    Returns the combined DataFrame.
    """
    all_records = []

    for sub in REDDIT_SUBREDDITS:
        try:
            records = collect_subreddit_posts(sub)
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Failed to collect r/{sub}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS * 2)   # extra pause between subreddits

    df = pd.DataFrame(all_records)

    if df.empty:
        logger.warning("No Reddit data collected — check your internet connection or subreddit names.")
        return df

    # ── Deduplicate on post/comment id ────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    logger.info(f"Deduplicated: {before} → {len(df)} records")

    # ── Save raw CSV ──────────────────────────────────────────────────────
    timestamp_tag = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_DATA_DIR, f"reddit_raw_{timestamp_tag}.csv")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved raw Reddit data → {out_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    df = run_reddit_collection()
    print(df.head())
    print(f"\nTotal records: {len(df)}")