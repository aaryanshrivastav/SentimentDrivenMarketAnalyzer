"""
company_reddit_search.py
========================
Interactive Stage 0 tool — takes a company name (or ticker) as input,
resolves it to a canonical ticker symbol, then scrapes Reddit for posts
and comments mentioning that ticker across configured subreddits.

Usage
-----
# Interactive prompt
python company_reddit_search.py

# Non-interactive (pass name directly)
python company_reddit_search.py "Apple"
python company_reddit_search.py "TSLA"
python company_reddit_search.py "Reliance Industries"

Output
------
data/raw/reddit_{TICKER}_{timestamp}.csv
Columns match reddit_collector.py exactly:
    id, source, subreddit, type, author, text, upvotes, downvotes,
    upvote_ratio, account_age_days, account_karma, post_flair,
    timestamp_utc, pull_timestamp_utc
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Project config — with standalone fallback
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.config import (
        REDDIT_SUBREDDITS, REDDIT_POST_LIMIT, REDDIT_COMMENT_LIMIT,
        RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE,
    )
except ModuleNotFoundError:
    REDDIT_SUBREDDITS  = ["wallstreetbets", "stocks", "investing", "StockMarket", "IndiaInvestments"]
    REDDIT_POST_LIMIT  = 200
    REDDIT_COMMENT_LIMIT = 50
    RAW_DATA_DIR       = os.path.join("data", "raw")
    TIMESTAMP_FORMAT   = "%Y-%m-%d %H:%M:%S"
    TIMEZONE           = timezone.utc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Company → Ticker resolution table
# Covers common names, aliases, abbreviations, and misspellings.
# Add more entries freely — keys are lowercased before lookup.
# ---------------------------------------------------------------------------

COMPANY_TO_TICKER: dict[str, str] = {
    # ── US equities ──────────────────────────────────────────────────────
    "apple":                    "AAPL",
    "apple inc":                "AAPL",
    "aapl":                     "AAPL",
    "tesla":                    "TSLA",
    "tesla inc":                "TSLA",
    "tsla":                     "TSLA",
    "nvidia":                   "NVDA",
    "nvidia corporation":       "NVDA",
    "nvda":                     "NVDA",
    "microsoft":                "MSFT",
    "microsoft corporation":    "MSFT",
    "msft":                     "MSFT",
    "google":                   "GOOGL",
    "alphabet":                 "GOOGL",
    "alphabet inc":             "GOOGL",
    "googl":                    "GOOGL",
    "goog":                     "GOOGL",
    "amazon":                   "AMZN",
    "amazon.com":               "AMZN",
    "amzn":                     "AMZN",
    "meta":                     "META",
    "meta platforms":           "META",
    "facebook":                 "META",
    "meta":                     "META",
    "netflix":                  "NFLX",
    "nflx":                     "NFLX",
    "amd":                      "AMD",
    "advanced micro devices":   "AMD",
    "intel":                    "INTC",
    "intel corporation":        "INTC",
    "intc":                     "INTC",
    "qualcomm":                 "QCOM",
    "qcom":                     "QCOM",
    "broadcom":                 "AVGO",
    "avgo":                     "AVGO",
    "salesforce":               "CRM",
    "crm":                      "CRM",
    "oracle":                   "ORCL",
    "orcl":                     "ORCL",
    "ibm":                      "IBM",
    "jpmorgan":                 "JPM",
    "jp morgan":                "JPM",
    "jpm":                      "JPM",
    "goldman sachs":            "GS",
    "goldman":                  "GS",
    "gs":                       "GS",
    "bank of america":          "BAC",
    "bac":                      "BAC",
    "wells fargo":              "WFC",
    "wfc":                      "WFC",
    "berkshire":                "BRK.B",
    "berkshire hathaway":       "BRK.B",
    "visa":                     "V",
    "mastercard":               "MA",
    "paypal":                   "PYPL",
    "pypl":                     "PYPL",
    "johnson & johnson":        "JNJ",
    "j&j":                      "JNJ",
    "jnj":                      "JNJ",
    "pfizer":                   "PFE",
    "pfe":                      "PFE",
    "exxon":                    "XOM",
    "exxon mobil":              "XOM",
    "xom":                      "XOM",
    "disney":                   "DIS",
    "walt disney":              "DIS",
    "dis":                      "DIS",
    "walmart":                  "WMT",
    "wmt":                      "WMT",
    "target":                   "TGT",
    "tgt":                      "TGT",
    "boeing":                   "BA",
    "ba":                       "BA",
    "uber":                     "UBER",
    "lyft":                     "LYFT",
    "airbnb":                   "ABNB",
    "abnb":                     "ABNB",
    "spotify":                  "SPOT",
    "spot":                     "SPOT",
    "palantir":                 "PLTR",
    "pltr":                     "PLTR",
    "snowflake":                "SNOW",
    "snow":                     "SNOW",
    "coinbase":                 "COIN",
    "coin":                     "COIN",
    "robinhood":                "HOOD",
    "hood":                     "HOOD",
    "arm":                      "ARM",
    "arm holdings":             "ARM",
    # ── Indian equities (NSE) ────────────────────────────────────────────
    "reliance":                 "RELIANCE.NS",
    "reliance industries":      "RELIANCE.NS",
    "ril":                      "RELIANCE.NS",
    "reliance.ns":              "RELIANCE.NS",
    "hdfc bank":                "HDFCBANK.NS",
    "hdfcbank":                 "HDFCBANK.NS",
    "hdfcbank.ns":              "HDFCBANK.NS",
    "hdfc":                     "HDFCBANK.NS",
    "tcs":                      "TCS.NS",
    "tata consultancy":         "TCS.NS",
    "tata consultancy services":"TCS.NS",
    "tcs.ns":                   "TCS.NS",
    "infosys":                  "INFY.NS",
    "infy":                     "INFY.NS",
    "infy.ns":                  "INFY.NS",
    "wipro":                    "WIPRO.NS",
    "wipro.ns":                 "WIPRO.NS",
    "icici bank":               "ICICIBANK.NS",
    "icicibank":                "ICICIBANK.NS",
    "icici":                    "ICICIBANK.NS",
    "hcl":                      "HCLTECH.NS",
    "hcltech":                  "HCLTECH.NS",
    "hcl technologies":         "HCLTECH.NS",
    "bharti airtel":            "BHARTIARTL.NS",
    "airtel":                   "BHARTIARTL.NS",
    "bhartiartl":               "BHARTIARTL.NS",
    "larsen":                   "LT.NS",
    "l&t":                      "LT.NS",
    "larsen and toubro":        "LT.NS",
    "lt.ns":                    "LT.NS",
    "axis bank":                "AXISBANK.NS",
    "axisbank":                 "AXISBANK.NS",
    "kotak":                    "KOTAKBANK.NS",
    "kotak mahindra":           "KOTAKBANK.NS",
    "kotakbank":                "KOTAKBANK.NS",
    "sun pharma":               "SUNPHARMA.NS",
    "sunpharma":                "SUNPHARMA.NS",
    "bajaj finance":            "BAJFINANCE.NS",
    "bajfinance":               "BAJFINANCE.NS",
    "ongc":                     "ONGC.NS",
    "maruti":                   "MARUTI.NS",
    "maruti suzuki":            "MARUTI.NS",
    "asian paints":             "ASIANPAINT.NS",
    "asianpaint":               "ASIANPAINT.NS",
    "titan":                    "TITAN.NS",
    "titan company":            "TITAN.NS",
    "nestle india":             "NESTLEIND.NS",
    "nestle":                   "NESTLEIND.NS",
    "nestleind":                "NESTLEIND.NS",
    "adani":                    "ADANIENT.NS",
    "adani enterprises":        "ADANIENT.NS",
    "adanient":                 "ADANIENT.NS",
    "sbi":                      "SBIN.NS",
    "state bank":               "SBIN.NS",
    "state bank of india":      "SBIN.NS",
    "sbin":                     "SBIN.NS",
}

# ---------------------------------------------------------------------------
# Online fallback — Yahoo Finance search API (no auth required)
# ---------------------------------------------------------------------------

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _yahoo_lookup(company_name: str) -> str | None:
    """
    Query Yahoo Finance search to resolve a company name → ticker.
    Returns the best-match equity ticker, or None on failure.
    """
    try:
        resp = requests.get(
            YAHOO_SEARCH_URL,
            headers=YAHOO_HEADERS,
            params={"q": company_name, "quotesCount": 5, "newsCount": 0},
            timeout=8,
        )
        if resp.status_code != 200:
            return None

        quotes = resp.json().get("quotes", [])
        # Prefer results of type EQUITY, then any match
        for q in quotes:
            if q.get("quoteType") == "EQUITY" and q.get("symbol"):
                logger.info("Yahoo Finance resolved '%s' → %s", company_name, q["symbol"])
                return q["symbol"].upper()
        # Fallback: first result regardless of type
        if quotes and quotes[0].get("symbol"):
            return quotes[0]["symbol"].upper()

    except Exception as exc:
        logger.debug("Yahoo Finance lookup failed: %s", exc)

    return None


def resolve_ticker(company_input: str) -> tuple[str, str]:
    """
    Resolve a free-text company name or raw ticker to a canonical ticker.

    Resolution order
    ----------------
    1. Local lookup table (COMPANY_TO_TICKER) — instant, no network
    2. If input looks like a bare ticker (all caps, ≤6 chars) → use as-is
    3. Yahoo Finance search API — network fallback

    Returns
    -------
    (ticker, market)   e.g. ("AAPL", "US") or ("RELIANCE.NS", "IN")
    Raises ValueError if resolution fails completely.
    """
    raw   = company_input.strip()
    lower = raw.lower()

    # ── 1. Local table ───────────────────────────────────────────────────
    if lower in COMPANY_TO_TICKER:
        ticker = COMPANY_TO_TICKER[lower]
        logger.info("Local table: '%s' → %s", raw, ticker)
        return ticker, _market(ticker)

    # ── 2. Bare-ticker heuristic ─────────────────────────────────────────
    bare = raw.upper().replace("-", "").replace(" ", "")
    if bare.isalpha() and len(bare) <= 6:
        logger.info("Bare-ticker heuristic: '%s' → %s", raw, bare)
        return bare, "US"                  # Assume US; Yahoo will correct

    # ── 3. Yahoo Finance API ─────────────────────────────────────────────
    logger.info("Trying Yahoo Finance for '%s' …", raw)
    ticker = _yahoo_lookup(raw)
    if ticker:
        return ticker, _market(ticker)

    raise ValueError(
        f"Could not resolve '{raw}' to a ticker.\n"
        f"  Try a ticker symbol directly (e.g. 'AAPL') or\n"
        f"  add the mapping to COMPANY_TO_TICKER in this script."
    )


def _market(ticker: str) -> str:
    return "IN" if ticker.upper().endswith((".NS", ".BO")) else "US"


def _strip_suffix(ticker: str) -> str:
    for s in (".NS", ".BO"):
        if ticker.upper().endswith(s):
            return ticker[: -len(s)]
    return ticker


# ---------------------------------------------------------------------------
# Reddit search query builder
# ---------------------------------------------------------------------------

def _build_queries(ticker: str, market: str) -> list[str]:
    """
    Return a ranked list of search queries for this ticker.

    US  tickers → ["$AAPL", "AAPL"]
    IN  tickers → ["RELIANCE", "RELIANCE.NS"]   (no $ prefix for Indian)
    """
    bare = _strip_suffix(ticker)
    if market == "US":
        return [f"${bare}", bare]
    else:
        return [bare, ticker]


# ---------------------------------------------------------------------------
# Reddit HTTP helpers (mirrored from reddit_collector.py)
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 3.0


def _get_json(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 15 * attempt
                logger.warning("429 rate-limited. Waiting %ds (attempt %d/%d).", wait, attempt, retries)
                time.sleep(wait)
            elif resp.status_code == 403:
                logger.error("403 Forbidden: %s — subreddit may be private.", url)
                return None
            else:
                logger.warning("HTTP %d for %s (attempt %d/%d).", resp.status_code, url, attempt, retries)
                time.sleep(REQUEST_DELAY * attempt)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error attempt %d/%d: %s", attempt, retries, exc)
            time.sleep(REQUEST_DELAY * attempt)
    return None


def _parse_timestamp(created_utc: float) -> str:
    return datetime.fromtimestamp(created_utc, tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)


# ---------------------------------------------------------------------------
# Comment fetcher
# ---------------------------------------------------------------------------

def _fetch_comments(subreddit: str, post_id: str, pull_time: str) -> list[dict]:
    url  = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = _get_json(url, params={"limit": REDDIT_COMMENT_LIMIT, "depth": 1})
    time.sleep(REQUEST_DELAY)
    if not data or len(data) < 2:
        return []

    records = []
    for item in data[1]["data"]["children"][:REDDIT_COMMENT_LIMIT]:
        if item["kind"] != "t1":
            continue
        c    = item["data"]
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


# ---------------------------------------------------------------------------
# Subreddit search
# ---------------------------------------------------------------------------

def _search_subreddit(
    subreddit: str,
    query: str,
    pull_time: str,
    limit: int = 100,
    fetch_comments: bool = True,
) -> list[dict]:
    """Search one subreddit for `query`, paginate up to `limit` posts."""
    records   = []
    collected = 0
    after     = None

    while collected < limit:
        batch  = min(100, limit - collected)
        url    = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q":           query,
            "sort":        "new",
            "t":           "year",          # go back up to a year
            "limit":       batch,
            "restrict_sr": "true",
        }
        if after:
            params["after"] = after

        data = _get_json(url, params=params)
        time.sleep(REQUEST_DELAY)
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

            if fetch_comments:
                comments = _fetch_comments(subreddit, p["id"], pull_time)
                records.extend(comments)

        collected += len(children)
        after      = data["data"].get("after")
        if not after:
            break

    return records


# ---------------------------------------------------------------------------
# Per-ticker Reddit collection
# ---------------------------------------------------------------------------

def collect_for_ticker(
    ticker: str,
    market: str,
    subreddits: list[str] | None = None,
    posts_per_query: int = 50,
    fetch_comments: bool = True,
) -> pd.DataFrame:
    """
    Search all configured subreddits for posts mentioning `ticker`.

    Parameters
    ----------
    ticker          : Canonical ticker, e.g. "AAPL" or "RELIANCE.NS"
    market          : "US" or "IN"
    subreddits      : Override list; defaults to REDDIT_SUBREDDITS from config
    posts_per_query : Max posts per (subreddit × query) combination
    fetch_comments  : Whether to pull top comments for each post

    Returns
    -------
    Deduplicated pd.DataFrame in the reddit_collector schema.
    """
    pull_time  = datetime.now(tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)
    subs       = subreddits or REDDIT_SUBREDDITS
    queries    = _build_queries(ticker, market)
    all_records: list[dict] = []

    logger.info(
        "Collecting Reddit data for %s [market=%s]  |  "
        "%d subreddits × %d queries × %d posts",
        ticker, market, len(subs), len(queries), posts_per_query,
    )

    for subreddit in subs:
        for query in queries:
            logger.info("  r/%-20s  query='%s'", subreddit, query)
            try:
                records = _search_subreddit(
                    subreddit, query, pull_time,
                    limit=posts_per_query,
                    fetch_comments=fetch_comments,
                )
                all_records.extend(records)
                logger.info("    → %d records", len(records))
            except Exception as exc:
                logger.error("  r/%s query='%s' failed: %s", subreddit, query, exc)

            time.sleep(REQUEST_DELAY * 2)

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No Reddit records found for %s.", ticker)
        return df

    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Deduplicated: %d → %d records for %s", before, len(df), ticker)
    return df


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, ticker: str) -> str:
    """Write df to data/raw/reddit_{ticker}_{timestamp}.csv; return path."""
    safe_ticker   = ticker.replace(".", "_")
    timestamp_tag = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    out_path      = os.path.join(RAW_DATA_DIR, f"reddit_{safe_ticker}_{timestamp_tag}.csv")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved → %s  (%d records)", out_path, len(df))
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Search Reddit for posts about a company.\n"
            "Accepts a company name (e.g. 'Apple', 'Reliance Industries')\n"
            "or a ticker symbol (e.g. 'AAPL', 'RELIANCE.NS')."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "company",
        nargs="?",
        default=None,
        help="Company name or ticker (prompted interactively if omitted)",
    )
    p.add_argument(
        "--subreddits", "-s",
        nargs="+",
        default=None,
        metavar="SUB",
        help="Override subreddits to search (e.g. --subreddits stocks investing)",
    )
    p.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        metavar="N",
        help="Max posts per subreddit/query combination (default: 50)",
    )
    p.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip fetching comments (faster)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Get company input ────────────────────────────────────────────────
    company_input = args.company
    if not company_input:
        print("\n" + "═" * 55)
        print("  Reddit Company Search — Stage 0 Collector")
        print("═" * 55)
        print("  Enter a company name or ticker symbol.")
        print("  Examples: Apple | TSLA | Reliance Industries | TCS\n")
        company_input = input("  Company / Ticker: ").strip()
        if not company_input:
            print("  No input provided. Exiting.")
            sys.exit(0)

    # ── Resolve ticker ───────────────────────────────────────────────────
    print()
    try:
        ticker, market = resolve_ticker(company_input)
    except ValueError as exc:
        print(f"  ✗ {exc}")
        sys.exit(1)

    print(f"  ✓ Resolved  →  {ticker}  [market={market}]")
    print(f"  Subreddits  →  {args.subreddits or REDDIT_SUBREDDITS}")
    print(f"  Limit       →  {args.limit} posts per (subreddit × query)")
    print(f"  Comments    →  {'no' if args.no_comments else 'yes'}")
    print()

    # ── Collect ──────────────────────────────────────────────────────────
    df = collect_for_ticker(
        ticker=ticker,
        market=market,
        subreddits=args.subreddits,
        posts_per_query=args.limit,
        fetch_comments=not args.no_comments,
    )

    if df.empty:
        print(f"\n  No posts found for {ticker}. Try a different subreddit or a broader query.")
        sys.exit(0)

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = save(df, ticker)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  Ticker          : {ticker}")
    print(f"  Total records   : {len(df)}")
    print(f"  Posts           : {(df['type'] == 'post').sum()}")
    print(f"  Comments        : {(df['type'] == 'comment').sum()}")
    print(f"  Output          : {out_path}")
    print(f"{'═' * 55}")
    print()
    print("Top 10 records:")
    print(
        df[["subreddit", "type", "author", "upvotes", "timestamp_utc", "text"]]
        .assign(text=df["text"].str[:80] + "…")
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()