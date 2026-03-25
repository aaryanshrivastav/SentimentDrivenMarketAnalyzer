"""
Stage 0 — Kaggle & HuggingFace Dataset Collector
==================================================
Pulls two pre-labelled, large-scale tweet/finance datasets and normalises
them to the project's shared output schema (identical to reddit_collector.py).

Sources
-------
KAGGLE DATASETS
1. "Stock Tweets for Sentiment Analysis and Prediction"  (equinxx/...)
   ~500 K tweets, tickers AAPL TSLA NVDA MSFT + others
   Auth: kaggle.json (KAGGLE_USERNAME + KAGGLE_KEY env-vars)

2. "Tweets of Indian Stocks from StockTwits"  (rutviknelluri/...)
   Indian market stock-tweet dataset
   Auth: kaggle.json

HUGGINGFACE DATASETS
3. "twitter-financial-news-sentiment"  (zeroshot/...)
   ~20 K labelled finance tweets
   Auth: none (public, `datasets` library)

4. "StockTwits with Emoji"  (ElKulako/stocktwits-emoji)
   StockTwits posts with emoji sentiment indicators
   Auth: none (public, `datasets` library)

5. "Twitter Financial News"  (lukecarlate/twitter_financial_news)
   Financial news and commentary from Twitter
   Auth: none (public, `datasets` library)

Output schema — exact match with reddit_collector.py
------------------------------------------------------
id, source, subreddit, type, author, text, upvotes, downvotes,
upvote_ratio, account_age_days, account_karma, post_flair,
timestamp_utc, pull_timestamp_utc

Extra columns added (not in reddit schema, appended at right)
--------------------------------------------------------------
sentiment_label   — original label from source dataset  (if present)
sentiment_score   — numeric score / confidence           (if present)
"""

import os
import sys
import logging
import hashlib
import zipfile
import tempfile
from datetime import datetime, timezone

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project config import — with standalone fallback
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.config import (
        US_TICKERS, IN_TICKERS,
        RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE,
    )
except ModuleNotFoundError:
    import zoneinfo
    US_TICKERS       = ["AAPL", "TSLA", "NVDA", "MSFT"]
    IN_TICKERS       = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"]
    RAW_DATA_DIR     = os.path.join("data", "raw")
    TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
    TIMEZONE         = timezone.utc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kaggle datasets
KAGGLE_DATASETS = [
    {
        "slug": "equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
        "preferred_csv": "stock_tweets.csv",
        "source": "kaggle",
        "source_tag": "kaggle_us",
    },
    {
        "slug": "rutviknelluri/tweets-of-indian-stocks-from-stocktwits",
        "preferred_csv": "",
        "source": "kaggle_india_stocktwits",
        "source_tag": "kaggle_india",
    },
]

# HuggingFace datasets
HF_DATASETS = [
    {
        "name": "zeroshot/twitter-financial-news-sentiment",
        "splits": ["train", "validation"],
        "source": "huggingface",
        "source_tag": "hf_twitter_news",
    },
    {
        "name": "ElKulako/stocktwits-emoji",
        "splits": ["train"],  # Adjust based on dataset splits availability
        "source": "huggingface",
        "source_tag": "hf_stocktwits_emoji",
    },
    {
        "name": "lukecarlate/twitter_financial_news",
        "splits": ["train"],  # Adjust based on dataset splits availability
        "source": "huggingface",
        "source_tag": "hf_twitter_financial",
    },
]

# Ticker helpers
_US_SET = {t.upper() for t in US_TICKERS}
_IN_SET = {t.upper() for t in IN_TICKERS}


def _strip_suffix(ticker: str) -> str:
    for s in (".NS", ".BO"):
        if ticker.upper().endswith(s):
            return ticker[: -len(s)]
    return ticker


def _market(ticker: str) -> str:
    return "IN" if ticker.upper().endswith((".NS", ".BO")) else "US"


# Canonical output column order (reddit_collector schema + extras)
SCHEMA_COLS = [
    "id", "source", "subreddit", "type", "author", "text",
    "upvotes", "downvotes", "upvote_ratio",
    "account_age_days", "account_karma",
    "post_flair", "timestamp_utc", "pull_timestamp_utc",
    # extras
    "sentiment_label", "sentiment_score",
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pull_time() -> str:
    return datetime.now(tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)


def _stable_id(source_tag: str, raw_id) -> str:
    """
    Generate a stable string id.
    Uses the raw id directly if non-empty; otherwise hashes source_tag + content.
    """
    if raw_id and str(raw_id).strip():
        return f"{source_tag}_{raw_id}"
    return hashlib.md5(str(raw_id).encode()).hexdigest()


def _normalise_timestamp(raw_ts, pull_time: str) -> str:
    """
    Try several common timestamp formats and return TIMESTAMP_FORMAT string.
    Falls back to pull_time on any failure.
    """
    if pd.isna(raw_ts) or not str(raw_ts).strip():
        return pull_time

    raw = str(raw_ts).strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%a %b %d %H:%M:%S +0000 %Y",   # Twitter created_at
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt.astimezone(TIMEZONE).strftime(TIMESTAMP_FORMAT)
        except ValueError:
            continue

    logger.debug("Could not parse timestamp '%s' — using pull_time.", raw)
    return pull_time


def _empty_record(pull_time: str) -> dict:
    """Return a blank schema-aligned dict to be populated by each source."""
    return {
        "id":                 "",
        "source":             "",
        "subreddit":          None,
        "type":               "post",
        "author":             "[unknown]",
        "text":               "",
        "upvotes":            None,
        "downvotes":          None,
        "upvote_ratio":       None,
        "account_age_days":   -1,
        "account_karma":      -1,
        "post_flair":         None,
        "timestamp_utc":      pull_time,
        "pull_timestamp_utc": pull_time,
        "sentiment_label":    None,
        "sentiment_score":    None,
    }


def _normalize_sentiment_label(raw_label) -> str | None:
    """
    Standardize sentiment labels while preserving unlabeled rows as None.
    """
    if raw_label is None or pd.isna(raw_label):
        return None

    txt = str(raw_label).strip()
    if not txt:
        return None

    low = txt.lower()
    if low in {"none", "nan", "na", "n/a", "null", "unlabeled", "unlabelled", "unknown", ""}:
        return None

    if low in {"bullish", "positive", "pos", "buy", "1"}:
        return "Bullish"
    if low in {"bearish", "negative", "neg", "sell", "-1"}:
        return "Bearish"
    if low in {"neutral", "neu", "0"}:
        return "Neutral"

    return txt


def _reorder(df: pd.DataFrame) -> pd.DataFrame:
    """Force canonical column order; add missing cols as None."""
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in df.columns if c not in SCHEMA_COLS]
    return df[SCHEMA_COLS + extra]


# ---------------------------------------------------------------------------
# SOURCE 1 — Kaggle
# ---------------------------------------------------------------------------

def _kaggle_credentials() -> tuple[str, str] | None:
    """
    Return (username, key) from environment variables or ~/.kaggle/kaggle.json.
    Returns None if neither is found (caller should log and skip).
    """
    username = os.environ.get("KAGGLE_USERNAME", "")
    key      = os.environ.get("KAGGLE_KEY", "")

    if username and key:
        return username, key

    kaggle_json = os.path.expanduser(os.path.join("~", ".kaggle", "kaggle.json"))
    if os.path.exists(kaggle_json):
        import json
        with open(kaggle_json) as fh:
            creds = json.load(fh)
        return creds.get("username", ""), creds.get("key", "")

    return None


def _download_kaggle_zip(username: str, key: str, dest_dir: str, dataset_slug: str) -> str | None:
    """
    Download the dataset zip via Kaggle's public REST API.
    Returns local path to the zip file, or None on failure.

    API endpoint:
        GET https://www.kaggle.com/api/v1/datasets/download/{slug}
    Auth: HTTP Basic with (username, api_key)
    """
    owner, dataset = dataset_slug.split("/")
    url   = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"
    auth  = (username, key)

    logger.info("Downloading Kaggle dataset: %s", dataset_slug)
    try:
        resp = requests.get(url, auth=auth, stream=True, timeout=120)

        if resp.status_code == 401:
            logger.error("Kaggle: 401 Unauthorised — check KAGGLE_USERNAME / KAGGLE_KEY.")
            return None
        if resp.status_code == 403:
            logger.error("Kaggle: 403 Forbidden — accept dataset terms on kaggle.com first.")
            return None
        if resp.status_code != 200:
            logger.error("Kaggle: HTTP %d for %s", resp.status_code, url)
            return None

        safe_name = dataset_slug.replace("/", "__")
        zip_path = os.path.join(dest_dir, f"{safe_name}.zip")
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):   # 1 MB chunks
                fh.write(chunk)

        logger.info("Kaggle zip saved → %s", zip_path)
        return zip_path

    except requests.exceptions.RequestException as exc:
        logger.error("Kaggle download failed: %s", exc)
        return None


def _find_csv_in_zip(zip_path: str, preferred_name: str) -> str | None:
    """Extract the target CSV from the zip; return path to extracted file."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        logger.debug("Zip contents: %s", names)

        # Try preferred filename first, then any .csv
        target = next(
            (n for n in names if os.path.basename(n).lower() == preferred_name.lower()),
            next((n for n in names if n.lower().endswith(".csv")), None),
        )

        if not target:
            logger.error("No CSV found in Kaggle zip. Contents: %s", names)
            return None

        extract_dir = os.path.dirname(zip_path)
        zf.extract(target, extract_dir)
        return os.path.join(extract_dir, target)


def _build_kaggle_records(
    raw_df: pd.DataFrame,
    pull_time: str,
    source_name: str = "kaggle",
    source_tag: str = "kaggle",
) -> list[dict]:
    """
    Normalise Kaggle stock-tweets CSV → schema records.

    Expected Kaggle CSV columns (may vary by dataset version)
    ----------------------------------------------------------
    Tweet                — tweet text
    Date                 — tweet date  (YYYY-MM-DD or similar)
    Stock Name / Ticker  — stock symbol (e.g. AAPL)
    Company Name         — company display name

    Sentiment columns (if present): sentiment, label, score, etc.
    """
    records = []

    # ── Flexible column detection ────────────────────────────────────────
    col_map: dict[str, str | None] = {
        "text":      next((c for c in raw_df.columns if c.lower() in
                           ("tweet", "text", "body", "content")), None),
        "date":      next((c for c in raw_df.columns if c.lower() in
                           ("date", "datetime", "created_at", "timestamp")), None),
        "ticker":    next((c for c in raw_df.columns if c.lower() in
                           ("ticker", "stock name", "stock_name", "symbol")), None),
        "sentiment": next((c for c in raw_df.columns if c.lower() in
                           ("sentiment", "label", "sentiment_label", "class")), None),
        "score":     next((c for c in raw_df.columns if c.lower() in
                           ("score", "confidence", "sentiment_score", "probability")), None),
    }

    logger.info("Kaggle column map: %s", col_map)
    if not col_map["text"]:
        logger.error("Cannot find text column in Kaggle CSV — columns: %s", list(raw_df.columns))
        return []

    for i, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Processing Kaggle records"):
        rec = _empty_record(pull_time)

        text = str(row[col_map["text"]]).strip() if col_map["text"] else ""
        if not text or text.lower() in ("nan", "[deleted]", "[removed]"):
            continue

        ticker_raw = (
            str(row[col_map["ticker"]]).strip().upper()
            if col_map["ticker"] else ""
        )

        rec["id"]              = _stable_id(source_tag, i)
        rec["source"]          = source_name
        rec["text"]            = text
        rec["post_flair"]      = ticker_raw or None
        rec["timestamp_utc"]   = _normalise_timestamp(
                                     row[col_map["date"]] if col_map["date"] else None,
                                     pull_time,
                                 )
        rec["sentiment_label"] = (
            _normalize_sentiment_label(row[col_map["sentiment"]]) if col_map["sentiment"] else None
        )
        rec["sentiment_score"] = (
            row[col_map["score"]] if col_map["score"] else None
        )

        records.append(rec)

    return records


def collect_kaggle(pull_time: str) -> list[dict]:
    """Download and normalise all configured Kaggle datasets."""
    creds = _kaggle_credentials()
    if not creds:
        logger.warning(
            "Kaggle credentials not found.\n"
            "  Set KAGGLE_USERNAME + KAGGLE_KEY env-vars, OR\n"
            "  place ~/.kaggle/kaggle.json on disk.\n"
            "  Skipping Kaggle source."
        )
        return []

    username, key = creds
    all_records: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        for ds in tqdm(KAGGLE_DATASETS, desc="Loading Kaggle datasets"):
            dataset_slug = ds["slug"]
            preferred_csv = ds.get("preferred_csv", "")
            source_name = ds.get("source", "kaggle")
            source_tag = ds.get("source_tag", source_name)

            zip_path = _download_kaggle_zip(username, key, tmp, dataset_slug)
            if not zip_path:
                continue

            csv_path = _find_csv_in_zip(zip_path, preferred_csv)
            if not csv_path:
                continue

            logger.info("Reading Kaggle CSV (%s): %s", source_name, csv_path)
            try:
                raw_df = pd.read_csv(csv_path, low_memory=False)
            except Exception as exc:
                logger.error("Failed to read Kaggle CSV (%s): %s", source_name, exc)
                continue

            logger.info("Kaggle raw shape (%s): %s", source_name, raw_df.shape)
            ds_records = _build_kaggle_records(
                raw_df,
                pull_time,
                source_name=source_name,
                source_tag=source_tag,
            )
            logger.info("Kaggle %s: %d valid records after normalisation.", source_name, len(ds_records))
            all_records.extend(ds_records)

    logger.info("Kaggle total: %d records across %d datasets.", len(all_records), len(KAGGLE_DATASETS))
    return all_records


# ---------------------------------------------------------------------------
# SOURCE 2 — HuggingFace
# ---------------------------------------------------------------------------

# HuggingFace dataset-specific label mappings
# Each dataset may use different label formats (integer, emoji, text, etc.)
HF_LABEL_MAPS = {
    # zeroshot/twitter-financial-news-sentiment: integer labels 0=Bearish, 1=Bullish, 2=Neutral
    "zeroshot/twitter-financial-news-sentiment": {0: "Bearish", 1: "Bullish", 2: "Neutral"},
    
    # ElKulako/stocktwits-emoji: may use emoji or text labels (auto-detect by structure)
    "ElKulako/stocktwits-emoji": {},  # Will auto-detect; fallback to text normalization
    
    # lukecarlate/twitter_financial_news: likely text labels (Bullish, Bearish, Neutral)
    "lukecarlate/twitter_financial_news": {},  # Will auto-detect; fallback to text normalization
}

# Default fallback mapping
HF_LABEL_MAP_DEFAULT = {0: "Bearish", 1: "Bullish", 2: "Neutral"}


def _hf_available() -> bool:
    """Return True if the `datasets` library is importable."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "`datasets` library not installed. "
            "Run: pip install datasets\n"
            "Skipping HuggingFace source."
        )
        return False


def _load_hf_split(dataset_name: str, split: str) -> pd.DataFrame | None:
    """Load one split of an HF dataset as a pandas DataFrame."""
    from datasets import load_dataset  # local import — only if library present
    try:
        logger.info("Loading HuggingFace split: %s/%s", dataset_name, split)
        ds = load_dataset(dataset_name, split=split)
        return ds.to_pandas()
    except Exception as exc:
        logger.warning("HuggingFace dataset '%s' split '%s' failed: %s", dataset_name, split, exc)
        return None


def _build_hf_records(raw_df: pd.DataFrame, dataset_name: str, split: str, pull_time: str) -> list[dict]:
    """
    Normalise HuggingFace dataset → schema records.
    
    Handles dataset-specific column naming and label formats:
    - zeroshot/twitter-financial-news-sentiment: integer labels (0=Bearish, 1=Bullish, 2=Neutral)
    - ElKulako/stocktwits-emoji: emoji or text labels (auto-detect)
    - lukecarlate/twitter_financial_news: text labels (Bullish, Bearish, Neutral)
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw data from HF dataset load_dataset().to_pandas()
    dataset_name : str
        HF dataset identifier (e.g., 'ElKulako/stocktwits-emoji')
    split : str
        Dataset split name (e.g., 'train', 'validation')
    pull_time : str
        Timestamp of collection in TIMESTAMP_FORMAT
    
    Returns
    -------
    list[dict]
        List of schema-aligned records
    """
    records = []
    
    if raw_df.empty:
        logger.warning("HF dataset '%s' split '%s' is empty", dataset_name, split)
        return []
    
    # Dataset-specific column detection
    logger.info("Detecting columns in %s/%s. Available: %s", dataset_name, split, list(raw_df.columns)[:5])
    
    # Text column — highly flexible to handle tweets, posts, sentences, news
    text_col = None
    for candidate in ["text", "tweet", "sentence", "content", "body", "post", "message"]:
        if candidate in raw_df.columns:
            text_col = candidate
            break
    
    if not text_col:
        # Last resort: find longest string column
        for col in raw_df.columns:
            if raw_df[col].dtype == 'object':
                text_col = col
                break
    
    if not text_col:
        logger.error("HF '%s' split '%s': no text column detected. Columns: %s", 
                    dataset_name, split, list(raw_df.columns))
        return []
    
    logger.info("Detected text column: '%s'", text_col)
    
    # Label column — dataset-specific detection
    label_col = None
    for candidate in ["label", "sentiment", "class", "sentiment_label", "sentiment_class"]:
        if candidate in raw_df.columns:
            label_col = candidate
            break
    
    # Get dataset-specific label mapping
    label_map = HF_LABEL_MAPS.get(dataset_name, {})
    if not label_map:
        label_map = {}  # Empty dict means auto-detect via text normalization
    
    logger.info("Processing %d rows from %s/%s", len(raw_df), dataset_name, split)
    
    for i, row in tqdm(raw_df.iterrows(), total=len(raw_df), 
                       desc=f"Processing {dataset_name.split('/')[-1]}/{split}"):
        text = str(row[text_col]).strip()
        if not text or text.lower() in ("nan", "[deleted]", "[removed]", ""):
            continue
        
        # Extract and normalize label
        label_raw = row[label_col] if label_col else None
        label_str = None
        
        if label_raw is not None and not pd.isna(label_raw):
            # Try integer→label mapping first (if available)
            if isinstance(label_raw, (int, float)) and label_map:
                label_int = int(label_raw)
                if label_int in label_map:
                    label_str = _normalize_sentiment_label(label_map[label_int])
                else:
                    # Integer but no mapping — try string conversion + normalization
                    label_str = _normalize_sentiment_label(str(label_raw))
            else:
                # String or emoji — normalize directly
                label_str = _normalize_sentiment_label(str(label_raw))
        
        # Build record
        rec = _empty_record(pull_time)
        rec["id"]              = _stable_id(f"hf_{dataset_name.replace('/', '_')}_{split}", i)
        rec["source"]          = "huggingface"
        rec["subreddit"]       = dataset_name.split("/")[-1]  # Store dataset name as pseudo-subreddit
        rec["text"]            = text
        rec["sentiment_label"] = label_str
        
        # Try to extract timestamps if present
        for ts_col in ["created_at", "timestamp", "date", "datetime"]:
            if ts_col in row and not pd.isna(row[ts_col]):
                rec["timestamp_utc"] = _normalise_timestamp(row[ts_col], pull_time)
                break
        
        records.append(rec)
    
    logger.info("HF %s/%s: %d valid records after filtering", dataset_name, split, len(records))
    return records

    return records


def collect_huggingface(pull_time: str) -> list[dict]:
    """Load and normalise all configured HuggingFace datasets and splits."""
    if not _hf_available():
        return []

    records: list[dict] = []
    total_splits = sum(len(ds.get("splits", [])) for ds in HF_DATASETS)

    for ds in tqdm(HF_DATASETS, desc="Loading HuggingFace datasets"):
        dataset_name = ds["name"]
        splits = ds.get("splits", ["train"])
        source_tag = ds.get("source_tag", "hf_unknown")

        for split in splits:
            raw_df = _load_hf_split(dataset_name, split)
            if raw_df is None:
                logger.warning("Skipping %s/%s (could not load)", dataset_name, split)
                continue

            logger.info("HF %s split '%s' raw shape: %s", dataset_name, split, raw_df.shape)
            split_records = _build_hf_records(raw_df, dataset_name, split, pull_time)
            logger.info("HF %s split '%s': %d valid records.", dataset_name, split, len(split_records))
            records.extend(split_records)

    logger.info("HuggingFace total: %d records across %d datasets and ~%d splits.", len(records), len(HF_DATASETS), total_splits)
    return records


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_kaggle_hf_collection() -> pd.DataFrame:
    """
    Collect from Kaggle and HuggingFace, merge, deduplicate, save.

    Returns
    -------
    pd.DataFrame
        Merged, deduplicated, schema-aligned records.
        Written to data/raw/kaggle_hf_raw_{timestamp}.csv.
    """
    pt = _pull_time()
    all_records: list[dict] = []

    logger.info("═══ Kaggle + HuggingFace collection started ═══")

    # ── Kaggle ────────────────────────────────────────────────────────────
    logger.info("── Source 1 / 2 : Kaggle (multi-dataset) ──")
    kaggle_records = collect_kaggle(pt)
    all_records.extend(kaggle_records)
    logger.info("Kaggle: %d records", len(kaggle_records))

    # ── HuggingFace ───────────────────────────────────────────────────────
    logger.info("── Source 2 / 2 : HuggingFace ──")
    hf_records = collect_huggingface(pt)
    all_records.extend(hf_records)
    logger.info("HuggingFace: %d records", len(hf_records))

    # ── Build DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No data collected from either source.")
        return df

    # ── Deduplication (by synthetic id) ──────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Deduplicated: %d → %d records (removed %d)", before, len(df), before - len(df))

    # ── Canonical column order ────────────────────────────────────────────
    df = _reorder(df)

    # ── Persist ──────────────────────────────────────────────────────────
    tag      = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_DATA_DIR, f"kaggle_hf_raw_{tag}.csv")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved → %s  (%d records)", out_path, len(df))

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df = run_kaggle_hf_collection()

    print(f"\n{'═' * 60}")
    print(f"  Total records : {len(df)}")
    print(f"{'═' * 60}")

    if not df.empty:
        print("\nRecords by source:")
        print(df["source"].value_counts().to_string())

        print("\nSentiment label distribution:")
        print(df["sentiment_label"].value_counts(dropna=False).to_string())

        print("\nSample (10 rows):")
        print(
            df[["source", "post_flair", "sentiment_label", "text"]]
            .head(10)
            .to_string(index=False)
        )