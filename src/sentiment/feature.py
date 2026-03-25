"""
STAGE 2 — MARKET FEATURE ENGINEERING
Reads raw market CSVs and produces a fused feature matrix ready for Stage 3.

Column contracts (from your data):
  prices_*  : timestamp_utc, Open, High, Low, Close, Volume, ticker
  vix_*     : timestamp_utc, vix_close, index
  india_vix : timestamp_utc, vix_close, index
  indices   : timestamp_utc, Close, index_name, ticker
  earnings  : expected columns handled below
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Use absolute path based on file location
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "market"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & MERGE PRICE FILES
# Two pulls per ticker (044917, 050259) → merge and deduplicate
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(data_dir: Path, ticker_filter: list[str] | None = None) -> pd.DataFrame:
    """
    Loads all prices_*.csv files, merges duplicate pulls per ticker,
    resamples to 1-hour OHLCV bars.
    """
    frames = []
    for f in sorted(data_dir.glob("prices_*.csv")):
        if "all" in f.name:
            continue                        # skip the combined dump for now
        df = pd.read_csv(f, parse_dates=["timestamp_utc"])
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No prices_*.csv files found.")

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    raw.columns = [c.strip() for c in raw.columns]

    if ticker_filter:
        allowed = {str(t).strip().upper() for t in ticker_filter if str(t).strip()}
        raw = raw[raw["ticker"].astype(str).str.upper().isin(allowed)].copy()

    # Deduplicate: same ticker + same timestamp from two pulls → keep one
    raw = raw.drop_duplicates(subset=["timestamp_utc", "ticker"])
    raw = raw.sort_values(["ticker", "timestamp_utc"])

    # Resample to 1-hour bars (in case data is minute-level)
    resampled = (
        raw.groupby("ticker")
           .apply(_resample_ohlcv)
           .reset_index(drop=True)
    )

    logger.info(f"Prices loaded: {resampled['ticker'].nunique()} tickers, "
                f"{len(resampled)} hourly bars")
    return resampled


def _resample_ohlcv(group: pd.DataFrame) -> pd.DataFrame:
    group = group.set_index("timestamp_utc").sort_index()
    ticker = group["ticker"].iloc[0]
    ohlcv = group[["Open","High","Low","Close","Volume"]].resample("1h").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])
    ohlcv["ticker"] = ticker
    ohlcv = ohlcv.reset_index()
    return ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TECHNICAL INDICATORS (per ticker)
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI, MACD, Bollinger Bands, ATR, OBV, returns, and volume features.
    All computed per-ticker to avoid cross-contamination.
    """
    out = df.groupby("ticker", group_keys=False).apply(_compute_technicals)
    return out.reset_index(drop=True)


def _compute_technicals(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()
    c = g["Close"]
    v = g["Volume"]
    h = g["High"]
    l = g["Low"]

    # ── Returns ──────────────────────────────────────────────────────
    g["return_1h"]  = c.pct_change(1, fill_method=None)
    g["return_3h"]  = c.pct_change(3, fill_method=None)
    g["return_24h"] = c.pct_change(24, fill_method=None)

    # ── RSI (14-period) ──────────────────────────────────────────────
    delta    = c.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    g["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────
    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    g["macd"]         = ema12 - ema26
    g["macd_signal"]  = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"]    = g["macd"] - g["macd_signal"]

    # ── Bollinger Bands (20-period, 2σ) ──────────────────────────────
    sma20             = c.rolling(20).mean()
    std20             = c.rolling(20).std()
    g["bb_upper"]     = sma20 + 2 * std20
    g["bb_lower"]     = sma20 - 2 * std20
    g["bb_width"]     = (g["bb_upper"] - g["bb_lower"]) / sma20
    g["bb_position"]  = (c - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-9)

    # ── ATR (Average True Range, 14-period) ──────────────────────────
    prev_close  = c.shift(1)
    tr          = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)
    g["atr_14"] = tr.rolling(14).mean()

    # ── Volume features ───────────────────────────────────────────────
    g["volume_change"] = v.pct_change(1, fill_method=None)
    g["volume_ma_ratio"] = v / v.rolling(24).mean()   # current vs 24h avg

    # ── OBV (On Balance Volume) ───────────────────────────────────────
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    g["obv"] = obv

    # ── Volatility ────────────────────────────────────────────────────
    g["volatility_24h"] = g["return_1h"].rolling(24).std()

    # ── Clean up inf/NaN values ───────────────────────────────────────
    # Replace inf with NaN, then fill NaN with 0 for most indicators
    g = g.replace([np.inf, -np.inf], np.nan)
    
    # For ratio-based indicators, 0 is a reasonable default
    ratio_cols = ["volume_change", "volume_ma_ratio", "bb_width", "bb_position"]
    for col in ratio_cols:
        if col in g.columns:
            g[col] = g[col].fillna(0)
    
    # For price-based indicators, forward fill then backfill
    price_cols = ["rsi_14", "macd", "macd_signal", "macd_hist", 
                  "bb_upper", "bb_lower", "atr_14", "volatility_24h"]
    for col in price_cols:
        if col in g.columns:
            g[col] = g[col].ffill().bfill().fillna(0)
    
    # For OBV, no action needed (cumsum of real values)
    # But ensure no NaN from initial row
    if "obv" in g.columns:
        g["obv"] = g["obv"].fillna(0)
    
    # Clip extreme values to reasonable bounds
    for col in g.columns:
        if col not in ["timestamp_utc", "ticker", "target"]:
            if g[col].dtype in [np.float64, np.float32]:
                # Clip to 5 std deviations from mean
                mean_val = g[col].mean()
                std_val = g[col].std()
                if std_val > 0:
                    lower_bound = mean_val - 5 * std_val
                    upper_bound = mean_val + 5 * std_val
                    g[col] = g[col].clip(lower=lower_bound, upper=upper_bound)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LOAD VIX & INDICES (market-wide fear / context)
# ─────────────────────────────────────────────────────────────────────────────

def load_vix(data_dir: Path) -> pd.DataFrame:
    """
    Loads US VIX and India VIX, returns a combined table keyed by timestamp.
    """
    frames = []
    for f in data_dir.glob("*vix*.csv"):
        df = pd.read_csv(f, parse_dates=["timestamp_utc"])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        frames.append(df)

    if not frames:
        logger.warning("No VIX files found.")
        return pd.DataFrame(columns=["timestamp_utc","vix_close","index"])

    vix = pd.concat(frames, ignore_index=True).drop_duplicates()
    vix = vix.sort_values("timestamp_utc")

    # Pivot so each index becomes its own column
    vix_pivot = vix.pivot_table(
        index="timestamp_utc", columns="index", values="vix_close", aggfunc="last"
    ).reset_index()
    vix_pivot.columns.name = None
    vix_pivot.columns = [
        "timestamp_utc" if c == "timestamp_utc" else f"vix_{str(c).lower().replace(' ','_')}"
        for c in vix_pivot.columns
    ]
    return vix_pivot


def load_indices(data_dir: Path) -> pd.DataFrame:
    """
    Loads index closes (S&P 500, NIFTY etc.) — used as market context features.
    """
    f = data_dir / next(
        (x.name for x in data_dir.glob("indices*.csv")), ""
    )
    if not f.exists():
        logger.warning("No indices file found.")
        return pd.DataFrame()

    df = pd.read_csv(f, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.drop_duplicates(subset=["timestamp_utc","ticker"])

    # Resample each index to hourly
    pivoted = df.pivot_table(
        index="timestamp_utc", columns="ticker", values="Close", aggfunc="last"
    ).resample("1h").last().reset_index()
    pivoted.columns.name = None
    pivoted.columns = [
        "timestamp_utc" if c == "timestamp_utc" else f"idx_{c}"
        for c in pivoted.columns
    ]
    # Add returns for each index
    for c in pivoted.columns:
        if c.startswith("idx_"):
            pivoted[f"{c}_return"] = pivoted[c].pct_change(1, fill_method=None)

    return pivoted


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LOAD EARNINGS EVENTS (binary flag features)
# ─────────────────────────────────────────────────────────────────────────────

def load_earnings(data_dir: Path) -> pd.DataFrame:
    """
    earnings_all.csv → adds two binary features per price bar:
      earnings_today     : 1 if this ticker has earnings within same calendar day
      earnings_tomorrow  : 1 if earnings are within the next 24 hours
    """
    f = data_dir / "earnings_all.csv"
    if not f.exists():
        logger.warning("earnings_all.csv not found — earnings features skipped.")
        return pd.DataFrame(columns=["ticker","earnings_date"])

    df = pd.read_csv(f)
    df.columns = [c.lower().strip() for c in df.columns]

    # Find date and ticker columns flexibly
    date_col   = next((c for c in df.columns if "date" in c or "time" in c), None)
    ticker_col = next((c for c in df.columns if "ticker" in c or "symbol" in c), None)

    if not date_col or not ticker_col:
        logger.warning(f"Couldn't identify columns in earnings_all.csv: {list(df.columns)}")
        return pd.DataFrame(columns=["ticker","earnings_date"])

    df = df[[ticker_col, date_col]].rename(
        columns={ticker_col: "ticker", date_col: "earnings_date"}
    )
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=True, errors="coerce")
    return df.dropna()


def add_earnings_flags(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    if earnings.empty:
        prices["earnings_today"]    = 0
        prices["earnings_tomorrow"] = 0
        return prices

    merged = prices.copy()
    merged["earnings_today"]    = 0
    merged["earnings_tomorrow"] = 0

    for ticker, group in merged.groupby("ticker"):
        e_dates = earnings[earnings["ticker"] == ticker]["earnings_date"]
        if e_dates.empty:
            continue
        e_dates_set = set(e_dates.dt.date)
        ts = group["timestamp_utc"]
        merged.loc[group.index, "earnings_today"] = (
            ts.dt.date.isin(e_dates_set).astype(int)
        )
        merged.loc[group.index, "earnings_tomorrow"] = (
            (ts + pd.Timedelta("1d")).dt.date.isin(e_dates_set).astype(int)
        )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CREATE TARGET VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, threshold: float = 0.005) -> pd.DataFrame:
    """
    Target: did price go UP by more than `threshold` in the NEXT hour?
      1 = Up   (return_next_1h >  +threshold)
      0 = Down (return_next_1h < -threshold)
      NaN = Uncertain (within threshold band) → dropped before training

    This is a forward-looking label — it MUST be shifted after computation.
    The shift happens naturally because we use .shift(-1) to look forward.
    """
    def _target_per_ticker(g):
        g = g.sort_values("timestamp_utc").copy()
        next_return = g["Close"].pct_change(1, fill_method=None).shift(-1)  # look 1 bar ahead
        g["target"] = np.where(
            next_return >  threshold, 1,
            np.where(next_return < -threshold, 0, np.nan)
        )
        return g

    return df.groupby("ticker", group_keys=False).apply(_target_per_ticker)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — FUSE EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    sentiment_features: pd.DataFrame = None,   # output of Stage 1C
    data_dir: Path = DATA_DIR,
    ticker_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns the full fused feature matrix, one row per (ticker, hour).
    Pass sentiment_features=None to build the price-only baseline.
    """

    # ── Market data ──────────────────────────────────────────────────
    prices   = load_prices(data_dir, ticker_filter=ticker_filter)
    prices   = add_technical_features(prices)

    vix      = load_vix(data_dir)
    indices  = load_indices(data_dir)
    earnings = load_earnings(data_dir)

    # ── Join VIX (timestamp only — same for all tickers) ─────────────
    if not vix.empty:
        vix["timestamp_utc"] = pd.to_datetime(vix["timestamp_utc"], utc=True)
        prices = prices.merge(vix, on="timestamp_utc", how="left")

    # ── Join indices ─────────────────────────────────────────────────
    if not indices.empty:
        indices["timestamp_utc"] = pd.to_datetime(indices["timestamp_utc"], utc=True)
        prices = prices.merge(indices, on="timestamp_utc", how="left")

    # ── Earnings flags ───────────────────────────────────────────────
    prices = add_earnings_flags(prices, earnings)

    # ── Join sentiment features (already lagged from Stage 1C) ───────
    if sentiment_features is not None:
        sentiment_features["timestamp_utc"] = pd.to_datetime(
            sentiment_features["window_start"], utc=True
        )
        prices = prices.merge(
            sentiment_features.drop(columns=["window_start","feature_source_window"], errors="ignore"),
            on=["ticker","timestamp_utc"],
            how="left",
        )

    # ── Target variable ──────────────────────────────────────────────
    prices = add_target(prices)

    # ── Drop rows without a target (last bar per ticker, uncertain zone)
    prices = prices.dropna(subset=["target"])
    prices["target"] = prices["target"].astype(int)

    # ── Final sort ───────────────────────────────────────────────────
    prices = prices.sort_values(["ticker","timestamp_utc"]).reset_index(drop=True)

    logger.info(f"\nFeature matrix ready: {prices.shape}")
    logger.info(f"Target distribution:\n{prices['target'].value_counts().to_string()}")
    logger.info(f"Columns ({len(prices.columns)}):\n{list(prices.columns)}")

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TRAIN/VAL/TEST SPLIT (time-based, never shuffle)
# ─────────────────────────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    # test_ratio is implicit: 1 - train - val = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split — per ticker to respect each ticker's own timeline.
    Returns (train, val, test).
    """
    train_frames, val_frames, test_frames = [], [], []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("timestamp_utc")
        n = len(g)
        i_train = int(n * train_ratio)
        i_val   = int(n * (train_ratio + val_ratio))

        train_frames.append(g.iloc[:i_train])
        val_frames.append(  g.iloc[i_train:i_val])
        test_frames.append( g.iloc[i_val:])

    train = pd.concat(train_frames).sort_values("timestamp_utc").reset_index(drop=True)
    val   = pd.concat(val_frames  ).sort_values("timestamp_utc").reset_index(drop=True)
    test  = pd.concat(test_frames ).sort_values("timestamp_utc").reset_index(drop=True)

    logger.info(f"Split sizes - Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build price-only baseline (no sentiment yet)
    matrix = build_feature_matrix(sentiment_features=None)
    
    # Save to output directory
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_matrix.csv"
    
    matrix.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}  ({matrix.shape[0]} rows x {matrix.shape[1]} cols)")

    train, val, test = time_split(matrix)
    print(f"Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")