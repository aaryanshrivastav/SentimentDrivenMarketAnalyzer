"""
STAGE 2A — MARKET DATA & TECHNICAL INDICATORS
==============================================
Data source : yfinance (OHLCV) + TA-Lib (indicators)
Macro data  : yfinance ^VIX, manual earnings/fed flags
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn(
        "TA-Lib not installed. Falling back to pandas-based indicator calculations. "
        "Install TA-Lib for faster, more accurate results: pip install TA-Lib"
    )

from datetime import datetime, timedelta
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RAW OHLCV + LOG RETURN + INTRADAY RANGE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and add basic price features.

    Parameters
    ----------
    ticker : e.g. "AAPL"
    start  : "YYYY-MM-DD"
    end    : "YYYY-MM-DD"  (defaults to today)

    Returns
    -------
    DataFrame with columns:
        Open, High, Low, Close, Volume,
        Log_Return, Intraday_Range
    """
    end = end or datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol or date range.")

    # Flatten MultiIndex columns produced by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    # ── Price features ─────────────────────────────────────────────────────────
    df["Log_Return"]      = np.log(df["Close"] / df["Close"].shift(1))
    df["Intraday_Range"]  = (df["High"] - df["Low"]) / df["Close"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TECHNICAL INDICATORS  (TA-Lib preferred, pandas fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _rsi_pandas(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_pandas(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all required technical indicators to the OHLCV DataFrame.

    Indicators added
    ----------------
    RSI_14          — Relative Strength Index (14-period)
    MACD            — MACD line (12/26 EMA diff)
    MACD_Signal     — Signal line (9-period EMA of MACD)
    MACD_Hist       — MACD histogram
    MA20, MA50      — Simple moving averages
    BB_Width        — Bollinger Band width  = (Upper - Lower) / Middle
    ATR_14          — Average True Range (14-period)
    Volume_ZScore   — (Volume - 20d mean) / 20d std
    Price_ZScore    — (Close  - 20d mean) / 20d std
    """
    out = df.copy()
    close  = out["Close"].astype(float)
    high   = out["High"].astype(float)
    low    = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    # ── RSI ────────────────────────────────────────────────────────────────────
    if TALIB_AVAILABLE:
        out["RSI_14"] = talib.RSI(close.values, timeperiod=14)
    else:
        out["RSI_14"] = _rsi_pandas(close, period=14)

    # ── MACD ───────────────────────────────────────────────────────────────────
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(
            close.values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        out["MACD"]       = macd
        out["MACD_Signal"] = signal
        out["MACD_Hist"]  = hist
    else:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["MACD"]        = ema12 - ema26
        out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_Hist"]   = out["MACD"] - out["MACD_Signal"]

    # ── Moving Averages ────────────────────────────────────────────────────────
    if TALIB_AVAILABLE:
        out["MA20"] = talib.SMA(close.values, timeperiod=20)
        out["MA50"] = talib.SMA(close.values, timeperiod=50)
    else:
        out["MA20"] = close.rolling(20).mean()
        out["MA50"] = close.rolling(50).mean()

    # ── Bollinger Band Width ───────────────────────────────────────────────────
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            close.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        out["BB_Width"] = (upper - lower) / np.where(middle != 0, middle, np.nan)
    else:
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std(ddof=0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        out["BB_Width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # ── ATR ────────────────────────────────────────────────────────────────────
    if TALIB_AVAILABLE:
        out["ATR_14"] = talib.ATR(high.values, low.values, close.values, timeperiod=14)
    else:
        out["ATR_14"] = _atr_pandas(high, low, close, period=14)

    # ── Volume Z-score (20-day rolling) ────────────────────────────────────────
    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std(ddof=1).replace(0, np.nan)
    out["Volume_ZScore"] = (volume - vol_mean) / vol_std

    # ── Price Z-score (20-day rolling) ─────────────────────────────────────────
    price_mean = close.rolling(20).mean()
    price_std  = close.rolling(20).std(ddof=1).replace(0, np.nan)
    out["Price_ZScore"] = (close - price_mean) / price_std

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MACRO / REGIME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_vix(start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch VIX OHLCV from yfinance (^VIX) and return a DataFrame
    with the closing price and daily change.

    Returns
    -------
    DataFrame indexed by Date with columns: VIX, VIX_Change
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    if vix_raw.empty:
        raise ValueError("Failed to fetch VIX data. Check your internet connection.")

    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    vix = vix_raw[["Close"]].rename(columns={"Close": "VIX"})
    vix["VIX_Change"] = vix["VIX"] - vix["VIX"].shift(1)
    return vix


def build_earnings_flag(dates: pd.DatetimeIndex, earnings_dates: list[str], window: int = 3) -> pd.Series:
    """
    Return a binary Series (1 / 0) indicating whether a given trading day
    falls within `window` calendar days of an earnings announcement.

    Parameters
    ----------
    dates          : DatetimeIndex from your OHLCV DataFrame
    earnings_dates : list of "YYYY-MM-DD" strings (known earnings dates)
    window         : look-ahead / look-back window in calendar days (default 3)

    Example
    -------
    >>> earnings_flag = build_earnings_flag(
    ...     df.index,
    ...     earnings_dates=["2024-01-25", "2024-04-25", "2024-07-30", "2024-10-28"]
    ... )
    """
    earnings_dt = [pd.Timestamp(d) for d in earnings_dates]
    flag = pd.Series(0, index=dates, name="earnings_flag")
    for ed in earnings_dt:
        mask = (dates >= ed - timedelta(days=window)) & (dates <= ed + timedelta(days=window))
        flag[mask] = 1
    return flag


def build_fed_event_flag(dates: pd.DatetimeIndex, fed_dates: list[str]) -> pd.Series:
    """
    Return a binary Series (1 / 0) for Fed meeting / announcement days.

    Parameters
    ----------
    dates     : DatetimeIndex from your OHLCV DataFrame
    fed_dates : list of "YYYY-MM-DD" strings (FOMC meeting dates)

    Example
    -------
    >>> fed_flag = build_fed_event_flag(
    ...     df.index,
    ...     fed_dates=["2024-01-31", "2024-03-20", "2024-05-01",
    ...                "2024-06-12", "2024-07-31", "2024-09-18",
    ...                "2024-11-07", "2024-12-18"]
    ... )
    """
    fed_ts = set(pd.Timestamp(d) for d in fed_dates)
    flag = pd.Series(
        [1 if d in fed_ts else 0 for d in dates],
        index=dates,
        name="fed_event_flag"
    )
    return flag


def compute_market_regime(vix: pd.Series) -> pd.Series:
    """
    Assign a market regime label based on VIX level.

    0 → low volatility   (VIX < 15)
    1 → normal           (15 ≤ VIX ≤ 25)
    2 → high volatility  (VIX > 25)
    """
    conditions = [vix < 15, (vix >= 15) & (vix <= 25), vix > 25]
    choices    = [0, 1, 2]
    regime = pd.Series(
        np.select(conditions, choices, default=np.nan),
        index=vix.index,
        name="market_regime"
    )
    return regime


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_dataframe(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    earnings_dates: Optional[list[str]] = None,
    fed_dates: Optional[list[str]] = None,
    earnings_window: int = 3,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    End-to-end Stage 2A pipeline.

    Returns a clean DataFrame with all price, technical, and macro features
    aligned on the same trading-day index.

    Parameters
    ----------
    ticker          : Stock ticker (e.g. "AAPL")
    start           : Start date "YYYY-MM-DD"
    end             : End date "YYYY-MM-DD" (default: today)
    earnings_dates  : List of earnings announcement dates
    fed_dates       : List of FOMC meeting dates
    earnings_window : ± calendar days around earnings (default 3)
    drop_na         : Drop rows with NaN (from indicator warm-up) if True

    Returns
    -------
    pd.DataFrame
    """
    earnings_dates = earnings_dates or []
    fed_dates      = fed_dates      or []
    end            = end or datetime.today().strftime("%Y-%m-%d")

    print(f"[Stage 2A] Fetching OHLCV for {ticker} ({start} to {end})...")
    df = fetch_ohlcv(ticker, start=start, end=end)

    print("[Stage 2A] Computing technical indicators...")
    df = compute_technical_indicators(df)

    print("[Stage 2A] Fetching VIX data...")
    vix_df = fetch_vix(start=start, end=end)

    # Align VIX on the stock's trading days (forward-fill any gaps)
    df = df.join(vix_df, how="left")
    df["VIX"]        = df["VIX"].ffill()
    df["VIX_Change"] = df["VIX_Change"].ffill()

    print("[Stage 2A] Building macro / regime features...")
    df["earnings_flag"]  = build_earnings_flag(df.index, earnings_dates, window=earnings_window)
    df["fed_event_flag"] = build_fed_event_flag(df.index, fed_dates)
    df["market_regime"]  = compute_market_regime(df["VIX"])

    if drop_na:
        before = len(df)
        df.dropna(inplace=True)
        print(f"[Stage 2A] Dropped {before - len(df)} rows during indicator warm-up. {len(df)} rows remain.")

    print(f"[Stage 2A] Done. Feature matrix shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  QUICK-LOOK DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the feature matrix."""
    print("\n" + "=" * 60)
    print("STAGE 2A — FEATURE MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape        : {df.shape}")
    print(f"Date range   : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns ({len(df.columns)}) :\n  " + "\n  ".join(df.columns.tolist()))
    print("\nDescriptive statistics:")
    print(df.describe().round(4).to_string())
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EXAMPLE USAGE  (run this file directly to see output)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Known AAPL earnings dates (quarterly) ─────────────────────────────────
    AAPL_EARNINGS = [
        "2023-02-02", "2023-05-04", "2023-08-03", "2023-11-02",
        "2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31",
        "2025-02-01", "2025-05-02", "2025-08-01", "2025-10-31",
    ]

    # ── FOMC meeting dates 2023-2024 ──────────────────────────────────────────
    FOMC_DATES = [
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-31", "2025-03-20", "2025-05-01", "2025-06-12",
        "2025-07-31", "2025-09-18", "2025-11-07", "2025-12-18",
    ]

    features = build_feature_dataframe(
        ticker          = "AAPL",
        start           = "2023-01-01",
        end             = "2025-12-31",
        earnings_dates  = AAPL_EARNINGS,
        fed_dates       = FOMC_DATES,
        earnings_window = 3,
        drop_na         = True,
    )

    print_feature_summary(features)

    # Preview last 5 rows
    print("Last 5 rows:")
    print(features.tail().to_string())

    # Optional: save to CSV
    # features.to_csv("stage2a_features_AAPL.csv")