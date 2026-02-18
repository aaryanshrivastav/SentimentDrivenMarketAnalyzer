"""
Stage 0 — Market Data Collector
Pulls historical + live OHLCV price data, VIX, India VIX, broad indices,
and earnings dates via yfinance.

Output:
    data/market/prices_{ticker}_{timestamp}.csv   — OHLCV per ticker
    data/market/prices_all_{timestamp}.csv        — combined OHLCV
    data/market/vix_{timestamp}.csv               — CBOE VIX
    data/market/india_vix_{timestamp}.csv         — NSE India VIX
    data/market/indices_{timestamp}.csv           — NIFTY50, SENSEX, S&P500, NASDAQ
    data/market/earnings_all.csv                  — upcoming earnings dates
"""

import yfinance as yf
yf.set_tz_cache_location("cache")
import pandas as pd
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    TICKERS, PRICE_INTERVAL, PRICE_PERIOD,
    VIX_TICKER, INDIA_VIX_TICKER, INDEX_TICKERS,
    MARKET_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def fetch_ohlcv(ticker: str, period: str = None, interval: str = None) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker using yfinance.
    Returns a DataFrame with columns: timestamp_utc, Open, High, Low, Close, Volume, ticker.
    """
    period   = period   or PRICE_PERIOD
    interval = interval or PRICE_INTERVAL

    logger.info(f"Fetching OHLCV: {ticker} | period={period} | interval={interval}")

    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        logger.warning(f"No OHLCV data returned for {ticker}")
        return df

    df.index = df.index.tz_convert("UTC")
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "timestamp_utc", "Date": "timestamp_utc"}, inplace=True)

    cols = [c for c in ["timestamp_utc", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    df["ticker"] = ticker
    df["timestamp_utc"] = df["timestamp_utc"].dt.strftime(TIMESTAMP_FORMAT)

    logger.info(f"{ticker}: {len(df)} rows ({df['timestamp_utc'].iloc[0]} → {df['timestamp_utc'].iloc[-1]})")
    return df


def fetch_volatility_index(ticker: str, label: str, period: str = None, interval: str = None) -> pd.DataFrame:
    """
    Download a volatility index (VIX or India VIX).
    Returns a DataFrame with columns: timestamp_utc, vix_close.
    """
    period   = period   or PRICE_PERIOD
    interval = interval or PRICE_INTERVAL

    logger.info(f"Fetching {label} ({ticker}) | period={period} | interval={interval}")
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        logger.warning(f"No data returned for {label}")
        return df

    df.index = df.index.tz_convert("UTC")
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "timestamp_utc", "Date": "timestamp_utc"}, inplace=True)
    df = df[["timestamp_utc", "Close"]].copy()
    df.rename(columns={"Close": "vix_close"}, inplace=True)
    df["index"] = label
    df["timestamp_utc"] = df["timestamp_utc"].dt.strftime(TIMESTAMP_FORMAT)

    logger.info(f"{label}: {len(df)} rows")
    return df


def fetch_indices(period: str = None, interval: str = None) -> pd.DataFrame:
    """
    Fetch broad market indices: NIFTY 50, SENSEX, S&P 500, NASDAQ.
    Returns a combined DataFrame with columns: timestamp_utc, Close, index_name.
    """
    period   = period   or PRICE_PERIOD
    interval = interval or PRICE_INTERVAL

    frames = []
    for name, ticker in INDEX_TICKERS.items():
        logger.info(f"Fetching index: {name} ({ticker})")
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning(f"No data for index {name}")
            continue
        df.index = df.index.tz_convert("UTC")
        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "timestamp_utc", "Date": "timestamp_utc"}, inplace=True)
        df = df[["timestamp_utc", "Close"]].copy()
        df["index_name"] = name
        df["ticker"] = ticker
        df["timestamp_utc"] = df["timestamp_utc"].dt.strftime(TIMESTAMP_FORMAT)
        frames.append(df)
        logger.info(f"{name}: {len(df)} rows")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Fetch upcoming earnings dates for a ticker.
    Returns a DataFrame with columns: ticker, earnings_date.
    """
    logger.info(f"Fetching earnings dates for {ticker}")
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            logger.warning(f"No earnings calendar found for {ticker}")
            return pd.DataFrame(columns=["ticker", "earnings_date"])

        earnings_dates = []
        if "Earnings Date" in cal.index:
            for val in cal.loc["Earnings Date"]:
                if pd.notna(val):
                    earnings_dates.append(str(val)[:10])

        df = pd.DataFrame({"ticker": ticker, "earnings_date": earnings_dates})
        logger.info(f"{ticker} earnings dates: {earnings_dates}")
        return df

    except Exception as e:
        logger.warning(f"Could not fetch earnings for {ticker}: {e}")
        return pd.DataFrame(columns=["ticker", "earnings_date"])


def run_market_data_collection(tickers: list = None) -> dict:
    """
    Entry point: fetch OHLCV for all tickers (US + India) + VIX + India VIX
    + broad indices + earnings dates.
    Saves individual CSVs. Returns dict of DataFrames.
    """
    if tickers is None:
        tickers = TICKERS

    os.makedirs(MARKET_DATA_DIR, exist_ok=True)
    timestamp_tag = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")

    results = {}

    # ── OHLCV per ticker (US + India) ────────────────────────────────────
    price_frames = []
    for ticker in tickers:
        df = fetch_ohlcv(ticker)
        if not df.empty:
            price_frames.append(df)
            path = os.path.join(MARKET_DATA_DIR, f"prices_{ticker.replace('.', '_')}_{timestamp_tag}.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved → {path}")
        results[f"prices_{ticker}"] = df

    if price_frames:
        combined = pd.concat(price_frames, ignore_index=True)
        combined_path = os.path.join(MARKET_DATA_DIR, f"prices_all_{timestamp_tag}.csv")
        combined.to_csv(combined_path, index=False)
        results["prices_all"] = combined
        logger.info(f"Combined prices saved → {combined_path}")

    # ── US VIX ───────────────────────────────────────────────────────────
    vix_df = fetch_volatility_index(VIX_TICKER, "US_VIX")
    if not vix_df.empty:
        vix_path = os.path.join(MARKET_DATA_DIR, f"vix_{timestamp_tag}.csv")
        vix_df.to_csv(vix_path, index=False)
        logger.info(f"US VIX saved → {vix_path}")
    results["vix"] = vix_df

    # ── India VIX ────────────────────────────────────────────────────────
    india_vix_df = fetch_volatility_index(INDIA_VIX_TICKER, "INDIA_VIX")
    if not india_vix_df.empty:
        india_vix_path = os.path.join(MARKET_DATA_DIR, f"india_vix_{timestamp_tag}.csv")
        india_vix_df.to_csv(india_vix_path, index=False)
        logger.info(f"India VIX saved → {india_vix_path}")
    results["india_vix"] = india_vix_df

    # ── Broad Indices (NIFTY50, SENSEX, S&P500, NASDAQ) ─────────────────
    indices_df = fetch_indices()
    if not indices_df.empty:
        indices_path = os.path.join(MARKET_DATA_DIR, f"indices_{timestamp_tag}.csv")
        indices_df.to_csv(indices_path, index=False)
        logger.info(f"Indices saved → {indices_path}")
    results["indices"] = indices_df

    # ── Earnings dates ────────────────────────────────────────────────────
    earnings_frames = []
    for ticker in tickers:
        df = fetch_earnings_dates(ticker)
        earnings_frames.append(df)
        if not df.empty:
            path = os.path.join(MARKET_DATA_DIR, f"earnings_{ticker.replace('.', '_')}.csv")
            df.to_csv(path, index=False)
    if earnings_frames:
        earnings_all = pd.concat(earnings_frames, ignore_index=True)
        results["earnings"] = earnings_all
        earnings_path = os.path.join(MARKET_DATA_DIR, "earnings_all.csv")
        earnings_all.to_csv(earnings_path, index=False)
        logger.info(f"Earnings saved → {earnings_path}")

    logger.info("Market data collection complete.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    data = run_market_data_collection()
    for key, df in data.items():
        if not df.empty:
            print(f"\n── {key} ({len(df)} rows) ──")
            print(df.head(3))