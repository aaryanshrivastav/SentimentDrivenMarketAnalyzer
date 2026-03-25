"""
STAGE 2B — GRANGER CAUSALITY TEST
==================================
One-time validation run BEFORE training.
Tests whether past sentiment scores statistically help predict future price
returns beyond what past price alone can explain.

Decision rule:
  p-value < 0.05 for ANY lag (1–4) → sentiment is a valid predictor → INCLUDE
  p-value ≥ 0.05 for ALL lags      → sentiment adds no signal         → EXCLUDE

Run this per ticker. Results are saved to a JSON registry so downstream
training code can query "should I use sentiment for TICKER?" automatically.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
import warnings
import os
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

warnings.filterwarnings("ignore")


def _is_constant(series: pd.Series) -> bool:
    clean = series.dropna()
    if len(clean) == 0:
        return True
    return clean.nunique() <= 1


def _inject_tiny_jitter(series: pd.Series, scale: float = 1e-8) -> pd.Series:
    """Inject deterministic epsilon trend to avoid constant-series numerical failures."""
    clean = series.copy()
    if len(clean) == 0:
        return clean
    eps = np.linspace(0.0, scale, num=len(clean))
    return clean + eps


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MAX_LAGS       = 4          # Test lags 1, 2, 3, 4
P_VALUE_THRESH = 0.05       # Significance threshold
REGISTRY_PATH  = Path("sentiment_registry.json")   # Persisted results


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STATIONARITY CHECK  (Granger requires stationary series)
# ══════════════════════════════════════════════════════════════════════════════

def check_stationarity(series: pd.Series, name: str = "series", verbose: bool = True) -> bool:
    """
    Run Augmented Dickey-Fuller test on a series.

    Returns True if stationary (p < 0.05), False otherwise.
    If non-stationary, the caller should first-difference the series.
    """
    clean = series.dropna()
    if len(clean) < 8:
        if verbose:
            print(f"  ADF [{name}]: insufficient length ({len(clean)}), treat as non-stationary")
        return False

    if _is_constant(clean):
        if verbose:
            print(f"  ADF [{name}]: constant series, treat as stationary for preprocessing")
        return True

    try:
        result = adfuller(clean, autolag="AIC")
    except ValueError as exc:
        if "constant" in str(exc).lower():
            if verbose:
                print(f"  ADF [{name}]: constant input encountered, treat as stationary")
            return True
        raise
    p_val  = result[1]
    is_stationary = p_val < P_VALUE_THRESH

    if verbose:
        status = "STATIONARY" if is_stationary else "NON-STATIONARY (will difference)"
        print(f"  ADF [{name}]: p={p_val:.4f}  -->  {status}")

    return is_stationary


def make_stationary(series: pd.Series, name: str = "series", verbose: bool = True) -> pd.Series:
    """
    Return the series as-is if stationary; otherwise return first-difference.
    At most two rounds of differencing are attempted.
    """
    for d in range(2):
        if check_stationarity(series, name=f"{name} (d={d})", verbose=verbose):
            return series
        series = series.diff().dropna()

    warnings.warn(f"[{name}] Could not achieve stationarity after 2 differences.")
    return series


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CORE GRANGER TEST  (sentiment → price)
# ══════════════════════════════════════════════════════════════════════════════

def run_granger_test(
    price_series: pd.Series,
    sentiment_series: pd.Series,
    ticker: str,
    max_lags: int = MAX_LAGS,
    p_thresh: float = P_VALUE_THRESH,
    verbose: bool = True,
) -> dict:
    """
    Test whether `sentiment_series` Granger-causes `price_series`.

    Parameters
    ----------
    price_series     : Log returns (or any price-derived stationary series)
    sentiment_series : Sentiment scores aligned to the same index
    ticker           : Used for labelling only
    max_lags         : Number of lags to test (1 … max_lags)
    p_thresh         : Significance level (default 0.05)
    verbose          : Print lag-by-lag results

    Returns
    -------
    dict with keys:
        ticker, lags_tested, lag_results, min_p_value,
        best_lag, sentiment_is_valid, recommendation
    """
    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  GRANGER CAUSALITY: sentiment to price  [{ticker}]")
        print(f"{'═' * 60}")

    # ── Align and clean ────────────────────────────────────────────────────────
    combined = pd.DataFrame({
        "price":     price_series,
        "sentiment": sentiment_series,
    }).dropna()

    if len(combined) < max_lags * 10:
        warnings.warn(
            f"[{ticker}] Only {len(combined)} aligned rows. "
            f"Need at least {max_lags * 10} for reliable results."
        )

    if verbose:
        print(f"\n  Aligned observations : {len(combined)}")
        print(f"  Date range           : {combined.index[0]} to {combined.index[-1]}\n")
        print("  ── Stationarity checks ──")

    # ── Enforce stationarity ───────────────────────────────────────────────────
    price_stat     = make_stationary(combined["price"],     name="price",     verbose=verbose)
    sentiment_stat = make_stationary(combined["sentiment"], name="sentiment", verbose=verbose)

    # Re-align after possible differencing
    test_df = pd.concat([price_stat, sentiment_stat], axis=1).dropna()
    test_df.columns = ["price", "sentiment"]

    if len(test_df) < max_lags * 5:
        warnings.warn(f"[{ticker}] Too few rows after stationarity/alignment: {len(test_df)}")
        return _build_result(ticker, max_lags, {}, p_thresh, failed=True)

    # Optional rescue for constant series after filling/alignment.
    allow_jitter = os.getenv("GRANGER_ALLOW_CONSTANT_JITTER", "1") == "1"
    if _is_constant(test_df["price"]) or _is_constant(test_df["sentiment"]):
        if allow_jitter:
            warnings.warn(
                f"[{ticker}] Constant series detected post-preprocessing; injecting tiny jitter for numerical stability."
            )
            if _is_constant(test_df["price"]):
                test_df["price"] = _inject_tiny_jitter(test_df["price"])
            if _is_constant(test_df["sentiment"]):
                test_df["sentiment"] = _inject_tiny_jitter(test_df["sentiment"])
        else:
            warnings.warn(
                f"[{ticker}] Constant series detected and jitter disabled; excluding ticker from Granger."
            )
            return _build_result(ticker, max_lags, {}, p_thresh, failed=True)

    if verbose:
        print(f"\n  ── Granger test (lags 1–{max_lags}) ──")
        print(f"  {'Lag':<6} {'F-stat':<12} {'p-value':<12} {'Significant?'}")
        print(f"  {'-'*46}")

    # ── Run tests ──────────────────────────────────────────────────────────────
    # grangercausalitytests expects [dependent, cause] column order
    gc_input = test_df[["price", "sentiment"]].values

    try:
        gc_results = grangercausalitytests(gc_input, maxlag=max_lags, verbose=False)
    except Exception as e:
        warnings.warn(f"[{ticker}] Granger test failed: {e}")
        return _build_result(ticker, max_lags, {}, p_thresh, failed=True)

    # ── Parse results ──────────────────────────────────────────────────────────
    lag_results: dict[int, dict] = {}

    for lag in range(1, max_lags + 1):
        # statsmodels returns multiple test statistics; we use the F-test
        f_stat = gc_results[lag][0]["ssr_ftest"][0]
        p_val  = gc_results[lag][0]["ssr_ftest"][1]
        sig    = bool(p_val < p_thresh)

        lag_results[lag] = {"f_stat": round(float(f_stat), 4), "p_value": round(float(p_val), 4), "significant": bool(sig)}

        if verbose:
            marker = "SIGNIFICANT" if sig else "not significant"
            print(f"  Lag {lag:<3}  F={f_stat:<10.4f}  p={p_val:<10.4f}  {marker}")

    return _build_result(ticker, max_lags, lag_results, p_thresh)


def _build_result(
    ticker: str,
    max_lags: int,
    lag_results: dict,
    p_thresh: float,
    failed: bool = False,
) -> dict:
    """Compile the final result dict and print a clear recommendation."""

    if failed or not lag_results:
        return {
            "ticker": ticker,
            "lags_tested": max_lags,
            "lag_results": {},
            "min_p_value": None,
            "best_lag": None,
            "sentiment_is_valid": False,
            "recommendation": "EXCLUDE — test failed or insufficient data",
        }

    p_values = {lag: info["p_value"] for lag, info in lag_results.items()}
    min_p    = min(p_values.values())
    best_lag = min(p_values, key=p_values.get)
    is_valid = min_p < p_thresh

    if is_valid:
        recommendation = (
            f"INCLUDE sentiment — significant at lag {best_lag} "
            f"(p={min_p:.4f} < {p_thresh})"
        )
    else:
        recommendation = (
            f"EXCLUDE sentiment — no lag reached p < {p_thresh} "
            f"(min p={min_p:.4f}). Use technical features only."
        )

    result = {
        "ticker": ticker,
        "lags_tested": max_lags,
        "lag_results": lag_results,
        "min_p_value": round(min_p, 4),
        "best_lag": best_lag,
        "sentiment_is_valid": is_valid,
        "recommendation": recommendation,
    }

    print(f"\n  ┌─ VERDICT {'─' * 48}")
    print(f"  │  {recommendation}")
    print(f"  └{'─' * 58}\n")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BATCH RUNNER  (multiple tickers at once)
# ══════════════════════════════════════════════════════════════════════════════

def run_granger_batch(
    ticker_data: dict[str, tuple[pd.Series, pd.Series]],
    max_lags: int = MAX_LAGS,
    p_thresh: float = P_VALUE_THRESH,
    save_registry: bool = True,
    registry_path: Path = REGISTRY_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Granger causality tests for multiple tickers and return a summary.

    Parameters
    ----------
    ticker_data : dict mapping ticker → (price_series, sentiment_series)
                  Both series must share the same DatetimeIndex.
    save_registry : Persist results to JSON for downstream model code.

    Returns
    -------
    pd.DataFrame — one row per ticker with all test results.

    Example
    -------
    >>> results_df = run_granger_batch({
    ...     "GME":  (gme_returns,  gme_sentiment),
    ...     "BRK-B": (brk_returns, brk_sentiment),
    ... })
    """
    all_results = []

    for ticker, (price, sentiment) in ticker_data.items():
        try:
            result = run_granger_test(
                price_series     = price,
                sentiment_series = sentiment,
                ticker           = ticker,
                max_lags         = max_lags,
                p_thresh         = p_thresh,
                verbose          = verbose,
            )
        except Exception as e:
            warnings.warn(f"[{ticker}] Granger test crashed and will be excluded: {e}")
            result = _build_result(ticker, max_lags, {}, p_thresh, failed=True)
        all_results.append(result)

    summary_df = _build_summary_table(all_results)

    if verbose:
        _print_batch_summary(summary_df)

    if save_registry:
        _save_registry(all_results, registry_path)

    return summary_df


def _build_summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        lag_p = {f"p_lag{k}": v["p_value"] for k, v in r.get("lag_results", {}).items()}
        rows.append({
            "ticker":              r["ticker"],
            "sentiment_is_valid":  r["sentiment_is_valid"],
            "min_p_value":         r["min_p_value"],
            "best_lag":            r["best_lag"],
            **lag_p,
            "recommendation":      r["recommendation"],
        })
    return pd.DataFrame(rows).set_index("ticker")


def _print_batch_summary(df: pd.DataFrame) -> None:
    include = df[df["sentiment_is_valid"]].index.tolist()
    exclude = df[~df["sentiment_is_valid"]].index.tolist()

    print("\n" + "═" * 60)
    print("  BATCH GRANGER SUMMARY")
    print("═" * 60)
    print(f"  Tickers tested   : {len(df)}")
    print(f"  Sentiment VALID  : {len(include)}  -  {include}")
    print(f"  Sentiment INVALID: {len(exclude)}  -  {exclude}")
    print("\n  Full results:")
    p_cols = [c for c in df.columns if c.startswith("p_lag")]
    display_cols = ["sentiment_is_valid", "min_p_value", "best_lag"] + p_cols
    print(df[display_cols].to_string())
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SENTIMENT REGISTRY  (persist & query results)
# ══════════════════════════════════════════════════════════════════════════════

def _save_registry(results: list[dict], path: Path = REGISTRY_PATH) -> None:
    """Persist Granger test results to JSON for downstream model consumption."""
    registry = {}
    if path.exists():
        with open(path, "r") as f:
            registry = json.load(f)

    for r in results:
        registry[r["ticker"]] = {
            "sentiment_is_valid": r["sentiment_is_valid"],
            "min_p_value":        r["min_p_value"],
            "best_lag":           r["best_lag"],
            "recommendation":     r["recommendation"],
            "lag_results":        r["lag_results"],
        }

    with open(path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"  [Registry] Saved results for {len(results)} ticker(s) to {path}")


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """Load the persisted Granger results registry."""
    if not path.exists():
        raise FileNotFoundError(
            f"Registry not found at '{path}'. Run run_granger_batch() first."
        )
    with open(path, "r") as f:
        return json.load(f)


def sentiment_is_valid_for(ticker: str, path: Path = REGISTRY_PATH) -> bool:
    """
    Quick boolean lookup: should sentiment features be used for this ticker?

    Usage (in training / inference code)
    -------------------------------------
    >>> if sentiment_is_valid_for("GME"):
    ...     features = pd.concat([tech_features, sentiment_features], axis=1)
    ... else:
    ...     features = tech_features
    """
    registry = load_registry(path)
    if ticker not in registry:
        warnings.warn(
            f"[{ticker}] not found in Granger registry. "
            f"Run the test first. Defaulting to EXCLUDE."
        )
        return False
    return registry[ticker]["sentiment_is_valid"]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SYNTHETIC DEMO  (runs without real data)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_data(
    n: int = 500,
    sentiment_leads_price: bool = True,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate synthetic price returns and sentiment scores.

    When `sentiment_leads_price=True`, the sentiment at t−1 is embedded
    into price at t, so Granger should detect the relationship.
    When False, sentiment is pure noise → Granger should NOT detect it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="h")

    sentiment = pd.Series(rng.normal(0, 1, n), index=dates, name="sentiment")

    if sentiment_leads_price:
        # Price is partially driven by yesterday's sentiment (lag-1 relationship)
        noise  = rng.normal(0, 0.5, n)
        price  = 0.4 * sentiment.shift(1).fillna(0) + noise
    else:
        # Price is pure random walk — sentiment has no predictive power
        price  = pd.Series(rng.normal(0, 1, n), index=dates)

    price = pd.Series(price.values, index=dates, name="price")
    return price, sentiment


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN — example usage
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n  STAGE 2B — GRANGER CAUSALITY  (synthetic demo)")

    # ── Synthetic ticker where sentiment DOES lead price (e.g. GME-type) ──────
    gme_price, gme_sentiment = _generate_synthetic_data(
        n=600, sentiment_leads_price=True, seed=1
    )

    # ── Synthetic ticker where sentiment does NOT lead price (e.g. BRK-B) ─────
    brk_price, brk_sentiment = _generate_synthetic_data(
        n=600, sentiment_leads_price=False, seed=2
    )

    # ── Batch test ─────────────────────────────────────────────────────────────
    results = run_granger_batch(
        ticker_data={
            "GME":   (gme_price,  gme_sentiment),
            "BRK-B": (brk_price,  brk_sentiment),
        },
        max_lags       = MAX_LAGS,
        p_thresh       = P_VALUE_THRESH,
        save_registry  = True,
        verbose        = True,
    )

    # ── How to use the registry in your training pipeline ─────────────────────
    print("── Registry lookup demo ──────────────────────────────────")
    for ticker in ["GME", "BRK-B"]:
        valid = sentiment_is_valid_for(ticker)
        feature_set = "technical + sentiment" if valid else "technical only"
        print(f"  {ticker:<8} - use {feature_set}")

    # ══════════════════════════════════════════════════════════════════════════
    # REAL-DATA USAGE TEMPLATE
    # ══════════════════════════════════════════════════════════════════════════
    # Replace the synthetic series with your actual data:
    #
    #   from stage2a_market_features import build_feature_dataframe
    #
    #   df = build_feature_dataframe("GME", start="2022-01-01", end="2024-12-31")
    #   price_returns = df["Log_Return"]
    #
    #   # sentiment_series must be on the same DatetimeIndex as price_returns
    #   # (hourly, daily — whatever resolution your NLP pipeline produces)
    #   sentiment = load_your_sentiment_scores("GME")   # pd.Series
    #
    #   result = run_granger_test(
    #       price_series     = price_returns,
    #       sentiment_series = sentiment,
    #       ticker           = "GME",
    #   )
    # ══════════════════════════════════════════════════════════════════════════