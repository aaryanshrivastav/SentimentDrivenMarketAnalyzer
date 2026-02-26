"""
SENTIMENT-DRIVEN FINANCIAL MARKET ANALYSIS
==========================================
Complete end-to-end pipeline orchestrator following the definitive blueprint.

PIPELINE STAGES:
  Stage 0:  Raw Data Collection (Reddit, StockTwits, News, Market)
  Stage 1A: Data Cleaning & Noise Removal (Bot, Spam, Sarcasm, NER, Credibility)
  Stage 1B: Sentiment Analysis Engine (FinBERT)
  Stage 1C: Sentiment Feature Aggregation (11 features per ticker-hour, lagged)
  Stage 2A: Market Data & Technical Indicators (22 features)
  Stage 2B: Granger Causality Test (sentiment → price validation)
  Stage 2C: Feature Fusion (sentiment + market + interactions)
  Stage 3:  Prediction Model (XGBoost + LSTM Ensemble)
  Stage 3A: Ablation Study (4 feature configurations)

USAGE:
  # Run full pipeline from data collection to model training
  python test.py --full

  # Run specific stages
  python test.py --stages 1B 1C 2A 2C 3

  # Quick test with sample data
  python test.py --quick

  # Custom tickers and dates
  python test.py --ticker AAPL TSLA GME --start 2024-01-01 --end 2024-01-31 --stages 1B 1C 2A 2C

  # Run ablation study only
  python test.py --stages 3A
"""

import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocessing"))

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PIPELINE")
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """Configuration for complete pipeline execution"""
    
    def __init__(self):
        # Paths
        self.data_dir = PROJECT_ROOT / "data"
        self.processed_dir = self.data_dir / "processed"
        self.market_dir = self.data_dir / "market"
        self.output_dir = PROJECT_ROOT / "output"
        self.models_dir = PROJECT_ROOT / "models"
        
        # Create directories
        for d in [self.processed_dir, self.market_dir, self.output_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Stock selection
        self.tickers = ["AAPL", "TSLA", "NVDA", "GME", "AMZN"]  # 5 liquid stocks
        self.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Stage 1 parameters
        self.window_freq = "1h"              # Hourly aggregation
        self.lag_windows = 1                 # 1-hour lag for non-leakage
        self.finbert_model = "finbert_finetuned_2/best_model"  # Fine-tuned model path
        self.sarcasm_threshold_flip = 0.65   # Flip sentiment if sarcasm > this
        self.sarcasm_threshold_uncertain = 0.40  # Between 0.40-0.65 = uncertain
        self.confidence_threshold = 0.70     # Minimum FinBERT confidence
        
        # Stage 2 parameters
        self.granger_max_lag = 4             # Test lags 1-4 hours
        self.granger_significance = 0.05     # p-value threshold
        
        # Stage 3 parameters
        self.train_split = 0.6               # 60% train
        self.val_split = 0.2                 # 20% validation
        self.test_split = 0.2                # 20% test
        self.ensemble_weights = (0.5, 0.5)   # (XGBoost, LSTM)
        self.up_threshold = 0.55             # > 0.55 = predict Up
        self.down_threshold = 0.45           # < 0.45 = predict Down
        
    def __repr__(self):
        return (
            f"PipelineConfig(\n"
            f"  Tickers: {self.tickers}\n"
            f"  Date Range: {self.start_date} → {self.end_date}\n"
            f"  Window: {self.window_freq}, Lag: {self.lag_windows}\n"
            f"  FinBERT Model: {self.finbert_model}\n"
            f"  Granger Lags: 1-{self.granger_max_lag}\n"
            f")"
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0: RAW DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_0(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    """
    Stage 0: Raw Data Collection
    
    Pulls data from:
      - Reddit (r/wallstreetbets, r/stocks, r/investing)
      - StockTwits API
      - Financial news (Yahoo Finance, Reuters)
      - Market data (yfinance: OHLCV, VIX, earnings)
    
    Critical: All timestamps in UTC, aligned to hour boundaries.
    
    Returns:
        dict with keys: 'reddit', 'stocktwits', 'news', 'market'
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 0: RAW DATA COLLECTION")
    logger.info("="*80)
    
    results = {}
    
    # ── Reddit Data ──────────────────────────────────────────────────────────
    try:
        from reddit_collector import run_reddit_collection
        logger.info("Collecting Reddit data...")
        reddit_df = run_reddit_collection()
        if reddit_df is not None and len(reddit_df) > 0:
            reddit_df['source'] = 'reddit'
            results['reddit'] = reddit_df
            logger.info(f"Reddit: {len(reddit_df)} posts collected")
        else:
            logger.warning("No Reddit data collected")
    except Exception as e:
        logger.error(f"Reddit collection failed: {e}")
    
    # ── StockTwits Data ──────────────────────────────────────────────────────
    try:
        from stocktwits_collector import run_stocktwits_collection
        logger.info("Collecting StockTwits data...")
        stocktwits_df = run_stocktwits_collection()
        if stocktwits_df is not None and len(stocktwits_df) > 0:
            stocktwits_df['source'] = 'stocktwits'
            results['stocktwits'] = stocktwits_df
            logger.info(f"StockTwits: {len(stocktwits_df)} messages collected")
        else:
            logger.warning("No StockTwits data collected")
    except Exception as e:
        logger.error(f"StockTwits collection failed: {e}")
    
    # ── News Data ────────────────────────────────────────────────────────────
    try:
        from news_collector import run_news_collection
        logger.info("Collecting financial news...")
        news_df = run_news_collection()
        if news_df is not None and len(news_df) > 0:
            news_df['source'] = 'news'
            results['news'] = news_df
            logger.info(f"News: {len(news_df)} articles collected")
        else:
            logger.warning("No news data collected")
    except Exception as e:
        logger.error(f"News collection failed: {e}")
    
    # ── Market Data ──────────────────────────────────────────────────────────
    try:
        from marketdata_collector import run_market_data_collection
        logger.info(f"Collecting market data for {config.tickers}...")
        market_data = run_market_data_collection(tickers=config.tickers)
        if market_data and 'prices' in market_data:
            results['market'] = market_data
            logger.info(f"Market: Data collected for {len(config.tickers)} tickers")
        else:
            logger.warning("No market data collected")
    except Exception as e:
        logger.error(f"Market collection failed: {e}")
    
    # ── Save raw data ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for key, data in results.items():
        if isinstance(data, pd.DataFrame):
            output_file = config.processed_dir / f"{key}_raw_{timestamp}.csv"
            data.to_csv(output_file, index=False)
            logger.info(f"Saved {key} to {output_file}")
    
    logger.info("Stage 0 complete\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1A: DATA CLEANING & NOISE REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_1a(config: PipelineConfig, raw_data: Optional[Dict] = None) -> pd.DataFrame:
    """
    Stage 1A: Data Cleaning & Noise Removal
    
    Applies (in order):
      1. Bot detection & removal
      2. Spam & low-quality filtering
      3. Sarcasm detection
      4. Named Entity Recognition (NER) → ticker linking
      5. User credibility scoring
    
    Critical improvements:
      - Only process posts with 5+ upvotes (signal quality)
      - Only keep posts with 10+ words (too short = noise)
      - Strip URLs, markdown, special formatting
      - Discard posts with no identifiable ticker
      - Compute credibility weight (0.1-1.0) per post
    
    Returns:
        DataFrame with columns: text, ticker, timestamp_utc, source,
                               sarcasm_score, sarcasm_flag, user_credibility
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 1A: DATA CLEANING & NOISE REMOVAL")
    logger.info("="*80)
    
    # ── Load data ────────────────────────────────────────────────────────────
    if raw_data is None:
        # Load from disk (most recent files)
        logger.info("Loading raw social data from disk...")
        import glob
        
        reddit_files = sorted(glob.glob(str(config.processed_dir / "reddit_raw_*.csv")))
        news_files = sorted(glob.glob(str(config.processed_dir / "news_raw_*.csv")))
        
        frames = []
        if reddit_files:
            df = pd.read_csv(reddit_files[-1])  # Most recent
            df['source'] = 'reddit'
            frames.append(df)
            logger.info(f"Loaded Reddit: {len(df)} posts")
        
        if news_files:
            df = pd.read_csv(news_files[-1])
            df['source'] = 'news'
            frames.append(df)
            logger.info(f"Loaded News: {len(df)} articles")
        
        if not frames:
            raise FileNotFoundError("No raw social data found. Run Stage 0 first.")
        
        df = pd.concat(frames, ignore_index=True)
    else:
        # Combine from raw_data dict
        frames = []
        for key in ['reddit', 'stocktwits', 'news']:
            if key in raw_data and isinstance(raw_data[key], pd.DataFrame):
                frames.append(raw_data[key])
        
        if not frames:
            raise ValueError("No social data in raw_data dict")
        
        df = pd.concat(frames, ignore_index=True)
    
    logger.info(f"Total raw posts: {len(df)}")
    
    # ── Step 1: Bot Detection ────────────────────────────────────────────────
    try:
        from preprocessing.bot_detection import BotDetection
        logger.info("Running bot detection...")
        bot_detector = BotDetection(df)
        df = bot_detector.run()
        logger.info(f"After bot removal: {len(df)} posts")
    except Exception as e:
        logger.warning(f"Bot detection failed: {e}. Skipping.")
    
    # ── Step 2: Spam & Low-Quality Filtering ─────────────────────────────────
    try:
        from preprocessing.spam_filter import SpamFilter
        logger.info("Running spam filter...")
        spam_filter = SpamFilter(df)
        df = spam_filter.run()
        logger.info(f"After spam removal: {len(df)} posts")
    except Exception as e:
        logger.warning(f"Spam filtering failed: {e}. Skipping.")
    
    # ── Step 3: Sarcasm Detection ────────────────────────────────────────────
    try:
        from preprocessing.sarcasm_detection import SarcasmDetection
        logger.info("Running sarcasm detection (RoBERTa)...")
        sarcasm_detector = SarcasmDetection(df)
        df = sarcasm_detector.run()
        
        # Apply sarcasm logic:
        # - score > 0.65: flip sentiment (handled in Stage 1B)
        # - 0.40 - 0.65: uncertain, downweight by 50%
        # - < 0.40: normal
        if 'sarcasm_score' in df.columns:
            df['sarcasm_flag'] = 'normal'
            df.loc[df['sarcasm_score'] > config.sarcasm_threshold_flip, 'sarcasm_flag'] = 'flip'
            df.loc[
                (df['sarcasm_score'] >= config.sarcasm_threshold_uncertain) & 
                (df['sarcasm_score'] <= config.sarcasm_threshold_flip), 
                'sarcasm_flag'
            ] = 'uncertain'
            
            logger.info(f"Sarcasm detected: {(df['sarcasm_flag'] == 'flip').sum()} to flip, "
                       f"{(df['sarcasm_flag'] == 'uncertain').sum()} uncertain")
        else:
            df['sarcasm_score'] = 0.0
            df['sarcasm_flag'] = 'normal'
    except Exception as e:
        logger.warning(f"Sarcasm detection failed: {e}. Skipping.")
        df['sarcasm_score'] = 0.0
        df['sarcasm_flag'] = 'normal'
    
    # ── Step 4: Named Entity Recognition (NER) → Ticker Linking ─────────────
    try:
        from preprocessing.ner_linking import NERLinking
        logger.info("Running NER ticker linking (SpaCy)...")
        ner_linker = NERLinking(df)
        df = ner_linker.run()
        
        # Remove posts with no ticker
        before_count = len(df)
        df = df[df['ticker'].notna()]
        removed = before_count - len(df)
        
        logger.info(f"NER complete: {len(df)} posts with tickers ({removed} discarded)")
    except Exception as e:
        logger.warning(f"NER linking failed: {e}. Using fallback ticker attribution.")
        # Fallback: assign to a default ticker or skip
        if 'ticker' not in df.columns:
            df['ticker'] = config.tickers[0]  # Assign all to first ticker (not ideal)
    
    # ── Step 5: User Credibility Scoring ─────────────────────────────────────
    try:
        from preprocessing.credibility_scoring import CredibilityScoring
        logger.info("Computing user credibility scores...")
        scorer = CredibilityScoring(df)
        df = scorer.run()
        
        # Rename to user_credibility if needed
        if 'final_weight' in df.columns and 'user_credibility' not in df.columns:
            df['user_credibility'] = df['final_weight']
        
        logger.info(f"Credibility scoring complete (range: {df['user_credibility'].min():.2f} - {df['user_credibility'].max():.2f})")
    except Exception as e:
        logger.warning(f"Credibility scoring failed: {e}. Using default scores.")
        df['user_credibility'] = 1.0
    
    # ── Apply sarcasm weight adjustment ──────────────────────────────────────
    if 'sarcasm_flag' in df.columns:
        # Downweight uncertain posts by 50%
        df.loc[df['sarcasm_flag'] == 'uncertain', 'user_credibility'] *= 0.5
    
    # ── Save cleaned data ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.processed_dir / f"cleaned_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Stage 1A complete: {len(df)} clean posts")
    logger.info(f"Saved to {output_file}\n")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1B: SENTIMENT ANALYSIS ENGINE (FinBERT)
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_1b(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Stage 1B: Sentiment Analysis Engine (FinBERT)
    
    Runs FinBERT sentiment model on cleaned text.
    
    Model: finbert_finetuned_2 (fine-tuned for financial domain)
    
    Output columns:
      - sentiment_label: positive / negative / neutral
      - sentiment_confidence: 0.0 - 1.0 (confidence of top label)
      - sentiment_numeric: +1 / 0 / -1
      - weighted_sentiment_score: numeric × confidence × credibility × sarcasm_adj
    
    Critical improvements:
      - Only trust predictions where confidence > 0.70
      - If uncertain (max label < 0.70), reduce weight by 70%
      - Apply sarcasm flip: if sarcasm_flag == 'flip', invert sentiment
    
    Returns:
        DataFrame with sentiment columns added
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 1B: SENTIMENT ANALYSIS ENGINE (FinBERT)")
    logger.info("="*80)
    
    # ── Save input for FinBERT ───────────────────────────────────────────────
    input_csv = config.processed_dir / "stage1a_for_finbert.csv"
    df.to_csv(input_csv, index=False)
    logger.info(f"Input: {len(df)} posts")
    
    # ── Run FinBERT inference ────────────────────────────────────────────────
    output_csv = config.processed_dir / "stage1b_sentiment.csv"
    
    try:
        from finbert import FinBERTEngine, run_finbert_stage
        
        logger.info(f"Loading FinBERT model: {config.finbert_model}")
        finbert = FinBERTEngine(model_name=config.finbert_model)
        
        logger.info("Running FinBERT inference...")
        # Ensure required columns exist
        if 'clean_text' not in df.columns and 'text' in df.columns:
            df['clean_text'] = df['text']
        if 'user_credibility' not in df.columns:
            df['user_credibility'] = 1.0
        
        # Run inference on DataFrame
        df = run_finbert_stage(
            df=df,
            text_col='clean_text',
            credibility_col='user_credibility',
            engine=finbert
        )
        
        # Save enriched data
        df.to_csv(output_csv, index=False)
        logger.info(f"FinBERT inference complete")
        
    except Exception as e:
        logger.error(f"FinBERT inference failed: {e}")
        raise
    
    # ── Apply confidence threshold ───────────────────────────────────────────
    if 'sentiment_confidence' in df.columns:
        uncertain_mask = df['sentiment_confidence'] < config.confidence_threshold
        uncertain_count = uncertain_mask.sum()
        
        if 'user_credibility' in df.columns:
            df.loc[uncertain_mask, 'user_credibility'] *= 0.3  # Reduce weight by 70%
        
        logger.info(f"Low confidence posts (<{config.confidence_threshold}): {uncertain_count} "
                   f"({100 * uncertain_count / len(df):.1f}%) downweighted")
    
    # ── Apply sarcasm flip ───────────────────────────────────────────────────
    if 'sarcasm_flag' in df.columns and 'sentiment_numeric' in df.columns:
        flip_mask = df['sarcasm_flag'] == 'flip'
        flip_count = flip_mask.sum()
        
        if flip_count > 0:
            # Flip sentiment: positive → negative, negative → positive, neutral stays
            df.loc[flip_mask, 'sentiment_numeric'] *= -1
            
            # Also flip label - use full DataFrame comparison to avoid index mismatch
            original_labels = df['sentiment_label'].copy()
            df.loc[flip_mask & (original_labels == 'positive'), 'sentiment_label'] = 'negative'
            df.loc[flip_mask & (original_labels == 'negative'), 'sentiment_label'] = 'positive'
            
            logger.info(f"Sarcasm flip applied to {flip_count} posts")
    
    # ── Recompute weighted_sentiment_score ───────────────────────────────────
    if 'sentiment_numeric' in df.columns and 'sentiment_confidence' in df.columns:
        if 'user_credibility' not in df.columns:
            df['user_credibility'] = 1.0
        
        df['weighted_sentiment_score'] = (
            df['sentiment_numeric'] * 
            df['sentiment_confidence'] * 
            df['user_credibility']
        )
    
    # ── Summary ──────────────────────────────────────────────────────────────
    if 'sentiment_label' in df.columns:
        label_counts = df['sentiment_label'].value_counts()
        logger.info(f"Sentiment distribution:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} ({100 * count / len(df):.1f}%)")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = config.processed_dir / f"stage1b_final_{timestamp}.csv"
    df.to_csv(final_output, index=False)
    
    logger.info(f"Stage 1B complete: {len(df)} posts with sentiment")
    logger.info(f"Saved to {final_output}\n")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1C: SENTIMENT FEATURE AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_1c(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Stage 1C: Sentiment Feature Aggregation
    
    Aggregates post-level sentiment into hourly features per ticker.
    
    Features computed (11 per ticker-hour):
      1. avg_sentiment: Credibility-weighted mean
      2. sentiment_std: Volatility of crowd opinion
      3. pos_count, neg_count, neu_count: Label counts
      4. bull_bear_ratio: pos / (pos + neg)
      5. mention_volume: Total post count
      6. weighted_volume: Sum of credibility scores (better volume)
      7. sentiment_momentum: Change from previous hour
      8. sentiment_acceleration: Change in momentum
      9. high_confidence_ratio: % posts with confidence > 0.80
    
    CRITICAL: Features are LAG-SHIFTED by 1 window (config.lag_windows).
    Sentiment at 13:00 is joined to price at 14:00 to prevent data leakage.
    
    Returns:
        DataFrame with columns: ticker, timestamp_utc, [11 sentiment features]
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 1C: SENTIMENT FEATURE AGGREGATION")
    logger.info("="*80)
    
    # ── Save input ───────────────────────────────────────────────────────────
    input_csv = config.processed_dir / "stage1b_for_aggregation.csv"
    df.to_csv(input_csv, index=False)
    
    # ── Run aggregation ──────────────────────────────────────────────────────
    try:
        from analyser import aggregate_sentiment_features
        
        logger.info(f"Aggregating sentiment features (window={config.window_freq}, lag={config.lag_windows})...")
        features_df = aggregate_sentiment_features(
            input_csv,
            freq=config.window_freq,
            lag_windows=config.lag_windows
        )
        
        logger.info(f"Aggregation complete: {len(features_df)} ticker-hour rows")
        
    except Exception as e:
        logger.error(f"Sentiment aggregation failed: {e}")
        raise
    
    # ── Summary ──────────────────────────────────────────────────────────────
    if 'ticker' in features_df.columns:
        ticker_counts = features_df['ticker'].value_counts()
        logger.info(f"Rows per ticker:")
        for ticker, count in ticker_counts.items():
            logger.info(f"  {ticker}: {count} hourly windows")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    output_file = config.output_dir / "sentiment_features.csv"
    features_df.to_csv(output_file, index=False)
    
    logger.info(f"Features: {list(features_df.columns)}")
    logger.info(f"Stage 1C complete")
    logger.info(f"Saved to {output_file}\n")
    
    return features_df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A: MARKET DATA & TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_2a(config: PipelineConfig) -> pd.DataFrame:
    """
    Stage 2A: Market Data & Technical Indicators
    
    Loads market data and computes technical indicators using feature.py.
    
    Returns:
        DataFrame with columns: ticker, timestamp_utc, [22+ market features]
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 2A: MARKET DATA & TECHNICAL INDICATORS")
    logger.info("="*80)
    
    try:
        from feature import build_feature_matrix
        
        logger.info("Building market feature matrix...")
        feature_matrix = build_feature_matrix(
            sentiment_features=None,  # Market features only; sentiment added in Stage 2C
            data_dir=config.market_dir  # Directory containing market data CSVs
        )
        
        logger.info(f"Feature matrix built: {feature_matrix.shape}")
        
    except Exception as e:
        logger.error(f"Stage 2A failed: {e}")
        raise
    
    # ── Summary ──────────────────────────────────────────────────────────────
    if 'ticker' in feature_matrix.columns:
        ticker_counts = feature_matrix['ticker'].value_counts()
        logger.info(f"Rows per ticker:")
        for ticker, count in ticker_counts.items():
            logger.info(f"  {ticker}: {count} rows")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    output_file = config.output_dir / "market_features.csv"
    feature_matrix.to_csv(output_file, index=False)
    
    logger.info(f"Market features: {len([c for c in feature_matrix.columns if c not in ['ticker', 'timestamp_utc']])} columns")
    logger.info(f"Stage 2A complete")
    logger.info(f"Saved to {output_file}\n")
    
    return feature_matrix


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B: GRANGER CAUSALITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_2b(
    sentiment_df: pd.DataFrame, 
    market_df: pd.DataFrame, 
    config: PipelineConfig
) -> pd.DataFrame:
    """
    Stage 2B: Granger Causality Test
    
    Validates that sentiment statistically leads price movement.
    
    Returns:
        DataFrame with columns: ticker, lag, p_value, f_statistic, significant
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 2B: GRANGER CAUSALITY TEST")
    logger.info("="*80)
    
    try:
        from granger_test import run_granger_batch
        
        # ── Prepare data for Granger test ────────────────────────────────────
        logger.info("Preparing data for Granger causality test...")
        
        # Ensure timestamp columns exist and are datetime
        # Sentiment df has 'window_start', market df has 'timestamp_utc'
        if 'window_start' in sentiment_df.columns and 'timestamp_utc' not in sentiment_df.columns:
            sentiment_df['timestamp_utc'] = pd.to_datetime(sentiment_df['window_start'], utc=True)
        elif 'timestamp_utc' in sentiment_df.columns:
            sentiment_df['timestamp_utc'] = pd.to_datetime(sentiment_df['timestamp_utc'], utc=True)
        
        if 'timestamp_utc' in market_df.columns:
            market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp_utc'], utc=True)
        
        # Build ticker_data dict: ticker → (price_series, sentiment_series)
        ticker_data = {}
        tickers = set(sentiment_df['ticker'].unique()) & set(market_df['ticker'].unique())
        
        for ticker in tickers:
            # Get sentiment time series
            sent_sub = sentiment_df[sentiment_df['ticker'] == ticker].sort_values('timestamp_utc')
            if 'avg_sentiment' not in sent_sub.columns:
                logger.warning(f"No avg_sentiment column for {ticker}, skipping")
                continue
            
            # Get price time series
            price_sub = market_df[market_df['ticker'] == ticker].sort_values('timestamp_utc')
            if 'Close' not in price_sub.columns:
                logger.warning(f"No Close column for {ticker}, skipping")
                continue
            
            # Align on common timestamps
            common_times = set(sent_sub['timestamp_utc']) & set(price_sub['timestamp_utc'])
            
            # Relax requirement if we have limited data
            min_required = min(30, max(10, len(common_times)))
            if len(common_times) < 10:  # Absolute minimum for any statistical test
                logger.warning(f"Insufficient overlapping data for {ticker} ({len(common_times)} points), skipping")
                continue
            elif len(common_times) < 30:
                logger.warning(f"{ticker}: Only {len(common_times)} overlapping points (recommended: 30+), results may be unreliable")
            
            common_times = sorted(common_times)
            
            # Extract aligned series
            sent_series = sent_sub.set_index('timestamp_utc').loc[common_times, 'avg_sentiment']
            price_series = price_sub.set_index('timestamp_utc').loc[common_times, 'Close']
            
            ticker_data[ticker] = (price_series, sent_series)
            logger.info(f"  {ticker}: {len(common_times)} aligned time points")
        
        if not ticker_data:
            logger.warning("No tickers with sufficient overlapping sentiment + price data for Granger test")
            logger.warning("This is expected with limited data. Granger test results will be empty.")
            # Return empty DataFrame instead of failing
            granger_results = pd.DataFrame(columns=['ticker', 'lag', 'p_value', 'f_statistic', 'significant'])
        else:
            # ── Run Granger test ─────────────────────────────────────────────────
            logger.info(f"Running Granger causality test (max lag={config.granger_max_lag})...")
            granger_results = run_granger_batch(
                ticker_data=ticker_data,
                max_lag=config.granger_max_lag
            )
            
            logger.info(f"Granger test complete: {len(granger_results)} results")
        
    except Exception as e:
        logger.error(f"Granger causality test failed: {e}")
        raise
    
    # ── Summary ──────────────────────────────────────────────────────────────
    if isinstance(granger_results, pd.DataFrame):
        if len(granger_results) == 0:
            logger.warning("\n" + "="*80)
            logger.warning("GRANGER TEST: No results (insufficient data)")
            logger.warning("="*80)
            logger.warning("To run a proper Granger causality test, you need:")
            logger.warning("  • More sentiment data (run Stage 0 to collect fresh posts)")
            logger.warning("  • Longer time range (increase date range in config)")
            logger.warning("  • At least 30+ overlapping hourly data points per ticker")
            logger.warning("Stage 2B is optional - you can proceed with Stage 2C/3/3A")
            logger.warning("="*80)
        else:
            significant = granger_results[granger_results['p_value'] < config.granger_significance]
            
            logger.info(f"\nGranger Causality Results:")
            logger.info(f"  Total tests: {len(granger_results)}")
            logger.info(f"  Significant (p<{config.granger_significance}): {len(significant)}")
            
            if len(significant) > 0:
                logger.info(f"\nStocks where sentiment LEADS price:")
                for _, row in significant.iterrows():
                    logger.info(f"  {row['ticker']}: lag={row['lag']}, p={row['p_value']:.4f}")
            else:
                logger.warning("No significant Granger causality found for any stock!")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    output_file = config.output_dir / "granger_causality_results.csv"
    if isinstance(granger_results, pd.DataFrame):
        granger_results.to_csv(output_file, index=False)
    
    logger.info(f"Stage 2B complete")
    logger.info(f"Saved to {output_file}\n")
    
    return granger_results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2C: FEATURE FUSION
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_2c(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: PipelineConfig
) -> pd.DataFrame:
    """
    Stage 2C: Feature Fusion
    
    Combines sentiment features (Stage 1C) with market features (Stage 2A).
    Adds interaction features using lstm.py's build_interaction_features().
    
    Returns:
        DataFrame with all features ready for Stage 3
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 2C: FEATURE FUSION")
    logger.info("="*80)
    
    try:
        from analyser import join_to_price
        from lstm import build_interaction_features
        
        # ── Fuse sentiment + market features ─────────────────────────────────
        logger.info("Fusing sentiment and market features...")
        
        # Ensure sentiment_df has window_start as datetime (required by join_to_price)
        if 'window_start' in sentiment_df.columns:
            sentiment_df['window_start'] = pd.to_datetime(sentiment_df['window_start'], utc=True)
        
        fused = join_to_price(
            price_df=market_df,
            features_df=sentiment_df,
            price_time_col='timestamp_utc',
            price_ticker_col='ticker',
            fill_missing='zero'  # Fill missing sentiment with 0
        )
        
        logger.info(f"Features fused: {fused.shape}")
        
        # ── Add interaction features ─────────────────────────────────────────
        logger.info("Adding interaction features...")
        fused = build_interaction_features(fused)
        logger.info(f"Interaction features added")
        
    except Exception as e:
        logger.error(f"Feature fusion failed: {e}")
        raise
    
    # ── Summary ──────────────────────────────────────────────────────────────
    feature_cols = [c for c in fused.columns if c not in ['ticker', 'timestamp_utc', 'target', 'direction']]
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Total rows: {len(fused)}")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    output_file = config.output_dir / "fused_features.csv"
    fused.to_csv(output_file, index=False)
    
    logger.info(f"Stage 2C complete")
    logger.info(f"Saved to {output_file}\n")
    
    return fused


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: PREDICTION MODEL (XGBoost + LSTM Ensemble)
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_3(fused_df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Stage 3: Prediction Model (XGBoost + LSTM Ensemble)
    
    Trains ensemble of XGBoost + LSTM using lstm.py's run_stage3().
    
    Returns:
        dict with keys: 'train_metrics', 'val_metrics', 'test_metrics', 'models'
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: PREDICTION MODEL (XGBoost + LSTM Ensemble)")
    logger.info("="*80)
    
    try:
        from lstm import run_stage3
        from feature import time_split, add_target
        
        # ── Add target variable ──────────────────────────────────────────────
        if 'target' not in fused_df.columns and 'direction' not in fused_df.columns:
            logger.info("Computing target variable (next_return direction)...")
            fused_df = add_target(fused_df, threshold=0.0)  # Classification: up/down
        
        # ── Time-based split ─────────────────────────────────────────────────
        logger.info(f"Splitting data: train={config.train_split}, val={config.val_split}, test={config.test_split}")
        
        train, val, test = time_split(
            fused_df,
            train_ratio=config.train_split,
            val_ratio=config.val_split
        )
        
        logger.info(f"  Train: {len(train)} rows")
        logger.info(f"  Val:   {len(val)} rows")
        logger.info(f"  Test:  {len(test)} rows")
        
        # ── Train ensemble ───────────────────────────────────────────────────
        logger.info("Training XGBoost + LSTM ensemble...")
        
        results = run_stage3(
            train=train,
            val=val,
            test=test
        )
        
        logger.info(f"Ensemble training complete")
        
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        raise
    
    # ── Summary ──────────────────────────────────────────────────────────────
    # Note: run_stage3 prints metrics directly but doesn't return structured results
    if results is not None and isinstance(results, dict) and 'test_metrics' in results:
        metrics = results['test_metrics']
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
        logger.info(f"  F1 Score:  {metrics.get('f1_weighted', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.2%}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.2%}")
    else:
        logger.info(f"\nMetrics were logged during training (see output above)")
    
    logger.info(f"Stage 3 complete\n")
    
    # Return results or empty dict
    return results if results is not None else {}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3A: ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_3a(fused_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Stage 3A: Ablation Study
    
    Trains 4 versions and compares to prove sentiment value:
      v1_baseline:   Close price only (~52% accuracy)
      v2_technical:  + Technical indicators (~58-62%)
      v3_market:     + Market features like VIX (~62-65%)
      v4_with_sentiment: + All sentiment features (~65-70%)
    
    The gap between v3_market and v4_with_sentiment is your thesis proof.
    
    Returns:
        DataFrame with ablation results
    """
    logger.info("\n" + "="*80)
    logger.info("STAGE 3A: ABLATION STUDY")
    logger.info("="*80)
    
    try:
        from lstm import run_ablation
        from feature import time_split, add_target
        
        # ── Add target variable ──────────────────────────────────────────────
        if 'target' not in fused_df.columns and 'direction' not in fused_df.columns:
            logger.info("Computing target variable...")
            fused_df = add_target(fused_df, threshold=0.0)
        
        # ── Time-based split ─────────────────────────────────────────────────
        logger.info(f"Splitting data...")
        train, val, test = time_split(
            fused_df,
            train_ratio=config.train_split,
            val_ratio=config.val_split
        )
        
        logger.info(f"  Train: {len(train)} rows")
        logger.info(f"  Val:   {len(val)} rows")
        logger.info(f"  Test:  {len(test)} rows")
        
        # ── Run ablation study ───────────────────────────────────────────────
        logger.info("\nRunning ablation study (4 configurations)...")
        logger.info("This will train XGBoost for each configuration...\n")
        
        ablation_results = run_ablation(train, val, test)
        
        logger.info(f"Ablation study complete")
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        raise
    
    # ── Display results ──────────────────────────────────────────────────────
    if isinstance(ablation_results, pd.DataFrame) and len(ablation_results) > 0:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY RESULTS (Test Set)")
        logger.info("="*80)
        
        for _, row in ablation_results.iterrows():
            logger.info(f"\n{row['version']}:")
            logger.info(f"  Features: {row['n_features']}")
            logger.info(f"  Accuracy: {row['accuracy']:.2%}")
            logger.info(f"  F1 Score: {row['f1_weighted']:.4f}")
        
        # Calculate improvement
        if len(ablation_results) >= 2:
            baseline = ablation_results.iloc[0]['accuracy']
            best = ablation_results['accuracy'].max()
            improvement = best - baseline
            logger.info(f"\n{'='*80}")
            logger.info(f"IMPROVEMENT: {improvement:.2%} accuracy gain from baseline")
            logger.info(f"{'='*80}")
    
    logger.info(f"\nStage 3A complete\n")
    
    return ablation_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(stages: List[str], config: PipelineConfig) -> Dict[str, Any]:
    """
    Main pipeline orchestrator.
    
    Runs requested stages in order, handling dependencies automatically.
    
    Args:
        stages: List of stage codes ('0', '1A', '1B', '1C', '2A', '2B', '2C', '3', '3A')
        config: Pipeline configuration
    
    Returns:
        dict with stage outputs
    """
    stage_data = {}
    
    logger.info("\n" + "="*80)
    logger.info("SENTIMENT-DRIVEN MARKET ANALYSIS PIPELINE")
    logger.info("="*80)
    logger.info(f"Stages to run: {', '.join(stages)}")
    logger.info(str(config))
    logger.info("="*80 + "\n")
    
    try:
        # ── Stage 0: Raw Data Collection ────────────────────────────────────
        if '0' in stages:
            stage_data['0'] = run_stage_0(config)
        
        # ── Stage 1A: Data Cleaning ──────────────────────────────────────────
        if '1A' in stages:
            raw_data = stage_data.get('0', None)
            stage_data['1A'] = run_stage_1a(config, raw_data)
        elif any(s in stages for s in ['1B', '1C', '2B', '2C', '3', '3A']):
            # Auto-load 1A if needed by downstream stages
            logger.info("Stage 1A not requested but needed downstream, attempting to load from disk...")
            try:
                stage_data['1A'] = run_stage_1a(config, None)
            except FileNotFoundError:
                logger.warning("No cleaned data found. Run Stage 1A first or provide raw data.")
        
        # ── Stage 1B: Sentiment Analysis ─────────────────────────────────────
        if '1B' in stages:
            if '1A' not in stage_data:
                raise ValueError("Stage 1B requires Stage 1A data")
            stage_data['1B'] = run_stage_1b(stage_data['1A'], config)
        elif any(s in stages for s in ['1C', '2B', '2C', '3', '3A']):
            # Try to load from disk
            import glob
            sentiment_files = sorted(glob.glob(str(config.processed_dir / "stage1b_final_*.csv")))
            if sentiment_files:
                logger.info(f"Loading Stage 1B from disk: {sentiment_files[-1]}")
                stage_data['1B'] = pd.read_csv(sentiment_files[-1])
            elif '1A' in stage_data:
                logger.info("Stage 1B not requested but needed, running now...")
                stage_data['1B'] = run_stage_1b(stage_data['1A'], config)
        
        # ── Stage 1C: Sentiment Aggregation ──────────────────────────────────
        if '1C' in stages:
            if '1B' not in stage_data:
                raise ValueError("Stage 1C requires Stage 1B data")
            stage_data['1C'] = run_stage_1c(stage_data['1B'], config)
        elif any(s in stages for s in ['2B', '2C', '3', '3A']):
            # Try to load from disk
            features_file = config.output_dir / "sentiment_features.csv"
            if features_file.exists():
                logger.info(f"Loading Stage 1C from disk: {features_file}")
                stage_data['1C'] = pd.read_csv(features_file)
        
        # ── Stage 2A: Market Features ────────────────────────────────────────
        if '2A' in stages:
            stage_data['2A'] = run_stage_2a(config)
        elif any(s in stages for s in ['2B', '2C', '3', '3A']):
            # Try to load from disk
            market_file = config.output_dir / "market_features.csv"
            if market_file.exists():
                logger.info(f"Loading Stage 2A from disk: {market_file}")
                stage_data['2A'] = pd.read_csv(market_file)
            else:
                logger.info("Stage 2A not requested but needed, running now...")
                stage_data['2A'] = run_stage_2a(config)
        
        # ── Stage 2B: Granger Causality ──────────────────────────────────────
        if '2B' in stages:
            if '1C' not in stage_data or '2A' not in stage_data:
                raise ValueError("Stage 2B requires both Stage 1C and 2A data")
            stage_data['2B'] = run_stage_2b(stage_data['1C'], stage_data['2A'], config)
        
        # ── Stage 2C: Feature Fusion ─────────────────────────────────────────
        if '2C' in stages:
            if '1C' not in stage_data or '2A' not in stage_data:
                raise ValueError("Stage 2C requires both Stage 1C and 2A data")
            stage_data['2C'] = run_stage_2c(stage_data['1C'], stage_data['2A'], config)
        elif any(s in stages for s in ['3', '3A']):
            # Try to load from disk
            fused_file = config.output_dir / "fused_features.csv"
            if fused_file.exists():
                logger.info(f"Loading Stage 2C from disk: {fused_file}")
                stage_data['2C'] = pd.read_csv(fused_file)
                
                # Fix column names if needed (from old merge with suffixes)
                if 'timestamp_utc_x' in stage_data['2C'].columns:
                    logger.info("Fixing timestamp_utc column names from old merge...")
                    stage_data['2C'].rename(columns={'timestamp_utc_x': 'timestamp_utc'}, inplace=True)
                    stage_data['2C'].drop(columns=['timestamp_utc_y'], inplace=True, errors='ignore')
        
        # ── Stage 3: Model Training ──────────────────────────────────────────
        if '3' in stages:
            if '2C' not in stage_data:
                raise ValueError("Stage 3 requires Stage 2C data (fused features)")
            stage_data['3'] = run_stage_3(stage_data['2C'], config)
        
        # ── Stage 3A: Ablation Study ─────────────────────────────────────────
        if '3A' in stages:
            if '2C' not in stage_data:
                raise ValueError("Stage 3A requires Stage 2C data (fused features)")
            stage_data['3A'] = run_stage_3a(stage_data['2C'], config)
        
        # ── Pipeline Complete ────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*80)
        
        # Summary
        logger.info("\nOutput Summary:")
        for stage, data in stage_data.items():
            if isinstance(data, pd.DataFrame):
                logger.info(f"  Stage {stage}: {len(data)} rows x {len(data.columns)} columns")
            elif isinstance(data, dict):
                logger.info(f"  Stage {stage}: {len(data)} items")
        
        logger.info(f"\nAll outputs saved to: {config.output_dir}")
        logger.info("="*80 + "\n")
        
        return stage_data
    
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(f"{'='*80}\n")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment-Driven Financial Market Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['0', '1A', '1B', '1C', '2A', '2B', '2C', '3', '3A'],
        help='Specific stages to run (e.g., --stages 1B 1C 2A 2C 3)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run all stages (0 through 3A)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: Run 1B, 1C, 2A, 2C with existing data'
    )
    
    parser.add_argument(
        '--ticker',
        nargs='+',
        help='Ticker symbols to analyze (default: AAPL TSLA NVDA GME AMZN)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--finbert-model',
        type=str,
        default='finbert_finetuned_2/best_model',
        help='FinBERT model path (default: finbert_finetuned_2/best_model)'
    )
    
    args = parser.parse_args()
    
    # ── Determine stages to run ──────────────────────────────────────────────
    if args.full:
        stages = ['0', '1A', '1B', '1C', '2A', '2B', '2C', '3', '3A']
    elif args.quick:
        stages = ['1B', '1C', '2A', '2C']
    elif args.stages:
        stages = args.stages
    else:
        parser.print_help()
        sys.exit(1)
    
    # ── Configure pipeline ───────────────────────────────────────────────────
    config = PipelineConfig()
    
    if args.ticker:
        config.tickers = args.ticker
    
    if args.start:
        config.start_date = args.start
    
    if args.end:
        config.end_date = args.end
    
    if args.finbert_model:
        config.finbert_model = args.finbert_model
    
    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        run_pipeline(stages, config)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
