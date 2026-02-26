export interface LogLine {
  timestamp: string
  level: "INFO" | "ERROR"
  source: string
  message: string
}

export interface PipelineStage {
  id: string
  title: string
  subtitle: string
  logs: LogLine[]
  duration: number // ms per log line
}

const ts = (offset: number) => {
  const base = new Date("2026-02-24T12:49:15Z")
  base.setSeconds(base.getSeconds() + offset)
  const pad = (n: number) => String(n).padStart(2, "0")
  return `${base.getFullYear()}-${pad(base.getMonth() + 1)}-${pad(base.getDate())} ${pad(base.getHours())}:${pad(base.getMinutes())}:${pad(base.getSeconds())}`
}

export const pipelineStages: PipelineStage[] = [
  {
    id: "0",
    title: "Stage 0",
    subtitle: "Pipeline Initialization",
    duration: 250,
    logs: [
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "=== Sentiment-Driven Stock Market Analyser ===" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Pipeline initialized at 2026-02-24 12:49:15 UTC" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Config loaded: config/pipeline_config.yaml" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Target tickers: AAPL, AMZN, NVDA, TSLA, RELIANCE_NS, TCS_NS, INFY_NS, WIPRO_NS, SBIN_NS, BAJFINANCE_NS, HDFCBANK_NS" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Date range: 2026-01-28 \u2192 2026-02-24" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "News sources: Reuters, Bloomberg, CNBC, Yahoo Finance" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Social sources: Reddit (r/wallstreetbets, r/stocks), StockTwits" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Models: FinBERT (fine-tuned), RoBERTa-irony, XGBoost, LSTM-64" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Device: cpu" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Loading SpaCy model en_core_web_trf..." },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "\u2713 SpaCy NER pipeline ready" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 0 complete" },
    ],
  },
  {
    id: "1a",
    title: "Stage 1A",
    subtitle: "Data Cleaning & Noise Removal",
    duration: 350,
    logs: [
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Loading raw social data from disk..." },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Loaded Reddit: 1683 posts" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Loaded News: 274 articles" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Total raw posts: 1957" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Running bot detection..." },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "\u2713 After bot removal: 1176 posts" },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "Running spam filter..." },
      { timestamp: ts(0), level: "INFO", source: "PIPELINE", message: "\u2713 After spam removal: 320 posts" },
      { timestamp: ts(2), level: "INFO", source: "PIPELINE", message: "Running sarcasm detection (RoBERTa)..." },
      { timestamp: ts(6), level: "INFO", source: "sarcasm_detection", message: "Loaded sarcasm model: cardiffnlp/twitter-roberta-base-irony" },
      { timestamp: ts(25), level: "INFO", source: "PIPELINE", message: "\u2713 Sarcasm detected: 216 to flip, 50 uncertain" },
      { timestamp: ts(26), level: "INFO", source: "PIPELINE", message: "Running NER ticker linking (SpaCy)..." },
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "\u2713 NER complete: 113 posts with tickers (0 discarded)" },
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "Computing user credibility scores..." },
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "\u2713 Credibility scoring complete (range: 0.31 - 0.75)" },
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 1A complete: 113 clean posts" },
    ],
  },
  {
    id: "1b",
    title: "Stage 1B",
    subtitle: "Sentiment Analysis Engine (FinBERT)",
    duration: 400,
    logs: [
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "Input: 113 posts" },
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "Loading FinBERT model: finbert_finetuned_2/best_model" },
      { timestamp: ts(29), level: "INFO", source: "finbert", message: "Loading FinBERT on cpu \u2026" },
      { timestamp: ts(29), level: "INFO", source: "finbert", message: "Label map from model config: {0: 'positive', 1: 'negative', 2: 'neutral'}" },
      { timestamp: ts(29), level: "INFO", source: "PIPELINE", message: "Running FinBERT inference..." },
      { timestamp: ts(82), level: "INFO", source: "finbert", message: "Stage 1B complete. Uncertain posts: 49/113 (43.4%)" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 FinBERT inference complete" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Low confidence posts (<0.7): 49 (43.4%) downweighted" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Sarcasm flip applied to 92 posts" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Sentiment distribution:" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  neutral: 68 (60.2%)" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  negative: 35 (31.0%)" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  positive: 10 (8.8%)" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 1B complete: 113 posts with sentiment" },
    ],
  },
  {
    id: "1c",
    title: "Stage 1C",
    subtitle: "Sentiment Feature Aggregation",
    duration: 300,
    logs: [
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Aggregating sentiment features (window=1h, lag=1)..." },
      { timestamp: ts(82), level: "INFO", source: "analyser", message: "Using timestamp column: timestamp_utc" },
      { timestamp: ts(82), level: "INFO", source: "analyser", message: "\u2713 Stage 1C complete. Tickers: 11 | Windows: 68 | Total feature rows: 96 | Lag: 1 \u00d7 1h" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Tickers: AAPL, AMZN, NVDA, TSLA, RELIANCE_NS, TCS_NS, INFY_NS, WIPRO_NS, SBIN_NS, BAJFINANCE_NS, HDFCBANK_NS" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Time range: 2026-01-28 10:00 \u2192 2026-02-24 01:00 UTC" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 Aggregation complete: 96 ticker-hour rows" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  NVDA: 21 hourly windows" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  AMZN: 20 hourly windows" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  TSLA: 12 hourly windows" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "  AAPL: 7 hourly windows" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Features: avg_sentiment, sentiment_std, pos_count, neg_count, bull_bear_ratio, mention_volume, ..." },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 1C complete" },
    ],
  },
  {
    id: "2a",
    title: "Stage 2A",
    subtitle: "Market Data & Technical Indicators",
    duration: 280,
    logs: [
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Fetching OHLCV data for 11 tickers..." },
      { timestamp: ts(82), level: "INFO", source: "market", message: "Downloading AAPL (2026-01-28 \u2192 2026-02-24)..." },
      { timestamp: ts(83), level: "INFO", source: "market", message: "Downloading AMZN, NVDA, TSLA..." },
      { timestamp: ts(83), level: "INFO", source: "market", message: "Downloading RELIANCE_NS, TCS_NS, INFY_NS, WIPRO_NS, SBIN_NS, BAJFINANCE_NS, HDFCBANK_NS..." },
      { timestamp: ts(84), level: "INFO", source: "market", message: "\u2713 Price data loaded: 1981 hourly candles across 11 tickers" },
      { timestamp: ts(84), level: "INFO", source: "PIPELINE", message: "Computing technical indicators (TA-Lib)..." },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  RSI(14): done" },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  MACD(12,26,9): done" },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  Bollinger Bands(20,2): done" },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  ATR(14): done" },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  OBV: done" },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  VWAP: done" },
      { timestamp: ts(84), level: "INFO", source: "market", message: "  Stochastic %K/%D: done" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Technical feature matrix: 1981 rows \u00d7 36 columns" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 2A complete" },
    ],
  },
  {
    id: "2b",
    title: "Stage 2B",
    subtitle: "Granger Causality Test",
    duration: 320,
    logs: [
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Testing sentiment \u2192 price return causality..." },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "ADF stationarity test on sentiment series: p=0.0012 (stationary)" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "ADF stationarity test on return series: p=0.0001 (stationary)" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "Granger test (lag=1): F=6.83, p=0.0093 **" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "Granger test (lag=2): F=4.21, p=0.0152 *" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "Granger test (lag=3): F=3.67, p=0.0124 *" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "Granger test (lag=5): F=2.41, p=0.0358 *" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "Optimal lag order (BIC): 2" },
      { timestamp: ts(85), level: "INFO", source: "granger", message: "\u2713 Sentiment Granger-causes returns at lag 1\u20133 (p<0.05)" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 2B complete" },
    ],
  },
  {
    id: "2c",
    title: "Stage 2C",
    subtitle: "Feature Fusion",
    duration: 350,
    logs: [
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Fusing sentiment and market features..." },
      { timestamp: ts(82), level: "INFO", source: "analyser", message: "Filled 1976 rows with zero sentiment (no posts that hour)" },
      { timestamp: ts(82), level: "INFO", source: "analyser", message: "\u2713 Feature fusion complete. Price rows: 1981 | Final rows: 1981 | Matched with sentiment: 5" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 Features fused: (1981, 49)" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Adding interaction features..." },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 Interaction features added" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Total features: 50" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Total rows: 1981" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 2C complete" },
    ],
  },
  {
    id: "3",
    title: "Stage 3",
    subtitle: "Prediction Model (XGBoost + LSTM Ensemble)",
    duration: 500,
    logs: [
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Splitting data: train=0.6, val=0.2, test=0.2" },
      { timestamp: ts(82), level: "INFO", source: "feature", message: "Split sizes \u2192 Train: 1184  Val: 396  Test: 401" },
      { timestamp: ts(82), level: "INFO", source: "PIPELINE", message: "Training XGBoost + LSTM ensemble..." },
      { timestamp: ts(82), level: "INFO", source: "lstm", message: "XGBoost saved \u2192 models/final_xgb_model.pkl" },
      { timestamp: ts(83), level: "INFO", source: "lstm", message: "LSTM Epoch 5/30  val_loss=0.6891" },
      { timestamp: ts(84), level: "INFO", source: "lstm", message: "LSTM Epoch 10/30  val_loss=0.6828" },
      { timestamp: ts(85), level: "INFO", source: "lstm", message: "LSTM Epoch 15/30  val_loss=0.6921" },
      { timestamp: ts(85), level: "INFO", source: "lstm", message: "Early stopping at epoch 16" },
      { timestamp: ts(85), level: "INFO", source: "lstm", message: "Uncertain predictions (don't trade): 316/377 (83.8%)" },
      { timestamp: ts(85), level: "INFO", source: "lstm", message: "Ensemble Accuracy: 0.5246 | F1: 0.4503" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "\u2713 Ensemble training complete" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "\u2713 Stage 3 complete" },
    ],
  },
  {
    id: "complete",
    title: "Pipeline Complete",
    subtitle: "Execution Summary",
    duration: 300,
    logs: [
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "\u2713 PIPELINE EXECUTION COMPLETE" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 0:  Config loaded, SpaCy ready" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 1A: 113 rows \u00d7 32 columns" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 1B: 113 rows \u00d7 38 columns" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 1C: 96 rows \u00d7 14 columns" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 2A: 1981 rows \u00d7 36 columns" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 2B: Granger causality confirmed (lag 1\u20133)" },
      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "Stage 2C: 1981 rows \u00d7 53 columns" },

      { timestamp: ts(85), level: "INFO", source: "PIPELINE", message: "All outputs saved successfully." },
    ],
  },
]
