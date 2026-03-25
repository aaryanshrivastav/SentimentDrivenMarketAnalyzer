"""
STAGE 3 — PREDICTION MODEL: XGBoost + LSTM Ensemble
Includes ablation study (Stage 3A) comparing 4 feature configurations.
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

SEED       = 42
SEQUENCE_LEN = 24        # 24 hourly bars = 1 day lookback for LSTM
ENSEMBLE_W_XGB  = 0.5
ENSEMBLE_W_LSTM = 0.5
UP_THRESH    = 0.55      # predict Up
DOWN_THRESH  = 0.45      # predict Down, between = Uncertain (don't trade)
OUTPUT_DIR   = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)


def _lstm_checkpoint_path(name: str, input_size: int) -> Path:
    """Checkpoint path scoped by input feature count to avoid shape-mismatch restores."""
    return OUTPUT_DIR / f"{name}_best_in{input_size}.pt"


def _read_threshold_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        if 0.0 < v < 1.0:
            return v
    except ValueError:
        pass
    logger.warning("Invalid %s='%s', using default %.3f", name, raw, default)
    return default


# Slightly narrower uncertain zone by default; configurable via env vars.
UP_THRESH = _read_threshold_env("ENSEMBLE_UP_THRESH", 0.52)
DOWN_THRESH = _read_threshold_env("ENSEMBLE_DOWN_THRESH", 0.48)
if DOWN_THRESH >= UP_THRESH:
    logger.warning(
        "Invalid threshold pair (down=%.3f, up=%.3f); resetting to 0.48/0.52",
        DOWN_THRESH,
        UP_THRESH,
    )
    DOWN_THRESH = 0.48
    UP_THRESH = 0.52


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS — used for ablation study
# ─────────────────────────────────────────────────────────────────────────────

# Version 1 — Baseline
PRICE_FEATURES = ["Close", "return_1h", "return_3h", "return_24h", "Volume"]

# Version 2 — + Technical indicators
TECHNICAL_FEATURES = PRICE_FEATURES + [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "atr_14",
    "volume_change", "volume_ma_ratio", "volatility_24h", "obv",
]

# Version 3 — + Sentiment
SENTIMENT_FEATURES = TECHNICAL_FEATURES + [
    "avg_sentiment", "sentiment_std", "bull_bear_ratio",
    "mention_volume", "weighted_volume", "sentiment_momentum",
    "sentiment_acceleration", "high_confidence_ratio",
    "pos_count", "neg_count", "neu_count",
    "sentiment_available", "sentiment_imputed",
]

# Version 4 — Full (+ events + interactions)
def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates interaction terms that XGBoost can't find on its own."""
    df = df.copy()
    if "rsi_14" in df.columns and "avg_sentiment" in df.columns:
        # Oversold AND positive sentiment = strong buy signal
        df["rsi_x_sentiment"]  = df["rsi_14"] * df["avg_sentiment"]
        df["macd_x_sentiment"] = df["macd"]   * df["avg_sentiment"]
        df["vol_x_sentiment"]  = df["volume_change"] * df["avg_sentiment"]
        df["sentiment_x_earnings_tomorrow"] = (
            df.get("avg_sentiment", 0) * df.get("earnings_tomorrow", 0)
        )
    return df

ALL_FEATURES = SENTIMENT_FEATURES + [
    "earnings_today", "earnings_tomorrow",
    "rsi_x_sentiment", "macd_x_sentiment",
    "vol_x_sentiment", "sentiment_x_earnings_tomorrow",
]

ABLATION_CONFIGS = {
    "v1_baseline":    PRICE_FEATURES,
    "v2_technical":   TECHNICAL_FEATURES,
    "v3_sentiment":   SENTIMENT_FEATURES,
    "v4_full":        ALL_FEATURES,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_available_features(df: pd.DataFrame, desired: list[str]) -> list[str]:
    """Return only features that actually exist in df."""
    available = [f for f in desired if f in df.columns]
    missing   = [f for f in desired if f not in df.columns]
    if missing:
        logger.warning(f"Missing features (will skip): {missing}")
    return available


def _apply_sentiment_quality_filter(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optionally filter to higher-confidence sentiment rows for sentiment-heavy variants."""
    if version not in {"v3_sentiment", "v4_full"}:
        return train, val, test

    mode = os.getenv("ABLATION_SENTIMENT_FILTER", "available_only").strip().lower()
    min_rows = int(os.getenv("ABLATION_MIN_ROWS", "200"))

    if mode == "off":
        return train, val, test

    if "sentiment_available" not in train.columns:
        logger.warning("sentiment_available not found; skipping sentiment-quality filtering for %s", version)
        return train, val, test

    def _mask(df: pd.DataFrame) -> pd.Series:
        avail = df.get("sentiment_available", 0).fillna(0).astype(int) == 1
        if mode == "available_only":
            return avail
        if mode == "available_or_not_imputed" and "sentiment_imputed" in df.columns:
            not_imp = df["sentiment_imputed"].fillna(0).astype(int) == 0
            return avail | not_imp
        return avail

    tr2 = train[_mask(train)].copy()
    v2 = val[_mask(val)].copy()
    te2 = test[_mask(test)].copy()

    # Fallback if filtered slices become too small for stable training/testing.
    if min(len(tr2), len(v2), len(te2)) < min_rows:
        logger.warning(
            "%s: sentiment-quality filter '%s' produced too few rows (train=%d, val=%d, test=%d). Using full split.",
            version, mode, len(tr2), len(v2), len(te2)
        )
        return train, val, test

    logger.info(
        "%s: sentiment-quality filter '%s' applied (train %d->%d, val %d->%d, test %d->%d)",
        version, mode, len(train), len(tr2), len(val), len(v2), len(test), len(te2)
    )
    return tr2, v2, te2


def _best_threshold_from_val(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    """Pick threshold maximizing validation accuracy, then F1 as tie-breaker."""
    best_t = 0.5
    best_acc = -1.0
    best_f1 = -1.0

    for t in np.linspace(0.35, 0.65, 31):
        preds = (prob_up >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
        if (acc > best_acc) or (acc == best_acc and f1 > best_f1):
            best_acc, best_f1, best_t = acc, f1, float(t)

    return best_t


def prep_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) as numpy arrays, dropping NaN and inf rows."""
    cols = feature_cols + ["target"]
    sub  = df[cols].copy()
    
    # Replace inf with NaN, then drop
    sub = sub.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN in any column
    initial_len = len(sub)
    sub = sub.dropna()
    if len(sub) < initial_len:
        logger.debug(f"Dropped {initial_len - len(sub)} rows with NaN/inf values")
    
    X = sub[feature_cols].values.astype(np.float32)
    y = sub["target"].values.astype(np.int64)
    
    # Final check for any remaining inf/nan
    if not np.isfinite(X).all():
        logger.warning(f"Warning: X still contains inf/nan after cleaning")
        # Replace any remaining inf/nan with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 8760) -> float:
    """Annualised Sharpe on hourly strategy returns."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def evaluate(y_true, y_pred, prices=None, label="") -> dict:
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("%s evaluation skipped: empty y_true/y_pred after filtering.", label)
        return {
            "label": label,
            "accuracy": float("nan"),
            "f1_weighted": float("nan"),
            "n_samples": 0,
        }

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    result = {"label": label, "accuracy": acc, "f1_weighted": f1, "n_samples": len(y_true)}

    if prices is not None:
        # Simulate: go long when predict=1, short when predict=0
        direction = np.where(y_pred == 1, 1, -1)
        strategy_returns = direction * prices
        result["sharpe"] = sharpe_ratio(strategy_returns)

    logger.info(f"\n{'-'*40}")
    logger.info(f"  {label}")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  F1       : {f1:.4f}")
    if "sharpe" in result:
        logger.info(f"  Sharpe   : {result['sharpe']:.4f}")
    logger.info(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Down", "Up"],
            zero_division=0,
        )
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train, y_train, X_val, y_val,
    name: str = "xgb"
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        use_label_encoder= False,
        eval_metric      = "logloss",
        early_stopping_rounds = 30,
        random_state     = SEED,
        n_jobs           = -1,
        device           = "cuda" if torch.cuda.is_available() else "cpu",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    joblib.dump(model, OUTPUT_DIR / f"{name}_model.pkl")
    logger.info(f"XGBoost saved to {OUTPUT_DIR}/{name}_model.pkl")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class FinLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden, 32)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(32, 2)          # binary: Down / Up

    def forward(self, x):
        out, _ = self.lstm1(x)
        out     = self.drop(out)
        out, _  = self.lstm2(out)
        out     = self.drop(out[:, -1, :])     # last timestep only
        out     = self.relu(self.fc1(out))
        return self.fc2(out)                   # logits


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Convert flat (N, F) arrays → sequences (N-seq_len, seq_len, F)."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def train_lstm(
    X_train, y_train, X_val, y_val,
    seq_len: int = SEQUENCE_LEN,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    name: str = "lstm",
) -> FinLSTM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_train.shape[1]

    # Build sequences
    Xs_tr, ys_tr = make_sequences(X_train, y_train, seq_len)
    Xs_val, ys_val = make_sequences(X_val, y_val, seq_len)

    tr_loader  = DataLoader(TensorDataset(
        torch.tensor(Xs_tr), torch.tensor(ys_tr)
    ), batch_size=batch_size, shuffle=False)   # do NOT shuffle time-series

    val_loader = DataLoader(TensorDataset(
        torch.tensor(Xs_val), torch.tensor(ys_val)
    ), batch_size=batch_size * 2, shuffle=False)

    model     = FinLSTM(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = float("inf")
    patience_ctr  = 0
    PATIENCE      = 7
    ckpt_path = _lstm_checkpoint_path(name=name, input_size=input_size)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses) if len(val_losses) > 0 else float("inf")
        scheduler.step(val_loss)

        if epoch % 5 == 0:
            logger.info(f"LSTM Epoch {epoch}/{epochs}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        logger.warning(
            "LSTM checkpoint not found at %s. Returning last-epoch weights.",
            ckpt_path,
        )
    return model


def lstm_predict_proba(model: FinLSTM, X: np.ndarray, seq_len: int) -> np.ndarray:
    """Returns probability of Up class for each sample (after seq_len offset)."""
    device = next(model.parameters()).device
    Xs, _  = make_sequences(X, np.zeros(len(X)), seq_len)
    loader  = DataLoader(torch.tensor(Xs), batch_size=512, shuffle=False)
    probs   = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(
    xgb_prob: np.ndarray,
    lstm_prob: np.ndarray,
    w_xgb: float = ENSEMBLE_W_XGB,
    w_lstm: float = ENSEMBLE_W_LSTM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (final_prob, predicted_label).
    Labels: 1=Up, 0=Down, -1=Uncertain (don't trade).
    """
    # Align lengths (LSTM is shorter by seq_len due to sequence windowing)
    min_len   = min(len(xgb_prob), len(lstm_prob))
    xgb_prob  = xgb_prob[-min_len:]
    lstm_prob = lstm_prob[-min_len:]

    final_prob = w_xgb * xgb_prob + w_lstm * lstm_prob
    labels = np.where(
        final_prob > UP_THRESH,   1,
        np.where(final_prob < DOWN_THRESH, 0, -1)
    )
    return final_prob, labels


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_xgb(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 6,
) -> list[dict]:
    """
    Trains XGBoost on expanding window, evaluates on next fold.
    Returns list of per-fold metric dicts.
    """
    df   = df.sort_values("timestamp_utc")
    fold_size = len(df) // (n_splits + 1)
    results   = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end   = fold_size * (i + 1)

        train_fold = df.iloc[:train_end]
        val_fold   = df.iloc[train_end:val_end]

        X_tr, y_tr  = prep_xy(train_fold, feature_cols)
        X_val, y_val = prep_xy(val_fold,  feature_cols)

        if len(np.unique(y_tr)) < 2 or len(X_val) == 0:
            continue

        scaler = StandardScaler().fit(X_tr)
        model  = train_xgboost(scaler.transform(X_tr), y_tr,
                               scaler.transform(X_val), y_val,
                               name=f"wf_fold{i}")
        preds  = model.predict(scaler.transform(X_val))
        res    = evaluate(y_val, preds, label=f"Walk-Forward Fold {i}")
        results.append(res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY (Stage 3A)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Trains XGBoost for all 4 feature configurations, reports metrics.
    This is your thesis experiment — the V3→V4 jump is what you report.
    """
    train = build_interaction_features(train)
    val   = build_interaction_features(val)
    test  = build_interaction_features(test)

    summary = []

    for version, desired_features in ABLATION_CONFIGS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"  ABLATION: {version}")
        logger.info(f"{'='*50}")

        tr_df, val_df, te_df = _apply_sentiment_quality_filter(train, val, test, version)

        feature_cols = get_available_features(tr_df, desired_features)
        if not feature_cols:
            logger.warning(f"No features available for {version} - skipping.")
            continue

        X_tr,  y_tr  = prep_xy(tr_df, feature_cols)
        X_val, y_val = prep_xy(val_df, feature_cols)
        X_te,  y_te  = prep_xy(te_df, feature_cols)

        if min(len(X_tr), len(X_val), len(X_te)) == 0:
            logger.warning("%s: empty split after preprocessing; skipping.", version)
            continue

        scaler  = StandardScaler().fit(X_tr)
        X_tr_s  = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s  = scaler.transform(X_te)

        xgb_model = train_xgboost(X_tr_s, y_tr, X_val_s, y_val, name=version)
        val_prob  = xgb_model.predict_proba(X_val_s)[:, 1]
        test_prob = xgb_model.predict_proba(X_te_s)[:, 1]

        tuned_t = _best_threshold_from_val(y_val, val_prob)
        logger.info("%s: tuned decision threshold=%.3f (from validation)", version, tuned_t)

        preds_val  = (val_prob >= tuned_t).astype(int)
        preds_test = (test_prob >= tuned_t).astype(int)

        val_result  = evaluate(y_val, preds_val,  label=f"{version} - Val")
        test_result = evaluate(y_te,  preds_test, label=f"{version} - Test")
        test_result["version"] = version
        test_result["n_features"] = len(feature_cols)
        test_result["decision_threshold"] = tuned_t
        summary.append(test_result)

    summary_df = pd.DataFrame(summary).sort_values("accuracy", ascending=False)
    print("\n-- ABLATION SUMMARY (Test Set) --")
    print(summary_df[["version","n_features","accuracy","f1_weighted"]].to_string(index=False))
    summary_df.to_csv("ablation_results.csv", index=False)
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    feature_cols: list[str] = None,
):
    """
    Trains the full XGBoost + LSTM ensemble on the best feature set.
    """
    if feature_cols is None:
        feature_cols = get_available_features(train, ALL_FEATURES)

    train = build_interaction_features(train)
    val   = build_interaction_features(val)
    test  = build_interaction_features(test)

    X_tr,  y_tr  = prep_xy(train, feature_cols)
    X_val, y_val = prep_xy(val,   feature_cols)
    X_te,  y_te  = prep_xy(test,  feature_cols)

    # Scale (fit on train only — no leakage)
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

    # ── XGBoost ──────────────────────────────────────────────────────
    xgb_model  = train_xgboost(X_tr_s, y_tr, X_val_s, y_val, name="final_xgb")
    xgb_prob_te = xgb_model.predict_proba(X_te_s)[:, 1]

    # ── LSTM ─────────────────────────────────────────────────────────
    lstm_model  = train_lstm(X_tr_s, y_tr, X_val_s, y_val, name="final_lstm")
    lstm_prob_te = lstm_predict_proba(lstm_model, X_te_s, SEQUENCE_LEN)

    # ── Ensemble ─────────────────────────────────────────────────────
    final_prob, ensemble_labels = ensemble_predict(xgb_prob_te, lstm_prob_te)

    # Align y_te with LSTM offset
    y_te_aligned = y_te[-len(ensemble_labels):]
    tradeable    = ensemble_labels != -1    # exclude Uncertain signals

    logger.info(
        "\nEnsemble thresholds: up=%.3f down=%.3f",
        UP_THRESH,
        DOWN_THRESH,
    )
    logger.info(f"Uncertain predictions (don't trade): "
                f"{(ensemble_labels == -1).sum()}/{len(ensemble_labels)} "
                f"({(ensemble_labels == -1).mean()*100:.1f}%)")

    if tradeable.sum() == 0:
        logger.warning(
            "No tradeable ensemble signals under current thresholds. "
            "Falling back to 0.5 cutoff for evaluation only."
        )
        fallback_labels = (final_prob >= 0.5).astype(int)
        evaluate(
            y_te_aligned,
            fallback_labels,
            label="Ensemble (fallback 0.5 cutoff)"
        )
    else:
        evaluate(
            y_te_aligned[tradeable],
            ensemble_labels[tradeable],
            label="Ensemble (tradeable signals only)"
        )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load feature matrix built by stage2_feature_engineering.py
    from src.sentiment.feature import build_feature_matrix, time_split

    matrix = build_feature_matrix()
    
    # Diagnostic: Check for data quality issues
    logger.info(f"\nFeature matrix shape: {matrix.shape}")
    inf_count = np.isinf(matrix.select_dtypes(include=[np.number]).values).sum()
    nan_count = matrix.select_dtypes(include=[np.number]).isna().sum().sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} inf values in feature matrix")
        inf_cols = [col for col in matrix.select_dtypes(include=[np.number]).columns 
                    if np.isinf(matrix[col]).any()]
        logger.warning(f"Columns with inf: {inf_cols}")
    if nan_count > 0:
        logger.info(f"Found {nan_count} NaN values (will be handled by prep_xy)")
    
    train, val, test = time_split(matrix)

    # Step 1: Ablation study
    run_ablation(train, val, test)

    # Step 2: Full ensemble on best feature set
    run_stage3(train, val, test)