import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import logging
import os
import re
from tqdm import tqdm


logger = logging.getLogger("sarcasm_detection")


class SarcasmDetection:

    def __init__(self, df, model_name=None):
        self.df = df.copy()
        self._normalize_input_schema()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cpu":
            self._configure_cpu_runtime()

        candidates = []
        if model_name:
            candidates.append(model_name)

        # Prefer a public model first, then legacy fallback.
        candidates.extend([
            "cardiffnlp/twitter-roberta-base-irony",
            "mrm8488/roberta-base-finetuned-sarcasm",
        ])

        self.model_name = None
        self.label_positive_index = 1
        last_error = None

        for candidate in candidates:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                self.model = AutoModelForSequenceClassification.from_pretrained(candidate)
                if self.device.type == "cpu":
                    self._maybe_quantize_model()
                self.model.to(self.device)
                self.model.eval()
                self.model_name = candidate
                self.label_positive_index = self._resolve_positive_index()
                logger.info("Loaded sarcasm model: %s", candidate)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Sarcasm model '%s' unavailable: %s", candidate, exc)

        if self.model_name is None:
            if os.getenv("ALLOW_SARCASM_FALLBACK", "0") == "1":
                logger.warning(
                    "No sarcasm model available. Using neutral fallback because ALLOW_SARCASM_FALLBACK=1."
                )
            else:
                raise RuntimeError(
                    "No sarcasm model could be loaded. Set ALLOW_SARCASM_FALLBACK=1 only if you accept degraded accuracy."
                ) from last_error

    def _configure_cpu_runtime(self):
        """Tune PyTorch CPU threading for better throughput on laptop CPUs."""
        cpu_count = os.cpu_count() or 4
        default_threads = max(2, min(8, cpu_count - 2))
        env_threads = os.getenv("SARCASM_CPU_THREADS", "")

        try:
            threads = int(env_threads) if env_threads else default_threads
            threads = max(1, threads)
            torch.set_num_threads(threads)
            logger.info("Sarcasm CPU threads set to %d", threads)
        except ValueError:
            logger.warning("Invalid SARCASM_CPU_THREADS='%s'. Using PyTorch defaults.", env_threads)

    def _maybe_quantize_model(self):
        """Apply dynamic int8 quantization on CPU to speed up transformer linear layers."""
        quantize = os.getenv("SARCASM_CPU_QUANTIZE", "1") == "1"
        if not quantize:
            return

        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            logger.info("Enabled dynamic quantization for sarcasm model (CPU)")
        except Exception as exc:
            logger.warning("Could not quantize sarcasm model, continuing without quantization: %s", exc)

    def _effective_batch_size(self) -> int:
        """Choose a conservative default batch size by device, overridable via env."""
        env_bs = os.getenv("SARCASM_BATCH_SIZE")
        if env_bs:
            try:
                parsed = int(env_bs)
                if parsed > 0:
                    return parsed
            except ValueError:
                logger.warning("Invalid SARCASM_BATCH_SIZE='%s'. Using defaults.", env_bs)

        return 128 if self.device.type == "cuda" else 96

    def _live_source_mask(self) -> pd.Series:
        live_sources = {"reddit", "stocktwits"}
        if "source" not in self.df.columns:
            return pd.Series([True] * len(self.df), index=self.df.index)
        return self.df["source"].astype(str).str.lower().isin(live_sources)

    def _first_existing(self, candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def _normalize_input_schema(self):
        """Map twitter-style kaggle_hf columns to the text/source fields expected by this stage."""
        text_col = self._first_existing(["text", "tweet", "Tweet", "content", "body"])
        if text_col and text_col != "text":
            self.df["text"] = self.df[text_col]
        elif "text" not in self.df.columns:
            self.df["text"] = ""

        source_col = self._first_existing(["source", "platform"])
        if source_col and source_col != "source":
            self.df["source"] = self.df[source_col]
        elif "source" not in self.df.columns:
            self.df["source"] = "unknown"

    def _prepare_text_for_model(self, text: str, source: str) -> str:
        """Light source-aware cleaning before sarcasm inference."""
        cleaned = str(text)
        src = str(source).lower()

        # Twitter-like streams: remove handles and normalize hashtags while keeping semantic terms.
        if src in {"huggingface", "kaggle", "kaggle_india_stocktwits", "stocktwits"}:
            cleaned = re.sub(r"@\w+", " ", cleaned)
            cleaned = re.sub(r"#", "", cleaned)

        cleaned = re.sub(r"http\S+|www\S+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _resolve_positive_index(self):
        id2label = getattr(self.model.config, "id2label", {}) or {}
        if id2label:
            for idx, label in id2label.items():
                label_text = str(label).lower()
                if "sarcas" in label_text or "irony" in label_text:
                    return int(idx)

        # Binary classifiers usually map positive class to index 1.
        if getattr(self.model.config, "num_labels", 2) >= 2:
            return 1

        return 0

    def predict_sarcasm(self, text):
        if self.model_name is None:
            return 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        sarcasm_prob = probs[:, self.label_positive_index].item()
        return sarcasm_prob

    def predict_sarcasm_batch(self, texts, batch_size: int):
        """Batch inference for significantly faster CPU/GPU throughput."""
        if self.model_name is None:
            return [0.0] * len(texts)

        if not texts:
            return []

        scores = []
        iterator = range(0, len(texts), batch_size)
        desc = f"Sarcasm inference ({self.device.type}, bs={batch_size})"

        for start in tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc=desc, unit="batch"):
            batch_texts = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=1)

            batch_scores = probs[:, self.label_positive_index].detach().cpu().tolist()
            scores.extend(float(s) for s in batch_scores)

        return scores

    def run(self):
        batch_size = self._effective_batch_size()
        only_live = os.getenv("SARCASM_ONLY_LIVE", "1") == "1"
        max_samples_env = os.getenv("SARCASM_MAX_SAMPLES", "").strip()
        max_samples = int(max_samples_env) if max_samples_env.isdigit() and int(max_samples_env) > 0 else None

        prepared_texts = [
            self._prepare_text_for_model(row.get("text", ""), row.get("source", "unknown"))
            for _, row in self.df.iterrows()
        ]

        scope_mask = self._live_source_mask() if only_live else pd.Series([True] * len(self.df), index=self.df.index)
        valid_idx = [i for i, txt in enumerate(prepared_texts) if txt and bool(scope_mask.iloc[i])]

        if max_samples is not None and len(valid_idx) > max_samples:
            # Deterministic cap so runs are reproducible and bounded on CPU.
            valid_idx = valid_idx[:max_samples]

        valid_texts = [prepared_texts[i] for i in valid_idx]

        logger.info(
            "Sarcasm inference scope: %d/%d rows (only_live=%s, max_samples=%s, device=%s, batch_size=%d)",
            len(valid_texts),
            len(self.df),
            only_live,
            max_samples if max_samples is not None else "none",
            self.device.type,
            batch_size,
        )

        sarcasm_scores = [0.0] * len(self.df)
        if valid_texts:
            valid_scores = self.predict_sarcasm_batch(valid_texts, batch_size=batch_size)
            for i, score in zip(valid_idx, valid_scores):
                sarcasm_scores[i] = score

        self.df['sarcasm_score'] = sarcasm_scores

        # Apply logic thresholds
        self.df['flip_flag'] = self.df['sarcasm_score'] > 0.65
        self.df['uncertain_flag'] = (
            (self.df['sarcasm_score'] >= 0.4) &
            (self.df['sarcasm_score'] <= 0.65)
        )

        return self.df
