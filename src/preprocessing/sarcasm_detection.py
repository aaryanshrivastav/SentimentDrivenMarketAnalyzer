import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import logging
import os


logger = logging.getLogger("sarcasm_detection")


class SarcasmDetection:

    def __init__(self, df, model_name=None):
        self.df = df.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def run(self):

        sarcasm_scores = []

        for text in self.df['text']:
            score = self.predict_sarcasm(str(text))
            sarcasm_scores.append(score)

        self.df['sarcasm_score'] = sarcasm_scores

        # Apply logic thresholds
        self.df['flip_flag'] = self.df['sarcasm_score'] > 0.65
        self.df['uncertain_flag'] = (
            (self.df['sarcasm_score'] >= 0.4) &
            (self.df['sarcasm_score'] <= 0.65)
        )

        return self.df
