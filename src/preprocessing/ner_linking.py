import pandas as pd
import re
import spacy
import os
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class NERLinking:

    def __init__(self, df, ticker_path="data/reference/sp500_tickers.csv"):
        self.df = df.copy()
        self.ticker_map = {}
        self.valid_tickers = set()
        self.alias_map = {
            "tesla": "TSLA",
            "gamestop": "GME",
            "apple": "AAPL",
            "nvidia": "NVDA",
            "amazon": "AMZN",
            "reliance": "RELIANCE_NS",
            "tcs": "TCS_NS",
            "infosys": "INFY_NS",
            "hdfc bank": "HDFCBANK_NS",
            "icici bank": "ICICIBANK_NS",
            "wipro": "WIPRO_NS",
            "state bank of india": "SBIN_NS",
            "sbi": "SBIN_NS",
            "tata motors": "TATAMOTORS_NS",
            "adani enterprises": "ADANIENT_NS",
            "bajaj finance": "BAJFINANCE_NS",
        }

        if os.path.exists(ticker_path):
            self.ticker_df = pd.read_csv(ticker_path)
            self.ticker_df['company'] = self.ticker_df['company'].str.lower()
            self.ticker_map = dict(
                zip(self.ticker_df['company'], self.ticker_df['ticker'])
            )
            self.valid_tickers = set(self.ticker_df['ticker'])
        else:
            self.valid_tickers = {
                "TSLA", "GME", "AAPL", "NVDA", "AMZN",
                "RELIANCE_NS", "TCS_NS", "INFY_NS", "HDFCBANK_NS", "ICICIBANK_NS",
                "WIPRO_NS", "SBIN_NS", "TATAMOTORS_NS", "ADANIENT_NS", "BAJFINANCE_NS",
            }

        self.enable_spacy = os.getenv("NER_ENABLE_SPACY", "1") == "1"
        self.spacy_batch_size = max(16, int(os.getenv("NER_SPACY_BATCH_SIZE", "256")))
        self.max_spacy_rows = None
        max_rows_env = os.getenv("NER_MAX_SPACY_ROWS", "").strip()
        if max_rows_env.isdigit() and int(max_rows_env) > 0:
            self.max_spacy_rows = int(max_rows_env)

        self.nlp = None
        if self.enable_spacy:
            # Load only what is needed for ORG extraction.
            self.nlp = spacy.load(
                "en_core_web_sm",
                disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
            )

    def _normalize_ticker(self, raw):
        if raw is None:
            return None
        t = str(raw).strip().upper()
        if not t:
            return None
        t = t.replace(".NS", "_NS")
        return t

    def _extract_plain_tickers(self, text):
        # Match plain symbols without $ prefix: AAPL, NVDA, RELIANCE, TCS.
        raw = re.findall(r'\b[A-Z]{2,10}(?:_NS)?\b', text.upper())
        mapped = []
        for token in raw:
            norm = self._normalize_ticker(token)
            # Map bare Indian symbols to _NS form.
            if norm in {"RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "WIPRO", "SBIN", "TATAMOTORS", "ADANIENT", "BAJFINANCE"}:
                norm = f"{norm}_NS"
            mapped.append(norm)
        return mapped

    def extract_dollar_tickers(self, text):
        matches = re.findall(r'\$([A-Z]{1,5})', text)
        return [self._normalize_ticker(t) for t in matches]

    def extract_company_names(self, text, doc=None):
        if doc is None and self.nlp is not None:
            doc = self.nlp(text)

        companies = []

        if doc is not None:
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    name = ent.text.lower()
                    if name in self.ticker_map:
                        companies.append(self.ticker_map[name])

        lowered = text.lower()
        for alias, ticker in self.alias_map.items():
            if alias in lowered:
                companies.append(ticker)

        return companies

    def _extract_hint_tickers(self, row):
        tickers_hint = []
        for col in ["ticker", "post_flair", "symbol"]:
            if col in row and not pd.isna(row[col]):
                hint = self._normalize_ticker(row[col])
                if hint:
                    tickers_hint.append(hint)
        return tickers_hint

    def _filter_valid(self, tickers):
        dedup = list(dict.fromkeys([t for t in tickers if t]))
        if self.valid_tickers:
            dedup = [t for t in dedup if t in self.valid_tickers]
        return dedup

    def run(self):
        if self.df.empty:
            return self.df

        new_rows = []
        base_tickers = {}
        need_spacy_idx = []

        # Fast path: regex + existing hints (much cheaper than spaCy).
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="NER fast-path", unit="row"):
            text = str(row.get("text", ""))

            tickers_dollar = self.extract_dollar_tickers(text)
            tickers_plain = self._extract_plain_tickers(text)
            tickers_hint = self._extract_hint_tickers(row)

            all_tickers = self._filter_valid(tickers_dollar + tickers_plain + tickers_hint)
            if all_tickers:
                base_tickers[idx] = all_tickers
            else:
                need_spacy_idx.append(idx)

        logger.info(
            "NER fast-path resolved %d/%d rows; spaCy needed for %d rows",
            len(base_tickers),
            len(self.df),
            len(need_spacy_idx),
        )

        # Slow path: spaCy ORG only for unresolved rows.
        if self.enable_spacy and self.nlp is not None and need_spacy_idx:
            spacy_indices = need_spacy_idx
            if self.max_spacy_rows is not None and len(spacy_indices) > self.max_spacy_rows:
                spacy_indices = spacy_indices[: self.max_spacy_rows]
                logger.info("NER spaCy capped to %d rows via NER_MAX_SPACY_ROWS", len(spacy_indices))

            spacy_texts = [str(self.df.at[i, "text"]) for i in spacy_indices]
            docs = self.nlp.pipe(spacy_texts, batch_size=self.spacy_batch_size)

            for i, doc in tqdm(
                zip(spacy_indices, docs),
                total=len(spacy_indices),
                desc=f"NER spaCy (bs={self.spacy_batch_size})",
                unit="row",
            ):
                text = str(self.df.at[i, "text"])
                tickers_org = self.extract_company_names(text, doc=doc)
                tickers = self._filter_valid(tickers_org)
                if tickers:
                    base_tickers[i] = tickers

        # Expand rows by detected tickers.
        for idx, tickers in tqdm(base_tickers.items(), total=len(base_tickers), desc="NER expand", unit="row"):
            row = self.df.loc[idx]
            for ticker in tickers:
                new_row = row.copy()
                new_row["ticker"] = ticker
                new_rows.append(new_row)

        logger.info("NER output rows: %d from %d input rows", len(new_rows), len(self.df))
        return pd.DataFrame(new_rows)
