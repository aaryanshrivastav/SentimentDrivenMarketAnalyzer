import pandas as pd
import re
import spacy
import os


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

        self.nlp = spacy.load("en_core_web_sm")

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

    def extract_company_names(self, text):
        doc = self.nlp(text)
        companies = []

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

    def run(self):

        new_rows = []

        for _, row in self.df.iterrows():

            text = str(row.get('text', ''))

            tickers_dollar = self.extract_dollar_tickers(text)
            tickers_plain = self._extract_plain_tickers(text)
            tickers_org = self.extract_company_names(text)
            tickers_hint = []

            for col in ["ticker", "post_flair", "symbol"]:
                if col in row and not pd.isna(row[col]):
                    hint = self._normalize_ticker(row[col])
                    if hint:
                        tickers_hint.append(hint)

            all_tickers = list(set(tickers_dollar + tickers_plain + tickers_org + tickers_hint))
            all_tickers = [t for t in all_tickers if t]
            if self.valid_tickers:
                all_tickers = [t for t in all_tickers if t in self.valid_tickers]

            if not all_tickers:
                continue  # discard if no ticker found

            for ticker in all_tickers:
                new_row = row.copy()
                new_row['ticker'] = ticker
                new_rows.append(new_row)

        return pd.DataFrame(new_rows)
