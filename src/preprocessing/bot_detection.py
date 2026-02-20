import pandas as pd
import re
import hashlib


class BotDetection:

    def __init__(self, df):
        self.df = df.copy()

    def _resolve_time_column(self):
        if "created_utc" in self.df.columns:
            return "created_utc"
        if "timestamp_utc" in self.df.columns:
            return "timestamp_utc"
        raise KeyError("Missing required time column: expected 'created_utc' or 'timestamp_utc'.")

    def frequency_filter(self):
        time_col = self._resolve_time_column()

        if time_col == "created_utc":
            self.df[time_col] = pd.to_datetime(self.df[time_col], unit='s', errors='coerce')
        else:
            self.df[time_col] = pd.to_datetime(self.df[time_col], errors='coerce', utc=True)

        self.df = self.df.dropna(subset=[time_col])
        self.df = self.df.sort_values(['author', time_col])

        self.df['post_count_last_hour'] = (
            self.df.groupby('author')
            .rolling('1H', on=time_col)
            .count()['text']
            .reset_index(level=0, drop=True)
        )

        self.df['freq_bot'] = self.df['post_count_last_hour'] > 15

    def username_filter(self):

        pattern1 = r'^[a-zA-Z]{2,}\d{4,}$'
        pattern2 = r'^[a-zA-Z0-9]{10,}$'

        def is_suspicious(username):
            if pd.isna(username):
                return True

            if re.match(pattern1, username):
                return True

            if re.match(pattern2, username):
                return True

            digit_ratio = sum(c.isdigit() for c in username) / len(username)
            if digit_ratio > 0.5:
                return True

            return False

        self.df['username_bot'] = self.df['author'].apply(is_suspicious)

    def duplicate_filter(self):

        self.df['text_normalized'] = (
            self.df['text']
            .str.lower()
            .str.replace(r'\W+', '', regex=True)
        )

        self.df['text_hash'] = self.df['text_normalized'].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )

        time_col = self._resolve_time_column()
        self.df = self.df.sort_values(time_col)

        self.df['duplicate_bot'] = self.df.duplicated(
            subset=['text_hash'],
            keep=False
        )

    def run(self):

        self.frequency_filter()
        self.username_filter()
        self.duplicate_filter()

        self.df['is_bot'] = (
            self.df['freq_bot'] |
            self.df['username_bot'] |
            self.df['duplicate_bot']
        )

        return self.df[~self.df['is_bot']]
