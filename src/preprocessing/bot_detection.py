import pandas as pd
import re
import hashlib
import logging

logger = logging.getLogger(__name__)


class BotDetection:

    def __init__(self, df):
        self.df = df.copy()
        self.removed_freq = 0
        self.removed_username = 0
        self.removed_duplicate = 0
        self._normalize_input_schema()

    def _first_existing(self, candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def _normalize_input_schema(self):
        """Map twitter-style kaggle_hf columns into the expected preprocessing schema."""
        text_col = self._first_existing(["text", "tweet", "Tweet", "content", "body"])
        if text_col and text_col != "text":
            self.df["text"] = self.df[text_col]
        elif "text" not in self.df.columns:
            self.df["text"] = ""

        author_col = self._first_existing(["author", "user", "username", "screen_name", "handle"])
        if author_col and author_col != "author":
            self.df["author"] = self.df[author_col]
        elif "author" not in self.df.columns:
            self.df["author"] = "[unknown]"

        source_col = self._first_existing(["source", "platform"])
        if source_col and source_col != "source":
            self.df["source"] = self.df[source_col]
        elif "source" not in self.df.columns:
            self.df["source"] = "unknown"

        if "timestamp_utc" not in self.df.columns:
            ts_col = self._first_existing(["created_utc", "timestamp", "created_at", "datetime", "date", "Date"])
            if ts_col:
                self.df["timestamp_utc"] = self.df[ts_col]

    def _is_live_social_source(self) -> pd.Series:
        """
        Strict bot heuristics should apply only to live social feeds.
        Kaggle/HF/static datasets are pre-curated and should be treated leniently.
        """
        if "source" not in self.df.columns:
            return pd.Series([True] * len(self.df), index=self.df.index)

        live_sources = {"reddit", "stocktwits"}
        return self.df["source"].astype(str).str.lower().isin(live_sources)

    def _is_placeholder_author(self) -> pd.Series:
        if "author" not in self.df.columns:
            return pd.Series([True] * len(self.df), index=self.df.index)

        author = self.df["author"].astype(str).str.strip().str.lower()
        placeholders = {"", "nan", "none", "null", "[unknown]", "unknown"}
        return author.isin(placeholders)

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

        live_mask = self._is_live_social_source() & (~self._is_placeholder_author())
        self.df['post_count_last_hour'] = 0
        self.df['freq_bot'] = False

        if live_mask.any():
            live_df = self.df.loc[live_mask, ['author', time_col, 'text']].copy()
            live_df['post_count_last_hour'] = (
                live_df.groupby('author')
                .rolling('1H', on=time_col)
                .count()['text']
                .reset_index(level=0, drop=True)
            )

            self.df.loc[live_df.index, 'post_count_last_hour'] = live_df['post_count_last_hour']
            # Moderately stricter: flag sustained high-frequency posting bursts.
            self.df.loc[live_df.index, 'freq_bot'] = live_df['post_count_last_hour'] > 120

        self.removed_freq = self.df['freq_bot'].sum()

    def username_filter(self):

        pattern1 = r'^[a-zA-Z]{2,}\d{4,}$'
        pattern2 = r'^[a-zA-Z0-9]{16,}$'

        live_mask = self._is_live_social_source()

        def is_suspicious(username):
            if pd.isna(username):
                return False  # Don't filter out missing usernames

            username = str(username).strip()
            if not username or username.lower() in {'[unknown]', 'unknown', 'none', 'null', 'nan'}:
                return False

            if re.match(pattern1, username):
                return True

            if re.match(pattern2, username):
                return True

            digit_ratio = sum(c.isdigit() for c in username) / len(username)
            if digit_ratio > 0.9 and len(username) >= 8:
                return True

            return False

        self.df['username_bot'] = False
        self.df.loc[live_mask, 'username_bot'] = self.df.loc[live_mask, 'author'].apply(is_suspicious)
        self.removed_username = self.df['username_bot'].sum()

    def duplicate_filter(self):
        # Only evaluate duplicates in live social streams.
        # Static datasets can legitimately contain repeated phrasing.
        
        self.df['text_normalized'] = (
            self.df['text'].fillna('').astype(str)
            .str.lower()
            .str.replace(r'\W+', '', regex=True)
        )

        self.df['text_hash'] = self.df['text_normalized'].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )

        time_col = self._resolve_time_column()
        self.df = self.df.sort_values(time_col)

        live_mask = self._is_live_social_source() & (~self._is_placeholder_author())
        self.df['duplicate_bot'] = False

        if live_mask.any():
            group_cols = ['source', 'author', 'text_hash'] if 'source' in self.df.columns else ['author', 'text_hash']
            grp = self.df.loc[live_mask].groupby(group_cols)[time_col]
            counts = grp.transform('size')
            span = (grp.transform('max') - grp.transform('min')).dt.total_seconds()

            # Slightly stricter repost burst rule for live feeds.
            self.df.loc[live_mask, 'duplicate_bot'] = (counts >= 4) & (span <= 2700)

        self.removed_duplicate = self.df['duplicate_bot'].sum()

    def run(self):

        self.frequency_filter()
        self.username_filter()
        self.duplicate_filter()

        self.df['is_bot'] = (
            self.df['freq_bot'] |
            self.df['username_bot'] |
            self.df['duplicate_bot']
        )

        total_removed = self.df['is_bot'].sum()
        logger.info(f"Bot detection breakdown: freq={self.removed_freq}, username={self.removed_username}, duplicate={self.removed_duplicate}, total_removed={total_removed}")
        
        return self.df[~self.df['is_bot']]
