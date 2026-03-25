import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


class SpamFilter:

    def __init__(self, df):
        self.df = df.copy()
        self.removed_low_upvotes = 0
        self.removed_short_posts = 0
        self.removed_url_only = 0
        self._normalize_input_schema()

    def _first_existing(self, candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def _normalize_input_schema(self):
        """Map twitter-like kaggle_hf columns to preprocessing schema."""
        text_col = self._first_existing(['text', 'tweet', 'Tweet', 'content', 'body'])
        if text_col and text_col != 'text':
            self.df['text'] = self.df[text_col]
        elif 'text' not in self.df.columns:
            self.df['text'] = ''

        source_col = self._first_existing(['source', 'platform'])
        if source_col and source_col != 'source':
            self.df['source'] = self.df[source_col]
        elif 'source' not in self.df.columns:
            self.df['source'] = 'unknown'

        # For twitter-style datasets, likes/favorites should behave like upvotes.
        if 'upvotes' not in self.df.columns:
            likes_col = self._first_existing(['likes', 'like_count', 'favorite_count', 'favorites'])
            if likes_col:
                self.df['upvotes'] = self.df[likes_col]

    def _is_live_social_source(self) -> pd.Series:
        if 'source' not in self.df.columns:
            return pd.Series([True] * len(self.df), index=self.df.index)

        live_sources = {'reddit', 'stocktwits'}
        return self.df['source'].astype(str).str.lower().isin(live_sources)

    def remove_low_upvotes(self, min_upvotes=6):

        upvote_col = None
        if 'ups' in self.df.columns:
            upvote_col = 'ups'
        elif 'upvotes' in self.df.columns:
            upvote_col = 'upvotes'

        if upvote_col is not None:
            live_mask = self._is_live_social_source()

            # Apply upvote threshold only to live social data where votes are meaningful.
            if live_mask.any():
                social = self.df.loc[live_mask, upvote_col]
                non_null_upvotes = social.notna().sum()
                avg_upvotes = social.fillna(0).mean()

                if non_null_upvotes > len(social) * 0.3 and avg_upvotes > 1:
                    before = len(self.df)
                    keep_mask = (~live_mask) | (self.df[upvote_col].fillna(0) >= min_upvotes)
                    self.df = self.df[keep_mask]
                    self.removed_low_upvotes = before - len(self.df)
                    logger.debug(f"Removed {self.removed_low_upvotes} low-upvote live-social posts")
                else:
                    logger.debug(
                        f"Skipping upvote filter for live-social rows: non_null={non_null_upvotes}/{len(social)}, avg={avg_upvotes:.2f}"
                    )

    def remove_short_posts(self, min_words_live=4, min_words_static=2):

        self.df['word_count'] = self.df['text'].apply(
            lambda x: len(str(x).split())
        )

        before = len(self.df)
        live_mask = self._is_live_social_source()
        keep_mask = (
            (live_mask & (self.df['word_count'] >= min_words_live)) |
            ((~live_mask) & (self.df['word_count'] >= min_words_static))
        )
        self.df = self.df[keep_mask]
        self.removed_short_posts = before - len(self.df)
        logger.debug(
            f"Removed {self.removed_short_posts} short posts (live<{min_words_live}, static<{min_words_static})"
        )

    def remove_url_only(self):

        def is_url_only(text):
            text = str(text).strip()
            return bool(re.fullmatch(r'https?://\S+', text))

        before = len(self.df)
        self.df = self.df[~self.df['text'].apply(is_url_only)]
        self.removed_url_only = before - len(self.df)
        logger.debug(f"Removed {self.removed_url_only} URL-only posts")

    def remove_low_signal_posts(self, min_alpha_chars=6):
        """Drop very low-information rows such as punctuation-only or symbol-heavy fragments."""
        def has_signal(text):
            alpha_chars = re.findall(r"[A-Za-z]", str(text))
            return len(alpha_chars) >= min_alpha_chars

        before = len(self.df)
        self.df = self.df[self.df['text'].apply(has_signal)]
        removed = before - len(self.df)
        if removed > 0:
            logger.debug(f"Removed {removed} low-signal posts")

    def clean_text(self):

        def clean(text):
            text = str(text)

            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)

            # Remove markdown symbols
            text = re.sub(r'\*|\_|`|\~|\>', '', text)

            # Remove ticker symbols like $TSLA
            text = re.sub(r'\$[A-Z]{1,5}', '', text)

            # Remove emojis and non-ascii
            text = text.encode('ascii', 'ignore').decode()

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        self.df['text'] = self.df['text'].apply(clean)

    def run(self):

        self.remove_low_upvotes()
        self.remove_short_posts()
        self.remove_url_only()
        self.clean_text()
        self.remove_low_signal_posts()

        total_removed = self.removed_low_upvotes + self.removed_short_posts + self.removed_url_only
        logger.info(f"Spam filter breakdown: low_upvotes={self.removed_low_upvotes}, short_posts={self.removed_short_posts}, url_only={self.removed_url_only}, total_removed={total_removed}")
        
        return self.df
