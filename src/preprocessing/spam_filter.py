import pandas as pd
import re


class SpamFilter:

    def __init__(self, df):
        self.df = df.copy()

    def remove_low_upvotes(self, min_upvotes=5):

        upvote_col = None
        if 'ups' in self.df.columns:
            upvote_col = 'ups'
        elif 'upvotes' in self.df.columns:
            upvote_col = 'upvotes'

        if upvote_col is not None:
            self.df = self.df[self.df[upvote_col].fillna(0) >= min_upvotes]

    def remove_short_posts(self, min_words=10):

        self.df['word_count'] = self.df['text'].apply(
            lambda x: len(str(x).split())
        )

        self.df = self.df[self.df['word_count'] >= min_words]

    def remove_url_only(self):

        def is_url_only(text):
            text = str(text).strip()
            return bool(re.fullmatch(r'https?://\S+', text))

        self.df = self.df[~self.df['text'].apply(is_url_only)]

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

        return self.df
