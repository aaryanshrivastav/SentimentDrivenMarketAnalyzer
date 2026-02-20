import pandas as pd
import numpy as np


class CredibilityScoring:

    def __init__(self, df):
        self.df = df.copy()

    def compute_engagement(self, row):

        source = row['source']

        if source == "reddit":
            if 'upvote_ratio' in row and not pd.isna(row['upvote_ratio']):
                return row['upvote_ratio']
            else:
                ups = row.get('ups')
                if pd.isna(ups):
                    ups = row.get('upvotes', 0)
                return ups / (ups + 1)

        elif source == "stocktwits":
            if 'likes' in row:
                return min(row['likes'] / 50, 1.0)
            else:
                return 0.5

        elif source == "news":
            return 1.0

        return 0.5

    def source_weight(self, source):

        weights = {
            "news": 1.0,
            "stocktwits": 0.7,
            "reddit": 0.5
        }

        return weights.get(source, 0.5)

    def run(self):

        self.df['engagement_score'] = self.df.apply(
            self.compute_engagement,
            axis=1
        )

        self.df['base_source_weight'] = self.df['source'].apply(
            self.source_weight
        )

        self.df['final_weight'] = (
            self.df['engagement_score'] +
            self.df['base_source_weight']
        ) / 2

        # Sarcasm uncertainty penalty
        if 'uncertain_flag' in self.df.columns:
            self.df.loc[self.df['uncertain_flag'] == True, 'final_weight'] *= 0.5

        # Clamp between 0.1 and 1.0
        self.df['final_weight'] = np.clip(self.df['final_weight'], 0.1, 1.0)

        return self.df
