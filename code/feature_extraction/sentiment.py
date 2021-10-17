"""
Extracts the sentiment of the tweets.
"""

import numpy as np

from textblob import TextBlob

from code.util import COLUMN_TWEET, FEATURE_SENTIMENT
from code.feature_extraction.feature_extractor import FeatureExtractor


class Sentiment(FeatureExtractor):
    """
    Class to extract the sentiment of a tweet.
    """

    def __init__(self, input_column=COLUMN_TWEET):
        super().__init__([input_column], FEATURE_SENTIMENT)

    def _get_values(self, inputs):
        sentiments = []

        for tweet in inputs[0]:
            sentiments.append(self._get_sentiment_values(tweet))

        return np.array(sentiments)

    def _get_sentiment_values(self, text):
        """
        Calculates the sentiment for the given text and returns it.

        :returns A tuple containing the polarity and the subjectivity of the text.
        """
        sentiment = TextBlob(text).sentiment
        return (sentiment.polarity, sentiment.subjectivity)

