"""
Weekday feature extractor:
Extracts the weekday of the posting date for each tweet
"""
import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_DATE, ISO_WEEKDAYS


# class for extracting the character-based length as a feature
class WeekdayExtractor(FeatureExtractor):

    # constructor
    def __init__(self, input_column=COLUMN_DATE):
        super().__init__([input_column], input_column)
        self.encoder = OneHotEncoder()
        self.feature_names = None

    def get_feature_name(self):
        return self.feature_names.tolist()

    # don't need to fit, so don't overwrite _set_variables()

    # compute the word length based on the inputs
    def _get_values(self, inputs):
        # Get weekdays from tweets
        weekdays = []
        for tweet in inputs[0]:
            tweet_date = datetime.datetime.strptime(tweet, "%Y-%m-%d")
            day_int = tweet_date.isoweekday()
            weekdays.append(ISO_WEEKDAYS[day_int])
        weekdays = np.array(weekdays)

        # One-Hot encode the extracted weekday
        self.encoder.fit(weekdays.reshape(-1, 1))
        encoded = self.encoder.transform(weekdays.reshape(-1, 1)).toarray()
        self.feature_names = self.encoder.get_feature_names(['weekday'])

        return encoded
