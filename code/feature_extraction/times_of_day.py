"""
Times of day feature extractor:
Extracts the time of day for each row.
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_TIME, FEATURE_TIMES_OF_DAY


class TimesOfDayExtractor(FeatureExtractor):
    """
    Extracts the times of day (i.e. morning, afternoon, evening, and night) as one hot encoded values.
    """

    def __init__(self, input_column: str = COLUMN_TIME):
        super().__init__([input_column], FEATURE_TIMES_OF_DAY)
        self.encoder = OneHotEncoder()

        # specify the max values for each time of day
        self.NIGHT_MAX_VALUE = '05:59:59'
        self.MORNING_MAX_VALUE = '11:59:59'
        self.AFTERNOON_MAX_VALUE = '17:59:59'
        self.EVENING_MAX_VALUE = '23:59:59'

        # specify the times of day category constants
        self.NIGHT = 1
        self.MORNING = 2
        self.AFTERNOON = 3
        self.EVENING = 4

    def _get_values(self, inputs):
        # create the data arrays and the categorical array
        time_arr = np.array(inputs[0])
        times_of_day = np.empty_like(time_arr)

        # fill the categorical array
        times_of_day[time_arr <= self.NIGHT_MAX_VALUE] = self.NIGHT
        times_of_day[(self.NIGHT_MAX_VALUE < time_arr) & (time_arr <= self.MORNING_MAX_VALUE)] = self.MORNING
        times_of_day[(self.MORNING_MAX_VALUE < time_arr) & (time_arr <= self.AFTERNOON_MAX_VALUE)] = self.AFTERNOON
        times_of_day[(self.AFTERNOON_MAX_VALUE < time_arr) & (time_arr <= self.EVENING_MAX_VALUE)] = self.EVENING

        # One-Hot encode the extracted times of day
        times_of_day = times_of_day.reshape(-1, 1)
        return self.encoder.fit_transform(times_of_day).toarray()
