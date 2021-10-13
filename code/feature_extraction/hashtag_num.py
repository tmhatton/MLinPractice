"""
Extracts the number of hashtags from the tweet
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_TWEET, COLUMN_HASHTAGS, SUFFIX_HASHTAG_NUM


class HashtagNum(FeatureExtractor):
    """
    Class to extract the number of hashtags from a tweet
    """

    def __init__(self, input_column=COLUMN_HASHTAGS):
        super().__init__([input_column], COLUMN_TWEET + SUFFIX_HASHTAG_NUM)

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        nums_of_hashtags = []

        for tweet in inputs[0]:
            hashtag_list = ast.literal_eval(tweet)
            nums_of_hashtags.append(len(hashtag_list))

        nums_of_hashtags = np.array(nums_of_hashtags).reshape(-1, 1)

        return nums_of_hashtags

