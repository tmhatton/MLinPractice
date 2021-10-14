"""
Extracts the number of mentions from the tweet
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_TWEET, COLUMN_MENTIONS, SUFFIX_MENTION_NUM


class MentionNum(FeatureExtractor):
    """
    Class to extract the number of mentions from a tweet
    """

    def __init__(self, input_column=COLUMN_MENTIONS):
        super().__init__([input_column], COLUMN_TWEET + SUFFIX_MENTION_NUM)

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        nums_of_mentions = []

        for tweet in inputs[0]:
            mention_list = ast.literal_eval(tweet)
            nums_of_mentions.append(len(mention_list))

        nums_of_mentions = np.array(nums_of_mentions).reshape(-1, 1)

        return nums_of_mentions

