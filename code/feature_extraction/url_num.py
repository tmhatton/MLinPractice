"""
Extracts the number of URLs from the tweet
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_TWEET, COLUMN_URLS, SUFFIX_URL_NUM


class URLsNum(FeatureExtractor):
    """
    Class to extract the number of URLs from a tweet
    """

    def __init__(self, input_column=COLUMN_URLS):
        super().__init__([input_column], COLUMN_TWEET + SUFFIX_URL_NUM)

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        nums_of_urls = []

        for tweet in inputs[0]:
            url_list = ast.literal_eval(tweet)
            nums_of_urls.append(len(url_list))

        nums_of_mentions = np.array(nums_of_urls).reshape(-1, 1)

        return nums_of_mentions

