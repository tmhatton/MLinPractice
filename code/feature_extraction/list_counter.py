"""
Extracts features that are based on a list and just have to be counted.
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_TWEET, COLUMN_URLS, SUFFIX_URL_NUM, COLUMN_PHOTOS, SUFFIX_PHOTOS_NUM, COLUMN_HASHTAGS, \
    SUFFIX_HASHTAG_NUM, COLUMN_MENTIONS, SUFFIX_MENTION_NUM


class ListCounter(FeatureExtractor):
    """
    Class to count the entries in a list like object.
    """

    def __init__(self, input_column, output_column):
        super().__init__([input_column], output_column)

    def _get_values(self, inputs):
        """Counts the number of values in a list like object."""
        nums_of_entries = []

        for input_as_string in inputs[0]:
            input_as_list = ast.literal_eval(input_as_string)
            nums_of_entries.append(len(input_as_list))

        nums_of_entries = np.array(nums_of_entries).reshape(-1, 1)

        return nums_of_entries


class URLsNum(ListCounter):
    """
    Class to extract the number of URLs from a tweet.
    """

    def __init__(self, input_column=COLUMN_URLS):
        super().__init__(input_column, COLUMN_TWEET + SUFFIX_URL_NUM)


class PhotosNum(ListCounter):
    """
    Class to extract the number of photos from a tweet.
    """

    def __init__(self, input_column=COLUMN_PHOTOS):
        super().__init__(input_column, COLUMN_PHOTOS + SUFFIX_PHOTOS_NUM)


class HashtagNum(ListCounter):
    """
    Class to extract the number of hashtags from a tweet.
    """

    def __init__(self, input_column=COLUMN_HASHTAGS):
        super().__init__(input_column, COLUMN_TWEET + SUFFIX_HASHTAG_NUM)


class MentionNum(ListCounter):
    """
    Class to extract the number of mentions from a tweet
    """

    def __init__(self, input_column=COLUMN_MENTIONS):
        super().__init__(input_column, COLUMN_TWEET + SUFFIX_MENTION_NUM)
