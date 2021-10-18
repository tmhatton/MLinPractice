"""
Extracts the number of attached videos from the tweet
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_VIDEOS, SUFFIX_VIDEOS_NUM


class VideosNum(FeatureExtractor):
    """
    Class to extracts the number of attached videos from the tweet
    """

    def __init__(self, input_column=COLUMN_VIDEOS):
        super().__init__([input_column], COLUMN_VIDEOS + SUFFIX_VIDEOS_NUM)

    # don't need to fit, so don't overwrite _set_variables()

    # get the number of attached videos from the tweet
    def _get_values(self, inputs):
        nums_of_videos = []

        for videos in inputs[0]:
            nums_of_videos.append(videos)

        nums_of_videos = np.array(nums_of_videos).reshape(-1, 1)

        return nums_of_videos


