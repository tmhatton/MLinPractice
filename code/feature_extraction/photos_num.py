"""
Extracts the number of photos from the tweet
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_PHOTOS, SUFFIX_PHOTOS_NUM


class PhotosNum(FeatureExtractor):
    """
    Class to extract the number of photos from a tweet
    """

    def __init__(self, input_column=COLUMN_PHOTOS):
        super().__init__([input_column], COLUMN_PHOTOS + SUFFIX_PHOTOS_NUM)

    # don't need to fit, so don't overwrite _set_variables()

    # get the number of photos in the tweet
    def _get_values(self, inputs):
        nums_of_photos = []

        for tweet in inputs[0]:
            photo_list = ast.literal_eval(tweet)
            nums_of_photos.append(len(photo_list))

        nums_of_photos = np.array(nums_of_photos).reshape(-1, 1)

        return nums_of_photos


