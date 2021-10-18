"""
Token length feature extractor:
Extracts the length of tokens for each tweet
"""
import ast
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import SUFFIX_TOKEN_LENGTH


# class for extracting the character-based length as a feature
class TokenLength(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], input_column + SUFFIX_TOKEN_LENGTH)

    # don't need to fit, so don't overwrite _set_variables()

    # compute the word length based on the inputs
    def _get_values(self, inputs):
        nums_of_tokens = []

        for tweet in inputs[0]:
            token_list = ast.literal_eval(tweet)
            nums_of_tokens.append(len(token_list))

        nums_of_tokens = np.array(nums_of_tokens).reshape(-1, 1)

        return nums_of_tokens
        # token_lengths = []
        #
        # for tweet in inputs[0]:
        #     token_lengths.append(len(tweet.split()))
        #
        # token_lengths = np.array(token_lengths).reshape(-1, 1)
        #
        # return token_lengths
