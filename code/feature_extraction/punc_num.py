#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of punctuation characters in the given column.
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import SUFFIX_PUNC_NUM
from string import punctuation


# class for extracting the number of punctuation characters in input
class PunctuationNum(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], input_column + SUFFIX_PUNC_NUM)

    # don't need to fit, so don't overwrite _set_variables()

    # compute the number of punctuation characters in input
    def _get_values(self, inputs):
        punc_num = []

        for tweet in inputs[0]:
            number = 0
            for character in tweet:
                if character in punctuation:
                    number = number + 1
            punc_num.append(number)

        punc_num = np.array(punc_num).reshape(-1, 1)
        return punc_num
