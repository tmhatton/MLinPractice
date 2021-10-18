#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of capital letters in the given column.
"""

import numpy as np
import pandas as pd

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import SUFFIX_CAP_LETTERS


# class for extracting the number of capital letters in input
class CapLettersNum(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], input_column + SUFFIX_CAP_LETTERS)

    # don't need to fit, so don't overwrite _set_variables()

    # compute the number of capital letters in input
    def _get_values(self, inputs):
        cap_letter_num = []

        for tweet in inputs[0]:
            number = 0
            for character in tweet:
                if character.isupper():
                    number = number + 1
            cap_letter_num.append(number)

        cap_letter_num = np.array(cap_letter_num).reshape(-1, 1)

        return cap_letter_num

