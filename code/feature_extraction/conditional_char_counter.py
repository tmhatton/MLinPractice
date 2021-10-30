"""
Extracts features that are based on characters and a condition if the characters should be counted.
"""
import string
import numpy as np

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import SUFFIX_PUNC_NUM


class ConditionalCharCounter(FeatureExtractor):
    """
    Class to count the characters in a text that fulfill a certain condition.
    """

    def __init__(self, input_column, output_column):
        super().__init__([input_column], output_column)

    def _get_values(self, inputs):
        """Counts the characters in a text that fulfill a certain condition."""
        num_of_chars = []

        for text in inputs[0]:
            number = 0
            for character in text:
                if self._check_condition(character):
                    number = number + 1
            num_of_chars.append(number)

        num_of_chars = np.array(num_of_chars).reshape(-1, 1)

        return num_of_chars

    def _check_condition(self, char: str):
        """Checks if the condition for counting the character is met."""
        return True


class PunctuationNum(ConditionalCharCounter):
    """
    Class to count the number of punctuation characters in an input.
    """

    def __init__(self, input_column):
        super().__init__(input_column, input_column + SUFFIX_PUNC_NUM)

    def _check_condition(self, char: str):
        return char in string.punctuation
