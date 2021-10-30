import unittest
import pandas as pd
from code.util import COLUMN_TWEET
from code.feature_extraction.conditional_char_counter import PunctuationNum, CapLettersNum


class PunctuationNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_TWEET
        self.extractor = PunctuationNum(self.INPUT_COLUMN)

    def test_punc_num(self):
        input_text = "Hallo ... das ist ein Text mit 40 ZEICHEN und 8 Satzzeichen!!!!?"
        output = [[8]]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        punc_num = self.extractor.fit_transform(input_df)

        self.assertEqual(punc_num, output)


class CapLetterNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_TWEET
        self.extractor = CapLettersNum(self.INPUT_COLUMN)

    def test_cap_letter_num(self):
        input_text = "Hallo, das ist ein Text mit 40 ZEICHEN."
        output = [[9]]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        char_length = self.extractor.fit_transform(input_df)
        print(self.extractor.fit_transform(input_df))
        self.assertEqual(char_length, output)


if __name__ == '__main__':
    unittest.main()
