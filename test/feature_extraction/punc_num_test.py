import unittest
import pandas as pd
from code.util import COLUMN_TWEET
from code.feature_extraction.punc_num import PunctuationNum

class PunctuationNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUM = COLUMN_TWEET
        self.extractor = PunctuationNum(self.INPUT_COLUM)

    def test_character_length(self):
        input_text = "Hallo ... das ist ein Text mit 40 ZEICHEN und 8 Satzzeichen!!!!?"
        output = [[8]]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUM] = [input_text]
        punc_num = self.extractor.fit_transform(input_df)
        print(self.extractor.fit_transform(input_df))
        self.assertEqual(punc_num, output)  # add assertion here


if __name__ == '__main__':
    unittest.main()
