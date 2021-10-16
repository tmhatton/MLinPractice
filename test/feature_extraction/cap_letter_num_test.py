import unittest
import pandas as pd
from code.util import COLUMN_TWEET
from code.feature_extraction.cap_letter_num import CapLettersNum

class CapLetterNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUM = COLUMN_TWEET
        self.extractor = CapLettersNum(self.INPUT_COLUM)

    def test_character_length(self):
        input_text = "Hallo, das ist ein Text mit 40 ZEICHEN."
        output = [[9]]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUM] = [input_text]
        char_length = self.extractor.fit_transform(input_df)
        print(self.extractor.fit_transform(input_df))
        self.assertEqual(char_length, output)  # add assertion here


if __name__ == '__main__':
    unittest.main()
