import unittest
import pandas as pd

from code.feature_extraction.url_num import URLsNum
from code.util import COLUMN_URLS


class URLsNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_URLS
        self.extractor = URLsNum(self.INPUT_COLUMN)

    def test_url_num(self):
        input = '''['www.google.com', 'www.apple.com', 'www.uos.de', 'www.example.com']'''
        input_df = pd.DataFrame([COLUMN_URLS])
        input_df[COLUMN_URLS] = [input]

        expected_output = [4]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
