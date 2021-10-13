import unittest
import pandas as pd

from code.feature_extraction.hashtag_num import HashtagNum
from code.util import COLUMN_HASHTAGS


class HashtagNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_HASHTAGS
        self.extractor = HashtagNum(self.INPUT_COLUMN)

    def test_hashtag_num(self):
        input = '''['hashtag', 'yolo', 'data']'''
        input_df = pd.DataFrame([COLUMN_HASHTAGS])
        input_df[COLUMN_HASHTAGS] = [input]

        expected_output = [3]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
