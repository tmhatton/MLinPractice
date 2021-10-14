import unittest
import pandas as pd

from code.feature_extraction.mention_num import MentionNum
from code.util import COLUMN_MENTIONS


class MentionNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_MENTIONS
        self.extractor = MentionNum(self.INPUT_COLUMN)

    def test_hashtag_num(self):
        input = '''[{'id': '2235729541', 'name': 'dogecoin', 'screen_name': 'dogecoin'}, {'id': '123432342', 'name': 'John Doe', 'screen_name': 'jodoe'}]'''
        input_df = pd.DataFrame([COLUMN_MENTIONS])
        input_df[COLUMN_MENTIONS] = [input]

        expected_output = [2]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
