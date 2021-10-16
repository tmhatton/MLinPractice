import unittest
import pandas as pd

from code.feature_extraction.token_length import TokenLength


class TokenLengthTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = "input"
        self.extractor = TokenLength(self.INPUT_COLUMN)

    def test_token_length(self):
        input_text = "['This', 'is', 'an', 'example', 'sentence']"
        output = [5]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        token_length = self.extractor.fit_transform(input_df)

        self.assertEqual(output, token_length)


if __name__ == '__main__':
    unittest.main()
