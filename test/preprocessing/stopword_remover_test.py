import unittest
import pandas as pd
from code.preprocessing.stopword_remover import StopwordRemover
from code.util import COLUMN_STOPWORDS


class StopwordRemoverTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = COLUMN_STOPWORDS

    def test_stopword_remover_tweet(self):
        remover = StopwordRemover(input_column=self.INPUT_COLUMN, use_tokens=False)
        input_text = "This is an example sentence"
        output_text = ['example', 'sentence']

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]

        removed = remover.fit_transform(input_df)
        self.assertEqual(output_text, removed[self.OUTPUT_COLUMN][0])

    def test_stopword_remover_tweet_tokenized(self):
        remover = StopwordRemover(input_column=self.INPUT_COLUMN, use_tokens=True)
        input_text = '["This", "is", "an", "example", "sentence"]'
        output_text = ['example', 'sentence']

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]

        removed = remover.fit_transform(input_df)
        self.assertEqual(output_text, removed[self.OUTPUT_COLUMN][0])


if __name__ == '__main__':
    unittest.main()
