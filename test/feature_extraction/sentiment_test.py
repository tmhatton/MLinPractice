import unittest
import pandas as pd
from code.feature_extraction.sentiment import Sentiment


class SentimentTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = 'input'
        self.extractor = Sentiment(self.INPUT_COLUMN)

    def test_sentiment_polarity(self):
        """
        Tests whether the polarity is calculated like expected. A negative sentence should have a value below zero,
        a neutral sentence should have a value of zero, and a positive sentence should have a value above zero.
        """
        inputs = ['This text is negative.', 'This text is neutral.', 'This text is positive.']
        input_df = pd.DataFrame(inputs, columns=[self.INPUT_COLUMN])

        sentiments = self.extractor.fit_transform(input_df)

        # assert the general output and ranges
        self.assertEqual(sentiments.shape, (3, 2), msg='Expected two outcomes (polarity and subjectivity) for each input.')
        self.assertGreaterEqual(sentiments[:, 0].min(), -1, msg='Minimal expected polarity value is too low.')
        self.assertLessEqual(sentiments[:, 0].max(), 1, msg='Maximal expected polarity value is too high.')

        # assert the polarities of the sentences
        self.assertLess(sentiments[0, 0], 0, msg='Polarity of the first sentence should be below zero.')
        self.assertEqual(sentiments[1, 0], 0, msg='Polarity of the second sentence should be zero.')
        self.assertGreater(sentiments[2, 0], 0, msg='Polarity of the third sentence should be above zero.')

    def test_sentiment_subjectivity(self):
        """
        Tests whether the subjectivity is calculated like expected. A neutral formulated sentence should have a value
        of around zero and a not neutral formulated sentence should have a value of higher than 0.2.
        """
        inputs = ['The weather is sunny.', 'I hate the weather at the moment.']
        input_df = pd.DataFrame(inputs, columns=[self.INPUT_COLUMN])

        sentiments = self.extractor.fit_transform(input_df)

        # assert the general output and ranges
        self.assertEqual(sentiments.shape, (2, 2), msg='Expected two outcomes (polarity and subjectivity) for each input.')
        self.assertGreaterEqual(sentiments[:, 1].min(), 0, msg='Minimal expected polarity value is too low.')
        self.assertLessEqual(sentiments[:, 1].max(), 1, msg='Maximal expected polarity value is too high.')

        # assert the subjectivities of the sentences
        self.assertEqual(sentiments[0, 1], 0, msg='Subjectivity of the first sentence should be zero.')
        self.assertGreaterEqual(sentiments[1, 1], 0.2, msg='Subjectivity of the second sentence should be greater or equal to 0.2.')


if __name__ == '__main__':
    unittest.main()
