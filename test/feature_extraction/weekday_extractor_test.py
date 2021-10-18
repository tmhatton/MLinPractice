import unittest
import pandas as pd

from code.feature_extraction.weekday_extractor import WeekdayExtractor
from code.util import COLUMN_DATE


class WeekdayExtractorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_DATE
        self.extractor = WeekdayExtractor(self.INPUT_COLUMN)

    def test_weekday_extraction(self):
        input = ['2021-10-16', '2021-10-15', '2021-10-08']
        input_df = pd.DataFrame()
        input_df[COLUMN_DATE] = input

        # Expected columns would be weekday_Fri and weekday_Sat, thus only two one-hot-encoded values
        expected_output = [[0, 1], [1, 0], [1, 0]]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output.tolist())


if __name__ == '__main__':
    unittest.main()
