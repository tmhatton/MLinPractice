import unittest
import pandas as pd
import numpy as np

from code.feature_extraction.times_of_day import TimesOfDayExtractor


class TimesOfDayExtractorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = 'time'
        self.extractor = TimesOfDayExtractor(self.INPUT_COLUMN)

    def test_fit_transform(self):
        """
        Tests whether all times are transformed to their correct one hot encoded category.
        """
        time_data = ['00:00:00', '03:23:37', '05:59:59',
                     '06:00:00', '07:21:10', '11:59:59',
                     '12:00:00', '17:02:01', '17:59:59',
                     '18:00:00', '19:47:58', '23:59:59']
        df = pd.DataFrame(time_data, columns=[self.INPUT_COLUMN])

        times_of_day_features = self.extractor.fit_transform(df)

        # assert the general output and ranges
        self.assertEqual(times_of_day_features.shape, (12, 4), msg='Expected four outcomes for each input.')
        self.assertEqual(times_of_day_features.min(), 0, msg='Minimal value is not zero.')
        self.assertEqual(times_of_day_features.max(), 1, msg='Maximal value is not one.')

        # assert that the correct number of ones and zeros are present
        unique, counts = np.unique(times_of_day_features, return_counts=True)
        self.assertEqual(dict(zip(unique, counts)), {0: 36, 1: 12}, 'Incorrect number of ones and zeros.')

        # assert each line
        self.assertTrue((times_of_day_features[0] == np.array([1, 0, 0, 0])).all(), '00:00:00 is not labeled as first category')
        self.assertTrue((times_of_day_features[1] == np.array([1, 0, 0, 0])).all(), '03:23:37 is not labeled as first category')
        self.assertTrue((times_of_day_features[2] == np.array([1, 0, 0, 0])).all(), '05:59:59 is not labeled as first category')
        self.assertTrue((times_of_day_features[3] == np.array([0, 1, 0, 0])).all(), '06:00:00 is not labeled as second category')
        self.assertTrue((times_of_day_features[4] == np.array([0, 1, 0, 0])).all(), '07:21:10 is not labeled as second category')
        self.assertTrue((times_of_day_features[5] == np.array([0, 1, 0, 0])).all(), '11:59:59 is not labeled as second category')
        self.assertTrue((times_of_day_features[6] == np.array([0, 0, 1, 0])).all(), '12:00:00 is not labeled as third category')
        self.assertTrue((times_of_day_features[7] == np.array([0, 0, 1, 0])).all(), '17:02:01 is not labeled as third category')
        self.assertTrue((times_of_day_features[8] == np.array([0, 0, 1, 0])).all(), '17:59:59 is not labeled as third category')
        self.assertTrue((times_of_day_features[9] == np.array([0, 0, 0, 1])).all(), '18:00:00 is not labeled as fourth category')
        self.assertTrue((times_of_day_features[10] == np.array([0, 0, 0, 1])).all(), '19:47:58 is not labeled as fourth category')
        self.assertTrue((times_of_day_features[11] == np.array([0, 0, 0, 1])).all(), '23:59:59 is not labeled as fourth category')


if __name__ == '__main__':
    unittest.main()
