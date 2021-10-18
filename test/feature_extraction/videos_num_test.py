import unittest
import pandas as pd
from code.feature_extraction.videos_num import VideosNum
from code.util import COLUMN_VIDEOS


class PhotosNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_VIDEOS
        self.extractor = VideosNum(self.INPUT_COLUMN)

    def test_photos_num(self):
        input = '''[0]'''
        input_df = pd.DataFrame([COLUMN_VIDEOS])
        input_df[COLUMN_VIDEOS] = [input]

        expected_output = [0]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
