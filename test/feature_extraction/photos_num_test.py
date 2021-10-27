import unittest
import pandas as pd
from code.feature_extraction.photos_num import PhotosNum
from code.util import COLUMN_PHOTOS


class PhotosNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_PHOTOS
        self.extractor = PhotosNum(self.INPUT_COLUMN)

    def test_photos_num(self):
        input = '''['www.hashtag.de/234234.jpg', 'www.yolo.us/g5h23g45f.png', 'www.data.it/246gkjnbvh2.jpg']'''
        input_df = pd.DataFrame([COLUMN_PHOTOS])
        input_df[COLUMN_PHOTOS] = [input]

        expected_output = [3]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
