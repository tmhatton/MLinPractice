import unittest
import pandas as pd

from code.feature_extraction.list_counter import PhotosNum, URLsNum, HashtagNum, MentionNum
from code.util import COLUMN_PHOTOS, COLUMN_URLS, COLUMN_HASHTAGS, COLUMN_MENTIONS


class PhotosNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_PHOTOS
        self.extractor = PhotosNum(self.INPUT_COLUMN)

    def test_photos_num(self):
        input_data = '''['www.hashtag.de/234234.jpg', 'www.yolo.us/g5h23g45f.png', 'www.data.it/246gkjnbvh2.jpg']'''
        input_df = pd.DataFrame([COLUMN_PHOTOS])
        input_df[COLUMN_PHOTOS] = [input_data]

        expected_output = [3]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


class URLsNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_URLS
        self.extractor = URLsNum(self.INPUT_COLUMN)

    def test_url_num(self):
        input_data = '''['www.google.com', 'www.apple.com', 'www.uos.de', 'www.example.com']'''
        input_df = pd.DataFrame([COLUMN_URLS])
        input_df[COLUMN_URLS] = [input_data]

        expected_output = [4]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


class HashtagNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_HASHTAGS
        self.extractor = HashtagNum(self.INPUT_COLUMN)

    def test_hashtag_num(self):
        input_data = '''['hashtag', 'yolo', 'data']'''
        input_df = pd.DataFrame([COLUMN_HASHTAGS])
        input_df[COLUMN_HASHTAGS] = [input_data]

        expected_output = [3]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


class MentionNumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = COLUMN_MENTIONS
        self.extractor = MentionNum(self.INPUT_COLUMN)

    def test_mention_num(self):
        input_data = '''[{'id': '2235729541', 'name': 'dogecoin', 'screen_name': 'dogecoin'}, {'id': '123432342', 'name': 'John Doe', 'screen_name': 'jodoe'}]'''
        input_df = pd.DataFrame([COLUMN_MENTIONS])
        input_df[COLUMN_MENTIONS] = [input_data]

        expected_output = [2]
        output = self.extractor.fit_transform(input_df)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
