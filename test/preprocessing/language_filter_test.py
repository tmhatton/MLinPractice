import unittest
import pandas as pd
import random

from code.preprocessing.language_filter import LanguageFilter


class LanguageFilterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.INPUT_COLUMN = 'input'
        self.CORRECT_LANG_ID = 'en'
        self.WRONG_LANG_ID_1 = 'de'
        self.WRONG_LANG_ID_2 = 'es'
        self.filter = LanguageFilter(lang_id=self.CORRECT_LANG_ID, input_column=self.INPUT_COLUMN)

    def test_execute(self):
        """
        Tests whether all non-english rows are removed from the data frame.
        """
        # create the data and validate it
        input_data = [self.CORRECT_LANG_ID] * 4
        input_data.extend([self.WRONG_LANG_ID_1] * 3)
        input_data.extend([self.WRONG_LANG_ID_2] * 2)
        random.shuffle(input_data)
        df = pd.DataFrame(input_data, columns=[self.INPUT_COLUMN])
        self.assertEqual(len(df), 9, 'Wrong number of rows before the execution of the filter.')

        # execute the filter
        filtered_df = self.filter.execute(df)

        # validate the filtered df
        self.assertEqual(len(filtered_df), 4, 'Wrong number of rows after the execution of the filter.')
        num_of_correct_lang_ids = (filtered_df[self.INPUT_COLUMN] == self.CORRECT_LANG_ID).sum()
        self.assertEqual(num_of_correct_lang_ids, 4, 'Data frame still contains wrong language ids.')


if __name__ == '__main__':
    unittest.main()
