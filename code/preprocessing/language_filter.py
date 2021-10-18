import pandas as pd
from code.preprocessing.filter import Filter
from code.util import COLUMN_LANGUAGE


class LanguageFilter(Filter):
    """
    Removes all rows from a given data frame that are not in a given language.
    """

    def __init__(self, lang_id: str, input_column: str = COLUMN_LANGUAGE):
        """
        Initializes a new LanguageFilter object.

        :param lang_id: The language identifier that should be kept.
        :param input_column: The input column of the data frame that contains the language identifier.
        """
        super().__init__()
        self.input_column = input_column
        self.lang_id = lang_id

    def _get_indices_to_remove(self, df: pd.DataFrame) -> pd.Index:
        return df[df[self.input_column] != self.lang_id].index
