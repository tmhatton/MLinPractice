import pandas as pd


class Filter:
    """
    Removes all rows from a given data frame that do not meet the filter criterion.
    """

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the filter and removes all rows from the data frame that do not meet the filter criterion.

        :param df: The data frame to clean.
        :return: The data frame without the rows that do not meet the filter criterion.
        """
        pass

    def _get_indices_to_remove(self, df: pd.DataFrame) -> pd.Index:
        """
        Collects and returns all indices that should be removed.

        :param df: The data frame to clean.
        :return: A collection of indices that should be removed.
        """
        # to be implemented by subclass
        pass
