"""
Preprocessor that removes stopwords from tweet
and stores cleaned-up tweet in new column called tweet_sw_removed
"""
import ast

from nltk.corpus import stopwords

from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_STOPWORDS


class StopwordRemover(Preprocessor):
    # constructor
    def __init__(self, input_column=COLUMN_TWEET, use_tokens=False):
        # input column "tweet" or "tweet_tokenized, new output column
        super().__init__([input_column], COLUMN_STOPWORDS)
        self.use_tokens = use_tokens

    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store stopwords for later reference
        self._stopwords = stopwords.words('english')

    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        column = []

        # Remove stopwords either from tokenized tweet if available
        if self.use_tokens:
            for tokens in inputs[0]:
                token_list = ast.literal_eval(tokens)
                column.append([token for token in token_list if token.lower() not in self._stopwords])

        # If not available, split tweet first into list of words
        else:
            for tweet in inputs[0]:
                word_list = tweet.split()
                column.append([word for word in word_list if word.lower() not in self._stopwords])

        return column
