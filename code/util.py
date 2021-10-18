#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_HASHTAGS = "hashtags"
COLUMN_MENTIONS = "mentions"
COLUMN_URLS = "urls"
COLUMN_PHOTOS = "photos"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"

SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_CHAR_LENGTH = "_char_length"
SUFFIX_TOKEN_LENGTH = "_token_length"
SUFFIX_HASHTAG_NUM = "_hashtag_num"
SUFFIX_MENTION_NUM = "_mention_num"
SUFFIX_URL_NUM = "_url_num"
SUFFIX_PUNC_NUM = "_punc_num"
SUFFIX_CAP_LETTERS = "_cap_letters_num"
SUFFIX_PHOTOS_NUM = "_photos_num"
