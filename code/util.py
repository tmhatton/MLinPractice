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
COLUMN_DATE = "date"
COLUMN_VIDEOS = "video"
COLUMN_LANGUAGE = "language"
COLUMN_TIME = "time"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_STOPWORDS = "tweet_no_stopwords"

SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_CHAR_LENGTH = "_char_length"
SUFFIX_TOKEN_LENGTH = "_token_length"
SUFFIX_HASHTAG_NUM = "_hashtag_num"
SUFFIX_MENTION_NUM = "_mention_num"
SUFFIX_URL_NUM = "_url_num"
SUFFIX_PUNC_NUM = "_punc_num"
SUFFIX_CAP_LETTERS = "_cap_letters_num"
SUFFIX_PHOTOS_NUM = "_photos_num"
SUFFIX_VIDEOS_NUM = "_videos_num"

# Weekday dictionary
ISO_WEEKDAYS = {1: "Mon",
                2: "Tue",
                3: "Wed",
                4: "Thu",
                5: "Fri",
                6: "Sat",
                7: "Sun"}


def flatten(t):
    flat_list = []
    for sublist in t:
        if isinstance(sublist, list):
            for item in sublist:
                flat_list.append(item)
        else:
            flat_list.append(sublist)
    return flat_list


FEATURE_SENTIMENT = "sentiment"
FEATURE_TIMES_OF_DAY = "times_of_day"
