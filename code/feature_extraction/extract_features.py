#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.list_counter import URLsNum, PhotosNum, HashtagNum, MentionNum, TokenNum
from code.feature_extraction.conditional_char_counter import PunctuationNum, CapLettersNum
from code.feature_extraction.videos_num import VideosNum
from code.feature_extraction.weekday_extractor import WeekdayExtractor
from code.feature_extraction.sentiment import Sentiment
from code.feature_extraction.times_of_day import TimesOfDayExtractor
from code.util import COLUMN_TWEET, COLUMN_LABEL, SUFFIX_TOKENIZED, COLUMN_STOPWORDS


# setting up CLI
parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output pickle file")
parser.add_argument("-e", "--export_file", help="create a pipeline and export to the given location", default=None)
parser.add_argument("-i", "--import_file", help="import an existing pipeline from the given location", default=None)
parser.add_argument("-c", "--char_length", action="store_true", help="compute the number of characters in the tweet")
parser.add_argument("-t", "--token_length", action="store_true", help="compute the number of words/tokens in the tweet")
parser.add_argument("--hashtag_num", action="store_true", help="compute the number hashtags in the tweet")
parser.add_argument("-m", "--mention_num", action="store_true", help="compute the number of mentions in the tweet")
parser.add_argument("-u", "--url_num", action="store_true", help="compute the number of URLs in the tweet")
parser.add_argument("--cap_letter", action="store_true", help="compute the number of capital letters in the tweet")
parser.add_argument("-p", "--punc_num", action="store_true", help="compute the number punctuation characters in the tweet")
parser.add_argument("--photos_num", action="store_true", help="compute the number of attached photos in the tweet")
parser.add_argument("--videos_num", action="store_true", help="compute the number of attached videos in the tweet")
parser.add_argument("-w", "--weekday", action="store_true", help="extract the one-hot encoded weekday of the tweet's date")
parser.add_argument("-s", "--sentiment", action="store_true", help="compute the sentiment (i.e. polarity and subjectivity) of the tweet")
parser.add_argument("-tod", "--times_of_day", action="store_true", help="extracts the one-hot encoded time of day from the tweet's time")

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")

if args.import_file is not None:
    # simply import an existing FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:  # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(CharacterLength(COLUMN_TWEET))
    if args.token_length:
        # number of tokens of tokenized tweet
        features.append(TokenNum(COLUMN_TWEET + SUFFIX_TOKENIZED))
        # number of tokens of tokenized tweet without stop words
        features.append(TokenNum(COLUMN_STOPWORDS))
    if args.hashtag_num:
        # number of hashtags of original tweet data from hashtags column
        features.append(HashtagNum())
    if args.mention_num:
        # number of mentions of original tweet data from mentions column
        features.append(MentionNum())
    if args.url_num:
        # number of URLs of original tweet data from urls column
        features.append(URLsNum())
    if args.punc_num:
        # number of punctuation characters in original tweet (without any changes)
        features.append(PunctuationNum(COLUMN_TWEET))
    if args.cap_letter:
        # number of capital letters in original tweet (without any changes)
        features.append(CapLettersNum(COLUMN_TWEET))
    if args.photos_num:
        # number of photos in the tweet
        features.append(PhotosNum())
    if args.videos_num:
        # number of videos in the tweet
        features.append(VideosNum())
    if args.weekday:
        # extract and one-hot-encode the weekday from the date of the tweet
        features.append(WeekdayExtractor())
    if args.sentiment:
        # calculates the sentiment of the tweets
        features.append(Sentiment())
    if args.times_of_day:
        # extract and one-hot-encode the times of day from the time of the tweet
        features.append(TimesOfDayExtractor())

    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)

    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)

# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array,
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)
