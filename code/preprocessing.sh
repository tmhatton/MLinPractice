#!/bin/bash

# create directory if not yet existing
mkdir -p code/data/preprocessing/split/

#install all nltk models
python -m nltk.downloader all

# add labels
echo "  creating labels"
python -m code.preprocessing.create_labels code/data/raw/ code/data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo "  general preprocessing"
python -m code.preprocessing.run_preprocessing code/data/preprocessing/labeled.csv code/data/preprocessing/preprocessed.csv --punctuation --tokenize -e code/data/preprocessing/pipeline.pickle

# split the data set
echo "  splitting the data set"
python -m code.preprocessing.split_data code/data/preprocessing/preprocessed.csv code/data/preprocessing/split/ -s 42