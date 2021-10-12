#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# setting up CLI
parser = argparse.ArgumentParser(description="Classifier")
parser.add_argument("input_file", help="path to the input pickle file")
parser.add_argument("-s", '--seed', type=int, help="seed for the random number generator", default=None)
parser.add_argument("-e", "--export_file", help="export the trained classifier to the given location", default=None)
parser.add_argument("-i", "--import_file", help="import a trained classifier from the given location", default=None)
parser.add_argument("-m", "--majority", action="store_true", help="majority class classifier")
parser.add_argument("-rnd", "--random", action="store_true", help="50-50 / Random classifier")
parser.add_argument("-a", "--accuracy", action="store_true", help="evaluate using accuracy")
parser.add_argument("-p", "--precision", action="store_true", help="evaluate using precision")
parser.add_argument("-r", "--recall", action="store_true", help="evaluate using recall")
parser.add_argument("-c", "--cohen", action="store_true", help="evaluate using cohen's kappa")
parser.add_argument("-f1", "--f1_score", action="store_true", help="evaluate using f1")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        classifier = pickle.load(f_in)

else:  # manually set up a classifier

    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        classifier = DummyClassifier(strategy="most_frequent", random_state=args.seed)
        classifier.fit(data["features"], data["labels"])
    elif args.random:
        print("    50-50 / random classifier")
        classifier = DummyClassifier(strategy="uniform", random_state=args.seed)
        classifier.fit(data["features"], data["labels"])

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.precision:
    evaluation_metrics.append(("precision", precision_score))
if args.recall:
    evaluation_metrics.append(("recall", recall_score))
if args.f1_score:
    evaluation_metrics.append(("f1_score", f1_score))
if args.cohen:
    evaluation_metrics.append(("cohen_kappa_score", cohen_kappa_score ))

# compute and print them
for metric_name, metric in evaluation_metrics:
    print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))

# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)
