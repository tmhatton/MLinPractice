#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# setting up CLI
parser = argparse.ArgumentParser(description="Classifier")
parser.add_argument("input_file", help="path to the input pickle file")
parser.add_argument("-s", '--seed', type=int, help="seed for the random number generator", default=None)
parser.add_argument("-e", "--export_file", help="export the trained classifier to the given location", default=None)
parser.add_argument("-i", "--import_file", help="import a trained classifier from the given location", default=None)
parser.add_argument("-m", "--majority", action="store_true", help="majority class classifier")
parser.add_argument("-rnd", "--random", action="store_true", help="50-50 / Random classifier")
parser.add_argument("-at", "--always_true", action="store_true", help="Always 'True' classifier")
parser.add_argument("--knn", type = int, help = "k-nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("-lf", "--label_frequency", action="store_true", help="Label Frequency classifier")
parser.add_argument("-af", "--always_false", action="store_true", help="Always 'False' classifier")
parser.add_argument("-svm", "--support_vector_machine", action="store_true", help="Support Vector Machines classifier")
parser.add_argument("-rf", "--random_forest", action="store_true", help="Random Forest classifier")
parser.add_argument("-nb", "--naive_bayes", action="store_true", help="Naive Bayes classifier")
parser.add_argument("-a", "--accuracy", action="store_true", help="evaluate using accuracy")
parser.add_argument("-p", "--precision", action="store_true", help="evaluate using precision")
parser.add_argument("-r", "--recall", action="store_true", help="evaluate using recall")
parser.add_argument("-c", "--cohen", action="store_true", help="evaluate using cohen's kappa")
parser.add_argument("-l", "--log_loss", action="store_true", help="evaluate using the log loss")
parser.add_argument("-f1", "--f1_score", action="store_true", help="evaluate using f1")
parser.add_argument("-auc", "--auc_roc", action="store_true", help="evaluate AUC-ROC score")

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
    elif args.always_true:
        print("    always 'true' classifier")
        classifier = DummyClassifier(strategy="constant", random_state=args.seed, constant=1)
        classifier.fit(data["features"], data["labels"])
    elif args.always_false:
        print("    always 'false' classifier")
        classifier = DummyClassifier(strategy="constant", random_state=args.seed, constant=0)
        classifier.fit(data["features"], data["labels"])
    elif args.label_frequency:
        # label frequency classifier
        print("    label frequency classifier")
        classifier = DummyClassifier(strategy="stratified", random_state=args.seed)
        classifier.fit(data["features"], data["labels"])
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn)
        classifier = make_pipeline(standardizer, knn_classifier)
        classifier.fit(data["features"], data["labels"].ravel())
    elif args.support_vector_machine:
        print("    Support Vector Machines classifier")
        classifier = SVC()
        classifier.fit(data["features"], data["labels"].ravel())
    elif args.random_forest:
        print("    Random Forest classifier")
        standardizer = StandardScaler()
        rf_classifier = RandomForestClassifier(n_estimators = 1000)
        classifier = make_pipeline(standardizer, rf_classifier)
        classifier.fit(data["features"], data["labels"].ravel())
    elif args.naive_bayes:
        print("    Naive Bayes classifier")
        classifier = GaussianNB()
        classifier.fit(data["features"], data["labels"].ravel())

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
    evaluation_metrics.append(("cohen_kappa_score", cohen_kappa_score))
if args.log_loss:
    evaluation_metrics.append(("log_loss", log_loss))
if args.auc_roc:
    evaluation_metrics.append(("roc_auc", roc_auc_score))

# compute and print them
for metric_name, metric in evaluation_metrics:
    print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))

# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)
