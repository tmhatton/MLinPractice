#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse
import pickle

from mlflow import log_metric, log_param, set_tracking_uri
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, log_loss, \
    roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# setting up CLI
parser = argparse.ArgumentParser(description="Classifier")
parser.add_argument("input_file", help="path to the input pickle file")
parser.add_argument("-s", '--seed', type=int, help="seed for the random number generator", default=None)
parser.add_argument("-e", "--export_file", help="export the trained classifier to the given location", default=None)
parser.add_argument("-i", "--import_file", help="import a trained classifier from the given location", default=None)
parser.add_argument("--log_folder", help="where to log the mlflow results", default="data/classification/mlflow")
parser.add_argument("-m", "--majority", action="store_true", help="majority class classifier")
parser.add_argument("-rnd", "--random", action="store_true", help="50-50 / Random classifier")
parser.add_argument("-at", "--always_true", action="store_true", help="Always 'True' classifier")
parser.add_argument("--knn", type=int, help="k-nearest neighbor classifier with the specified value of k", default=None)
parser.add_argument("-lf", "--label_frequency", action="store_true", help="Label Frequency classifier")
parser.add_argument("-af", "--always_false", action="store_true", help="Always 'False' classifier")
parser.add_argument("-svm", "--support_vector_machine", action="store_true", help="Support Vector Machines classifier")
parser.add_argument("--svm_kernel", help="SVM kernel", choices=["linear", "poly", "rbf", "sigmoid", "precomputed"], default='rbf')
parser.add_argument("--svm_C", type=float, help="SVM regularization parameter. Must be strictly positive", default=1.0)
parser.add_argument("-rf", "--random_forest", action="store_true", help="Random Forest classifier")
parser.add_argument("--rf_n", type=int, help="Random Forest Number of Estimators", default=100)
parser.add_argument("--rf_criterion", type=str, help="Random Forest split criterion", choices=['gini', 'entropy'], default='gini')
parser.add_argument("--rf_weights", type=int, help="Random Forest class weights. 1 = balanced, 0 = not_balanced", default=0)
parser.add_argument("-lsvc", "--linear_svc", action="store_true")
parser.add_argument("-dt", "--decision_tree", action="store_true")
parser.add_argument("-nb", "--naive_bayes", action="store_true", help="Naive Bayes classifier")
parser.add_argument("-a", "--accuracy", action="store_true", help="evaluate using accuracy")
parser.add_argument("-p", "--precision", action="store_true", help="evaluate using precision")
parser.add_argument("-r", "--recall", action="store_true", help="evaluate using recall")
parser.add_argument("-c", "--cohen", action="store_true", help="evaluate using cohen's kappa")
parser.add_argument("-l", "--log_loss", action="store_true", help="evaluate using the log loss")
parser.add_argument("-f1", "--f1_score", action="store_true", help="evaluate using f1")
parser.add_argument("-auc", "--auc_roc", action="store_true", help="evaluate AUC-ROC score")

args = parser.parse_args()
params = {}

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

set_tracking_uri(args.log_folder)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)

    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)

    log_param("dataset", "validation")

else:  # manually set up a classifier

    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        classifier = DummyClassifier(strategy="most_frequent", random_state=args.seed)
    elif args.random:
        print("    50-50 / random classifier")
        classifier = DummyClassifier(strategy="uniform", random_state=args.seed)
    elif args.always_true:
        print("    always 'true' classifier")
        classifier = DummyClassifier(strategy="constant", random_state=args.seed, constant=1)
    elif args.always_false:
        print("    always 'false' classifier")
        classifier = DummyClassifier(strategy="constant", random_state=args.seed, constant=0)
    elif args.label_frequency:
        # label frequency classifier
        print("    label frequency classifier")
        classifier = DummyClassifier(strategy="stratified", random_state=args.seed)
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs=-1)
        classifier = make_pipeline(standardizer, knn_classifier)
    elif args.support_vector_machine:
        print("    Support Vector Machines classifier")
        log_param("classifier", "svm")
        log_param("kernel", args.svm_kernel)
        log_param("C", args.svm_C)
        params = {"classifier": "svm", "kernel": args.svm_kernel, "C": args.svm_C}
        classifier = SVC(kernel=args.svm_kernel, C=args.svm_C)
    elif args.linear_svc:
        print("    Linear Support Vector Machine classifier")
        classifier = LinearSVC(class_weight='balanced', max_iter=100_000)
    elif args.random_forest:
        print("    Random Forest classifier")
        log_param("classifier", "random_forest")
        log_param("n_estimators", args.rf_n)
        log_param("criterion", args.rf_criterion)
        rf_weights = 'balanced' if args.rf_weights == 1 else None
        log_param("class_weights", rf_weights)
        standardizer = StandardScaler()
        rf_classifier = RandomForestClassifier(n_estimators=args.rf_n, criterion=args.rf_criterion,
                                               class_weight=rf_weights, n_jobs=-1)
        classifier = make_pipeline(standardizer, rf_classifier)
    elif args.decision_tree:
        print("    Decision Tree classifier")
        classifier = DecisionTreeClassifier(class_weight='balanced')
    elif args.naive_bayes:
        print("    Naive Bayes classifier")
        classifier = GaussianNB()

    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

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
    metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))
    log_metric(metric_name, metric_value)

# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
