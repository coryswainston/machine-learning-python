# -*- coding: utf-8 -*-
"""
A basic program to predict types of flowers

@author: coryswainston
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from classifiers import HardCodedClassifier

# get the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split and randomize the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# first run it through a real classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)
gaussian_pred = classifier.predict(X_test)
gaussian_results = gaussian_pred == y_test
print("\nGaussianNB results:")
print(gaussian_results)

# this one is just a hard-coded result every time
hc_classifier = HardCodedClassifier()
hc_classifier.fit(X_train, y_train)
hc_pred = hc_classifier.predict(X_test)
hc_results = hc_pred == y_test
print("\nHard coded results:")
print(hc_results)

# compare accuracy
print("\nAccuracy:")

gaussian_correct_set = len([i for i in gaussian_results if i])
gaussian_accuracy = gaussian_correct_set / len(gaussian_results)
print(f"GuassianNB: {gaussian_accuracy:.2f}")

hc_correct_set = len([i for i in hc_results if i])
hc_accuracy = hc_correct_set / len(hc_results)
print(f"HardCodedClassifier: {hc_accuracy:.2f}")

