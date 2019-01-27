# -*- coding: utf-8 -*-
"""
Prediction using k nearest neighbors algorithm

@author: coryswainston
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from classifiers import KNearestNeighbors

# get the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

def run_knn(X, y):
    # split and randomize the data
    test_size = .30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # get k value
    k = int(input("Enter number of neighbors: "))
    while k < 1 or k > 20:
        k = int(input("Enter a value between 1 and 20: "))
    
    # first run it through a real classifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    results = prediction == y_test
    print("\nsklearn Results:")
    print(results)
    
    # this one is just a hard-coded result every time
    my_classifier = KNearestNeighbors(k)
    my_classifier.fit(X_train, y_train)
    my_prediction = my_classifier.predict(X_test)
    my_results = my_prediction == y_test
    print("\nMy results:")
    print(my_results)
    
    # compare accuracy
    print("\nAccuracy:")
    
    correct_set = len([i for i in results if i])
    accuracy = correct_set / len(results)
    print(f"sklearn KNeighbors: {accuracy:.2f}")
    
    my_correct_set = len([i for i in my_results if i])
    my_accuracy = my_correct_set / len(my_results)
    print(f"My algorithm: {my_accuracy:.2f}")

run_knn(X, y)