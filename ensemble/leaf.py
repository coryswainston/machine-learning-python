# -*- coding: utf-8 -*-
"""
Ensemble learning assignment

@author: coryswainston
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier



# load a dataset
cols = ["class","number","eccentricity","aspect ratio","elongation",
        "solidity","convexity","iso factor","max indentation depth",
        "lobedness","avg intensity","avg contrast","smoothness",
        "third moment","uniformity","entropy"]
data = pd.read_csv('data/leaf.csv', 
                   index_col=False,
                   names=cols,
                   skipinitialspace=True)

#################### PREPARE THE DATA ########################
X = data.drop(columns=['class','number']).values
y = data['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


################### NEURAL NETWORK ##########################
print("MLP:")
mlp = MLPClassifier(hidden_layer_sizes=(30,30), learning_rate_init=0.01, max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(accuracy_score(y_test, predictions))

################### NAIVE BAYES #############################
print("Naive Bayes:")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
print(accuracy_score(y_test, predictions))

################### K NEAREST NEIGHBORS #######################
print("KNN:")
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))



print ("\n")


###################### BAGGING ################################
print("Bagging:")
bagging = BaggingClassifier(base_estimator=knn, max_samples=.9, max_features=.9)
#bagging = BaggingClassifier(base_estimator=gnb, max_samples=0.5, max_features=0.5)
#bagging = BaggingClassifier(base_estimator=mlp, max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)
predictions = bagging.predict(X_test)
print(accuracy_score(y_test, predictions))

###################### ADABOOST #############################
print("AdaBoost:")
ab = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
ab.fit(X_train, y_train)
predictions = ab.predict(X_test)
print(accuracy_score(y_test, predictions))

####################### RANDOM FOREST ############################
print("Random forest:")
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(accuracy_score(y_test, predictions))

