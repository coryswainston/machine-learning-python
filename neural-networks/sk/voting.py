#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using neural networks for a simple binary classification

@author: coryswainston
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing

#################### DATA PREPARATION ####################

# load voting dataset
cols = ["class", "handicapped-infants", "water-project", 
        "budget-res", "physician-fee", "el-salvador", "religious-groups",
        "anti-satellite", "nicaragua", "mx-missile", "immigration", "synfuels", 
        "education", "superfund", "crime", "duty-free", "south-africa"]
data = pd.read_csv('data/voting.data.txt', 
                   index_col=False, 
                   names=cols, 
                   skipinitialspace=True)
og_data = data.copy()

# prepare the dataset for a neural network
#   handle missing data, normalize numeric values, etc.
data.replace('y', 1, inplace=True)
data.replace('n', 0, inplace=True)
data.replace('?', 1, inplace=True) #TODO fix this
lb_enc = preprocessing.LabelEncoder()
lb_enc.fit(['democrat', 'republican'])
data["class"] = lb_enc.transform(data["class"])


y = data["class"].values
X = data.drop(columns=['class']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



################ TRAINING AND PREDICTION ###################
## EXPERIMENT WITH LAYERS
print("Testing with 1 hidden layer")
for i in range(20,40):
    mlp = MLPClassifier(hidden_layer_sizes=(i), learning_rate_init=0.01, max_iter=400)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(f"Accuracy with 1 layer of {i} nodes is {accuracy_score(y_test, predictions)}")

print("Testing with 2 hidden layers")
for i in range(20,40):
    mlp = MLPClassifier(hidden_layer_sizes=(i,i), learning_rate_init=0.01, max_iter=400)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(f"Accuracy with 2 layers of {i} nodes is {accuracy_score(y_test, predictions)}")

print("Testing with 3 hidden layers")
for i in range(20,40):
    mlp = MLPClassifier(hidden_layer_sizes=(i,i,i), learning_rate_init=0.01, max_iter=400)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(f"Accuracy with 3 layers of {i} nodes is {accuracy_score(y_test, predictions)}")


## EXPERIMENT WITH SOLVERS
mlp = MLPClassifier(hidden_layer_sizes=(30,), solver="sgd", max_iter=400)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(f"Accuracy with sgd solver is {accuracy_score(y_test, predictions)}")


mlp = MLPClassifier(hidden_layer_sizes=(30,), solver="sgd", max_iter=400, learning_rate="adaptive")
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(f"Accuracy with sgd solver (adaptive learning rate) is {accuracy_score(y_test, predictions)}")

mlp = MLPClassifier(hidden_layer_sizes=(30,), solver="lbfgs", max_iter=400)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(f"Accuracy with lbfgs solver is {accuracy_score(y_test, predictions)}")



## EXPERIMENT WITH LEARNING RATES
for i in range(-4,-1):
    for j in range(1,5):
        rate = 10**i * j * 2
        
        mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, learning_rate_init=rate)
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)
        print(f"Accuracy with {rate} learning rate is {accuracy_score(y_test, predictions)}")
        