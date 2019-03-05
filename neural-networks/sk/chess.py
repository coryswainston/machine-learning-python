# -*- coding: utf-8 -*-
"""
Classification on a more nuanced chess dataset using MLP

@author: coryswainston
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing

#################### DATA PREPARATION ####################

# load voting dataset
cols = ["wk-file","wk-rank","wr-file","wr-rank","bk-file","bk-rank","depth-of-win"]
data = pd.read_csv('data/chess.data.txt', 
                   index_col=False, 
                   names=cols, 
                   skipinitialspace=True)
og_data = data.copy()

# prepare the dataset for a neural network
#   handle missing data, normalize numeric values, etc.
lb_enc = preprocessing.LabelEncoder()
lb_enc.fit(['a','b','c','d','e','f','g','h'])
data["wk-file"] = lb_enc.transform(data["wk-file"])
data["wr-file"] = lb_enc.transform(data["wr-file"])
data["bk-file"] = lb_enc.transform(data["bk-file"])


y = data["depth-of-win"].values
X = data.drop(columns=['depth-of-win']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

################ TRAINING AND PREDICTION ###################
## Test different layer sizes
print("Testing with 1 hidden layer")
for i in range(1,8):
    mlp = MLPClassifier(hidden_layer_sizes=(i*10), learning_rate_init=0.01, max_iter=400)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(f"Accuracy with 1 layer of {i*10} nodes is {accuracy_score(y_test, predictions)}")

print("Testing with 2 hidden layers")
for i in range(1,8):
    mlp = MLPClassifier(hidden_layer_sizes=(i*10,i*10), learning_rate_init=0.01, max_iter=400)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(f"Accuracy with 2 layers of {i*10} nodes is {accuracy_score(y_test, predictions)}")

print("Testing with 3 hidden layers")
for i in range(1,8):
    mlp = MLPClassifier(hidden_layer_sizes=(i*10,i*10,i*10), learning_rate_init=0.01, max_iter=400)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(f"Accuracy with 3 layers of {i*10} nodes is {accuracy_score(y_test, predictions)}")


## Test different learning rates
for i in range(1,10):
    mlp = MLPClassifier(hidden_layer_sizes=(60,60,60), max_iter=500, 
                        early_stopping=False, verbose=False, 
                        learning_rate_init=(i * 0.001))
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    print(f"Accuracy is {accuracy_score(y_test, predictions)}")
