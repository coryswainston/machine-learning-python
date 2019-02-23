#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using neural networks with sklearn

@author: coryswainston
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing

#################### DATA PREPARATION ####################

# load at least one dataset
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



################# TRAINING AND PREDICTION ######################

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), learning_rate_init=0.01)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

print("\n\nDATA FOR VOTING DATASET (binary data)\n")
print(classification_report(y_test,predictions))
