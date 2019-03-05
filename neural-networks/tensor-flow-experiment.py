# -*- coding: utf-8 -*-
"""
Using neural networks

@author: coryswainston
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# load at least one dataset
cols = ["class", "handicapped-infants", "water-project", 
        "budget-res", "physician-fee", "el-salvador", "religious-groups",
        "anti-satellite", "nicaragua", "mx-missile", "immigration", "synfuels", 
        "education", "superfund", "crime", "duty-free", "south-africa"]
data = pd.read_csv('data/voting.data.txt', 
                   index_col=False, 
                   names=cols, 
                   skipinitialspace=True,
                   na_values="?")
og_data = data.copy()

# prepare the dataset for a neural network
#   handle missing data, normalize numeric values, etc.
data.replace('y', 1, inplace=True)
data.replace('n', 0, inplace=True)
lb_enc = preprocessing.LabelEncoder()
lb_enc.fit(['democrat', 'republican'])
data["class"] = lb_enc.transform(data["class"])


y = data["class"].values
X = data.drop(columns=['class']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# train a neural network
model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(16,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, batch_size=32)

result = model.predict(X_test, batch_size=32)
print(result)


# make predictions

# experiment with at least three sets of hyper-parameters
#   (number of layers, nodes per layer,
#    learning rates, activation functions, etc.)
