# -*- coding: utf-8 -*-
"""
Using neural networks

@author: coryswainston
"""

import numpy as np
import pandas as pd
#import tensorflow as tf
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
data.replace('y', 'c')
data.replace('n', 0)


# train a neural network

# make predictions

# experiment with at least three sets of hyper-parameters
#   (number of layers, nodes per layer,
#    learning rates, activation functions, etc.)
