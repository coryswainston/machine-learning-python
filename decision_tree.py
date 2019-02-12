# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:04:12 2019

@author: coryswainston
"""

from classifiers import ID3DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype


# import car data
car_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
car_data = pd.read_csv('data/car.data.txt', 
                       index_col=False, 
                       names=car_cols, 
                       skipinitialspace=True)

# a method to replace categorical data with numbers
def replace_cat_data(data, col_name, order):
    data[col_name] = data[col_name].astype(CategoricalDtype(categories=order, ordered=True))
    data[col_name] = data[col_name].cat.codes

# take care of categorical data
category_fixes = {"doors": {"5more": 5},
                  "persons": {"more": 5}}
car_data.replace(category_fixes, inplace=True)
maint_order = ["low", "med", "high", "vhigh"]
lug_boot_order = ["small", "med", "big"]
for i in ["buying", "maint", "safety"]:
    replace_cat_data(car_data, i, maint_order)
replace_cat_data(car_data, "lug_boot", lug_boot_order)

y = car_data[['safety']]
X = car_data.drop(columns=['safety'])

# randomize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# predict
skt = DecisionTreeClassifier()
skt.fit(X_train, y_train)
res = skt.predict(X_test)
car_results = res == y_test.values.ravel()
correct_set = len([i for i in car_results if i])
accuracy = correct_set / len(car_results)
print(accuracy)

# predict
dt = ID3DecisionTree()
dt.fit(X_train, y_train)
res = dt.predict(X_test)
car_results = res == y_test
correct_set = len([i for i in car_results if i])
accuracy = correct_set / len(car_results)
print(accuracy)
