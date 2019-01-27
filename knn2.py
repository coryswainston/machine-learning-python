# -*- coding: utf-8 -*-
"""
Using knn with more interesting data

@author: coryswainston
"""

import numpy as np
import pandas as pd
from classifiers import KNearestNeighbors
from knn import run_knn

# a method to replace categorical data with numbers
def replace_cat_data(data, col_name):
    data[col_name] = data[col_name].astype('category')
    data[col_name] = data[col_name].cat.codes

# import the car data
car_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
car_data = pd.read_csv('data/car.data.txt', 
                       index_col=False, 
                       names=car_cols, 
                       skipinitialspace=True)

# take care of categorical data
category_fixes = {"doors": {"5more": 5},
                  "persons": {"more": 5}}
car_data.replace(category_fixes, inplace=True)
for i in ["buying", "maint", "lug_boot", "safety"]:
    replace_cat_data(car_data, i)

# convert to numpy arrays
np_car_data = car_data.values
car_X, car_y = np.hsplit(np_car_data, [car_data.shape[1] - 1])

# predict
run_knn(car_X, car_y)


# import the mpg data
mpg_cols = ["mpg", "cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model year", "origin", "car name"]
mpg_data = pd.read_csv('data/auto-mpg.data.txt', 
                       index_col=False, 
                       names=mpg_cols, 
                       delim_whitespace=True,
                       na_values=["?"])

# import the student data 
mat_data = pd.read_csv('data/student-mat.csv',
                       index_col=False,
                       sep=';')
