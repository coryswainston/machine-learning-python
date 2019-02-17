# -*- coding: utf-8 -*-
"""
Using knn with more interesting data

@author: coryswainston
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# a method to replace categorical data with numbers
def replace_cat_data(data, col_name, order):
    data[col_name] = data[col_name].astype(CategoricalDtype(categories=order, ordered=True))
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
maint_order = ["low", "med", "high", "vhigh"]
lug_boot_order = ["small", "med", "big"]
for i in ["buying", "maint", "safety"]:
    replace_cat_data(car_data, i, maint_order)
replace_cat_data(car_data, "lug_boot", lug_boot_order)

# convert to numpy arrays
np_car_data = car_data.values.astype(int)
car_X, car_y = np.hsplit(np_car_data, [car_data.shape[1] - 1])
car_y = np.ravel(car_y)

print(car_y)

# predict
test_size = .50
X_train, X_test, y_train, y_test = train_test_split(car_X, car_y, test_size=test_size)
k = int(input("Enter number of neighbors: "))
while k < 1 or k > 20:
    k = int(input("Enter a value between 1 and 20: "))
c = KNeighborsClassifier(n_neighbors=k)
c.fit(X_train, y_train)
p = c.predict(X_test)
print(p)
print("Y test")
print(y_test)
car_results = p == y_test
#print("\nCar results:")
#print(car_results)
correct_set = len([i for i in car_results if i])
accuracy = correct_set / len(car_results)
print(accuracy)


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
