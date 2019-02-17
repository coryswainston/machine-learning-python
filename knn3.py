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
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


############### READ CSV ####################
# import the car data
car_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
car_data = pd.read_csv('data/car.data.txt', 
                       index_col=False, 
                       names=car_cols, 
                       skipinitialspace=True)
car_data_bak = car_data.copy()


############### PREPROCESSING ####################

# take care of doors, persons
category_fixes = {"doors": {"5more": 5},
                  "persons": {"more": 5}}
car_data.replace(category_fixes, inplace=True)

# still need to buying, maint, lugboot, safety
lugboot = ["small", "med", "big"]
maint_plus = ["low", "med", "high", "vhigh"]

# lug_boot
lb_enc = preprocessing.LabelEncoder()
lb_enc.fit(lugboot)
car_data["lug_boot"] = lb_enc.transform(car_data["lug_boot"])

# maint, buying, safety
mp_enc = preprocessing.LabelEncoder()
mp_enc.fit(maint_plus)
car_data["maint"] = mp_enc.transform(car_data["maint"])
car_data["buying"] = mp_enc.transform(car_data["buying"])
car_data["safety"] = mp_enc.transform(car_data["safety"])









# convert to numpy arrays
car_X = car_data.drop(columns=["safety"])
car_y = car_data["safety"]

car_X = car_X.values
car_y = car_y.values.ravel()





# predict
test_size = .30
# X_train, X_test, y_train, y_test = train_test_split(car_X, car_y, test_size=test_size)

X_train, X_test, y_train, y_test = car_X[:20], car_X[20:], car_y[:20], car_y[20:]



# k = int(input("Enter number of neighbors: "))
# while k < 1 or k > 20:
#     k = int(input("Enter a value between 1 and 20: "))
k = 8
c = KNeighborsClassifier(n_neighbors=k)
c.fit(X_train, y_train)
p = c.predict(X_test)

print(car_data_bak[:5])
print(X_train[:5])
print(y_train[:5])


print(p[:10])
print(y_test[:10])
acc_score = accuracy_score(p, y_test)
print(acc_score)










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
