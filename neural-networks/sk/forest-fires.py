# -*- coding: utf-8 -*-
"""
Using neural networks for a more complex regression task

@author: coryswainston
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

#################### DATA PREPARATION ####################

# load forest fire dataset
data = pd.read_csv('data/forestfires.csv', 
                   index_col=False, 
                   skipinitialspace=True)
og_data = data.copy()

# prepare the dataset for a neural network
#   handle missing data, normalize numeric values, etc.
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
          'aug', 'sep', 'oct', 'nov', 'dec']
days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

lb_enc = preprocessing.LabelEncoder()
lb_enc.fit(months)
data['month'] = lb_enc.transform(data['month'])

lb_enc.fit(days)
data['day'] = lb_enc.transform(data['day'])

y = data['area'].values
X = data.drop(columns=['area', 'rain']).values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



################ TRAINING AND PREDICTION ###################

mlp = MLPRegressor(hidden_layer_sizes=(30,30,30), early_stopping=True,
                   max_iter=10000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

print("\n\nR2 SCORE")
print(r2_score(y_test,predictions))

plt.plot(predictions)
plt.show()
plt.plot(y_test)
plt.show()