# -*- coding: utf-8 -*-
"""
Regression task with popularity of news articles

@author: coryswainston
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


#################### DATA PREPARATION ####################

# load forest fire dataset
data = pd.read_csv('data/news.csv', 
                   index_col=False, 
                   skipinitialspace=True)
og_data = data.copy()

y = data['shares'].values
X = data.drop(columns=['url', 'shares']).values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


################ TRAINING AND PREDICTION ###################

#### TEST WITH DIFFERENT NUMBERS OF NODES
for i in range(1,30):
    layer_size = i * 2
    mlp = MLPRegressor(hidden_layer_sizes=(layer_size,), solver='adam', max_iter=2000, early_stopping=True, verbose=False)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    
    print(mean_absolute_error(y_test,predictions))
    
    plt.plot(predictions)
    plt.show()
    plt.plot(y_test)
    plt.show()