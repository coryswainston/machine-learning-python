# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:43:05 2019

@author: swain
"""

import pandas as pd

column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", 
                "capital-loss", "hours-per-week", "native-country"]

data = pd.read_csv('adult.data.txt', index_col = False, names = column_names, 
                   skipinitialspace = True, na_values=["?"])

