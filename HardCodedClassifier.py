# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:54:01 2019

@author: swain
"""

class HardCodedClassifier:
    def __init__(self):
        return
    
    def fit(self, data, targets):
        print("called fit()")
        
    def predict(self, data):
        return [0 for i in data]