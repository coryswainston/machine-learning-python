# -*- coding: utf-8 -*-
"""
File for different classifiers

@author: coryswainston
"""
import numpy as np

class HardCodedClassifier:
    def __init__(self):
        return
    
    def fit(self, data, targets):
        # do nothin
        return
        
    def predict(self, data):
        # we're bound to be right sometimes?
        return [0 for i in data]
    
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        
    def fit(self, data, targets):
        self.data = data
        self.targets = targets
        
    def predict(self, data):
        assert(np.shape(data)[1] == np.shape(self.data)[1])
        targets = []                
        for row in data:
            targets.append(self._get_target_for_row(row))                
        return targets
    
    def _get_target_for_row(self, row):
        # get the distance of each datapoint along with its target
        distances_by_target = []
        for srow, starget in zip(self.data, self.targets):
            distances = [(x-y)**2 for x, y in zip(srow, row)]
            distances_by_target.append((starget, sum(distances)))
        # sort from smallest to largest
        distances_by_target.sort(key=lambda tup: tup[1])
        # get k nearest neighbors
        nearest_neighbors = [i[0] for i in distances_by_target[0:self.k]]
        # find the most common classification
        return max(set(nearest_neighbors), key=nearest_neighbors.count) 
        