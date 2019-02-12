# -*- coding: utf-8 -*-
"""
File for different classifiers

@author: coryswainston
"""
import numpy as np
import math

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

class ID3DecisionTree:
    def __init__(self):
        return
        
    def fit(self, data, targets):
        # data and targets are both pandas dataframes
        self.data = data
        self.targets = targets
        self.classes = set(self.targets.values.ravel())
        self.entropy = 1 # calculate entropy by determining number of each target
        self.tree = {}
        
        self._build_tree(self.tree, self.data, self.targets)
        
    def predict(self, data):
        # traverse a tree
        return []
    
    def _build_tree(self, branch, data, targets):
        if len(set(targets.values.ravel())) == 1:
            print(targets)
            return targets[0]
        
        columns = data.columns.values
        if len(columns) == 1:
            return max(set(targets), key=targets.count)

        best_info_gain = 0        
        for c in columns:
            info_gain = self.entropy - self._calc_entropy(data[c], targets)
            if info_gain > best_info_gain:
                best_column = c
        
        # best column is a new node        
        # foreach value
        for v in set(data[best_column].values):
            subset = data.loc[data[best_column] == v]
            subset_targets = targets.loc[data.index.values]
            branch[best_column] = {}
            branch[best_column][v] = {}
            branch[best_column][v] = self._build_tree(branch[best_column][v],
                  subset, subset_targets)

    
    def _calc_entropy(self, column, targets):
        values = set(column.values)
        
        m = {}
        for v in values:
            m[v] = {}
            for c in self.classes:
                m[v][c] = 0
                
        # iterate through entire column and populate map
        for i,t in zip(column, targets.values.ravel()):
            if i in m and t in m[i]:
                m[i][t] += 1
        
        # calculate entropy for each value
        entropy = 0
        for v in values:
            entropy += self._calc_value_entropy(m[v])        
        
        return entropy
        
    def _calc_value_entropy(self, frequencies):
        entropy = 0
        print(frequencies)
        for k,v in frequencies.items():
            break

        return entropy            