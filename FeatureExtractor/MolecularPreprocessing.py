import numpy as np
import os
import sys
import csv

def remove_constant_features(feature_matrix):

    print("Features remaining before constant feature removal: %d" % feature_matrix.shape[1])

    # V1
    # CONSTANT_FEATURE_REMOVAL_RATIO = .7 
    # all_constant_features = [] 
    # for j in range(feature_matrix.shape[1]): 
    #     feature_column = feature_matrix[:,j].tolist() 
    #     # Count the number of occurences of each value in the feature array for feature_value in feature_column
    #     for feature_value in feature_column:
    #         feature_value_count = feature_column.count(feature_value) 
    #         if feature_value_count >= CONSTANT_FEATURE_REMOVAL_RATIO * len(feature_column):
    #             all_constant_features.append(j) 
    #             break

    # V2
    all_constant_features = []
    for j in range(feature_matrix.shape[1]):
        feature_column = feature_matrix[1:,j]
        if (np.array_equal(feature_column,[feature_column[0]] * len(feature_column))):
            all_constant_features.append(j)

    feature_matrix = np.delete(feature_matrix,all_constant_features,1) 

    print("Features remaining after constant feature removal: %d" % feature_matrix.shape[1])

    return feature_matrix
