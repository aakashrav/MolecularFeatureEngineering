#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

__author__="Aakash Ravi"

def identify_uniform_features( feature_matrix, num_features ):
    "This function takes in a correlation matrix and computes \
    features which do not differ a lot in the data set. \
    This is done by taking the diagonal values of the covariance \
    matrix, which correspond to the variation of a certain feature \
    and taking the square root, obtaining the standard deviation. \
    We then divide the standard deviation by the mean of the feature, \
    this way we have a normalized score that we can compare accross \
    features. We can then use this score to identify the 'best' features \
    and return their indices."
    
    # Avoid degenerate cases when our dataset is sometimes empty
    if feature_matrix == []:
        print("ERROR: empty feature matrix, couldn't identify \
            uniform features")
        return []

    cv_matrix = np.cov(feature_matrix, None, rowvar=0)

    # Take diagonal variance values and compute the standard deviation
    d = np.diag(cv_matrix)
    # Compute the standard deviation
    std_deviation = np.sqrt(d)
    # Divide by the mean for the feature
    mean_features = np.mean(feature_matrix, axis=0)
    variance_score = np.divide(std_deviation,mean_features)
    
    # Take the features with the highest scores
    temp = num_features * -1
    indices = np.argpartition(variance_score,temp)[temp:]

    return indices

# TODO: Implement on the next version(?)
def identify_correlated_uniform_features( feature_matrix, uniform_indices, \
    num_features, threshold ):
    "This function ideally takes the output indices from identify_uniform_features \
    and finds features that are correlated to these output indices. This will help in \
    identifying features that are highly *positively* correlated to these uniform \
    indices, giving us an even broader range of features to choose from."

