#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model

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
    and return their indices. This score is also known as the 'Coefficient \
    of Variance'"
    
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
    # We need to take the absolute value of the mean since the mean may be
    # negative. We only care about the ratio between the mean and standard
    # deviation, so dividing by the absolute value suffices.
    variance_score = np.divide(std_deviation,np.absolute(mean_features))
    
    # Take the features with the lowest scores -
    # these correspond to features with the lowest variation
    indices = np.argpartition(variance_score,num_features)[0:num_features]

    return indices

def get_top_features( feature_matrix, num_features ):
    "This function performs performs logistic regression on our sample fragment \
    data and finds coefficients for the features. Using these coefficients the \
    function will return the most important features that correspond to the active \
    molecules by choosing the features that correspond to the highest coefficient values."

    log_reg = linear_model.LogisticRegression(solver = 'liblinear')
    TRAINING_DATA = np.array(feature_matrix)[0:len(feature_matrix)*.8,0:len(feature_matrix[0])-1]
    TEST_DATA = np.array(feature_matrix)[len(feature_matrix)*.8:len(feature_matrix), \
    0:len(feature_matrix[0])-1]

    TRAINING_RESULTS = np.array(feature_matrix)[0:len(feature_matrix)*.8,len(feature_matrix[0])-1]
    TEST_RESULTS = np.array(feature_matrix)[len(feature_matrix) *.8:len(feature_matrix), \
    len(feature_matrix[0])-1]
    
    print(log_reg.fit(TRAINING_DATA, TRAINING_RESULTS))


# TODO: Implement on the next version(?)
def identify_correlated_uniform_features( feature_matrix, uniform_indices, \
    num_features, threshold ):
    "This function ideally takes the output indices from identify_uniform_features \
    and finds features that are correlated to these output indices. This will help in \
    identifying features that are highly *positively* correlated to these uniform \
    indices, giving us an even broader range of features to choose from."

