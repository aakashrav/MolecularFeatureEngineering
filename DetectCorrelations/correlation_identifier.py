#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model

__author__="Aakash Ravi"

def identify_uniform_features( feature_matrix, num_features ):
    "This function takes in a matrix of size (n x m), where n is the \
    number of fragments, and m the number of features. It then computes \
    features which do not differ a lot in the data set. \
    This is done by taking the diagonal values of the covariance \
    matrix, which correspond to the variation of a certain feature \
    and taking the square root, obtaining the standard deviation. \
    We then divide the standard deviation by the mean of the feature, \
    this way we have a normalized score that we can compare accross \
    features. We can then use this score to identify the 'best' features- \
    features with the lowest variance -  and return their indices. \
    This score is also known as the 'Coefficient \
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


def identify_correlated_features( feature_matrix, \
    num_features, threshold = .30 ):
    "This function takes as input the feature_matrix, and returns a subset of features that are \
    highly representative of all the features. This subset will be in the form of a vector containing \
    the indices of the subset of features. \
    This is done by finding features with a lot of 'neighbors' in the correlation matrix. A feature \
    i has neighbor feature j, if corr(i,j) >= threshold (so neighbors are highly correlated). We will \
    then identify num_features features with the highest amount of neighbors. Credits to this method \
    goes to Ondrej Micka."

    # Avoid degenerate cases when our dataset is sometimes empty
    if feature_matrix == []:
        print("ERROR: empty feature matrix, couldn't identify \
            uniform features")
        return []

    cv_matrix = np.cov(feature_matrix, None, rowvar=0)

    neighbor_matrix = get_neighbor_matrix(cv_matrix, threshold)
    # Vector holding the degree (number of neighbors) for every feature
    degree_vector = []
    for row in neighbor_matrix:
        deg = len(filter(lambda x: x == 1, row))
        # We subtract -1 since a feature is always perfectly correlated to itself
        degree_vector.append(deg - 1) 
    
    if degree_vector == []:
        max_degree_feature = 0
    else:
        max_degree_feature = max(degree_vector)
        index_of_max_feature = degree_vector.index(max_degree_feature)

    # Keep track of all features that have some sort of correlation
    features_with_neighbors = [True]*len(degree_vector)

    # This vector will keep track of features that have been removed from consideration,
    # because they were heavily correlated with other features. It's usage will become
    # clear later.
    unecessary_features = []
            
    # While there are correlated features, we choose the feature with highest degree, 
    # the one with the most neighbors, as a representant of some 'neighbor class'. We
    # then delete all features that are correlated with this representant (if it wasn't)
    # already chosen)
    significant_features = []
    while(max_degree_feature > 0):
        significant_features.append(index_of_max_feature)

        # We start to clean up the neighbor matrix by making sure all neighbors of our 
        # chosen representative no longer count as neighbors for other feaures since
        # they will be removed.
        for j in range(0,len(cv_matrix)):
            # Perform for every neighbor of our chosen 'max' feature
            if (j != index_of_max_feature) and \
            features_with_neighbors[j] and \
            (cv_matrix[index_of_max_feature][j] >= threshold):
                # First reduce the degree of all j's neighbors, since we will be removing it
                for k in range(0,len(cv_matrix)):
                    if features_with_neighbors[k] and (cv_matrix[k][j]>=threshold):
                        degree_vector[k] -= 1
                # Add the feature to the list of unecessary features
                unecessary_features.append(j)

        # Next, we finally remove all neighbors of i, since we already chose i as one of our features
        # and we don't want correlated features 
        for j in range(0,len(cv_matrix)):
            if (j != index_of_max_feature) and \
            features_with_neighbors[j] and \
            (cv_matrix[index_of_max_feature][j] >= threshold):
                degree_vector[j] = 0
                features_with_neighbors[j] = False

        # Then move on to the next feature with neighbors, until we have chosen all of them
        max_degree_feature = max(degree_vector)
        index_of_max_feature = degree_vector.index(max_degree_feature)

    # Only keep the representants of each 'neighbor class' found from the previous
    # method, as well as features that are not correlated heavily with any other features.
    all_features = np.arange(len(feature_matrix[0]))
    non_redundant_features = np.delete(all_features, unecessary_features, 0)
    significant_features.extend(non_redundant_features)

    # Return the requested amount of significant features
    if (len(significant_features) < num_features):
        return significant_features
    else:
        return significant_features[1:num_features]

def get_neighbor_matrix(covariance_matrix, threshold):
    "Returns a matrix M, where M(i,j)=M(j,i)=1 if cov(feature i, feature j)>=threshold, \
    and M(i,j)=M(j,i)=0 otherwise."
    
    neighbor_matrix = np.zeros(shape=(len(covariance_matrix),len(covariance_matrix)))

    for i in range(0, len(covariance_matrix)):
        for j in range(0, len(covariance_matrix[i])):
            if covariance_matrix[i][j] >= threshold:
                neighbor_matrix[i][j] =1

    return neighbor_matrix


