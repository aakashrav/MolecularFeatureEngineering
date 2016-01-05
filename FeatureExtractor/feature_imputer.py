#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A helper function working as a black box imputer for feature vectors from PADEL
"""

from sklearn.preprocessing import Imputer
import numpy as np

def ImputeFeatures( fragment_feature_matrix ):
    "This function will use sklearn's built in imputer to impute values of the feature matrix"

    # For missing values encoded as np.nan, use the string value “NaN”
    imp = Imputer(missing_values="NaN", strategy='mean', axis=0)
    # Fit and transform in one step and return
    return imp.fit_transform(fragment_feature_matrix)

def FinalImpute( fragment_feature_matrix ):
    "This function will perform a final impute on the data; it will remove \
    all data rows that contain a NaN value. This method should only be called \
    as a last result where even after the imputation of the prior method, we \
    still have that some features for some molecules are still NaN. \
    The only way this may happen is in the unlikely case that all fragments \
    for a particular molecule have a feature as NaN. As unlikely as \
    this is the data we get from PADEL is quite messy and therefore this method is a necessary evil."

    nan_row_indices = []

    # Find rows that contain a NaN value
    # O(n^2) complexity, TODO: Make this faster
    for i in range (0, len(fragment_feature_matrix) -1):
        for j in range (0, len(fragment_feature_matrix[0])-1):
            if np.isnan(fragment_feature_matrix[i][j]):
                nan_row_indices.append(i)

    # Remove only our chosen rows
    clean_feature_matrix = np.delete(fragment_feature_matrix, nan_row_indices, 0)
    return clean_feature_matrix
