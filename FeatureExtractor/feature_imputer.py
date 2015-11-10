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
