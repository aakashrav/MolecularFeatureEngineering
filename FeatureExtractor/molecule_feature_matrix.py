#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this module is to provide functions to retreive features of fragments 
that are contained in the molecule, and impute the feature vectors so that they are not sparse, 
i.e. there are no missing values
"""

import os
from sklearn.preprocessing import Imputer
import json
import numpy as np
import re
import feature_imputer

__author__="Aakash Ravi"

def retrieve_features( descriptor_file, molecules_to_fragments_file,
    molecule_smiles, active_flag, feature_array ):

    "Parameter 'molecule_smiles' is an array of molecule smiles we want to fetch the features for, \
    the function returns a matrix of *average* features of the *fragments* for each smile in the array. \
    The function also requires an active_flag, which will then trigger an automatic adding of a '1' feature \
    to each molecule, indicating that it is active. Similarly with '0' and inactive. \
    Finally, the feature_array is an array of features that the function will deem relevant. \
    We may have some degenerate features (i.e. where all the values are 0) so the function will \
    remove those features from the array and return only the non degenrate ones."

    # Define the final feature matrix holding the average features of the molecule
    # The average is computed by taking the features of each fragment of the molecule
    # and dividing it by the number of fragments in that molecule
    final_feature_matrix = []

    # Take care of degenerate case where no molecules are inputted for analysis
    if molecule_smiles == []:
        print("ERROR: No molecules inputted for feature retrieval")
        return []

    with open(molecules_to_fragments_file, 'r') as input_stream:

        molecules_to_fragments = json.load(input_stream)

        # Perform this feature averaging procedure for each molecule in the array
        # 'molecule_smiles', and return a matrix of found values
        for molecule_index in range(0, len(molecule_smiles)):

            # Sometimes the desired SMILE may not have any fragment data, so we just continue
            # And notify the user that no fragments were found for this SMILES
            try:
                molecule = molecules_to_fragments[molecule_smiles[molecule_index]]
                print("Starting analysis for " + molecule_smiles[molecule_index] + "...")
            except KeyError:
                print("Fragments not available for this molecule: ", molecule_smiles[molecule_index])
                continue

            fragments = molecule['fragments']

            # We now traverse through the feature data from PADEL and obtain features
            # for each fragment, and average it over all fragments
            with open(descriptor_file,'r') as input_stream:

                header = input_stream.readline().rstrip().split(',')

                # Feature matrix for the features of various fragments
                # We add one extra column because we want to append the 1 or 0
                # based on activity or inactivity. Adding this column beforehand
                # will be much more efficient than calling .append and having to reallocate
                # memory after
                molecule_feature_matrix = np.zeros([len(fragments), len(header)], dtype=np.float)

                index =0

                for line in input_stream:
                    line = line.rstrip().split(',')
                    name = line[0][1:-1]

                    # We found a desired fragment of the molecule, so add this to the 
                    # list of features
                    if name in fragments:
                        print("Found fragment: " + name)

                        # Here we just copy the newly obtained features into our feature matrix
                        # If the feature is not a number, we explicitly input np.nan
                        for j in range(1, len(line)-1):
                            try:
                                molecule_feature_matrix[index][j] = float(line[j])
                            except ValueError:
                                molecule_feature_matrix[index][j] = np.nan
                        index+=1


                # Here we call the imputer API to impute NaN
                # values in our molecule_feature_matrix.
                # We do this since some fragments may have NaN values for some features 
                # so we impute them using known values from other fragments
                feature_imputer.ImputeFeatures(molecule_feature_matrix)

                # Compute average features over all fragments to 
                # return final averaged features of the molecule
                final_feature_vector = np.mean(molecule_feature_matrix, axis=0)

                # Add '1' at the end if its an active molecule '0' if not
                if active_flag:
                    final_feature_vector[len(header)-1] = 1
                else:
                    final_feature_vector[len(header)-1] = 0

                # And finally we add the new feature vector to our matrix of averaged features
                # At the end, this matrix will contain the averaged features for every molecule
                # inputted in parameter 'molecule_smiles'
                final_feature_matrix.append(final_feature_vector)
                print(molecule_smiles[molecule_index] + ' successfully analyzed')

        # Identify non degenerate features
        # Cannot index nested list with a multidimensional index,
        # so need to do some fancy indexing
        non_degenerate_features = \
        identify_non_degenerate_features( np.array(final_feature_matrix)[:,0:(len(final_feature_matrix[0])-1)], \
            feature_array)
        
        # Add the final characteristic feature
        non_degenerate_features.append(len(final_feature_matrix[0])-1)
        # Take only the non degenerate features from the feature matrix
        non_degenerate_feature_matrix = np.array(final_feature_matrix)[:,non_degenerate_features]

        return non_degenerate_feature_matrix

def identify_non_degenerate_features( feature_matrix, feature_array ):

    "This function will remove any features that have the exact same value \
    for all molecules. Such an event is incredibly unlikely as this implies that \
    all features in every fragment of every molecule have the same value - an \
    incredibly unlikely event. Thus we deem such features redundant and remove them."

    non_degenerate_feature_indices = []

    # Loop through all the features of the feature_matrix
    for j in range(0,len(feature_matrix[0])):

        # Cannot index nested list with a multidimensional index.
        # so need to do some fancy indexing
        current_feature = np.array(feature_matrix)[:,j]

        # Val is the feature value for one observations, if all observations
        # have this same val, we know that the feature is degenerate
        val = current_feature[0]
        equal_range =0

        for i in range (0,len(current_feature)):
            if current_feature[i] == val:
                equal_range +=1
            else:
                break

        # If equal range is not equal to the length, we know that every observation
        # didn't have the same value for this feature, therefore it is not degenerate.
        # And so we add it to the list of non-degenerate indices and also add this
        # indice to the feature_array to keep track of the list of important features
        if equal_range != len(current_feature):
            non_degenerate_feature_indices.append(j)
            feature_array.append(j)

    return non_degenerate_feature_indices
