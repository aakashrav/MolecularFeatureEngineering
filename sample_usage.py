#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module that extracts the set of actives and inactives for a specific dataset and returns
a matrix with the set of features for each active and inactive along with a 1 or 0 depending
on their activity w.r.t. the target
"""

__author__="Aakash Ravi"

import os
import json
import numpy as np
from FeatureExtractor import obtain_molecules
from FeatureExtractor import molecule_feature_matrix
from FeatureExtractor import feature_imputer
from DetectCorrelations import correlation_identifier

DATASET_NUMBER = '03'
DATASET_DIRECTORY = '5HT1A_Agonist/'
FRAGMENTS_DIRECTORY = 'fragments/'
DESCRIPTORS_DIRECTORY = 'descriptors/'
ELKI_CSV_FILE = '/Users/AakashRavi/Desktop/Aakash/Education/ChemicalInformatics/ELKI/subclutest.csv'

def main():
    dataset_file = os.path.dirname(os.path.realpath(__file__)) + \
    '/' + DATASET_DIRECTORY + DATASET_NUMBER + '/'

    fragments_file = os.path.dirname(os.path.realpath(__file__)) + \
    '/' + FRAGMENTS_DIRECTORY + '5HT1A_Agonist-fragments.json'

    descriptors_file = os.path.dirname(os.path.realpath(__file__)) + \
    '/' + DESCRIPTORS_DIRECTORY + '5HT1A_Agonist-fragments.csv'

    # First fetch the desired set of actives and inactives
    actives = obtain_molecules.get_actives(dataset_file + 'known-ligands.smi')
    inactives = obtain_molecules.get_inactives(dataset_file + 'known-decoys.smi')

    feature_array = []
    feature_array_inactives = []

    # Fetch the features for all our actives, fully imputed
    actives_feature_matrix = molecule_feature_matrix.retrieve_features(descriptors_file, \
        fragments_file ,actives[0:3], 1, feature_array)

    # Print the active feature matrix
    print("Feature matrix for the actives:")
    for i in range(0,len(actives_feature_matrix)):
            print(actives_feature_matrix[i])

    # Fetch the features for all our inactives, fully imputed
    inactives_feature_matrix = molecule_feature_matrix.retrieve_features(descriptors_file,\
        fragments_file, inactives[0:3], 0, feature_array_inactives)

    # Print the inactive feature matrix
    print("Feature matrix for the inactives:")
    for i in range(0,len(inactives_feature_matrix)):
            print(inactives_feature_matrix[i])
    
    key_indices_actives = []
    # Compute the key features for the active reactants
    # Splice the matrix so we don't count the final characteristic feature
    # (1 or 0 ). We limit the amount of features we want to be the top 100.
    # This amount can be changed to fit our needs.

    # key_indices_actives = correlation_identifier.identify_uniform_features( \
    #     np.array(actives_feature_matrix)[:,0:len(actives_feature_matrix[0])-1], 100)

    key_indices_actives = correlation_identifier.identify_correlated_features( \
        np.array(actives_feature_matrix)[:,0:len(actives_feature_matrix[0])-1], 100)

    print("Key features for the active features:")
    print(key_indices_actives)

    non_correlated_active_matrix = np.array(actives_feature_matrix)[:,key_indices_actives]
    print("Non correlated feature matrix for the actives")
    for i in range(0,len(non_correlated_active_matrix)):
            print(non_correlated_active_matrix[i])

    with open(ELKI_CSV_FILE,'w') as f_handle:
        print(len(non_correlated_active_matrix))
        print(len(non_correlated_active_matrix[0]))  
        np.savetxt(f_handle, non_correlated_active_matrix, delimiter=",", fmt="%f")     
    
    # key_indices_inactives = []
    # # No inactives are in the dataset, so can't really do anything with them
    # if inactives_feature_matrix != []:
    # # Perform the same procedure for the inactive reactants
    #     key_indices_inactives = correlation_identifier.identify_uniform_features( \
    #         np.array(inactives_feature_matrix)[:,0:len(actives_feature_matrix[0])-1], 100)

    # # print("Key features for the inactive features:")
    # # print(key_indices_inactives)

    # Take union of the key features to obtain our full set of key features
    # full_key_feature_set = np.union1d(key_indices_actives, key_indices_inactives)

    # print("Full set of key feature indices: ")
    # print(full_key_feature_set)
    
    # for i in range(0, len(actives_feature_matrix)):
    #     for j in range(0, len(actives_feature_matrix[0])):
    #         print(actives_feature_matrix[i][j])

    # correlation_identifier.get_top_features(actives_feature_matrix, 5)


if __name__ == '__main__':
    main()
