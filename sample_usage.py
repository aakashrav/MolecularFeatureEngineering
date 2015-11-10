#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module that extracts the set of actives and inactives for a specific dataset and returns
a matrix with the set of features for each active and inactive along with a 1 or 0 depending
on their activity w.r.t. the target
"""

__author__="Aakash Ravi"

import os
import json
from FeatureExtractor import obtain_molecules
from FeatureExtractor import molecule_feature_matrix

DATASET_NUMBER = '05'
DATASET_DIRECTORY = '5HT1A_Agonist/'
FRAGMENTS_DIRECTORY = 'fragments/'
DESCRIPTORS_DIRECTORY = 'descriptors/'

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

    # print("Actives: %s\n", actives)
    # print("Inactives: %s\n", inactives)

    # Fetch the features for all our actives, fully imputed
    actives_feature_matrix = molecule_feature_matrix.retrieve_features(descriptors_file, \
        fragments_file ,actives[1:5], 1)

    # Fetch the features for all our inactives, fully imputed
    inactives_feature_matrix = molecule_feature_matrix.retrieve_features(descriptors_file,\
        fragments_file, inactives[1:5], 0)

    # Print the active feature matrix
    print("Feature matrix for the actives:")
    for i in range(0,len(actives_feature_matrix)):
            print(actives_feature_matrix[i])

    # Print the inactive feature matrix
    print("Feature matrix for the inactives:")
    for i in range(0,len(inactives_feature_matrix)):
            print(inactives_feature_matrix[i])

if __name__ == '__main__':
    main()
