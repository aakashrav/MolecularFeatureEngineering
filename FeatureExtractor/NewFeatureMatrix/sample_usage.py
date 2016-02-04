#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module that extracts the set of actives and inactives for a specific dataset and returns
a matrix with the set of features for each active and inactive along with a 1 or 0 depending
on their activity w.r.t. the target
"""

__author__="Aakash Ravi"

import sys
sys.path.append('..')
import os
import json
import numpy as np
import obtain_molecules
import molecule_feature_matrix

DATASET_NUMBER = '03'
DATASET_DIRECTORY = '/../../5HT1A_Agonist/'
FRAGMENTS_DIRECTORY = '/../../fragments/'
DESCRIPTORS_DIRECTORY = '/../../descriptors/'
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
    actives_feature_matrix, inactives_feature_matrix = molecule_feature_matrix.retrieve_features(descriptors_file, \
        fragments_file ,actives[0:3], inactives,feature_array, output_details=False)

    # Print the active feature matrix
    print("Feature matrix for the actives:")
    print(actives_feature_matrix.shape)
    for i in range(0,len(actives_feature_matrix)):
        for j in range(0, len(actives_feature_matrix[0])):
            if actives_feature_matrix[i][j] == np.nan:
                print(actives_feature_matrix[i][j])
            # print(actives_feature_matrix[i])

if __name__ == '__main__':
    main()
