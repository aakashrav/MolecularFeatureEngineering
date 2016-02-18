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
DATASET_DIRECTORY = '/../SDFActivesInactivesDataset/'
FRAGMENTS_DIRECTORY = '/../fragments/'
DESCRIPTORS_DIRECTORY = '/../descriptors/'
ELKI_CSV_FILE = '/Users/AakashRavi/Desktop/Aakash/Education/ChemicalInformatics/ELKI/subclutest.csv'
REAL=0


# DATASET_NUMBER = '03'
# DATASET_DIRECTORY = '/../SDFActivesInactivesDataset/'
# FRAGMENTS_DIRECTORY = '../'
# DESCRIPTORS_DIRECTORY = '../'
# ELKI_CSV_FILE = '/Users/AakashRavi/Desktop/Aakash/Education/ChemicalInformatics/ELKI/subclutest.csv'
# REAL=1

def main():

    ####------ NON-SDF VERSION ----#
    # dataset_file = os.path.dirname(os.path.realpath(__file__)) + \
    # '/' + DATASET_DIRECTORY + DATASET_NUMBER + '/'

    # fragments_file = os.path.dirname(os.path.realpath(__file__)) + \
    # '/' + FRAGMENTS_DIRECTORY + '5HT1A_Agonist-fragments.json'

    # descriptors_file = os.path.dirname(os.path.realpath(__file__)) + \
    # '/' + DESCRIPTORS_DIRECTORY + '5HT1A_Agonist-fragments.csv'

    #  # First fetch the desired set of actives and inactives
    # actives = obtain_molecules.get_molecules(dataset_file + 'known-ligands.smi')
    # inactives = obtain_molecules.get_molecules(dataset_file + 'known-decoys.smi')

    #  # Fetch the features for all our actives and inactives, fully imputed
    # if REAL==0:
    #     molecule_feature_matrix.retrieve_features(descriptors_file, \
    #         fragments_file ,actives[0:3], inactives, output_details=False)
    # else:
    #     molecule_feature_matrix.retrieve_features(descriptors_file, \
    #         fragments_file ,actives, inactives, output_details=False)

    ####------ NON-SDF VERSION ----#

    ####------ SDF VERSION ---- ##

    actives_dataset_file = os.path.dirname(os.path.realpath(__file__)) + \
        DATASET_DIRECTORY + "Hydrogen-Bonds_2/actives_1"

    inactives_dataset_file = os.path.dirname(os.path.realpath(__file__)) + \
        DATASET_DIRECTORY + "Hydrogen-Bonds_2/inactives"

    descriptors_file = os.path.dirname(os.path.realpath(__file__)) + \
    '/' + DESCRIPTORS_DIRECTORY + 'features.csv'

    fragments_file = os.path.dirname(os.path.realpath(__file__)) + \
     '/' + FRAGMENTS_DIRECTORY + 'SDF_Fragments.json'

    # Fetch the SDF Actives and Inactives List
    actives = obtain_molecules.get_sdf_molecules(actives_dataset_file)
    inactives = obtain_molecules.get_sdf_molecules(inactives_dataset_file)
    
    # Fetch the features for all our actives and inactives, fully imputed
    if REAL==0:
        molecule_feature_matrix.retrieve_sdf_features(descriptors_file, \
            fragments_file ,actives[0:3], inactives, output_details=False)
    else:
        molecule_feature_matrix.retrieve_sdf_features(descriptors_file, \
            fragments_file ,actives, inactives, output_details=False)

    ####------ SDF VERSION ----##
    
    # # Print the active feature matrix
    # print("Feature matrix for the actives:")
    # print(actives_feature_matrix.shape)
    # for i in range(0,len(actives_feature_matrix)):
    #     for j in range(0, len(actives_feature_matrix[0])):
    #         if actives_feature_matrix[i][j] == np.nan:
    #             print(actives_feature_matrix[i][j])
    #         # print(actives_feature_matrix[i])

if __name__ == '__main__':
    main()
