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
import molecular_clusters
import cluster_analysis

DATASET_NUMBER = '03'
DATASET_DIRECTORY = '/../SDFActivesInactivesDataset/'
FRAGMENTS_DIRECTORY = '/../fragments/'
DESCRIPTORS_DIRECTORY = '/../descriptors/'

def main():

    actives_dataset_file = os.path.dirname(os.path.realpath(__file__)) + \
        DATASET_DIRECTORY + "IC-50_8/actives"

    inactives_dataset_file = os.path.dirname(os.path.realpath(__file__)) + \
        DATASET_DIRECTORY + "IC-50_8/inactives"

    descriptors_file = os.path.dirname(os.path.realpath(__file__)) + \
    '/' + DESCRIPTORS_DIRECTORY + 'features.csv'

    fragments_file = os.path.dirname(os.path.realpath(__file__)) + \
     '/' + FRAGMENTS_DIRECTORY + 'SDF_Fragments.json'

    # Fetch the SDF Actives and Inactives List
    actives = obtain_molecules.get_sdf_molecules(actives_dataset_file)
    inactives = obtain_molecules.get_sdf_molecules(inactives_dataset_file)
    
    print "Starting molecular feature matrix creation.. "
    # Retrieve the molecular feature matrix corresponding to our dataset and 
    # flush it to file
    molecule_feature_matrix.retrieve_sdf_features(descriptors_file, \
        fragments_file ,actives, inactives, output_details=False)
    print "Finished molecular feature matrix creation.. "

    print "Starting search of molecular clusters"
    # Find the clusters using ELKI
    molecular_clusters.main()
    print "Finished search of molecular clusters"

    print "Starting analysis and pruning of found clusters"
    # Analyze the clusters and output the most pure and diverse ones
    cluster_analysis.main()
    print "Finished analysis and pruning of clusters! Clusters available in data directory"

if __name__ == '__main__':
    main()
