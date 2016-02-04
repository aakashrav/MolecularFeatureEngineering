#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this module is to provide functions to retreive features of fragments 
that are contained in the molecule, and impute the feature vectors so that they are not sparse, 
i.e. there are no missing values
"""

__author__="Aakash Ravi"

import os
import json
import numpy as np
import re
import time
from datetime import datetime

NON_FRAGMENT_VERSION = 0
DESCRIPTOR_TO_RAM = 0


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def compute_feature_median(non_imputed_feature_matrix, descriptor_indice, molecule_keys):
    global_descriptor_average_array = []
    # Get the unique molecule keys
    unique_molecules = np.unique(molecule_keys)
    # Go from range 0 to maximum molecular index
    for molecule_i in unique_molecules:
        # Get indices of all other fragments that belong to the same molecule
        molecule_fragment_indices = np.where(molecule_keys == molecule_i)
        fragments_for_molecule = non_imputed_feature_matrix[molecule_fragment_indices,:]
        # Get the values for this specific descriptor
        # print(fragments_for_molecule)
        descriptor_values = fragments_for_molecule[:,descriptor_indice]
        # If there doesn't exist any nan descriptor values for fragments of this molecule
        # we add it to the global average array.
        if (np.isnan(descriptor_values).any()):
            continue
        else:
            global_descriptor_average_array.append(np.mean(descriptor_values))

    # If descriptor is defined for no molecule, it is a degenerate descriptor,
    # So we return np.nan, else we return the median of the averages
    if (len(global_descriptor_average_array) == 0):
        return np.nan
    else:
        return np.median(global_descriptor_average_array)

def secondary_feature_impute(inactives_feature_matrix, degenerate_features, \
    global_median_cache):
    
    if (inactives_feature_matrix is None):
        print("Empty matrix; no secondary imputation done, continuing on..")
        return None

    # First remove all degenerate features calculated from the standard impute
    np.delete(inactives_feature_matrix, degenerate_features, 1)

    for fragment in range(0, len(inactives_feature_matrix)-1):

        nan_descriptors = np.where(np.isnan(non_imputed_feature_matrix[fragment]) == True)
        
        # For any non-numerical descriptors we find, we just set it to the pre-computed
        # global median for that descriptor
        for j in nan_descriptors:
            inactives_feature_matrix[fragment,j] = global_median_cache[0,j]

    return inactives_feature_matrix
    

def standard_feature_impute(non_imputed_feature_matrix):
    
    if (non_imputed_feature_matrix is None):
        print("Empty matrix; no standard imputation done, continuing on..")
        return None

    imputed_feature_matrix = np.copy(non_imputed_feature_matrix)

    # Keep track of indicies corresponding to the molecule of each fragment- just the first column
    molecule_keys = non_imputed_feature_matrix[:,0]
    # Keep track of degenerate features
    degenerate_features = []

    # Cache globally imputed descriptors, initialize the array with NaN
    global_median_cache = np.empty([1,non_imputed_feature_matrix.shape[1]], dtype=np.float)

    for descriptor in range(0,len(global_median_cache)-1):
        global_descriptor_median = compute_feature_median(non_imputed_feature_matrix, descriptor,
                molecule_keys)
        if (global_descriptor_median==np.nan):
            np.delete(imputed_feature_matrix, descriptor, 1)
            np.delete(global_median_cache, descriptor,0)
            degenerate_features.append(descriptor)
        else:
            global_median_cache[descriptor] = global_descriptor_median
    
    # Now loop through the fragments and impute
    for fragment in range(0, len(non_imputed_feature_matrix)-1):

        # Obtain all descriptors that have non-numerical values for this fragment
        nan_descriptors = np.where(np.isnan(non_imputed_feature_matrix[fragment]) == True)

        for j in nan_descriptors:
            imputed_feature_matrix[fragment, j] = global_median_cache[0,j]

    return [imputed_feature_matrix, global_median_cache, degenerate_features]


def load_matrix(descriptor_file, molecules_to_fragments_file,
    molecule_smiles, output_details=1,
    descriptors_map=None, descriptors=None):

    with open(molecules_to_fragments_file, 'r') as input_stream:

        molecules_to_fragments = json.load(input_stream)

        # Keep a list of already found fragments so we don't
        # redundantly retreive the same fragments
        found_fragments = []
        
        if (descriptors is not None):
            non_imputed_feature_matrix = np.empty((0, descriptors.shape[1]), np.float)
        else:
            non_imputed_feature_matrix = None

        # Add all fragment data for each molecule in the array
        # 'molecule_smiles', and return a matrix of found values
        for molecule_index in range(0, len(molecule_smiles)):

            # Sometimes the desired SMILE may not have any fragment data, so we just continue
            # And notify the user that no fragments were found for this SMILES
            try:
                molecule = molecules_to_fragments[molecule_smiles[molecule_index]]
                if output_details:
                    print("Starting analysis of {} ...".format(molecule_smiles[molecule_index]))
            except KeyError:
                print("Fragments not available for molecule ", molecule_smiles[molecule_index])
                continue

            fragments = molecule['fragments']
            time_start = time.time()

            with open(descriptor_file,'r') as input_stream:
                for f in fragments:
                    header = input_stream.readline().rstrip().split(',')
                    
                    # Use the descriptors file on disk
                    if (descriptors_map is None):
                        for line in input_stream:
                            line = line.rstrip().split(',')
                            name = line[0][1:-1]
                            
                            if (non_imputed_feature_matrix is None):
                                non_imputed_feature_matrix = np.empty((0, len(line)), np.float)

                            # We found a desired fragment of the molecule
                            if (name in fragments) and (name not in found_fragments):
                                print("Found fragment: " + name)
                                found_fragments.append(name)

                                # First append the molecule index, so we know which molecule
                                # the fragment came from in further stages of the pipeline
                                molecule_descriptor_row = np.empty((1, len(line)), np.float)
                                molecule_descriptor_row[0] = molecule_index

                                # Next we just copy the newly obtained features into our feature matrix
                                # If the feature is not a number, we explicitly input np.nan
                                # Need the for loop since we may have NaNs
                                for j in range(1, molecule_descriptor_row.shape[1]):
                                    try:
                                        molecule_descriptor_row[0,j] = float(line[j])
                                    except ValueError:
                                        molecule_descriptor_row[0,j] = np.nan

                                # Finally append the row to our matrix
                                non_imputed_feature_matrix = np.vstack((non_imputed_feature_matrix, \
                                                                molecule_descriptor_row))

                    # Use the descriptors dictionary in RAM
                    else:
                        # If we already found the fragment, we continue on; will save us time and space
                        if f in found_fragments:
                            continue;
                        try:
                            ix_f = descriptors_map[f]
                            # Append the current fragment to the feature matrix
                            # non_imputed_feature_matrix = np.append(non_imputed_feature_matrix,
                            #                                     [np.append(descriptors[ix_f, :], 0)], axis=0)
                            non_imputed_feature_matrix = np.vstack((non_imputed_feature_matrix,
                                                                descriptors[ix_f, :]))
                            found_fragments.append(f)
                        except KeyError:
                            # print("Fragment: " + f + " for molecule " + molecule_smiles[molecule_index] + " NOT found")
                            continue

                # time_end = time.time()s
                #print('Time for getting features of a molecule: {0}'.format(time_end-time_start))

    return non_imputed_feature_matrix


def read_descriptor_file(descriptor_file_name):
    print("[{}] Reading descriptors file...".format(str(datetime.now())))
    # Read in fragments descriptors into an NP array
    descriptors = None
    # Store mapping between SMILES and indeces in a dictionary
    descriptors_smiles_to_ix = {}
    with open(descriptor_file_name, 'r') as descriptor_file:
        #Header serves to find out the number of descriptors
        header = descriptor_file.readline().rstrip().split(',')
        descriptors = np.empty((0, len(header)-1), np.float)
        #Adding rows into the NP array one by one is expensive. Therefore we read rows
        #int a python list in batches and regularly flush them into the NP array
        aux_descriptors = []
        ix = 0
        for line in descriptor_file:
            line_split = line.rstrip().split(',')
            descriptors_smiles_to_ix[line_split[0].strip('"\'')] = ix
            #descriptors_all_values = np.append(descriptors_all_values, [np.array(line_split[1:])], axis=0)
            aux_descriptors.append([float(x) if isfloat(x) else float('nan') for x in line_split[1:]])
            ix += 1
            if ix % 1000 == 0:
                descriptors = np.vstack((descriptors, np.asarray(aux_descriptors)))
                del aux_descriptors[:]
                break
            #print '{0}.'.format(ix)
        if len(aux_descriptors) > 0:
            descriptors = np.vstack((descriptors, np.asarray(aux_descriptors)))
            del aux_descriptors[:]

    return [descriptors_smiles_to_ix, descriptors]


def retrieve_features(descriptor_file, molecules_to_fragments_file,
    active_molecule_smiles, inactive_molecule_smiles, feature_array,
    output_details = False):

    """Parameter 'molecule_smiles' is an array of molecule smiles we want to fetch the features for,
    the function returns a matrix of *average* features of the *fragments* for each smile in the array.
    The function also requires an active_flag, which will then trigger an automatic adding of a '1' feature
    to each molecule, indicating that it is active. Similarly with '0' and inactive.
    Finally, the feature_array is an array of features that the function will deem relevant.
    We may have some degenerate features (i.e. where all the values are 0) so the function will
    remove those features from the array and return only the non degenerate ones."""
    
    descriptors_map = None
    descriptors = None

    if DESCRIPTOR_TO_RAM:
        # First load the descriptors into RAM
        descriptors_map, descriptors = read_descriptor_file(descriptor_file)

    # Load the non-imputed actives feature matrix
    non_imputed_feature_matrix = load_matrix(descriptor_file, molecules_to_fragments_file,
        active_molecule_smiles, output_details=1, 
        descriptors_map = descriptors_map, descriptors=descriptors)

    # Impute the actives, keeping track of degenerate features and any global medians
    actives_feature_matrix,global_median_cache,degenerate_features = standard_feature_impute(non_imputed_feature_matrix)

    # Load the non-imputed actives feature matrix
    # TODO: LOAD INACTIVES ON THE FLY..
    inactives_feature_matrix = []
    inactives_feature_matrix = load_matrix(descriptor_file, molecules_to_fragments_file, \
        inactive_molecule_smiles, output_details=1, \
        descriptors_map = descriptors_map, descriptors= descriptors)

    # Now, use the results from the actives imputation to impute the inactives
    secondary_feature_impute(inactives_feature_matrix, degenerate_features,
        global_median_cache)
    
    # Remove existing statistics file
    if (os.path.isfile("molecule_statistics")):
        os.remove("molecule_statistics")

    with open("molecule_statistics", 'a') as f_handle:
        f_handle.write("Degenerate features:\n")
        np.savetxt(f_handle, degenerate_features, delimiter=',')

        f_handle.write("Global medians:\n")
        np.savetxt(f_handle, global_median_cache, delimiter=',')

    return [actives_feature_matrix, inactives_feature_matrix]
    

