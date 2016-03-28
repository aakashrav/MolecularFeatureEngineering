#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this module is to provide functions to retrieve features of fragments 
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
import neighborhood_extractor
import pickle
import csv
import config
import shutil

FLUSH_BUFFER_SIZE = config.FLUSH_BUFFER_SIZE
DESCRIPTOR_TO_RAM = config.DESCRIPTOR_TO_RAM
NUM_FEATURES = config.NUM_FEATURES
COVARIANCE_THRESHOLD = config.COVARIANCE_THRESHOLD
DATA_DIRECTORY = config.DATA_DIRECTORY
DEBUG = config.DEBUG

fragment_number_name_mapping = {}
actives_fragment_molecule_mapping = {}
inactives_fragment_molecule_mapping = {}


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def _compute_feature_median(non_imputed_feature_matrix, descriptor_indice, molecule_keys):
    global_descriptor_average_array = []
    # Get the unique molecule keys
    unique_molecules = np.unique(molecule_keys)
    # Go from range 0 to maximum molecular index
    for molecule_i in unique_molecules:
        # Get indices of all other fragments that belong to the same molecule
        molecule_fragment_indices = np.where(molecule_keys == molecule_i)
        fragments_for_molecule = non_imputed_feature_matrix[molecule_fragment_indices]
        # Get the values for this specific descriptor
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

def _actives_feature_impute(feature_matrix, descriptor_matrix):
    
    if (feature_matrix is None):
        print("Empty matrix; no standard imputation done, continuing on..")
        return None

    # Keep track of indices corresponding to the molecule of each fragment- just the first column
    molecule_keys = feature_matrix[1:,0]
    # Delete the molecule_index feature, which has already been placed in molecule_keys for
    # further usage
    feature_matrix = np.delete(feature_matrix, 0, 1)
    # Keep track of all the features for further usage
    all_features = np.arange(feature_matrix.shape[1])
    # Keep track of degenerate features
    degenerate_features = []

    # Cache globally imputed descriptors
    global_median_cache = np.empty([1,feature_matrix.shape[1]], dtype=np.float)

    for descriptor in range(0,feature_matrix.shape[1]):
        global_descriptor_median = _compute_feature_median(feature_matrix, descriptor,
                molecule_keys)
        if (np.isnan(global_descriptor_median)):
            degenerate_features.append(descriptor)
        global_median_cache[0,descriptor] = global_descriptor_median
    
    # Recompute the significant features before beginning imputation
    if DESCRIPTOR_TO_RAM:
        neighborhood_extractor.extract_features(NUM_FEATURES,descriptor_matrix,COVARIANCE_THRESHOLD)

    print "Actives imputation: starting out with %d features" % (feature_matrix.shape[1])
    significant_features = np.genfromtxt(os.path.join(DATA_DIRECTORY,'significant_features'),delimiter=',')
    redundant_features = [i for i in all_features if i not in significant_features]

    feature_matrix = np.delete(feature_matrix, redundant_features, 1)
    print "Actives imputation: removed %d features with constant features and covariance neighborhoods, now have %d features, with the NUM_FEATURES \
    parameters set to %d" % (len(redundant_features), len(significant_features), NUM_FEATURES)
    # Remove the redundant features from the degenerate features, since they have already
    # been removed
    for feature in redundant_features:
        try:
            degenerate_features = np.delete(degenerate_features,feature,1)
        except IndexError:
            continue
    
    # Now loop through the fragments and impute
    for fragment in range(1, len(feature_matrix)):
        # Obtain all descriptors that have non-numerical values for this fragment
        nan_descriptors = np.where(np.isnan(feature_matrix[fragment]) == True)
        for j in nan_descriptors:
            feature_matrix[fragment, j] = global_median_cache[0,j]

    # First remove all degenerate features
    non_degenerate_feature_matrix_one = np.delete(feature_matrix, degenerate_features, 1)
    print "Actives imputation: removed degenerate features, now have %d features" % (non_degenerate_feature_matrix_one.shape[1])

    # Remove existing dataset files and flush new actives data
    with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'w+') as f_handle:
        np.savetxt(f_handle, non_degenerate_feature_matrix_one[1:,:], delimiter=',',fmt='%5.5f')

    # Update degenerate features
    used_features = [ x-1 for x in non_degenerate_feature_matrix_one[0]]
    degenerate_features = [i for i in all_features if i not in used_features]
    
    return [global_median_cache, degenerate_features, used_features]

def _read_descriptor_file(descriptor_file_name):
    print("[{}] Reading descriptors file...".format(str(datetime.now())))
    # Read in fragments descriptors into an NP array
    descriptors = None
    # Store mapping between SMILES and indeces in a dictionary
    descriptors_smiles_to_ix = {}
    with open(descriptor_file_name, 'r') as descriptor_file:
        #Header serves to find out the number of descriptors
        header = descriptor_file.readline().rstrip().split(',')

        if DEBUG:
            with open(os.path.join(DATA_DIRECTORY,'all_descriptors.csv'),'wb+') as f_handle:
                csv.writer(f_handle).writerow(header)

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

def _flush_metadata(global_median_cache, used_features):

    # Flush new metadata
    # Save the global median cache structure as CSV
    with open(os.path.join(DATA_DIRECTORY,"global_median_cache.csv"), 'wb+') as f_handle:
        np.savetxt(f_handle, global_median_cache, delimiter=',', fmt='%5.5f')

    # We create a mapping between the indices of all the descriptors
    # to indices of the new descriptors that have no degenerate features
    old_new_descriptor_mapping = {}
    for i in range(len(used_features)):
        old_new_descriptor_mapping[i] = used_features[i]
    with open(os.path.join(DATA_DIRECTORY,"used_features.pkl"),'wb+') as f_handle:
        pickle.dump(old_new_descriptor_mapping, f_handle, pickle.HIGHEST_PROTOCOL)
    
    # Save the fragment number to name mapping dictionaries in a pickle file
    with open(os.path.join(DATA_DIRECTORY,"fragment_number_name_mapping.pkl"), 'wb+') as f_handle:
        pickle.dump(fragment_number_name_mapping,f_handle, pickle.HIGHEST_PROTOCOL)
    # print(fragment_number_name_mapping)

    # Save the fragments to molecules mapping(s) dictionar(ies) in a pickle file
    with open(os.path.join(DATA_DIRECTORY,"actives_fragment_molecule_mapping.pkl"), 'wb+') as f_handle:
        pickle.dump(actives_fragment_molecule_mapping, f_handle, pickle.HIGHEST_PROTOCOL)
    # print(actives_fragment_molecule_mapping)

    # Save the fragments to molecules mapping(s) dictionar(ies) in a pickle file
    with open(os.path.join(DATA_DIRECTORY,"inactives_fragment_molecule_mapping.pkl"), 'wb+') as f_handle:
        pickle.dump(inactives_fragment_molecule_mapping, f_handle, pickle.HIGHEST_PROTOCOL)
    # print(inactives_fragment_molecule_mapping)

def _load_matrix_sdf(descriptor_file, molecules_to_fragments_file,
    molecule_sdfs, output_details=0,
    descriptors_map=None, descriptors=None):

    with open(molecules_to_fragments_file, 'r') as input_stream:

        molecules_to_fragments = json.load(input_stream)

        # Keep a list of already found fragments for metadata storage
        found_fragments = []
        
        if (descriptors is not None):
            non_imputed_feature_matrix = np.empty((0, descriptors.shape[1]+1), np.float)

            # Append the descriptor number columns
            descriptor_numbers = np.arange(descriptors.shape[1]+1).reshape((1, descriptors.shape[1]+1))
            non_imputed_feature_matrix = np.append(non_imputed_feature_matrix,
                                                        descriptor_numbers, axis=0)

        else:
            non_imputed_feature_matrix = None

        FRAGMENT_COUNT = 0

        # Add all fragment data for each molecule in the array
        # 'molecule_sfds', and return a matrix of found values
        for molecule_index in range(0, len(molecule_sdfs)):

            # Sometimes the desired SMILE may not have any fragment data, so we just continue
            # And notify the user that no fragments were found for this SMILES
            try:
                full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments 
                            if molecule["name"] == molecule_sdfs[molecule_index]]
                # First index is actual fragments, since there 
                # can exist only one key value pair for the molecule and its fragments
                full_fragments = full_fragments[0]
                fragments = [fragment["smiles"] for fragment in full_fragments]

                # if output_details:
                #     print("Starting analysis of {} ...".format(molecule_sdfs[molecule_index]))
            except KeyError:
                print("Fragments not available for molecule ", molecule_sdfs[molecule_index])
                continue

            with open(descriptor_file,'r') as input_stream:
                # Use the descriptors file on disk
                if (descriptors_map is None):
                    header = input_stream.readline().rstrip().split(',')

                    if DEBUG:
                        with open(os.path.join(DATA_DIRECTORY,'all_descriptors.csv'),'wb+') as f_handle:
                            csv.writer(f_handle).writerow(header)

                    for line in input_stream:
                        line = line.rstrip().split(',')
                        name = line[0][1:-1]
                            
                        if (non_imputed_feature_matrix is None):
                            non_imputed_feature_matrix = np.empty((0, len(line)), np.float)
                            # Append the descriptor number columns
                            descriptor_numbers = np.arange(len(line))
                            non_imputed_feature_matrix = np.vstack((non_imputed_feature_matrix, \
                                descriptor_numbers))
                        
                        if (name in fragments) and (name in found_fragments):
                            fragment_number_name_mapping[FRAGMENT_COUNT] = name
                            actives_fragment_molecule_mapping[name].append(molecule_index)

                        elif (name in fragments) and (name not in found_fragments):
                            found_fragments.append(name)
                            fragment_number_name_mapping[FRAGMENT_COUNT] = name
                            actives_fragment_molecule_mapping[name] = [molecule_index]
                        else:
                            continue

                        # First append the molecule index, so we know which molecule
                        # the fragment came from in further stages of the pipeline
                        molecule_descriptor_row = np.empty((1, len(line)), np.float)
                        molecule_descriptor_row[0,0] = molecule_index

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

                        FRAGMENT_COUNT+=1

                # Use the descriptors dictionary in RAM
                else:
                    for f in fragments:
                        # If we already found the fragment, we continue on; will save us time and space
                        if f in found_fragments:
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            actives_fragment_molecule_mapping[f].append(molecule_index)
                        else:
                            found_fragments.append(f)
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            actives_fragment_molecule_mapping[f] = [molecule_index]
                        try:
                            ix_f = descriptors_map[f]
                        
                            non_imputed_feature_matrix = np.append(non_imputed_feature_matrix,
                                                        [np.insert(descriptors[ix_f], 0, molecule_index)], axis=0)
                            FRAGMENT_COUNT+=1
                        except KeyError:
                            print("Key error")
                            continue
    
    print("Number of active fragments: %d\n" % FRAGMENT_COUNT)
    print("Number of active molecules: %d\n" % len(molecule_sdfs))

    return [non_imputed_feature_matrix, FRAGMENT_COUNT]

def _inactives_load_impute_sdf(degenerate_features, \
    global_median_cache, descriptor_file, molecules_to_fragments_file,
    molecule_sdfs, FRAGMENT_COUNT, output_details=0,
    descriptors_map=None, descriptors=None):

    # For debugging purposes
    OLD_FRAGMENT_COUNT = FRAGMENT_COUNT

    with open(molecules_to_fragments_file, 'r') as input_stream:

        molecules_to_fragments = json.load(input_stream)

        # Keep a list of already found fragments for metadata storage
        found_fragments = []
        
        # Create a temporary holding space for the fragments, and periodically flush to disk
        inactives_feature_matrix = np.empty([FLUSH_BUFFER_SIZE, global_median_cache.shape[1]], np.float)

        FLUSH_COUNT = 0

        # Add all fragment data for each molecule in the array
        # 'molecule_sdfs', and return a matrix of found values
        for molecule_index in range(0, len(molecule_sdfs)):

            # Sometimes the desired SMILE may not have any fragment data, so we just continue
            # And notify the user that no fragments were found for this SMILES
            try:
                full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments 
                            if molecule["name"] == molecule_sdfs[molecule_index]]
                # First index is actual fragments, since there 
                # can exist only one key value pair for the molecule and its fragments
                full_fragments = full_fragments[0]
                fragments = [fragment["smiles"] for fragment in full_fragments]

                # if output_details:
                    # print("Starting analysis of {} ...".format(molecule_sdfs[molecule_index]))
            except KeyError:
                print("Fragments not available for molecule ", molecule_sdfs[molecule_index])
                continue

            with open(descriptor_file,'r') as input_stream:
                # Use the descriptors file on disk
                if (descriptors_map is None):
                    header = input_stream.readline().rstrip().split(',')
                    
                    for line in input_stream:
                        line = line.rstrip().split(',')
                        name = line[0][1:-1]
                        
                        if (name in fragments) and (name in found_fragments):
                            fragment_number_name_mapping[FRAGMENT_COUNT] = name
                            inactives_fragment_molecule_mapping[name].append(molecule_index)

                        # The fragment found is not already added to the matrix
                        elif (name in fragments) and (name not in found_fragments):
                            found_fragments.append(name)
                            fragment_number_name_mapping[FRAGMENT_COUNT] = name
                            inactives_fragment_molecule_mapping[name] = [molecule_index]
                        else:
                            continue

                        molecule_descriptor_row = np.empty([1, len(line)-1], np.float)

                        # Next we just copy the newly obtained features into our feature matrix
                        # If we encounter a nan, either we impute it with the global median cache from the actives
                        # Or we delete it as we consider it a degenerate feature
                        for j in range(1, len(line)):
                            try:
                                molecule_descriptor_row[0,j-1] = float(line[j])
                            except ValueError:
                                if (j-1 in degenerate_features):
                                    molecule_descriptor_row[0,j-1] = np.nan
                                else:
                                    molecule_descriptor_row[0,j-1] = global_median_cache[0,j-1]

                        if (FLUSH_COUNT == FLUSH_BUFFER_SIZE):
                            # Flush only the non degenerate descriptors to file
                            all_descriptors = np.arange(global_median_cache.shape[1])
                            non_degenerate_descriptors = np.delete(all_descriptors, degenerate_features)
                            with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'a') as f_handle:
                                np.savetxt(f_handle, inactives_feature_matrix[:,non_degenerate_descriptors], delimiter=',', fmt='%5.5f')
                            FLUSH_COUNT=0
                            
                        inactives_feature_matrix[FLUSH_COUNT]= molecule_descriptor_row
                        FLUSH_COUNT+=1
                        FRAGMENT_COUNT+=1

                # Use the descriptors dictionary in RAM
                else:
                    for f in fragments:
                        # If we already found the fragment, we continue on; will save us time and space
                        if f in found_fragments:
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            inactives_fragment_molecule_mapping[f].append(molecule_index)
                        else:
                            found_fragments.append(f)
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            inactives_fragment_molecule_mapping[f] = [molecule_index]
                        try:
                            ix_f = descriptors_map[f]
                            current_fragment = descriptors[ix_f]

                            # Obtain all descriptors that have non-numerical values for this fragment
                            nan_descriptors = np.where(np.isnan(current_fragment) == True)

                            for j in nan_descriptors:
                                current_fragment[j] = global_median_cache[0,j]

                            if (FLUSH_COUNT == FLUSH_BUFFER_SIZE):
                                # Flush only the non-degenerate descriptors to file
                                all_descriptors = np.arange(global_median_cache.shape[1])
                                non_degenerate_descriptors = np.delete(all_descriptors, degenerate_features)
                                with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'a') as f_handle:
                                    np.savetxt(f_handle, inactives_feature_matrix[:,non_degenerate_descriptors], delimiter=',', fmt='%5.5f')
                                FLUSH_COUNT=0

                            inactives_feature_matrix[FLUSH_COUNT] = current_fragment
                            FLUSH_COUNT+=1
                            FRAGMENT_COUNT+=1

                        except KeyError:
                            print("Key error")
                            continue
        
        print("Number of inactive fragments: %d\n" % (FRAGMENT_COUNT - OLD_FRAGMENT_COUNT))
        print("Number of inactive molecules: %d\n" % len(molecule_sdfs))

        # At the end, flush whatever inactives fragments we have left
        if (FLUSH_COUNT % FLUSH_BUFFER_SIZE != 0):
            all_descriptors = np.arange(global_median_cache.shape[1])
            non_degenerate_descriptors = np.delete(all_descriptors, degenerate_features)
            with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'a') as f_handle:
                np.savetxt(f_handle, inactives_feature_matrix[0:FLUSH_COUNT,non_degenerate_descriptors], delimiter=',', fmt='%5.5f')

def normalize_features(molecule_feature_matrix):

    normalized_feature_matrix = np.empty(molecule_feature_matrix.shape)

    # Normalize the values of each fragment for each feature
    for feature in range(molecule_feature_matrix.shape[1]):
        # Get the minimum accross the feature values
        max_feature = np.amax(molecule_feature_matrix[:,feature])
        # Get the maximum accross the feature values
        min_feature = np.amin(molecule_feature_matrix[:,feature])
        # Normalize each fragment's feature value
        for fragment in range(molecule_feature_matrix.shape[0]):
            normalized_feature_matrix[fragment,feature] = (molecule_feature_matrix[fragment,feature] - min_feature) / (max_feature - min_feature)
            
    return normalized_feature_matrix


def retrieve_sdf_features(descriptor_file, sdf_molecules_to_fragments_file,
    active_molecules, inactive_molecules, output_details=False):
    
    #TODO: description
    descriptors_map = None
    descriptors = None

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
    else:
        shutil.rmtree(DATA_DIRECTORY)
        os.makedirs(DATA_DIRECTORY)

    if DESCRIPTOR_TO_RAM:
        # First load the descriptors into RAM
        descriptors_map, descriptors = _read_descriptor_file(descriptor_file)

    # Load the non-imputed actives feature matrix
    [non_imputed_feature_matrix,FRAGMENT_COUNT] = _load_matrix_sdf(descriptor_file, sdf_molecules_to_fragments_file,
        active_molecules, output_details=1, 
        descriptors_map = descriptors_map, descriptors=descriptors)

    # Impute the actives, keeping track of degenerate features and any global medians
    global_median_cache,degenerate_features,used_features = _actives_feature_impute(non_imputed_feature_matrix,descriptors)
    
    # Load inactives matrix using the results from the actives imputation to impute the inactives matrix
    _inactives_load_impute_sdf(degenerate_features,
        global_median_cache, descriptor_file, sdf_molecules_to_fragments_file, \
        inactive_molecules, FRAGMENT_COUNT, output_details=1, \
        descriptors_map = descriptors_map, descriptors= descriptors)

    # Flush statistics on molecules
    _flush_metadata(global_median_cache, used_features)

def main():
    from numpy import random
    feature_matrix = np.empty([11,10])
    for i in range(feature_matrix.shape[0]):
        cur_fragment = np.random.randint(0,200,size=feature_matrix.shape[1]).reshape(1,feature_matrix.shape[1])
        feature_matrix[i] = cur_fragment

    feature_matrix = normalize_features(feature_matrix)
    print feature_matrix

if __name__ == '__main__':
    main()
    