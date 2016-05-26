#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__= "Aakash Ravi"
__email__= "aakash_ravi@hotmail.com"

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
import sys
import subprocess
import molecular_clusters
import cluster_analysis
import MolecularPreprocessing
from rdkit.ML.Scoring import Scoring

FLUSH_BUFFER_SIZE = config.FLUSH_BUFFER_SIZE
DESCRIPTOR_TO_RAM = config.DESCRIPTOR_TO_RAM
NUM_FEATURES = config.NUM_FEATURES
COVARIANCE_THRESHOLD = config.COVARIANCE_THRESHOLD
DATA_DIRECTORY = config.DATA_DIRECTORY
DEBUG = config.DEBUG

# Store feature max and min for feature normalization
feature_max = {}
feature_min = {}

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
            if (np.mean(descriptor_values)==np.nan):
                print(descriptor_values)
                input("What now?")

            global_descriptor_average_array.append(np.mean(descriptor_values))
    
    # If descriptor is defined for no molecule, it is a degenerate descriptor,
    # So we return np.nan, else we return the median of the averages
    if (len(global_descriptor_average_array) == 0):
        return np.nan
    else:
        return np.median(global_descriptor_average_array)

def _actives_feature_impute(feature_matrix, descriptor_matrix, descriptors_map=None, active_fragments=None,
    inactive_fragments=None):
    
    if (feature_matrix is None):
        print("Empty matrix; no standard imputation done, continuing...")
        return [None,None,None]

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

    print("Starting creation of global median cache")

    for descriptor in range(0,feature_matrix.shape[1]):
        global_descriptor_median = _compute_feature_median(feature_matrix, descriptor,
                molecule_keys)
        if (np.isnan(global_descriptor_median)):
            degenerate_features.append(descriptor)
        global_median_cache[0,descriptor] = global_descriptor_median
    
    print("Finished creation of global median cache")

    # Recompute the significant features before beginning imputation
    if DESCRIPTOR_TO_RAM:
        neighborhood_extractor.extract_features(NUM_FEATURES,descriptor_matrix,COVARIANCE_THRESHOLD,descriptors_map,active_fragments,inactive_fragments)


    print("Actives imputation: starting out with %d features" % (feature_matrix.shape[1]))
    significant_features = np.genfromtxt(os.path.join(config.DATA_DIRECTORY,'significant_features'),delimiter=',')
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

    # First remove all degenerate features
    feature_matrix = np.delete(feature_matrix, degenerate_features, 1)
    print "Actives imputation: removed degenerate features, now have %d features" % (feature_matrix.shape[1])
    
    # Now loop through the fragments and impute
    for fragment in range(1, len(feature_matrix)):
        # Obtain all descriptors that have non-numerical values for this fragment
        nan_descriptors = np.where(np.isnan(feature_matrix[fragment]) == True)
        for j in nan_descriptors:
            feature_matrix[fragment, j] = global_median_cache[0,j]

    # Remove existing dataset files and flush new actives data
    with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'w+') as f_handle:
        np.savetxt(f_handle, feature_matrix[1:,:], delimiter=',',fmt='%5.5f')

    # Store the feature max and min for feature normalization
    for feature in range(feature_matrix[1:,:].shape[1]):
        # Get the maximum accross the feature values
        max_feature = np.amax(feature_matrix[1:,feature])
        # Get the minimum accross the feature values
        min_feature = np.amin(feature_matrix[1:,feature])
        
        # Update the dictionaries holding the feature maximums and minimums for later use
        feature_max[feature] = max_feature
        feature_min[feature] = min_feature

    # Update degenerate features
    used_features = [ x-1 for x in feature_matrix[0]]
    degenerate_features = [i for i in all_features if i not in used_features]
    
    return [global_median_cache, degenerate_features, used_features]

def _read_descriptor_file(descriptor_file_name):
    print("[{}] Reading descriptors file...".format(str(datetime.now())))
    print("Descriptor file name: %s" % descriptor_file_name)

    # Read in fragments descriptors into an NP array
    descriptors = None
    # Store mapping between SMILES and indeces in a dictionary
    descriptors_smiles_to_ix = {}
    with open(descriptor_file_name, 'r') as descriptor_file:
        #Header serves to find out the number of descriptors
        header = descriptor_file.readline().rstrip().split(',')

        # if DEBUG:
        #     with open(os.path.join(DATA_DIRECTORY,'all_descriptors.csv'),'wb+') as f_handle:
        #         csv.writer(f_handle).writerow(header)

        descriptors = np.empty((0, len(header)-1), np.float)
        #Adding rows into the NP array one by one is expensive. Therefore we read rows
        #int a python list in batches and regularly flush them into the NP array
        aux_descriptors = []
        ix = 0
        for line in descriptor_file:
            line_split = line.rstrip().split(',')
            descriptors_smiles_to_ix[line_split[0].strip('"\'')] = ix

            aux_descriptors.append([float(x) if isfloat(x) else float('nan') for x in line_split[1:]])
            ix += 1
            if ix % 1000 == 0:
                descriptors = np.vstack((descriptors, np.asarray(aux_descriptors)))
                del aux_descriptors[:]

        if len(aux_descriptors) > 0:
            descriptors = np.vstack((descriptors, np.asarray(aux_descriptors)))
            del aux_descriptors[:]

    return [descriptors_smiles_to_ix, descriptors]

def _flush_metadata(global_median_cache, used_features):

    # Flush all the newly obtained metadata about the molecular feature matrix

    # Keep track of the values of the global median cache
    with open(os.path.join(DATA_DIRECTORY,"global_median_cache.pkl"), 'wb+') as f_handle:
        pickle.dump(global_median_cache, f_handle, pickle.HIGHEST_PROTOCOL)

    # Keep track of which features are used in the final, fully imputed, molecular
    # feature matrix.
    with open(os.path.join(DATA_DIRECTORY,"used_features.pkl"),'wb+') as f_handle:
        pickle.dump(used_features, f_handle, pickle.HIGHEST_PROTOCOL)
    
    # Save the fragment number to name mapping dictionaries in a pickle file
    with open(os.path.join(DATA_DIRECTORY,"fragment_number_name_mapping.pkl"), 'wb+') as f_handle:
        pickle.dump(fragment_number_name_mapping,f_handle, pickle.HIGHEST_PROTOCOL)

    # Save the fragments to molecules mapping(s) dictionar(ies) in a pickle file
    with open(os.path.join(DATA_DIRECTORY,"actives_fragment_molecule_mapping.pkl"), 'wb+') as f_handle:
        pickle.dump(actives_fragment_molecule_mapping, f_handle, pickle.HIGHEST_PROTOCOL)

    # Save the fragments to molecules mapping(s) dictionar(ies) in a pickle file
    with open(os.path.join(DATA_DIRECTORY,"inactives_fragment_molecule_mapping.pkl"), 'wb+') as f_handle:
        pickle.dump(inactives_fragment_molecule_mapping, f_handle, pickle.HIGHEST_PROTOCOL)

# def _load_matrix_sdf(descriptor_file, molecules_to_fragments_file,
#     molecule_sdfs, output_details=0,
#     descriptors_map=None, descriptors=None):
def _load_matrix_sdf(descriptor_file,molecules_to_fragments,output_details=0,
    descriptors_map=None, descriptors=None):

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
    molecule_index = 0
    # Add all fragment data for each molecule in the array
    # 'molecule_sfds', and return a matrix of found values
    # for molecule_index in range(0, len(molecule_sdfs)):
    for molecule in molecules_to_fragments:

        # Sometimes the desired SMILE may not have any fragment data, so we just continue
        # And notify the user that no fragments were found for this SMILES
        # try:
        #     # full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments] 
        #     #             # if molecule["name"] == molecule_sdfs[molecule_index]]

        #     # First index is actual fragments, since there 
        #     # can exist only one key value pair for the molecule and its fragments
        #     full_fragments = full_fragments[0]
        #     fragments = [fragment["smiles"] for fragment in full_fragments]

        #     # if output_details:
        #     #     print("Starting analysis of {} ...".format(molecule_sdfs[molecule_index]))
        # except KeyError:
        #     print("Fragments not available for molecule ", molecule_sdfs[molecule_index])
        #     continue

        full_fragments = molecule["fragments"]
        fragments = [fragment["smiles"] for fragment in full_fragments]

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
                    try:
                        ix_f = descriptors_map[f]
                    
                        non_imputed_feature_matrix = np.append(non_imputed_feature_matrix,
                                                    [np.insert(descriptors[ix_f], 0, molecule_index)], axis=0)

                        if f in found_fragments:
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            actives_fragment_molecule_mapping[f].append(molecule_index)
                        else:
                            found_fragments.append(f)
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            actives_fragment_molecule_mapping[f] = [molecule_index]

                        FRAGMENT_COUNT+=1
                    except KeyError:
                        print("Key error extracting actives from feature matrix.")
                        print("Affected fragment %s" % f)
                        continue

            molecule_index += 1
    
    # print("Number of active fragments: %d\n" % FRAGMENT_COUNT)
    # print("Number of active molecules: %d\n" % len(molecule_sdfs))

    return [non_imputed_feature_matrix, FRAGMENT_COUNT]

def _inactives_load_impute_sdf(degenerate_features, \
    global_median_cache, descriptor_file, molecules_to_fragments,
    FRAGMENT_COUNT, output_details=0,
    descriptors_map=None, descriptors=None):

    # For debugging purposes
    OLD_FRAGMENT_COUNT = FRAGMENT_COUNT

    # Keep a list of already found fragments for metadata storage
    found_fragments = []
    
    # Create a temporary holding space for the fragments, and periodically flush to disk
    inactives_feature_matrix = np.empty([FLUSH_BUFFER_SIZE, global_median_cache.shape[1]], np.float)

    FLUSH_COUNT = 0
    molecule_index = 0
    # Add all fragment data for each molecule in the array
    # 'molecule_sdfs', and return a matrix of found values
    # for molecule_index in range(0, len(molecule_sdfs)):
    for molecule in molecules_to_fragments:

        # # Sometimes the desired SMILE may not have any fragment data, so we just continue
        # # And notify the user that no fragments were found for this SMILES
        # try:
        #     full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments 
        #                 if molecule["name"] == molecule_sdfs[molecule_index]]
        #     # First index is actual fragments, since there 
        #     # can exist only one key value pair for the molecule and its fragments
        #     full_fragments = full_fragments[0]
        #     fragments = [fragment["smiles"] for fragment in full_fragments]

        #     # if output_details:
        #         # print("Starting analysis of {} ...".format(molecule_sdfs[molecule_index]))
        # except KeyError:
        #     print("Fragments not available for molecule ", molecule_sdfs[molecule_index])
        #     continue

        full_fragments = molecule["fragments"]
        fragments = [fragment["smiles"] for fragment in full_fragments]

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
                        non_degenerate_inactives_feature_matrix = inactives_feature_matrix[:,non_degenerate_descriptors]

                        with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'a') as f_handle:
                            np.savetxt(f_handle, non_degenerate_inactives_feature_matrix, delimiter=',', fmt='%5.5f')
                        FLUSH_COUNT=0

                        for feature in range(non_degenerate_inactives_feature_matrix.shape[1]):
                            # Get the maximum accross the feature values
                            max_feature = np.amax(non_degenerate_inactives_feature_matrix[:,feature])
                            # Get the minimum accross the feature values
                            min_feature = np.amin(non_degenerate_inactives_feature_matrix[:,feature])
                            
                            if (max_feature > feature_max[feature]):
                                feature_max[feature] = max_feature
                            if (min_feature < feature_min[feature]):
                                feature_min[feature] = min_feature
                        
                    inactives_feature_matrix[FLUSH_COUNT]= molecule_descriptor_row
                    FLUSH_COUNT+=1
                    FRAGMENT_COUNT+=1

            # Use the descriptors dictionary in RAM
            else:
                for f in fragments:
                    # If we already found the fragment, we continue on; will save us time and space
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
                            non_degenerate_inactives_feature_matrix = inactives_feature_matrix[:,non_degenerate_descriptors]

                            with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'a') as f_handle:
                                np.savetxt(f_handle, non_degenerate_inactives_feature_matrix, delimiter=',', fmt='%5.5f')
                            FLUSH_COUNT=0
                            
                            for feature in range(non_degenerate_inactives_feature_matrix.shape[1]):
                                # Get the maximum accross the feature values
                                max_feature = np.amax(non_degenerate_inactives_feature_matrix[:,feature])
                                # Get the minimum accross the feature values
                                min_feature = np.amin(non_degenerate_inactives_feature_matrix[:,feature])
                            
                                if (max_feature > feature_max[feature]):
                                    feature_max[feature] = max_feature
                                if (min_feature < feature_min[feature]):
                                    feature_min[feature] = min_feature

                        if f in found_fragments:
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            inactives_fragment_molecule_mapping[f].append(molecule_index)
                        else:
                            found_fragments.append(f)
                            fragment_number_name_mapping[FRAGMENT_COUNT] = f
                            inactives_fragment_molecule_mapping[f] = [molecule_index]


                        inactives_feature_matrix[FLUSH_COUNT] = current_fragment
                        FLUSH_COUNT+=1
                        FRAGMENT_COUNT+=1

                    except KeyError:
                        print("Key error extracting inactives from feature matrix.")
                        print("Affected fragment %s" % f)
                        continue

            molecule_index += 1
    
    # print("Number of inactive fragments: %d\n" % (FRAGMENT_COUNT - OLD_FRAGMENT_COUNT))
    # print("Number of inactive molecules: %d\n" % len(molecule_sdfs))

    # At the end, flush whatever inactives fragments we have left
    if (FLUSH_COUNT % FLUSH_BUFFER_SIZE != 0):
        all_descriptors = np.arange(global_median_cache.shape[1])
        non_degenerate_descriptors = np.delete(all_descriptors, degenerate_features)
        non_degenerate_inactives_feature_matrix = inactives_feature_matrix[:,non_degenerate_descriptors]

        with open(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),'a') as f_handle:
            np.savetxt(f_handle, non_degenerate_inactives_feature_matrix[0:FLUSH_COUNT], delimiter=',', fmt='%5.5f')

        for feature in range(non_degenerate_inactives_feature_matrix.shape[1]):
            # Get the maximum accross the feature values
            max_feature = np.amax(non_degenerate_inactives_feature_matrix[:,feature])
            # Get the minimum accross the feature values
            min_feature = np.amin(non_degenerate_inactives_feature_matrix[:,feature])
                            
            if (max_feature > feature_max[feature]):
                feature_max[feature] = max_feature
            if (min_feature < feature_min[feature]):
                feature_min[feature] = min_feature


def normalize_features(molecule_feature_matrix_file, DATA_DIRECTORY, feature_max=None, feature_min=None):
    
    # Remove any existing temp file
    open(os.path.join(DATA_DIRECTORY,"temp_file"),'w+')
    normalized_feature_matrix = None
    max_feature_array = []
    min_feature_array = []

    # for feature in range(len(feature_max)):
    #     if (feature_max[feature] == feature_min[feature]):
    #         print(feature)

    with open(os.path.join(DATA_DIRECTORY,molecule_feature_matrix_file),'r') as f_handle:

        if (feature_max is not None) and (feature_min is not None):
            reader = csv.reader(f_handle)
            data_observations_left = True
            while(data_observations_left):
                try:
                    next_observation = next(reader)
                except StopIteration:
                    data_observations_left = False
                    continue
                next_observation = np.asarray(next_observation).astype(np.float)

                # Normalize each feature's value
                for feature in range(len(next_observation)):
                    # if feature_max[feature] == feature_min[feature]:
                    #     # print("Divide by zero, feature %d" % feature)
                    #     # print "%d %d" % (feature_max[feature], feature_min[feature])
                    #     # For degenerate features, set all the observations to the same
                    #     # value in range [0,1] - in this case 1.
                    #     next_observation[feature] = 1
                    #     continue
                    
                    try:
                        next_observation[feature] = (next_observation[feature] - feature_min[feature]) / (feature_max[feature] - feature_min[feature])
                    except RuntimeWarning:
                        print("Runtime warning:")
                        print(feature_max[feature])
                        print(feature_min[feature])
                        print(feature)

                # Flush the new normalized vector into the new file
                with open(os.path.join(DATA_DIRECTORY,"temp_file"),'a') as f_handle:
                    np.savetxt(f_handle, next_observation.reshape(1,len(next_observation)), delimiter=',',fmt='%5.5f')
            
            max_feature_array = feature_max
            min_feature_array = feature_min
        else:
            molecule_feature_matrix = np.asarray(np.genfromtxt(molecule_feature_matrix_file, delimiter=',')).astype(np.float)
            normalized_feature_matrix = np.empty(molecule_feature_matrix.shape).reshape(molecule_feature_matrix.shape[0],molecule_feature_matrix.shape[1])
            # Normalize the values of each fragment for each feature
            for feature in range(molecule_feature_matrix.shape[1]):
                # Get the minimum accross the feature values
                max_feature = np.amax(molecule_feature_matrix[:,feature])
                # Get the maximum accross the feature values
                min_feature = np.amin(molecule_feature_matrix[:,feature])
                if max_feature == min_feature:
                    # print("Divide by zero!")
                    # print molecule_feature_matrix[:,feature]
                    # For degenerate features, set all the observations to the same
                    # value in range [0,1] - in this case 1.
                    for fragment in range(molecule_feature_matrix.shape[0]):
                        normalized_feature_matrix[fragment,feature] = 1
                    continue

                # Normalize each fragment's feature value
                for fragment in range(molecule_feature_matrix.shape[0]):
                    normalized_feature_matrix[fragment,feature] = (molecule_feature_matrix[fragment,feature] - min_feature) / (max_feature - min_feature)
            
                max_feature_array.append(max_feature)
                min_feature_array.append(min_feature)

            with open(os.path.join(DATA_DIRECTORY,"temp_file"),'w+') as f_handle:
                np.savetxt(f_handle, normalized_feature_matrix, delimiter=',',fmt='%5.5f')

    # Rename the temporary file as the original matrix, to be consistent
    # TODO:FIND BETTER WORKAROUND FOR THIS
    os.remove(os.path.join(DATA_DIRECTORY,molecule_feature_matrix_file))
    os.rename(os.path.join(DATA_DIRECTORY,"temp_file"), molecule_feature_matrix_file)
    return [max_feature_array,min_feature_array]

# def create_feature_matrix(descriptor_file, sdf_molecules_to_fragments_file,
#     active_molecules, inactive_molecules, output_details=False):
def create_feature_matrix(features_file,active_fragments,inactive_fragments,descriptors_map=None,descriptors=None,output_details=False):
    #TODO: description

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
    else:
        shutil.rmtree(DATA_DIRECTORY)
        os.makedirs(DATA_DIRECTORY)

    if (descriptors_map == None) or (descriptors == None):
        descriptors_map, descriptors = _read_descriptor_file(features_file)

    # Load the non-imputed actives feature matrix
    [non_imputed_feature_matrix,FRAGMENT_COUNT] = _load_matrix_sdf(features_file,active_fragments,output_details=1, 
        descriptors_map = descriptors_map, descriptors=descriptors)

    print("Beginning imputation of actives feature matrix and creation of global median cache.")
    # Impute the actives, keeping track of degenerate features and any global medians
    global_median_cache,degenerate_features,used_features = _actives_feature_impute(non_imputed_feature_matrix,descriptors,None,None,None)

    
    print("Beginning creation of inactives feature matrix.")
    # Load inactives matrix using the results from the actives imputation to impute the inactives matrix
    _inactives_load_impute_sdf(degenerate_features,
        global_median_cache, features_file, inactive_fragments, FRAGMENT_COUNT, output_details=1, \
        descriptors_map = descriptors_map, descriptors= descriptors)

    # Normalize the features
    normalize_features(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),DATA_DIRECTORY,feature_max,feature_min)

    # Flush statistics on molecules
    _flush_metadata(global_median_cache, used_features)

def _compute_subspace_distance(point_1,point_2,subspace):
    "A helper function that computes the distance between point 1 and point 2, when projected to the \
    specified subspace."
    
    distance = 0
    for i in range(len(point_1)):
        if i in subspace:
            distance+=((point_1[i] - point_2[i])**2)
    
    # Return the square root of the aggregated distance
    return np.sqrt(distance)

def get_score(molecule):
    return molecule["score"]


def get_AUC(molecule_names_and_activity, molecules_to_fragments, descriptors_map, descriptors, MODEL_DIRECTORY):

    sorted_activity_list = []

    for test_molecule in molecule_names_and_activity:
        full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments 
                            if molecule["name"] == test_molecule["name"]]

        # First index is actual fragments, since there 
        # can exist only one key value pair for the molecule and its fragments
        full_fragments = full_fragments[0]
        fragments = [fragment["smiles"] for fragment in full_fragments]

        with open(os.path.join(ROOT_DIR_METADATA,'global_median_cache.pkl'),'r') as f_handle:
            global_median_cache = pickle.load(f_handle)

        with open(os.path.join(ROOT_DIR_METADATA,'used_features.pkl'),'r') as f_handle:
            used_features = pickle.load(f_handle)

        found_fragments = []
        feature_matrix = np.empty((0,len(used_features)))

        # Create the feature matrix for the fragments of this particular molecule
        for f in fragments:
            # If we already found the fragment, we continue on; will save us time and space
            if f in found_fragments:
                continue
            else:
                found_fragments.append(f)
                
                try:
                    ix_f = descriptors_map[f]
                    current_fragment = descriptors[ix_f].reshape(1,len(descriptors[ix_f]))

                    # Obtain all feature values that have non-numerical values for this fragment
                    nan_descriptors = np.where(np.isnan(current_fragment) == True)

                    # Impute these non-numerical values with the values from the global median cache
                    # which was again, obtained from the training set.
                    for j in nan_descriptors:
                        current_fragment[0,j] = global_median_cache[0,j]

                    # Finally, project the features of the current fragment into only the non-degenerate
                    # feature space as learned from the training set.
                    current_fragment = current_fragment[:,used_features]

                    # Append this fragment to our feature matrix
                    np.vstack((feature_matrix,current_fragment))

                except KeyError:
                    print("Key error during AUC calculation!")
                    continue

        with open(os.path.join(MODEL_DIRECTORY,"molecular_cluster_model.pkl"),'r') as f_handle:
            molecular_cluster_model = pickle.load(f_handle)

        if len(molecular_cluster_model) == 0:
            print "No clusters in model; can't evaluate new molecule."
            return -1

        distance_array = []
        for i in range(feature_matrix.shape[0]):
            closest_centroid_distance = np.min([ _compute_subspace_distance(feature_matrix[i],molecular_cluster_model[j]['centroid'],molecular_cluster_model[j]['subspace']) \
                for j in range(len(molecular_cluster_model))])

            distance_array.append(closest_centroid_distance)

        score = np.mean(np.asarray(distance_array))
        sorted_activity_list.append({"score":score,"activity":test_molecule["activity"]})

    sorted_activity_list = sorted(sorted_activity_list,key=get_score)
    return Scoring.CalcAUC(sorted_activity_list, "activity")




def molecular_model_creation(features_file,active_fragments,inactive_fragments,features_map,features_matrix,num_active_molecules,num_inactive_molecules):

    # actives_dataset_file = os.path.join(config.INPUT_DATA_DIRECTORY,str(DATASET_NUMBER), str(DATASET_NUMBER) + "_ligands.json")
    # inactives_dataset_file = os.path.join(config.INPUT_DATA_DIRECTORY,str(DATASET_NUMBER), str(DATASET_NUMBER) + "_decoys.json")
    # features_file = os.path.join(config.INPUT_DATA_DIRECTORY,str(DATASET_NUMBER), str(DATASET_NUMBER) + "_features.csv")
    # fragments_file = os.path.join(config.INPUT_DATA_DIRECTORY,str(DATASET_NUMBER), str(DATASET_NUMBER) + "_fragments.json")
    
    # os.environ["DATASET_NUMBER"] = str(DATASET_NUMBER)

    # result = subprocess.call(['bash', './data_preprocessing.sh'])

    # # Fetch the SDF Actives and Inactives List
    # with open(actives_dataset_file,'r') as f_handle:
    #     actives = json.load(f_handle)
    # with open(inactives_dataset_file,'r') as f_handle:
    #     inactives = json.load(f_handle)

    print "Starting molecular feature model creation..."
    # Retrieve the molecular feature matrix corresponding to our dataset and 
    # flush it to file
    # create_feature_matrix(features_file, fragments_file ,actives, inactives, output_details=False)
    create_feature_matrix(features_file, active_fragments, inactive_fragments,features_map,features_matrix,output_details=False)
    print "Finished molecular feature matrix creation."

    print "Starting search of molecular clusters..."
    # Find the clusters using ELKI
    molecular_clusters.find_clusters(CLUSTER_FILENAME = os.path.join(config.DATA_DIRECTORY,"detected_clusters"),
        FEATURE_MATRIX_FILE = os.path.join(config.DATA_DIRECTORY,"molecular_feature_matrix.csv"),
        ELKI_EXECUTABLE=config.ELKI_EXECUTABLE,num_active_molecules=num_active_molecules,num_inactive_molecules=num_inactive_molecules)
    print "Finished search of molecular clusters."

    # Analyze the clusters and output the most pure and diverse ones
    print "Starting analysis and pruning of found clusters..."
    PURITY_THRESHOLD = .5
    DIVERSITY_THRESHOLD = num_active_molecules * .6
    DIVERSITY_PERCENTAGE = False
    cluster_analysis.create_cluster_centroid_model(PURITY_THRESHOLD, DIVERSITY_THRESHOLD, DIVERSITY_PERCENTAGE)
    print "Finished analysis and pruning of clusters! Clusters model available in data directory for querying."

def main():
    actives_fragment_file = sys.argv[1] # Check
    inactives_fragment_file = sys.argv[2] # Check
    features_file = sys.argv[3]
    training_test_split_file = sys.argv[4] # Check
    MOLECULAR_MODEL_DIRECTORY = os.path.join(DATA_DIRECTORY,"ClustersModel")

    with open(training_test_split_file,"r+") as f_handle:
        training_test_molecules = json.load(f_handle)

    active_training_molecule_names = [molecule["name"] for molecule in training_test_molecules["data"]["train"]["ligands"]]
    inactive_training_molecule_names = [molecule["name"] for molecule in training_test_molecules["data"]["train"]["decoys"]]

    with open(actives_fragment_file,"r+") as f_handle:
        actives_molecule_to_fragments = json.load(f_handle)
    with open(inactives_fragment_file,"r+") as f_handle:
        inactives_molecule_to_fragments = json.load(f_handle)

    print("Extracting active and inactive training molecules...")

    active_training_molecules = [molecule for molecule in actives_molecule_to_fragments \
                                    if molecule["name"] in active_training_molecule_names]
    
    inactive_training_molecules = [molecule for molecule in inactives_molecule_to_fragments \
                                    if molecule["name"] in inactive_training_molecule_names]

    
    print("Reading the features file into memory")

    features_map, features = _read_descriptor_file(features_file)
    
    print("Removing constant features in feature matrix...")

    # Preprocessing: remove constant features
    # output_features_file = os.path.join(os.getcwd(),"output_features.csv")
    # MolecularPreprocessing.remove_constant_features(features_file,output_features_file)
    features = MolecularPreprocessing.remove_constant_features(features)

    # Create the molecular model
    molecular_model_creation(features_file,active_training_molecules,inactive_training_molecules,features_map,features,len(active_training_molecules),len(inactive_training_molecules))
    # molecular_model_creation(active_training_molecules,inactive_training_molecules,features_file,len(active_training_molecules),len(inactive_training_molecules))
    
    print("Finished molecular feature model creation...")

    testing_molecules = training_test_molecules["data"]["test"]

    # Combined active and inactive molecular fragments
    full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

    print("Getting AUC Score for current dataset...")
    # Get the AUC score for the testing data
    print(get_AUC(testing_molecules,full_molecules_to_fragments,features_map,features,MOLECULAR_MODEL_DIRECTORY))

if __name__ == '__main__':
    main()
    