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
import importlib
import subprocess
import molecular_clusters
import cluster_analysis
import MolecularPreprocessing
from rdkit.ML.Scoring import Scoring

FLUSH_BUFFER_SIZE = config.FLUSH_BUFFER_SIZE
DESCRIPTOR_TO_RAM = config.DESCRIPTOR_TO_RAM
NUM_FEATURES = config.NUM_FEATURES
CORRELATION_THRESHOLD = config.CORRELATION_THRESHOLD
DATA_DIRECTORY = os.path.join(os.path.dirname(sys.argv[1]),"FragmentDescriptorData")
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
        # If there doesn't exist any infinite descriptor values for fragments of this molecule,
        # we add it to the global average array.
        if (np.isfinite(descriptor_values).all()):
            global_descriptor_average_array.append(np.mean(descriptor_values))
        else:
            continue

    # If descriptor is defined for no molecule, it is a degenerate descriptor,
    # So we return np.nan, else we return the median of the averages
    if (len(global_descriptor_average_array) == 0):
        return np.nan
    else:
        return np.median(global_descriptor_average_array)

def _actives_feature_impute(feature_matrix):
    
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

    print("Starting creation of global median cache...")

    for descriptor in range(0,feature_matrix.shape[1]):
        global_descriptor_median = _compute_feature_median(feature_matrix, descriptor,
                molecule_keys)
        if (not np.isfinite(global_descriptor_median)):
            degenerate_features.append(descriptor)
        global_median_cache[0,descriptor] = global_descriptor_median
    
    print("Imputing fragments according to global median cache...")

    # Now, loop through the fragments and impute using the global median cache
    for fragment in range(1, len(feature_matrix)):
        # Obtain all descriptors that have non-numerical values for this fragment
        nan_descriptors = np.where(np.isfinite(feature_matrix[fragment]) != True)
        for j in nan_descriptors:
            feature_matrix[fragment, j] = global_median_cache[0,j]

    # Then remove all the degenerate features from the feature matrix
    feature_matrix = np.delete(feature_matrix, degenerate_features, 1)
    print "Actives imputation: removed degenerate features, now have %d features" % (feature_matrix.shape[1])

    # # Compute the significant features using the correlation neighborhoods method
    # if DESCRIPTOR_TO_RAM:
    #     neighborhood_extractor.extract_features(NUM_FEATURES,feature_matrix,CORRELATION_THRESHOLD)

    # significant_features = np.genfromtxt(os.path.join(config.DATA_DIRECTORY,'significant_features'),delimiter=',')
    # redundant_features = [i for i in range(feature_matrix.shape[1]) if i not in significant_features]

    # # Remove the redundant features from the feature matrix
    # feature_matrix = np.delete(feature_matrix, redundant_features, 1)
    # print "Actives imputation: removed %d features with constant features and covariance neighborhoods, now have %d features, with the NUM_FEATURES parameters set to %d" % (len(redundant_features), len(significant_features), NUM_FEATURES)

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

def _load_matrix_sdf(molecules_to_fragments,
    descriptors_map, descriptors):

    # Keep a list of already found fragments for metadata storage
    found_fragments = []
    
    non_imputed_feature_matrix = np.empty((0, descriptors.shape[1]+1), np.float)

    # Append the descriptor number columns
    descriptor_numbers = np.arange(descriptors.shape[1]+1).reshape((1, descriptors.shape[1]+1))
    non_imputed_feature_matrix = np.append(non_imputed_feature_matrix,
                                                descriptor_numbers, axis=0)

    FRAGMENT_COUNT = 0
    molecule_index = 0

    for molecule in molecules_to_fragments:

        full_fragments = molecule["fragments"]
        fragments = [fragment["smiles"] for fragment in full_fragments]

        for f in fragments:

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

    return [non_imputed_feature_matrix, FRAGMENT_COUNT]

def _inactives_load_impute_sdf(degenerate_features, \
    global_median_cache, molecules_to_fragments, \
    FRAGMENT_COUNT, descriptors_map, descriptors):

    # For debugging purposes
    OLD_FRAGMENT_COUNT = FRAGMENT_COUNT

    # Keep a list of already found fragments for metadata storage
    found_fragments = []
    
    # Create a temporary holding space for the fragments, and periodically flush to disk
    inactives_feature_matrix = np.empty([FLUSH_BUFFER_SIZE, global_median_cache.shape[1]], np.float)

    FLUSH_COUNT = 0
    molecule_index = 0

    for molecule in molecules_to_fragments:

        full_fragments = molecule["fragments"]
        fragments = [fragment["smiles"] for fragment in full_fragments]

        for f in fragments:
            # If we already found the fragment, we continue on; will save us time and space
            try:
                ix_f = descriptors_map[f]
                current_fragment = descriptors[ix_f]

                # Obtain all descriptors that have non-numerical values for this fragment
                nan_descriptors = np.where(np.isfinite(current_fragment) != True)

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
                print("Affected fragment: %s" % f)
                continue

        molecule_index += 1
    

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


def _normalize_features(molecule_feature_matrix_file, DATA_DIRECTORY, feature_max=None, feature_min=None):
    
    # Remove any existing temp file
    open(os.path.join(DATA_DIRECTORY,"temp_file"),'w+')
    normalized_feature_matrix = None
    max_feature_array = []
    min_feature_array = []

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
                    next_observation[feature] = (next_observation[feature] - feature_min[feature]) / (feature_max[feature] - feature_min[feature])
                    # Handle degenerate cases where, due to some rounding errors, the maximum
                    # and minimum are exactly the same and we therefore get a division by zero.
                    if not (np.isfinite(next_observation[feature])):
                        next_observation[feature] = feature_max[feature]


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

def _create_feature_matrix(active_fragments,inactive_fragments,descriptors_map,descriptors):
    #TODO: description

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
    else:
        shutil.rmtree(DATA_DIRECTORY)
        os.makedirs(DATA_DIRECTORY)

    # Load the non-imputed actives' feature matrix
    [non_imputed_feature_matrix,FRAGMENT_COUNT] = _load_matrix_sdf(active_fragments, 
        descriptors_map = descriptors_map, descriptors=descriptors)

    print("Beginning imputation of actives feature matrix and creation of global median cache...")
    # Impute the actives, keeping track of degenerate features and any global medians
    global_median_cache,degenerate_features,used_features = _actives_feature_impute(non_imputed_feature_matrix)

    
    print("Beginning creation of inactives feature matrix using significant features...")
    # Load inactives matrix using the results from the actives imputation to impute the inactives matrix
    _inactives_load_impute_sdf(degenerate_features,
        global_median_cache, inactive_fragments, FRAGMENT_COUNT, \
        descriptors_map = descriptors_map, descriptors= descriptors)

    print("Normalizing feature matrix...")
    # Normalize the features
    _normalize_features(os.path.join(DATA_DIRECTORY,"molecular_feature_matrix.csv"),DATA_DIRECTORY,feature_max,feature_min)

    # Flush statistics on molecules
    _flush_metadata(global_median_cache, used_features)

    return [global_median_cache,used_features]

def _compute_subspace_distance(point_1,point_2,subspace):
    "A helper function that computes the distance between point 1 and point 2, when projected to the \
    specified subspace."
    
    distance = 0
    for i in range(len(point_1)):
        if subspace[i] == 1:
            distance+=(point_1[i]-point_2[i])**2
    
    # Return the square root of the aggregated distance
    return np.sqrt(distance)

def get_score(molecule):
    return molecule["score"]

def get_ranking(molecule):
    return molecule["ranking"]

def get_activity(molecule):
    return molecule["activity"]

# Cluster ranking
def get_AUC(molecule_names_and_activity, molecules_to_fragments, descriptors_map, descriptors, MODEL_DIRECTORY, \
    global_median_cache,used_features,scoring_method, descriptor_csv_file, \
    bayes_subspace = None, bayes_centroid = None, single_cluster_model = None, bayes_feature_file = None):
    
    cluster_rankings_list = []
    final_sorted_activity_list = []
    molecule_fragment_matrices = {}

    for test_molecule in molecule_names_and_activity:
        final_sorted_activity_list.append({"name":test_molecule["name"],"ranking":0,"activity":test_molecule["activity"]})

    if not ((bayes_subspace is not None) and (bayes_centroid is not None)):
        
        if single_cluster_model is not None:
            molecular_cluster_model = [single_cluster_model]
        else:
            with open(os.path.join(MODEL_DIRECTORY,"molecular_cluster_model.pkl"),'r') as f_handle:
                molecular_cluster_model = pickle.load(f_handle)

        if len(molecular_cluster_model) == 0:
            print "No clusters found in model; can't evaluate any new test molecules..."
            return -1

    for test_molecule in molecule_names_and_activity:

        full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments 
                            if molecule["name"] == test_molecule["name"]]

        # First index is actual fragments, since there 
        # can exist only one key value pair for the molecule and its fragments
        full_fragments = full_fragments[0]
        fragments = [fragment["smiles"] for fragment in full_fragments]

        found_fragments = []

        if (bayes_subspace is not None) and (bayes_centroid is not None):
            feature_matrix = np.empty((0,len(bayes_subspace)))
        else:
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

                    if not ((bayes_subspace is not None) and (bayes_centroid is not None)):
                        # Obtain all feature values that have non-numerical values for this fragment
                        nan_descriptors = np.where(np.isfinite(current_fragment) != True)

                        # Impute these non-numerical values with the values from the global median cache
                        # which was again, obtained from the training set.
                        for j in nan_descriptors:
                            current_fragment[0,j] = global_median_cache[0,j]

                        # Finally, project the features of the current fragment into only the non-degenerate
                        # feature space as learned from the training set.
                        current_fragment = current_fragment[:,used_features]

                    # Append this fragment to our feature matrix
                    feature_matrix = np.vstack((feature_matrix,current_fragment))

                except KeyError:
                    print("Key error during AUC calculation!")
                    continue

        if not ((bayes_subspace is not None) and (bayes_centroid is not None)):
            # Normalize the test molecule fragments
            for row in range(feature_matrix.shape[0]):
                for feature in range(feature_matrix.shape[1]):
                    feature_matrix[row,feature] = (feature_matrix[row,feature] - feature_min[feature]) / (feature_max[feature] - feature_min[feature])
                    if not (np.isfinite(feature_matrix[row,feature])):
                        feature_matrix[row,feature] = feature_max[feature]

        molecule_fragment_matrices[test_molecule["name"]] = feature_matrix

    if (bayes_subspace is not None) and (bayes_centroid is not None):

        bayes_sorted_activity_list = []
        distance_array = []

        for test_molecule in molecule_names_and_activity:

            for i in range(molecule_fragment_matrices[test_molecule["name"]].shape[0]):
                current_centroid_distance = _compute_subspace_distance(molecule_fragment_matrices[test_molecule["name"]][i],bayes_centroid,bayes_subspace)
                distance_array.append(current_centroid_distance)
            
            # No fragments are found for this molecule, so we continue since we can't evaluate it.
            if (len(distance_array) == 0):
                continue

            if scoring_method == 1:
                score = np.mean(np.asarray(distance_array))
            else:
                score = np.min(np.asarray(distance_array))

            score = np.around(score, decimals=10)
            bayes_sorted_activity_list.append({"name":test_molecule["name"],"score":score,"activity":test_molecule["activity"]})
        
        bayes_sorted_activity_list = sorted(bayes_sorted_activity_list,key=get_score)
        return Scoring.CalcAUC(bayes_sorted_activity_list,"activity")

    for cluster_model in molecular_cluster_model:
        
        cluster_sorted_activity_list = []
        distance_array = []

        for test_molecule in molecule_names_and_activity:

            # Loop over fragments of the molecule
            for i in range(molecule_fragment_matrices[test_molecule["name"]].shape[0]):
                # closest_centroid_distance = np.min([ _compute_subspace_distance(feature_matrix[i],molecular_cluster_model[j]['centroid'],molecular_cluster_model[j]['subspace']) \
                #     for j in range(len(molecular_cluster_model))])
                
                current_centroid_distance = _compute_subspace_distance(molecule_fragment_matrices[test_molecule["name"]][i],cluster_model['centroid'],cluster_model['subspace'])
                distance_array.append(current_centroid_distance)
            
            # No fragments are found for this molecule, so we continue since we can't evaluate it.
            if (len(distance_array) == 0):
                continue

            if scoring_method == 1:
                score = np.mean(np.asarray(distance_array))
            else:
                score = np.min(np.asarray(distance_array))

            score = np.around(score, decimals=10)
            cluster_sorted_activity_list.append({"name":test_molecule["name"],"score":score,"activity":test_molecule["activity"]})

        cluster_sorted_activity_list = sorted(cluster_sorted_activity_list,key=get_score)
        # Append the current cluster's ranking to the cluster ranking list.
        cluster_rankings_list.append(cluster_sorted_activity_list)
        
        # unique,counts = np.unique(cluster_model['subspace'],return_counts=True)
        
        # print cluster_model['subspace']
        # try:
        #     print(dict(zip(unique,counts))[1])
        # except KeyError:
        #     print(0)
        # print(cluster_sorted_activity_list[0:400])

    # Compute the average ranking of each molecule from all the cluster rankings.
    for molecule in final_sorted_activity_list:
        total_ranking = 0
        num_rankings = 0
        for cluster_activity_list in cluster_rankings_list:
            for index,value in enumerate(cluster_activity_list):
                if value["name"] == molecule["name"]:
                    total_ranking+=index
                    num_rankings+=1
        molecule["ranking"] = int(total_ranking/num_rankings)

    # First sort based on the secondary key, the activity in reverse.
    # final_sorted_activity_list = sorted(final_sorted_activity_list,key=get_activity,reverse=True)
    # Then sort based on the primary key, the average ranking.
    final_sorted_activity_list = sorted(final_sorted_activity_list, key=get_ranking)

    # Get the most important features
    important_features_full = []
    with open(descriptor_csv_file,'r') as f_handle:
        reader = csv.reader(f_handle, delimiter=',')
        features = next(reader)
        for cluster_model in molecular_cluster_model:
            important_features = [ [features[index], ((feature_max[index] - feature_min[index]) * cluster_model['centroid'][index]) + feature_min[index]] for index,el in enumerate(cluster_model['subspace']) if el != 0]
            important_features_full.append(important_features)

    with open(descriptor_csv_file,'r') as f_handle:
        reader = csv.reader(f_handle, delimiter=',')
        features_next = next(reader)
        bayes_subspace = [np.float(0)] * len(features_next)
        bayes_centroid = [np.float(0)] * len(features_next)

        with open(bayes_feature_file,'r') as bayes_handle:
            bayes_features = csv.reader(bayes_handle, delimiter=',')
            for row in bayes_features:
                bayes_subspace[features_next.index(row[0])] = 1
                bayes_centroid[features_next.index(row[0])] = (np.float(row[1]) - feature_min[features_next.index(row[0])]) / (feature_max[features_next.index(row[0])] - feature_min[features_next.index(row[0])])
                if not (np.isfinite(bayes_centroid[features_next.index(row[0])])):
                    bayes_centroid[features_next.index(row[0])] = feature_max[features_next.index(row[0])]

    num_common_dimensions_array = []
    centroid_distance_array = []

    for cluster_model in molecular_cluster_model:
        common_dimensions = []
        for i in range(len(bayes_subspace)):
            if (bayes_subspace[i] == 1) and (cluster_model['subspace'][i] == 1):
                common_dimensions.append(1)
            else:
                common_dimensions.append(0)

        num_common_dimensions_array.append(common_dimensions.count(1))
        if (common_dimensions.count(1) != 0):
            c_distance = _compute_subspace_distance(cluster_model['centroid'],bayes_centroid,common_dimensions)/common_dimensions.count(1)
            centroid_distance_array.append(c_distance)

    if single_cluster_model is not None:
        return Scoring.CalcAUC(final_sorted_activity_list, "activity")
    else: 
        return len(molecular_cluster_model), np.mean(num_common_dimensions_array), np.mean(centroid_distance_array), np.std(num_common_dimensions_array), np.std(centroid_distance_array), Scoring.CalcAUC(final_sorted_activity_list, "activity")


# Cluster ranking
def get_AUC_Single(active_cv_molecules, inactive_cv_molecules, molecules_to_fragments, descriptors_map, descriptors, MODEL_DIRECTORY, \
    global_median_cache,used_features,scoring_method, descriptor_csv_file, \
    bayes_subspace = None, bayes_centroid = None, single_cluster_model = None, bayes_feature_file = None):
    
    cluster_rankings_list = []
    final_sorted_activity_list = []
    molecule_fragment_matrices = {}

    if single_cluster_model is not None:
        for test_molecule in active_cv_molecules:
            final_sorted_activity_list.append({"name":test_molecule["name"],"ranking":0,"activity":1})
        for test_molecule in inactive_cv_molecules:
            final_sorted_activity_list.append({"name":test_molecule["name"],"ranking":0,"activity":0})
    else:
        for test_molecule in molecule_names_and_activity:
            final_sorted_activity_list.append({"name":test_molecule["name"],"ranking":0,"activity":test_molecule["activity"]})

    if not ((bayes_subspace is not None) and (bayes_centroid is not None)):
        
        if single_cluster_model is not None:
            molecular_cluster_model = [single_cluster_model]
        else:
            with open(os.path.join(MODEL_DIRECTORY,"molecular_cluster_model.pkl"),'r') as f_handle:
                molecular_cluster_model = pickle.load(f_handle)

        if len(molecular_cluster_model) == 0:
            print "No clusters found in model; can't evaluate any new test molecules..."
            return -1

    molecule_names_and_activity = active_cv_molecules + inactive_cv_molecules

    for test_molecule in molecule_names_and_activity:

        full_fragments = [molecule["fragments"] for molecule in molecules_to_fragments 
                            if molecule["name"] == test_molecule["name"]]

        # First index is actual fragments, since there 
        # can exist only one key value pair for the molecule and its fragments
        full_fragments = full_fragments[0]
        fragments = [fragment["smiles"] for fragment in full_fragments]

        found_fragments = []

        if (bayes_subspace is not None) and (bayes_centroid is not None):
            feature_matrix = np.empty((0,len(bayes_subspace)))
        else:
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

                    if not ((bayes_subspace is not None) and (bayes_centroid is not None)):
                        # Obtain all feature values that have non-numerical values for this fragment
                        nan_descriptors = np.where(np.isfinite(current_fragment) != True)

                        # Impute these non-numerical values with the values from the global median cache
                        # which was again, obtained from the training set.
                        for j in nan_descriptors:
                            current_fragment[0,j] = global_median_cache[0,j]

                        # Finally, project the features of the current fragment into only the non-degenerate
                        # feature space as learned from the training set.
                        current_fragment = current_fragment[:,used_features]

                    # Append this fragment to our feature matrix
                    feature_matrix = np.vstack((feature_matrix,current_fragment))

                except KeyError:
                    print("Key error during AUC calculation!")
                    continue

        if not ((bayes_subspace is not None) and (bayes_centroid is not None)):
            # Normalize the test molecule fragments
            for row in range(feature_matrix.shape[0]):
                for feature in range(feature_matrix.shape[1]):
                    feature_matrix[row,feature] = (feature_matrix[row,feature] - feature_min[feature]) / (feature_max[feature] - feature_min[feature])
                    if not (np.isfinite(feature_matrix[row,feature])):
                        feature_matrix[row,feature] = feature_max[feature]

        molecule_fragment_matrices[test_molecule["name"]] = feature_matrix

    if (bayes_subspace is not None) and (bayes_centroid is not None):

        bayes_sorted_activity_list = []
        distance_array = []

        for test_molecule in molecule_names_and_activity:

            for i in range(molecule_fragment_matrices[test_molecule["name"]].shape[0]):
                current_centroid_distance = _compute_subspace_distance(molecule_fragment_matrices[test_molecule["name"]][i],bayes_centroid,bayes_subspace)
                distance_array.append(current_centroid_distance)
            
            # No fragments are found for this molecule, so we continue since we can't evaluate it.
            if (len(distance_array) == 0):
                continue

            if scoring_method == 1:
                score = np.mean(np.asarray(distance_array))
            else:
                score = np.min(np.asarray(distance_array))

            score = np.around(score, decimals=10)
            bayes_sorted_activity_list.append({"name":test_molecule["name"],"score":score,"activity":test_molecule["activity"]})
        
        bayes_sorted_activity_list = sorted(bayes_sorted_activity_list,key=get_score)
        return Scoring.CalcAUC(bayes_sorted_activity_list,"activity")

    for cluster_model in molecular_cluster_model:
        
        cluster_sorted_activity_list = []
        distance_array = []

        for test_molecule in molecule_names_and_activity:

            # Loop over fragments of the molecule
            for i in range(molecule_fragment_matrices[test_molecule["name"]].shape[0]):
                # closest_centroid_distance = np.min([ _compute_subspace_distance(feature_matrix[i],molecular_cluster_model[j]['centroid'],molecular_cluster_model[j]['subspace']) \
                #     for j in range(len(molecular_cluster_model))])
                
                current_centroid_distance = _compute_subspace_distance(molecule_fragment_matrices[test_molecule["name"]][i],cluster_model['centroid'],cluster_model['subspace'])
                distance_array.append(current_centroid_distance)
            
            # No fragments are found for this molecule, so we continue since we can't evaluate it.
            if (len(distance_array) == 0):
                continue

            if scoring_method == 1:
                score = np.mean(np.asarray(distance_array))
            else:
                score = np.min(np.asarray(distance_array))

            score = np.around(score, decimals=10)
            cluster_sorted_activity_list.append({"name":test_molecule["name"],"score":score})

        cluster_sorted_activity_list = sorted(cluster_sorted_activity_list,key=get_score)
        # Append the current cluster's ranking to the cluster ranking list.
        cluster_rankings_list.append(cluster_sorted_activity_list)
        
        # unique,counts = np.unique(cluster_model['subspace'],return_counts=True)
        
        # print cluster_model['subspace']
        # try:
        #     print(dict(zip(unique,counts))[1])
        # except KeyError:
        #     print(0)
        # print(cluster_sorted_activity_list[0:400])

    # Compute the average ranking of each molecule from all the cluster rankings.
    for molecule in final_sorted_activity_list:
        total_ranking = 0
        num_rankings = 0
        for cluster_activity_list in cluster_rankings_list:
            for index,value in enumerate(cluster_activity_list):
                if value["name"] == molecule["name"]:
                    total_ranking+=index
                    num_rankings+=1
        molecule["ranking"] = int(total_ranking/num_rankings)

    # First sort based on the secondary key, the activity in reverse.
    # final_sorted_activity_list = sorted(final_sorted_activity_list,key=get_activity,reverse=True)
    # Then sort based on the primary key, the average ranking.
    final_sorted_activity_list = sorted(final_sorted_activity_list, key=get_ranking)

    # Get the most important features
    important_features_full = []
    with open(descriptor_csv_file,'r') as f_handle:
        reader = csv.reader(f_handle, delimiter=',')
        features = next(reader)
        for cluster_model in molecular_cluster_model:
            important_features = [ [features[index], ((feature_max[index] - feature_min[index]) * cluster_model['centroid'][index]) + feature_min[index]] for index,el in enumerate(cluster_model['subspace']) if el != 0]
            important_features_full.append(important_features)

    with open(descriptor_csv_file,'r') as f_handle:
        reader = csv.reader(f_handle, delimiter=',')
        features_next = next(reader)
        bayes_subspace = [np.float(0)] * len(features_next)
        bayes_centroid = [np.float(0)] * len(features_next)

        with open(bayes_feature_file,'r') as bayes_handle:
            bayes_features = csv.reader(bayes_handle, delimiter=',')
            for row in bayes_features:
                bayes_subspace[features_next.index(row[0])] = 1
                bayes_centroid[features_next.index(row[0])] = (np.float(row[1]) - feature_min[features_next.index(row[0])]) / (feature_max[features_next.index(row[0])] - feature_min[features_next.index(row[0])])
                if not (np.isfinite(bayes_centroid[features_next.index(row[0])])):
                    bayes_centroid[features_next.index(row[0])] = feature_max[features_next.index(row[0])]

    num_common_dimensions_array = []
    centroid_distance_array = []

    for cluster_model in molecular_cluster_model:
        common_dimensions = []
        for i in range(len(bayes_subspace)):
            if (bayes_subspace[i] == 1) and (cluster_model['subspace'][i] == 1):
                common_dimensions.append(1)
            else:
                common_dimensions.append(0)

        num_common_dimensions_array.append(common_dimensions.count(1))
        if (common_dimensions.count(1) != 0):
            c_distance = _compute_subspace_distance(cluster_model['centroid'],bayes_centroid,common_dimensions)/common_dimensions.count(1)
            centroid_distance_array.append(c_distance)

    if single_cluster_model is not None:
        return Scoring.CalcAUC(final_sorted_activity_list, "activity")
    else: 
        return [len(molecular_cluster_model), np.mean(num_common_dimensions_array), np.mean(centroid_distance_array), np.std(num_common_dimensions_array), np.std(centroid_distance_array), Scoring.CalcAUC(final_sorted_activity_list, "activity")]


def _molecular_model_creation(active_fragments,inactive_fragments,features_map, \
    features_matrix,num_active_molecules,num_inactive_molecules,parameter_dictionary, ALG_TYPE, \
    DIVERSITY_CHECK=True, PURITY_CHECK=True):

    # Retrieve the molecular feature matrix corresponding to our dataset and 
    # flush it to file
    print("Creating molecular feature matrix...")
    [global_median_cache,used_features] = _create_feature_matrix(active_fragments, inactive_fragments,features_map,features_matrix)
    print("Finished molecular feature matrix creation...")

    print "Starting search for molecular clusters..."
    # Find the clusters using ELKI
    molecular_clusters.find_clusters(parameter_dictionary, ALG_TYPE, CLUSTER_FILENAME = os.path.join(DATA_DIRECTORY,"detected_clusters"),
        FEATURE_MATRIX_FILE = os.path.join(config.DATA_DIRECTORY,"molecular_feature_matrix.csv"),
        ELKI_EXECUTABLE=config.ELKI_EXECUTABLE,num_active_molecules=num_active_molecules,num_inactive_molecules=num_inactive_molecules)
        # epsilon=parameter_dictionary["epsilon"],mu_ratio=parameter_dictionary["mu_ratio"], ALG_TYPE)

    print "Finished search for molecular clusters..."

    # Analyze the clusters and output the most pure and diverse ones
    print "Starting analysis and pruning of found clusters..."
    # PURITY_THRESHOLD = .5
    # PURITY_THRESHOLD = parameter_dictionary["PURITY_THRESHOLD"]
    PURITY_THRESHOLD = .5
    DIVERSITY_THRESHOLD = num_active_molecules * .5
    # DIVERSITY_THRESHOLD = num_active_molecules * parameter_dictionary["DIVERSITY_THRESHOLD"]
    DIVERSITY_PERCENTAGE = False
    cluster_analysis.create_cluster_centroid_model(PURITY_THRESHOLD, DIVERSITY_THRESHOLD, DIVERSITY_PERCENTAGE, ALG_TYPE, len(used_features), DIVERSITY_CHECK, PURITY_CHECK)
    print "Finished analysis and pruning of clusters! Clusters' model available in data directory for querying with \
    new test molecules..."

    return [global_median_cache,used_features]

def main():
    sys.path.append(os.path.dirname(sys.argv[1]))
    # config_file = importlib.import_module('params.py')

    # from config_file import *
    from params import *
    training_test_split_file = sys.argv[2]

    MOLECULAR_MODEL_DIRECTORY = os.path.join(DATA_DIRECTORY,"ClustersModel")

    with open(training_test_split_file,"r+") as f_handle:
        training_test_molecules = json.load(f_handle)

    active_training_molecule_names = [molecule["name"] for index,molecule in enumerate(training_test_molecules["data"]["train"]["ligands"]) if (index<=int(.9*len(training_test_molecules["data"]["train"]["ligands"])))]
    inactive_training_molecule_names = [molecule["name"] for index,molecule in enumerate(training_test_molecules["data"]["train"]["decoys"]) if (index<=int(.9*len(training_test_molecules["data"]["train"]["decoys"])))]
    active_cv_molecule_names = [molecule for index,molecule in enumerate(training_test_molecules["data"]["train"]["ligands"]) if (index>int(.9*len(training_test_molecules["data"]["train"]["ligands"])))]
    inactive_cv_molecule_names = [molecule for index,molecule in enumerate(training_test_molecules["data"]["train"]["decoys"]) if (index>int(.9*len(training_test_molecules["data"]["train"]["decoys"])))]

    full_cv_molecules = active_cv_molecule_names + inactive_cv_molecule_names

    with open(actives_fragment_file,"r+") as f_handle:
        actives_molecule_to_fragments = json.load(f_handle)
    with open(inactives_fragment_file,"r+") as f_handle:
        inactives_molecule_to_fragments = json.load(f_handle)

    print("Extracting active and inactive training molecules...")

    active_training_molecules = [molecule for molecule in actives_molecule_to_fragments \
                                    if molecule["name"] in active_training_molecule_names]
    
    inactive_training_molecules = [molecule for molecule in inactives_molecule_to_fragments \
                                    if molecule["name"] in inactive_training_molecule_names]

    
    print("Reading the features file into memory...")

    features_map, features = _read_descriptor_file(features_file)
    
    print("Removing constant features in feature matrix...")

    # Preprocessing: remove constant features
    features = MolecularPreprocessing.remove_constant_features(features)

    print("Creating molecular feature model...")

    if bayes_scoring is not None:

        with open(descriptor_csv_file,'r') as f_handle:
            reader = csv.reader(f_handle, delimiter=',')
            features_next = next(reader)
            bayes_subspace = [np.float(0)] * len(features_next)
            bayes_centroid = [np.float(0)] * len(features_next)

            with open(bayes_model_file,'r') as bayes_handle:
                bayes_features = csv.reader(bayes_handle, delimiter=',')
                for row in bayes_features:
                    bayes_subspace[features_next.index(row[0])] = 1
                    bayes_centroid[features_next.index(row[0])] = np.float(row[1])

                testing_molecules = training_test_molecules["data"]["test"]
                full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

                active_cv_molecules = [molecule for molecule in actives_molecule_to_fragments \
                                    if molecule["name"] in active_training_molecule_names_cv]
    
                inactive_cv_molecules = [molecule for molecule in inactives_molecule_to_fragments \
                                    if molecule["name"] in inactive_training_molecule_names_cv]

                AUC_SCORE = get_AUC(molecule_names_and_activity=testing_molecules,molecules_to_fragments=full_molecules_to_fragments,descriptors_map=features_map,descriptors=features,MODEL_DIRECTORY=MOLECULAR_MODEL_DIRECTORY,global_median_cache=None,used_features=None,scoring_method=1,descriptor_csv_file=descriptor_csv_file,bayes_subspace=bayes_subspace,bayes_centroid=bayes_centroid,single_cluster_model = None, bayes_feature_file = None)
                print("Bayes scoring method 1: ")
                print(AUC_SCORE)
                AUC_SCORE = get_AUC(molecule_names_and_activity=testing_molecules,molecules_to_fragments=full_molecules_to_fragments,descriptors_map=features_map,descriptors=features,MODEL_DIRECTORY=MOLECULAR_MODEL_DIRECTORY,global_median_cache=None,used_features=None,scoring_method=2,descriptor_csv_file=descriptor_csv_file,bayes_subspace=bayes_subspace,bayes_centroid=bayes_centroid,single_cluster_model = None, bayes_feature_file = None)
                print("Bayes scoring method 2: ")
                print(AUC_SCORE)

        return



    with open(results_file,'w+') as f_handle:
        if ALG_TYPE == 'DISH':
            # for mu_ratio in [.2,.4,.6,.8]:
            for mu_ratio in [.2,.6,.9]: # NEW
            # for mu_ratio in [.2]: # 5HT2B
            # for mu_ratio in [.2,.4,.9]: # Another test
            # for mu_ratio in [.4]: # V2R
            # for mu_ratio in [0.8]: # DRD1
                # for epsilon in [.1,.4,.6,.8]:
                for epsilon in [.000000005,.0000005,.000005]: # NEW
                # for epsilon in [.1]: #5HT2B
                # for epsilon in [.1,.5,.7]: # Another test 
                # for epsilon in [.1]: #V2R
                # for epsilon in [0.4]: #DRD1
            # for DIVERSITY_THRESHOLD in [.1,.2,.3,.5,.6,.7,.8,.9,1.0]: # CONCISE THIS FOR EACH DATASET...
            # for DIVERSITY_THRESHOLD in [.2,.6,.9]: # NEW
            # for DIVERSITY_THRESHOLD in [.3]: # 5HT2B
            # for DIVERSITY_THRESHOLD in [.2,.4]: # Another test
            # for DIVERSITY_THRESHOLD in [1.0]: #V2R
            # for DIVERSITY_THRESHOLD in [0.8]: # DRD1
                # for PURITY_THRESHOLD in [.2,.4,.6,.8]:
                # for PURITY_THRESHOLD in [.2,.6,.9]: # NEW
                # for PURITY_THRESHOLD in [.2]: #5HT2B
                # for PURITY_THRESHOLD in [.1,.4,.7]: # Another test
                # for PURITY_THRESHOLD in [.2]: #V2R
                # for PURITY_THRESHOLD in [0.2]: #DRD1
                    for scoring_method in [1,2]: # NEW
                    # for scoring_method in [2]: #5HT2B
                    # for scoring_method in [1,2]: # Another test
                    # for scoring_method in [1]: #V2R 
                    # for scoring_method in [2]: # DRD1
                        for DIVERSITY_CHECK in [True, False]:
                            # for PURITY_CHECK in [True, False]:

                            DIVERSITY_THRESHOLD = .5
                            PURITY_THRESHOLD = .3
                            PURITY_CHECK = True

                            parameter_dictionary = {"DIVERSITY_THRESHOLD":DIVERSITY_THRESHOLD, \
                                "PURITY_THRESHOLD":PURITY_THRESHOLD,"scoring_method":scoring_method,
                                "mu_ratio":mu_ratio,"epsilon":epsilon}

                            # Create the molecular model
                            [global_median_cache, used_features] = _molecular_model_creation(active_training_molecules,inactive_training_molecules,features_map,features,len(active_training_molecules),len(inactive_training_molecules),parameter_dictionary, ALG_TYPE, DIVERSITY_CHECK, PURITY_CHECK)

                            print("Finished creating molecular feature model, beginning testing...")

                            testing_molecules = training_test_molecules["data"]["test"]

                            # Combined active and inactive molecular fragments
                            full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

                            # HERE: Choose the cluster from the centroid model that exhibits the best AUC, that will be our new model.
                            with open(os.path.join(MOLECULAR_MODEL_DIRECTORY,"molecular_cluster_model.pkl"),'r') as model_f_handle:
                                molecular_cluster_model = pickle.load(model_f_handle)

                            best_cluster_model = None
                            best_AUC_score = 0
                            for cluster_model in molecular_cluster_model:
                                AUC_SCORE = get_AUC_Single(active_cv_molecules=active_cv_molecule_names,inactive_cv_molecules=inactive_cv_molecule_names,molecules_to_fragments=full_molecules_to_fragments,descriptors_map=features_map,descriptors=features,MODEL_DIRECTORY=MOLECULAR_MODEL_DIRECTORY,global_median_cache=global_median_cache,used_features=used_features,scoring_method=parameter_dictionary["scoring_method"],descriptor_csv_file=descriptor_csv_file,bayes_subspace=None,bayes_centroid=None,single_cluster_model = cluster_model, bayes_feature_file = bayes_model_file)
                                if AUC_SCORE > best_AUC_score:
                                    best_cluster_model = cluster_model
                                    best_AUC_score = AUC_SCORE

                            # Handle case where no clusters are found
                            if best_cluster_model is None:
                                best_cluster_model = []
                            else:
                                best_cluster_model = [best_cluster_model]

                            with open(os.path.join(MOLECULAR_MODEL_DIRECTORY,"molecular_cluster_model.pkl"),'w+') as model_f_handle:
                                pickle.dump(best_cluster_model, model_f_handle, protocol=pickle.HIGHEST_PROTOCOL)

                            print("Getting AUC Score for current dataset...")
                            # Get the AUC score for the testing data
                            f_handle.write("AUC Score for the current parameters:\n")
                            json.dump(parameter_dictionary, f_handle)
                            f_handle.write("\n")
                            f_handle.write("Diversity Check: ")
                            f_handle.write(str(DIVERSITY_CHECK))
                            f_handle.write(" ")
                            f_handle.write("Purity Check: ")
                            f_handle.write(str(PURITY_CHECK))
                            f_handle.write("\n")
                            AUC_SCORE = get_AUC(molecule_names_and_activity=testing_molecules,molecules_to_fragments=full_molecules_to_fragments,descriptors_map=features_map,descriptors=features,MODEL_DIRECTORY=MOLECULAR_MODEL_DIRECTORY,global_median_cache=global_median_cache,used_features=used_features,scoring_method=parameter_dictionary["scoring_method"],descriptor_csv_file=descriptor_csv_file,bayes_subspace=None,bayes_centroid=None,single_cluster_model = None, bayes_feature_file = bayes_model_file)
                            f_handle.write(str(AUC_SCORE))
                            f_handle.write("\n")
                            f_handle.write("\n")
                            return AUC_SCORE[5]
        elif ALG_TYPE == 'DeLiClu':
            # for mu_ratio in [.2,.4,.6,.8]:
            for minpts_ratio in [.2,.6,.9]: # NEW
            # for mu_ratio in [.2]: # 5HT2B
            # for mu_ratio in [.2,.4,.9]: # Another test
            # for mu_ratio in [.4]: # V2R
            # for mu_ratio in [0.8]: # DRD1
                # for DIVERSITY_THRESHOLD in [.1,.2,.3,.5,.6,.7,.8,.9,1.0]: # CONCISE THIS FOR EACH DATASET...
                for DIVERSITY_THRESHOLD in [.2,.6,.9]: # NEW
                # for DIVERSITY_THRESHOLD in [.3]: # 5HT2B
                # for DIVERSITY_THRESHOLD in [.2,.4]: # Another test
                # for DIVERSITY_THRESHOLD in [1.0]: #V2R
                # for DIVERSITY_THRESHOLD in [0.8]: # DRD1
                    # for PURITY_THRESHOLD in [.2,.4,.6,.8]:
                    for PURITY_THRESHOLD in [.2,.6,.9]: # NEW
                    # for PURITY_THRESHOLD in [.2]: #5HT2B
                    # for PURITY_THRESHOLD in [.1,.4,.7]: # Another test
                    # for PURITY_THRESHOLD in [.2]: #V2R
                    # for PURITY_THRESHOLD in [0.2]: #DRD1
                        for scoring_method in [1,2]: # NEW
                        # for scoring_method in [2]: #5HT2B
                        # for scoring_method in [1,2]: # Another test
                        # for scoring_method in [1]: #V2R 
                        # for scoring_method in [2]: # DRD1
                            for DIVERSITY_CHECK in [True, False]:
                                for PURITY_CHECK in [True, False]:

                                    parameter_dictionary = {"DIVERSITY_THRESHOLD":DIVERSITY_THRESHOLD, \
                                        "PURITY_THRESHOLD":PURITY_THRESHOLD,"scoring_method":scoring_method,
                                        "minpts_ratio":minpts_ratio}

                                    # Create the molecular model
                                    [global_median_cache, used_features] = _molecular_model_creation(active_training_molecules,inactive_training_molecules,features_map,features,len(active_training_molecules),len(inactive_training_molecules),parameter_dictionary, ALG_TYPE, DIVERSITY_CHECK, PURITY_CHECK)

                                    print("Finished creating molecular feature model, beginning testing...")

                                    testing_molecules = training_test_molecules["data"]["test"]

                                    # Combined active and inactive molecular fragments
                                    full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

                                    print("Getting AUC Score for current dataset...")
                                    # Get the AUC score for the testing data
                                    f_handle.write("AUC Score for the current parameters:\n")
                                    json.dump(parameter_dictionary, f_handle)
                                    f_handle.write("\n")
                                    AUC_SCORE = get_AUC(testing_molecules,full_molecules_to_fragments,features_map,features,MOLECULAR_MODEL_DIRECTORY,global_median_cache,used_features,parameter_dictionary["scoring_method"])
                                    f_handle.write(str(AUC_SCORE))
                                    f_handle.write("\n")
        elif ALG_TYPE == 'P3C':
            # for mu_ratio in [.2,.4,.6,.8]:
            for threshold in [.00001,.0000000001,.0000000000000001]: # NEW
            # for mu_ratio in [.2]: # 5HT2B
            # for mu_ratio in [.2,.4,.9]: # Another test
            # for mu_ratio in [.4]: # V2R
            # for mu_ratio in [0.8]: # DRD1
                # for DIVERSITY_THRESHOLD in [.1,.2,.3,.5,.6,.7,.8,.9,1.0]: # CONCISE THIS FOR EACH DATASET...
                for DIVERSITY_THRESHOLD in [.2,.6,.9]: # NEW
                # for DIVERSITY_THRESHOLD in [.3]: # 5HT2B
                # for DIVERSITY_THRESHOLD in [.2,.4]: # Another test
                # for DIVERSITY_THRESHOLD in [1.0]: #V2R
                # for DIVERSITY_THRESHOLD in [0.8]: # DRD1
                    # for PURITY_THRESHOLD in [.2,.4,.6,.8]:
                    for PURITY_THRESHOLD in [.2,.6,.9]: # NEW
                    # for PURITY_THRESHOLD in [.2]: #5HT2B
                    # for PURITY_THRESHOLD in [.1,.4,.7]: # Another test
                    # for PURITY_THRESHOLD in [.2]: #V2R
                    # for PURITY_THRESHOLD in [0.2]: #DRD1
                        for scoring_method in [1,2]: # NEW
                        # for scoring_method in [2]: #5HT2B
                        # for scoring_method in [1,2]: # Another test
                        # for scoring_method in [1]: #V2R 
                        # for scoring_method in [2]: # DRD1
                            for DIVERSITY_CHECK in [True, False]:
                                for PURITY_CHECK in [True, False]:

                                    parameter_dictionary = {"DIVERSITY_THRESHOLD":DIVERSITY_THRESHOLD, \
                                        "PURITY_THRESHOLD":PURITY_THRESHOLD,"scoring_method":scoring_method,
                                        "threshold":threshold}

                                    # Create the molecular model
                                    [global_median_cache, used_features] = _molecular_model_creation(active_training_molecules,inactive_training_molecules,features_map,features,len(active_training_molecules),len(inactive_training_molecules),parameter_dictionary, ALG_TYPE, DIVERSITY_CHECK, PURITY_CHECK)

                                    print("Finished creating molecular feature model, beginning testing...")

                                    testing_molecules = training_test_molecules["data"]["test"]

                                    # Combined active and inactive molecular fragments
                                    full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

                                    print("Getting AUC Score for current dataset...")
                                    # Get the AUC score for the testing data
                                    f_handle.write("AUC Score for the current parameters:\n")
                                    json.dump(parameter_dictionary, f_handle)
                                    f_handle.write("\n")
                                    AUC_SCORE = get_AUC(testing_molecules,full_molecules_to_fragments,features_map,features,MOLECULAR_MODEL_DIRECTORY,global_median_cache,used_features,parameter_dictionary["scoring_method"])
                                    f_handle.write(str(AUC_SCORE))
                                    f_handle.write("\n")
        elif ALG_TYPE == 'HiSC':
            # for mu_ratio in [.2,.4,.6,.8]:
            for k_ratio in [.2,.6,.9]: # NEW
            # for mu_ratio in [.2]: # 5HT2B
            # for mu_ratio in [.2,.4,.9]: # Another test
            # for mu_ratio in [.4]: # V2R
            # for mu_ratio in [0.8]: # DRD1
                # for epsilon in [.1,.4,.6,.8]:
                for alpha in [.2,.6,.9]: # NEW
                # for epsilon in [.1]: #5HT2B
                # for epsilon in [.1,.5,.7]: # Another test 
                # for epsilon in [.1]: #V2R
                # for epsilon in [0.4]: #DRD1
                    # for DIVERSITY_THRESHOLD in [.1,.2,.3,.5,.6,.7,.8,.9,1.0]: # CONCISE THIS FOR EACH DATASET...
                    for DIVERSITY_THRESHOLD in [.2,.6,.9]: # NEW
                    # for DIVERSITY_THRESHOLD in [.3]: # 5HT2B
                    # for DIVERSITY_THRESHOLD in [.2,.4]: # Another test
                    # for DIVERSITY_THRESHOLD in [1.0]: #V2R
                    # for DIVERSITY_THRESHOLD in [0.8]: # DRD1
                        # for PURITY_THRESHOLD in [.2,.4,.6,.8]:
                        for PURITY_THRESHOLD in [.2,.6,.9]: # NEW
                        # for PURITY_THRESHOLD in [.2]: #5HT2B
                        # for PURITY_THRESHOLD in [.1,.4,.7]: # Another test
                        # for PURITY_THRESHOLD in [.2]: #V2R
                        # for PURITY_THRESHOLD in [0.2]: #DRD1
                            for scoring_method in [1,2]: # NEW
                            # for scoring_method in [2]: #5HT2B
                            # for scoring_method in [1,2]: # Another test
                            # for scoring_method in [1]: #V2R 
                            # for scoring_method in [2]: # DRD1
                                for DIVERSITY_CHECK in [True, False]:
                                    for PURITY_CHECK in [True, False]:

                                        parameter_dictionary = {"DIVERSITY_THRESHOLD":DIVERSITY_THRESHOLD, \
                                            "PURITY_THRESHOLD":PURITY_THRESHOLD,"scoring_method":scoring_method,
                                            "k_ratio":k_ratio,"alpha":alpha}

                                        # Create the molecular model
                                        [global_median_cache, used_features] = _molecular_model_creation(active_training_molecules,inactive_training_molecules,features_map,features,len(active_training_molecules),len(inactive_training_molecules),parameter_dictionary, ALG_TYPE, DIVERSITY_CHECK, PURITY_CHECK)

                                        print("Finished creating molecular feature model, beginning testing...")

                                        testing_molecules = training_test_molecules["data"]["test"]

                                        # Combined active and inactive molecular fragments
                                        full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

                                        print("Getting AUC Score for current dataset...")
                                        # Get the AUC score for the testing data
                                        f_handle.write("AUC Score for the current parameters:\n")
                                        json.dump(parameter_dictionary, f_handle)
                                        f_handle.write("\n")
                                        AUC_SCORE = get_AUC(testing_molecules,full_molecules_to_fragments,features_map,features,MOLECULAR_MODEL_DIRECTORY,global_median_cache,used_features,parameter_dictionary["scoring_method"])
                                        f_handle.write(str(AUC_SCORE))
                                        f_handle.write("\n")
        elif ALG_TYPE == 'HiCO':
            # for mu_ratio in [.2,.4,.6,.8]:
            for mu_ratio in [.2,.6,.9]: # NEW
            # for mu_ratio in [.2]: # 5HT2B
            # for mu_ratio in [.2,.4,.9]: # Another test
            # for mu_ratio in [.4]: # V2R
            # for mu_ratio in [0.8]: # DRD1
                # for epsilon in [.1,.4,.6,.8]:
                for alpha in [.2,.6,.9]: # NEW
                # for epsilon in [.1]: #5HT2B
                # for epsilon in [.1,.5,.7]: # Another test 
                # for epsilon in [.1]: #V2R
                # for epsilon in [0.4]: #DRD1
                    # for DIVERSITY_THRESHOLD in [.1,.2,.3,.5,.6,.7,.8,.9,1.0]: # CONCISE THIS FOR EACH DATASET...
                    for DIVERSITY_THRESHOLD in [.2,.6,.9]: # NEW
                    # for DIVERSITY_THRESHOLD in [.3]: # 5HT2B
                    # for DIVERSITY_THRESHOLD in [.2,.4]: # Another test
                    # for DIVERSITY_THRESHOLD in [1.0]: #V2R
                    # for DIVERSITY_THRESHOLD in [0.8]: # DRD1
                        # for PURITY_THRESHOLD in [.2,.4,.6,.8]:
                        for PURITY_THRESHOLD in [.2,.6,.9]: # NEW
                        # for PURITY_THRESHOLD in [.2]: #5HT2B
                        # for PURITY_THRESHOLD in [.1,.4,.7]: # Another test
                        # for PURITY_THRESHOLD in [.2]: #V2R
                        # for PURITY_THRESHOLD in [0.2]: #DRD1
                            for scoring_method in [1,2]: # NEW
                            # for scoring_method in [2]: #5HT2B
                            # for scoring_method in [1,2]: # Another test
                            # for scoring_method in [1]: #V2R 
                            # for scoring_method in [2]: # DRD1
                                for DIVERSITY_CHECK in [True, False]:
                                    for PURITY_CHECK in [True, False]:

                                        parameter_dictionary = {"DIVERSITY_THRESHOLD":DIVERSITY_THRESHOLD, \
                                            "PURITY_THRESHOLD":PURITY_THRESHOLD,"scoring_method":scoring_method,
                                            "mu_ratio":mu_ratio,"alpha":alpha}

                                        # Create the molecular model
                                        [global_median_cache, used_features] = _molecular_model_creation(active_training_molecules,inactive_training_molecules,features_map,features,len(active_training_molecules),len(inactive_training_molecules),parameter_dictionary, ALG_TYPE, DIVERSITY_CHECK, PURITY_CHECK)

                                        print("Finished creating molecular feature model, beginning testing...")

                                        testing_molecules = training_test_molecules["data"]["test"]

                                        # Combined active and inactive molecular fragments
                                        full_molecules_to_fragments = actives_molecule_to_fragments + inactives_molecule_to_fragments

                                        print("Getting AUC Score for current dataset...")
                                        # Get the AUC score for the testing data
                                        f_handle.write("AUC Score for the current parameters:\n")
                                        json.dump(parameter_dictionary, f_handle)
                                        f_handle.write("\n")
                                        AUC_SCORE = get_AUC(testing_molecules,full_molecules_to_fragments,features_map,features,MOLECULAR_MODEL_DIRECTORY,global_median_cache,used_features,parameter_dictionary["scoring_method"])
                                        f_handle.write(str(AUC_SCORE))
                                        f_handle.write("\n")

    
    print("Finished computation of AUCs.")
if __name__ == '__main__':
    main()
    