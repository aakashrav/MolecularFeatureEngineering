import numpy as np
import math
import csv
import random
from sklearn.datasets.samples_generator import make_blobs
import config
import pickle
import os
import config
import shutil
import sys

def AddMolecularData(all_clusters, number_of_active_molecules, number_of_inactive_molecules, 
    diversity_threshold, purity_threshold, num_diverse_pure_clusters, DATA_DIRECTORY,
    diversity_percentage = False, difficult_version = False):
    if diversity_percentage:
        diversity_threshold = int(diversity_threshold * number_of_active_molecules)
    current_diverse_pure_clusters = 0
    
    # Prevent parameter combinations that may lead to severe errors
    if (num_diverse_pure_clusters > len(all_clusters)):
            raise ValueError("Current num_diverse_clusters is greater than the actual number of existing clusters provided")
    
    actives_fragments_to_molecule_mapping = {}
    inactives_fragments_to_molecule_mapping = {}
    # active_molecule_list = np.arange(0,number_of_active_molecules).reshape(1,number_of_active_molecules)
    # inactive_molecule_list = np.arange(0,number_of_inactive_molecules).reshape(1,number_of_inactive_molecules)
    
    with open(os.path.join(DATA_DIRECTORY,"generated_test_clusters.pkl"),'w+') as f_handle:

        # Keep track of all metadata for future tests
        generated_clusters = {}
        generated_clusters["num_clusters"] = len(all_clusters)
        generated_clusters["centroids"] = []
        generated_clusters["cluster_radii"] = []
        generated_clusters["num_active_molecules"] = number_of_active_molecules
        generated_clusters["num_inactive_molecules"] = number_of_inactive_molecules
        generated_clusters["diversity"] = diversity_threshold
        generated_clusters["purity"] = purity_threshold
        generated_clusters["num_diverse_pure_clusters"] = num_diverse_pure_clusters
        generated_clusters["clusters"] = []
        generated_clusters["diverse_clusters"] = []
        counter = 0

        for cluster in all_clusters:
            already_assigned_fragments = []
            num_points = len(cluster)
            generated_clusters["clusters"].append([])
            centroid = [0] * len(cluster[0][1:])
            min_vals = cluster[0][1:]
            max_vals = cluster[0][1:]

            if current_diverse_pure_clusters < num_diverse_pure_clusters:
                # First deal with diversity
                # Up to the user to make sure that each test cluster has enough points to
                # even be able to be considered diverse.
                max_fragments = np.min([num_points, diversity_threshold])
                for i in range(max_fragments):
                    actives_fragments_to_molecule_mapping[cluster[i][0]] = [i % number_of_active_molecules]
                    already_assigned_fragments.append(i)
                    generated_clusters["clusters"][counter].append(cluster[i][0])
                    centroid+=cluster[i][1:]
                    min_vals = np.minimum(min_vals,cluster[i][1:])
                    max_vals = np.maximum(max_vals,cluster[i][1:])
            
                # Then deal with purity
                number_of_active_fragments_needed = int(np.ceil(purity_threshold * num_points))
                for i in range(number_of_active_fragments_needed):
                    if i not in already_assigned_fragments:
                        actives_fragments_to_molecule_mapping[cluster[i][0]] = [i % number_of_active_molecules]
                        already_assigned_fragments.append(i)
                        generated_clusters["clusters"][counter].append(cluster[i][0])
                        centroid+=cluster[i][1:]
                        min_vals = np.minimum(min_vals,cluster[i][1:])
                        max_vals = np.maximum(max_vals,cluster[i][1:])

                # Assign the rest to inactives
                for i in range(max_fragments,num_points):
                    if i not in already_assigned_fragments:
                        inactives_fragments_to_molecule_mapping[cluster[i][0]] = [i % number_of_inactive_molecules]
                        generated_clusters["clusters"][counter].append(cluster[i][0])
                        centroid+=cluster[i][1:]
                        min_vals = np.minimum(min_vals,cluster[i][1:])
                        max_vals = np.maximum(max_vals,cluster[i][1:])


                current_diverse_pure_clusters+=1
                generated_clusters["diverse_clusters"].append(counter)
        
            else:
                # In difficult version the rest of the clusters all have only inactive fragments
                if difficult_version:
                    for i in range(num_points):
                        # Choose an inactive molecule at random, doesn't matter which one
                        inactives_fragments_to_molecule_mapping[cluster[i][0]] = [i % number_of_inactive_molecules]
                        generated_clusters["clusters"][counter].append(cluster[i][0])
                        centroid+=cluster[i][1:]
                        min_vals = np.minimum(min_vals,cluster[i][1:])
                        max_vals = np.maximum(max_vals,cluster[i][1:])
            
                # Else in the easy version assign each cluster to actives or inactives randomly
                else:
                    active = np.random.randint(2)
                    if active:
                        for i in range(num_points):
                            # Choose an active molecule at random, doesn't matter which one
                            actives_fragments_to_molecule_mapping[cluster[i][0]] = [i % number_of_active_molecules]
                            generated_clusters["clusters"][counter].append(cluster[i][0])
                            centroid+=cluster[i][1:]
                            min_vals = np.minimum(min_vals,cluster[i][1:])
                            max_vals = np.maximum(max_vals,cluster[i][1:])
                    else:
                        for i in range(num_points):
                            # Choose an inactive molecule at random, doesn't matter which one
                            inactives_fragments_to_molecule_mapping[cluster[i][0]] = [i % number_of_inactive_molecules]
                            generated_clusters["clusters"][counter].append(cluster[i][0])
                            centroid+=cluster[i][1:]
                            min_vals = np.minimum(min_vals,cluster[i][1:])
                            max_vals = np.maximum(max_vals,cluster[i][1:])

            centroid/=num_points
            cluster_radius = 0
            for i in range(len(min_vals)):
                max_distance = np.maximum(centroid[i] - min_vals[i],centroid[i]-max_vals[i])
                cluster_radius+=max_distance**2
            cluter_radius = np.sqrt(cluster_radius) 

            generated_clusters["cluster_radii"].append(cluster_radius)
            generated_clusters["centroids"].append(centroid)
            counter+=1

        pickle.dump(generated_clusters,f_handle,pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(DATA_DIRECTORY,"test_actives_fragment_molecule_mapping.pkl"),'wb+') as f_handle:
         pickle.dump(actives_fragments_to_molecule_mapping, f_handle, pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(DATA_DIRECTORY,"test_inactives_fragment_molecule_mapping.pkl"),'wb+') as f_handle:
        pickle.dump(inactives_fragments_to_molecule_mapping, f_handle, pickle.HIGHEST_PROTOCOL)

# Flush the clusters to file
def FlushData(clusters, DATA_DIRECTORY):
    # Create new file
    open(os.path.join(DATA_DIRECTORY,"test_molecular_feature_matrix.csv"), 'w+')
    with open(os.path.join(DATA_DIRECTORY,"test_molecular_feature_matrix.csv"), 'a') as f_handle:
        for cluster in clusters:
            # Flush everything except the identification key
            new_cluster = np.delete(cluster, 0, 1)
            np.savetxt(f_handle, new_cluster, delimiter=",", fmt="%f")

# Method for labelling each raw data point in each cluster with an identification key
def LabelDataWithKeys(clusters):
    labelled_clusters = []
    counter = 0

    for i in range(clusters.shape[0]):
        current_cluster = clusters[i]
        new_cluster = np.empty([current_cluster.shape[0],current_cluster.shape[1]+1],dtype=np.float)

        for j in range(current_cluster.shape[0]):
            current_vector = current_cluster[j]
            # Insert an identification key
            new_vector = np.insert(current_vector,0,counter)
            # new_cluster.append(new_vector)
            new_cluster[j] = new_vector
            counter = counter +1

        labelled_clusters.append(new_cluster)

    return np.asarray(labelled_clusters)

# Generates several subspace clusters, using random seed values for the center
def GenerateSeveralClusters(clustered_dimensions, unclustered_dimensions, 
    amount_of_clusters, points_per_cluster, deviation_per_dimension, unclustered_subspace_range,
    distance_ratio_between_clusters, max_shifting_range, purity):
    
    clusters = []

    # Generate a starting cluster center, working as a seed to generate future clusters.
    center_seed = np.empty([1,clustered_dimensions], dtype=int)
    random_center_seed = random.randint(0,max_shifting_range)
    # Make sure that the clustered dimensions are quite close to eachother
    center_seed.fill(random_center_seed)

    # Compute the cluster's radius
    cluster_radius = ComputeClusterRadiusFromDeviation(deviation_per_dimension, clustered_dimensions)

    # Compute the minimum physical distance between clusters, should be the radius of a cluster
    # multiplied by the distance ratio between clusters.
    min_inter_cluster_distance = distance_ratio_between_clusters * cluster_radius

    for j in range(amount_of_clusters):
        # Generate cluster with a specific center
        cluster = GenerateSubspaceCluster(clustered_dimensions, unclustered_dimensions,
            points_per_cluster[j], cluster_radius, center_seed, unclustered_subspace_range,
            purity)
        # Add this cluster to our list of clusters
        clusters.append(cluster)
        # Generate a new center for the next cluster, we will shift the center by our
        # at least our min_inter_cluster_distance but at most our max_shifting_range
        for k in range(center_seed.shape[1]):

            # Sometimes user specifies an dimensional distance above the max shifting range.
            # To avoid errors, if this is the case then we just use the max shifting range
            if (min_inter_cluster_distance > max_shifting_range):
                random_shift = max_shifting_range
            else:
                random_shift = random.randint(min_inter_cluster_distance, max_shifting_range)

            # Randomly choose positive or negative shift
            pos_or_neg = random.randint(0,1)
            if pos_or_neg == 1:
                random_shift *= -1

            # If the shift would cause a negative center seed value, 
            # we keep it positive by making the shift positive again
            if ( (center_seed[0,k] + random_shift) < 0):
                random_shift *= -1

            center_seed[0,k]+=random_shift

    return np.asarray(clusters)

    
# Generate a single subspace cluster using the center_vector seed and a radius
# for the cluster
def GenerateSubspaceCluster(clustered_dimensions, unclustered_dimensions, points_in_cluster, 
    cluster_radius, center_seed, unclustered_subspace_range, purity):

    cluster = np.empty([points_in_cluster,clustered_dimensions+unclustered_dimensions],dtype=np.float)

    unclustered_seed = np.zeros([1,unclustered_dimensions],dtype=int)
    center_vector = np.append(center_seed, unclustered_seed)

    # Calculate the amount of 'impure' points - points that are clustered on some dimensions
    # of the subspace cluster, but in other dimensions are random. Such points correspond
    # to 'noisy' points or even certain inactive compounds that exhibit some features of the
    # active, but don't exhibit other features.
    num_pure_points = int((purity) * points_in_cluster)
    num_impure_points = points_in_cluster - num_pure_points

    # First generate the pure points in the cluster
    clustered_vectors, labels_true = make_blobs(n_samples=num_pure_points, centers=center_vector,
        cluster_std=cluster_radius, random_state=1)
    
    pre_random_permutation_cluster = np.empty([num_pure_points + num_impure_points,\
        clustered_vectors.shape[1]],dtype=np.float)

    # Now add random data for the unclustered dimensions
    for i in range(num_pure_points):
        for k in range(clustered_dimensions-1,(clustered_dimensions + unclustered_dimensions)):
            random_number_in_range = random.randint(0,unclustered_subspace_range)
            clustered_vectors[i,k] = random_number_in_range

        # Add this final vector to the list of points    
        # pre_random_permutation_cluster.append(clustered_vectors[i])
        pre_random_permutation_cluster[i] = clustered_vectors[i]

    # Now generate the impure center seed.
    # Randomly choose the number of dimensions deviating from pure points
    # Must be atleast 2 so we get some impurity
    num_impure_dimensions = random.randint(2,clustered_dimensions)
    # Generate random center seeds just for the impure dimensions
    for i in range (num_impure_dimensions):
        random_number_in_range = random.randint(0,unclustered_subspace_range)
        center_vector[i] = random_number_in_range
    # Generate the impure points
    clustered_vectors, labels_true = make_blobs(n_samples=num_impure_points, centers=center_vector,
        cluster_std=cluster_radius, random_state=1)

    # Now add random data for the unclustered dimensions, just as before
    j = 0
    for i in range(num_pure_points, (num_pure_points + num_impure_points)):
        for k in range(clustered_dimensions-1,(clustered_dimensions + unclustered_dimensions)):
            random_number_in_range = random.randint(0,unclustered_subspace_range)
            clustered_vectors[j,k] = random_number_in_range

        # Add this final vector to the list of points 
        pre_random_permutation_cluster[i] = clustered_vectors[j]
        j+=1


    # Shuffle the cluster's dimensions so that we have some variety
    random_permutation = np.random.permutation(np.arange(clustered_dimensions + unclustered_dimensions))

    # Sort through the cluster and map the data points according to this permutation
    # of the dimensions
    for i in range(pre_random_permutation_cluster.shape[0]):
        permuted_data_point = []
        PerformPermutationMapping(pre_random_permutation_cluster[i],permuted_data_point, \
            random_permutation)
        cluster[i] = permuted_data_point

    return cluster

# Perform the permutation on the arr, and deposit contents into newarr
def PerformPermutationMapping(arr,newarr,permutation):
    for j in range(arr.shape[0]):
        newarr.append(arr[permutation[j]])

# The deviation per dimension is only defined on a per dimensional basis, so we
# use the deviation per dimension along with the number of dimensions to calculate
# the total Euclidean distance from the cluster center, where the cluster center
# is w.l.o.g. the origin in our point space.
def ComputeClusterRadiusFromDeviation(deviation_per_dimension, num_dimensions):
    "This helper method computes the total cluster radius of each dimension of the cluster \
    deviates by amount 'deviation_per_dimension'"

    deviation_vector = np.array([1,num_dimensions])
    deviation_vector.fill(deviation_per_dimension)

    # A simple dot product, and square root will calculate the Euclidean distance
    final_deviation = np.ceil(np.sqrt(np.dot(deviation_vector, deviation_vector)))

    return final_deviation

# Generates various test clusters according to the input parameters
# First argument shall be output directory for the cluster feature matrix and the
# cluster IDs and fragments
# Second argument shall be number of clustered dimensions (must be less than 10)
# Third argument shall be the intercluster distance
# Fourth argument should be the density of the clusters (will roughly translate into number of points
# per cluster, assuming a constant cluster deviation per dimension of 10)
# Fifth argument shall be the actives vs inactives molecule ratio, should be less than 1, and will
# be taken as a ratio out of 100 to reflect real scenarios
# The number of clusters is set as  10 + N(0,2), 
def main():

    # Number of clustered dimensions
    num_clustered_dimensions = int(sys.argv[2])
    # Number of unclustered dimensions
    num_unclustered_dimensions = 50 - num_clustered_dimensions
    # The maximum distance between two sequentially generated clusters (intercluster distance)
    max_shifting_range = int(sys.argv[3])
    # The minimum distance between two clusters
    # Argument needed due to backward compatibility, for now we will just keep it the same
    # as maximum shifting range.
    distance_ratio_between_clusters = max_shifting_range
    # Amount of clusters 
    amount_of_clusters = int(10 + np.random.normal(0, 2, 1)[0])
    # Density, or points per cluster
    amount_of_points = [int(sys.argv[4])] * amount_of_clusters 
    # Radius (or deviation from the center) in each dimension
    deviation_per_dimension = 10
    #  Range of noise for the unclustered dimensions - shall be constant
    unclustered_noise_range = 100
    # Purity of the cluster; that is, 95% of the cluster's points are pure,
    # and the rest are some noise points that don't really belong in the cluster
    # Will add extra serendipity to our clustering and we will keep it constant
    purity = .95

    clusters = GenerateSeveralClusters(num_unclustered_dimensions, num_unclustered_dimensions,
        amount_of_clusters, amount_of_points, deviation_per_dimension, unclustered_noise_range,
        distance_ratio_between_clusters, max_shifting_range, purity)
    
    # Label the data points in our clusters with identification keys
    labelled_clusters = LabelDataWithKeys(clusters)
    
    # Flush the labelled clusters
    FlushData(labelled_clusters, sys.argv[1])
    
    num_active_molecules = int(float(sys.argv[5]) * 100)
    num_inactive_molecules = 100 - num_active_molecules
    # Add molecular data with values for purity, diversity, etc. 
    data_with_classes = AddMolecularData(labelled_clusters, num_active_molecules, num_inactive_molecules, 7, .6, 4, sys.argv[1], diversity_percentage=False,difficult_version=True)
    # data_with_classes = generate_clusters.AddMolecularData(labelled_clusters, 10, 40, .5, .6, 2,diversity_percentage=True,difficult_version=True)

if __name__ == '__main__':
    main()
