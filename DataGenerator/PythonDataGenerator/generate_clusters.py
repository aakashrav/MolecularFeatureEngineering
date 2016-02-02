import numpy as np
import math
import csv
import random
import math
from sklearn.datasets.samples_generator import make_blobs

# class DataPointClass:
# int key - corresponds to the identification of this specific data point.
# using this key, we can perform an Join on the list of data points to obtain
# the coordinates of this point
#
# string molecule - the specific molecule that this data point belongs to
#
# boolean active_flag - 1 or 0 flag depending on whether this point's molecule
# is an active or inactive
class DataPointClass:
    def __init__(self, init_key, init_molecule, active):
        self.key = int(init_key)
        self.molecule = init_molecule
        self.active_flag = active

# Simple syntactic sugar. Just generates an instance of DataPointClass
# for use in AddClassData
def PlainDataToClassData(key, molecule_name, active_flag):
    new_data_with_class = DataPointClass(key, molecule_name, active_flag)
    return new_data_with_class

# AddClassData
# all_clusters- an array of clusters, each clusters itself an array of data points
#
# active_molecule_list - a list of active molecules to classify the data points within
# the clusters into
#
# inactive_molecule_list - same principle as active_molecule_list but with inactive
# molecules
#
# active_inactive_cluster_ratio - ratio of active vs inactive clusters that the method
# should initialize. The closer to 1 the ratio the higher percentage of clusters
# will be clusters of active fragments.
def AddClassData(all_clusters, active_molecule_list, inactive_molecule_list, 
    active_inactive_cluster_ratio):
    
    # Use the ratio to figure out how many active clusters are needed out of all clusters
    num_clusters = len(all_clusters)
    num_active_clusters = int(math.floor(num_clusters * active_inactive_cluster_ratio))

    data_points_with_classes = []
    
    # Initialize active clusters, with the corresponding data points in this cluster
    # being assigned to active molecules
    for i in range(0, num_active_clusters):
        current_active_cluster = all_clusters[i]

        for j in range(0, len(current_active_cluster)):
            random_molecule_index = random.randint(0,len(active_molecule_list)-1)

            class_vector = PlainDataToClassData(current_active_cluster[j][0], \
                active_molecule_list[random_molecule_index], active_flag=1)

            data_points_with_classes.append(class_vector)

    # Initialize inactive clusters; exactly the same as active clusters, except
    # we use the remaining clusters from the list of all clusters
    for i in range(num_active_clusters, len(all_clusters)):
        current_inactive_cluster = all_clusters[i]

        for j in range(0, len(current_inactive_cluster)):
            random_molecule_index = random.randint(0,len(inactive_molecule_list)-1)

            class_vector = PlainDataToClassData(current_inactive_cluster[j][0], \
                inactive_molecule_list[random_molecule_index], active_flag=0)
            
            data_points_with_classes.append(class_vector)

    return data_points_with_classes

# Method for flushing the identification key of a data point, and the data point itself
def FlushData(output_filename, data):
    # Write the column name entries
    with open(output_filename, 'w') as f_handle:
        f_handle.write("key" + ',')
        for i in range(1, len(data[0][0])):
            if i < len(data[0][0])-1:
                f_handle.write("Dimension " + str(i) + ',')
            else:
                f_handle.write("Dimension " + str(i) + '\n')
    # Append the rest
    with open(output_filename,'a') as f_handle:
        for i in range(0, len(data)):
            np.savetxt(f_handle, data[i], delimiter=",", fmt="%f")

# Method for flushing the key of a data point along with the specific molecule
# and activity class that this data point belongs to. Can then do a Join with
# the raw data points using the identification key and get all the information about
# a specific point.
def FlushClassData(output_filename, data):
    # Write the first entry
    with open(output_filename, 'w') as f_handle:
        f_handle.write("key" + ',' + "molecule" + ',' + "active_flag" + '\n')
    # Append the rest
    with open(output_filename, 'a') as f_handle:
        for i in range(0, len(data)):
            # f_handle.write("%d,%s,%d,\n" %(data[i].key, data[i].molecule,data[i].active_flag))
            f_handle.write(str(data[i].key) + "," + str(data[i].molecule) + "," + str(data[i].active_flag) + "\n")

# Method for labelling each raw data point in each cluster with an identification key
def LabelDataWithKeys(clusters):
    labelled_clusters = []
    counter = 0

    for i in range(0, len(clusters)):
        current_cluster = clusters[i]
        new_cluster = []
        for j in range(0, len(current_cluster)):
            current_vector = current_cluster[j]
            new_vector = np.insert(current_vector,0,counter)
            counter = counter +1
            new_cluster.append(new_vector)
        labelled_clusters.append(new_cluster)
    
    return labelled_clusters

# Generates several subspace clusters, using random seed values for the center
def GenerateSeveralClusters(clusters, clustered_dimensions, unclustered_dimensions, 
    amount_of_clusters, points_per_cluster, deviation_per_dimension, unclustered_subspace_range,
    distance_ratio_between_clusters, max_shifting_range, purity):

    # Generate a starting cluster center, working as a seed to generate future clusters.
    center_seed = []
    random_center_seed = random.randint(0,max_shifting_range)
    # Make sure that the clustered dimensions are quite close to eachother
    for i in range(0, clustered_dimensions):
        center_seed.append(random_center_seed)

    # Compute the cluster's radius
    cluster_radius = ComputeClusterRadiusFromDeviation(deviation_per_dimension, clustered_dimensions)

    # Compute the minimum physical distance between clusters, should be the radius of a cluster
    # multiplied by the distance ration between clusters.
    min_inter_cluster_distance = distance_ratio_between_clusters * cluster_radius

    for j in range(0, amount_of_clusters):
        cluster = []
        # Generate cluster with a specific center
        GenerateSubspaceCluster(cluster, clustered_dimensions, unclustered_dimensions,
            points_per_cluster[j], cluster_radius, center_seed, unclustered_subspace_range,
            purity)
        # Add this cluster to our list of clusters
        clusters.append(cluster)
        # Generate a new center for the next cluster, we will shift the center by our
        # at least our min_inter_cluster_distance but at most our max_shifting_range
        for k in range(0, len(center_seed)):

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
            if ( (center_seed[k] + random_shift) < 0):
                random_shift *= -1

            center_seed[k]+=random_shift

    
# Generate a single subspace cluster using the center_vector seed and a radius
# for the cluster
def GenerateSubspaceCluster(cluster, clustered_dimensions, unclustered_dimensions, points_in_cluster, 
    cluster_radius, center_seed, unclustered_subspace_range, purity):
    # Create a local copy of the center_seed since we will be manipulating
    # the values
    center_vector = []
    center_vector.extend(center_seed)
    #Add some dummy centers for the unclustered dimensions
    for i in range(0, unclustered_dimensions):
        center_vector.append(0)

    # # Cluster_radius corresponds to total Euclidean distance from the center.
    # # Therefore we compute the individual deviation for each dimension (assuming the deviation in each
    # # dimension is uniform)
    # deviation_squared = math.pow(cluster_radius,2)
    # deviation_divided_by_number_of_clusterable_dimensions = deviation_squared / clustered_dimensions
    # final_deviation_per_dimension = math.sqrt(deviation_divided_by_number_of_clusterable_dimensions)

    # Calculate the amount of 'impure' points - points that are clustered on some dimensions
    # of the subspace cluster, but in other dimensions are random. Such points correspond
    # to 'noisy' points or even certain inactive compounds that exhibit some features of the
    # active, but don't exhibit other features.
    num_pure_points = int((purity) * points_in_cluster)
    num_impure_points = points_in_cluster - num_pure_points

    # First generate the pure points in the cluster
    clustered_vectors, labels_true = make_blobs(n_samples=num_pure_points, centers=center_vector,
        cluster_std=cluster_radius, random_state=1)
    
    pre_random_permutation_cluster = []

    # Now add random data for the unclustered dimensions
    for i in range(0, len(clustered_vectors)):
        for k in range(clustered_dimensions,(clustered_dimensions + unclustered_dimensions)):

            random_number_in_range = random.randint(0,unclustered_subspace_range)
            clustered_vectors[i,k] = random_number_in_range

        # Add this final vector to the list of points    
        pre_random_permutation_cluster.append(clustered_vectors[i])

    # Now generate the impure center seed.
    # Randomly choose the number of dimensions deviating from pure points
    # Must be atleast 2 so we get some impurity
    num_impure_dimensions = random.randint(2,clustered_dimensions)
    # Now, generate random center seeds just for the impure dimensions
    for i in range (0, num_impure_dimensions):
        random_number_in_range = random.randint(0,unclustered_subspace_range)
        center_vector[i] = random_number_in_range

    # Now actually generate the impure points
    clustered_vectors, labels_true = make_blobs(n_samples=num_impure_points, centers=center_vector,
        cluster_std=cluster_radius, random_state=1)

    # Now add random data for the unclustered dimensions, just as before
    for i in range(0, len(clustered_vectors)):
        for k in range(clustered_dimensions,(clustered_dimensions + unclustered_dimensions)):

            random_number_in_range = random.randint(0,unclustered_subspace_range)
            clustered_vectors[i,k] = random_number_in_range

        # Add this final vector to the list of points    
        pre_random_permutation_cluster.append(clustered_vectors[i])


    # Shuffle the cluster's dimensions so that we have some variety
    random_permutation = np.arange(clustered_dimensions + unclustered_dimensions)
    np.random.shuffle(random_permutation)

    # Sort through the cluster and map the data points according to this permutation
    # of the dimensions
    for i in range(0, len(pre_random_permutation_cluster)):
        permuted_data_point = []
        PerformPermutationMapping(pre_random_permutation_cluster[i],permuted_data_point, \
            random_permutation)
        cluster.append(permuted_data_point)

    # Return the cluster for good measure
    return cluster

# Perform the permutation on the arr, and deposit contents into newarr
def PerformPermutationMapping(arr,newarr,permutation):
    for j in range(0,len(arr)):
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
    final_deviation = math.ceil(math.sqrt(np.dot(deviation_vector, deviation_vector)))
    # final_deviation = np.dot(deviation_vector, deviation_vector)

    return final_deviation

