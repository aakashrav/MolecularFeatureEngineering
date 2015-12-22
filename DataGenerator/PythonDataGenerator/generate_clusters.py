import numpy as np
import math
import random
from sklearn.datasets.samples_generator import make_blobs

def GenerateSubspaceCluster(clusters, clustered_dimensions, unclustered_dimensions, amount, 
    total_deviation, center_vector, unclustered_subspace_range):

    #Add some dummy centers for the unclustered dimensions
    for i in range(0, unclustered_dimensions):
        center_vector.append(0)
    # Total_deviation corresponds to total Euclidean distance from the center.
    # Therefore we compute the individual deviation for each dimension (assuming the deviation in each
    # dimension is uniform)
    deviation_squared = math.pow(total_deviation,2)
    deviation_divided_by_number_of_clusterable_dimensions = deviation_squared / clustered_dimensions
    final_deviation_per_dimension = math.sqrt(deviation_divided_by_number_of_clusterable_dimensions) 

    clustered_vectors, labels_true = make_blobs(n_samples=amount, centers=center_vector,
        cluster_std=final_deviation_per_dimension, random_state=1)

    # Now add random noise for the unclustered dimensions
    for i in range(1, len(clustered_vectors)):
        for k in range(clustered_dimensions,(clustered_dimensions + unclustered_dimensions)):

            random_number_in_range = random.randint(0,unclustered_subspace_range)
            clustered_vectors[i,k] = random_number_in_range

        # Add this final vector to the list of points    
        clusters.append(clustered_vectors[i])

    #Return the cluster for good measure
    return clusters

def main():
    clusters = []
    cluster_center = [5,5,5]
    num_clustered_dimensions = 3
    num_unclustered_dimensions = 3
    amount_of_points = 20
    total_center_deviation = 4
    unclustered_noise_range  = 100

    GenerateSubspaceCluster(clusters,num_clustered_dimensions,num_unclustered_dimensions,amount_of_points,
        total_center_deviation, cluster_center, unclustered_noise_range)

    for j in range(1,len(clusters)):
        print clusters[j]

if __name__ == '__main__':
    main()

