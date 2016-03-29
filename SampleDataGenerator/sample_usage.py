import os
import generate_clusters
import config
import shutil

def main():
    
    if os.path.exists(config.TEST_DATA_DIRECTORY):
        shutil.rmtree(config.TEST_DATA_DIRECTORY)
    os.makedirs(config.TEST_DATA_DIRECTORY)

    # Number of clustered dimensions
    num_clustered_dimensions = 3
    # Number of unclustered dimensions
    num_unclustered_dimensions = 3
    # Amount of clusters 
    amount_of_clusters = 11
    # Points per cluster , VERY DENSE CLUSTER
    # amount_of_points = [22,420,533,23,128,140,443,5,12,241,30]
    # Very light cluster
    amount_of_points = [10,14,13,23,12,18,10,10,10,10,14]
    # Radius (or deviation from the center) in each dimension
    deviation_per_dimension = 10
    #  Range of noise for the unclustered dimensions
    unclustered_noise_range = 100
    # The minimum distance between two clusters
    distance_ratio_between_clusters = 3
    # The maximum distance between two sequentially generated clusters
    max_shifting_range = 200
    # Purity of the cluster; that is, 95% of the cluster's points are pure,
    # and the rest are some noise points that don't really belong in the cluster
    purity = .95

    clusters = generate_clusters.GenerateSeveralClusters(num_unclustered_dimensions, num_unclustered_dimensions,
        amount_of_clusters, amount_of_points, deviation_per_dimension, unclustered_noise_range,
        distance_ratio_between_clusters, max_shifting_range, purity)
    
    # Label the data points in our clusters with identification keys
    labelled_clusters = generate_clusters.LabelDataWithKeys(clusters)
    
    # Flush the labelled clusters
    generate_clusters.FlushData(labelled_clusters)
    
    # Add molecular data with values for purity, diversity, etc. for a more realistic dataset
    data_with_classes = generate_clusters.AddMolecularData(labelled_clusters, 10, 40, 7, .6, 2,diversity_percentage=False,difficult_version=True)
    # data_with_classes = generate_clusters.AddMolecularData(labelled_clusters, 10, 40, .5, .6, 2,diversity_percentage=True,difficult_version=True)


if __name__ == '__main__':
    main()
