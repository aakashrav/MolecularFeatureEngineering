import generate_clusters

def main():

    # Number of clustered dimensions
    num_clustered_dimensions = 3
    # Number of unclustered dimensions
    num_unclustered_dimensions = 3
    # Amount of clusters 
    amount_of_clusters = 11
    # Points per cluster
    amount_of_points = [22,42,5,23,18,10,43,5,12,21,30]
    # Radius (or deviation from the center) in each dimension
    deviation_per_dimension = 10
    #  Range of noise for the unclustered dimensions
    unclustered_noise_range = 100
    # The minimum distance between two clusters
    distance_ratio_between_clusters = 3
    # The maximum distance between two sequentially generated clusters
    max_shifting_range = 200
    # Purity of the cluster; that is, 95% of the cluster's points are pure,
    # and the rest are some noise points
    purity = .95

    clusters = generate_clusters.GenerateSeveralClusters(num_unclustered_dimensions, num_unclustered_dimensions,
        amount_of_clusters, amount_of_points, deviation_per_dimension, unclustered_noise_range,
        distance_ratio_between_clusters, max_shifting_range, purity)
    
    # Label the data points in our clusters with identification keys
    labelled_clusters = generate_clusters.LabelDataWithKeys(clusters)
    
    # Flush the laballed clusters
    generate_clusters.FlushData("labelled_clusters2.csv",labelled_clusters)

    # Add class data about the point (the molecule of the point, its activity, etc.)
    # We would like a ratio of 60% active clusters and 40% inactive clusters
    data_with_classes = generate_clusters.AddClassData(labelled_clusters, ["C15H20O4","Al2Be3O18Si6"], \
        ["C6H10O7","C12H24O2"], .6)

    # Flush the data points
    generate_clusters.FlushClassData("fragment_data_points.csv", data_with_classes)


if __name__ == '__main__':
    main()
