__author__ = "David Hoksza"
__email__ = "david.hoksza@mff.cuni.cz"
__license__ = 'X11'

import csv
import math
import argparse
import numpy as np
import config

# import matplotlib.pyplot as plt


"""
Processing fragments' features files in the CSV format. Each line
consists of a fragment SMILES and a list of features. Header lists
feature names.
"""


def to_float(x):
    try:
        a = float(x)
        if np.isinf(a): a = float('nan')
    except ValueError:
        return float('nan')
    else:
        return a


def clusters_to_join(clusters, corr_matrix, corr_threshold):
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            correlated = True
            for corr_i in clusters[i]:
                if not correlated: break
                for corr_j in clusters[j]:
                    if abs(corr_matrix[corr_i][corr_j]) < corr_threshold:
                        correlated = False
                        break
            if correlated: return [i, j]

    return [-1, -1]


def process(csv_file_name, out_file_name, log_file_name, corr_threshold):
    features = []
    feature_names = []
    fragment_names = []
    #Reading features from CSV
    for row in csv.reader(open(csv_file_name, "r")):
        if not row: continue
        if len(features) == 0:
            for feature_name in row[1:]:
                feature_names.append(feature_name)
                features.append([])
        else:
            fragment_names.append(row[0])
            for i in range(1, len(row)):
                features[i-1].append(row[i])

    #Convert to numbers
    features = [[to_float(y) for y in x] for x in features]

    #Removal of constant features
    for i in range(len(features)-1, -1, -1):
        if np.isnan(features[i]).all() or np.nanmax(features[i]) - np.nanmin(features[i]) == 0:
            del features[i]
            del feature_names[i]

    #Get medians and impute them
    for i in range(len(features)):
        median = np.median(np.asarray([x for x in features[i] if not math.isnan(x)]))
        features[i] = [median if math.isnan(x) else x for x in features[i]]

    corr_matrix = [[0 for x in range(len(features))] for x in range(len(features))]
    for i in range(len(features)):
        a = np.asarray(features[i])
        for j in range(i, len(features)):
            c = np.corrcoef(a, np.asarray(features[j]))[0, 1]
            corr_matrix[i][j] = c
            corr_matrix[j][i] = c

    #Add each feature into its own cluster
    clusters = []
    for i in range(len(features)):
        clusters.append([i])

    while True:
        [i, j] = clusters_to_join(clusters, corr_matrix, corr_threshold)
        if i == -1: break
        clusters[i] += clusters[j]
        clusters.pop(j)

    #Print clusters
    with open(log_file_name, "w") as fl:
        for cluster in clusters:
            str_cluster = feature_names[cluster[0]] + ": "
            for ix_feature in cluster:
                str_cluster += " " + feature_names[ix_feature]
            fl.write(str_cluster + "\n")

    #Leave out all but first features of each cluster from the feature matrix
    #and print out the resulting matrix
    ixs_features_to_remove = sorted([j for i in clusters for j in i[1:]])
    for ix in reversed(ixs_features_to_remove):
        feature_names.pop(ix)
        features.pop(ix)

    # Get only a subset of all the chosen features, due to time complexity constraints.
    if len(features) > config.NUM_FEATURES:
        features = features[0:config.NUM_FEATURES]
        feature_names = feature_names[0:config.NUM_FEATURES]

    with open(out_file_name, "w") as fo:
        line = "Name"
        for name in feature_names: line += ",{}".format(name)
        fo.write(line + "\n")
        for i in range(len(features[0])):
            line = fragment_names[i]
            for feature in features: line += ",{}".format(feature[i])
            fo.write(line + "\n")

    #fig, ax = plt.subplots()
    #heatmap = ax.pcolor(corrMatrix, cmap=plt.cm.Blues, vmin=0, vmax=2)

    #ax.set_xticklabels(feature_names, minor=False)
    #ax.set_yticklabels(feature_names, minor=False)
    #plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="Features CSV file")
    parser.add_argument("-c", default=0.7, help="Correlation threshold")
    parser.add_argument("-o", required=True, help="Output processed features file")
    parser.add_argument("-l", required=True, help="Log file (features correlation clusters)")
    args = parser.parse_args()


    process(args.f, args.o, args.l, float(args.c))
