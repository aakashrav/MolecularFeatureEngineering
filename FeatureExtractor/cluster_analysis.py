from __future__ import division
import numpy as np
import os
import pickle
import math
import csv
import sys
import molecule_feature_matrix
import shutil
import config
import molecular_clusters
from numpy import genfromtxt
import time

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.mlab as mlab


class Cluster:
    def __init__(self):
        self.points = []
        self.num_points = 0

    def set_points(self,point_tuples_array):
        self.points = point_tuples_array
        self.num_points = len(point_tuples_array)

    def get_points(self):
        return self.points

    def set_id_points(self, point_id_array):
        self.point_ids = point_id_array
        #TODO THROW ERROR IF LENGTH OF THE ARRAY NOT THE SAME AS THE NUMBER OF POINTS

    def get_id_points(self):
        return self.point_ids
    
    def set_subspace_mask(self,bitmask_tuple):
        self.subspace_mask = bitmask_tuple
    
    def get_subspace_mask(self):
        return self.subspace_mask

    def get_num_points(self):
        return self.num_points
    
    def get_min_descriptor_values(self):
        return self.min_descriptor_values

    def set_min_descriptor_values(self, new_values):
        self.min_descriptor_values = new_values

    def get_max_descriptor_values(self):
        return self.max_descriptor_values

    def set_max_descriptor_values(self, new_values):
        self.max_descriptor_values = new_values

    def get_descriptor_averages(self):
        return self.descriptor_averages
    
    def set_descriptor_averages(self, new_values):
        self.descriptor_averages = new_values


def extract_clusters_from_file(filename):
    clusters = []

    with open(filename,"r") as f_handle:
        lines = f_handle.read().splitlines()

        for i in range(len(lines)):
            lines[i] = lines[i].lstrip('#')
            lines[i] = lines[i].lstrip(' ')

        i = 0
        while (i != len(lines)):
            if (len(lines[i]) < 8):
                continue
            while (i < len(lines) and lines[i][0:7]!="Cluster"):
                i+=1

            new_cluster = Cluster()
            point_tuples = []
            point_id_array = []

            # Define the subspaces for which the cluster is defined
            cluster_dimensions = lines[i].split(' ')[1]
            cluster_dimensions =  cluster_dimensions.split('_')[1]

            subspace_tuple = ()
            for char in range(len(cluster_dimensions)):
                subspace_tuple += (int(cluster_dimensions[char]),)

            new_cluster.set_subspace_mask(subspace_tuple)
            
            while(i < len(lines) and lines[i][0:2]!="ID"):
                i+=1

            while(i < len(lines) and lines[i][0:2] == "ID"):
                string_arr = lines[i].split(' ')
                curr_tuple = ()

                # Add the ID of the fragment vector as well
                id = string_arr[0].split('=')[1]
                point_id_array.append(int(id))

                for j in range(1,len(string_arr)):
                    curr_tuple += (float(string_arr[j]),)

                point_tuples.append(curr_tuple)
                i+=1

            new_cluster.set_points(point_tuples)
            new_cluster.set_id_points(point_id_array)

            clusters.append(new_cluster)
    
    return clusters

def prune_clusters(clusters, fragment_name_number_mapping, active_fragment_molecule_mapping, inactive_fragment_molecule_mapping, \
    diversity_threshold = 5, percentage = False, \
    purity_threshold = .3, test = False):

    print("Clusters before %d" % (len(clusters)))

    if percentage:
        diversity_threshold = diversity_threshold * len(active_fragment_molecule_mapping) 

    # Learn a purity threshold weighting from the intrinsic properties of the data
    purity_threshold_weighting = compute_threshold_weighting(active_fragment_molecule_mapping, inactive_fragment_molecule_mapping)
    purity_threshold /= purity_threshold_weighting

    print("Purity threshold with weighting: %f" % purity_threshold)
    
    degenerate_clusters = []
    
    # First prune clusters that are not diverse enough with respect to activity
    for i in range(len(clusters)):
        all_points = clusters[i].get_points()
        unique_active_molecules = []
        for j in range(len(all_points)):
            # In test versions we don't have the fragment name to number mapping
            if not test:
                point_fragment_id = clusters[i].get_id_points()[j]
                fragment_name = fragment_name_number_mapping[point_fragment_id]
            else:
                fragment_name = clusters[i].get_id_points()[j]

            try:
                molecules_of_fragment = active_fragment_molecule_mapping[fragment_name]
            except KeyError:
                # Fragment only occurs in inactive molecules
                continue
                
            for molecule in molecules_of_fragment:
                if molecule not in unique_active_molecules:
                    unique_active_molecules.append(molecule)

        if len(unique_active_molecules) < diversity_threshold:
            degenerate_clusters.append(i)

    # Then prune clusters that are not pure enough
    for i in range(len(clusters)):
        all_points = clusters[i].get_points()
        num_actives_in_cluster = 0
        num_inactives_in_cluster = 0
        for j in range(len(all_points)):
            point_fragment_id = clusters[i].get_id_points()[j]
            
            if not test:
                point_fragment_id = clusters[i].get_id_points()[j]
                fragment_name = fragment_name_number_mapping[point_fragment_id]
            else:
                fragment_name = clusters[i].get_id_points()[j]

            try:
                molecules_of_fragment = active_fragment_molecule_mapping[fragment_name]
                num_actives_in_cluster+=len(molecules_of_fragment)
            except KeyError:
                # Clear exception data and continue
                sys.exc_clear()

            try:
                molecules_of_fragment = inactive_fragment_molecule_mapping[fragment_name]
                num_inactives_in_cluster+=len(molecules_of_fragment)
            except KeyError:
                continue
        
        cluster_activity = num_actives_in_cluster/(num_actives_in_cluster + num_inactives_in_cluster)
        if (cluster_activity < purity_threshold):
            degenerate_clusters.append(i)
    
    # Get final unique list of all non-significant clusters
    degenerate_clusters = np.unique(degenerate_clusters)

    print("Degenerate clusters: %s" % degenerate_clusters)
    significant_clusters = [cluster for index, cluster in enumerate(clusters) if index not in degenerate_clusters]

    print("Clusters after %d" % (len(significant_clusters)))
    return significant_clusters


def compute_threshold_weighting(actives_fragment_molecule_mapping, inactives_fragment_molecule_mapping):

    num_active_fragments = len(actives_fragment_molecule_mapping)
    num_inactive_fragments = len(inactives_fragment_molecule_mapping)
    
    # Num inactives -10 
    if (num_inactive_fragments - 10 >  num_active_fragments):
        threshold_weighting = 1 + (math.log10(num_inactive_fragments - num_active_fragments))
    else:
        threshold_weighting = 1

    return threshold_weighting

def compute_cluster_centroid(cluster):
    centroid = [0] * len(cluster.get_points()[0])
    for fragment in cluster.get_points():
        centroid = [centroid[i]+fragment_val for i,fragment_val in enumerate(fragment)]
    centroid = [x / len(cluster.get_points()) for x in centroid]
    return centroid

def compute_subspace_distance(centroid_one,centroid_two,subspace):
    squared_distance = 0
    for i in range(len(centroid_one)):
        if subspace[i] == 1:
            squared_distance+=(centroid_one[i] - centroid_two[i])**2

    return np.sqrt(squared_distance)

def compute_cluster_radius(cluster):
    cluster_centroid = compute_cluster_centroid(cluster)
    cluster_maximum_values = cluster.get_points()[0]
    cluster_minimum_values = cluster.get_points()[0]
    for i in range(len(cluster.get_points())):
        cluster_minimum_values = np.minimum(cluster_minimum_values,cluster.get_points()[i])
        cluster_maximum_values = np.maximum(cluster_maximum_values,cluster.get_points()[i])

    cluster_radius = 0
    for i in range(len(cluster_maximum_values)):
        if cluster.get_subspace_mask()[i] == 1:
            max_distance = np.maximum(np.absolute(cluster_centroid[i] - cluster_minimum_values[i]),np.absolute(cluster_centroid[i]-cluster_maximum_values[i]))
            cluster_radius+=max_distance**2
    
    return np.sqrt(cluster_radius) 

def compute_cluster_extreme_values(cluster):
    cluster_maximum_values = cluster.get_points()[0]
    cluster_minimum_values = cluster.get_points()[0]
    for i in range(len(cluster.get_points())):
        cluster_minimum_values = np.minimum(cluster_minimum_values,cluster.get_points()[i])
        cluster_maximum_values = np.maximum(cluster_maximum_values,cluster.get_points()[i])

    return {"min":cluster_minimum_values,"max":cluster_maximum_values}

def check_cube_intersection(cluster1extremes,cluster2extremes,cluster_subspace_mask):
    counter=0
    for i in range(len(cluster1extremes["min"])):
        if i in cluster_subspace_mask:
            if ((cluster1extremes["max"][i]>=cluster2extremes["min"][i]) and (cluster1extremes["max"][i]<=cluster2extremes["max"][i])) \
            or ((cluster2extremes["max"][i]>=cluster1extremes["min"][i]) and (cluster2extremes["max"][i]<=cluster1extremes["max"][i])):
                counter+=1
            else:
                break
    if counter == len(cluster1extremes["min"])-1:
        return True
    else:
        return False

def check_subspace_dimensions_match(list,tuple):
    print(tuple)
    for i in range(len(tuple)):
        if tuple[i] != list[0][i]:
            return False
    
    return True


def create_cluster_centroid_model(purity_threshold, diversity_threshold, diversity_percentage):

    DATA_DIRECTORY = config.DATA_DIRECTORY

    CLUSTER_DIRECTORY = os.path.join(DATA_DIRECTORY,"ClustersModel")
    if os.path.exists(CLUSTER_DIRECTORY):
        shutil.rmtree(CLUSTER_DIRECTORY, ignore_errors=True)
    os.makedirs(CLUSTER_DIRECTORY)

    clusters = extract_clusters_from_file(os.path.join(DATA_DIRECTORY,"detected_clusters"))

    with open(os.path.join(DATA_DIRECTORY,"fragment_number_name_mapping.pkl"), 'rb') as f_handle:
        fragment_number_name_mapping = pickle.load(f_handle)

    with open(os.path.join(DATA_DIRECTORY,"actives_fragment_molecule_mapping.pkl"), 'rb') as f_handle:
        active_fragment_molecule_mapping = pickle.load(f_handle)
    
    with open(os.path.join(DATA_DIRECTORY,"inactives_fragment_molecule_mapping.pkl"),'rb') as f_handle:
        inactive_fragment_molecule_mapping = pickle.load(f_handle)

    # clusters = prune_clusters(clusters, fragment_number_name_mapping, \
    #     active_fragment_molecule_mapping, inactive_fragment_molecule_mapping,\
    #      diversity_threshold=diversity_threshold, percentage=diversity_percentage, purity_threshold=purity_threshold)

    molecular_cluster_model = []
    for cluster in clusters:
        molecular_cluster_model.append({'centroid':compute_cluster_centroid(cluster),'subspace':cluster.get_subspace_mask()})

    with open(os.path.join(CLUSTER_DIRECTORY,'molecular_cluster_model.pkl'),'w+') as f_handle:
        pickle.dump(molecular_cluster_model, f_handle, pickle.HIGHEST_PROTOCOL)

def dish_main():

    DATA_DIRECTORY = config.TEST_DATA_DIRECTORY

    with open('./final_clustering_score','w+') as f_handle:
        f_handle.write("ICD,Density,Epsilon,Mu,Score\n")
    
    # Process each genereated test clusters and dump evaluation metrics
    final_score = 0
    dataset_counter = 0

    # Keep track of which values acheived the highest and lowest scores, respectively.
    maximum_score = 0
    maximum_score_params = []
    minimum_score = 1
    minimum_score_params = []

    PARAM_1 = "epsilon"
    PARAM_2 = "mu"

    scatterplot_x = [.002,.02,.03,.04]
    scatterplot_y = [3,6,9,10]
    scatterplot_scores = np.zeros((4,4))
    scatterplot_scores_num = np.zeros((4,4))

    dataset_scores = []

    row_dict = {.002:0,.02:1,.03:2,.04:3}
    col_dict = {3:0,6:1,9:2,10:3}

    for subdir in os.listdir(DATA_DIRECTORY):
        CURRENT_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,subdir)

        with open(os.path.join(CURRENT_DATA_DIRECTORY,"generated_test_clusters.pkl"),'rb') as f_handle:
            clusters_metadata = pickle.load(f_handle)
        
        with open(os.path.join(CURRENT_DATA_DIRECTORY,"test_molecular_feature_matrix.csv"),'r') as f_handle:
            feature_matrix = genfromtxt(f_handle, delimiter=',')

        max_feature_vals, min_feature_vals = molecule_feature_matrix._normalize_features(os.path.join(CURRENT_DATA_DIRECTORY,"test_molecular_feature_matrix.csv"),CURRENT_DATA_DIRECTORY,None,None)

        with open(os.path.join(CURRENT_DATA_DIRECTORY,"parameters.pkl"),'r') as f_handle:
            parameters = pickle.load(f_handle)

        molecular_clusters.find_clusters(os.path.join(CURRENT_DATA_DIRECTORY,"detected_clusters"),
            os.path.join(CURRENT_DATA_DIRECTORY,"test_molecular_feature_matrix.csv"),config.ELKI_EXECUTABLE,
            num_active_molecules=clusters_metadata["num_active_molecules"],num_inactive_molecules=clusters_metadata["num_inactive_molecules"],
             epsilon = float(parameters["epsilon"]), mu=int(parameters["mu"]))
        
        # Get all the detected clusters
        clusters = extract_clusters_from_file(os.path.join(CURRENT_DATA_DIRECTORY,"detected_clusters"))

        with open(os.path.join(CURRENT_DATA_DIRECTORY,"test_actives_fragment_molecule_mapping.pkl"), 'rb') as f_handle:
            active_fragment_molecule_mapping = pickle.load(f_handle)
    
        with open(os.path.join(CURRENT_DATA_DIRECTORY,"test_inactives_fragment_molecule_mapping.pkl"),'rb') as f_handle:
            inactive_fragment_molecule_mapping = pickle.load(f_handle)

        
        print("Clusters detected %d" % len(clusters))
        print("Clusters generated %d" % len(clusters_metadata["centroids"]))

        detected_clusters = 0
        generated_clusters = (len(clusters_metadata["centroids"]))

        for cluster in clusters:
            cluster_centroid = compute_cluster_centroid(cluster)
            denormalized_cluster_centroid = []
            for i in range(len(cluster_centroid)):
                denormalized_cluster_centroid.append((cluster_centroid[i] * (max_feature_vals[i] - min_feature_vals[i])) + min_feature_vals[i])
            cluster_centroid = denormalized_cluster_centroid

            intersecting_centroids = [index for index,val in enumerate(clusters_metadata["cluster_subspace_dimensions"]) if check_subspace_dimensions_match(val.tolist(),cluster.get_subspace_mask())]
            if intersecting_centroids == []:
                continue
            nearest_centroid_distance = compute_subspace_distance(cluster_centroid, clusters_metadata["centroids"][intersecting_centroids[0]],cluster.get_subspace_mask())
            minimum_centroid_index=intersecting_centroids[0]
            for i in range(1,len(intersecting_centroids)):

                # Check that the clusters actually intersect (i.e. are not parallel clusters in the same subspace)
                current_centroid_distance = compute_subspace_distance(cluster_centroid, clusters_metadata["centroids"][intersecting_centroids[i]],cluster.get_subspace_mask())

                if check_cube_intersection(compute_cluster_extreme_values(cluster),clusters_metadata["cluster_extreme_values"][intersecting_centroids[i]],cluster.get_subspace_mask()) and \
                current_centroid_distance < nearest_centroid_distance:
                    nearest_centroid_distance = current_centroid_distance
                    minimum_centroid_index = intersecting_centroids[i]

            print("Detected a cluster! Centroid distance: %d", nearest_centroid_distance)

            del clusters_metadata["centroids"][minimum_centroid_index]
            del clusters_metadata["cluster_subspace_dimensions"][minimum_centroid_index]
            clusters_metadata["cluster_extreme_values"]["min"] = np.delete(clusters_metadata["cluster_extreme_values"]["min"], minimum_centroid_index)
            clusters_metadata["cluster_extreme_values"]["max"] = np.delete(clusters_metadata["cluster_extreme_values"]["max"], minimum_centroid_index)

            detected_clusters+=1

        final_score_current_set = float(detected_clusters/generated_clusters)

        if final_score_current_set > maximum_score:
            maximum_score = final_score_current_set
            maximum_score_params = []
            maximum_score_params.append({'icd':parameters["icd"],'density':parameters["density"],'epsilon':parameters["epsilon"],'mu':parameters["mu"]})
        elif final_score_current_set == maximum_score:
            maximum_score_params.append({'icd':parameters["icd"],'density':parameters["density"],'epsilon':parameters["epsilon"],'mu':parameters["mu"]})
        elif final_score_current_set < minimum_score:
            minimum_score = final_score_current_set
            minimum_score_params = []
            minimum_score_params.append({'icd':parameters["icd"],'density':parameters["density"],'epsilon':parameters["epsilon"],'mu':parameters["mu"]})
        elif final_score_current_set == minimum_score:
            minimum_score_params.append({'icd':parameters["icd"],'density':parameters["density"],'epsilon':parameters["epsilon"],'mu':parameters["mu"]})
        else:
            pass

        print("Final score for current set of clusters %.2f" % final_score_current_set)
        
        scatterplot_scores_num[row_dict[parameters[PARAM_1]],col_dict[parameters[PARAM_2]]] +=1
        scatterplot_scores[row_dict[parameters[PARAM_1]],col_dict[parameters[PARAM_2]]] += final_score_current_set

        dataset_scores.append(final_score_current_set)

        with open('./final_clustering_score','a') as f_handle:
            f_handle.write("%f,%f,%f,%f,%f\n" % (parameters["icd"],parameters["density"],parameters["epsilon"],parameters["mu"],final_score_current_set))

    print("Final mean score over all datasets: %.2f\n" % np.mean(dataset_scores))
    print("Final variance score over all datasets %.2f\n" % np.var(dataset_scores))
    print("Final maximum score over all datasets %.2f\n" % np.amax(dataset_scores))
    print("Final minimum score over all datasets % .2f\n" % np.amin(dataset_scores))

    with open('./final_clustering_score','a') as f_handle:
            f_handle.write("\nMaximum score and corresponding parameters:\n")
            for parameters in maximum_score_params:
                f_handle.write("%f,%f,%f,%f, Score: %f\n" % (parameters["icd"],parameters["density"],parameters["epsilon"],parameters["mu"], maximum_score))

            f_handle.write("\nMinimum score and corresponding parameters:\n")
            for parameters in minimum_score_params:
                f_handle.write("%f,%f,%f,%f, Score: %f\n" % (parameters["icd"],parameters["density"],parameters["epsilon"],parameters["mu"], minimum_score))

    # Take the average
    scatterplot_scores = scatterplot_scores/scatterplot_scores_num

    print(scatterplot_x)
    print(scatterplot_y)

    # #setup the 2D grid with Numpy
    # x, y = np.meshgrid(scatterplot_x, scatterplot_y)

    # plt.xlabel('Epsilon')
    # plt.ylabel('NumPoints')

    # #now just plug the data into pcolormesh, it's that easy!
    # plt.pcolormesh(x, y, scatterplot_scores)
    # plt.colorbar() #need a colorbar to show the intensity scale
    # plt.show() #boom

    # plt.clf()

    # # The histogram of the data
    # n, bins, patches = plt.hist(dataset_scores, 50, normed=1, facecolor='green', alpha=0.75)
    # # plt.scatter(scatterplot_x,scatterplot_y)


    # plt.xlabel('Score')
    # plt.ylabel('Datasets')
    # plt.title('Histogram of the number of datasets achieving each score')
    # plt.axis([np.amin(dataset_scores), np.amax(dataset_scores),0,len(dataset_scores)/4])
    # plt.grid(True)

    # plt.show()

if __name__ == "__main__":
    dish_main()
    # create_cluster_centroid_model(.6, 20, False)
        
