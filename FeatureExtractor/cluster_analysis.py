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

_dish_clustering_parameters_method = {"silhouette":0, "calinsky":1}


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


def compute_silhouette(point, cluster, other_clusters, point_index):
    
    dissimilarity_indices = [0,0]

    intracluster_dissimilarity = 0

    # If cluster is a noise cluster, we use all dimensions
    if (any(cluster.get_subspace_mask())):
        projected_point = [value if cluster.get_subspace_mask()[index] == 1 else 0 for index,value in enumerate(point)]
    # else:
    #     For now when we deal with noisy clusters, we just return -1 for the cluster
    #     projected_point = point
    #     return ((-1)/cluster.get_num_points())


    for other_point in [value for index,value in enumerate(cluster.get_points()) if index!=point_index]:

        # If cluster is a noise cluster, we use all dimensions
        if (any(cluster.get_subspace_mask())):
            projected_other_point = [value if cluster.get_subspace_mask()[index] == 1 else 0 for index,value in enumerate(other_point)]
        # else:
        #     projected_other_point = other_point

        intracluster_dissimilarity+= np.linalg.norm(np.subtract(projected_point, projected_other_point))

    intracluster_dissimilarity /= cluster.get_num_points()
    dissimilarity_indices[0] = intracluster_dissimilarity
    
    # Dissimilarity must always be positive, so we ensure that the following code updates this value
    min_alternate_dissimilarity = -1

    for cluster in other_clusters:
        intercluster_dissimilarity = 0
        for other_point in cluster.get_points():

            projected_other_point = [value if cluster.get_subspace_mask()[index] == 1 else 0 for index,value in enumerate(other_point)]

            intercluster_dissimilarity+= np.linalg.norm(np.subtract(projected_point, projected_other_point))

        intercluster_dissimilarity /= cluster.get_num_points()
        
        # We are in the first iteration, so update
        if (min_alternate_dissimilarity == -1):
        	min_alternate_dissimilarity = intercluster_dissimilarity
        # Found a new lowest dissimilarity value
        elif (min_alternate_dissimilarity > intercluster_dissimilarity):
        	min_alternate_dissimilarity = intercluster_dissimilarity
        else:
        	continue
    
    # No update to the min alternate dissimilarity has been made, so we return error value
    if (min_alternate_dissimilarity == -1):
        return -3

    dissimilarity_indices[1] = min_alternate_dissimilarity
    
    silhouette = (dissimilarity_indices[1] - dissimilarity_indices[0])/max(dissimilarity_indices[0], dissimilarity_indices[1])
    
    return silhouette



def calculate_clustering_metric(method,clusters):

    try:
        choice = _dish_clustering_parameters_method[method.lower()]
    except KeyError:
        return -2

    if (choice == 0):

        silhouettes = []

        for i in range(len(clusters)):
            # If the cluster is a noise cluster, move on
            if not(any(clusters[i].get_subspace_mask())):
                print "Skipped!"
                continue
            cluster_silhouettes = [0] * clusters[i].get_num_points()
            other_clusters  = [cluster for index,cluster in enumerate(clusters) if index!=i]
            for j in range(clusters[i].get_num_points()):
                cluster_silhouettes[j] = compute_silhouette(clusters[i].get_points()[j],\
                    clusters[i], other_clusters, j)

            silhouettes.extend(cluster_silhouettes)
        
        try:
            return sum(silhouettes)/len(silhouettes)
        except ZeroDivisionError:
            print "Silhouettes have length 0: %d" % (len(silhouettes))
            return -2
        

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
    purity_threshold = .3):

    print "Clusters before %d" % (len(clusters))

    if percentage:
        diversity_threshold = diversity_threshold * len(active_fragment_molecule_mapping) 

    # Learn a purity threshold weighting from the intrinsic properties of the data
    purity_threshold_weighting = compute_threshold_weighting(active_fragment_molecule_mapping, inactive_fragment_molecule_mapping)
    purity_threshold /= purity_threshold_weighting

    print "Purity threshold with weighting: %f" % purity_threshold
    
    degenerate_clusters = []
    
    # First prune clusters that are not diverse enough with respect to activity
    for i in range(len(clusters)):
        all_points = clusters[i].get_points()
        unique_active_molecules = []
        for j in range(len(all_points)):
            point_fragment_id = clusters[i].get_id_points()[j]
            fragment_name = fragment_name_number_mapping[point_fragment_id]

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
            fragment_name = fragment_name_number_mapping[point_fragment_id]

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

    print "Degenerate clusters: %s" % degenerate_clusters
    significant_clusters = [cluster for index, cluster in enumerate(clusters) if index not in degenerate_clusters]

    print "Clusters after %d" % (len(significant_clusters))
    return significant_clusters


def compute_threshold_weighting(actives_fragment_molecule_mapping, inactives_fragment_molecule_mapping):

    num_active_fragments = len(actives_fragment_molecule_mapping)
    num_inactive_fragments = len(inactives_fragment_molecule_mapping)
    
    # Num inactives -10 
    if (num_inactive_fragments - 10 >  num_active_fragments):
        threshold_weighting = 1 + (math.log10(len(inactive_fragments) - len(active_fragments)))
    else:
        threshold_weighting = 1

    return threshold_weighting


def main():

    DATA_DIRECTORY = config.DATA_DIRECTORY

    CLUSTER_DIRECTORY = os.path.join(DATA_DIRECTORY,"Clusters")
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

    clusters = prune_clusters(clusters, fragment_number_name_mapping, \
        active_fragment_molecule_mapping, inactive_fragment_molecule_mapping,\
         diversity_threshold=config.CLUSTER_DIVERSITY_THRESHOLD, percentage=config.CLUSTER_DIVERSITY_PERCENTAGE, purity_threshold=config.CLUSTER_PURITY_THRESHOLD)
    
    with open(os.path.join(DATA_DIRECTORY,'used_features.pkl'), 'rb') as f_handle:
        used_feature_mapping = pickle.load(f_handle)

    # Read all the descriptor names           
    with open(os.path.join(DATA_DIRECTORY,'all_descriptors.csv')) as f_handle:
        reader = csv.reader(f_handle)
        all_descriptor_names = next(reader)
        all_descriptor_names = np.array(all_descriptor_names)
    
    cluster_count = 1
    for cluster in clusters:
        with open(os.path.join(CLUSTER_DIRECTORY, "Cluster_" + str(cluster_count)), "w+") as f_handle:
            
            significant_indices = np.where(np.array(cluster.get_subspace_mask()) == 1)[0]
            significant_features = [old_descriptor for new_descriptor,old_descriptor in used_feature_mapping.iteritems() \
                                   if new_descriptor in significant_indices]

              
            f_handle.write("Cluster descriptors:\n")
            if len(significant_features) == 0:
                f_handle.write("Noise cluster - no descriptors")
            else:
                for descriptor in all_descriptor_names[significant_features]:
                    f_handle.write("%s " % descriptor)
            f_handle.write('\n\n')
            
            with open(os.path.join(DATA_DIRECTORY,"fragment_number_name_mapping.pkl"), 'rb') as dict_handle:
                fragment_number_name_mapping = pickle.load(dict_handle) 
                f_handle.write("Cluster fragments: \n")
                fragment_count = 0
                for fragment_number in cluster.get_id_points():
                    fragment_name = fragment_number_name_mapping[fragment_number]
                    f_handle.write("Fragment %d: %s\n" % (fragment_count,fragment_name))
                    fragment_count+=1
                f_handle.write('\n\n')


            f_handle.write("Descriptor details:\n")
            cluster_points = np.array(cluster.get_points())
            cluster_points = cluster_points.reshape(len(cluster.get_points()), len(cluster.get_points()[0]))

            cluster_median = np.median(cluster_points, axis=0)
            descriptor_max = np.amax(cluster_points,axis=0)
            descriptor_min = np.amin(cluster_points, axis=0)

            descriptor_count = 0
            for descriptor in all_descriptor_names[significant_features]:
                f_handle.write("%s Median: %5.5f, Max: %5.5f, Min: %5.5f, Range: %5.5f\n\n" % \
                    (descriptor, cluster_median[descriptor_count], descriptor_max[descriptor_count], \
                        descriptor_min[descriptor_count], descriptor_max[descriptor_count] - descriptor_min[descriptor_count]))
                descriptor_count+=1

            cluster_count+=1

    # silhouette_metric = calculate_clustering_metric("silhouette", active_clusters

    # if (silhouette_metric == -2):
    #     print "Error on silhouette metric %d" % (-2)
    # elif (silhouette_metric == -3):
    #     # TODO, JUST CHECK LENGTH AT BEGINNING AND RETURN
    #     print "Only one input cluster, silhouette metric irrelevant"
    # else:
    #     print "Silhouette: %d" % (silhouette_metric)

if __name__ == "__main__":
    main()
        