from __future__ import division
import numpy as np
import os
import pickle

# 30 AND 5!!
# 25 and 10

_dish_clustering_parameters_method = {"silhouette":0, "calinsky":1}


class Cluster:
    def __init__(self):
        self.points = []
        self.num_points = 0

    def set_points(self,point_tuples_array):
        self.points = point_tuples_array
        self.num_points = len(point_tuples_array)

    def set_id_points(self, point_id_array):
        self.point_ids = point_id_array
        #TODO THROW ERROR IF LENGTH OF THE ARRAY NOT THE SAME AS THE NUMBER OF POINTS

    def get_id_points(self):
        return self.point_ids
    
    def set_subspace_mask(self,bitmask_tuple):
        self.subspace_mask = bitmask_tuple
    
    def get_subspace_mask(self):
        return self.subspace_mask

    def get_points(self):
        return self.points

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

def prune_clusters(clusters, f_to_m_dict_file, num_unique_molecules):
    print "Clusters before %d" % (len(clusters))

    with open(f_to_m_dict_file, 'rb') as f_handle:
        f_to_m_dict = pickle.load(f_handle)
    
    degenerate_clusters = []
    
    # First prune clusters that only contain fragments from a few distinct molecules
    for i in range(len(clusters)):
        all_points = clusters[i].get_points()
        unique_molecules = []
        for j in range(len(all_points)):
            if (f_to_m_dict[clusters[i].get_id_points()[j]] not in unique_molecules):
                unique_molecules.append(f_to_m_dict[clusters[i].get_id_points()[j]])
        if len(unique_molecules) < num_unique_molecules:
            degenerate_clusters.append(i)

    # # Then prune clusters that don't occur in intermediate sized subspaces
    # # TODO: Make this more pythonic
    # for i in range(len(clusters)):
    #     dimensionality = 0
    #     for j in clusters[i].get_subspace_mask():
    #         if j == 1:
    #             dimensionality+=1
        
    #     print "dimensionality: %d, min: %d, max: %d\n" % (dimensionality, len(clusters[i].get_subspace_mask())/3, 2*len(clusters[i].get_subspace_mask())/3)
    #     if not(len(clusters[i].get_subspace_mask())/9 < dimensionality < 8* len(clusters[i].get_subspace_mask())/9):
    #         print "Here"
    #         if i not in degenerate_clusters:
    #             degenerate_clusters.append(i)
    
    print degenerate_clusters
    non_degenerate_clusters = [cluster for index, cluster in enumerate(clusters) if index not in degenerate_clusters]

    print "Clusters after %d" % (len(non_degenerate_clusters))
    return non_degenerate_clusters


def prune_degenerate_activity_clusters(active_clusters, inactive_clusters):
    
    print "Active clusters before %d" % (len(active_clusters))

    degenerate_active_clusters = []

    for i in range(len(active_clusters)):
        cluster_subspaces = active_clusters[i].get_subspace_mask()
        for j in range(len(inactive_clusters)):
            inactive_cluster_subspaces = inactive_clusters[j].get_subspace_mask()
            if (cluster_subspaces == inactive_cluster_subspaces):
                degenerate_active_clusters.append(i)
                break

    non_degenerate_active_clusters = [cluster for index, cluster in enumerate(active_clusters) \
                                    if index not in degenerate_active_clusters]
    
    print "Active clusters after %d" % (len(non_degenerate_active_clusters))
    
    return non_degenerate_active_clusters




def main():
    # Actives
    active_clusters = extract_clusters_from_file("sample_output")
    active_clusters = prune_clusters(active_clusters, "actives_fragment_molecule_mapping.pkl", 4)

    # Inactives
    inactive_clusters = extract_clusters_from_file("sample_output_inactives")
    inactive_clusters = prune_clusters(inactive_clusters, "inactives_fragment_molecule_mapping.pkl", 16)

    active_clusters = prune_degenerate_activity_clusters(active_clusters, inactive_clusters)

    # # Actives
    # active_clusters = extract_clusters_from_file("sample_output2")
    # active_clusters = prune_clusters(active_clusters, "actives_fragment_molecule_mapping2.pkl", 10)

    # # Inactives
    # inactive_clusters = extract_clusters_from_file("sample_output_inactives_2")
    # inactive_clusters = prune_clusters(inactive_clusters, "inactives_fragment_molecule_mapping2.pkl", 20)

    # active_clusters = prune_degenerate_activity_clusters(active_clusters, inactive_clusters)

    silhouette_metric = calculate_clustering_metric("silhouette", active_clusters)

    for cluster in active_clusters:
        print(cluster.get_subspace_mask())

    if (silhouette_metric == -2):
        print "Error on silhouette metric %d" % (-2)
    elif (silhouette_metric == -3):
        # TODO, JUST CHECK LENGTH AT BEGINNING AND RETURN
        print "Only one input cluster, silhouette metric irrelevant"
    else:
        print "Silhouette: %d" % (silhouette_metric)

if __name__ == "__main__":
    main()
        
