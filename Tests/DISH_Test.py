import config
import os
import numpy as numpy
import pickle


## TODO: IMPORT THIS ##
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

    print "Degenerate clusters: %s" % degenerate_clusters
    significant_clusters = [cluster for index, cluster in enumerate(clusters) if index not in degenerate_clusters]

    print "Clusters after %d" % (len(significant_clusters))
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

## TODO: IMPORT THIS ##

def main():

    DATA_DIRECTORY = config.TEST_DATA_DIRECTORY
    
    # Process each genereated test clusters and dump evaluation metrics
    for directory in os.walk(DATA_DIRECTORY):
        CURRENT_DATA_DIRECTORY = directory[0]

        with open(os.path.join(CURRENT_DATA_DIRECTORY,"generated_test_clusters.pkl"),'rb') as f_handle:
            clusters_metadata = pickle.load(f_handle)

        normalize_features(os.path.join(CURRENT_DATA_DIRECTORY,"test_molecular_feature_matrix.csv"),CURRENT_DATA_DIRECTORY,None,None)

        molecular_clusters.find_clusters(os.path.join(CURRENT_DATA_DIRECTORY,"detected_clusters"),
            os.path.join(CURRENT_DATA_DIRECTORY,"test_molecular_feature_matrix.csv"),config.ELKI_EXECUTABLE,
            num_active_molecules=clusters_metadata["num_active_molecules"],num_inactive_molecules=clusters_metadata["num_inactive_molecules"],
             epsilon = .3, mu=10)
        
        # Get all the detected clusters
        clusters = extract_clusters_from_file(os.path.join(CURRENT_DATA_DIRECTORY,"detected_clusters"))

        with open(os.path.join(CURRENT_DATA_DIRECTORY,"test_actives_fragment_molecule_mapping.pkl"), 'rb') as f_handle:
            active_fragment_molecule_mapping = pickle.load(f_handle)
    
        with open(os.path.join(CURRENT_DATA_DIRECTORY,"test_inactives_fragment_molecule_mapping.pkl"),'rb') as f_handle:
            inactive_fragment_molecule_mapping = pickle.load(f_handle)

        # Get the pruned clusters
        pruned_clusters = prune_clusters(clusters, None, \
            active_fragment_molecule_mapping, inactive_fragment_molecule_mapping,\
            diversity_threshold=7, percentage=False, purity_threshold=.6,test=True)

        with open(config.TEST_STATISTICS_FILE,'w+') as f_handle:
            f_handle.write("All clusters found: \n")
            for index,cluster in enumerate(clusters):
                f_handle.write("%d " % index)
            f_handle.write("\n")

            f_handle.write("Degenerate clusters found:\n"):
                found_denerate_clusters = [index for index,cluster in enumerate(clusters) if cluster not in pruned_clusters]
                for i in found_denerate_clusters:
                    f_handle.write("%d " % i)
                f_handle.write("\n")

            f_handle.write("Remaining significant clusters:\n"):
                for index,cluster in pruned_clusters:
                    f_handle.write("Fragments: \n")
                    for fragment_id in cluster.get_id_points():
                        f_handle.write("%d " % fragment_id)

                    f_handle.write("\n")

                    # Check to see how close the cluster is to the nearest actual cluster
                    # Via computing centroid and radius.

if __name__ == "__main__":
    main()
        