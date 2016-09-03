# from __future__ import division
import ast
import numpy as np
import os
import subprocess
import pickle

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

def extract_clusters_from_file_DiSH(filename):
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
    
    clusters = [cluster for cluster in clusters if not all(v == 0 for v in cluster.get_subspace_mask())]
    return clusters

def extract_clusters_from_file_P3C(filename,dimensions):
    clusters = []

    with open(filename,"r") as f_handle:
        lines = f_handle.read().splitlines()

        for i in range(len(lines)):
            lines[i] = lines[i].lstrip('#')
            lines[i] = lines[i].lstrip(' ')

        i = 0
        while (i != len(lines)):
            while (i < len(lines) and lines[i][0:7]!="Cluster"):
                i+=1

            if (i == len(lines)):
                break

            new_cluster = Cluster()
            point_tuples = []
            point_id_array = [] 

            print(lines[i])
            if(lines[i].split(' ')[1] == 'Noise'):
                # Get to the subspaces
                for j in range(0,3):
                    i+=1  
                subspace_tuple = ()
                for dimension in range(dimensions):
                    subspace_tuple+=(0,)
            else:
                # Regular cluster
                # Define the subspaces for which the cluster is defined
                # Get to the subspaces
                for j in range(0,6):
                    i+=1  
                subspace = ast.literal_eval(lines[i][22:])

                subspace_tuple = ()
                for dimension in range(1,dimensions+1):
                    if dimension in subspace:
                        subspace_tuple += (1,)
                    else:
                        subspace_tuple += (0,)

            new_cluster.set_subspace_mask(subspace_tuple)
            
            while(i < len(lines) and lines[i][0:2]!="ID"):
                i+=1

            while(i < len(lines) and lines[i][0:2] == "ID"):
                string_arr = lines[i].split(' ')
                curr_tuple = ()

                # Add the ID of the fragment vector
                id = string_arr[0].split('=')[1]
                point_id_array.append(int(id))

                for j in range(1,len(string_arr)):
                    try:
                        curr_tuple += (float(string_arr[j]),)
                    except ValueError:
                        curr_tuple += ((string_arr[j]),)

                point_tuples.append(curr_tuple)
                i+=1

            new_cluster.set_points(point_tuples)
            new_cluster.set_id_points(point_id_array)

            clusters.append(new_cluster)

    clusters = [cluster for cluster in clusters if not all(v == 0 for v in cluster.get_subspace_mask())]
    return clusters

def check_subspace_dimensions_match(dim1,dim2):
    for i in range(len(dim1)):
        if dim1[i] != dim2[i]:
            return False
    
    return True

def compute_cluster_centroid(cluster):
    centroid = [0] * len(cluster.get_points()[0])
    for fragment in cluster.get_points():
        centroid = [centroid[i]+fragment_val for i,fragment_val in enumerate(fragment)]
    centroid = [x / len(cluster.get_points()) for x in centroid]
    return centroid

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

def CheckIntersection(cluster1,cluster2):
    "Checks whether the first cluster center and the second cluster center intersect. \
    We define intersect to mean dist(cluster_seed_one[i],cluster_seed_two[i]) <= 2*radius \
    for any i = 0,...,num_clustered_dimensions"

    if not (check_subspace_dimensions_match(cluster1.get_subspace_mask(),cluster2.get_subspace_mask())):
        return [False,0]
    
    centroid_1 = compute_cluster_centroid(cluster1)
    centroid_2 = compute_cluster_centroid(cluster2)
    radius_1 = compute_cluster_radius(cluster1)
    radius_2 = compute_cluster_radius(cluster2)
    radius = np.amax(radius_1,radius_2)
    if (np.absolute(np.array(centroid_1) - np.array(centroid_2)) > 2*radius).any():
        return [False,1]

    return [True,None]

def compare_clusterings(clusters_1,clusters_2):
    for cluster1 in clusters_1:
        for cluster2 in clusters_2:
            if (CheckIntersection(cluster1,cluster2)):
                return True

    return False


def normalize_features(molecule_feature_matrix_file, DATA_DIRECTORY, feature_max=None, feature_min=None):
    
    # Remove any existing temp file
    open(os.path.join(DATA_DIRECTORY,"temp_file"),'w+')
    normalized_feature_matrix = None
    max_feature_array = []
    min_feature_array = []

    with open(os.path.join(DATA_DIRECTORY,molecule_feature_matrix_file),'r') as f_handle:

        if (feature_max is not None) and (feature_min is not None):
            reader = csv.reader(f_handle)
            data_observations_left = True
            while(data_observations_left):
                try:
                    next_observation = next(reader)
                except StopIteration:
                    data_observations_left = False
                    continue
                next_observation = np.asarray(next_observation).astype(np.float)

                # Normalize each feature's value
                for feature in range(len(next_observation)):
                    next_observation[feature] = (next_observation[feature] - feature_min[feature]) / (feature_max[feature] - feature_min[feature])
                    # Handle degenerate cases where, due to some rounding errors, the maximum
                    # and minimum are exactly the same and we therefore get a division by zero.
                    if not (np.isfinite(next_observation[feature])):
                        next_observation[feature] = feature_max[feature]


                # Flush the new normalized vector into the new file
                with open(os.path.join(DATA_DIRECTORY,"temp_file"),'a') as f_handle:
                    np.savetxt(f_handle, next_observation.reshape(1,len(next_observation)), delimiter=',',fmt='%5.5f')
            
            max_feature_array = feature_max
            min_feature_array = feature_min

        else:
            molecule_feature_matrix = np.asarray(np.genfromtxt(os.path.join(DATA_DIRECTORY,molecule_feature_matrix_file), delimiter=',')).astype(np.float)
            normalized_feature_matrix = np.empty(molecule_feature_matrix.shape).reshape(molecule_feature_matrix.shape[0],molecule_feature_matrix.shape[1])
            # Normalize the values of each fragment for each feature
            for feature in range(molecule_feature_matrix.shape[1]):
                # Get the minimum accross the feature values
                max_feature = np.amax(molecule_feature_matrix[:,feature])
                # Get the maximum accross the feature values
                min_feature = np.amin(molecule_feature_matrix[:,feature])
                if max_feature == min_feature:
                    # print("Divide by zero!")
                    # print molecule_feature_matrix[:,feature]
                    # For degenerate features, set all the observations to the same
                    # value in range [0,1] - in this case 1.
                    for fragment in range(molecule_feature_matrix.shape[0]):
                        normalized_feature_matrix[fragment,feature] = 1
                    continue

                # Normalize each fragment's feature value
                for fragment in range(molecule_feature_matrix.shape[0]):
                    normalized_feature_matrix[fragment,feature] = (molecule_feature_matrix[fragment,feature] - min_feature) / (max_feature - min_feature)
            
                max_feature_array.append(max_feature)
                min_feature_array.append(min_feature)

            with open(os.path.join(DATA_DIRECTORY,"temp_file"),'w+') as f_handle:
                np.savetxt(f_handle, normalized_feature_matrix, delimiter=',',fmt='%5.5f')

    # Rename the temporary file as the original matrix, to be consistent
    # TODO:FIND BETTER WORKAROUND FOR THIS
    os.remove(os.path.join(DATA_DIRECTORY,molecule_feature_matrix_file))
    subprocess.call(["mv",os.path.join(DATA_DIRECTORY,"temp_file"),os.path.join(DATA_DIRECTORY,molecule_feature_matrix_file)])
    print(molecule_feature_matrix_file)
    return [max_feature_array,min_feature_array]

def compute_subspace_distance(centroid_one,centroid_two,subspace):
    squared_distance = 0
    for i in range(len(centroid_one)):
        if subspace[i] == 1:
            squared_distance+=(centroid_one[i] - centroid_two[i])**2

    return np.sqrt(squared_distance)


# if __name__ == "__main__":

#     num_intersections = 0
#     num_separate_dimensions = 0
#     num_parallel_clusters = 0

#     result = subprocess.call(["touch","./output_p3c","./output_dish"])

#     for i in range(0,256):

#         with open(os.path.join("../TestFragmentDescriptorData", str(i),"parameters.pkl"),'rb') as f_handle:
#             parameters = pickle.load(f_handle)

#         with open(os.path.join("../TestFragmentDescriptorData", str(i),"generated_test_clusters.pkl"),'rb') as f_handle:
#             clusters_metadata = pickle.load(f_handle)

#         max_feature_vals, min_feature_vals = normalize_features("test_molecular_feature_matrix.csv",os.path.join("../TestFragmentDescriptorData", str(i)))

#         result = subprocess.call(['java', '-jar', "../ELKI/elki-bundle-0.7.0.jar",'KDDCLIApplication','-dbc.in',str(os.path.join("../TestFragmentDescriptorData", str(i), "test_molecular_feature_matrix.csv")),'-dbc.filter', \
#         'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.P3C','-p3c.alpha',str(float(.1)),'-p3c.threshold',str(float(parameters["poisson_threshold"])),'-out',"./output_p3c"])

#         result = subprocess.call(['java', '-jar', "../ELKI/elki-bundle-0.7.0.jar",'KDDCLIApplication','-dbc.in',str(os.path.join("../TestFragmentDescriptorData", str(i), "test_molecular_feature_matrix.csv")),'-dbc.filter', \
#         'FixedDBIDsFilter','-time','-algorithm','clustering.subspace.DiSH','-dish.epsilon',str(float(parameters["epsilon"])),'-dish.mu',str(int(parameters["mu"])),'-out',"./output_dish"])

#         clusters_p3c = extract_clusters_from_file_P3C("./output_p3c",50)
#         clusters_dish = extract_clusters_from_file_DiSH("./output_dish")

#         print("Number of DISH clusters: ")
#         print(len(clusters_dish))
#         print("Number of P3C clusters: ")
#         print(len(clusters_p3c))
#         print("Clusters generated %d\n" % len(clusters_metadata["centroids"]))

#         for cluster1  in clusters_p3c:
#             for cluster2 in clusters_dish:
#                 [result, error_type] = CheckIntersection(cluster1,cluster2)

#                 if result is True:
#                     num_intersections+=1
#                 else: 
#                     if error_type == 0:
#                         num_separate_dimensions+=1
#                     else:
#                         num_parallel_clusters +=1

#     print("Number of Intersections: ")
#     print(num_intersections)
#     print("Separate dimensions: ")
#     print(num_separate_dimensions)
#     print("Parallel Clusters: ")
#     print(num_parallel_clusters)

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

def compute_cluster_extreme_values(cluster):
    cluster_maximum_values = cluster.get_points()[0]
    cluster_minimum_values = cluster.get_points()[0]
    for i in range(len(cluster.get_points())):
        cluster_minimum_values = np.minimum(cluster_minimum_values,cluster.get_points()[i])
        cluster_maximum_values = np.maximum(cluster_maximum_values,cluster.get_points()[i])

    return {"min":cluster_minimum_values,"max":cluster_maximum_values}

if __name__ == "__main__":

    result = subprocess.call(["touch","./output_p3c","./output_dish"])

    for i in range(0,256):

        with open(os.path.join("../TestFragmentDescriptorData", str(i),"parameters.pkl"),'rb') as f_handle:
            parameters = pickle.load(f_handle)

        with open(os.path.join("../TestFragmentDescriptorData", str(i),"generated_test_clusters.pkl"),'rb') as f_handle:
            clusters_metadata = pickle.load(f_handle)

        max_feature_vals, min_feature_vals = normalize_features("test_molecular_feature_matrix.csv",os.path.join("../TestFragmentDescriptorData", str(i)))

        result = subprocess.call(['java', '-jar', "../ELKI/elki-bundle-0.7.0.jar",'KDDCLIApplication','-dbc.in',str(os.path.join("../TestFragmentDescriptorData", str(i), "test_molecular_feature_matrix.csv")),'-dbc.filter', \
        'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.P3C','-p3c.threshold',str(.0000000000000001),'-out',"./output_p3c"])

        clusters_p3c = extract_clusters_from_file_P3C("./output_p3c",100)
        # for cluster in clusters_p3c:
        #     print(cluster.get_subspace_mask())

        # for i in range(clusters_metadata["num_clusters"]):
        #     print clusters_metadata["cluster_subspace_dimensions"][i]

        detected_clusters = 0
        generated_clusters = len(clusters_metadata["centroids"])

        for cluster in clusters_p3c:
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
        with open("./p3c_results",'w+') as p3c_handle:
            p3c_handle.write(str(detected_clusters))
            p3c_handle.write("\n")
            p3c_handle.write(str(generated_clusters))
            p3c_handle.write("\n")
            p3c_handle.write(str(final_score_current_set))
            p3c_handle.write("\n\n")
        print("Done with one! \n")
