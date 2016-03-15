import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from MolecularFeatureEngineering import config
import config
import subprocess

def find_clusters(CLUSTER_FILENAME,FEATURE_MATRIX_FILE,ELKI_EXECUTABLE):
    try:
        os.remove(CLUSTER_FILENAME)
    except OSError:
        pass

    with open(CLUSTER_FILENAME,'w+') as f_handle:
        
        feature_matrix = np.loadtxt(open(FEATURE_MATRIX_FILE,"rb"),delimiter=",",skiprows=0)

        # Compute appropriate parameters mu and epsilon
        max_feature_val_array = feature_matrix.max(axis=0)
        # min_feature_val_array = feature_matrix.min(axis=0)
        # range_feature_val_array = np.subtract(max_feature_val_array,min_feature_val_array)
        percentile_value_array = [(.3 * max_feature) for max_feature in max_feature_val_array]
        # final_epsilon_array = np.add(min_feature_val_array,percentile_value_array)
        
        # Remove all zero values - we don't want an epsilon of zero to be calculated
        percentile_value_array = [i for i in percentile_value_array if i != 0]
        epsilon = np.median(percentile_value_array)
        # Sometimes median is 0, in which case we take the mean
        if epsilon == 0:
            epsilon = np.mean(percentile_value_array)

        epsilon = 10 # FIX HERE
        
        print "Computed epsilon for molecular matrix: %5.5f" % epsilon
        # Mu will be fairly constant, for now. 
        mu = 25
        print "Computed mu for molecular matrix: %d" % mu

        mu = 50 # FIX HERE

        result = subprocess.call(['sudo','java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
        	'FixedDBIDsFilter','-time','-algorithm','clustering.subspace.DiSH','-dish.epsilon',\
        	str(epsilon),'-dish.mu',str(mu),'-out',CLUSTER_FILENAME])

        if result == 0:
        	print "Cluster detection finished!"
        else:
            print "Error in cluster detection"

def main():
    CLUSTER_FILENAME = os.path.join(config.DATA_DIRECTORY,"detected_clusters")
    FEATURE_MATRIX_FILE = os.path.join(config.DATA_DIRECTORY,"molecular_feature_matrix.csv")
    find_clusters(CLUSTER_FILENAME,FEATURE_MATRIX_FILE,config.ELKI_EXECUTABLE)

if __name__ == '__main__':
    main()
