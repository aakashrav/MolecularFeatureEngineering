import sys
import os
import numpy as np
import config
import subprocess

def find_clusters(parameter_dictionary,ALG_TYPE,CLUSTER_FILENAME,FEATURE_MATRIX_FILE,ELKI_EXECUTABLE,
    num_active_molecules,num_inactive_molecules):
    
    with open(CLUSTER_FILENAME,'w+') as f_handle:
        
        feature_matrix = np.loadtxt(open(FEATURE_MATRIX_FILE,"rb"),delimiter=",",skiprows=0)
        
        # Calculate epsilon and mu for this specific dataset.
        # Epsilon will be the median of the standard deviations of each of the columns in the 
        # feature matrix.
        # standard_deviations_columns = [np.std(feature_matrix[:,i]) for i in range(feature_matrix.shape[1])]
        # epsilon = np.median(standard_deviations_columns)

        # Mu will be the number of active molecules divided by the amount of binding sites
        # For now set binding sites to 5, TODO: Customize this number
        num_binding_sites = 5
        # mu = int(np.ceil(num_active_molecules/num_binding_sites))
        
        # print "Computed epsilon for molecular matrix: %5.5f" % parameter_dictionary['epsilon']
        # print "Computed mu for molecular matrix: %d" % mu
        
        # Call DiSH via ELKI
        if ALG_TYPE == 'DeLiClu':
            minpts = int(np.ceil(num_active_molecules * parameter_dictionary['minpts_ratio']))
            result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
                'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.optics.DeLiClu','-db.index',\
                'de.lmu.ifi.dbs.elki.index.tree.spatial.rstarvariants.deliclu.DeLiCluTreeFactory', '-deliclu.minpts',str(minpts),'-out',CLUSTER_FILENAME])
        elif ALG_TYPE == 'P3C':
            # result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
            #     'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.P3C','-p3c.alpha',\
            #     str(parameter_dictionary['alpha']),'-out',CLUSTER_FILENAME])
            result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
                'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.P3C','-out',CLUSTER_FILENAME])
            result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
                'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.P3C','-p3c.em.maxiter',str(parameter_dictionary['iterations']),'-out',CLUSTER_FILENAME])
        elif ALG_TYPE == 'HiSC':
            k = int(np.ceil(num_active_molecules * parameter_dictionary['k_ratio']))
            result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
                'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.HiSC','-hisc.alpha',\
                str(parameter_dictionary['alpha']),'-hisc.k',str(k),'-out',CLUSTER_FILENAME])
        elif ALG_TYPE == 'HiCO':
            mu = int(np.ceil(num_active_molecules * parameter_dictionary['mu_ratio']))
            result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
                'FixedDBIDsFilter','-time','-algorithm','de.lmu.ifi.dbs.elki.algorithm.clustering.correlation.HiCO','-hico.alpha',\
                str(parameter_dictionary['alpha']),'-hico.mu',str(mu),'-out',CLUSTER_FILENAME])
        else: # Default is DiSH
            mu = int(np.ceil(num_active_molecules * parameter_dictionary['mu_ratio']))
            result = subprocess.call(['java', '-jar', ELKI_EXECUTABLE,'KDDCLIApplication','-dbc.in',FEATURE_MATRIX_FILE,'-dbc.filter', \
            	'FixedDBIDsFilter','-time','-algorithm','clustering.subspace.DiSH','-dish.epsilon',\
            	str(parameter_dictionary['epsilon']),'-dish.mu',str(mu),'-out',CLUSTER_FILENAME])

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
