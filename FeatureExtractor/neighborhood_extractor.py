import numpy as np
import correlation_identifier
import config
import os

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def _read_descriptor_file(descriptor_file_name):
    # Read in fragments descriptors into an NP array
    descriptors = None
    with open(descriptor_file_name, 'r') as descriptor_file:
        #Header serves to find out the number of descriptors
        header = descriptor_file.readline().rstrip().split(',')

        descriptors = np.empty((0, len(header)-1), np.float)
        #Adding rows into the NP array one by one is expensive. Therefore we read rows
        #int a python list in batches and regularly flush them into the NP array
        aux_descriptors = []
        ix = 0
        for line in descriptor_file:
            line_split = line.rstrip().split(',')
            aux_descriptors.append([float(x) if isfloat(x) else float('nan') for x in line_split[1:]])
            ix += 1
            if ix % 1000 == 0:
                descriptors = np.vstack((descriptors, np.asarray(aux_descriptors)))
                del aux_descriptors[:]
                break
        if len(aux_descriptors) > 0:
            descriptors = np.vstack((descriptors, np.asarray(aux_descriptors)))
            del aux_descriptors[:]

    return descriptors

def extract_features(NUM_FEATURES, descriptor_matrix, covariance_threshold = .80):
    all_features = np.arange(descriptor_matrix.shape[1])
    # Keep track of the features
    feature_matrix = np.vstack((all_features, descriptor_matrix))

    #Remove all descriptors that have the same value for all fragments, they are degenerate
    all_constant_features = []
    for j in range(feature_matrix.shape[1]):
        feature_column = feature_matrix[1:,j]
        if (np.array_equal(feature_column,[feature_column[0]] * len(feature_column))):
            all_constant_features.append(j)

    significant_features = [feature for feature in all_features if feature not in all_constant_features]
    np.delete(feature_matrix,all_constant_features,1)

    correlation_representatives = correlation_identifier.identify_correlated_features(feature_matrix[1:], NUM_FEATURES,covariance_threshold)
    feature_matrix = feature_matrix[:,correlation_representatives]
    significant_features = feature_matrix[0]

    with open(os.path.join(config.DATA_DIRECTORY,'significant_features'),'wb+') as f_handle:
        np.savetxt(f_handle, significant_features, delimiter=",", fmt="%d")

if __name__ == '__main__':
    feature_matrix = _read_descriptor_file(config.FRAGMENT_FEATURES_FILE)
    extract_features(50,feature_matrix)