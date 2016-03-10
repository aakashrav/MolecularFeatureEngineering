import numpy as np
import correlation_identifier

NUM_FEATURES = 200

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

def main():
    feature_matrix = _read_descriptor_file("../features.csv")
    significant_features = correlation_identifier.identify_correlated_features(feature_matrix, NUM_FEATURES)
    with open('significant_features','wb+') as f_handle:
    	print "Writing"
        for feature in significant_features:
        	f_handle.write("%s " % feature)


if __name__ == '__main__':
	main()
