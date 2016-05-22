import numpy as np
import os
import sys
import csv

def _create_feature_matrix(features_file):
    feature_matrix = None
    molecule_names = []

    with open(features_file,'r') as input_stream:

        reader = csv.reader(input_stream)
        # Gets the first line
        header = next(reader)

        # for line in input_stream:
        for line in reader:
            # line = line.rstrip().split(',')
            name = line[0][1:-1]
                            
            if (feature_matrix is None):
                feature_matrix = np.empty((0, len(header)-1))

            molecule_descriptor_row = np.empty((1, len(header)-1))
            molecule_names.append(name)

            # Next we just copy the newly obtained features into our feature matrix
            # If the feature is not a number, we explicitly input np.nan
            # Need the for loop since we may have NaNs
            for j in range(0, molecule_descriptor_row.shape[1]):
                try:
                    molecule_descriptor_row[0,j] = float(line[j])
                except ValueError:
                    molecule_descriptor_row[0,j] = np.nan
                # except IndexError:
                #     for k in range(j,molecule_descriptor_row.shape[1]):
                #         molecule_descriptor_row[0,k] = np.nan
                #     break

            feature_matrix = np.vstack((feature_matrix, molecule_descriptor_row))
    
    feature_matrix = np.hstack((np.asarray(molecule_names).reshape(feature_matrix.shape[0],1),feature_matrix))
    feature_matrix = np.vstack((np.asarray(header).reshape(1,feature_matrix.shape[1]),feature_matrix))
    return feature_matrix

def remove_constant_features(features_file = None, output_features_file = None):
    if (features_file == None) or (output_features_file == None):
        features_file = sys.argv[1]
        output_features_file = sys.argv[2]

    feature_matrix = _create_feature_matrix(features_file) 

    # V1
    # CONSTANT_FEATURE_REMOVAL_RATIO = .7 
    # all_constant_features = [] 
    # for j in range(feature_matrix.shape[1]): 
    #     feature_column = feature_matrix[:,j].tolist() 
    #     # Count the number of occurences of each value in the feature array for feature_value in feature_column
    #     for feature_value in feature_column:
    #         feature_value_count = feature_column.count(feature_value) 
    #         if feature_value_count >= CONSTANT_FEATURE_REMOVAL_RATIO * len(feature_column):
    #             all_constant_features.append(j) 
    #             break

    # V2
    all_constant_features = []
    for j in range(feature_matrix.shape[1]):
        feature_column = feature_matrix[1:,j]
        if (np.array_equal(feature_column,[feature_column[0]] * len(feature_column))):
            all_constant_features.append(j)

    feature_matrix = np.delete(feature_matrix,all_constant_features,1) 

    # for i in range(0,len(feature_matrix[1])-1):
    #     header_string = str(feature_matrix[1,i]) + ","


    # Save the data manually, NumPy savetxt seems to be giving problems
    with open(output_features_file,'w+') as f_handle:
        for row in feature_matrix:
            for i in range(0,len(row)-1):
                f_handle.write(str(row[i])+",")
            f_handle.write(row[len(row)-1])
            f_handle.write("\n")



    # header_string+=str(feature_matrix[1,feature_matrix.shape[1]-1])
    # print header_string

    # with open(features_file,"wb+") as f_handle:
    #     np.savetxt(f_handle, feature_matrix[1:feature_matrix.shape[0]-1], delimiter=',', fmt="%s "+("%f "*(feature_matrix.shape[1]-2))+"%f", header=header_string)

if __name__ == "__main__":
    remove_constant_features(None,None)