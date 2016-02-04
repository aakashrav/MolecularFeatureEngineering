__author__ = 'David Hoksza'

import os
import json
import numpy as np
from rdkit import Chem
from FeatureExtractor import molecule_feature_matrix
from DetectCorrelations import correlation_identifier
from datetime import datetime
import sys


def main():
    NotImplemented

if __name__ == '__main__':
    BENCHMARK_ROOT_DIR = 'd:/projects/VSB/datasets/2011_GLL&GDD/'
    TYPE_OF_SELECTION = 'rand-50-rest-500-rest-20'
    OUT_DIR = 'd:/data/vsb/'



    selectionsDir = BENCHMARK_ROOT_DIR + 'definition/selections/' + TYPE_OF_SELECTION
    moleculesDir = BENCHMARK_ROOT_DIR + 'definition/molecules/'

    for target in os.listdir(selectionsDir):
        print('[{}] Start processing target {}'.format(str(datetime.now()), target))
        targetSelectionsDir = selectionsDir + '/' + target
        targetMoleculesDir = moleculesDir + '/' + target

        #Read in all actives and inactives definitions from SDF (keys = ids) and obtain their SMILES
        sdfActives = targetMoleculesDir + '_Ligands.sdf'
        sdfInActives = targetMoleculesDir + '_Decoys.sdf'
        actives = {} #dictionary id -> smiles
        inactives = {}

        for spec in [[sdfActives, actives, 'actives'], [sdfInActives, inactives, 'inactives']]:
            print('[{}] Reading {}...'.format(str(datetime.now()), spec[2]))
            supplier = Chem.SDMolSupplier(spec[0])
            cnt = 1
            for molecule in supplier:
                if molecule is None:
                    sys.stderr.writelines("Error reading molecule {0} from {1}".format(cnt, spec[0]))
                else:
                    spec[1][molecule.GetProp('_Name')] = Chem.MolToSmiles(molecule)
                cnt += 1
                
        #Get all fragments and their descriptors
        non_degenerate_features = []
        [descriptors_map, descriptors] = \
            molecule_feature_matrix.read_descriptor_file(targetMoleculesDir+'-fragments.csv')

        for run_id in os.listdir(targetSelectionsDir):
            print('[{}] Start processing run {}'.format(str(datetime.now()), run_id))
            out_filename_base = OUT_DIR + 'target_' + target + '-run_' + run_id
            with open(targetSelectionsDir + '/' + run_id) as jsonFile:

                #Read in actives and inactives from SDF (keys = ids)
                runDefinition = json.load(jsonFile)               

                for [type_desc, type_name, map_id_smiles] in [['known actives', 'known-ligands', actives],
                                                              ['known inactives', 'known-decoys', inactives],
                                                              ['actives', 'ligands', actives],
                                                              ['inactives', 'decoys', inactives]]:

                    print("[{}] Starting processing {} of target {}, run id {} ...".format(
                        str(datetime.now()), type_desc, target, run_id))

                    #Get SMILES from the Ids
                    smiles = []
                    for idMolecule in runDefinition[type_name]:
                        try:
                            smiles.append(map_id_smiles[idMolecule])
                        except KeyError:
                            sys.stderr.writelines("SMILES string for molecule {} not found.".format(idMolecule))

                    # Get descriptors of fragments for given dataset
                    # and obtain their features, impute values and remove degenerate features
                    feature_matrix = molecule_feature_matrix.retrieve_features(descriptors_map, descriptors,
                                                                               targetMoleculesDir+'-fragments.json',
                                                                               smiles, 1, non_degenerate_features)
                    #Identify features which are representatives of correlation clusters
                    key_indices_actives = correlation_identifier.identify_correlated_features(
                        np.array(feature_matrix)[:, 0:-1], 100)
                    non_correlated_features_matrix = np.array(feature_matrix)[:, key_indices_actives]
                    #print("Non correlated feature matrix for the actives")
                    #for i in range(0,len(non_correlated_features_matrix)):
                    #        print(non_correlated_features_matrix[i])
                    with open(out_filename_base + '_' + type_desc.replace(' ', '_'), 'w') as f:
                        print(non_correlated_features_matrix.shape)
                        print("[{}] Storing non-correlated feature matrix...".format(str(datetime.now())))
                        np.savetxt(f, non_correlated_features_matrix, delimiter=",", fmt="%f")

            print('[{}] Processing of run {} finished'.format(str(datetime.now()), run_id))
        print('[{}] Processing of target {} finished'.format(str(datetime.now()), target))

    main()
