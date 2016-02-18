#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this script is to create a list of actives and inactives from one of the datasets
given for a target
"""

__author__ = "Aakash Ravi"

import os
import json
import re

def get_molecules( molecule_file ):
    "This function retreives the actives or inactives from the corresponding data file provided in the argument"

    # Create an empty array of active names
    known_actives = []

    with open(molecule_file, 'r') as input_stream:
        for line in input_stream:
            # Line contains some control characters that we don't want
            line = line[0:-2]
            known_actives.append(line)
    return known_actives

def stupid_test(filename):
    with open(filename, 'r') as f_handle:
        molecules_to_fragments = json.load(f_handle)
    
    # actual_m_list = []
    # for i in range(0, len(mlist)):
    #     actual_json = re.sub("u'", '', json.dumps(mlist[i]))
    #     actual_m_list.append(actual_json)

    # json_list = json.loads(json.dumps(actual_m_list))
    # print(json_list)
    # for i in range(0, len(json_list)):
    #     print(json_list[i]["fragments"])

    # print(mlist)
    # print(mlist[0]["name"])
    fragments = []
    for i in range(0, len(molecules_to_fragments)):
        # if (molecules_to_fragments[i]["name"] == molecule_sdfs[molecule_index]):
        for j in range(0,len(molecules_to_fragments[i]["fragments"])):
            fragments.append(molecules_to_fragments[i]["fragments"][j]["smiles"])
    # print(mlist[0]["fragments"][:]["smiles"])

def get_sdf_molecules( molecule_directory ):
    "This function retrieves the actives or inactives SDF Identifier numbers from the data file"

    molecule_list = []

    for subdir, dirs, files in os.walk(molecule_directory):
        for file in files:
            if file.endswith('.sdf'):
                with open(molecule_directory+'/'+file,'r') as f_handle: 
                    lines = f_handle.read().splitlines()
                    molecule_sdf = lines[0]
                molecule_list.append(molecule_sdf)
                f_handle.close()
            else:
                continue
    # print(molecule_list)
    # with open(molecule_file, 'r') as f_handle:
    #     molecule_list = json.load(f_handle)

    return molecule_list

if __name__ == '__main__':
    # get_sdf_molecules("../SDFActivesInactivesDataset/Hydrogen-Bonds_4/actives_1.json")
    stupid_test("../fragments/SDF_Fragments.json")
