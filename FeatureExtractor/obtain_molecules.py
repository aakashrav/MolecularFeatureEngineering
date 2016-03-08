#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this script is to create a list of actives and inactives from one of the datasets
given for a target
"""

__author__ = "Aakash Ravi"

import os
import json
import re

# def get_molecules( molecule_file ):
#     "This function retreives the actives or inactives from the corresponding data file provided in the argument"

#     # Create an empty array of active names
#     known_actives = []

#     with open(molecule_file, 'r') as input_stream:
#         for line in input_stream:
#             # Line contains some control characters that we don't want
#             line = line[0:-2]
#             known_actives.append(line)
#     return known_actives

def get_sdf_molecules( molecule_directory ):
    "This function retrieves the actives or inactives SDF Identifier numbers from the data file"

    molecule_list = []

    for subdir, dirs, files in os.walk(molecule_directory):
        for file in files:
            if file.endswith('.sdf'):
                with open(os.path.join(molecule_directory,file),'r') as f_handle: 
                    lines = f_handle.read().splitlines()
                    molecule_sdf = lines[0]
                molecule_list.append(molecule_sdf)
                f_handle.close()
            else:
                continue

    return molecule_list

if __name__ == '__main__':
    get_sdf_molecules("../SDFActivesInactivesDataset/Hydrogen-Bonds_4/actives_1.json")
