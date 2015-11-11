#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this script is to create a list of actives and inactives from one of the datasets
given for a target
"""

__author__ = "Aakash Ravi"

import os

def get_actives( molecules_to_fragments_file ):
    "This function retreives the actives from the corresponding data file provided in the argument"

    # Create an empty array of active names
    known_actives = []

    with open(molecules_to_fragments_file, 'r') as input_stream:
        for line in input_stream:
            # Line contains some control characters that we don't want
            line = line[0:-2]
            known_actives.append(line)
    return known_actives

def get_inactives( molecules_to_fragments_file ):
    "This function retrieves the inactives from the corresponding data file provided in the argument"

    # Create an empty array of inactive names
    known_inactives = []

    with open(molecules_to_fragments_file, 'r') as input_stream:
        for line in input_stream:
            # Line contains some control characters that we don't want
            line = line[0:-2]
            known_inactives.append(line)
    return known_inactives
