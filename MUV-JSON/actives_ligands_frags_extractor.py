import json
import argparse
import os
import sys

__author__ = "Aakash Ravi"
__email__ = "aakash_ravi@hotmail.com"

def main():
    actives_molecules_file = sys.argv[1]
    inactives_molecules_file = sys.argv[2]
    frag_file = sys.argv[3]
    DATA_DIRECTORY = sys.argv[4]

    with open(actives_molecules_file,"r") as f_handle:
        active_molecules = json.load(f_handle)

    with open(inactives_molecules_file,"r") as f_handle:
        inactive_molecules = json.load(f_handle)

    with open(frag_file,"r") as f_handle:
        molecule_fragments = json.load(f_handle)

    active_molecule_fragments = [molecule for molecule in molecule_fragments if molecule["name"] in active_molecules]

    inactive_molecule_fragments = [molecule for molecule in molecule_fragments if molecule["name"] in inactive_molecules]

    with open(os.path.join(DATA_DIRECTORY,'active_fragments.json'), 'w+') as outfile:
        json.dump(active_molecule_fragments, outfile)

    with open(os.path.join(DATA_DIRECTORY,'inactive_fragments.json'), 'w+') as outfile:
        json.dump(inactive_molecule_fragments, outfile)

if __name__ == "__main__":
    main()

