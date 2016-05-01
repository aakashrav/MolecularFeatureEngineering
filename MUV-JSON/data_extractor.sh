# clone this script
# https://github.com/skodape/biochem-tools/tree/master/molecule_preparation

#http://siret.ms.mff.cuni.cz/skoda/datasets/8.0-8.5.zip
#http://siret.ms.mff.cuni.cz/skoda/datasets/8.5-9.0.zip
#http://siret.ms.mff.cuni.cz/skoda/datasets/9.0-9.5.zip
#http://siret.ms.mff.cuni.cz/skoda/datasets/9.8-1.0.zip


# extract features fragmetns from known actives and decoys
# We've done this already
#python biochem-tools/molecule_preparation/extract_fragments.py -i 5HT1F_Agonist_Decoys.sdf -o 5HT1F_Agonist_Decoys.frags.json
#C:/Python27-64/python.exe C:/Projects/GitHub/biochem-tools/molecule_preparation/extract_fragments.py -i 5HT1F_Agonist_Ligands.sdf -o 5HT1F_Agonist_Ligandsfrags.json

# Extract features for every fragment of every known active and decoy, DONE SEPARATELY
# SO WE GET TWO SEPARATE FEATURE FILES..THEN WE CAN COMBINE THEM AFTER REMOVAL OF CONSTANT FEATURES
python ./actives_ligands_frags_extractor.py ./8/8_ligands.json ./8/8_decoys.json ./8/8_fragments.json ./8/
python ./biochem-tools/molecule_preparation/compute_descriptors.py -i ./8/active_fragments.json -o ./8/8_ligands_features.csv -p /Users/AakashRavi/Desktop/Aakash/Education/ChemicalInformatics/MolecularFeatureEngineering/MUV-JSON/PaDEL-Descriptor -f
python ./biochem-tools/molecule_preparation/compute_descriptors.py -i ./8/inactive_fragments.json -o ./8/8_decoys_features.csv -p /Users/AakashRavi/Desktop/Aakash/Education/ChemicalInformatics/MolecularFeatureEngineering/MUV-JSON/PaDEL-Descriptor -f
# Script to join features of active and inactive fragments
cat ./8/8_ligands_features.csv ./8/8_decoys_features.csv | sort -r | uniq > ./8/8_features.csv
python ./remove_constant_features.py ./8/8_features.csv
rm ./8/8_ligands_features.csv ./8/8_decoys_features.csv

# http://www.yapcwsoft.com/dd/padeldescriptor/
