import os

DEBUG = True

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"FragmentDescriptorData")
TEST_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"TestFragmentDescriptorData")
ELKI_EXECUTABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"ELKI/elki-bundle-0.7.0.jar")
FRAGMENT_FEATURES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "descriptors/features.csv")

# Parameters for creation of feature matrix
FLUSH_BUFFER_SIZE = 100
DESCRIPTOR_TO_RAM = 1

# Parameters for determining significant features using covariance neighborhoods
NUM_FEATURES = 50
COVARIANCE_THRESHOLD = .80

# Cluster extraction and pruning parameters
CLUSTER_PURITY_THRESHOLD = .6
CLUSTER_DIVERSITY_THRESHOLD = 20
CLUSTER_DIVERSITY_PERCENTAGE = False
