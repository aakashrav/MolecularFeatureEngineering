import os

DEBUG = True

DATA_DIRECTORY = os.path.join(os.getcwd(),"FragmentDescriptorData")
TEST_DATA_DIRECTORY = os.path.join(os.getcwd(),"TestFragmentDescriptorData")
ELKI_EXECUTABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"ELKI/elki-bundle-0.7.0.jar")
INPUT_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'MUV-JSON')
JAVA_EXECUTABLE = '/storage/brno2/home/ravia/jdk1.8.0_102/bin/java'

# Parameters for creation of feature matrix
FLUSH_BUFFER_SIZE = 100
DESCRIPTOR_TO_RAM = 1

# Parameters for determining significant features using correlation neighborhoods
NUM_FEATURES = 60
CORRELATION_THRESHOLD = .80

# Cluster extraction and pruning parameters
CLUSTER_PURITY_THRESHOLD = .6
CLUSTER_DIVERSITY_THRESHOLD = 20
CLUSTER_DIVERSITY_PERCENTAGE = False
