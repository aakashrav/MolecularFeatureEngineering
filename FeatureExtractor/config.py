import os

DEBUG = True

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"FragmentDescriptorData")
ELKI_EXECUTABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"ELKI/elki-bundle-0.7.0.jar")
FRAGMENT_FEATURES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "descriptors/features.csv")

FLUSH_BUFFER_SIZE = 100
DESCRIPTOR_TO_RAM = 0
NUM_FEATURES = 50
COVARIANCE_THRESHOLD = .80
