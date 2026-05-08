from models.model_params import prm_params
from utils.load_data import *
from utils.split_dataset import *


################################
# Prepare data for experiments #
################################

def prepare_segments(split_ratio=0.8):
  """ Prepare the training and test data for experiments """
  dataname = prm_params['dataname']
  centroids, segments, labels, lengths = load_segments(dataname, prm_params)
  segments_train, segments_test = split_train_test_set(segments, split_ratio)
  labels_train, labels_test = split_train_test_set(labels, split_ratio)
  lengths_train, lengths_test = split_train_test_set(lengths, split_ratio)
  train_set = (segments_train, labels_train, lengths_train)
  test_set = (segments_test, labels_test, lengths_test)
  return train_set, test_set, centroids









