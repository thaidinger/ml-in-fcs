import torch
torch.set_default_dtype(torch.float)

# from models.model_params import prm_params
from utils.prepare_data import *
from utils.markov_processing import *



###########################
# Prepare sampling inputs #
###########################

def sampling_inputs(split_raito=0.8):
  """ Prepare the inputs for sampling module """
  _, test_set, patterns = prepare_segments(split_raito)
  segments, labels, lengths = test_set
  chain_pattern, chain_length, chain_magnitude = construct_markov_chain(segments, labels, lengths)
  chain_states = np.stack((chain_pattern, chain_length, chain_magnitude), axis=1)
  return segments, chain_states, patterns



#################################
# Initialize the first segement #
#################################

def get_init_state_by_index(sample_idx=None):
  """ Initialize the state of first segment for sampling by given index """
  subseqs_real, states_real, _ = sampling_inputs()
  if sample_idx is None:
    sample_idx = np.random.randint(0, len(states_real))
  init_state = torch.tensor(states_real[sample_idx]).view(1, -1)
  init_segment = subseqs_real[sample_idx]
  return init_state, init_segment, sample_idx


def get_init_state(init_state=None, init_segment=None):
  """ Initialize the states """
  _, test_set, _ = sampling_inputs()
  segments_test, labels_test, lengths_test = test_set()
  sample_idx = np.random.randint(0, len(segments_test))
  init_segment = segments_test[sample_idx]
  init_pattern = labels_test[sample_idx]
  init_length = lengths_test[sample_idx]
  init_magnitude = max(init_segment) - min(init_segment)
  init_state = np.stack((init_pattern, init_length, init_magnitude), axis=0)
  init_state = torch.tensor(init_state).view(1, -1)
  return init_state, init_segment









