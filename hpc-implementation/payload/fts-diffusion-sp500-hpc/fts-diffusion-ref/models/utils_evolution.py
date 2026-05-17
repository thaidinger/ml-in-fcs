import torch
from torch.utils.data import DataLoader, TensorDataset
torch.set_default_dtype(torch.float)

from utils.markov_processing import *



#####################################################
# Construct dataloader for pattern evolution module #
#####################################################

def Dataset2Chain(dataset):
  """ Convert the train or test set to Markov chain of states (pattern, length, magnitude) """
  segments, labels, lengths = dataset
  chain = construct_markov_chain(segments, labels, lengths)
  return chain


def Chain2TensorDataset(chain_pattern, chain_length, chain_magnitude, catlen=True):
  """ Tensorize the Markov chain of states (pattern, length, magnitude) """
  states_pairs = construct_markov_states_pairs(chain_pattern, chain_length, chain_magnitude, catlen)
  states_pairs = torch.tensor(states_pairs, dtype=float)
  dataset = TensorDataset(states_pairs)
  return dataset


def Chain2Dataloader(chain, batch_size, shuffle=False):
  """ Convert the Markov chain of states (pattern, length, magnitude) to dataloader """
  chain_pattern, chain_length, chain_magnitude = chain
  tensor_dataset = Chain2TensorDataset(chain_pattern, chain_length, chain_magnitude, catlen=True)
  dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader


def construct_dataloader_evolution(segments, labels, lengths, batch_size=16, shuffle=False):
  """ Contruct dataloader for training evolution module """
  chain = Dataset2Chain((segments, labels, lengths))
  dataloader = Chain2Dataloader(chain, batch_size, shuffle)
  return dataloader









