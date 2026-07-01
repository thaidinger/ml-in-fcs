import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
torch.set_default_dtype(torch.float)

from utils.utils_clustering import *

data_path = 'data/'
res_path = 'res/'



##########################
# Construct Markov chain #
##########################

def construct_markov_chain(subsequences, labels, segmentation):
  """ Construct the Markov chain of states (pattern, length, magnitude) """
  series_pattern = labels.astype(int)
  series_length = segmentation.astype(int)
  series_magnitude = np.array([max(seg)-min(seg) for seg in subsequences], dtype=float)
  return series_pattern, series_length, series_magnitude


def construct_states_pairs(chain):
  """ Construct the (current states, next states) pairs """
  return np.array([[curr, next] for curr, next in zip(chain[:-1], chain[1:])])


def construct_markov_states_pairs(chain_pattern, chain_length, chain_magnitude, l_min=10, catlen=True):
  """ Construct the Markov chain of states pairs of (current states, next states) """
  if not catlen:
    chain_length = chain_length/l_min    # Scale the range of values for regression
  chain = np.stack((chain_pattern, chain_length, chain_magnitude), axis=1)
  states_pairs = construct_states_pairs(chain)
  return states_pairs



############################
# Markov transition matrix #
############################

def compute_transition_matrix(state_series):
  """ Compute the Markov transition matrix """
  value_min, value_max = min(state_series), max(state_series) + 1
  M = [[0]*(value_max-value_min) for _ in range(value_min, value_max)]
  for (i,j) in zip(state_series, state_series[1:]):
    M[i-value_min][j-value_min] += 1
  for row in M:
    s = sum(row)
    if s > 0:
      row[:] = [f/s for f in row]
  return np.array(M)


def magnitude_quantization(series, n_bins=10):
  """ Discretize continuous state (magnitude) via quantization """
  min_val = series.min()
  max_val = series.max()
  bin_edges = np.linspace(min_val, max_val, n_bins + 1)
  series_quantized = np.digitize(series, bin_edges, right=False)
  series_quantized = np.clip(series_quantized - 1, 0, n_bins - 1)
  # bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
  # series_recovered = bin_midpoints[series_quantized]
  return series_quantized.astype(int)#, series_recovered


def compute_transition_matrix_continuous(states, n_bins=10):
  # Compute the Markov transition matrix of continuous states
  states[:,-1] = magnitude_quantization(states[:,-1], n_bins)
  states = states.astype(int)
  # Compute transition matrix of each variable
  n_dims = len(states[0])
  transit_matrices = []
  for i in range(n_dims):
    transit_matrices.append(compute_transition_matrix(states[:,i]))
  return transit_matrices


def show_transition_matrix(state_series):
  """ Show the Markov transition probability matrix """
  transition_matrix = compute_transition_matrix(state_series)
  df = pd.DataFrame(transition_matrix)
  print(df.round(3))
  return transition_matrix


def diff_transition_matrix(mat1, mat2, ord=1):
  """ Show the difference between two Markov transition probability matrix """
  diff_norm = np.linalg.norm(mat1 - mat2, ord=ord)
  return diff_norm









