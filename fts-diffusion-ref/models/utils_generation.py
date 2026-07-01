import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_default_dtype(torch.float)
import yfinance as yf

from utils.utils_clustering import *
from utils.series_processing import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)



#################################################
# Tensorize the segments and reference patterns #
#################################################

class PairedDataset(Dataset):
  """
    Construct the dataset for a variable-length input and a fixed-length input (inputs as list of tensors)
  """
  def __init__(self, variable_length_seqs, fixed_length_seqs, custom_pad_length=None):
    # self.variable_length_seqs = sorted(variable_length_seqs, key=len, reverse=True)
    self.variable_length_seqs = variable_length_seqs
    self.fixed_length_seqs = fixed_length_seqs
    assert len(self.variable_length_seqs) == len(self.fixed_length_seqs)
    self.custom_pad_length = custom_pad_length
    if self.custom_pad_length is None:
      self.custom_pad_length = max(len(seq) for seq in self.variable_length_seqs)

  def __len__(self):
    return len(self.variable_length_seqs)

  def __getitem__(self, idx):
    return self.variable_length_seqs[idx], self.fixed_length_seqs[idx]

  def collate_fn(self, batch):
    variable_seqs, fixed_seqs = zip(*batch)
    variable_seqs_len = [len(seq) for seq in variable_seqs]
    # padded_variable_seqs = pad_sequence(variable_seqs, batch_first=True)
    padded_variable_seqs = torch.zeros((len(variable_seqs), self.custom_pad_length), dtype=torch.float32)
    for i, seq in enumerate(variable_seqs):
      end = min(len(seq), self.custom_pad_length)
      padded_variable_seqs[i, :end] = seq[:end]
    fixed_seqs_tensor = torch.stack(fixed_seqs, dim=0)
    variable_seqs_len_tensor = torch.tensor(variable_seqs_len)
    return padded_variable_seqs.float(), fixed_seqs_tensor.float(), variable_seqs_len_tensor.int()


def rescale_segments_fixed_length(segments):
  """ Rescale fixed-length segments into unit segments """
  min_vals = segments.min(dim=1, keepdim=True).values
  max_vals = segments.max(dim=1, keepdim=True).values
  magnitudes = max_vals - min_vals
  rescaled_segments = (segments - min_vals) / magnitudes
  return rescaled_segments

def rescale_segment_variable_length(segment):
  """ Rescale each variable-length segment into a unit segment """
  min_vals = min(segment)
  max_vals = max(segment)
  magnitude = max_vals - min_vals
  rescaled_segment = (segment - min_vals) / magnitude
  return rescaled_segment

def rescale_segments_variable_length(segments):
  """ Rescale variable-length segments into unit segments """
  rescaled_segments = [rescale_segment_variable_length(segment) for segment in segments]
  return rescaled_segments

def rescale_segments(segments, if_vary):
  """ Rescale segments into unit segments """
  if if_vary:
    return rescale_segments_variable_length(segments)
  else:
    return rescale_segments_fixed_length(segments)


def tensorize_segments_fixed_length(segments):
  """ Tensorize the fixed-length segments """
  segments_tensor = torch.tensor(segments.astype(float), dtype=torch.float)
  return segments_tensor

def tensorize_segments_variable_length(segments):
  """ Tensorize the variable-length segments """
  segments_tensors = [torch.tensor(segment) for segment in segments]
  return segments_tensors


def tensorize_patterns_raw(ref_patterns, labels):
  """ Tensorize the reference patterns of raw prices """
  patterns_series = np.array([ref_patterns[p] for p in labels], dtype=float)
  patterns_tensor = torch.tensor(patterns_series.astype(float), dtype=torch.float)
  return patterns_tensor

def tensorize_patterns_firstorder(segments, labels, max_length):
  """ Tensorize the reference patterns of returns """
  n_patterns = len(set(labels))
  centroids = compute_centroids(n_patterns, segments, labels, size=max_length)
  patterns_series = np.array([centroids[p] for p in labels])
  patterns_tensor = torch.tensor(patterns_series, dtype=torch.float)
  return patterns_tensor


def check_variable_lengths(segments):
  """ Check whether the lengths of segments are variable """
  lengths = [len(segment) for segment in segments]
  max_length = max(lengths)
  if_vary = len(set(lengths)) != 1
  return if_vary, max_length



#######################################################
# Construct dataloader for training generation module #
#######################################################

def construct_dataloader_generation(
  segments, ref_patterns, labels,
  batch_size=16, 
  shuffle=False,
  first_order=False,
  rescale=False):
  """ Construct dataloader of segments and patterns for training generation module """
  if_vary, max_length = check_variable_lengths(segments)
  if if_vary:
    tensor_x = tensorize_segments_variable_length(segments)
  else:
    tensor_x = tensorize_segments_fixed_length(segments)
  if first_order:
    tensor_x = prices2returns_batch_variable_length(tensor_x)
    tensor_p = tensorize_patterns_firstorder(tensor_x, labels, max_length)
  else:
    tensor_p = tensorize_patterns_raw(ref_patterns, labels)
  if rescale:
    tensor_x = rescale_segments(tensor_x, if_vary)
    tensor_p = rescale_segments(tensor_p, if_vary)
  paired_list = list(zip(tensor_x, tensor_p))
  paired_list.sort(key=lambda x: len(x[0]), reverse=True)
  tensor_x, tensor_p = zip(*paired_list)
  dataset = PairedDataset(tensor_x, tensor_p)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
  return dataloader



#################################
# Convert data from cuda to cpu #
#################################

def cuda2cpu(tensors):
  """ Convert the tensors from cuda to cpu """
  tensors_cpu = []
  for tensor in tensors:
    tensors_cpu.append(tensor.cpu())
  return tensors_cpu









