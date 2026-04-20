import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from experiments.stylized_facts import *
from models.model_params import prm_params
from models.sampling import generate_timeseries_ftsdiffusion
from utils.load_data import *

data_path = 'data/'
fig_path = 'figs/'

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



########################
# Qualitative analysis #
########################

def qualitative_analysis(store_res=False):
  """ Qualitative analysis in experiments """
  dataname = prm_params['dataname']
  patterns, segments, labels, lengths = load_segments(dataname, prm_params)
  threshold = find_threshold(segments, 252 * 10)
  sample_idx = np.random.randint(0, threshold)
  real_fts = get_qual_experiment_real_fts(segments, sample_idx)
  syn_fts = get_qual_experiment_syn_fts(patterns, segments, labels, lengths, sample_idx)
  real_fts = (real_fts[1:] - real_fts[:-1]) / real_fts[:-1]
  syn_fts = (syn_fts[1:] - syn_fts[:-1]) / syn_fts[:-1]
  plot_stylized_facts(real_fts, syn_fts, lags=90, store_res=store_res)
  if store_res:
    print("Results of qualitative analysis stored.")


def plot_stylized_facts(real_fts, syn_fts, lags=90, store_res=False):
  """ Plot the stylized facts for real and synthetic financial time series """
  fig, axs = plt.subplots(2, 3)
  # fig, axs = plt.subplots(2, 3, figsize=(20,9.5))
  # plt.rc('xtick', labelsize=37)
  # plt.rc('ytick', labelsize=37)
  # plt.rc('axes', labelsize=45)
  # plt.rc('legend', fontsize=30)
  fig.tight_layout()
  
  # Plot stylized facts of actual FTS
  plot_heavy_tail(real_fts, axs[0,0], axs[0,1], real=True)
  plot_returns_autocorr(abs(real_fts), lags, ax=axs[0,2], real=True)
  
  # Plot stylized facts of synthetic FTS
  plot_heavy_tail(syn_fts, axs[1,0], axs[1,1])
  plot_returns_autocorr(abs(syn_fts), lags, ax=axs[1,2])
  # plt.subplots_adjust(wspace=0.45, hspace=0.59)

  # Store the results before show to avoid blank figures on some backends.
  if store_res:
    fig.savefig(fig_path + 'stylized_fact.pdf', bbox_inches='tight')

  plt.show()



################
# Prepare data #
################

def find_threshold(segments, target_length=252*10):
  """ Determine the sampling threshold to ensure the combined time series can achieve the target length """
  threshold_idx = len(segments)
  sum_length = 0
  for segment in reversed(segments):
    sum_length += len(segment)
    threshold_idx -= 1
    if sum_length > target_length:
      break
  return threshold_idx


def get_qual_experiment_real_fts(segments, idx):
  """ Get the actual timeseries """
  real_timeseries = np.concatenate(segments[idx:])
  return real_timeseries


def get_qual_experiment_syn_fts(patterns, segments, labels, lengths, idx):
  """ Get the synthetic timeseries """
  first_state, first_segment = get_first_segment(segments, labels, lengths, idx)
  syn_timeseries = generate_timeseries_ftsdiffusion(252*10, first_state, first_segment, patterns)
  return syn_timeseries


def get_first_segment(segments, labels, lengths, idx):
  """ Get the first segment for generation """
  init_segment = segments[idx]
  init_pattern = labels[idx]
  init_length = lengths[idx]
  init_magnitude = max(init_segment) - min(init_segment)
  init_state = np.stack((init_pattern, init_length, init_magnitude), axis=0)
  init_state = torch.tensor(init_state).view(1, -1)
  return init_state, init_segment









