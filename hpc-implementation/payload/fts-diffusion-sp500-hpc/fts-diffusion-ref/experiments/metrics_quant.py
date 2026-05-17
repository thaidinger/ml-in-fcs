import numpy as np
import pandas as pd
import yfinance as yf
import torch
from scipy import stats
from tqdm import tqdm
import warnings

from models.sampling import generate_timeseries_ftsdiffusion



#########################
# Goodness of fit tests #
#########################

def ks_test(real, syn):
  """ Kolmogorov-Smirnov test """
  return stats.ks_2samp(real, syn)[1]

def ad_test(real, syn):
  """ Anderson-Darling test """
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return stats.anderson_ksamp([real, syn])[2]



############################
# Distribution discrepancy #
############################

def w_distance(real, syn):
  """ Wasserstein distance """
  return stats.wasserstein_distance(real, syn)


#############################################################################
# Compute the results of goodness of fit tests and distribution discrepancy #
#############################################################################

def distribution_tests(n_rounds, patterns, segments, labels, lengths, threshold):
  """ Perform the quantitative experiments using distribution tests """
  ks_stats = []
  ad_stats = []
  for _ in tqdm(range(n_rounds)):
    sample_idx = np.random.randint(0, threshold)
    real_fts = get_quant_experiment_real_fts(segments, sample_idx)
    syn_fts = get_quant_experiment_syn_fts(patterns, segments, labels, lengths, sample_idx)
    real_sample = (real_fts[1:] - real_fts[:-1]) / real_fts[:-1]
    syn_sample = (syn_fts[1:] - syn_fts[:-1]) / syn_fts[:-1]
    ks_stats.append(ks_test(real_sample, syn_sample))
    ad_stats.append(ad_test(real_sample, syn_sample))

  results = {
  'KS': {'mean': np.mean(ks_stats), 'std': np.std(ks_stats, ddof=1)},
  'AD': {'mean': np.mean(ad_stats), 'std': np.std(ad_stats, ddof=1)}
  }
  results_df = pd.DataFrame(results)

  print(results_df)



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


def get_quant_experiment_real_fts(segments, idx, series_length=60):
  """ Get the actual timeseries """
  real_timeseries = np.concatenate(segments[idx:])
  return real_timeseries[:series_length]


def get_quant_experiment_syn_fts(patterns, segments, labels, lengths, idx, series_length=60):
  """ Get the synthetic timeseries """
  first_state, first_segment = get_first_segment(segments, labels, lengths, idx)
  syn_timeseries = generate_timeseries_ftsdiffusion(series_length, first_state, first_segment, patterns)
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









