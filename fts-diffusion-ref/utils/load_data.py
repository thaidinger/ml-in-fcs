import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
torch.set_default_dtype(torch.float)

from utils.utils_clustering import *
from utils.markov_processing import *
from utils.series_processing import *

data_path = 'data/'
res_path = 'res/'



#########################
# Load time series data #
#########################

def get_fts(
    ticker='^GSPC',
    fts_name='sp500',
    start_date='1980-01-01', 
    end_date='2020-01-01'):
  """ Get financial time series from yfinance """
  df = yf.download(ticker, start=start_date, end=end_date, progress=False).reset_index()
  df['Close'].to_csv(data_path + fts_name + '_timeseries.csv', index=False)


def load_actual_fts(dataname):
  """ Load actual fts """
  filename = f"{dataname}_timeseries.csv"
  filepath = data_path + filename
  if not os.path.exists(filepath):
    # Backward-compatible fallback for older naming conventions.
    filepath = data_path + f"data_{dataname}_timeseries.csv"
  series = pd.read_csv(filepath).values
  return series


def load_ground_truth(dataname, params):
  """ Load ground truth segments in simulated investigation """
  n_clusters = params['k']
  l_min, l_max = params['l_min'], params['l_max']
  barycenter = params['barycenter']
  filename = f'data_{dataname}_l{l_min}-{l_max}'
  series = pd.read_csv(data_path + f"{filename}_timeseries.csv").values[:,1]
  labels = pd.read_csv(data_path + f"{filename}_labels.csv").values[:,1]
  segmentation = pd.read_csv(data_path + f"{filename}_segmentation.csv").values[:,1]
  subsequences = np.array([series[segmentation[i]:segmentation[i+1]] for i in range(len(segmentation)-1)], dtype=object)
  subsequences_norm = normalize_segments(subsequences)
  centroids = compute_centroids(n_clusters, subsequences_norm, labels, barycenter)
  return series, centroids, subsequences, labels, segmentation



#############################################
# Load SISC-resulting segments and patterns #
#############################################

def load_segments(dataname, params):
  """ Load SISC learned segmentation and clustering results """
  dict_init = {'kmeans++': 'kmpp', 'random_sample': 'rs', 'random_noise': 'rn'}
  n_clusters = params['k']
  l_min, l_max = params['l_min'], params['l_max']
  barycenter = params['barycenter']
  init_strategy = params['init_strategy']
  filename = f'sisc_{dataname}_k{n_clusters}_l{l_min}-{l_max}_{barycenter[:4]}_{dict_init[init_strategy]}'
  centroids = pd.read_csv(res_path + filename + '_centroids.csv').values[:,1:]
  labels = pd.read_csv(res_path + filename + '_labels.csv').values[:,1]
  segmentation = pd.read_csv(res_path + filename + '_segmentation.csv').values[:,1]
  segmentation = np.array([segmentation[i+1]-segmentation[i] for i in range(len(segmentation)-1)], dtype=int)
  df_subsequences = pd.read_csv(res_path + filename + '_subsequences.csv')
  subsequences = df_subsequences.values[:,1]
  subsequences = np.array([np.float64(subsequences[i].strip('[]').split()) for i in range(len(subsequences))], dtype=object)
  return centroids, subsequences, labels, segmentation









