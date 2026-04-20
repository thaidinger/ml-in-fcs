import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from dtaidistance import dtw as dtai_dtw
from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient
import matplotlib.pyplot as plt

res_path = 'res/'



################################
# Load SISC-resulting clusters #
################################

def load_sisc_res(
    dataname,
    n_clusters,
    l_min, l_max,
    init_strategy='kmeans++',
    barycenter='dba'):
  """ Load learned segmentation and clustering results """
  dict_init = {'kmeans++': 'kmpp', 'random_sample': 'rs', 'random_noise': 'rn'}
  filename = f'sisc_{dataname}_k{n_clusters}_l{l_min}-{l_max}_{barycenter[:4]}_{dict_init[init_strategy]}'
  df_centroids = pd.read_csv(res_path + filename + '_centroids.csv')
  df_labels = pd.read_csv(res_path + filename+'_labels.csv')
  df_subsequences = pd.read_csv(res_path + filename + '_subsequences.csv')
  df_segmentation = pd.read_csv(res_path + filename + '_segmentation.csv')
  subsequences = df_subsequences.values[:,1]
  subsequences = np.array([np.float64(subsequences[i].strip('[]').split()) for i in range(len(subsequences))], dtype=object)
  return df_centroids.values[:,1:], subsequences, df_labels.values[:,1], df_segmentation.values[:,1]



########################################
# Cluster-related computation for SISC #
########################################

def normalize_segments(segments):
  """ Normalize the segments into the unit scale in magnitude """
  segments_norm = []
  for seg in segments:
    max_value = max(seg)
    min_value = min(seg)
    seg_norm = (seg - min_value) / (max_value - min_value)
    segments_norm.append(seg_norm)
  return np.array(segments_norm, dtype=object)


def compute_centroids(
    n_patterns,
    segments,
    labels=None,
    barycenter='dba',
    gamma=.001,
    size=None):
  """ Compute the centroids of segments in each cluster """
  segments = np.array(segments.copy(), dtype=object)
  if n_patterns==1:
    if barycenter=='dba':
      return dtw_barycenter_averaging(segments, barycenter_size=size, tol=1e-5).flatten().astype(float)
    elif barycenter == 'softdtw':
      return softdtw_barycenter(segments, gamma=gamma, tol=1e-5).flatten().astype(float)
    elif barycenter=='dbasubgrad':
      return dtw_barycenter_averaging_subgradient(segments, barycenter_size=size, tol=1e-5).flatten().astype(float)
  else:
    centroids = []
    for i in range(n_patterns):
      idx_i = np.where(labels == i)[0]
      segments_i = segments[idx_i]
      if barycenter == 'dba':
        centroid = dtw_barycenter_averaging(segments_i, barycenter_size=size, tol=1e-5).flatten()
      elif barycenter == 'softdtw':
        centroid = softdtw_barycenter(segments_i, gamma=gamma, tol=1e-5).flatten()
      elif barycenter=='dbasubgrad':
        centroid = dtw_barycenter_averaging_subgradient(segments_i, barycenter_size=size, tol=1e-5).flatten()
      centroids.append(centroid.astype(float))
    return np.array(centroids)


def compute_label_alignment(real, pred):
  """ Compute the label aligment between learned clusters and the ground-truth (if applicable) """
  K = len(real)
  alignment = np.zeros(K)
  candidate = np.arange(K)
  # Greedily find the nearest learned centroid for each ground-truth centroid
  for i in range(K):
    distances = [dtai_dtw.distance_fast(real[i].astype(np.double), pred[j].astype(np.double), use_pruning=True) for j in candidate]
    select = np.argmin(distances)
    alignment[i] = candidate[select]
    candidate = np.delete(candidate, select)
  return alignment.astype(int)


def compute_label_alignment_hungarian(real, pred):
  """ Compute the label aligment between learned clusters and the ground-truth (if applicable) using hungarian algorithm """
  K = len(real)
  distance_matrix = np.zeros((K,K))
  for i in range(K):
    for j in range(K):
      distance_matrix[i,j] = dtai_dtw.distance_fast(real[i].astype(np.double), pred[j].astype(np.double), use_pruning=True)
  row_ind, col_ind = linear_sum_assignment(distance_matrix)
  alignment = col_ind
  return alignment.astype(int)


def align_labels(labels, align):
  """ Align the labels of learned clusters with the ground-truth (if applicable) """
  labels_aligned = [np.where(align==label) for label in labels]
  return np.array(labels_aligned, dtype=object).flatten()


def label_series_from_seg(segmentation, labels):
  """ Get the label series from segmentation """
  N = len(labels)
  label_series = []
  for i in range(N):
    label_series.extend([labels[i]] * (segmentation[i+1]-segmentation[i]))
  return np.array(label_series)



######################
# SISC Visualization #
######################

def plot_real_timeseries(timeseries, name=''):
  plt.figure(figsize=(10, 2))
  plt.plot(timeseries)
  plt.title(f"{name} Time Series Data")


def plot_real_clusters(
    n_patterns,
    timeseries,
    segmentation,
    labels=None,
    barycenter='dba',
    gamma=.001):
  """ Show ground-truth clusters including centroid(s) and segments """
  subsequences = np.array([timeseries[segmentation[i]:segmentation[i+1]] for i in range(len(segmentation)-1)], dtype=object)
  subsequences = normalize_segments(subsequences)
  centroids = compute_centroids(n_patterns, subsequences, labels, barycenter=barycenter, gamma=gamma)
  n_rows = n_patterns
  n_cols = 5
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
  for i in range(n_patterns):
    # Plot centroid
    centroid = centroids[i].flatten() if n_patterns!=1 else centroids.flatten()
    ax_c = axs[i,0] if n_patterns!=1 else axs[0]
    ax_c.plot(centroid, color='red')
    ax_c.get_xaxis().set_visible(False)
    ax_c.get_yaxis().set_visible(False)
    # Plot segments
    subsequences_i = subsequences[labels==i] if n_patterns!=1 else subsequences
    for j in range(1, n_cols):
      idx = np.random.choice(len(subsequences_i))
      segment = subsequences_i[idx]
      ax_s = axs[i, j] if n_patterns!=1 else axs[j]
      ax_s.plot(segment)
      ax_s.get_xaxis().set_visible(False)
      ax_s.get_yaxis().set_visible(False)
  plt.show()
  return centroids


def plot_res_clusters(
    n_patterns,
    centroids,
    subsequences,
    labels=None):
  """ Show learned clusters including centroid(s) and segments """
  n_rows = n_patterns
  n_cols = 5
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
  for i, centroid in enumerate(centroids):
    # Plot centroid
    centroid = centroids[i].flatten() if n_patterns!=1 else centroids.flatten()
    ax_c = axs[i,0] if n_patterns!=1 else axs[0]
    ax_c.plot(centroid, color='red')
    ax_c.get_xaxis().set_visible(False)
    ax_c.get_yaxis().set_visible(False)
    # Plot segments
    subsequences_i = subsequences[labels==i] if n_patterns!=1 else subsequences
    for j in range(1, n_cols):
      idx = np.random.choice(len(subsequences_i))
      segment = subsequences_i[idx]
      ax_s = axs[i, j] if n_patterns!=1 else axs[j]
      ax_s.plot(segment)
      ax_s.get_xaxis().set_visible(False)
      ax_s.get_yaxis().set_visible(False)
  plt.show()


def plot_res_centroids(
    n_patterns,
    centroids,
    align):
  """ Show learned centroid(s) with the order aligned with the ground truth """
  fig,axs = plt.subplots(1, n_patterns, figsize=(2 * n_patterns, 2))
  for i, _ in enumerate(centroids):
    centroid = centroids[align[i]]
    axs[i].plot(centroid, color='red', label=f"p{str(i+1)}")
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  plt.show()


def plot_centroids_comparison(real, pred, align=None):
  """ Show ground-truth and learned centroid(s) with aligned order """
  if align is None:
    align = compute_label_alignment(real, pred)
  K = len(real)
  fig, axs = plt.subplots(2, K, figsize=(K * 2, 4))
  for i in range(K):
    axs[0, i].plot(real[i], color='red')
    axs[0, i].get_xaxis().set_visible(False)
    axs[0, i].get_yaxis().set_visible(False)
    axs[1, i].plot(pred[align[i]])
    axs[1, i].get_xaxis().set_visible(False)
    axs[1, i].get_yaxis().set_visible(False)
  plt.show()









