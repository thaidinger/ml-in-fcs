import numpy as np
from dtaidistance import dtw as dtai_dtw
import torch
torch.set_default_dtype(torch.float)

from utils.utils_clustering import *

data_path = 'data/'
res_path = 'res/'



##################
# Cluster losses #
##################

def compute_within_cluster_loss(centroids, sequences, labels):
  """ Compute the within-clutser loss using DTW to measure the total discrepancy within clusters  """
  n_clusters = len(centroids)
  sequences = normalize_segments(sequences)
  n_seqs = len(sequences)
  loss_total = 0
  loss_clusters = np.zeros(n_clusters)
  for i in range(n_clusters):
    sequences_i = sequences[labels==i]
    n_seqs_i = len(sequences_i)
    if n_seqs_i==0:
      loss_clusters[i] = 0
    else:
      loss_i = sum([dtai_dtw.distance_fast(seq.astype(np.double), centroids[i].astype(np.double), use_pruning=True) for seq in sequences_i])
      loss_total += loss_i
      loss_clusters[i] = loss_i/n_seqs_i
  return loss_total/n_seqs, np.mean(loss_clusters)



####################
# Silhouette score #
####################

def compute_silhouette_score_dtw(K, X, labels):
  """ Compute the silhouette score using DTW to measure the degree of separation between clusters """
  if len(np.unique(labels))==1:
    return np.NaN
  N = len(X)
  # Initialize the a(i) and b(i) vectors
  a = np.zeros(N)
  b = np.zeros(N)
  # Compute a(i)
  for k in range(K):
    cluster_k = np.where(labels==k)[0]
    for i in cluster_k:
      a[i] = np.mean([dtai_dtw.distance_fast(X[i].astype(np.double), X[j].astype(np.double), use_pruning=True) for j in cluster_k if i!=j])
  # Compute b(i)
  for i in range(N):
    b[i] = min([np.mean([dtai_dtw.distance_fast(X[i].astype(np.double), X[j].astype(np.double), use_pruning=True) for j in np.where(labels==k)[0]]) for k in range(K) if labels[i]!=k])
  # Compute the silhouette scores
  eps = 1e-5
  s = (b - a) / (np.maximum(a, b) + eps)
  return np.mean(s)



###############################
# Intersection of Union (IoU) #
# (Only for simulated data)   #
###############################

def compute_iou(A, B):
  """ Compute the intersection of union to measure the overlaps between learned and ground-truth segmentation """
  IoU_values = []
  for b in B:
    max_iou = 0
    for a in A:
      intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
      union = max(a[1], b[1]) - min(a[0], b[0])
      iou = intersection / union
      max_iou = max(max_iou, iou)
    IoU_values.append(max_iou)
  return sum(IoU_values) / len(IoU_values)

def IoU(real, pred):
  """ Compute the intersection of union to measure the overlaps between learned and ground-truth segmentation """
  A = np.array([(real[i], real[i+1]) for i in range(len(real)-1)])
  B = np.array([(pred[i], pred[i+1]) for i in range(len(pred)-1)])
  iou = compute_iou(A, B)
  print(f"IoU: {iou}")



#############################
# Cluster assignment error  #
# (Only for simulated data) #
#############################

def compute_clutser_assign_error(real_label_series, pred_label_series):
  """ Compute the cluster assignment error to measure the difference between learned and ground-truth cluster labels """
  T = min(len(real_label_series), len(pred_label_series))
  error = sum([real_label_series[t]!=pred_label_series[t] for t in range(T)])/T
  return error



##########################################################
# Discrepancy between leanred and ground-truth centroids #
# (Only for simulated data)                              #
##########################################################

def compute_perunit_dtw_percentage_error(real, pred):
  """ Compute per-unit DTW percentage error between learned and ground-truth centroids """
  dtw = dtai_dtw.distance_fast(real.astype(np.double), pred.astype(np.double), use_pruning=True)
  l = len(real)
  dtw_avg = dtw/l
  error_perunit = [dtw_avg/real[i] if real[i]!=0 else dtw_avg for i in range(l)]
  return np.mean(error_perunit)

def compute_perunit_dtw_error(real, pred):
  """ Compute per-unit DTW error between centroids """
  dtw = dtai_dtw.distance_fast(real.astype(np.double), pred.astype(np.double), use_pruning=True)
  l = len(real)
  dtw_avg = dtw/l
  return dtw_avg









