import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import torch
torch.set_default_dtype(torch.float)
import matplotlib.pyplot as plt
from tqdm import tqdm

# from models.utils_recognition import *
from utils.utils_clustering import *

data_path = 'data/'
res_path = 'res/'



##################################################
# Scale-Invariant Subsequences Clustering (SISC) #
##################################################

class SISC:
  """
    Scale-Invariant Subsequences Clustering (SISC)
    Args
      n_clusters: the number of clusters
      l_min, l_max: minimum and maximum length of segments, respectively
      gamma: (optional) parameter to control smoothness in soft-DTW
  """
  def __init__(self, n_clusters, l_min=10, l_max=20, gamma=.5):
    self.n_clusters = n_clusters
    self.l_min = l_min
    self.l_max = l_max
    self.gamma = gamma
    self.T = None
    self.centroids = None
    self.labels = None
    self.subsequences = None
    self.segmentation = None
    self.total_loss = np.inf
    self.min_loss = np.inf
    self.first_iter = True
    self.hist_loss = []
    self.converge = 0

  ''' ~fit: greedy K-Means based subsequence clustering '''
  def fit(self, series, 
          max_iters,
          init_strategy='kmeans++', 
          barycenter='dba',
          plot_progress=False,
          store_res=False, dataname=None):
    
    # Initialization
    self.T = len(series)
    if self.first_iter:
      self.centroids = self.init_centroids(series, strategy=init_strategy)
      if plot_progress:
        self.plot_centroids(0)
      self.first_iter = False

    # Start learning
    for iter in tqdm(range(max_iters)):
      curr_pos = 0
      new_subsequences = []
      new_segmentation = [0]
      new_labels = []

      # Greedy segmentation through the entire time series
      while curr_pos+self.l_min < self.T:
        if curr_pos+self.l_max >= self.T:
          curr_seg, curr_seg_len, curr_seg_label = self.compute_optimal_segment(series[curr_pos:])
        else:
          curr_seg, curr_seg_len, curr_seg_label = self.compute_optimal_segment(series[curr_pos:curr_pos + self.l_max])
        new_subsequences.append(curr_seg)
        new_labels.append(curr_seg_label)
        curr_pos += curr_seg_len
        new_segmentation.append(curr_pos)
      new_subsequences = np.array(new_subsequences, dtype=object)
      new_labels = np.array(new_labels)
      new_centroids, new_loss = self.update_centroids(new_subsequences, new_labels, barycenter)
      self.hist_loss.append(new_loss)

      # Save results
      if store_res:
        self.save_results(new_loss, dataname, init_strategy, barycenter)

      # Display progress
      PLOT_ITER = (iter == 0) or ((iter+1) % 10 == 0)
      if PLOT_ITER:
        print(f"-- Iter {iter+1} --")
        print(f"Total Loss: {new_loss}")
        if plot_progress:
          self.plot_centroids(iter+1)

      # Early-stop criteria
      if self.stop_criteria(new_centroids, new_loss, epsilon=1e-6):
        print(f"-- Converge and stop at iter {iter+1} --")
        break

      # Update results
      self.total_loss = new_loss
      self.centroids = new_centroids
      self.labels = new_labels
      self.subsequences = new_subsequences
      self.segmentation = new_segmentation

    # print(f"-- Terminate at iter {max_iters} --")


  ''' ~func: choose the optimal segment with the minimum DTW between all centroids at each position '''
  def compute_optimal_segment(self, sequence):
    subsequences = self._possible_segments(sequence, norm=False)
    min_dist_label_pairs = np.array([self._compute_nearest_dtw(seq) for seq in subsequences])
    min_dist_pos = np.argmin(min_dist_label_pairs[:, 0])
    seq_optimal = subsequences[min_dist_pos]
    l_optimal = self.l_min + min_dist_pos
    label_optimal = min_dist_label_pairs[min_dist_pos, 1]
    return seq_optimal, int(l_optimal), int(label_optimal)


  ''' ~func: initialize the centroids at the begining '''
  def init_centroids(self, series, strategy='kmeans++'):
    if strategy=='random_noise':
      return np.random.normal(0, 0.2, (self.n_clusters, self.l_max))
    elif strategy=='random_sample':
      random_indices = np.random.choice(self.T, size=self.n_clusters, replace=False)
      return [series[idx:idx + self.l_max] for idx in random_indices]
    elif strategy=='kmeans++':
      return self._kmeanspp_init(series)
    else:
      print("Invailid strategy for cluster initialization. Use K-Means++ instead.")
      return self._kmeanspp_init(series)
    

  ''' ~func: update the centroids given the new segmented subsequences and labels '''
  def update_centroids(self, subsequences, labels, barycenter='dba'):
    new_centroids = []
    total_loss = 0.0
    for i in range(self.n_clusters):
      cluster_segments = subsequences[labels==i]
      cluster_segments = self._normalize_segments(cluster_segments)
      if len(cluster_segments) >= 1:
        centroid = self._compute_centroid(cluster_segments, barycenter)
      else:
        centroid = self.centroids[i]
      new_centroids.append(centroid)
      inner_loss = self._compute_inner_loss(cluster_segments, centroid)
      total_loss += inner_loss
    return np.array(new_centroids, dtype=object), total_loss


  ''' ~util: compute the DTW distance with scaled magnitude '''
  def _compute_scaled_dtw(self, a, b):
    a = self._normalize_arr(a)
    b = self._normalize_arr(b)
    return dtai_dtw.distance(a, b)
    ## return dtai_dtw.distance_fast(a, b, use_pruning=True)


  ''' ~util: normalize for scaled magnitude '''
  def _normalize_arr(self, arr):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_arr = scaler.fit_transform(arr.reshape(-1, 1))
    return normalized_arr.flatten()


  ''' ~util: find the possible segnments in a given sequence in length of l_max '''
  def _possible_segments(self, sequence, norm=True):
    if norm:
      subseqs = [self._normalize_arr(sequence[:l]) for l in range(self.l_min, self.l_max+1)]
    else:
      subseqs = [sequence[:l] for l in range(self.l_min, self.l_max+1)]
    return np.array(subseqs, dtype=object)


  ''' ~util: find the centroid with the minimum DTW distance '''
  def _compute_nearest_dtw(self, subsequence):
    distances = []
    for i, centroid in enumerate(self.centroids):
      distance = self._compute_scaled_dtw(subsequence, centroid)
      distances.append(distance)
    return min(np.array(distances)), int(np.argmin(np.array(distances)))


  ''' ~util: compute centroid given the subsequences with a cluster '''
  def _compute_centroid(self, subsequences, barycenter='dba', init_barycenter=None):
    # Use DTW
    if barycenter=='dba':
      return dtw_barycenter_averaging(subsequences, barycenter_size=self.l_max, tol=1e-5).flatten()
    # Use soft-DTW
    elif barycenter=='softdtw':
      subseq_lengths = [len(subseq) for subseq in subsequences]
      index_maxlen = np.argmax(subseq_lengths)
      interp_f = interp1d(np.linspace(0, 1, len(subsequences[index_maxlen])), subsequences[index_maxlen])
      subsequences[index_maxlen] = interp_f(np.linspace(0, 1, self.l_max))
      return softdtw_barycenter(subsequences, gamma=self.gamma, tol=1e-5).flatten()


  ''' ~util: normalize the segments into the standard scale in magnitude '''
  def _normalize_segments(self, segments):
    segments_norm = []
    for seg in segments:
      max_value = max(seg)
      min_value = min(seg)
      seg_norm = (seg-min_value)/(max_value-min_value)
      segments_norm.append(seg_norm)
    return np.array(segments_norm, dtype=object)


  ''' ~util: compute the inner cluster loss '''
  def _compute_inner_loss(self, subsequences, centroid):
    inner_loss = 0.0
    for subseq in subsequences:
      inner_loss += self._compute_scaled_dtw(subseq, centroid)
    return inner_loss


  ''' ~func: plot the centroids '''
  def plot_centroids(self, iter):
    n_rows = 1
    n_cols = self.n_clusters
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows * 0.7))
    for i, centroid in enumerate(self.centroids):
      axs[i].plot(centroid)
      axs[i].get_xaxis().set_visible(False)
      axs[i].get_yaxis().set_visible(False)
    plt.show()


  ''' ~util: plot the total and inner cluster loss '''
  def _compute_cluster_loss(self, clusters, centroids):
    total_loss = 0.0
    inner_losses = []
    for i, cluster in enumerate(clusters):
      inner_loss = 0.0
      n_subseqs = len(cluster)
      for subseq in cluster:
        inner_loss += self._compute_scaled_dtw(subseq, centroids[i])
      total_loss += inner_loss
      inner_losses.append(inner_loss)
    return total_loss, np.array(inner_losses)


  ''' ~util: k-means++ initialization '''
  def _kmeanspp_init(self, series):
    centroids = []
    first_centroid = True
    remain_idxs = np.arange(self.T-self.l_max+1)
    # iteratively add the next centroid probably far from the existings following k-means++
    while len(centroids)<self.n_clusters:
      select_idx, remain_idxs = self._select_next_centroid(series,
                                                           remain_idxs, centroids,
                                                           first_centroid)
      centroids.append(series[select_idx:select_idx+self.l_max])
    return np.array(centroids, dtype=object)


  ''' ~util: select the next centroid following k-means++ '''
  def _select_next_centroid(self, series, remain_idxs, centroids, first_centroid):
    if first_centroid:
      weight = np.ones(len(remain_idxs))/len(remain_idxs)
      first_centroid = False
    else:
      weight = self._weight_next_centroid(series, remain_idxs, centroids)
    select_idx = np.random.choice(remain_idxs, p=weight)
    remain_idxs = np.delete(remain_idxs, np.where(remain_idxs==select_idx))
    return select_idx, remain_idxs


  ''' ~util: compute the k-means++ probability of the next centroid '''
  def _weight_next_centroid(self, series, remain_idxs, centroids):
    weights = np.zeros(len(remain_idxs))
    for idx in remain_idxs:
      seq = series[idx:idx+self.l_max]
      dist = [self._compute_scaled_dtw(seq, c) for c in centroids]
      weights[idx] += min(dist)
    return weights/sum(weights)


  ''' ~func: early stop criteria '''
  def stop_criteria(self, new_centroids, new_loss, epsilon=1e-6):
    for old, new in zip(self.centroids, new_centroids):
      dtw = self._compute_scaled_dtw(old, new)
      if dtw > epsilon:
        self.coverage = 0
        return False
    if abs(self.total_loss - new_loss) > epsilon:
      self.coverage = 0
      return False
    if self.coverage <= 3:
      self.coverage += 1
      return False
    return True


  ''' ~func: save the results '''
  def save_results(self, new_loss, dataname, init_strategy, barycenter):
    if new_loss<=self.min_loss:
      self.min_loss = new_loss
      df_centroids = pd.DataFrame(self.centroids)
      df_labels = pd.DataFrame(self.labels)
      df_subsequences = pd.DataFrame(self.subsequences)
      df_segmentation = pd.DataFrame(self.segmentation)
      dict_init = {'kmeans++': 'kmpp',
                  'random_sample': 'rs',
                  'random_noise': 'rn',
                  'reference':'ref'}
      filename = f"sisc_{dataname}_k{self.n_clusters}_l{self.l_min}-{self.l_max}_{barycenter[:4]}_{dict_init[init_strategy]}"
      df_centroids.to_csv(res_path + filename +'_centroids.csv')
      df_labels.to_csv(res_path + filename + '_labels.csv')
      df_subsequences.to_csv(res_path + filename + '_subsequences.csv')
      df_segmentation.to_csv(res_path + filename + '_segmentation.csv')


############################
# Train the SISC algorithm #
############################

def train_recognition_module(
    data, dataname=None,
    n_clusters=13,
    l_min=10, l_max=21,
    max_iters=100,
    init_strategy='kmeans++',
    barycenter='dba', gamma=0.5,
    plot_progress=False,
    plot_loss=True,
    store_res=False
    ):
  
  # Run SISC
  sisc = SISC(n_clusters, l_min, l_max, gamma)
  sisc.fit(series=data, 
           max_iters=max_iters,
           init_strategy=init_strategy, 
           barycenter=barycenter,
           plot_progress=plot_progress,
           store_res=store_res, dataname=dataname)
  centroids = sisc.centroids
  labels = sisc.labels
  subsequences = sisc.subsequences
  segmentation = sisc.segmentation

  # Save results
  if store_res:
    df_centroids = pd.DataFrame(centroids)
    df_labels = pd.DataFrame(labels)
    df_subsequences = pd.DataFrame(subsequences)
    df_segmentation = pd.DataFrame(segmentation)
    dict_init = {'kmeans++': 'kmpp',
                 'random_sample': 'rs',
                 'random_noise': 'rn',
                 'reference':'ref'}
    filename = f"sisc_{dataname}_k{n_clusters}_l{l_min}-{l_max}_{barycenter[:4]}_{dict_init[init_strategy]}"
    df_centroids.to_csv(res_path + filename + '_centroids.csv')
    df_labels.to_csv(res_path + filename + '_labels.csv')
    df_subsequences.to_csv(res_path + filename + '_subsequences.csv')
    df_segmentation.to_csv(res_path + filename + '_segmentation.csv')
  
  # Plot training loss
  if plot_loss:
    hist_loss = np.array(sisc.hist_loss)
    plt.figure(figsize=(4,1))
    plt.plot(hist_loss, label='total_loss')
    plt.legend(loc='upper right')

  return centroids, labels, subsequences, segmentation









