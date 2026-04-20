import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
torch.set_default_dtype(torch.float)

from utils.prepare_data import *
from models.sampling import *

res_path = 'res/'
fig_path = 'figs/'



##################
# Setup for TMTR #
##################

def setup_dowmstream_tmtr(mix_length):
  """ Setup for the downstream experiment under the TMTR setting """
  downstream_timeseries, segments_test, labels_test, lengths_test = get_downstream_data()
  real_timeseries = downstream_timeseries[:mix_length]
  first_segment = init_first_segment(segments_test, labels_test, lengths_test)
  test_timeseries = downstream_timeseries[mix_length:]
  return real_timeseries, first_segment, test_timeseries



#######################################
# Create the mixture dataset fot TMTR #
#######################################

def create_mixture_dataset(mix_length, proportion, 
                           real_timeseries, syn_timeseries,
                           window_size, scaler):
  """ Create the mixture dataset with a certain proportion of synthetic data for TMTR """
  if proportion == 0:
    return create_mix_dataset(real_timeseries, mix_length, window_size, scaler)
  if proportion == 1:
    return create_mix_dataset(syn_timeseries, mix_length, window_size, scaler)
  syn_length = int(mix_length * proportion)
  real_length = mix_length - syn_length
  mix_dataset_real = create_mix_dataset(real_timeseries, real_length, window_size, scaler)
  mix_dataset_syn = create_mix_dataset(syn_timeseries, syn_length, window_size, scaler)
  mix_dataset = concat_datasets_downstream(mix_dataset_real, mix_dataset_syn)
  return mix_dataset


def sample_tmtr_timeseries(timeseries, length):
  """ Sample the timeseries of the given length for mixture component of real or synthetic """
  total_length = len(timeseries)
  if total_length == length:
    return timeseries
  start_idx = np.random.randint(0, total_length - length)
  return timeseries[start_idx:start_idx + length]


def create_mix_dataset(timeseries, length, window_size, scaler):
  """ Create the TensorDataset for mixture component of real or synthetic """
  mix_timeseries = sample_tmtr_timeseries(timeseries, length)
  mix_dataset = Timeseries2Dataset_Downstream(mix_timeseries, window_size, scaler)
  return mix_dataset



##################
# Setup for TATR #
##################

def setup_dowmstream_tatr(window_size):
  """ Setup for the downstream experiment under the TATR setting """
  downstream_timeseries, segments_test, labels_test, lengths_test = get_downstream_data()
  init_timeseries = downstream_timeseries[:252*5]
  init_dataset, scaler = init_tatr_set(init_timeseries, window_size)
  first_segment = init_first_segment(segments_test, labels_test, lengths_test)
  test_timeseries = downstream_timeseries[252*5:]
  test_dataset = init_tatr_set(test_timeseries, window_size, scaler)
  setup_string = "Setup for TATR completed.\n"
  setup_string += f"Length of initial training time series: {len(init_timeseries)}\n"
  setup_string += f"Length of initial test time series: {len(test_timeseries)}"
  print(setup_string)
  return init_dataset, first_segment, test_dataset, scaler



##############################
# Initalize the TATR dataset #
##############################

def init_tatr_set(timeseries, window_size=64, scaler=None):
  """ Initialize the TATR dataset with real data """
  return Timeseries2Dataset_Downstream(timeseries, window_size, scaler)



##################################
# Generate synthetic time series #
##################################

def generate_syn_timeseries_downstream(length, init_state, init_segment, model='fts-diffusion'):
  """ Provide synthetic timeseries to augment the dataset """
  if model.lower()=='fts-diffusion':
    return generate_timeseries_ftsdiffusion(length, init_state, init_segment)


def create_syn_dataset(syn_timeseries, window_size=64, scaler=None, datatype='prices'):
  """ Convert synthetic data to dataset """
  if datatype == 'returns':
    syn_timeseries = (syn_timeseries[1:] - syn_timeseries[:-1]) / syn_timeseries[:-1]
  if datatype == 'volatility':
    syn_timeseries = (syn_timeseries[1:] - syn_timeseries[:-1]) / syn_timeseries[:-1]
    syn_timeseries = pd.Series(syn_timeseries).rolling(window=5).std().dropna().values
  return Timeseries2Dataset_Downstream(syn_timeseries, window_size, scaler)



##########################################
# Data processing for both TMTR and TATR #
##########################################

def get_downstream_data():
  """ Get the downstream timeseries """
  _, test_set, _ = prepare_segments()
  segments_test, labels_test, lengths_test = test_set
  downstream_timeseries = np.concatenate(segments_test)
  return downstream_timeseries, segments_test, labels_test, lengths_test


def copy_dataset_downstream(dataset):
  """ Copy the dataset """
  dataset_copy = dataset.clone().detach()
  dataset_copy.requires_grad = False
  return dataset_copy


def init_first_segment(segments_test, labels_test, lengths_test):
  """ Get the initial state of the first segment """
  init_segment = segments_test[0]
  init_pattern = labels_test[0]
  init_length = lengths_test[0]
  init_magnitude = max(init_segment) - min(init_segment)
  init_state = np.stack((init_pattern, init_length, init_magnitude), axis=0)
  init_state = torch.tensor(init_state).view(1, -1)
  return init_state, init_segment


def Timeseries2Dataset_Downstream(timeseries, window_size=64, scaler=None):
  """ Convert the downstream timeseries to rolling samples """
  if scaler is None:
    scaler =MinMaxScaler(feature_range=(-1, 1))
    timeseries = scaler.fit_transform(timeseries.reshape(-1, 1))
    timeseries = torch.tensor(timeseries).squeeze(1)
    dataset = timeseries.unfold(0, window_size, 1).float()
    return dataset, scaler
  else:
    timeseries = scaler.transform(timeseries.reshape(-1, 1))
    timeseries = torch.tensor(timeseries).squeeze(1)
    dataset = timeseries.unfold(0, window_size, 1).float()
    return dataset


def concat_datasets_downstream(dataset_1, dataset_2):
  """ Concatenate the datasets """
  return torch.cat((dataset_1, dataset_2), dim=0)


def construct_dataloader_downstream(dataset, batch_size=16):
  """ Construct the dataloader for the augmented dataset """
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)



#################################################
# Plot the results of the downstream experiment #
#################################################

def plot_dowmstream_tmtr(df, store_fig=False):
  """ Plot the results of the downstream experiment under the TMTR setting """
  error_avg = df['avg'].values
  error_min = df['min'].values
  error_max = df['max'].values
  n_proportions = len(error_avg)
  x_range = (np.arange(n_proportions) * (100 / (n_proportions - 1))).astype(int)
  plt.plot(x_range, error_avg)
  plt.fill_between(x_range, error_min, error_max, alpha=0.2)
  plt.xlabel('Syn. Prop. (%)')
  plt.ylabel('MAPE')
  if store_fig:
    plt.gcf().savefig(fig_path + "res_tmtr.pdf", bbox_inches='tight')


def plot_dowmstream_tatr(df, aug_length, store_fig=True):
  """ Plot the results of the downstream experiment under the TATR setting """
  error_avg = df['avg'].values
  error_min = df['min'].values
  error_max = df['max'].values
  x_range = np.arange(1, len(error_avg) + 1) * (252 * 10)
  plt.plot(x_range, error_avg)
  plt.fill_between(x_range, error_min, error_max, alpha=0.2)
  plt.axhline(y=error_avg[0], color='gray', linestyle='--')
  plt.xlabel('Aug. Size')
  plt.ylabel('MAPE')
  if store_fig:
    plt.gcf().savefig(fig_path + "res_tatr.pdf", bbox_inches='tight')


def summarize_results(exp, dataname, ahead, lstm_hidden_dim, lstm_loss, datatype='prices'):
  """ Summarize the downstream results """
  filename = f"res_{exp}_{dataname}-{datatype}_{ahead}ahead_h{lstm_hidden_dim}_{lstm_loss.lower()}.csv"
  filepath = res_path + filename
  if not os.path.exists(filepath):
    # Backward-compatible fallback for older file naming.
    filepath = res_path + f"res_{exp}_{dataname}_{ahead}ahead_h{lstm_hidden_dim}_{lstm_loss.lower()}.csv"
  df_res = pd.read_csv(filepath)
  errors = df_res.values
  n_runs, n_iters = errors.shape
  summary = np.zeros((3, n_iters))
  for i in range(n_iters):
    res = errors[:, i]
    res.sort()
    pencentile = int(np.ceil(len(res) * 0.025))
    summary[0, i] = np.mean(res[pencentile:-pencentile])
    summary[1, i] = res[pencentile]
    summary[2, i] = res[-pencentile]
  df_summary = pd.DataFrame()
  df_summary['avg'] = summary[0, :]
  df_summary['min'] = summary[1, :]
  df_summary['max'] = summary[2, :]
  df_summary.to_csv(res_path + f"res_{exp}_summary.csv")
  return df_summary









