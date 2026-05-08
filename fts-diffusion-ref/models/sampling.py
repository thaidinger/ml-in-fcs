import numpy as np
import pandas as pd
import random
import torch
from torch.nn import functional as F
torch.set_default_dtype(torch.float)
from tqdm import tqdm

from models.utils_sampling import *
from models.pattern_generation_module import *
from models.pattern_evolution_module import *
from models.load_models import *

data_path = 'data/'



#########################################
# Predict the state of the next segment #
#########################################

def state_evolution_ftsdiffusion(model, state, l_min):
  """ Predict the next state by employing pattern evolution module in FTS-Diffuson """
  n_patterns = model.n_patterns
  range_length = model.range_length
  pred = model(state)

  pred_pattern_values = pred[:, :n_patterns]
  probabilities_pattern = F.softmax(pred_pattern_values, dim=1)
  pred_pattern = torch.argmax(probabilities_pattern, dim=1).unsqueeze(0)
  pred_length_values = pred[:, n_patterns:n_patterns+range_length]
  probabilities_length = F.softmax(pred_length_values, dim=1)
  pred_length = torch.argmax(probabilities_length, dim=1).unsqueeze(0) + l_min
  pred_magnitude = pred[:, n_patterns+range_length:].float()

  next_state = torch.cat((pred_pattern, pred_length, pred_magnitude), dim=1)

  return next_state



#############################
# Generate the next segment #
#############################

def segment_generation_ftsdiffusion(model, state, patterns):
  """ Generate segments by employing FTS-Diffuson model """
  p_t, l_t, m_t = state[0]
  p_t, l_t = p_t.long().detach().cpu().numpy(), l_t.long().unsqueeze(0).detach().cpu().numpy()
  pattern = torch.tensor(patterns[p_t]).unsqueeze(0).float()
  x_, _ = model.generate(pattern, l_t)
  new_segment = x_.squeeze(0)[:l_t.squeeze(0)]
  # m_new = (max(new_segment)- min(new_segment))
  # new_segment = (new_segment / m_new) * m_t.squeeze(0)
  new_segment = new_segment * m_t.squeeze(0)
  
  return new_segment.detach().cpu().numpy()



######################################
# Generate new synthetic time series #
######################################

def sampling_timeseries_ftsdiffusion(
    T,
    sample_idx=None,
    plot_ts=False,
    store_ts=False,
    store_actual=False,
    dataname=''):
  """ Generate the synthetic time series using FTS-Diffusion (for sampling module) """
  model = load_ftsdiffusion()
  l_min = prm_params['l_min']
  subseqs_real, _, patterns = sampling_inputs()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  patterns = patterns.to(device)
  
  # Initialize the first segment
  state, first_segment = get_init_state_by_index(sample_idx)
  timeseries = list(first_segment)
  states = [state.squeeze(1).detach().cpu().numpy()]

  # Generate new synthetic time series
  pbar = tqdm(total = T)
  curr_T = len(timeseries)
  pbar.update(curr_T)
  while curr_T < T:
    state = torch.tensor(state).to(device)
    state = state_evolution_ftsdiffusion(model['evolution'], state, l_min)
    segment = segment_generation_ftsdiffusion(model['generation'], state, patterns)
    segment = segment - segment[0] + timeseries[-1]
    segment_length = len(segment)
    update_by = min(segment_length, T - curr_T)
    pbar.update(update_by)
    states.append(state.squeeze(1).detach().cpu().numpy())
    timeseries.extend(segment)
    curr_T += segment_length
  pbar.close()
  timeseries = np.array(timeseries[:T])
  states = np.array(states).reshape(-1, 3)

  ## Plot the generated synthetic time series
  if plot_ts:
    plt.figure(figsize=(10,3))
    plt.plot(timeseries)

  # Store the generated synthetic time series (and the actual time series at the same period, if applicable)
  if store_ts:
    df_ts = pd.DataFrame(timeseries)
    df_states = pd.DataFrame(states)
    df_ts.to_csv(data_path + f"syn_{dataname}_ftsdiffusion_{T}.csv")
    df_states.to_csv(data_path + f"syn_{dataname}_ftsdiffusion_{T}_states.csv")
    print(f"syn_{dataname}_ftsdiffusion_{T} stored.")
    if store_actual:
      actual_ts = np.concatenate(subseqs_real[sample_idx:])
      actual_ts_sampled = actual_ts[:T]
      df_actual = pd.DataFrame(actual_ts_sampled)
      df_actual.to_csv(data_path + f"syn_{dataname}_actual_{T}.csv")
      print(f"syn_{dataname}_actual_{T} stored.")

  return timeseries


def generate_timeseries_ftsdiffusion(T, init_state=None, init_segment=None, patterns=None):
  """ Generate the synthetic time series using FTS-Diffusion (for downstream experiment) """
  model = load_ftsdiffusion()
  l_min = prm_params['l_min']
  if init_state is None:
    state, first_segment = get_init_state(init_state, init_segment)
  else:
    state, first_segment = init_state, init_segment
  if patterns is None:
    _, _, patterns = sampling_inputs()
  timeseries = list(first_segment)
  curr_T = len(timeseries)
  while curr_T < T:
    state = state_evolution_ftsdiffusion(model['evolution'], state, l_min)
    segment = segment_generation_ftsdiffusion(model['generation'], state, patterns)
    segment = segment - segment[0] + timeseries[-1]
    segment_length = len(segment)
    timeseries.extend(segment)
    curr_T += segment_length
  timeseries = np.array(timeseries[:T])
  return timeseries









