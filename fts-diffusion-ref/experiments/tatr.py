import numpy as np
import pandas as pd
from tqdm import tqdm

from models.model_params import prm_params
from experiments.utils_downstream import *
from experiments.predictor_lstm import *

res_path = 'res/'



##############################################
# Train on Augmentation, Test on Real (TATR) #
##############################################

def TATR(datatype='prices',
         n_runs=100, n_augmentations=100, aug_length=252, 
         n_epochs=100, window_size=64, ahead=1,
         lstm_hidden_dim=64, lstm_loss='mse',
         store_res=True):
  """ Downstream experiment under the TATR setting """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Trained on device: {device}")
  dataname = prm_params['dataname']
  init_dataset, first_segment, test_dataset, scaler = setup_dowmstream_tatr(window_size)
  init_state, init_segment = first_segment
  # init_state, init_segment = torch.tensor(init_state).to(device), torch.tensor(init_segment).to(device)
  errors = np.zeros((n_runs, n_augmentations + 1))
  for run in tqdm(range(n_runs)):
    print(f"[TATR] Run {run+1}/{n_runs} started.", flush=True)
    curr_dataset = copy_dataset_downstream(init_dataset)
    for aug_time in tqdm(range(n_augmentations + 1)):
      if aug_time > 0:
        print(f"[TATR] Run {run+1}/{n_runs} - augmentation {aug_time}/{n_augmentations}: generating synthetic series...", flush=True)
        syn_timeseries = generate_syn_timeseries_downstream(aug_length, 
                                                            init_state, init_segment,
                                                            model='fts-diffusion')
        syn_dataset = create_syn_dataset(syn_timeseries, window_size, scaler, datatype)
        curr_dataset = concat_datasets_downstream(curr_dataset, syn_dataset)
      print(f"[TATR] Run {run+1}/{n_runs} - augmentation {aug_time}/{n_augmentations}: training LSTM...", flush=True)
      # Train on augmentation
      model = separate_train_lstm_predictor(n_epochs,
                                            curr_dataset,# dataloader_train,
                                            input_dim=1,
                                            hidden_dim=lstm_hidden_dim,
                                            output_dim=ahead,
                                            n_layers=2,
                                            criterion=lstm_loss,
                                            verbose=True)
      # Test on real
      error = test_on_real(model, test_dataset, scaler, criterion='mape')
      errors[run, aug_time] = error
      print(f"Run {run:2d} - Aug {aug_time:2d} Size: {len(curr_dataset) + window_size - 1} MAPE: {error:.7f}", flush=True)
      if store_res:
        df_errors = pd.DataFrame(errors)
        filename = f"res_tatr_{dataname}-{datatype}_{ahead}ahead_h{lstm_hidden_dim}_{lstm_loss.lower()}.csv"
        df_errors.to_csv(res_path + filename, index=False)
  print(f"Run {run} completed.", flush=True)









