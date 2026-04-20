import numpy as np
import pandas as pd
from tqdm import tqdm

from models.model_params import prm_params
from experiments.utils_downstream import *
from experiments.predictor_lstm import *



#########################################
# Train on Mixture, Test on Real (TMTR) #
#########################################

def TMTR(datatype='prices',
         n_runs=20, n_proportions=10, mix_length=252*5, 
         n_epochs=200, window_size=64, ahead=1,
         lstm_hidden_dim=64, lstm_loss='mse',
         store_res=True):
  """ Downstream experiment under the TMTR setting """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Trained on device: {device}")
  dataname = prm_params['dataname']
  real_timeseries, first_segment, test_timeseries = setup_dowmstream_tmtr(mix_length)
  init_state, init_segment = first_segment
  # init_state, init_segment = torch.tensor(init_state).to(device), torch.tensor(init_segment).to(device)
  _, scaler = Timeseries2Dataset_Downstream(test_timeseries, window_size)
  test_dataset = Timeseries2Dataset_Downstream(test_timeseries, window_size, scaler)
  errors = np.zeros((n_runs, n_proportions + 1))
  for run in tqdm(range(n_runs)):
    print(f"[TMTR] Run {run+1}/{n_runs}: generating synthetic series...", flush=True)
    syn_timeseries = generate_syn_timeseries_downstream(mix_length, 
                                                        init_state, init_segment,
                                                        model='fts-diffusion')
    for proportion_idx, proportion in enumerate(tqdm(range(0, 101, int(100 / n_proportions)))):
      proportion = proportion / 100
      print(f"[TMTR] Run {run+1}/{n_runs} - proportion {int(proportion * 100)}%: training LSTM...", flush=True)
      mix_dataset = create_mixture_dataset(mix_length, proportion, real_timeseries, syn_timeseries, window_size, scaler)
      # Train on mixture
      model = separate_train_lstm_predictor(n_epochs, 
                                            mix_dataset,# dataloader_train,
                                            input_dim=1, 
                                            hidden_dim=lstm_hidden_dim, 
                                            output_dim=ahead,
                                            n_layers=2,
                                            criterion=lstm_loss,
                                            verbose=True)
      # Test on real
      error = test_on_real(model, test_dataset, scaler, criterion='mape')
      errors[run, proportion_idx] = error
      print(f"Run {run:2d} - Syn. Proportion {int(proportion * 100):3d}% MAPE: {error:.7f}", flush=True)
      if store_res:
        df_errors = pd.DataFrame(errors)
        filename = f"res_tmtr_{dataname}-{datatype}_{ahead}ahead_h{lstm_hidden_dim}_{lstm_loss.lower()}.csv"
        df_errors.to_csv(res_path + filename, index=False)
  print(f"Run {run} completed.", flush=True)

      







