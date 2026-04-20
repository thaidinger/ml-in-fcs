import numpy as np
import random
import torch

# Setting reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



#################################################
# Hyperparameters of pattern recognition module #
#################################################

prm_params = {
  'dataname': 'sp500',
  'k': 14,
  'l_min': 10,
  'l_max': 21,
  'max_iters': 20,
  'barycenter': 'dba',
  'init_strategy': 'kmeans++'
}


################################################
# Hyperparameters of pattern generation module #
################################################

pgm_params = {
  'sae_input_dim': 1,
  'sae_hidden_dim': 1,
  'sae_output_dim': 1,
  'sae_custom_pad_length': prm_params['l_max'],
  'pcdm_n_steps': 30,
  'pcdm_series_length': prm_params['l_max'],
  'pcdm_latent_dim': 1,
  'pcdm_time_embed_dim': 32,
  'pcdm_time_hidden_dim': 32,
  'pcdm_channels': [48, 64, 80, 80, 64, 48],
  'pcdm_min_beta': 1e-4,
  'pcdm_max_beta': 0.02,
  'n_patterns': prm_params['k'],
  'n_steps': 30,
  'batch_size': 32,
  'n_epochs': 30,
  'lr': 4e-4,
  'loss_weights': [0.98, 0.01],
  'pad_weight': 1,
  'scale_weight': 0.01
}


###############################################
# Hyperparameters of pattern evolution module #
###############################################

pem_params = {
  'evo_embed_dim': 196,
  'evo_hidden_dim': 32,
  'n_patterns': prm_params['k'],
  'batch_size': 32,
  'n_epochs': 60,
  'lr': 4e-4,
  'loss_weights': [0.05, 0.01, 0.94]
}









