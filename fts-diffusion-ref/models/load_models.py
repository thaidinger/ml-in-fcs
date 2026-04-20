import torch
import os
torch.set_default_dtype(torch.float)

from models.scaling_autoencoder import *
from models.pattern_conditioned_diffusion import *
from models.pattern_generation_module import *
from models.pattern_evolution_module import *
from models.model_params import *

model_path = 'trained_models/'



##############################################
# Load the trained pattern generation module #
##############################################

def get_pgm_name(state_dict=True):
  """ Get the stored name of the trained pattern generation module """
  dataname = prm_params['dataname']
  channels = pgm_params['pcdm_channels']
  n_patterns = pgm_params['n_patterns']
  n_steps = pgm_params['pcdm_n_steps']
  lr = pgm_params['lr']
  lr_str = "{:.0e}".format(lr)
  loss_weights = pgm_params['loss_weights']
  pad_weight = pgm_params['pad_weight']
  scale_weight = pgm_params['scale_weight']
  # pgm_name = f"pgm_c{min(channels)}-{max(channels)}_{dataname}_k{n_patterns}_n{n_steps}_lr{lr_str}_dw{loss_weights[1]}_pw{pad_weight}_sw{scale_weight}"
  pgm_name = f"pgm-2_c{min(channels)}-{max(channels)}_{dataname}_k{n_patterns}_n{n_steps}_lr{lr_str}_dw{loss_weights[1]}_pw{pad_weight}_sw{scale_weight}"
  if state_dict:
    return pgm_name + ".pth"
  else:
    return pgm_name + ".pt"


def build_sae(device):
  """ Instantiate the scaling AE """
  input_dim = pgm_params['sae_input_dim']
  hidden_dim = pgm_params['sae_hidden_dim']
  output_dim = pgm_params['sae_output_dim']
  custom_pad_length = pgm_params['sae_custom_pad_length']
  sae = ScalingAE(input_dim, hidden_dim, output_dim, custom_pad_length, device)
  return sae


def build_pcdm(device):
  """ Instantiate the pattern-conditioned diffusion network """
  n_steps = pgm_params['pcdm_n_steps']
  series_length = pgm_params['pcdm_series_length']
  channels = pgm_params['pcdm_channels']
  latent_dim = pgm_params['pcdm_latent_dim']
  time_embed_dim= pgm_params['pcdm_time_embed_dim']
  time_hidden_dim = pgm_params['pcdm_time_hidden_dim']
  min_beta, max_beta = pgm_params['pcdm_min_beta'], pgm_params['pcdm_max_beta']
  tcn = TCN2(n_steps, series_length, channels,
             input_dim=latent_dim,
             time_embed_dim=time_embed_dim, time_hidden_dim=time_hidden_dim,
             device=device)
  pcdm = PCDM(tcn,
              n_steps=n_steps,
              min_beta=min_beta, max_beta=max_beta,
              device=device)
  return pcdm


def build_pgm(device):
  """ Instantiate the pattern generation module """
  sae = build_sae(device).to(device)
  pcdm = build_pcdm(device).to(device)
  pgm = PatternGenerationModule(sae, pcdm, condition=True, device=device)
  return pgm


def load_pattern_generation_module(state_dict=True):
  """ Load the trained pattern generation module with default (or input) hyper-parameters """
  filename = get_pgm_name(state_dict)
  filepath = model_path + filename
  if not os.path.exists(filepath):
    legacy_filepath = filepath + '.pth'
    if os.path.exists(legacy_filepath):
      filepath = legacy_filepath
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pgm = build_pgm(device)
  pgm.load_state_dict(torch.load(filepath, map_location=device))
  pgm.eval()
  return pgm



#############################################
# Load the trained pattern evolution module #
#############################################
def get_pem_name(state_dict=True):
  dataname = prm_params['dataname']
  n_patterns = pem_params['n_patterns']
  evo_embed_dim = pem_params['evo_embed_dim']
  evo_hidden_dim = pem_params['evo_hidden_dim']
  lr = pem_params['lr']
  lr_str = "{:.0e}".format(lr)
  loss_weights = pem_params['loss_weights']
  pem_name = f"pem_{dataname}_k{n_patterns}_e{evo_embed_dim}_h{evo_hidden_dim}_lr{lr_str}_pw{loss_weights[0]}_lw{loss_weights[1]}_mw{loss_weights[2]}"
  if state_dict:
    return pem_name + ".pth"
  else:
    return pem_name + ".pt"


def build_pem(device):
  """ Instantiate the pattern evolution network """
  n_patterns = pem_params['n_patterns']
  evo_embed_dim = pem_params['evo_embed_dim']
  evo_hidden_dim = pem_params['evo_hidden_dim']
  length_range = prm_params['l_max'] - prm_params['l_min'] + 1
  pem = PatternEvolutionModule(n_patterns, length_range, evo_embed_dim, evo_hidden_dim, device)
  return pem


def load_pattern_evolution_module(state_dict=True):
  """ Load the trained pattern evolution module with default (or input) hyper-parameters """
  filename = get_pem_name(state_dict)
  filepath = model_path + filename
  if not os.path.exists(filepath):
    legacy_filepath = filepath + '.pth'
    if os.path.exists(legacy_filepath):
      filepath = legacy_filepath
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pem = build_pem(device).to(device)
  pem.load_state_dict(torch.load(filepath, map_location=device))
  pem.eval()
  return pem



##############################
# Load trained FTS-Diffusion #
##############################

def load_ftsdiffusion():
  """ Load the trained FTS-Diffuion (pattern generation and evolution modules) """
  pgm = load_pattern_generation_module()
  pem = load_pattern_evolution_module()
  model_ftsdiffusion = {'generation': pgm, 'evolution': pem}
  return model_ftsdiffusion









