import os
import torch
torch.set_default_dtype(torch.float)

from models.pattern_recognition_module import *
from models.scaling_autoencoder import *
from models.pattern_conditioned_diffusion import *
from models.pattern_generation_module import *
from models.pattern_evolution_module import *
from models.model_params import *
from models.load_models import *
from utils.prepare_data import *

model_path = 'trained_models/'



##############################################
# Train FTS-Diffusion (generation-evolution) #
##############################################

def train_ftsdiffusion(fts, train_test_split=0.8, store_model=True):
  """ Train FTS-Diffusion (generation and evolution module) """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if _has_recognition_artifacts():
    print("Found existing SISC artifacts in res/. Skipping recognition stage.")
  else:
    train_ftsdiffusion_recognition(fts, store_model)
  has_pgm = _has_generation_artifact()
  has_pem = _has_evolution_artifact()
  if has_pgm and has_pem:
    print("Found existing generation/evolution checkpoints in trained_models/. Skipping training stage.")
    return
  dataloader_pgm, dataloader_pem = get_training_data(train_test_split)
  if has_pgm:
    print("Found existing generation checkpoint. Skipping generation training.")
  else:
    train_ftsdiffusion_generation(dataloader_pgm, device, store_model)
  if has_pem:
    print("Found existing evolution checkpoint. Skipping evolution training.")
  else:
    train_ftsdiffusion_evolution(dataloader_pem, device, store_model)


def _has_recognition_artifacts():
  dataname = prm_params['dataname']
  n_clusters = prm_params['k']
  l_min, l_max = prm_params['l_min'], prm_params['l_max']
  barycenter = prm_params['barycenter']
  init_strategy = prm_params['init_strategy']
  dict_init = {'kmeans++': 'kmpp', 'random_sample': 'rs', 'random_noise': 'rn'}
  if init_strategy not in dict_init:
    return False
  prefix = f"res/sisc_{dataname}_k{n_clusters}_l{l_min}-{l_max}_{barycenter[:4]}_{dict_init[init_strategy]}"
  required_files = [
    prefix + '_centroids.csv',
    prefix + '_labels.csv',
    prefix + '_subsequences.csv',
    prefix + '_segmentation.csv',
  ]
  return all(os.path.exists(path) for path in required_files)


def _has_generation_artifact():
  filename = get_pgm_name(state_dict=True)
  filepath = os.path.join(model_path, filename)
  return os.path.exists(filepath) or os.path.exists(filepath + '.pth')


def _has_evolution_artifact():
  filename = get_pem_name(state_dict=True)
  filepath = os.path.join(model_path, filename)
  return os.path.exists(filepath) or os.path.exists(filepath + '.pth')


def train_ftsdiffusion_recognition(fts, store_model=True):
  """ Train the pattern recognition module in FTS-Diffusion """
  dataname = prm_params['dataname']
  _ = train_recognition_module(
    fts, dataname=dataname,
    n_clusters=prm_params['k'],
    l_min=prm_params['l_min'], l_max=prm_params['l_max'],
    max_iters=prm_params['max_iters'],
    init_strategy=prm_params['init_strategy'],
    barycenter=prm_params['barycenter'],
    plot_progress=False,
    plot_loss=True,
    store_res=store_model)


def train_ftsdiffusion_generation(dataloader, device, store_model=True):
  """ Train the pattern generation module in FTS-Diffusion """
  pgm = build_pgm(device).to(device)
  n_epochs = pgm_params['n_epochs']
  lr = pgm_params['lr']
  loss_weights = pgm_params['loss_weights']
  pad_weight = pgm_params['pad_weight']
  scale_weight = pgm_params['scale_weight']
  store_name = get_pgm_name()
  optimizer = optim.Adam(pgm.parameters(), lr)
  train_generation_module(
    pgm,
    dataloader,
    n_epochs, optimizer,
    loss_weights,
    pad_weight=pad_weight,
    scale_weight=scale_weight,
    condition=True,
    device=device,
    store_model=store_model, 
    store_name=store_name)


def train_ftsdiffusion_evolution(dataloader, device, store_model=True):
  """ Train the pattern evolution module in FTS-Diffusion """
  pem = build_pem(device).to(device)
  n_epochs = pem_params['n_epochs']
  lr = pem_params['lr']
  loss_weights = pem_params['loss_weights']
  store_name = get_pem_name()
  train_evolution_module(
    pem, 
    dataloader,
    n_epochs=n_epochs,
    lr=lr,
    loss_weights=loss_weights,
    store_model=store_model,
    store_name=store_name)



##########################
# Setup training dataset #
##########################

def get_training_data(train_test_split=0.8):
  """ Get the training data """
  train_set, _, patterns = prepare_segments(train_test_split)
  segments, labels, lengths = train_set
  pgm_batch_size = pgm_params['batch_size']
  dataloader_pgm = construct_dataloader_generation(
    segments, patterns, labels,
    batch_size=pgm_batch_size, shuffle=True,
    first_order=False, rescale=True)
  pem_batch_size = pem_params['batch_size']
  dataloader_pem = construct_dataloader_evolution(
    segments, labels, lengths,
    batch_size=pem_batch_size, shuffle=False)
  return dataloader_pgm, dataloader_pem









