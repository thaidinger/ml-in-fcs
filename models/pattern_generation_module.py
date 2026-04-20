import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
torch.set_default_dtype(torch.float)
from tslearn.metrics import SoftDTWLossPyTorch as SoftDTWLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.scaling_autoencoder import *
from models.pattern_conditioned_diffusion import *
from models.utils_generation import *

model_path = 'trained_models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _checkpoint_paths(store_name):
  if store_name.endswith('.pth'):
    base = store_name[:-4]
  elif store_name.endswith('.pt'):
    base = store_name[:-3]
  else:
    base = store_name
  return model_path + base + '.pth', model_path + base + '.pt'



#############################
# Pattern generation module #
#############################

class PatternGenerationModule(nn.Module):
  """
    Pattern generation module incorporating scaling AE and pattern-conditioned diffusion network (random noise schedule for diffusion network)
    Args
      sae: torch network, scaling AE network
      pcdm: torch network, pattern-conditioned diffusion network
      condition: bool, True for learning conditioned on patterns in pcdm
  """
  def __init__(self, sae, pcdm, condition=True, device=None):
    super().__init__()
    self.device = device
    self.sae = sae.to(device)
    self.pcdm = pcdm.to(device)
    self.n_steps = pcdm.n_steps
    self.condition = condition

  def forward(self, x, p, lengths):
    x, p = x.to(self.device), p.to(self.device)
    batch_size = x.shape[0]
    # SAE encoder
    z, (z_hidden, z_cell) = self.sae.encoder(x, lengths)
    # SAE decoder
    packed_z = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
    x_ = self.sae.decoder(packed_z)
    x_ = x_.squeeze(-1)
    z_out = z.reshape(x_.size())

    # PCDM diffusion
    z = z.reshape(batch_size, 1, -1)
    p = p.unsqueeze(1)
    if self.condition:
      z = z - p
    epsilon = torch.randn_like(z).to(self.device)
    t = torch.randint(0, self.n_steps, (batch_size,)).to(self.device)
    z_noisy = self.pcdm.forward(z, t, epsilon, p).to(self.device)
    # PCDM denoising
    epsilon_theta = self.pcdm.backward(z_noisy, t, p).to(self.device)

    return x_, epsilon, epsilon_theta, z_out

  def generate(self, p, lengths):
    self.sae.eval()
    self.pcdm.eval()
    p = p.to(self.device)
    with torch.no_grad():
      p = p.unsqueeze(1)
      # Sample noise
      batch_size, n_channels, series_len = p.shape
      z_noisy = torch.randn_like(p).to(self.device)
      # PCDM denoising
      z_ = self.denoising_process(z_noisy, p,
                                  batch_size, n_channels, series_len).to(self.device)
      # SAE decoder
      if self.condition:
        z_ = z_ + p
      z_ = z_.reshape(batch_size, -1, 1)
      packed_z = pack_padded_sequence(z_, lengths, batch_first=True, enforce_sorted=False)
      x_ = self.sae.decoder(packed_z)
      x_ = x_.squeeze(-1)
    return x_, z_.reshape(x_.size())

  def denoising_process(self, z_noisy, p,
                        batch_size, n_channels, series_len):
    z_ = z_noisy
    for _, t in enumerate(list(range(self.n_steps))[::-1]):
      timestep = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
      e_theta = self.pcdm.backward(z_, timestep, p)
      alpha_t = self.pcdm.alphas[t]
      alpha_t_bar = self.pcdm.alpha_bars[t]
      z_ = (1 / alpha_t.sqrt()) * (z_ - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * e_theta)
      # Found it also works without the control of magnitude during the denoising process
      if t > 0:
        eta = torch.randn(batch_size, n_channels, series_len).to(self.device)
        beta_t = self.pcdm.betas[t]
        prev_alpha_t_bar = self.pcdm.alpha_bars[t-1] if t > 0 else self.pcdm.alphas[0]
        beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
        sigma_t = beta_tilda_t.sqrt()
        z_ += sigma_t * eta
    return z_



#######################################
# Train the pattern generation module #
#######################################

def train_generation_module(
    pgm,
    dataloader,
    n_epochs,
    optimizer,
    loss_weights,
    pad_weight=1,
    scale_weight=0,
    condition=True,
    device=None,
    store_model=False, store_name="pgm"):
  pgm.to(device)
  pgm.train()
  mse = nn.MSELoss(reduction='sum')
  softdtw = SoftDTWLoss(gamma=.1)
  hist_loss = []
  hist_loss_sae = []
  hist_loss_pcdm = []
  best_loss = float("inf")
  for epoch in tqdm(range(n_epochs)):
    pgm.train()
    total_loss = 0.0
    total_loss_sae = 0.0
    total_loss_pcdm = 0.0
    n_batches = 0
    for x, p, lengths in dataloader:
      x, p = x.to(device), p.to(device)
      optimizer.zero_grad()
      batch_size = x.shape[0]

      x_0_, epsilon, epsilon_theta, z = pgm.forward(x, p, lengths)

      # Compute the loss and update the params.
      mask_data = (torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).int().to(device)
      mask_pad = (torch.ones_like(mask_data) - mask_data).to(device)
      loss_sae = loss_weights[0] * (mse(x * mask_data, x_0_ * mask_data) + pad_weight * mse(x * mask_pad, x_0_ * mask_pad))
      if scale_weight != 0: # Optional: soft DTW between original data and latent representation, better interpretability
        loss_scale = scale_weight * softdtw(x.unsqueeze(1), z.unsqueeze(1)).sum()
        loss_sae += loss_scale
      loss_pcdm = loss_weights[-1] * mse(epsilon, epsilon_theta)
      loss = loss_sae + loss_pcdm
      total_loss += loss.item()
      total_loss_sae += loss_sae.item()
      total_loss_pcdm += loss_pcdm.item()
      loss.backward()
      optimizer.step()
      n_batches += 1

    epoch_loss = total_loss / n_batches
    hist_loss.append(epoch_loss)
    hist_loss_sae.append(total_loss_sae / n_batches)
    hist_loss_pcdm.append(total_loss_pcdm / n_batches)
    log_string = f"Epoch {epoch + 1:3d}/{n_epochs:3d} - loss: {epoch_loss:.5f}"
    log_string += f" | loss_sae: {total_loss_sae / n_batches:.5f} loss_pcdm: {total_loss_pcdm / n_batches:.5f}"

    # Store the optimal model
    if best_loss > epoch_loss:
      log_string += " --> Best model ever"
      best_loss = epoch_loss
      if store_model:
        state_dict_path, full_model_path = _checkpoint_paths(store_name)
        torch.save(pgm.state_dict(), state_dict_path)
        torch.save(pgm, full_model_path)
        log_string += " (stored)"
    print(log_string)

  # Plot the loss
  plt.figure(figsize=(4, 1.5))
  plt.plot(hist_loss, color='red', label='l_total')
  plt.plot(hist_loss_sae, color='orange', label='l_sae')
  plt.plot(hist_loss_pcdm, color='blue', label='l_pcdm')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(loc='upper right')









