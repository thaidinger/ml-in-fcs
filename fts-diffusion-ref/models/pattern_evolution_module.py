import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
torch.set_default_dtype(torch.float)
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.utils_evolution import *
from utils.markov_processing import *

model_path = 'trained_models/'


def _checkpoint_paths(store_name):
  if store_name.endswith('.pth'):
    base = store_name[:-4]
  elif store_name.endswith('.pt'):
    base = store_name[:-3]
  else:
    base = store_name
  return model_path + base + '.pth', model_path + base + '.pt'



#############################
# Pattern evolution network #
#############################

class PatternEvolutionModule(nn.Module):
  """
    Pattern evolution network
  """
  def __init__(self, n_patterns, range_length, embedding_dim, hidden_dim, device):
    super().__init__()
    self.n_patterns = n_patterns
    self.range_length = range_length
    self.embedding_p = nn.Embedding(num_embeddings=n_patterns, embedding_dim=embedding_dim)
    self.embedding_l = nn.Embedding(num_embeddings=range_length, embedding_dim=embedding_dim)
    self.fc1 = nn.Linear(embedding_dim * 2 + 1, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, n_patterns + range_length + 1)    # n_patterns for pattern types, 1 for length, 1 for magnitude
    self.device = device

  def forward(self, x):
    x = x.to(self.device)
    x_pattern = x[:, 0].long()
    x_length = x[:, 1].long()
    x_magnitude = x[:, 2].float().unsqueeze(1)
    x_pattern_embed = self.embedding_p(x_pattern)
    x_length_embed = self.embedding_l(x_length - 10)    # embedding starts from index 0
    x_concat = torch.cat((x_pattern_embed, x_length_embed, x_magnitude), dim=1)
    h = torch.relu(self.fc1(x_concat))
    h = torch.relu(self.fc2(h))
    y = self.fc3(h)
    return y



#######################################
# Train the pattern evolution network #
#######################################

def train_evolution_module(
    model,
    dataloader,
    n_epochs=100,
    lr=0.001,
    loss_weights=[1,1,1],
    store_model=False,
    store_name='pem.pt'):
  """ Training phase of the pattern evolution module """
  # Set the loss function and optimizer
  n_patterns = model.n_patterns
  range_length = model.range_length
  criterion_pattern = nn.CrossEntropyLoss()
  # criterion_scale = nn.MSELoss()
  criterion_scale = nn.L1Loss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  hist = []
  hist_p = []
  hist_l = []
  hist_m = []
  best_loss = float("inf")
  # Training
  for epoch in tqdm(range(n_epochs)):
    epoch_loss = 0.0
    epoch_loss_p = 0.0
    epoch_loss_l = 0.0
    epoch_loss_m = 0.0
    n_batches = 0
    for batch in dataloader:
      optimizer.zero_grad()
      batch = batch[0]
      batch_x, batch_y = batch[:, 0, :], batch[:, 1, :]
      target_pattern = batch_y[:, 0].long()
      target_length = (batch_y[:, 1] - 10).long()
      target_magnitude = batch_y[:, 2].float().view(-1, 1)
      # print(target_pattern.shape, target_length.shape, target_magnitude.shape)

      # Get model prediction
      pred_y = model(batch_x)
      pred_pattern = pred_y[:, :n_patterns]
      pred_length = pred_y[:, n_patterns:n_patterns+range_length]
      pred_magnitude = pred_y[:, n_patterns+range_length:].float().view(-1,1)
      # print(pred_pattern.shape, pred_length.shape, pred_magnitude.shape)

      # Compute loss
      loss_pattern = criterion_pattern(pred_pattern, target_pattern) * loss_weights[0]
      loss_length = criterion_pattern(pred_length, target_length) * loss_weights[1]
      loss_magnitude = criterion_scale(pred_magnitude, target_magnitude) * loss_weights[2]
      loss = loss_pattern + loss_length + loss_magnitude
      epoch_loss += loss.item()
      epoch_loss_p += loss_pattern.item()
      epoch_loss_l += loss_length.item()
      epoch_loss_m += loss_magnitude.item()
      n_batches += 1
      loss.backward()
      optimizer.step()
      # break

    epoch_loss = epoch_loss / n_batches
    epoch_loss_p = epoch_loss_p / n_batches
    epoch_loss_l = epoch_loss_l / n_batches
    epoch_loss_m = epoch_loss_m / n_batches
    hist.append(epoch_loss)
    hist_p.append(epoch_loss_p)
    hist_l.append(epoch_loss_l)
    hist_m.append(epoch_loss_m)

    log_string = f"Epoch {epoch + 1:3d}/{n_epochs:3d} - loss: {epoch_loss:.5f}"
    log_string += f" | loss_p: {epoch_loss_p:.5f}, loss_l: {epoch_loss_l:.5f}, loss_m: {epoch_loss_m:.5f}"
    if epoch_loss < best_loss:
      log_string += " --> Best model ever"
      best_loss = epoch_loss
      if store_model:
        state_dict_path, full_model_path = _checkpoint_paths(store_name)
        torch.save(model.state_dict(), state_dict_path)
        torch.save(model, full_model_path)
        log_string += " (stored)"
    print(log_string)

  # Plot training history
  fig, axs = plt.subplots(1, 4, figsize=(10,2))
  x_range = np.arange(n_epochs)
  axs[0].plot(x_range, hist, label='loss')
  axs[1].plot(x_range, hist_p, label='loss_p')
  axs[2].plot(x_range, hist_l, label='loss_l')
  axs[3].plot(x_range, hist_m, label='loss_m')
  plt.legend(loc='upper right')
  plt.tight_layout()









