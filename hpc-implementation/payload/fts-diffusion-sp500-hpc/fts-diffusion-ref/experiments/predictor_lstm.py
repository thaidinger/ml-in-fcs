import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
# from tslearn.metrics import SoftDTWLossPyTorch as SoftDTWLoss
import matplotlib.pyplot as plt
from tqdm import tqdm



###############################
# Downstream LSTM-based model #
###############################

class LSTMPredictor(nn.Module):
  """ LSTM-based predictor """
  def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.n_layers = n_layers
    self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.device = device

  def forward(self, x):
    h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
    c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
    if x.dim() == 2:
      x = x.unsqueeze(1)
    lstm_out, _ = self.lstm(x, (h0.detach(), c0.detach()))
    last_step = lstm_out[:, -1, :]
    y = self.fc(last_step)
    return y



#############################################
# Separately train the downstream predictor #
#############################################

def separate_train_lstm_predictor(
    n_epochs,
    dataset, # dataloader,
    input_dim,
    hidden_dim,
    output_dim,
    n_layers,
  criterion,
  verbose=False):
  """ Separately train the LSTM-based predictor on the given dataset """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = LSTMPredictor(input_dim, hidden_dim, output_dim, n_layers, device).to(device)
  criterion = set_loss_fn(criterion)
  optimizer = optim.Adam(model.parameters(), lr=1e-2)
  for epoch in range(n_epochs):
    model.train()
    X, y = dataset[:, :-output_dim].unsqueeze(-1).to(device), dataset[:, -output_dim:].to(device)
    y_ = model(X)
    loss = criterion(y_, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if verbose:
      print(f"[LSTM] Epoch {epoch+1}/{n_epochs} loss={loss.item():.6f}", flush=True)
  return model


def set_loss_fn(criterion='mse'):
  """  """
  criterion = criterion.lower()
  if criterion == 'mse':
    return nn.MSELoss()
  elif criterion == 'mae':
    return nn.L1Loss()



##############################################
# Test the downstream predictor on real data #
##############################################

def test_on_real(model, dataset, scaler, criterion='mape', plot_fig=False):
  """ Test the trained downstream model on real test dataset """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  ahead = model.output_dim
  with torch.no_grad():
    X, y = dataset[:, :-ahead].unsqueeze(-1).to(device), dataset[:, -ahead:].to(device)
    y_ = model(X)
  reals = scaler.inverse_transform(y.detach().cpu().numpy())
  preds = scaler.inverse_transform(y_.detach().cpu().numpy())
  if plot_fig:
    plt.figure(figsize=(7, 2))
    plt.plot(reals, color='red')
    plt.plot(preds, color='blue')
  if criterion=='mae':
    return MAE(reals, preds)
  else:
    return MAPE(reals, preds)









