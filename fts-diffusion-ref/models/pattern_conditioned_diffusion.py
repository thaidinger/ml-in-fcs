import numpy as np
import math
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
from tqdm import tqdm



#########################################
# Pattern-conditioned diffusion network #
#########################################

class PCDM(nn.Module):
  """
  Pattern-conditioned diffusion model
  Args
    network: nerural network for denoising process
    n_steps: diffusion steps
    min_beta, max_beta: range of pre-scheduled beta
    device: device
  """
  def __init__(self, network,
               n_steps=100, 
               min_beta=1e-4, max_beta=0.02,
               device=None):
    super().__init__()
    self.network = network.to(device)
    self.n_steps = n_steps
    self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
    self.alphas = 1. - self.betas
    self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    self.device = device

  def forward(self, x_0, t, eta=None, p=None):
    """
    Forward (diffusion) process
    Input
      x_0: original data
      t: current diffusion step
      eta: noise to be added
      p: reference pattern
    Output
      x_t: diffused data at diffusion step t
    """
    n = len(x_0)
    a_bar = self.alpha_bars[t]
    if eta is None:
      eta = torch.randn_like(x_0).to(self.device)
    x_t = a_bar.sqrt().reshape(n, 1, 1) * x_0 + (1 - a_bar).sqrt().reshape(n, 1, 1) * eta
    return x_t

  def backward(self, x, t, p):
    """
    Backward (denoising) process
    Input
      x: diffused data at diffusion step t
      t: current diffusion step
      p: reference pattern
    Output
      epsilon: predicted noise gradient at diffusion step t
    """
    return self.network(x, t, p)



class TimestepEmbedding(nn.Module):
  """
  Embedding of timesteps using sinusoidal positional embedding
  Args
    max_time_steps: number of diffusion steps
    time_embed_dim: time embedding dimension (as series length)
  Input
    t: tensor of time
  Output
    emb: tensor of time embedding, shape (n, dim)
  """
  def __init__(self, n_timesteps, time_embed_dim, time_hidden_dim):
    super().__init__()
    self.n = n_timesteps
    self.embed_dim = time_embed_dim
    self.hidden_dim = time_hidden_dim
    self.fc1 = nn.Linear(time_embed_dim, time_hidden_dim)
    self.fc2 = nn.Linear(time_hidden_dim, time_hidden_dim)
    self.activation = nn.ReLU()

  def _sinusoidal_embedding(self, t):
    t = t.unsqueeze(1)
    div_term = torch.exp(torch.arange(0.0, self.embed_dim, 2.0) * -(math.log(10000.0) / self.embed_dim)).to(t.device)
    pe = torch.zeros(t.size(0), self.embed_dim).to(t.device)
    pe[:, 0::2] = torch.sin(t * div_term)
    pe[:, 1::2] = torch.cos(t * div_term)
    return pe

  def forward(self, t):
    sinusoidal_emb = self._sinusoidal_embedding(t)
    t_emb = self.fc1(sinusoidal_emb)
    t_emb = self.activation(t_emb)
    t_emb = self.fc2(t_emb)
    return t_emb



class TemporalBlock(nn.Module):
  """
    Temporal block for TCN
  """
  def __init__(self,
               in_channels, out_channels,
               kernel_size, stride, padding, dilation,
               dropout=0.2):

    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.conv1 = nn.Conv1d(in_channels, out_channels,
                           kernel_size, stride=stride, padding=padding, dilation=dilation)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)
    self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    self.act_fn = nn.ReLU()

  def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01).float()
        self.conv2.weight.data.normal_(0, 0.01).float()
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01).float()

  def forward(self, x):
    out = self.conv1(x)
    out = self.act_fn(out)
    out = self.dropout(out)
    out = self.conv2(out)
    out = self.act_fn(out)
    res = x if self.downsample is None else self.downsample(x)
    return self.act_fn(out + res)



class TCN(nn.Module):
  """
    Temporal convolution network
    Args
      n_timesteps: number of diffusion steps
      series_len: length of series
      channels: hidden channels of each layer in TCN
      kernel_size: kernel size of convolution
      dropout: p in Dropout
      time_embed_dim: embedding dimension in timestep embedding
      time_hidden_dim: hidden dimension in timestep embedding
    Input
      x_t: noisy x at diffusion step t
      t: diffusion step t
      p: reference pattern
    Output
      x_t_minus_1: removing noise from diffusion step t to t-1
  """
  def __init__(self,
               n_timesteps,
               series_len,
               channels,
               kernel_size=3,
               dropout=0.2,
               input_dim=1,
               time_embed_dim=32, time_hidden_dim=32,
               device=None):
    super().__init__()
    self.n_timesteps = n_timesteps
    self.series_len = series_len
    self.input_dim = input_dim
    self.time_embed_dim = time_embed_dim
    self.time_hidden_dim = time_hidden_dim
    self.n_layers = len(channels)
    self.channels = channels
    self.kernel_size = 3
    self.time_embedding = TimestepEmbedding(n_timesteps, time_embed_dim, time_hidden_dim).to(device)
    self.layers = []

    for i in range(self.n_layers):
      dilation_size = 2 ** i
      padding = (kernel_size - 1) * dilation_size // 2
      in_channels = time_hidden_dim + input_dim + 1 if i==0 else self.channels[i-1]
      out_channels = self.channels[i]
      block = TemporalBlock(in_channels, out_channels,
                            kernel_size, stride=1, dilation=dilation_size, padding=padding,
                            dropout=dropout)
      block.init_weights()
      self.layers += [block]
    self.layers += [nn.Conv1d(channels[-1], 1, 1)]
    self.tcn = nn.Sequential(*self.layers).to(device)

  def forward(self, x_t, t, p):
    t_emb = self.time_embedding(t).unsqueeze(-1).expand(-1, -1, self.series_len)
    x = torch.cat((x_t, t_emb, p), dim=1)
    x_t_minus_1 = self.tcn(x)
    return x_t_minus_1



class TCN2(nn.Module):
  """
    Temporal convolution network (2nd version: z^i - p, empirically learn faster)
    Args
      n_timesteps: number of diffusion steps
      series_len: length of series
      channels: hidden channels of each layer in TCN2
      kernel_size: kernel size of convolution
      dropout: p in Dropout
      time_embed_dim: embedding dimension in timestep embedding
      time_hidden_dim: hidden dimension in timestep embedding
    Input
      x_t: noisy x at diffusion step t
      t: diffusion step t
      p: reference pattern
    Output
      x_t_minus_1: removing noise from diffusion step t to t-1
  """
  def __init__(self,
               n_timesteps,
               series_len,
               channels,
               kernel_size=3,
               dropout=0.2,
               input_dim=1,
               time_embed_dim=32, time_hidden_dim=32,
               device=None):
    super().__init__()
    self.device = device
    self.n_timesteps = n_timesteps
    self.series_len = series_len
    self.input_dim = input_dim
    self.time_embed_dim = time_embed_dim
    self.time_hidden_dim = time_hidden_dim
    self.n_layers = len(channels)
    self.channels = channels
    self.kernel_size = 3
    self.time_embedding = TimestepEmbedding(n_timesteps, time_embed_dim, time_hidden_dim).to(device)
    self.layers = []
    
    for i in range(self.n_layers):
      dilation_size = 2 ** i
      padding = (kernel_size - 1) * dilation_size // 2
      in_channels = time_hidden_dim + input_dim + 1 if i==0 else self.channels[i-1]
      out_channels = self.channels[i]
      block = TemporalBlock(in_channels, out_channels,
                            kernel_size, stride=1, dilation=dilation_size, padding=padding,
                            dropout=dropout)
      block.init_weights()
      self.layers += [block]
    self.layers += [nn.Conv1d(channels[-1], 1, 1)]
    self.tcn = nn.Sequential(*self.layers).to(device)

  def forward(self, x_t, t, p):
    x_t, p = x_t.to(self.device), p.to(self.device)
    x_t = x_t - p
    t_emb = self.time_embedding(t).unsqueeze(-1).expand(-1, -1, self.series_len)
    x = torch.cat((x_t, t_emb, p), dim=1)
    x_t_minus_1 = self.tcn(x)
    return x_t_minus_1









