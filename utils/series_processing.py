import numpy as np
import torch
torch.set_default_dtype(torch.float)



#############################
# Convert prices to returns #
#############################

def prices2returns(prices, last_prices=None):
  """ Construct returns series from prices series """
  last_digit = prices[0] if last_prices==None else last_prices[-1]
  prices_series = torch.cat((last_digit.view(1), prices), dim=0)
  returns_series = (prices_series[1:] - prices_series[:-1]) / prices_series[:-1]
  return returns_series

def prices2returns_batch(batch_prices):
  """ Construct returns series from prices series (for a batch of tensors) """
  batch_returns = []
  batch_returns.append(prices2returns(batch_prices[0]))
  for i in range(1, len(batch_prices)):
    batch_returns.append(prices2returns(batch_prices[i], batch_prices[i-1]))
  return batch_returns

def prices2returns_batch_fixed_length(batch_prices):
  """ Construct returns series from prices series (for batch with fixed-length inputs) """
  batch_returns = torch.empty_like(batch_prices)
  batch_returns[0] = prices2returns(batch_prices[0])
  for i in range(1, len(batch_prices)):
    batch_returns[i] = prices2returns(batch_prices[i], batch_prices[i-1])
  return batch_returns

def prices2returns_batch_variable_length(batch_prices):
  """ Construct returns series from prices series (for batch with variable-length inputs) """
  batch_returns = []
  batch_returns.append(prices2returns(batch_prices[0]))
  for i in range(1, len(batch_prices)):
    batch_returns.append(prices2returns(batch_prices[i], batch_prices[i-1]))
  return batch_returns



#############################
# Convert returns to prices #
#############################

def returns2prices(returns, start=1, dtype='numpy'):
  """ Construct prices series from returns series """
  res = [start]
  for r in returns:
      res.append(res[-1]*(1+r))
  if dtype=='tensor':
    return torch.tensor(res[1:])
  else:
    return np.array(res[1:]).flatten()

def returns2prices_batch(batch_returns, start, dtype='numpy'):
  """ Construct prices series from returns series (for a batch of tensors) """
  batch_prices = torch.empty_like(batch_returns)
  batch_prices[0] = returns2prices(batch_returns[0], start, dtype=dtype)
  for i in range(1, len(batch_returns)):
    batch_prices[i] = returns2prices(batch_returns[i], batch_prices[i-1,-1], dtype=dtype)
  return batch_prices









