import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.load_models import *
from models.utils_sampling import *

data_path = 'data/'

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



##############################
# Load stored synthetic data #
##############################

def load_syn_ts(dataname, modelname, T):
  """ Load the stored synthetic time series """
  syn_ts = pd.read_csv(data_path + f"syn_{dataname}_{modelname}_{T}.csv").values[:,1]
  return syn_ts


def load_actual_ts(dataname, T):
  """ Load the stored actual time series """
  actual_ts = pd.read_csv(data_path + f"syn_{dataname}_actual_{T}.csv").values[:,1]
  return actual_ts









