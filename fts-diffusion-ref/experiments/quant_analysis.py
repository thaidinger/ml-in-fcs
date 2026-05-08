import numpy as np
import random
import torch
from experiments.metrics_quant import *
from models.model_params import prm_params
from utils.load_data import *

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



#########################
# Quantitative analysis #
#########################

def quantitative_analysis():
  """ Quantitative analysis in experiments """
  dataname = prm_params['dataname']
  patterns, segments, labels, lengths = load_segments(dataname, prm_params)
  threshold = find_threshold(segments, 252)
  distribution_tests(20, patterns, segments, labels, lengths, threshold)









