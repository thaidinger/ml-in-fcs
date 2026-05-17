#############################################
# Important Note                            #
# Due to the inherentrandomness, suggest to #
# train and select the modules separately   #
#############################################



from utils.load_data import *
from models.model_params import *
from models.pattern_recognition_module import *
from models.train_ftsdiffusion import *



def train_all():
  dataname = prm_params['dataname']
  # Get the historical time series (S&P 500 as example)
  get_fts(ticker='^GSPC', fts_name=dataname, start_date='1980-01-01',  end_date='2020-01-01')
  fts = load_actual_fts(dataname).squeeze()
  # Train the modules
  train_ftsdiffusion(fts)









