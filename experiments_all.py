from experiments.qual_analysis import *
from experiments.quant_analysis import *
from experiments.downstream import *



def run_experiments(store_res=True):
  print("[Experiments] Starting qualitative analysis...", flush=True)
  # Qualitative analysis of stylized facts
  qualitative_analysis(store_res=store_res)
  print("[Experiments] Qualitative analysis complete.", flush=True)

  print("[Experiments] Starting quantitative analysis...", flush=True)
  # Quantitative analysis of distribution tests
  quantitative_analysis()
  print("[Experiments] Quantitative analysis complete.", flush=True)

  print("[Experiments] Starting downstream experiments (TMTR/TATR)...", flush=True)
  # Downstream experiments of TMTR and TATR
  downstream_experiments(ahead=1, store_fig=store_res)
  print("[Experiments] Downstream experiments complete.", flush=True)









