import numpy as np

met = np.load("/work/summarisation/Output_results/mt5_metrics-50k.npy", allow_pickle=True)
met

res = np.load("/work/summarisation/Output_results/mt5_results-50k.npy", allow_pickle=True)
res[:20]

rou = np.load("/work/summarisation/Output_results/mt5_rouge-50k.npy", allow_pickle=True)
rou

#-----MT5 50K NROWS WITH 10 EPOCHS (run 16/12):
import numpy as np
met = np.load('mt5_50k_ep10_16_12_2021.npy', allow_pickle=True) #metrics
res = np.load('results_MT5_50k_ep10_16_12_2021.npy', allow_pickle=True) #results
rou = np.load('rouge_MT5_50k_ep10_16_12_2021.npy', allow_pickle=True) #rouge_output
#rouge: 0.54787443, 0.28235127, 0.35684455
res[:5]

