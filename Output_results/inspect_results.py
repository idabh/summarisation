import numpy as np

met = np.load("/work/summarisation/Output_results/mt5_metrics-50k.npy", allow_pickle=True)
met

res = np.load("/work/summarisation/Output_results/mt5_results-50k.npy", allow_pickle=True)
res[:20]

rou = np.load("/work/summarisation/Output_results/mt5_rouge-50k.npy", allow_pickle=True)
rou
