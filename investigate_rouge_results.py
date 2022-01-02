# Results investigation
import numpy as np

#Extractive mt5 test results
test_results = np.load('./mT5_results/mt530-122947_ex_prefix_test.npy', allow_pickle = True)
test_results
d = test_results.flatten()
d[0]

#Abstractive mt5 test results
test_results = np.load('./mT5_results/mt530-144605_abs_prefix_test.npy', allow_pickle = True)
test_results
d = test_results.flatten()
d[0]

#######
#Random mt5 test results
test_results = np.load('./mT5_results/mt530-160149_prefix_test.npy', allow_pickle = True)
test_results
d = test_results.flatten()
d[0]


#######
#Mixed mt5 test results
test_results = np.load('./mT5_results/mt530-133709_mix_prefix_test.npy', allow_pickle = True)
test_results
d = test_results.flatten()
d[0]

