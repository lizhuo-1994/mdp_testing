import scipy.stats as stats
import os, json
import numpy as np

with open('results/mdpfuzz_metrics.json') as f:
    data = json.load(f)

data = np.array(data)

density = data[:, 0] 
norm_density = data[:, 2]  
semi_density = data[:, 4] 
norm_semi_density = data[:, 6] 
norm_novelty = data[:, 8]
failures = data[:, 10]

def smooth(data_list):

    smoothed = []
    data_sum = 0
    for i in range(len(data_list)):
        data_sum = data_sum + data_list[i]
        if i % 10 == 0 and i > 0:
            data_sum = data_sum / 10
            smoothed.append(data_sum)
            data_sum = 0

    return smoothed


density = smooth(density)
norm_density = smooth(norm_density) 
semi_density = smooth(semi_density) 
norm_semi_density = smooth(norm_semi_density)
norm_novelty = smooth(norm_novelty)
failures = smooth(failures)

tau, p_value = stats.kendalltau(density, failures)
print('density \t tau:', tau, '\t p-value:', p_value)

tau, p_value = stats.kendalltau(norm_density, failures)
print('norm_density \t tau:', tau, '\t p-value:', p_value)

tau, p_value = stats.kendalltau(semi_density, failures)
print('semi_density \t tau:', tau, '\t p-value:', p_value)

tau, p_value = stats.kendalltau(norm_semi_density, failures)
print('norm_semi_density \t tau:', tau, '\t p-value:', p_value)

tau, p_value = stats.kendalltau(norm_novelty, failures)
print('norm_novelty \t tau:', tau, '\t p-value:', p_value)