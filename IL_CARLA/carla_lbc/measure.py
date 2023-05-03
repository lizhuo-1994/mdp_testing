import scipy.stats as stats
import os, json
import numpy as np

with open('results/measure_metrics.json') as f:
    data = json.load(f)

data = np.array(data)
data = np.nan_to_num(data)

density = data[:, 0] 
norm_density = data[:, 2]  
semi_density = data[:, 4] 
norm_semi_density = data[:, 6] 
norm_novelty = data[:, 8]
norm_novelty= [1 if i == 0 else i for i in norm_novelty]
norm_novelty = np.array(norm_novelty)
failures = data[:, 10]

def smooth(data_list):

    smoothed = []
    data_sum = 0
    for i in range(len(data_list)):
        data_sum = data_sum + data_list[i]
        if i % 12 == 0 and i > 0:
            data_sum = data_sum / 12 
            smoothed.append(data_sum)
            data_sum = 0

    return smoothed


# density = smooth(density)
# norm_density = smooth(norm_density) 
# semi_density = smooth(semi_density) 
# norm_semi_density = smooth(norm_semi_density)
# norm_novelty = smooth(norm_novelty)
# failures = smooth(failures)

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