import os,sys
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats
# envs = ['RL_BipedalWalker']
# methods = ['fuzz', 'generative+novelty_diffusion']
from umap import UMAP

titles = {
    'RL_BipedalWalker' : 'RL_BipedalWalker',
    'RL_CARLA' : 'RL_CARLA',
    'IL_CARLA' : 'IL_CARLA',
    'ACAS_Xu' : 'ACAS_Xu',
    'CoopNavi' : 'CoopNavi'
}

times = {
    'RL_BipedalWalker' : 2,
    'RL_CARLA' : 2,
    'IL_CARLA' : 2,
    'ACAS_Xu' : 1,
    'CoopNavi' : 1
}

data_points = {
    'RL_BipedalWalker' : 5,
    'RL_CARLA' : 5,
    'IL_CARLA' : 5,
    'ACAS_Xu' : 5,
    'CoopNavi' : 5
}

markers = {
    'fuzz' : 'o',
    'generative+novelty_diffusion' : '*',
    'generative+density_diffusion' : '^',
    'generative+sensitivity_diffusion' : 's',
    'generative+performance_diffusion' : '+',
    'generative_diffusion' : 'D',
}

labels = {
    'fuzz' : 'MDPFuzz',
    'generative+novelty_diffusion' : 'Novelty (Ours)',
    'generative+density_diffusion' : 'Desnity',
    'generative+sensitivity_diffusion' : 'Sensitivity',
    'generative+performance_diffusion' : 'Reward',
    'generative_diffusion' : 'No Guidance',
}

colors = {
    'fuzz' : 'red',
    'generative+novelty_diffusion' : 'orange',
    'generative+density_diffusion' : 'green',
    'generative+sensitivity_diffusion' : 'blue',
    'generative+performance_diffusion' : 'pink',
    'generative_diffusion' : 'grey',
}

markersizes = {
    'fuzz' : 5,
    'generative+novelty_diffusion' : 5,
    'generative+density_diffusion' : 5,
    'generative+sensitivity_diffusion' : 5,
    'generative+performance_diffusion' : 5,
    'generative_diffusion' : 5
}



with open('results/mdpfuzz_information.json' , 'r') as f:
    m_info = json.load(f)

with open('results/mdpfuzz_novelty_dict.json' , 'r') as f:
    m_novelty_dict = json.load(f)

m_states = []
m_failure_flag = []
m_failure_ids = []
m_abstract_ids = []
m_occurrences = []
m_novelty = []
m_failure_occurrences = []
m_failure_novelty = []

for item in m_info:
    # s = np.asarray(item[0])
    m_states.append(item[0])
    m_failure_flag.append(item[1])
    if item[1]:
        m_failure_ids.append(item[2])
    m_abstract_ids.append(item[2])
        
m_abstract_ids = list(set(m_abstract_ids))
m_failure_ids = list(set(m_failure_ids))
m_occurrences = list(m_novelty_dict.values())
m_max_occ = max(m_occurrences)
m_min_occ = min(m_occurrences)
m_novelty = (np.array(m_occurrences) - m_min_occ) / (m_max_occ - m_min_occ)
for id in m_failure_ids:
    m_failure_occurrences.append(m_novelty_dict[id])

m_failure_novelty = (np.array(m_failure_occurrences) - m_min_occ) / (m_max_occ - m_min_occ)
m_novelty = 1 - m_novelty
m_failure_novelty = 1 - m_failure_novelty

print(len(m_novelty_dict.keys()), len(m_failure_ids))
print(sum(m_occurrences), sum(m_failure_occurrences))

print(m_failure_ids)

m_states = np.array(m_states)
m_states = np.nan_to_num(m_states)
m_states= m_states.astype(np.float)
# m_s_states = PCA(n_components=2).fit_transform(m_states)
m_s_states  = TSNE(n_components=2,init='random', perplexity=10).fit_transform(m_states)
# m_s_states = UMAP(n_components=2, init='random', random_state=0).fit_transform(np.array(m_states ))
plt.scatter(m_s_states[:,0], m_s_states[:,1], label = 'Mdpfuzz', color = 'red', s = 20, marker = '*')
    

with open('results/generative+novelty_information.json' , 'r') as f:
    our_info = json.load(f)

with open('results/generative+novelty_novelty_dict.json' , 'r') as f:
    novelty_dict = json.load(f)

states = []
failure_flag = []
failure_ids = []
abstract_ids = []
occurrences = []
novelty = []
failure_occurrences = []
failure_novelty = []

for item in our_info:
    # s = np.asarray(item[0])
    states.append(item[0])
    failure_flag.append(item[1])
    if item[1]:
        failure_ids.append(item[2])
    abstract_ids.append(item[2])
        
abstract_ids = list(set(abstract_ids))
failure_ids = list(set(failure_ids))
occurrences = list(novelty_dict.values())
max_occ = max(occurrences)
min_occ = min(occurrences)
novelty = (np.array(occurrences) - min_occ) / (max_occ - min_occ)
for id in failure_ids:
    failure_occurrences.append(novelty_dict[id])

failure_novelty = (np.array(failure_occurrences) - min_occ) / (max_occ - min_occ)
novelty = 1 - novelty
failure_novelty = 1 - failure_novelty

print(len(novelty_dict.keys()), len(failure_ids))
print(sum(occurrences), sum(failure_occurrences))

print(failure_ids)
states = np.array(states)
states = np.nan_to_num(states)
states= states.astype(np.float)
# s_states = PCA(n_components=2).fit_transform(states)
s_states = TSNE(n_components=2,init='random', perplexity=10).fit_transform(states)
# s_states = UMAP(n_components=2, init='random', random_state=0, min_dist).fit_transform(np.array(states ))
plt.scatter(s_states[:,0], s_states[:,1], label = 'Ours', color = 'orange', s = 20, marker = '*')
    

plt.legend(fontsize=10)
plt.title('RL_BipedalWalker', fontsize=14)
plt.grid()
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plt.savefig('./IL_CARLA/carla_lbc/termination_IL_CARLA.png')

common = list(set(abstract_ids) & set(m_abstract_ids))
print(len(states), len(m_states), len(abstract_ids), len(m_abstract_ids), len(common))


common = list(set(failure_ids) & set(m_failure_ids))
print(len(failure_ids), len(m_failure_ids), len(common))

# sns.set(style="darkgrid") 
# # Make default density plot
# sns.kdeplot(occurrences, color='blue',clip=(0.0, max_occ))
# sns.kdeplot(failure_occurrences, color='red',clip=(0.0, max_occ))
# plt.title('density of occurrences')
# plt.show()


# sns.set(style="darkgrid") 
# # Make default density plot
# sns.kdeplot(novelty, color='blue', clip=(0.0, 1.0))
# sns.kdeplot(failure_novelty, color='red', clip=(0.0, 1.0))
# plt.title('density of novelty')
# plt.show()