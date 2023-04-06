import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal
import copy
from scipy.spatial import distance

def compute_novelty(data_list):
    data_list = np.array(data_list)
    result_list = np.linalg.norm(data_list, axis=1)
    norm_novelties = normalize_data(result_list, np.min(result_list), np.max(result_list))
    return norm_novelties



def compute_sensitivity(case, cases_list, performance_list, episode_reward):
    if cases_list == []:
        return episode_reward

    distances = distance.cdist([case], cases_list, "cosine")[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index] 
    
    sensitivity = abs(performance_list[min_index] - episode_reward)
    # a larger sensitivity is the better
    return sensitivity



def normalize_data(data, lbound, ubound):
    if lbound == ubound:
        return 0
    norm_data = (data - lbound) / (ubound - lbound)
    norm_data = np.clip(norm_data,0,1)
    return norm_data

def case_clip(original_case):
    target_case = copy.deepcopy(original_case)
    for i in range(len(original_case)):
        if original_case[i] < 0.333:
            target_case[i] = 1
        elif original_case[i] < 0.667 and original_case[i] > 0.333:
            target_case[i] = 2
        else:
            target_case[i] = 3
    return target_case

## Define the NN architecture
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 512),
            nn.Linear(512, output_dim)
        )
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)
        out = self.activation(out)
        return out

class AbstractModel():
    def __init__(self):
        self.initial = []
        self.final = []

class Grid(AbstractModel):
    '''
    Multiple DTMCs from a set of sets of traces
    traces: a set of sets of traces
    '''
    def __init__(self, min_val, max_val, grid_num, clipped=True):
        super().__init__()
        self.min = min_val
        self.max = max_val
        self.k = grid_num
        self.dim = max_val.shape[0]
        self.total_states = pow(grid_num,self.dim)
        self.unit = (max_val - min_val) / self.k
        self.clipped = clipped
        
    def state_abstract(self, con_states):
        con_states = con_states
        lower_bound = self.min
        upper_bound = self.max
        unit = (upper_bound - lower_bound)/self.k
        abs_states = np.zeros(con_states.shape[0],dtype=np.int8)
        
        #print(lower_bound)
        #print(upper_bound)
        indixes = np.where(unit == 0)[0]
        unit[indixes] = 1
        #print('unit:\t', unit)
        
        tmp = ((con_states-self.min)/unit).astype(int)
        if self.clipped:
            tmp = np.clip(tmp, 0, self.k-1)
            
        dims = tmp.shape[1]
        for i in range(dims):
            abs_states = abs_states + tmp[:,i]*pow(self.k, i)
#         abs_states = np.expand_dims(abs_states,axis=-1)
        abs_states = [str(item) for item in abs_states]
        return abs_states
    
    def extract_abs_trace(self,dones,abs_states,abs_rewards, abs_values):

        end_idx = np.where(np.abs(dones)==1)[0]
        all_traces = []
        all_rewards = []
        all_values = []

        start = 0
        for cur_end in end_idx:
            all_rewards.append(abs_rewards[start : cur_end + 1])
            all_values.append(abs_values[start : cur_end + 1])
            all_traces.append(abs_states[start : cur_end + 1])
            start = cur_end + 1
           
        return all_traces,all_rewards,all_values 

def calculate_init_bounds(x2, y2):
    bound1 = np.arcsin(-y2/np.linalg.norm([x2, y2]))
    if x2 > 0:
        bound1 = np.pi - bound1
    if bound1 > np.pi / 2:
        return np.pi / 2, bound1
    else:
        return bound1, np.pi / 2

def map_test_case(acas_speed, row, theta, auto_theta): 
    acas_speed     = acas_speed * (1100-10) + 10
    row            = row * (60261-1000) + 1000
    theta          = theta * 2 * np.pi - np.pi
    x2             = row * np.cos(theta)
    y2             = row * np.sin(theta)
    bound1, bound2 = calculate_init_bounds(x2, y2)
    auto_theta     = auto_theta * (bound2-bound1) + bound1

    return acas_speed, x2, y2, auto_theta

class Memory:

    def __init__(self, size = 100):
        self.ptr = 0
        self.size = size
        self.count = 0
        self.max_density = 0
        self.min_density = 0
        self.max_novelty = 1
        self.min_novelty = 0
        self.max_sensitivity = 0
        self.min_sensitivity = 0
        self.max_performance = 0
        self.min_performance = 0
        self.case_list = [None for i in range(self.size)]
        self.density_list = [0 for i in range(self.size)]
        self.novelty_list = [0 for i in range(self.size)]
        self.sensitivity_list = [0 for i in range(self.size)]
        self.performance_list = [0 for i in range(self.size)]

    def append(self, case, density, sensitivity, performance, novelty):

        self.max_density = max(self.max_density, density)
        self.min_density = min(self.min_density, density)
        self.max_novelty = max(self.max_novelty, novelty)
        self.min_novelty = min(self.min_novelty, novelty)
        self.max_sensitivity = max(self.max_sensitivity, sensitivity)
        self.min_sensitivity = min(self.min_sensitivity, sensitivity)
        self.max_performance = max(self.max_performance, performance)
        self.min_performance = min(self.min_performance, performance)

        self.case_list[self.ptr] = case
        self.density_list[self.ptr]  = density
        self.sensitivity_list[self.ptr]  = sensitivity
        self.performance_list[self.ptr]  = performance
        self.ptr += 1
        self.count += 1
        if self.ptr == self.size:
            self.ptr = 0

    def get_index(self):
        if self.count < self.size:
            return self.count
        else:
            return self.size

    def clear(self):
        self.ptr = 0
        self.count = 0
        self.case_list = [None for i in range(self.size)]
        self.density_list = [0 for i in range(self.size)]
        self.novelty_list = [0 for i in range(self.size)]
        self.sensitivity_list = [0 for i in range(self.size)]
        self.performance_list = [0 for i in range(self.size)]

    def get_cases(self):
        return self.case_list[0: self.get_index()]

    def get_densities(self):
        return self.density_list[0: self.get_index()]

    def get_novelties(self):
        return self.novelty_list[0: self.get_index()]

    def get_sensitivities(self):
        return self.sensitivity_list[0: self.get_index()]

    def get_performances(self):
        return self.performance_list[0: self.get_index()]

    def get_rewards(self):
        return self.reward_list[0: self.get_index()]

class Carla_ENV:
    def __init__(self):
        self.start_pose = None
        self.target_pose = None
        self.vehicles = None
        self.weather = None
        self.min = -1
        self.max = 1
        self.start_scope = 101
        self.yaw_scope = 5
        self.weather_scope = 13
        self.target_scope = 101


    def from_vector(self, vector_info):
        vector_info = np.clip(vector_info, self.min, self.max)
        self.start_pose = int(((vector_info[0] -  self.min) /  (self.max -  self.min)) * self.start_scope)
        self.start_pose = np.clip(self.start_pose, 0, self.start_scope - 1)

        if self.start_pose in [39,41,42,48,68,79]:
            self.start_pose = 1

        self.start_pose_x = vector_info[1]
        self.start_pose_y = vector_info[2]
        self.start_pose_yaw = vector_info[3] * self.yaw_scope

        self.target_pose = int(((vector_info[4] -  self.min) /  (self.max -  self.min)) * self.target_scope)
        self.target_pose = np.clip(self.target_pose, 0, self.target_scope - 1)

        self.weather = int(((vector_info[5] -  self.min) /  (self.max -  self.min)) * self.weather_scope)
        
        self.vehicles = []
        for i in range(self.target_scope - 1):
            v_x = vector_info[6 + i * 2]
            v_y = vector_info[7 + i * 2] 
            self.vehicles.append((v_x, v_y))

class Density:

    def __init__(self):
        self.GMM = None
        self.GMMupdate = None
        self.GMMK = 10
        self.GMM_cond = None
        self.GMMupdate_cond = None
        self.GMMK_cond = 10
        self.GMMthreshold = 0.1

    def flatten_states(self, states):
        # TODO: flatten the states here
        states = np.array(states)
        states_cond = np.zeros((states.shape[0]-1, states.shape[1] * 2))
        for i in range(states.shape[0]-1):
            states_cond[i] = np.hstack((states[i], states[i + 1]))

        return states, states_cond

    def GMMinit(self, data_corpus, data_corpus_cond):
        res = []
        for i in range(self.GMMK):
            temp = dict()
            temp[0] = 1 / self.GMMK
            temp[1] = temp[0] * np.mean(data_corpus[i:i+15], axis=0)
            # temp[2] = covariances[i]
            temp[2] = np.zeros((data_corpus.shape[1], data_corpus.shape[1]))
            for j in range(i, i+15):
                temp[2] += temp[0] * np.matmul(data_corpus[j: j+1].T, data_corpus[j: j+1])
            temp[2] /= 15
            res.append(temp)

        weights = np.zeros(self.GMMK)
        means = np.zeros((self.GMMK, data_corpus.shape[1]))
        covariances = np.zeros((self.GMMK, data_corpus.shape[1], data_corpus.shape[1]))
        for i in range(self.GMMK):
            weights[i] = res[i][0]
            means[i] = res[i][1] / res[i][0]
            covariances[i] = np.eye(data_corpus.shape[1])

        self.GMM = dict()
        self.GMM['means'] = copy.deepcopy(means)
        self.GMM['weights'] = copy.deepcopy(weights)
        self.GMM['covariances'] = copy.deepcopy(covariances)

        res_cond = []
        for i in range(self.GMMK_cond):
            temp_cond = dict()
            temp_cond[0] = 1 / self.GMMK_cond
            temp_cond[1] = temp_cond[0] * np.mean(data_corpus_cond[i:i+15], axis=0)
            # temp[2] = covariances[i]
            temp_cond[2] = np.zeros((data_corpus_cond.shape[1], data_corpus_cond.shape[1]))
            for j in range(i, i+15):
                temp_cond[2] += temp_cond[0] * np.matmul(data_corpus_cond[j: j+1].T, data_corpus_cond[j: j+1])
            temp_cond[2] /= 15
            res_cond.append(temp_cond)

        weights_cond = np.zeros(self.GMMK_cond)
        means_cond = np.zeros((self.GMMK_cond, data_corpus_cond.shape[1]))
        covariances_cond = np.zeros((self.GMMK_cond, data_corpus_cond.shape[1], data_corpus_cond.shape[1]))
        for i in range(self.GMMK_cond):
            weights_cond[i] = res_cond[i][0]
            means_cond[i] = res_cond[i][1] / res_cond[i][0]
            covariances_cond[i] = np.eye(data_corpus_cond.shape[1])

        self.GMM_cond = dict()
        self.GMM_cond['means'] = copy.deepcopy(means_cond)
        self.GMM_cond['weights'] = copy.deepcopy(weights_cond)
        self.GMM_cond['covariances'] = copy.deepcopy(covariances_cond)

        return res, res_cond

    def get_mdp_pdf(self, states_seq, states_seq_cond):
        first_frame = states_seq[0:1]
        GMMpdf = np.zeros(self.GMMK)
        for k in range(self.GMMK):
            # print(k, np.linalg.det(self.GMM['covariances'][k]))
            GMMpdf[k] = self.GMM['weights'][k] * multivariate_normal.pdf(first_frame, self.GMM['means'][k], self.GMM['covariances'][k])
        GMMpdf += 1e-5
        GMMpdfvalue = np.sum(GMMpdf)
        first_frame_pdf = GMMpdf

        single_frame_pdf = np.zeros((states_seq.shape[0], self.GMMK))
        other_frame_pdf = np.zeros((states_seq_cond.shape[0], self.GMMK_cond))

        for i in range(states_seq.shape[0]):
            for k in range(self.GMMK):
                single_frame_pdf[i, k] = self.GMM['weights'][k] * multivariate_normal.pdf(states_seq[i], self.GMM['means'][k], self.GMM['covariances'][k])
        single_frame_pdf += 1e-5

        for i in range(states_seq_cond.shape[0]):
            for k in range(self.GMMK_cond):
                other_frame_pdf[i, k] = self.GMM_cond['weights'][k] * multivariate_normal.pdf(states_seq_cond[i], self.GMM_cond['means'][k], self.GMM_cond['covariances'][k])
            other_frame_pdf[i] += 1e-5
            # HACK: p(x1 | x0) = p(x1, x0) / p(x0)
            GMMpdfvalue *= np.min([np.sum(other_frame_pdf[i]) / np.sum(single_frame_pdf[i]), 1.0])
        return GMMpdfvalue, GMMpdf, other_frame_pdf

    def state_coverage(self, states_seq):
        states_seq, states_seq_cond = self.flatten_states(states_seq)
        # if len(self.corpus) < 20:
        #     return 0
        # el
        if self.GMM == None:
            # TODO: initialize
            GMMresult, GMMresult_cond = self.GMMinit(states_seq, states_seq_cond)
            self.GMMupdate = dict()
            self.GMMupdate_cond = dict()
            self.GMMupdate['iter'] = 10
            self.GMMupdate['threshold'] = 0.05
            self.GMMupdate['S'] = copy.deepcopy(GMMresult)
            self.GMMupdate_cond['S'] = copy.deepcopy(GMMresult_cond)

        GMMpdfvalue, GMMpdf, other_frame_pdf = self.get_mdp_pdf(states_seq, states_seq_cond)
        # GMMpdf = np.zeros(self.GMMK)
        # for k in range(self.GMMK):
        #     GMMpdf[k] = self.GMM['weights'][k] * multivariate_normal.pdf(states_seq, self.GMM['means'][k], self.GMM['covariances'][k])
        # GMMpdfvalue = np.sum(GMMpdf)

        first_frame = states_seq[0:1, :]
        
        # HACK: modified here
        # GMMpdfvalue = np.sum(GMMpdf)

        if GMMpdfvalue < self.GMMthreshold:
            # TODO: single update here
            gamma = 1.0 / (self.GMMupdate['iter'])
            GMMpdf /= np.sum(GMMpdf)
            new_S = copy.deepcopy(self.GMMupdate['S'])

            for i in range(self.GMMK):
                new_S[i][0] = self.GMMupdate['S'][i][0] + gamma * (GMMpdf[i] - self.GMMupdate['S'][i][0])
                new_S[i][1] = self.GMMupdate['S'][i][1] + gamma * (GMMpdf[i]*first_frame - self.GMMupdate['S'][i][1])
                new_S[i][2] = self.GMMupdate['S'][i][2] + gamma * (GMMpdf[i]*np.matmul(first_frame.T, first_frame) - self.GMMupdate['S'][i][2])

            self.GMMupdate['S'] = copy.deepcopy(new_S)

            for i in range(self.GMMK):
                self.GMM['weights'][i] = new_S[i][0]
                self.GMM['means'][i] = new_S[i][1] / new_S[i][0]
                self.GMM['covariances'][i] = (new_S[i][2] - np.matmul(self.GMM['means'][i].reshape(1, -1).T, new_S[i][1])) / new_S[i][0]
                W, V = np.linalg.eigh(self.GMM['covariances'][i])
                W = np.maximum(W, 1e-3)
                D = np.diag(W)
                reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
                self.GMM['covariances'][i] = copy.deepcopy(reconstruction)
                # print(i, np.linalg.det(self.GMM['covariances'][i]))

            # TODO: cond updates here
            cond_choices = np.argsort(np.sum(other_frame_pdf, axis=1))
            for cond_index in cond_choices[:cond_choices.shape[0] // 10]:
                GMMpdf_cond = other_frame_pdf[cond_index]
                GMMpdf_cond /= np.sum(GMMpdf_cond)
                current_frame = states_seq_cond[cond_index: cond_index + 1, :]
                new_S_cond = copy.deepcopy(self.GMMupdate_cond['S'])

                for i in range(self.GMMK_cond):
                    new_S_cond[i][0] = self.GMMupdate_cond['S'][i][0] + gamma * (GMMpdf_cond[i] - self.GMMupdate_cond['S'][i][0])
                    new_S_cond[i][1] = self.GMMupdate_cond['S'][i][1] + gamma * (GMMpdf_cond[i]*current_frame - self.GMMupdate_cond['S'][i][1])
                    new_S_cond[i][2] = self.GMMupdate_cond['S'][i][2] + gamma * (GMMpdf_cond[i]*np.matmul(current_frame.T, current_frame) - self.GMMupdate_cond['S'][i][2])

                self.GMMupdate_cond['S'] = copy.deepcopy(new_S_cond)

                for i in range(self.GMMK_cond):
                    self.GMM_cond['weights'][i] = new_S_cond[i][0]
                    self.GMM_cond['means'][i] = new_S_cond[i][1] / new_S_cond[i][0]
                    self.GMM_cond['covariances'][i] = (new_S_cond[i][2] - np.matmul(self.GMM_cond['means'][i].reshape(1, -1).T, new_S_cond[i][1])) / new_S_cond[i][0]
                    W, V = np.linalg.eigh(self.GMM_cond['covariances'][i])
                    W = np.maximum(W, 1e-3)
                    D = np.diag(W)
                    reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
                    self.GMM_cond['covariances'][i] = copy.deepcopy(reconstruction)

        return GMMpdfvalue