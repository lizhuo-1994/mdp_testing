if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTester
import tensorflow.contrib.layers as layers

import tqdm, sys
import os,json, random

from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid
from diffusion import Diffusion

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='spread', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    ######################## parameters for generative testing ############################################
    parser.add_argument("--method", help="select the guidance for testing", default="generative", type=str, required=False)
    parser.add_argument("--hour", help="test time", default=1, type=int)
    parser.add_argument("--step", help="number of normal cases at each training step", default=50, type=int)
    parser.add_argument("--grid", help="state abstraction granularity", default=2, type=int)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag, verify_func=scenario.verify)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTester
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def get_observe(env):
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
        state.append(agent.state.p_vel)
        state.append(agent.state.c)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return list(np.array(state).flatten())

def get_init_state(env):
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return state

def get_collision_num(env):
    collisions = 0
    for i, agent in enumerate(env.world.agents):
        for j, agent_other in enumerate(env.world.agents):
            if i == j:
                continue
            delta_pos = agent.state.p_pos - agent_other.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = (agent.size + agent_other.size)
            if dist < dist_min:
                collisions += 1
    return collisions / 2



def test(arglist):
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        U.load_state(arglist.load_dir)

        episode_rewards = 0.0  # sum of rewards for all agents

        sequence = []
        collisions = 0
        init_state = get_init_state(env)
        episode_step = 0
        train_step = 0


        case_dimension = 12
        diffusion_model = Diffusion(batch_size = 1, epoch = 1, data_size = case_dimension, training_step_per_spoch = 25, num_diffusion_step = 25)
        diffusion_model.setup()
        memory_model = Memory(size = 100)
        density_model = Density()


        min_obs = np.array([-2 for i in range(24)])
        max_obs = np.array([ 2 for i in range(24)])
        novelty_grid = Grid(min_obs, max_obs, arglist.grid)
        novelty_dict = dict()

        sensitivity_list = []
        performance_list = []
        novelty_list = []
        normal_case_list = []
        metric_list = []
        diffusion_failure_list = []
        random_failure_list = []
        diffusion_failure_count = []
        random_failure_count = []
        start_time = time.time()
        current_time = time.time()
        cur_step = 0
        failure_by_diffusion = 0

        information_list = []
        diffusion_failure_clusters= []

        # HACK: set time here
        while current_time - start_time < 3600 * arglist.hour:

            current_time = time.time()
            if cur_step > 0 and cur_step % arglist.step == 0:
                normal_case_list = np.array(normal_case_list)
                metric_list      = np.array(metric_list)

                ######################################################## add guidance to loss here #################################################
                if arglist.method == 'generative':
                    metrics = None
                elif arglist.method == 'generative+density':
                    metrics = metric_list[:, [0]]
                elif arglist.method == 'generative+sensitivity':
                    metrics = metric_list[:, [1]]
                elif arglist.method == 'generative+performance':
                    metrics = metric_list[:, [2]]
                elif arglist.method == 'generative+novelty':
                    metrics = metric_list[:, [3]]
                else:
                    print('Please check the method parameters!')
                    return

                diffusion_model.train(normal_case_list, metrics, arglist.method)
                normal_case_list = []
                metric_list = []
                memory_model.clear() 

                for _ in range(50):
                    failure_flag = False
                    test_case = diffusion_model.generate()
                    test_case = list(test_case.reshape(6,2))
                    obs_n = env.reset(test_case[0:3], test_case[3:]) 
                    agent_flag, landmark_flag = env.verify_func(env.world)
                    if agent_flag or landmark_flag:
                        continue
                    test_case = np.array(test_case).flatten().tolist()
                    episode_step = 0
                    collisions = 0
                    sequence = []
                    episode_rewards = 0
                    terminal = False
                    collisions = 0
                    done = False

                    while True:
                        # get action
                        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                        # environment step
                        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                        episode_step += 1
                        sequence.append(get_observe(env))
                        done = all(done_n)
                        terminal = (episode_step >= arglist.max_episode_len)
                        collisions += get_collision_num(env)
                        obs_n = new_obs_n
                        for i, rew in enumerate(rew_n):
                            episode_rewards += rew
                        if terminal and collisions > 5 and not done:
                            failure_flag = True
                            current_time = time.time()
                            regular_time = (current_time - start_time) / 3600
                            failure_by_diffusion += 1
                            print('Diffusion Failure case found:', [regular_time, failure_by_diffusion, test_case])
                            diffusion_failure_count.append([regular_time, failure_by_diffusion, test_case])
                            break
                        if terminal or done:
                            break
                    
                    ################################################## compare density and novelty ######################################

                    abstract_id = novelty_grid.state_abstract(np.array([sequence[-1]]))[0]
                    if abstract_id in novelty_dict.keys():
                        novelty_dict[abstract_id] += 1
                    else:
                        novelty_dict[abstract_id] = 1
                    novelty = novelty_dict[abstract_id]
                    # norm_novelty = normalize_data(novelty, memory_model.min_novelty, memory_model.max_novelty)
                    # norm_novelty = 1 - norm_novelty
                    # norm_novelty = novelty
                    norm_novelty = 1 / (math.e ** (novelty - 1))

                    ############################################# add to training #######################################################3
                    normal_case_list.append(test_case)
                    metric_list.append([0, 0, 0, norm_novelty])
                    memory_model.append(test_case, 0, 0, 0, novelty)

                    if failure_flag:
                        diffusion_failure_clusters.append(abstract_id)

                    print(failure_flag, abstract_id, len(novelty_dict.keys()), len(set(diffusion_failure_clusters)))
                    information_list.append([sequence[-1], failure_flag, abstract_id, norm_novelty])
            else:
                normal_case = np.random.uniform(-1,1,12)
                normal_case = list(normal_case.reshape(6,2))
                obs_n = env.reset(normal_case[0:3], normal_case[3:])
                normal_case = get_init_state(env)
                normal_case = np.array(normal_case).flatten()
                episode_rewards = 0
                collisions = 0
                episode_step = 0
                sequence = []
                terminal = False
                done = False
                while True:
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                    # environment step
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    episode_step += 1
                    sequence.append(get_observe(env))
                    done = all(done_n)
                    terminal = (episode_step >= arglist.max_episode_len)
                    collisions += get_collision_num(env)
                    obs_n = new_obs_n

                    for i, rew in enumerate(rew_n):
                        episode_rewards += rew

                    if terminal or done:
                        break


                ########################## Density, Sensitivity and other guidance ############################
                normal_case_list.append(normal_case)
                density, norm_density = 0, 0
                sensitivity, norm_sensitivity = 0, 0
                performance, norm_performance = 0, 0
                novelty, norm_novelty = 0, 0
                cases_list = memory_model.get_cases()

                if 'density' in arglist.method:
                    density_list = memory_model.get_densities()
                    density = density_model.state_coverage(sequence)
                    norm_density = normalize_data(density, memory_model.min_density, memory_model.max_density)
                
                if 'sensitivity' in arglist.method:
                    sensitivity_list = memory_model.get_sensitivities()
                    sensitivity = compute_sensitivity(normal_case, cases_list, performance_list, episode_reward)
                    norm_sensitivity = normalize_data(sensitivity, memory_model.min_sensitivity, memory_model.max_sensitivity)
                    norm_sensitivity = 1 - norm_sensitivity
                    metric = norm_sensitivity
                
                if 'performance' in arglist.method:
                    performance_list = memory_model.get_performances()
                    performance = episode_reward
                    norm_performance = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
                
                if 'novelty' in  arglist.method:
                    abstract_id = novelty_grid.state_abstract(np.array([sequence[-1]]))[0]
                    if abstract_id in novelty_dict.keys():
                        novelty_dict[abstract_id] += 1
                    else:
                        novelty_dict[abstract_id] = 1
                    novelty = novelty_dict[abstract_id]
                    # norm_novelty = normalize_data(novelty, memory_model.min_novelty, memory_model.max_novelty)
                    # norm_novelty = 1 - norm_novelty
                    # norm_novelty = novelty
                    norm_novelty = 1 / (math.e ** (novelty - 1))

                metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
                memory_model.append(normal_case, density, sensitivity, performance, novelty)
            cur_step += 1

    os.makedirs('results', exist_ok=True)
    with open('results/' + arglist.method + '_diffusion_failure_count.json', 'w') as f:
        json.dump(diffusion_failure_count, f)
        
    with open('results/' + arglist.method + '_information.json', 'w') as f:
        json.dump(information_list, f)

    with open('results/' + arglist.method + '_novelty_dict.json', 'w') as f:
        json.dump(novelty_dict, f)

if __name__ == '__main__':
    arglist = parse_args()
    test(arglist)
