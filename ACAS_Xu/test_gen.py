if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.load_model import read_onnx, load_repair
import torch, time, pickle, copy, argparse, sys
import os, random, json


from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid, map_test_case, calculate_init_bounds
from diffusion import Diffusion

class ACASagent:
    def __init__(self, acas_speed):
        self.x = 0
        self.y = 0
        self.theta = np.pi / 2
        self.speed = acas_speed
        self.interval = 0.1
        self.model_1 = read_onnx(1, 2)
        # self.model_1 = load_repair(model_index=1)
        self.model_2 = read_onnx(2, 2)
        self.model_3 = read_onnx(3, 2)
        self.model_4 = read_onnx(4, 2)
        # self.model_4 = load_repair(model_index=4)
        self.model_5 = read_onnx(5, 2)
        # self.model_5 = load_repair(model_index=5)
        self.prev_action = 0
        self.current_active = None

    def step(self, action):
        if action == 1:
            self.theta = self.theta + 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 2:
            self.theta = self.theta - 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 3:
            self.theta = self.theta + 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 4:
            self.theta = self.theta - 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        self.x = self.x + self.speed * np.cos(self.theta) * self.interval
        self.y = self.y + self.speed * np.sin(self.theta) * self.interval

    def act(self, inputs):
        inputs = torch.Tensor(inputs)
        # action = np.random.randint(5)
        if self.prev_action == 0:
            model = self.model_1
        elif self.prev_action == 1:
            model = self.model_2
        elif self.prev_action == 2:
            model = self.model_3
        elif self.prev_action == 3:
            model = self.model_4
        elif self.prev_action == 4:
            model = self.model_5
        action, active = model(inputs)
        # action = model(inputs)
        self.current_active = [action.clone().detach().numpy(), active.clone().detach().numpy()]
        action = action.argmin()
        self.prev_action = action
        return action

    def act_proof(self, direction):
        return direction


class Autoagent:
    def __init__(self, x, y, auto_theta, speed=None):
        self.x = x
        self.y = y
        self.theta = auto_theta
        self.speed = speed
        self.interval = 0.1

    def step(self, action):
        if action == 1:
            self.theta = self.theta + 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 2:
            self.theta = self.theta - 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 3:
            self.theta = self.theta + 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 4:
            self.theta = self.theta - 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        self.x = self.x + self.speed * np.cos(self.theta) * self.interval
        self.y = self.y + self.speed * np.sin(self.theta) * self.interval
    
    def act(self):
        # action = np.random.randint(5)
        action = 0
        return action

class env:
    def __init__(self, acas_speed, x2, y2, auto_theta):
        self.ownship = ACASagent(acas_speed)
        self.inturder = Autoagent(x2, y2, auto_theta)
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        # if self.inturder.x - self.ownship.x < 0:
        #     self.alpha = np.pi - self.alpha
        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

        if x2 == 0:
            if y2 > 0:
                self.inturder.speed = self.ownship.speed / 2
            else:
                self.inturder.speed = np.min([self.ownship.speed * 2, 1600])
        elif self.ownship.theta == self.inturder.theta:
            self.inturder.speed = self.ownship.speed
        else:
            self.inturder.speed = self.ownship.speed * np.sin(self.alpha) / np.sin(self.alpha + self.ownship.theta - self.inturder.theta)

        if self.inturder.speed < 0:
            self.inturder.theta = self.inturder.theta + np.pi
            while self.inturder.theta > np.pi:
                self.inturder.theta -= 2 * np.pi
            self.inturder.speed = -self.inturder.speed
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed

    def update_params(self):
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed
        self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row)
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta

        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

    def step(self):
        acas_act = self.ownship.act([self.row, self.alpha, self.phi, self.Vown, self.Vint])
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()
        # time.sleep(0.1)

    def step_proof(self, direction):
        acas_act = self.ownship.act_proof(direction)
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()


def verify(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()

    air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air2.update_params()

    min_dis1 = air1.row
    min_dis2 = air2.row
    for j in range(100):
        air1.step_proof(3)
        if min_dis1 > air1.row:
            min_dis1 = air1.row
    for j in range(100):
        air2.step_proof(4)
        if min_dis2 > air2.row:
            min_dis2 = air2.row
    # print(min_dis1, min_dis2)
    if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
        return True
    else:
        return False

def normalize_state(x):
    y = copy.deepcopy(x)
    y = np.array(y)
    y = y - np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    y = y / np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
    return y.tolist()

def reward_func(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()
    gamma = 0.99
    min_dis1 = np.inf
    reward = 0
    collide_flag = False
    states_seq = []

    for j in range(100):
        air1.step()
        reward = reward * gamma + air1.row / 60261.0
        states_seq.append(normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint]))
        if air1.row < dis_threshold:
            collide_flag = True
            reward -= 100

    return reward, collide_flag, states_seq


def testing(args):

    case_dimension = 4
    diffusion_model = Diffusion(batch_size = 1, epoch = 10, data_size = case_dimension, training_step_per_spoch = 50, num_diffusion_step = 25)
    diffusion_model.setup()
    memory_model = Memory(size = 100)
    density_model = Density()


    min_obs = np.array([-1 for i in range(5)])
    max_obs = np.array([1 for i in range(5)])
    novelty_grid = Grid(min_obs, max_obs, args.grid)
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

    while current_time - start_time < 3600 * args.hour:

        current_time = time.time()
        if cur_step > 0 and cur_step % args.step == 0:
            normal_case_list = np.array(normal_case_list)
            metric_list      = np.array(metric_list)

            ######################################################## add guidance to loss here #################################################
            if args.method == 'generative':
                metrics = None
            elif args.method == 'generative+density':
                metrics = metric_list[:, [0]]
            elif args.method == 'generative+sensitivity':
                metrics = metric_list[:, [1]]
            elif args.method == 'generative+performance':
                metrics = metric_list[:, [2]]
            elif args.method == 'generative+novelty':
                metrics = metric_list[:, [3]]
            else:
                print('Please check the method parameters!')
                return

            diffusion_model.train(normal_case_list, metrics, args.method)
            normal_case_list = []
            metric_list = []
            memory_model.clear() 

            for _ in range(50):
                test_case = diffusion_model.generate()
                [acas_speed, row, theta, auto_theta] = test_case
                acas_speed, x2, y2, auto_theta = map_test_case(acas_speed, row, theta, auto_theta)
                test_case = [acas_speed, x2, y2, auto_theta]
                if verify(acas_speed, x2, y2, auto_theta) == False:
                    continue
                reward, collide_flag, states_seq = reward_func(acas_speed, x2, y2, auto_theta)
                if collide_flag:
                    current_time = time.time()
                    regular_time = (current_time - start_time) / 3600
                    failure_by_diffusion += 1
                    print('Diffusion Failure case found:', [regular_time, failure_by_diffusion, test_case])
                    diffusion_failure_count.append([regular_time, failure_by_diffusion, test_case])

        else:

            normal_case = np.random.uniform(0,1,4)
            [acas_speed, row, theta, auto_theta] = normal_case
            acas_speed, x2, y2, auto_theta = map_test_case(acas_speed, row, theta, auto_theta)


            if verify(acas_speed, x2, y2, auto_theta) == False:
                continue

            reward, collide_flag, sequence = reward_func(acas_speed, x2, y2, auto_theta)
            if collide_flag:
                print('random detected!!!!!!!!!!!!!!!!')

            ########################## Density, Sensitivity and other guidance ############################
            cases_list = memory_model.get_cases()
            density_list = memory_model.get_densities()
            sensitivity_list = memory_model.get_sensitivities()
            performance_list = memory_model.get_performances()

            density = density_model.state_coverage(sequence)
            sensitivity = compute_sensitivity(normal_case, cases_list, performance_list, reward)
            performance = reward

            abstract_id = novelty_grid.state_abstract(np.array([sequence[-1]]))[0]

            if abstract_id in novelty_dict.keys():
                novelty_dict[abstract_id] += 1
            else:
                novelty_dict[abstract_id] = 1
            novelty = novelty_dict[abstract_id]

            norm_density = normalize_data(density, memory_model.min_density, memory_model.max_density)
            norm_sensitivity = normalize_data(sensitivity, memory_model.min_sensitivity, memory_model.max_sensitivity)
            norm_performance = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
            norm_novelty = normalize_data(novelty, memory_model.min_novelty, memory_model.max_novelty)

            # a larger sensitivity or novelty is the better
            norm_sensitivity = 1 - norm_sensitivity
            norm_novelty     = 1 - norm_novelty


            normal_case_list.append(normal_case)
            metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
            memory_model.append(normal_case, density, sensitivity, performance, novelty)
        cur_step += 1

    os.makedirs('results', exist_ok=True)
    with open('results/' + args.method + '_diffusion_failure_count.json', 'w') as f:
        json.dump(diffusion_failure_count, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--terminate", type=float, default=1.0)
    parser.add_argument("--seed_size", type=float, default=500)
    parser.add_argument("--picklepath", type=str, default='')

    ######################## parameters for generative testing ############################################
    parser.add_argument("--method", help="select the guidance for testing", default="generative", type=str, required=False)
    parser.add_argument("--hour", help="test time", default=1, type=int)
    parser.add_argument("--step", help="number of normal cases at each training step", default=100, type=int)
    parser.add_argument("--grid", help="state abstraction granularity", default=10, type=int)

    args = parser.parse_args()

    testing(args)