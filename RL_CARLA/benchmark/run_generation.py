import pandas as pd
import tqdm
import random
import bird_view.utils.carla_utils as cu
import numpy as np
import copy
import time, os, json, sys
import pickle
import traceback
import carla
from carla import ColorConverter
from carla import WeatherParameters


from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid, Carla_ENV
from diffusion import Diffusion

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    reward = 0.0
    reward += np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    reward += 0.2 * (cur_speed_norm - prev_speed_norm)
    if cur_collid:
        reward -= 100 * cur_speed_norm
    if cur_invade:
        reward -= cur_speed_norm

    return reward

def collect_corpus():
    return None

def get_index(sequence):
    result = np.zeros(17)
    mins = [-10, 90, -1, -1, -10, -10, -1, -20, -20, -5, -10, 90, 0, 1, 50, 50, 0]
    maxs = [200, 350, 1, 1, 10, 10, 1, 20, 20, 5, 200, 350, 0.5, 4, 300, 300, 15]
    for i in range(sequence.shape[0]):
        if sequence[i] < mins[i]:
            result[i] = 0
        elif sequence[i] > maxs[i]:
            result[i] = 10000
        else:
            result_i = (sequence[i] - mins[i]) / (maxs[i] - mins[i]) * 10000 + 1
            if np.isnan(result_i):
                result[i] = 10000
            else:
                result[i] = int(result_i)
    return result

def update_dict(storage, sequences):
    for i in range(len(sequences)):
        index = get_index(sequences[i])
        for j in range(index.shape[0]):
            storage[int(j * 10000 + index[j])] = 1
    return storage

def run_single(env, weather, start, target, agent_maker, seed, replay=False, em_guide=True, args = None):
    # HACK: deterministic vehicle spawns.
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    carla_env = Carla_ENV()
    ############################################## get the valid scope of locations ########################
    spawn_points = env._map.get_spawn_points()

    agent = agent_maker()
    tsne_data = []
    min_seq = []
    max_seq = []
    k_storage_state_cvg = {}
    count_of_GMM = 0

    case_dimension = 1 + 3 + 1 + 1 + 2 * 100
    diffusion_model = Diffusion(batch_size = 1, epoch = 100, data_size = case_dimension, training_step_per_spoch = 100, num_diffusion_step = 50)
    diffusion_model.setup()
    memory_model = Memory(size = 100)
    density_model = Density()


    min_obs = np.array([-10, 90, -1, -1, -10, -10, -1, -20, -20, -5, -10, 90, 0, 1, 50, 50, 0])
    max_obs = np.array([200, 350, 1, 1, 10, 10, 1, 20, 20, 5, 200, 350, 0.5, 4, 300, 300, 15])
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
            terminate_list = []
            memory_model.clear() 

            for _ in range(100):
                try:
                    env.replay = False
                    env.fuzzer = None
                    env.generative = True
                    env.first_run = False
                    generated_case = diffusion_model.generate()
                    carla_env.from_vector(generated_case)
                    env.test_settings = carla_env
                    initial_check = env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
                    start_pose = env._start_pose

                    test_case = []
                    test_case = test_case + [env._start_pose.location.x, env._start_pose.location.y, env._start_pose.rotation.yaw]
                    for i in range(len(carla_env.vehicles)):
                        test_case = test_case + [spawn_points[i].location.x + carla_env.vehicles[i][0], spawn_points[i].location.y + carla_env.vehicles[i][1]]
                    test_case = test_case + [spawn_points[carla_env.target_pose].location.x, spawn_points[carla_env.target_pose].location.y]
                    test_case = test_case + [carla_env.weather]
                    test_case = [ float('%.2f' % elem) for elem in test_case]

                    if initial_check == False:
                        print('Trigger initial collision!!!')
                        continue

                    diagnostics = list()
                    esult = {
                        "weather": weather,
                        "start": start,
                        "target": target,
                        "success": None,
                        "t": None,
                        "total_lights_ran": None,
                        "total_lights": None,
                        "collided": None,
                    }
                    seq_entropy = 0

                    first_reward_flag = True
                    total_reward = 0
                    sequence = []
                    while env.tick():
                        observations = env.get_observations()
                        if first_reward_flag == False:
                            cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                            cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                        if first_reward_flag:
                            first_reward_flag = False
                            prev_distance = env._local_planner.distance_to_goal
                            prev_speed = observations['velocity']
                            prev_invaded_frame_number = env._invaded_frame_number
                            prev_collided_frame_number = env._collided_frame_number
                            cur_invade = False
                            cur_collid = False
                            if env.invaded:
                                cur_invade = True
                            if env.collided:
                                cur_collid = True

                        reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                        total_reward += reward
                        prev_distance = env._local_planner.distance_to_goal
                        prev_speed = observations['velocity']
                        prev_invaded_frame_number = env._invaded_frame_number
                        prev_collided_frame_number = env._collided_frame_number

                        control, current_entropy, _ = agent.run_step(observations)

                        temp = copy.deepcopy(observations['node'])
                        temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                        vehicle_index = np.nonzero(observations['vehicle'])
                        vehicle_obs = np.zeros(3)
                        vehicle_obs[0] = vehicle_index[0].mean()
                        vehicle_obs[1] = vehicle_index[1].mean()
                        vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                        temp = np.hstack((temp, vehicle_obs))

                        seq_entropy += current_entropy
                        diagnostic = env.apply_control(control)
                        diagnostic.pop("viz_img")
                        diagnostics.append(diagnostic)
                        sequence.append(temp)

                        if env.is_failure() or env.is_success() or env._tick > 100:
                            break

                    if env.is_failure() or env.collided:
                        if test_case in diffusion_failure_list:
                            pass
                        else:
                            current_time = time.time()
                            regular_time = (current_time - start_time) / 3600
                            failure_by_diffusion += 1
                            print('Diffusion Failure case found:', [regular_time, failure_by_diffusion, test_case])
                            diffusion_failure_count.append([regular_time, failure_by_diffusion, test_case])
                except Exception as e: 
                    print(e)
                    print(carla_env.start_pose, carla_env.target_pose)
                    print(traceback.format_exc())
                    continue
        else:
            try:
                env.replay = False
                env.fuzzer = None
                env.generative = True
                env.first_run = False

                normal_case = np.random.uniform(-1,1, case_dimension)
                carla_env.from_vector(normal_case)
                env.test_settings = carla_env
                initial_check = env.init(start=carla_env.start_pose, target=carla_env.target_pose, weather=carla_env.weather)
                start_pose = env._start_pose

                ################################################ save the carla env embedding ############################################################

                test_case = []
                test_case = test_case + [env._start_pose.location.x, env._start_pose.location.y, env._start_pose.rotation.yaw]
                for i in range(len(carla_env.vehicles)):
                    test_case = test_case + [spawn_points[i].location.x + carla_env.vehicles[i][0], spawn_points[i].location.y + carla_env.vehicles[i][1]]
                test_case = test_case + [spawn_points[carla_env.target_pose].location.x, spawn_points[carla_env.target_pose].location.y]
                test_case = test_case + [carla_env.weather]
                test_case = [ float('%.2f' % elem) for elem in test_case]

                diagnostics = list()
                esult = {
                    "weather": weather,
                    "start": start,
                    "target": target,
                    "success": None,
                    "t": None,
                    "total_lights_ran": None,
                    "total_lights": None,
                    "collided": None,
                }
                seq_entropy = 0

                first_reward_flag = True
                total_reward = 0
                sequence = []
                while env.tick():
                    observations = env.get_observations()
                    if first_reward_flag == False:
                        cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                        cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                    if first_reward_flag:
                        first_reward_flag = False
                        prev_distance = env._local_planner.distance_to_goal
                        prev_speed = observations['velocity']
                        prev_invaded_frame_number = env._invaded_frame_number
                        prev_collided_frame_number = env._collided_frame_number
                        cur_invade = False
                        cur_collid = False
                        if env.invaded:
                            cur_invade = True
                        if env.collided:
                            cur_collid = True

                    reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                    total_reward += reward
                    prev_distance = env._local_planner.distance_to_goal
                    prev_speed = observations['velocity']
                    prev_invaded_frame_number = env._invaded_frame_number
                    prev_collided_frame_number = env._collided_frame_number

                    control, current_entropy, _ = agent.run_step(observations)

                    temp = copy.deepcopy(observations['node'])
                    temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                    vehicle_index = np.nonzero(observations['vehicle'])
                    vehicle_obs = np.zeros(3)
                    vehicle_obs[0] = vehicle_index[0].mean()
                    vehicle_obs[1] = vehicle_index[1].mean()
                    vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                    temp = np.hstack((temp, vehicle_obs))

                    seq_entropy += current_entropy
                    diagnostic = env.apply_control(control)
                    diagnostic.pop("viz_img")
                    diagnostics.append(diagnostic)
                    sequence.append(temp)

                    if env.is_failure() or env.is_success() or env._tick > 100:
                        break

                # if env.is_failure() or env.collided:
                #     if test_case in diffusion_failure_list:
                #         pass
                #     else:
                #         current_time = time.time()
                #         regular_time = (current_time - start_time) / 3600
                #         failure_by_diffusion += 1
                #         print('Random Failure case found:', [regular_time, failure_by_diffusion, test_case])
                #         diffusion_failure_count.append([regular_time, failure_by_diffusion, test_case])
                

                ########################## Density, Sensitivity and other guidance ############################
                cases_list = memory_model.get_cases()
                density_list = memory_model.get_densities()
                sensitivity_list = memory_model.get_sensitivities()
                performance_list = memory_model.get_performances()

                density = density_model.state_coverage(sequence)
                sensitivity = compute_sensitivity(normal_case, cases_list, performance_list, total_reward)
                performance = total_reward


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

            except Exception as e: 
                print(e)
                print(carla_env.start_pose, carla_env.target_pose)
                print(traceback.format_exc())
                continue

        cur_step += 1


    print(args.method)
    os.makedirs('results', exist_ok=True)
    with open('results/' + args.method + '_diffusion_failure_count.json', 'w') as f:
        json.dump(diffusion_failure_count, f)
    print(os.getcwd)


    
def run_benchmark(agent_maker, env, benchmark_dir, seed, resume, max_run=5, replay=False, em_guide=True, args = None):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / "summary.csv"
    diagnostics_dir = benchmark_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in env.all_tasks:
        if (
            resume
            and len(summary) > 0
            and (
                (summary["start"] == start)
                & (summary["target"] == target)
                & (summary["weather"] == weather)
            ).any()
        ):
            print(weather, start, target)
            continue

        diagnostics_csv = str(diagnostics_dir / ("%s.csv" % run_name))

        run_single(env, weather, start, target, agent_maker, seed, replay, em_guide, args)
        break

        # summary = summary.append(result, ignore_index=True)

        # # Do this every timestep just in case.
        # pd.DataFrame(summary).to_csv(summary_csv, index=False)
        # pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)