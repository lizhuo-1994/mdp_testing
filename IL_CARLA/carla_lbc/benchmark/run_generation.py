from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import time, copy, pickle, json, os, sys, traceback
import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu
from bird_view.models.common import crop_birdview
from fuzz.fuzz import fuzzing
from fuzz.replayer import replayer
import carla
from carla import ColorConverter
from carla import WeatherParameters

from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid, Carla_ENV
from diffusion import Diffusion

def _paint(observations, control, diagnostic, debug, env, show=False):
    import cv2
    import numpy as np
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2
    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    def _stick_together(a, b, axis=1):
        if axis == 1:
            h = min(a.shape[0], b.shape[0])
            r1 = h / a.shape[0]
            r2 = h / b.shape[0]
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
            return np.concatenate([a, b], 1)
            
        else:
            h = min(a.shape[1], b.shape[1])
            r1 = h / a.shape[1]
            r2 = h / b.shape[1]
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
            return np.concatenate([a, b], 0)

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)

    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(observations['command'], '???')

    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED
    
    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED       
 
    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y-S:y+S+1,x-S:x+S+1] = RED
        else:
            canvas[y-S:y+S+1,x-S:x+S+1] = RED
    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)
        try:
            birdview[x,y] = [155, 0, 155]
        except:
            pass
    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x-R:x+R+1,y-R:y+R+1] = [0, 155, 155]
    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)
    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))
    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)
    full = canvas

    if show:
        bzu.show_image('canvas', full)
    bzu.add_to_video(full)

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

def save_pickle(replayer):
    corpus = []
    total_crash = len(replayer.corpus)
    for i in range(total_crash):
        single_crash = []
        temp_trans = replayer.corpus[i][0]
        single_crash.append([temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll])
        temp_vehicleinfo = replayer.corpus[i][1]
        total_vehicle = len(temp_vehicleinfo)
        vehicle_info_crash = []
        for j in range(total_vehicle):
            temp_blue_print = temp_vehicleinfo[j][0]
            temp_trans = temp_vehicleinfo[j][1]
            temp_color = temp_vehicleinfo[j][2]
            temp_vehicle_id = temp_vehicleinfo[j][3]
            vehicle_info_crash.append([temp_blue_print.id, temp_blue_print.tags, temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll, temp_color, temp_vehicle_id])
        corpus.append([single_crash, vehicle_info_crash])
        replayer.envsetting[i][2] = replayer.envsetting[i][3]

    replayer.corpus = corpus
    replayer.original = []
    with open('./results/crash.pkl', 'wb') as handle:
        pickle.dump(replayer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Pickle Saved!!!')

def load_pickle(pickle_path, prints):
    with open(pickle_path, 'rb') as handle:
        replayer = pickle.load(handle)
    corpus = []
    envsetting = []
    total_crash = len(replayer.corpus)

    for i in range(total_crash):
        temp_trans = replayer.corpus[i][0][0]
        single_trans = carla.Transform(carla.Location(x=temp_trans[0], y=temp_trans[1], z=temp_trans[2]), carla.Rotation(pitch=temp_trans[3], yaw=temp_trans[4], roll=temp_trans[5]))
        vehicle_info_crash = replayer.corpus[i][1]
        total_vehicle = len(vehicle_info_crash)
        singel_vehicle = []
        for j in range(total_vehicle):
            blue_print = prints.filter(vehicle_info_crash[j][0])[0]
            assert blue_print.tags == vehicle_info_crash[j][1]
            blue_print.set_attribute("role_name", "autopilot")
            color = vehicle_info_crash[j][8]
            vehicle_id = vehicle_info_crash[j][9]
            if color != None:
                blue_print.set_attribute("color", color)
            if vehicle_id != None:
                blue_print.set_attribute("driver_id", vehicle_id)

            trans = carla.Transform(carla.Location(x=vehicle_info_crash[j][2], y=vehicle_info_crash[j][3], z=vehicle_info_crash[j][4]), carla.Rotation(pitch=vehicle_info_crash[j][5], yaw=vehicle_info_crash[j][6], roll=vehicle_info_crash[j][7]))
            singel_vehicle.append((blue_print, trans))
        corpus.append((single_trans, singel_vehicle))
        envsetting.append([replayer.envsetting[i][0], replayer.envsetting[i][1], cu.PRESET_WEATHERS[replayer.envsetting[i][2]], replayer.envsetting[i][3]])
    replayer.corpus = corpus
    replayer.envsetting = envsetting
    return replayer

def run_single(env, weather, start, target, agent_maker, seed, show=False, replay=False, em_guide=True, args = None):
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    print('replay = ', replay)
    tsne_data = []
    agent = agent_maker()
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
                        # tsne_data.append(copy.deepcopy(current_tsne))

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

                        if env._tick > 200:
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

                seq_entropy = 0
                first_reward_flag = True
                total_reward = 0
                sequence = []
                while env.tick():
                    observations = env.get_observations()
                    if first_reward_flag == False:
                        cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                        cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                    else:
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


                    sequence.append(copy.deepcopy(temp))

                    if env.is_failure() or env.is_success() or env._tick > 200:
                        break

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

    os.makedirs('carla_lbc/results', exist_ok=True)
    with open('carla_lbc/results/' + args.method + '_diffusion_failure_count.json', 'w') as f:
        json.dump(diffusion_failure_count, f)

    return None, None


def run_benchmark(agent_maker, env, benchmark_dir, seed, resume, max_run=5, show=False, replay=False, em_guide=True, args = None):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / 'summary.csv'
    diagnostics_dir = benchmark_dir / 'diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in env.all_tasks:
        if resume and len(summary) > 0 and ((summary['start'] == start) \
                       & (summary['target'] == target) \
                       & (summary['weather'] == weather)).any():
            print (weather, start, target)
            continue

        diagnostics_csv = str(diagnostics_dir / ('%s.csv' % run_name))
        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, show=show, replay=replay, em_guide=em_guide, args = args)
        exit()

        summary = summary.append(result, ignore_index=True)
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)
        break
