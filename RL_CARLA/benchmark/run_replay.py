from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import time, copy, pickle, json, os, sys, traceback, random
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

def load_test_cases(file_path):
    test_cases = []
    with open(file_path, 'r') as f:
        results = json.load(f)
    for r in results:
        test_cases.append(r[2])
    return test_cases


def run_single(env, weather, start, target, agent_maker, seed, show=False, replay=False, em_guide=True, args = None):
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    tsne_data = []
    agent = agent_maker()
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    carla_env = Carla_ENV()
    ############################################## get the valid scope of locations ########################
    spawn_points = env._map.get_spawn_points()

    agent = agent_maker()
    
    env.replay = True
    env.fuzzer = None
    env.generative = False
    env.first_run = False

    #test_cases = load_test_cases('results/fuzz_failure_count.json')
    test_cases = load_test_cases('results/generative+novelty_diffusion_failure_count.json')

    for i in range(len(test_cases)):
        try:
            bzu.init_video(save_dir='./results/new-videos/generative/', save_path=str(i))
            test_case = test_cases[i]
            carla_env.from_test_case(test_case)
            env.test_settings = carla_env
            initial_check = env.init(start=0, target=1, weather=carla_env.weather)
            start_pose = env._start_pose

            ################################################ save the carla env embedding ############################################################

            seq_entropy = 0
            first_reward_flag = True
            total_reward = 0
            sequence = []
            diagnostics = list()
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

                if show:
                    _paint(observations, control, diagnostic, agent.debug, env, show=show)

                diagnostic.pop('viz_img')
                diagnostics.append(diagnostic)
                sequence.append(temp)

                if env.is_failure() or env.is_success() or env._tick > 200:
                    break
                
            print('test_case: ', i, '. The len is: ', len(sequence), '. is_failure: ', env.is_failure(), '. is_success: ', env.is_success())
            if env.is_failure() or env.collided:
                print('failure replicated')
                
        except Exception as e: 
            print(e)
            print(traceback.format_exc())
            continue

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
