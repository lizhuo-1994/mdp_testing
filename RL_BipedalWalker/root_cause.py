import argparse, importlib, os, sys, time, copy, tqdm, pickle, gym, yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
import json, random, pickle


from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid
from diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )


    ######################## parameters for generative testing ############################################
    parser.add_argument("--method", help="select the guidance for testing", default="generative", type=str, required=False)
    parser.add_argument("--hour", help="test time", default=12, type=int)
    parser.add_argument("--step", help="number of normal cases at each training step", default=100, type=int)
    parser.add_argument("--grid", help="state abstraction granularity", default=2, type=int)
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    ##################################################################################################

    case_dimension = 15
    diffusion_model = Diffusion(batch_size = 1, epoch = 100, data_size = case_dimension, training_step_per_spoch = 50, num_diffusion_step = 50)
    diffusion_model.setup()
    memory_model = Memory(size = 100)
    density_model = Density()


    ################################### nvovelty computation ########################################
    min_obs = np.array([-5 for i in range(env.observation_space.shape[0])])
    max_obs = np.array([5 for i in range(env.observation_space.shape[0])])
    novelty_grid = Grid(min_obs, max_obs, args.grid)
    novelty_dict = dict()



    np.random.seed()
    states = np.random.randint(low=1, high=4, size=15)
    obs = env.reset(states)

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    ep_len = 0

    total_step = 100000
    val_step = 1000
    cur_step = 0
    wins = 0
    lose = 0
    failure_by_diffusion = 0
    failure_by_random = 0
    done = False
    regular_time = 0
    normal_case_list = []
    metric_list = []
    density_list = []

    sensitivity_list = []
    performance_list = []
    novelty_list = []
    diffusion_failure_list = []
    random_failure_list = []
    diffusion_failure_count = []
    random_failure_count = []
    start_time = time.time()
    current_time = time.time()

    #######################################################################################
    trajectory_list = []
    termination_list = []
    failure_flag = False

    while current_time - start_time < 3600 * args.hour:


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
            elif args.method == 'generative+baseline':
                metrics = metric_list[:, [0,1,2]]
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

            for _ in range(val_step):
                failure_flag = False
                seed = random.randint(1,1000)
                env.seed(seed)
                state = None
                test_case = diffusion_model.generate()
                obs = env.reset(test_case)
                sequences = [obs[0]]
                episode_reward = 0.0
                # print('states ', states)
                for _ in range(args.n_timesteps):
                    action, state = model.predict(obs, state=state, deterministic=deterministic)
                    obs, reward, done, infos = env.step(action)
                    sequences.append(obs[0])
                    episode_reward += reward[0]
                    if done:
                        break
                if done or episode_reward < 10:
                    failure_flag = True
                    save_case = test_case.tolist()
                    if save_case in random_failure_list or save_case in diffusion_failure_list:
                        pass
                    else:
                        lose += 1
                        done = False
                        regular_time = (current_time - start_time) / 3600
                        diffusion_failure_list.append(save_case)
                        failure_by_diffusion += 1
                        print(regular_time, failure_by_diffusion, save_case)
                        diffusion_failure_count.append([regular_time, failure_by_diffusion, save_case])
                else:
                    wins += 1

                trajectory_list.append(sequences)
                termination_list.append(sequences[-1])

            trajectory_list = np.array(trajectory_list)
            termination_list = np.array(termination_list)
            os.makedirs('results', exist_ok=True)
            np.save('results/' + args.method + '+trajectory.npy', trajectory_list)
            np.save('results/' + args.method + '+termination.npy', termination_list)

            exit()
                    
        else:
            seed = np.random.randint(1,1000)
            env.seed(seed)
            state = None
            normal_case = np.random.randint(low=1, high=4, size=15)
            
            obs = env.reset(normal_case)
            sequences = [obs[0]]
            episode_reward = 0.0
            # print('states ', states)
            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                sequences.append(obs[0])
                episode_reward += reward[0]
                if done:
                    break
                
            ########################## Density, Sensitivity and other guidance ############################
            cases_list = memory_model.get_cases()
            density_list = memory_model.get_densities()
            sensitivity_list = memory_model.get_sensitivities()
            performance_list = memory_model.get_performances()

            density = density_model.state_coverage(sequences)
            sensitivity = compute_sensitivity(normal_case, cases_list, performance_list, episode_reward)
            performance = episode_reward

            ############################ calculate novelty ################################################
            abstract_id = novelty_grid.state_abstract(np.array([sequences[-1]]))[0]
            if abstract_id in novelty_dict.keys():
                novelty_dict[abstract_id] += 1
            else:
                novelty_dict[abstract_id] = 1
            novelty = novelty_dict[abstract_id]

            norm_density = normalize_data(density, memory_model.min_density, memory_model.max_density)
            norm_sensitivity = normalize_data(sensitivity, memory_model.min_sensitivity, memory_model.max_sensitivity)
            norm_performance = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
            norm_novelty = normalize_data(novelty, memory_model.min_novelty, memory_model.max_novelty)


            # print('density: ', density, '\t norm_density:\t ', norm_density)
            # print('sensitivity: ', sensitivity, '\t norm_sensitivity:\t ', norm_sensitivity)
            # print('performance: ', performance, '\t norm_performance:\t ', norm_performance)

            # a larger sensitivity or novelty is the better
            norm_sensitivity = 1 - norm_sensitivity
            norm_novelty     = 1 - norm_novelty


            normal_case_list.append(normal_case)
            metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
            memory_model.append(normal_case, density, sensitivity, performance, novelty)

            print(cur_step, norm_density, norm_sensitivity, norm_performance, norm_novelty)

        cur_step += 1
        current_time = time.time()
    
    os.makedirs('results', exist_ok=True)
    with open('results/' + args.method + '_diffusion_failure_count.json', 'w') as f:
        json.dump(diffusion_failure_count, f)




if __name__ == '__main__':  
    main()