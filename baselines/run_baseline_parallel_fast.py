from os.path import exists
from pathlib import Path
import datetime
import os
import uuid
import sys
import numpy as np
import importlib
import torch
import time
#from red_gym_env import RedGymEnv # deprecated. done dynamically via conf.gym_env now
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # sleep 5s
        # time.sleep(5)
        # env_class = import_file('RedGymEnv', env_conf['gym_env'])
        print('\nSEARCHING FOR CODE FROM: ' + env_conf['gym_env'])
        if env_conf['gym_env'] == 'default':
            from red_gym_env import RedGymEnv
            env = RedGymEnv(env_conf)
        else:
            sys.path.append(os.getcwd())
            RedGymEnvClass = getattr(importlib.import_module(env_conf['gym_env']), 'RedGymEnv')
            print('\nRedGymEnvClass Found: ', RedGymEnvClass)
            env = RedGymEnvClass(env_conf)
        path_str = '' + os.path.abspath(sys.modules[env.__class__.__module__].__file__)
        print('\nEXECUTING CODE FROM:' + path_str)

        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':


    ep_length = 256 * 100
    print('\nCWD: ', Path.cwd())

    #anticipates .\Eureka\eureka\outputs\eureka\2023-11-13_13-04-20 CWD, we just want .\Eureka :
    eureka_root_dir = str(Path.cwd().parent.parent.parent.parent)
    relative_eureka_root_dir = '../../../../'
    pokemon_root_dir = eureka_root_dir + '/PokemonRedExperiments'
    sess_path = Path(pokemon_root_dir + f'/sessions/session_{str(uuid.uuid4())[:8]}')
    env_config = {
            'headless': False, 'save_final_state': True, 'early_stop': False,
            'action_freq': 24, 'init_state': pokemon_root_dir + '/has_pokedex_nballs.state', 'max_steps': ep_length, 
            'print_rewards': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
            'gb_path': pokemon_root_dir + '/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
            'use_screen_explore': True, 'extra_buttons': True,
            'gym_env': 'default' # default, overwritten by eureka generations
        }
    
    # add stdin params to env_config
    print('\nsys.argv: ', sys.argv)
    for arg in sys.argv[1:]:
        k, v = arg.split('=')
        env_config[k] = v

    print('\n CONFIG: ', env_config)
    
    num_cpu = 1  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 1
    # put a checkpoint here you want to start from

    # get latest from sessions folder:
    session_folders = [f for f in os.listdir(pokemon_root_dir + '/sessions') if os.path.isdir(os.path.join(pokemon_root_dir + '/sessions', f))]
    # filter for sessions with any .zip file inside:
    session_folders = [pokemon_root_dir + '/sessions/' + f for f in session_folders if len([x for x in os.listdir(pokemon_root_dir + '/sessions/' + f) if x.endswith('.zip')]) > 0]
    # get latest session:
    latest_session = None
    if len(session_folders) > 0:
        latest_session = max(session_folders, key=os.path.getctime)
        print('\nlatest session: ', latest_session)
    # get .zip file inside latest session:
    latest_session_file = None
    if latest_session:
        latest_session_file = max([pokemon_root_dir + '/sessions' + latest_session.split('/sessions')[1] + '/' + x for x in os.listdir(latest_session) if x.endswith('.zip')], key=os.path.getctime)
        latest_session_file = latest_session_file.split('.zip')[0]
        print('\nlatest session file: ', latest_session_file)
        

    if (latest_session_file):
        session_file = latest_session_file
    else:
        session_file = pokemon_root_dir + 'sessions/session_4da05e87_main_good/poke_439746560_steps' # pretrained
    #session_file = 'session_e41c9eff/poke_38207488_steps' # fresh
    
    if exists(session_file + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(session_file, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('\nERROR could not find checkpoint')
        
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu, callback=checkpoint_callback)

    print('\nPokemon emulation done')

    # dump config dict
    exp_date = 'red_gym_env' + '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_dir = os.path.join('runs', exp_date)
    os.makedirs(Path.cwd() / experiment_dir, exist_ok=True)
    os.makedirs(Path.cwd() / experiment_dir / "nn", exist_ok=True)
    os.makedirs(Path.cwd() / experiment_dir / "summaries", exist_ok=True)

    # process self.agent_stats
    # step	x	y	map	last_action	pcount	levels	ptypes	hp	frames	deaths	badge	event	heal
    # summarize and write to summaries text
    """ 
    model = model.env
    with open(Path.cwd() / experiment_dir / "summaries" / "agent_stats.txt", "w") as f:
        # get sums and averages, mins and maxes
        stats = {
            'average': {},
            'sum': {},
            'min': {},
            'max': {},
            'variance': {},
            'count': {}
        }
        # get last row:
        data = model.agent_stats[-1]
        for k, v in data.items():
            stats['sum'][k] = v
            stats['min'][k] = min(stats['min'][k], v)
            stats['max'][k] = max(stats['max'][k], v)
            stats['variance'][k] = v ** 2
            stats['average'][k] = stats['sum'][k]
            stats['count'][k] = 1

        # averag all rewards
        for i, data in model.all_runs:
            for k, v in data.items():
                if k not in stats['sum']:
                    stats['sum'][k] = 0
                    stats['min'][k] = v
                    stats['max'][k] = v
                    stats['variance'][k] = 0
                stats['sum'][k] += v
                stats['min'][k] = min(stats['min'][k], v)
                stats['max'][k] = max(stats['max'][k], v)
                stats['variance'][k] += v ** 2
                stats['average'][k] = stats['sum'][k] / i
                stats['count'][k] += 1
        # calculate variance
        for k, v in stats['variance'].items():
            stats['variance'][k] = v / stats['count'][k] - stats['average'][k] ** 2
            # write to file
            f.write(f"{k}: total {stats['sum'][k]}, average {stats['average'][k]} +/- {stats['variance'][k]}, min: {stats['min'][k]}, max: {stats['max'][k]}\n")
        f.write(f"consecutive_successes: {model.successes}\n")
    """
        
    summaries = Path.cwd() / experiment_dir / "summaries"
    print("Network Directory: ", Path.cwd() / experiment_dir / "nn")
    print("Tensorboard Directory: ", summaries)
    env.close()
    sys.exit()
