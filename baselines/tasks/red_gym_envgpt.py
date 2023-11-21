
import sys
import uuid 
import os
from math import floor, sqrt
import json
import torch
from pathlib import Path
from typing import *

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

class RedGymEnv(Env):


    def __init__(
        self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []
        self.max_opponent_poke = 0
        self.max_poke_levels = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.total_healing_rew = 0
        self.died_count = 0
        self.step_count = 0
        self.total_reward = 0
        self.rew_buf = torch.zeros(1)
        self.rew_dict = {}
        self.extras = {}

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        
        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)
            
        self.reset()

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem()

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []
        
        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()
       
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.step_count = 0
        self.total_reward, progress_reward = self.compute_total_reward()
        self.progress_reward = progress_reward
        self.reset_count += 1
        return self.render(), {}
    
    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)
        
    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    
    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
            frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)

        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords()
            
        self.update_heal_reward()

        new_reward, new_prog = self.step_reward()
        
        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))
    
    def append_agent_stats(self, action):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        if self.use_screen_explore:
            expl = ('frames', self.knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self.seen_coords))
        self.agent_stats.append({
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'last_action': action,
            'pcount': self.read_m(0xD163), 'levels': levels, 'ptypes': self.read_party(),
            'hp': self.read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self.died_count, 'badge': self.get_badges(),
            'event': self.progress_reward['event'], 'healr': self.total_healing_rew
        })

    def update_frame_knn_index(self, frame_vec):
        
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )
    
    def update_seen_coords(self):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}
        
        self.seen_coords[coord_string] = self.step_count

    def step_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        new_total, progress_reward = self.compute_total_reward()
        self.progress_reward = progress_reward
        new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.read_hp_fraction() > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )
    
    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['level'] * 100 / self.reward_scale, 
                self.read_hp_fraction()*2000, 
                prog['explore'] * 150 / (self.explore_weight * self.reward_scale))
               #(prog['events'], 
               # prog['levels'] + prog['party_xp'], 
               # prog['explore'])

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')
    
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew
    
    def get_knn_reward(self):
        
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post
    
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1
                
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )

    def compute_total_reward(self):

        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        self.num_poke = self.read_m(0xD163)
        self.poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        self.money = self.read_money() - 975 # subtract starting money
        self.seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        self.all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        self.oak_parcel = self.read_bit(0xD74E, 1) 
        self.oak_pokedex = self.read_bit(0xD74B, 5)
        self.opponent_level = self.read_m(0xCFF3)
        self.level_rew = self.get_levels_reward()
        self.max_opponent_level = max(self.max_opponent_level, self.opponent_level)
        self.enemy_poke_count = self.read_m(0xD89C)
        self.badges = self.bit_count(self.read_m(0xD356))
        self.knn_count = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        self.max_poke_levels = max(self.max_poke_levels, self.get_levels_sum())
        self.knn_reward = self.get_knn_reward()
    

        total_reward, sub_rewards = self.compute_reward()

        # event, level, explore
        # add defaults for special sub_rewards if blank:
        sub_rewards['event'] = sub_rewards.get('event', sub_rewards.get('event_reward', self.all_events_score))
        sub_rewards['level'] = sub_rewards.get('level', sub_rewards.get('level_reward', self.level_rew))
        sub_rewards['explore'] = sub_rewards.get('explore', sub_rewards.get('explore_reward', self.knn_reward))

        # if tensor, reduce total_reward from tensor to avg float
        if isinstance(total_reward, torch.Tensor):
            total_reward = total_reward.mean().item()
            # same for each sub_reward
            sub_rewards = {k: v.mean().item() for k, v in sub_rewards.items()}
        return total_reward, sub_rewards
    
    # def compute_reward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    def compute_reward(self):
        total_reward, sub_rewards = compute_success(self.num_poke, self.poke_xps, self.money, self.seen_poke_count, self.all_events_score, self.badges, self.knn_reward, self.level_rew, self.max_opponent_level, self.total_healing_rew, self.died_count)
        # TODO define compute_success function below
        # total_reward, sub_rewards = compute_success(...)
        return total_reward, sub_rewards
    
    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
    
    def update_max_op_level(self):
        #opponent_level = self.read_m(0xCFE8) - 5 # base level
        opponent_level = max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2
    
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))
    

# general reward function guide:
def compute_success(
        # some_var: float,
        # TODO add any class variables from RedGymEnv you need here as input vars
    ) -> Tuple[float, Dict[str, float]]:
    # TODO make better reward function!!!
    # use conditional scenarios to compute reward logic

    # explore mode:

    # battle mode:
    
    # ...etc

    # return tensor total as weighted sum of sub_rewards
    # total_reward = 
    return total_reward, sub_rewards

# best reward function so far:
def compute_success(
        num_poke: float, poke_xps: List[float], money: float, seen_poke_count: float,
        all_events_score: float, badges: float, knn_reward: float
    ) -> Tuple[float, Dict[str, float]]:

    # Adjustments to scale and temperature parameters
    money_temp, poke_count_temp, xp_temp = 0.00001, 2.0, 0.1
    events_temp, badges_temp, explore_temp = 0.2, 20.0, 0.8

    # Normalize and scale money assuming the reward increases linearly with money, capping at 60000 for better gradient
    money_reward = money_temp * min(money, 60000)

    # Enhance seen Pokemon count by a temperature parameter, since Pokemon can be caught frequently
    poke_count_reward = poke_count_temp * (seen_poke_count / 150.0)

    # Scale experience by counting XP points rewarding leveling up Pokemon rather than catching many Pokemon with low XP
    xp_reward = xp_temp * sum(poke_xps) / (100.0 * 6.0) # Assuming 100 is max level

    # Scale events reward to be more sensitive to game progression events
    events_reward = events_temp * (all_events_score / 50.0)

    # Badges are very important so increase the temperature parameter significantly to encourage collecting badges
    badges_reward = badges_temp * (badges / 8.0)

    # Exploration should be sensitive but not overly so, considering the KNN count as an exploration metric
    explore_reward = explore_temp * (knn_reward / 128.0)

    # Compute each sub-reward and compile into a dictionary
    sub_rewards = {
        'money_reward': money_reward,
        'poke_count_reward': poke_count_reward,
        'xp_reward': xp_reward,
        'events_reward': events_reward,
        'badges_reward': badges_reward,
        'explore_reward': explore_reward
    }

    # Compute the total reward
    total_reward = sum(sub_rewards.values())

    return total_reward, sub_rewards

# write new reward function here:

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
def compute_success(
    num_poke: float, 
    poke_xps: List[float], 
    money: float, 
    seen_poke_count: float,
    all_events_score: float, 
    badges: float, 
    knn_reward: float,
    level_rew: float,
    max_opponent_level: float,
    total_healing_rew: float,
    died_count: float
) -> Tuple[float, Dict[str, float]]:

    # Adjust scale and temperature parameters for each reward component
    money_temp, poke_count_temp, xp_temp = 0.00001, 2.0, 0.1
    events_temp, badges_temp, explore_temp = 0.2, 20.0, 0.8
    level_temp, healing_temp, death_penalty_temp = 0.05, 0.002, -1.0
    opponent_level_temp = 0.01
    max_reward_cap = 60000  # Cap for normalization of money reward

    # Normalize and scale each reward component
    money_reward = money_temp * min(money, max_reward_cap)
    poke_count_reward = poke_count_temp * (seen_poke_count / 150.0)
    xp_reward = xp_temp * torch.tensor(poke_xps).sum() / (100.0 * 6.0)
    events_reward = events_temp * (all_events_score / 50.0)
    badges_reward = badges_temp * (badges / 8.0)
    explore_reward = explore_temp * (knn_reward / 128.0)
    level_reward = level_temp * (level_rew / 100.0)
    healing_reward = healing_temp * total_healing_rew
    death_penalty = death_penalty_temp * died_count
    opponent_level_reward = opponent_level_temp * (max_opponent_level / 100.0) 
    
    # Combine rewards into a dictionary
    sub_rewards = {
        'money_reward': money_reward,
        'poke_count_reward': poke_count_reward,
        'xp_reward': xp_reward,
        'events_reward': events_reward,
        'badges_reward': badges_reward,
        'explore_reward': explore_reward,
        'level_reward': level_reward,
        'healing_reward': healing_reward,
        'death_penalty': death_penalty,
        'opponent_level_reward': opponent_level_reward
    }
    
    # Define weights for each reward component if needed, otherwise equally weight them
    reward_weights = {
        'money_reward': 1.0,
        'poke_count_reward': 1.0,
        'xp_reward': 1.0,
        'events_reward': 1.0,
        'badges_reward': 1.0,
        'explore_reward': 1.0,
        'level_reward': 1.0,
        'healing_reward': 1.0,
        'death_penalty': 1.0,
        'opponent_level_reward': 1.0
    }
    
    # Compute the total weighted reward
    total_reward = sum(sub_rewards[key] * reward_weights[key] for key in sub_rewards)

    return total_reward, sub_rewards
