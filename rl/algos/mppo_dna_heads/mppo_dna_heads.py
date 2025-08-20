# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# This file contains a modified version of CleanRL's PPO-DNA implementation:
# https://github.com/vwxyzjn/cleanrl/blob/caabea4c5b856f429baa2af8bc973d4994d4c330/cleanrl/ppo_dna_atari_envpool.py
import os
import sys
import random
import logging
import time
import copy
import enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import kl_divergence as kld
from torch.utils.mobile_optimizer import optimize_for_mobile
from rl.world.util.timer import Timer
from rl.world.util.misc import timer_stats

# import tyro
import warnings

from .. import common


if os.getenv("PYDEBUG", None) == "1":
    def excepthook(exc_type, exc_value, tb):
        import ipdb
        ipdb.post_mortem(tb)

    sys.excepthook = excepthook


@dataclass
class ScheduleArgs:
    # const / lin_decay / exp_decay
    mode: str = "const"
    start: float = 2.5e-4
    end: float = 0
    rate: float = 10


@dataclass
class EnvArgs:
    max_steps: int = 500
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    user_timeout: int = 30
    vcmi_timeout: int = 30
    boot_timeout: int = 30
    random_heroes: int = 1
    random_obstacles: int = 1
    town_chance: int = 0
    random_stack_chance: int = 0,
    warmachine_chance: int = 0
    random_terrain_chance: int = 0
    tight_formation_chance: int = 0
    battlefield_pattern: str = ""
    mana_min: int = 0
    mana_max: int = 0
    reward_step_fixed: float = -1
    reward_dmg_mult: float = 1
    reward_term_mult: float = 1
    reward_relval_mult: float = 1
    swap_sides: int = 0

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class NetworkArgs:
    attention: dict = field(default_factory=dict)
    encoder_other: list[dict] = field(default_factory=list)
    encoder_hexes: list[dict] = field(default_factory=list)
    encoder_merged: list[dict] = field(default_factory=list)
    actor: dict = field(default_factory=dict)
    critic: dict = field(default_factory=dict)


@dataclass
class State:
    seed: int = -1
    resumes: int = 0
    map_swaps: int = 0  # DEPRECATED
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0
    current_rollout: int = 0
    global_second: int = 0
    current_second: int = 0
    global_episode: int = 0
    current_episode: int = 0

    ep_length_queue: deque = field(default_factory=lambda: deque(maxlen=100))

    ep_rew_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_net_value_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_is_success_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class Args:
    run_id: str
    group_id: str
    run_name: Optional[str] = None
    trial_id: Optional[str] = None
    wandb_project: Optional[str] = None
    resume: bool = False
    overwrite: list = field(default_factory=list)
    notes: Optional[str] = None
    tags: Optional[list] = field(default_factory=list)
    loglevel: int = logging.DEBUG

    agent_load_file: Optional[str] = None
    vsteps_total: int = 0
    seconds_total: int = 0
    rollouts_per_mapchange: int = 0
    rollouts_per_log: int = 1
    rollouts_per_table_log: int = 10
    success_rate_target: Optional[float] = None
    ep_rew_mean_target: Optional[float] = None
    quit_on_target: bool = False
    mapside: str = "both"
    mapmask: str = ""  # DEPRECATED
    randomize_maps: bool = False
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"
    out_dir: str = ""
    out_dir_abs: str = ""  # auto-expanded on start

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])

    lr_schedule: ScheduleArgs = field(default_factory=lambda: ScheduleArgs())  # used if nn-specific lr_schedule["start"] is 0
    lr_schedule_value: ScheduleArgs = field(default_factory=lambda: ScheduleArgs(start=0))
    lr_schedule_policy: ScheduleArgs = field(default_factory=lambda: ScheduleArgs(start=0))
    lr_schedule_distill: ScheduleArgs = field(default_factory=lambda: ScheduleArgs(start=0))
    clip_coef: float = 0.2
    clip_vloss: bool = False
    distill_beta: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 1.2   # not used
    gae_lambda: float = 0.95  # used if nn-specific gae_lambda is 0
    gae_lambda_policy: float = 0
    gae_lambda_value: float = 0
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    num_envs: int = 1
    envmaps: list = field(default_factory=lambda: ["gym/generated/4096/4x1024.vmap"])

    num_minibatches: int = 4  # used if nn-specific num_minibatches is 0
    num_minibatches_distill: int = 0
    num_minibatches_policy: int = 0
    num_minibatches_value: int = 0
    num_steps: int = 128
    stats_buffer_size: int = 100

    update_epochs: int = 4   # used if nn-specific update_epochs is 0
    update_epochs_distill: int = 0
    update_epochs_policy: int = 0
    update_epochs_value: int = 0
    weight_decay: float = 0.0
    target_kl: float = None

    logparams: dict = field(default_factory=dict)
    cfg_file: Optional[str] = None
    seed: int = 42
    skip_wandb_init: bool = False
    skip_wandb_log_code: bool = False

    env: EnvArgs = field(default_factory=lambda: EnvArgs())
    env_version: int = 0
    env_wrappers: list = field(default_factory=list)
    network: NetworkArgs = field(default_factory=lambda: NetworkArgs())

    def __post_init__(self):
        if not isinstance(self.env, EnvArgs):
            self.env = EnvArgs(**self.env)
        if not isinstance(self.lr_schedule, ScheduleArgs):
            self.lr_schedule = ScheduleArgs(**self.lr_schedule)
        if not isinstance(self.lr_schedule_value, ScheduleArgs):
            self.lr_schedule_value = ScheduleArgs(**self.lr_schedule_value)
        if not isinstance(self.lr_schedule_policy, ScheduleArgs):
            self.lr_schedule_policy = ScheduleArgs(**self.lr_schedule_policy)
        if not isinstance(self.lr_schedule_distill, ScheduleArgs):
            self.lr_schedule_distill = ScheduleArgs(**self.lr_schedule_distill)
        if not isinstance(self.network, NetworkArgs):
            self.network = NetworkArgs(**self.network)

        for a in ["distill", "policy", "value"]:
            if getattr(self, f"update_epochs_{a}") == 0:
                setattr(self, f"update_epochs_{a}", self.update_epochs)

            if getattr(self, f"num_minibatches_{a}") == 0:
                setattr(self, f"num_minibatches_{a}", self.num_minibatches)

            if a != "distill" and getattr(self, f"gae_lambda_{a}") == 0:
                setattr(self, f"gae_lambda_{a}", self.gae_lambda)

            if getattr(self, f"lr_schedule_{a}").start == 0:
                setattr(self, f"lr_schedule_{a}", self.lr_schedule)

        common.coerce_dataclass_ints(self)


# 1. Adds 1 hex of padding to the input
#
#               0 0 0 0
#    1 2 3     0 1 2 3 0
#     4 5 6 =>  0 4 5 6 0
#    7 8 9     0 7 8 9 0
#               0 0 0 0
#
# 2. Simulates a Conv2d with kernel_size=2, padding=1
#
#  For the above example (grid of 9 total hexes), this would result in:
#
#  1 => [0, 0, 0, 1, 2, 0, 4]
#  2 => [0, 0, 1, 2, 3, 4, 5]
#  3 => [0, 0, 2, 3, 0, 5, 6]
#  4 => [1, 2, 0, 4, 5, 7, 8]
#  ...
#  9 => [5, 6, 8, 9, 0, 0, 0]
#
# Input: (B, ...) reshapeable to (B, Y, X, E)
# Output: (B, 165, out_channels)
#
class HexConv(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        padded_offsets0 = torch.tensor([-17, -16, -1, 0, 1, 17, 18])
        padded_offsets1 = torch.tensor([-18, -17, -1, 0, 1, 16, 17])
        padded_convinds = torch.zeros(11, 15, 7, dtype=int)

        for y in range(1, 12):
            for x in range(1, 16):
                padded_hexind = y * 17 + x
                padded_offsets = padded_offsets0 if y % 2 == 0 else padded_offsets1
                padded_convinds[y-1, x-1] = padded_offsets + padded_hexind

        self.register_buffer("padded_convinds", padded_convinds.flatten())
        self.fc = nn.LazyLinear(out_features=out_channels)

    def forward(self, x):
        b, _, hexdim = x.shape
        x = x.view(b, 11, 15, -1)
        padded_x = x.new_zeros((b, 13, 17, hexdim))  # +2 hexes in X and Y coords
        padded_x[:, 1:12, 1:16, :] = x
        padded_x = padded_x.view(b, -1, hexdim)
        fc_input = padded_x[:, self.padded_convinds, :].view(b, 165, -1)
        return self.fc(fc_input)


class HexConvResLayer(nn.Module):
    def __init__(self, channels, act={"t": "LeakyReLU"}):
        super().__init__()

        self.act = AgentNN.build_layer(act)
        self.body = nn.Sequential(
            HexConv(channels),
            self.act,
            HexConv(channels),
        )

    def forward(self, x):
        return self.act(self.body(x).add(x))


class HexConvResBlock(nn.Module):
    def __init__(self, channels, depth=1, act={"t": "LeakyReLU"}):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(depth):
            self.layers.append(HexConvResLayer(channels, act))

    def forward(self, x):
        assert x.is_contiguous
        return self.layers(x)


class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


def calcind(y, x):
    if y in range(0, 11) and x in range(0, 15):
        return y * 15 + x
    else:
        return -1


OFFSETS_0 = [   # even rows offsets  /\    /\    /\    /\    /\
    (-1, 1),    # AMOVE_0           /  \  /  \  /  \  /  \  /  \
    (0, 1),     # AMOVE_1          /    \/(11)\/(5) \/(0) \/(6) \
    (1, 1),     # AMOVE_2         ||    ||-1-1||-1 0||-1+1||-1+2||   =row 5
    (1, 0),     # AMOVE_3          \    /\    /\    /\    /\    /\
    (0, -1),    # AMOVE_4           \  /  \  /  \  /  \  /  \  /  \
    (-1, 0),    # AMOVE_5            \/(10)\/(4) \/    \/(1) \/(7) \
    (-1, 2),    # AMOVE_6            ||0 -2||0 -1||0  0||0 +1||0 +2||=row 6
    (0, 2),     # AMOVE_7            /\    /\    /\    /\    /\    /
    (1, 2),     # AMOVE_8           /  \  /  \  /  \  /  \  /  \  /
    (1, -1),    # AMOVE_9          /    \/(9) \/(3) \/(2) \/(8) \/
    (0, -2),    # AMOVE_10        ||    ||+1-1||+1 0||+1+1||+1+2||   =row 7
    (-1, -1),   # AMOVE_11         \    /\    /\    /\    /\    /\
]               # -------------------------------------------------------------
OFFSETS_1 = [   # odd rows offsets   \/(11)\/(5) \/(0) \/(6) \/    \
    (-1, 0),    # AMOVE_0            ||-1-2||-1-1||-1 0||-1+1||    ||=row 8
    (0, 1),     # AMOVE_1            /\    /\    /\    /\    /\    /
    (1, 0),     # AMOVE_2           /  \  /  \  /  \  /  \  /  \  /
    (1, -1),    # AMOVE_3          /(10)\/(4) \/    \/(1) \/(7) \/
    (0, -1),    # AMOVE_4         ||0 -2||0 -1||0  0||0 +1||0 +2||   =row 9
    (-1, -1),   # AMOVE_5          \    /\    /\    /\    /\    /\
    (-1, 1),    # AMOVE_6           \  /  \  /  \  /  \  /  \  /  \
    (0, 2),     # AMOVE_7            \/(9) \/(3) \/(2) \/(8) \/    \
    (1, 1),     # AMOVE_8            ||+1-2||+1-1||+1 0||+1+1||    ||=row 10
    (1, -2),    # AMOVE_9            /\    /\    /\    /\    /\    /
    (0, -2),    # AMOVE_10          /  \  /  \  /  \  /  \  /  \  /
    (-1, -2),   # AMOVE_11         /    \/    \/    \/    \/    \/
]

OFFSETS_1D_0 = [y * 15 + x for y, x in OFFSETS_0]
OFFSETS_1D_1 = [y * 15 + x for y, x in OFFSETS_1]

from vcmi_gym.envs.v12.pyconnector import (
    STATE_SIZE,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    STATE_SIZE_HEXES,
    N_ACTIONS,
    N_HEX_ACTIONS,
    N_NONHEX_ACTIONS,
    GLOBAL_ATTR_MAP,
    GLOBAL_ACT_MAP,
    HEX_ATTR_MAP,
    HEX_ACT_MAP,
)

DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES

assert DIM_OBS == STATE_SIZE

# Helper constants for reconstructing obs/mask
STATE_HEXES_INDEX_START = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
HEX_MASK_INDEX_START = HEX_ATTR_MAP["ACTION_MASK"][1]
HEX_MASK_INDEX_END = HEX_MASK_INDEX_START + HEX_ATTR_MAP["ACTION_MASK"][2]


class ActionData:
    def __init__(self, act0, act0_dist, hex1, hex1_dist, hex2, hex2_dist, action):
        self.act0 = act0
        self.act0_dist = act0_dist
        self.act0_logprob = act0_dist.log_prob(act0)
        self.act0_entropy = act0_dist.entropy()

        self.hex1 = hex1
        self.hex1_dist = hex1_dist
        self.hex1_logprob = hex1_dist.log_prob(hex1)
        self.hex1_entropy = hex1_dist.entropy()

        self.hex2 = hex2
        self.hex2_dist = hex2_dist
        self.hex2_logprob = hex2_dist.log_prob(hex2)
        self.hex2_entropy = hex2_dist.entropy()

        self.action = action
        self.logprob = self.act0_logprob + self.hex1_logprob + self.hex2_logprob
        self.entropy = self.act0_entropy + self.hex1_entropy + self.hex2_entropy


class MainAction(enum.IntEnum):
    WAIT = 0
    MOVE = enum.auto()
    AMOVE = enum.auto()
    SHOOT = enum.auto()


class AgentNN(nn.Module):
    @classmethod
    def build_layer(cls, spec):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")
        layer_cls = getattr(torch.nn, t, None) or globals()[t]

        for k, v in kwargs.items():
            if isinstance(v, dict) and "t" in v:
                kwargs[k] = cls.build_layer(v)

        return layer_cls(**kwargs)

    def __init__(self, network, dim_other, dim_hexes):
        super().__init__()

        self.dim_other = dim_other
        self.dim_hexes = dim_hexes

        self.encoder_other = torch.nn.Sequential()
        self.encoder_hexes = torch.nn.Sequential()
        self.encoder_merged = torch.nn.Sequential()

        # Maps [mainact, hex1, hex2] -> VCMI action
        action_table = torch.zeros([4, 165, 165], dtype=torch.long)
        action_table[MainAction.WAIT, :, :] = GLOBAL_ACT_MAP["WAIT"]  # WAIT is irrelevant of hex

        # Map VCMI action -> [mainact, hex1, hex2]
        # XXX: don't use -1 for hex1 and hex2 even if action has no hex (e.g. WAIT)
        #      This is because a -1 index brakes torch.gather. Using 0 is OK,
        #      since the "logits" for hex1 and/or hex2 for those actions are
        #      replaced by -inf (see CategoricalMasked) which blocks gradient flow.
        inverse_table = torch.zeros([N_ACTIONS, 3], dtype=torch.long)
        inverse_table[GLOBAL_ACT_MAP["WAIT"]] = torch.tensor([MainAction.WAIT, 0, 0])

        amove_hexes = torch.zeros([165, 12], dtype=torch.long)

        for y in range(11):
            o = OFFSETS_0 if y % 2 == 0 else OFFSETS_1
            for x in range(15):
                hex1 = calcind(y, x)
                amove_hexes[hex1, :] = torch.tensor([calcind(y + oy, x + ox) for oy, ox in o])

                move_action = N_NONHEX_ACTIONS + hex1*N_HEX_ACTIONS + HEX_ACT_MAP["MOVE"]
                action_table[MainAction.MOVE, hex1, :] = move_action
                inverse_table[move_action] = torch.tensor([MainAction.MOVE, hex1, 0])

                shoot_action = N_NONHEX_ACTIONS + hex1*N_HEX_ACTIONS + HEX_ACT_MAP["SHOOT"]
                action_table[MainAction.SHOOT, hex1, :] = shoot_action
                inverse_table[shoot_action] = torch.tensor([MainAction.SHOOT, hex1, 0])

                for amove, (oy, ox) in enumerate(o):
                    hex2 = calcind(y + oy, x + ox)
                    amove_hexes[hex1, amove] = hex2
                    if hex2 >= 0:
                        amove_action = N_NONHEX_ACTIONS + hex1*N_HEX_ACTIONS + amove
                        action_table[MainAction.AMOVE, hex1, hex2] = amove_action
                        inverse_table[amove_action] = torch.tensor([MainAction.AMOVE, hex1, hex2])

        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0))
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1)
        self.register_buffer("action_table", action_table)
        self.register_buffer("inverse_table", inverse_table)

        for spec in network.encoder_other:
            layer = AgentNN.build_layer(spec)
            self.encoder_other.append(layer)

        for spec in network.encoder_hexes:
            layer = AgentNN.build_layer(spec)
            self.encoder_hexes.append(layer)

        for spec in network.encoder_merged:
            layer = AgentNN.build_layer(spec)
            self.encoder_merged.append(layer)

        self.actor = AgentNN.build_layer(network.actor)
        self.critic = AgentNN.build_layer(network.critic)

        # Init lazy layers
        with torch.no_grad():
            self.get_actdata_and_value(torch.randn([2, dim_other + dim_hexes]))

        common.layer_init(self.actor, gain=0.01)
        common.layer_init(self.critic, gain=1.0)

    def encode(self, x):
        other, hexes = torch.split(x, [self.dim_other, self.dim_hexes], dim=1)
        z_other = self.encoder_other(other)
        z_hexes = self.encoder_hexes(hexes)
        merged = torch.cat((z_other, z_hexes.flatten(start_dim=1)), dim=1)
        return self.encoder_merged(merged)

    def get_value(self, x):
        return self.critic(self.encode(x))

    def _get_actdata(self, obs, z_merged, action=None, deterministic=False):
        B = obs.shape[0]

        if action is None:
            act0, hex1, hex2 = None, None, None
        else:
            act0, hex1, hex2 = self.inverse_table[action].unbind(1)

        act0_logits, hex1_logits, hex2_logits = self.actor(z_merged).split([len(MainAction), 165, 165], dim=-1)

        # 1. MASK_HEX1 - ie. allowed hex#1 for each action
        mask_hex1 = torch.zeros(B, 4, 165, dtype=torch.bool, device=obs.device)
        hexobs = obs[:, -STATE_SIZE_HEXES:].view([-1, 165, STATE_SIZE_ONE_HEX])

        # 1.1 for 0=WAIT: nothing to do (all zeros)
        # 1.2 for 1=MOVE: Take MOVE bit from obs's action mask
        movemask = hexobs[:, :, HEX_ATTR_MAP["ACTION_MASK"][1] + HEX_ACT_MAP["MOVE"]]
        mask_hex1[:, 1, :] = movemask

        # 1.3 for 2=AMOVE: Take any(AMOVEX) bits from obs's action mask
        amovemask = hexobs[:, :, torch.arange(12) + HEX_ATTR_MAP["ACTION_MASK"][1]].bool()
        mask_hex1[:, 2, :] = amovemask.any(dim=-1)

        # 1.4 for 3=SHOOT: Take SHOOT bit from obs's action mask
        shootmask = hexobs[:, :, HEX_ATTR_MAP["ACTION_MASK"][1] + HEX_ACT_MAP["SHOOT"]]
        mask_hex1[:, 3, :] = shootmask

        # 2. MASK_HEX2 - ie. allowed hex2 for each (action, hex1) combo
        mask_hex2 = torch.zeros([B, 4, 165, 165], dtype=torch.bool, device=obs.device)

        # 2.1 for 0=WAIT: nothing to do (all zeros)
        # 2.2 for 1=MOVE: nothing to do (all zeros)
        # 2.3 for 2=AMOVE: For each SRC hex, create a DST hex mask of allowed hexes
        dest = self.amove_hexes.expand(B, -1, -1)
        valid = amovemask & self.amove_hexes_valid.expand_as(dest)
        b_idx = torch.arange(B, device=obs.device).view(B, 1, 1).expand_as(dest)
        s_idx = torch.arange(165, device=obs.device).view(1, 165, 1).expand_as(dest)

        # Select only valid triples and write
        b_sel = b_idx[valid]
        s_sel = s_idx[valid]
        t_sel = dest[valid]

        mask_hex2[b_sel, 2, s_sel, t_sel] = True

        # 2.4 for 3=SHOOT: nothing to do (all zeros)

        # 3. MASK_ACTION - ie. allowed main action mask
        mask_act0 = torch.zeros(B, 4, dtype=torch.bool, device=obs.device)

        # 0=WAIT
        mask_act0[:, 0] = obs[:, GLOBAL_ATTR_MAP["ACTION_MASK"][1] + GLOBAL_ACT_MAP["WAIT"]]

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_act0[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1)

        # Next, we sample:
        #
        # 1. Sample MAIN ACTION
        dist_act0 = common.CategoricalMasked(logits=act0_logits, mask=mask_act0)

        if act0 is None:
            if deterministic:
                act0 = torch.argmax(dist_act0.probs, dim=1)
            else:
                act0 = dist_act0.sample()

        # 2. Sample HEX1 (with mask corresponding to the main action)
        dist_hex1 = common.CategoricalMasked(
            logits=hex1_logits,
            # Fancy (faster) version of mask_hex1[torch.arange(B, device=device), mainact]
            mask=mask_hex1.gather(1, act0.view(B, 1, 1).expand(B, 1, 165)).squeeze(1)  # (B, 165)
        )

        if hex1 is None:
            if deterministic:
                hex1 = torch.argmax(dist_hex1.probs, dim=1)
            else:
                hex1 = dist_hex1.sample()

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        dist_hex2 = common.CategoricalMasked(
            logits=hex2_logits,
            mask=(
                mask_hex2.gather(1, act0.view(B, 1, 1, 1).expand(B, 1, 165, 165)).
                squeeze(1).  # (B, 165, 165)
                gather(1, hex1.view(B, 1, 1).expand(B, 1, 165)).
                squeeze(1)  # (B, 165)
            )
        )

        if hex2 is None:
            if deterministic:
                hex2 = torch.argmax(dist_hex2.probs, dim=1)
            else:
                hex2 = dist_hex2.sample()

        if action is None:
            action = self.action_table[act0, hex1, hex2]

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    def get_actdata(self, obs, action=None, deterministic=False):
        z_merged = self.encode(obs)
        return self._get_actdata(obs, z_merged, action, deterministic)

    def get_actdata_and_value(self, obs, action=None, deterministic=False):
        z_merged = self.encode(obs)
        actdata = self._get_actdata(obs, z_merged, action, deterministic)
        value = self.critic(z_merged)
        return actdata, value

    # Inference (deterministic)
    def predict(self, b_obs):
        with torch.no_grad():
            b_obs = torch.as_tensor(b_obs)

            # Return unbatched action if input was unbatched
            if b_obs.ndim == 1:
                b_obs = b_obs.unsqueeze(dim=0)
                b_actdata = self.get_actdata(b_obs, deterministic=True)
                return b_actdata.v[0].cpu().item()
            else:
                b_actdata = self.get_actdata(b_obs, deterministic=True)
                return b_actdata.v.cpu().numpy()


class Agent(nn.Module):
    """
    Store a "clean" version of the agent: create a fresh one and copy the attrs
    manually (for nn's and optimizers - copy their states).
    This prevents issues if the agent contains wandb hooks at save time.
    """
    @staticmethod
    def save(agent, agent_file):
        print("Saving agent to %s" % agent_file)
        if not os.path.isabs(agent_file):
            warnings.warn(
                f"path {agent_file} is not absolute!"
                " If VCMI is started in a thread, the current directory is changed."
                f" CWD: {os.getcwd()}"
            )

        attrs = ["args", "dim_other", "dim_hexes", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN_value.load_state_dict(agent.NN_value.state_dict(), strict=True)
        clean_agent.NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)
        clean_agent.optimizer_value.load_state_dict(agent.optimizer_value.state_dict())
        clean_agent.optimizer_policy.load_state_dict(agent.optimizer_policy.state_dict())
        clean_agent.optimizer_distill.load_state_dict(agent.optimizer_distill.state_dict())
        torch.save(clean_agent, agent_file)

    @staticmethod
    def jsave(agent, jagent_file):
        print("Saving JIT agent to %s" % jagent_file)
        attrs = ["args", "dim_other", "dim_hexes", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN_value.load_state_dict(agent.NN_value.state_dict(), strict=True)
        clean_agent.NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)
        clean_agent.optimizer_value.load_state_dict(agent.optimizer_value.state_dict())
        clean_agent.optimizer_policy.load_state_dict(agent.optimizer_policy.state_dict())
        clean_agent.optimizer_distill.load_state_dict(agent.optimizer_distill.state_dict())
        jagent = JitAgent()
        jagent.env_version = clean_agent.env_version
        jagent.dim_other = clean_agent.dim_other
        jagent.dim_hexes = clean_agent.dim_hexes

        # v3+
        jagent.encoder_policy = clean_agent.NN_policy.encoder
        jagent.encoder_value = clean_agent.NN_value.encoder

        # common
        jagent.actor = clean_agent.NN_policy.actor
        jagent.critic = clean_agent.NN_value.critic

        jagent_optimized = optimize_for_mobile(torch.jit.script(jagent), preserved_methods=["get_version", "predict", "get_value"])
        jagent_optimized._save_for_lite_interpreter(jagent_file)

    @staticmethod
    def load(agent_file, device="cpu"):
        print("Loading agent from %s (device: %s)" % (agent_file, device))
        return torch.load(agent_file, map_location=device, weights_only=False)

    def __init__(self, args, dim_other, dim_hexes, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self._optimizer_state = None  # needed for save/load
        self.dim_other = dim_other  # needed for save/load
        self.dim_hexes = dim_hexes  # needed for save/load
        self.NN_value = AgentNN(args.network, dim_other, dim_hexes)
        self.NN_policy = AgentNN(args.network, dim_other, dim_hexes)
        self.NN_value.to(device)
        self.NN_policy.to(device)
        self.optimizer_value = torch.optim.AdamW(self.NN_value.parameters(), eps=1e-5)
        self.optimizer_policy = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.optimizer_distill = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.predict = self.NN_policy.predict
        self.state = state or State()


class JitAgent(nn.Module):
    """ TorchScript version of Agent (inference only) """

    def __init__(self):
        super().__init__()
        # XXX: these are overwritten after object is initialized
        self.obs_splitter = nn.Identity()
        self.encoder_policy = nn.Identity()
        self.encoder_value = nn.Identity()
        self.actor = nn.Identity()
        self.critic = nn.Identity()
        self.env_version = 0

    # Inference
    # XXX: attention is not handled here
    @torch.jit.export
    def predict(self, obs, deterministic: bool = False) -> int:
        b_obs = obs.unsqueeze(dim=0)
        encoded = self.encoder(b_obs)
        action_logits = self.actor(encoded)
        probs = self.categorical_masked(logits0=action_logits)

        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = self.sample(probs, action_logits)

        return action.int().item()

    @torch.jit.export
    def forward(self, obs) -> torch.Tensor:
        b_obs = obs.unsqueeze(dim=0)
        encoded = self.encoder(b_obs)

        return torch.cat(dim=1, tensors=(self.actor(encoded), self.critic(encoded)))

    @torch.jit.export
    def get_value(self, obs) -> float:
        b_obs = obs.unsqueeze(dim=0)
        encoded = self.encoder(b_obs)
        value = self.critic(encoded)
        return value.float().item()

    @torch.jit.export
    def get_version(self) -> int:
        return self.env_version

    # Implement SerializableCategoricalMasked as a function
    # (lite interpreted does not support instantiating the class)
    @torch.jit.export
    def categorical_masked(self, logits0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_value = torch.tensor(-((2 - 2**-23) * 2**127), dtype=logits0.dtype)

        # logits
        logits1 = torch.where(mask, logits0, mask_value)
        logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @torch.jit.export
    def sample(self, probs: torch.Tensor, action_logits: torch.Tensor) -> torch.Tensor:
        num_events = action_logits.size()[-1]
        probs_2d = probs.reshape(-1, num_events)
        samples_2d = torch.multinomial(probs_2d, 1, True).T
        batch_shape = action_logits.size()[:-1]
        return samples_2d.reshape(batch_shape)


def compute_advantages(rewards, dones, values, next_done, next_value, gamma, gae_lambda):
    total_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(total_steps)):
        if t == total_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def main(args, agent=None):
    args.out_dir = args.out_dir_template.format(group_id=args.group_id, run_id=args.run_id)
    args.out_dir_abs = os.path.abspath(args.out_dir)

    LOG = logging.getLogger("mppo_dna")
    LOG.setLevel(args.loglevel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert isinstance(args, Args)

    if agent is None:
        agent, args = common.maybe_resume(Agent, args, device_name=device.type)

    # update out_dir (may have changed after load)
    args.out_dir = args.out_dir_template.format(group_id=args.group_id, run_id=args.run_id)
    args.out_dir_abs = os.path.abspath(args.out_dir)

    if args.seconds_total:
        assert not args.vsteps_total, "cannot have both vsteps_total and seconds_total"
        rollouts_total = 0
    else:
        rollouts_total = args.vsteps_total // args.num_steps

    # Re-initialize to prevent errors from newly introduced args when loading/resuming
    # TODO: handle removed args
    args = Args(**vars(args))

    if os.getenv("NO_WANDB") == "true":
        args.wandb_project = None

    printargs = asdict(args).copy()

    # Logger
    if not any(LOG.handlers):
        formatter = logging.Formatter(f"-- %(asctime)s %(levelname)s [{args.group_id}/{args.run_id}] %(message)s")
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = None
        loghandler = logging.StreamHandler()
        loghandler.setFormatter(formatter)
        LOG.addHandler(loghandler)

    LOG.info("Args: %s" % printargs)

    out_dir = args.out_dir_template.format(seed=args.seed, group_id=args.group_id, run_id=args.run_id)
    LOG.info("Out dir: %s" % out_dir)

    num_envs = args.num_envs

    lr_schedule_fn_value = common.schedule_fn(args.lr_schedule_value)
    lr_schedule_fn_policy = common.schedule_fn(args.lr_schedule_policy)
    lr_schedule_fn_distill = common.schedule_fn(args.lr_schedule_distill)

    batch_size_policy = int(num_envs * args.num_steps)
    batch_size_value = int(num_envs * args.num_steps)
    batch_size_distill = int(num_envs * args.num_steps)
    minibatch_size_policy = int(batch_size_policy // args.num_minibatches_policy)
    minibatch_size_value = int(batch_size_value // args.num_minibatches_value)
    minibatch_size_distill = int(batch_size_distill // args.num_minibatches_distill)

    save_ts = None
    permasave_ts = None

    if args.agent_load_file and not agent:
        f = args.agent_load_file
        agent = Agent.load(f, device=device)
        agent.args = args
        agent.state.current_timestep = 0
        agent.state.current_vstep = 0
        agent.state.current_rollout = 0
        agent.state.current_second = 0
        agent.state.current_episode = 0

        # backup = "%s/loaded-%s" % (os.path.dirname(f), os.path.basename(f))
        # with open(f, 'rb') as fsrc:
        #     with open(backup, 'wb') as fdst:
        #         shutil.copyfileobj(fsrc, fdst)
        #         LOG.info("Wrote backup %s" % backup)

    common.validate_tags(args.tags)

    printargs = asdict(args).copy()
    LOG.info("Args (after loading): %s" % printargs)

    seed = args.seed

    # XXX: seed logic is buggy, do not use
    #      (this seed was never used to re-play trainings anyway)
    #      Just generate a random non-0 seed every time

    # if args.seed:
    #     seed = args.seed
    # elif agent and agent.state.seed:
    #     seed = agent.state.seed
    # else:

    # XXX: make sure the new seed is never 0
    # while seed == 0:
    #     seed = np.random.randint(2**31 - 1)

    wrappers = args.env_wrappers

    assert args.env_version == 12
    from vcmi_gym import VcmiEnv_v12 as VcmiEnv

    obs_space = VcmiEnv.OBSERVATION_SPACE
    act_space = VcmiEnv.ACTION_SPACE

    if agent is None:
        # TODO: robust mechanism ensuring these don't get mixed up
        dim_other = VcmiEnv.STATE_SIZE_GLOBAL + 2*VcmiEnv.STATE_SIZE_ONE_PLAYER
        dim_hexes = VcmiEnv.STATE_SIZE_HEXES
        assert VcmiEnv.STATE_SIZE == dim_other + dim_hexes
        agent = Agent(args, dim_other, dim_hexes, device=device)

    # TRY NOT TO MODIFY: seeding
    LOG.info("RNG master seed: %s" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    timers = {name: Timer(name) for name in ["all", "env", "forward", "backward", "loss"]}

    # this one is separate and not logged to wandb
    last_eval_timer = Timer("last_eval")
    last_eval_timer.start()

    try:
        seeds = [np.random.randint(2**31) for i in range(args.num_envs)]
        envs = common.create_venv(VcmiEnv, args, seeds, sync=(args.num_envs == 1))

        eval_args = args.__class__(**vars(args))
        eval_args.envmaps = ["gym/generated/evaluation/8x512.vmap"]
        eval_args.opponent_sbm_probs = [0, 1, 0]  # BattleAI
        eval_args.env.random_stack_chance = 0
        eval_args.env.num_envs = 1
        eval_venv = common.create_venv(VcmiEnv, eval_args, seeds, sync=False)

        agent.state.seed = seed

        if args.wandb_project:
            import wandb
            common.setup_wandb(args, agent, __file__)

            # For wandb.log, commit=True by default
            # for wandb_log, commit=False by default
            def wandb_log(*args, **kwargs):
                wandb.log(*args, **dict({"commit": False}, **kwargs))
        else:
            def wandb_log(*args, **kwargs):
                pass

        common.log_params(args, wandb_log)

        if args.resume:
            agent.state.resumes += 1
            wandb_log({"global/resumes": agent.state.resumes})

        # print("Agent state: %s" % asdict(agent.state))

        assert act_space.shape == ()

        # attn = agent.NN.attention is not None
        attn = False

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, num_envs) + obs_space["observation"].shape).to(device)
        actions = torch.zeros((args.num_steps, num_envs) + act_space.shape, dtype=torch.int64).to(device)
        logprobs = torch.zeros((args.num_steps, num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, num_envs)).to(device)
        dones = torch.zeros((args.num_steps, num_envs)).to(device)
        values = torch.zeros((args.num_steps, num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(next_obs, device=device)
        next_done = torch.zeros(num_envs, device=device)

        progress = 0
        map_rollouts = 0
        start_time = time.time()
        global_start_second = agent.state.global_second

        timers["all"].start()

        while progress < 1:
            [v.reset(start=(k == "all")) for k, v in timers.items()]

            if args.vsteps_total:
                progress = agent.state.current_vstep / args.vsteps_total
            elif args.seconds_total:
                progress = agent.state.current_second / args.seconds_total
            else:
                progress = 0

            agent.optimizer_value.param_groups[0]["lr"] = lr_schedule_fn_value(progress)
            agent.optimizer_policy.param_groups[0]["lr"] = lr_schedule_fn_policy(progress)
            agent.optimizer_distill.param_groups[0]["lr"] = lr_schedule_fn_distill(progress)

            episode_count = 0

            # XXX: eval during experience collection
            agent.eval()

            if agent.state.current_rollout == 0 or last_eval_timer.peek() > 1800:
                last_eval_timer.reset(start=True)
                eval_log = {
                    "eval/ep_rew_mean": 0,
                    "eval/ep_len_mean": 0,
                    "eval/ep_value_mean": 0,
                    "eval/ep_is_success_mean": 0,
                    "eval/num_episodes": 0,
                }

                with torch.no_grad():
                    t = lambda x: torch.as_tensor(x, device=device)
                    LOG.info("Evaluating...")
                    e_obs, _ = eval_venv.reset()
                    for vstep in range(0, 1000):
                        e_obs = t(e_obs)
                        e_actdata = agent.NN_policy.get_actdata(e_obs)
                        e_obs, e_rew, e_term, e_trunc, e_info = eval_venv.step(e_actdata.action.cpu().numpy())

                        # See notes/gym_vector.txt
                        if "_final_info" in e_info:
                            done_ids = np.flatnonzero(e_info["_final_info"])
                            final_info = e_info["final_info"]
                            eval_log["eval/ep_rew_mean"] += float(sum(final_info["episode"]["r"][done_ids]))
                            eval_log["eval/ep_len_mean"] += float(sum(final_info["episode"]["l"][done_ids]))
                            eval_log["eval/ep_value_mean"] += float(sum(final_info["net_value"][done_ids]))
                            eval_log["eval/ep_is_success_mean"] += float(sum(final_info["is_success"][done_ids]))
                            eval_log["eval/num_episodes"] += len(done_ids)

                if eval_log["eval/num_episodes"] > 0:
                    eval_log["eval/ep_rew_mean"] /= eval_log["eval/num_episodes"]
                    eval_log["eval/ep_len_mean"] /= eval_log["eval/num_episodes"]
                    eval_log["eval/ep_value_mean"] /= eval_log["eval/num_episodes"]
                    eval_log["eval/ep_is_success_mean"] /= eval_log["eval/num_episodes"]

                LOG.info("Done evaluating: %s" % eval_log)
                wandb_log(eval_log, commit=True)

            # tstart = time.time()
            for step in range(0, args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    with timers["forward"]:
                        actdata = agent.NN_policy.get_actdata(next_obs)
                        value = agent.NN_value.get_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = actdata.action
                logprobs[step] = actdata.logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                with timers["env"]:
                    next_obs, reward, terminations, truncations, infos = envs.step(actdata.action.cpu().numpy())

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs, device=device)
                next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)

                # XXX SIMO: SB3 does bootstrapping for truncated episodes here
                # https://github.com/DLR-RM/stable-baselines3/pull/658

                # See notes/gym_vector.txt
                if "_final_info" in infos:
                    done_ids = np.flatnonzero(infos["_final_info"])
                    final_infos = infos["final_info"]
                    agent.state.ep_rew_queue.extend(final_infos["episode"]["r"][done_ids])
                    agent.state.ep_length_queue.extend(final_infos["episode"]["l"][done_ids])
                    agent.state.ep_net_value_queue.extend(final_infos["net_value"][done_ids])
                    agent.state.ep_is_success_queue.extend(final_infos["is_success"][done_ids])
                    agent.state.current_episode += 1
                    agent.state.global_episode += 1
                    episode_count += 1

                agent.state.current_vstep += 1
                agent.state.current_timestep += num_envs
                agent.state.global_timestep += num_envs
                agent.state.current_second = int(time.time() - start_time)
                agent.state.global_second = global_start_second + agent.state.current_second

            # print("SAMPLE TIME: %.2f" % (time.time() - tstart))
            # tstart = time.time()

            # bootstrap value if not done
            with torch.no_grad():
                with timers["forward"]:
                    next_value = agent.NN_value.get_value(next_obs).reshape(1, -1)

                advantages, _ = compute_advantages(
                    rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_policy
                )
                _, returns = compute_advantages(rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_value)

            # flatten the batch
            b_obs = obs.flatten(end_dim=1)
            b_logprobs = logprobs.flatten(end_dim=1)
            b_actions = actions.flatten(end_dim=1)
            b_advantages = advantages.flatten(end_dim=1)
            b_returns = returns.flatten(end_dim=1)
            b_values = values.flatten(end_dim=1)

            # Policy network optimization
            b_inds = np.arange(batch_size_policy)
            clipfracs = []

            agent.train()
            for epoch in range(args.update_epochs_policy):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_policy, minibatch_size_policy):
                    end = start + minibatch_size_policy
                    mb_inds = b_inds[start:end]

                    with timers["forward"]:
                        actdata = agent.NN_policy.get_actdata(b_obs[mb_inds], action=b_actions[mb_inds])

                    with timers["loss"]:
                        logratio = actdata.logprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        entropy_loss = actdata.entropy.mean()
                        policy_loss = pg_loss - args.ent_coef * entropy_loss

                    with timers["backward"]:
                        agent.optimizer_policy.zero_grad()
                        policy_loss.backward()
                        nn.utils.clip_grad_norm_(agent.NN_policy.parameters(), args.max_grad_norm)
                        agent.optimizer_policy.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # Value network optimization
            for epoch in range(args.update_epochs_value):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_value, minibatch_size_value):
                    end = start + minibatch_size_value
                    mb_inds = b_inds[start:end]

                    with timers["forward"]:
                        newvalue = agent.NN_value.get_value(b_obs[mb_inds])
                    newvalue = newvalue.view(-1)

                    with timers["loss"]:
                        # Value loss
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            # XXX: SIMO: SB3 does not multiply by 0.5 here
                            #            (ie. SB3's vf_coef is essentially x2)
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    with timers["backward"]:
                        agent.optimizer_value.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.NN_value.parameters(), args.max_grad_norm)
                        agent.optimizer_value.step()

            # Value network to policy network distillation
            agent.NN_policy.zero_grad(True)  # don't clone gradients
            old_NN_policy = copy.deepcopy(agent.NN_policy).to(device)
            old_NN_policy.eval()
            for epoch in range(args.update_epochs_distill):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_distill, minibatch_size_distill):
                    end = start + minibatch_size_distill
                    mb_inds = b_inds[start:end]
                    # Compute policy and value targets
                    with torch.no_grad():
                        with timers["forward"]:
                            old_actdata = old_NN_policy.get_actdata(b_obs[mb_inds])
                            value_target = agent.NN_value.get_value(b_obs[mb_inds])

                    # XXX: must pass action=<old_action> to ensure masks for hex1 and hex2 are the same
                    #     (if actions differ, masks will differ and KLD will become NaN)
                    new_actdata, new_value = agent.NN_policy.get_actdata_and_value(b_obs[mb_inds], action=old_actdata.action)

                    with timers["loss"]:
                        # Distillation loss
                        policy_kl_loss = (
                            kld(old_actdata.act0_dist, new_actdata.act0_dist)
                            + kld(old_actdata.hex1_dist, new_actdata.hex1_dist)
                            + kld(old_actdata.hex2_dist, new_actdata.hex2_dist)
                        ).mean()

                        value_loss = 0.5 * (new_value.view(-1) - value_target).square().mean()
                        distill_loss = value_loss + args.distill_beta * policy_kl_loss

                    with timers["backward"]:
                        agent.optimizer_distill.zero_grad()
                        distill_loss.backward()
                        nn.utils.clip_grad_norm_(agent.NN_policy.parameters(), args.max_grad_norm)
                        agent.optimizer_distill.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            ep_rew_mean = common.safe_mean(agent.state.ep_rew_queue)
            ep_value_mean = common.safe_mean(agent.state.ep_net_value_queue)
            ep_is_success_mean = common.safe_mean(agent.state.ep_is_success_queue)
            ep_length_mean = common.safe_mean(agent.state.ep_length_queue)

            if episode_count > 0:
                assert ep_rew_mean is not np.nan
                assert ep_value_mean is not np.nan
                assert ep_is_success_mean is not np.nan
                agent.state.rollout_rew_queue_100.append(ep_rew_mean)
                agent.state.rollout_rew_queue_1000.append(ep_rew_mean)
                agent.state.rollout_net_value_queue_100.append(ep_value_mean)
                agent.state.rollout_net_value_queue_1000.append(ep_value_mean)
                agent.state.rollout_is_success_queue_100.append(ep_is_success_mean)
                agent.state.rollout_is_success_queue_1000.append(ep_is_success_mean)

            wandb_log({"params/policy_learning_rate": agent.optimizer_policy.param_groups[0]["lr"]})
            wandb_log({"params/value_learning_rate": agent.optimizer_value.param_groups[0]["lr"]})
            wandb_log({"params/distill_learning_rate": agent.optimizer_distill.param_groups[0]["lr"]})
            wandb_log({"losses/value_loss": v_loss.item()})
            wandb_log({"losses/policy_loss": pg_loss.item()})
            wandb_log({"losses/distill_loss": distill_loss.item()})
            wandb_log({"losses/entropy": entropy_loss.item()})
            wandb_log({"losses/old_approx_kl": old_approx_kl.item()})
            wandb_log({"losses/approx_kl": approx_kl.item()})
            wandb_log({"losses/clipfrac": np.mean(clipfracs)})
            wandb_log({"losses/explained_variance": explained_var})
            wandb_log({"rollout/ep_count": episode_count})

            if episode_count > 0:
                assert ep_rew_mean is not np.nan
                assert ep_value_mean is not np.nan
                assert ep_is_success_mean is not np.nan
                agent.state.rollout_rew_queue_100.append(ep_rew_mean)
                agent.state.rollout_rew_queue_1000.append(ep_rew_mean)
                agent.state.rollout_net_value_queue_100.append(ep_value_mean)
                agent.state.rollout_net_value_queue_1000.append(ep_value_mean)
                agent.state.rollout_is_success_queue_100.append(ep_is_success_mean)
                agent.state.rollout_is_success_queue_1000.append(ep_is_success_mean)

                wandb_log({"rollout/ep_rew_mean": ep_rew_mean})
                wandb_log({"rollout/ep_value_mean": ep_value_mean})
                wandb_log({"rollout/ep_success_rate": ep_is_success_mean})
                wandb_log({"rollout/ep_len_mean": ep_length_mean})

            wandb_log({"rollout100/ep_value_mean": common.safe_mean(agent.state.rollout_net_value_queue_100)})
            wandb_log({"rollout1000/ep_value_mean": common.safe_mean(agent.state.rollout_net_value_queue_1000)})
            wandb_log({"rollout100/ep_rew_mean": common.safe_mean(agent.state.rollout_rew_queue_100)})
            wandb_log({"rollout1000/ep_rew_mean": common.safe_mean(agent.state.rollout_rew_queue_1000)})
            wandb_log({"rollout100/ep_success_rate": common.safe_mean(agent.state.rollout_is_success_queue_100)})
            wandb_log({"rollout1000/ep_success_rate": common.safe_mean(agent.state.rollout_is_success_queue_1000)})
            wandb_log({"global/num_rollouts": agent.state.current_rollout})
            wandb_log({"global/num_timesteps": agent.state.current_timestep})
            wandb_log({"global/num_seconds": agent.state.current_second})
            wandb_log({"global/num_episode": agent.state.current_episode})

            if rollouts_total:
                wandb_log({"global/progress": progress})

            # XXX: maybe use a less volatile metric here (eg. 100 or 1000-average)
            if args.success_rate_target and ep_is_success_mean >= args.success_rate_target:
                LOG.info("Early stopping due to: success rate > %.2f (%.2f)" % (args.success_rate_target, ep_is_success_mean))

                if args.quit_on_target:
                    # XXX: break?
                    sys.exit(0)
                else:
                    raise Exception("Not implemented: map change on target")

            # XXX: maybe use a less volatile metric here (eg. 100 or 1000-average)
            if args.ep_rew_mean_target and ep_rew_mean >= args.ep_rew_mean_target:
                LOG.info("Early stopping due to: ep_rew_mean > %.2f (%.2f)" % (args.ep_rew_mean_target, ep_rew_mean))

                if args.quit_on_target:
                    # XXX: break?
                    sys.exit(0)
                else:
                    raise Exception("Not implemented: map change on target")

            if agent.state.current_rollout > 0 and agent.state.current_rollout % args.rollouts_per_log == 0:
                tstats = timer_stats(timers)

                wandb_log(tstats)
                wandb_log({
                    "global/global_num_timesteps": agent.state.global_timestep,
                    "global/global_num_seconds": agent.state.global_second
                }, commit=True)  # commit on final log line

                for k, v in tstats.items():
                    print("%-20s: %.4f " % (k, v))

                LOG.debug("rollout=%d vstep=%d rew=%.2f net_value=%.2f is_success=%.2f losses=%.1f|%.1f|%.1f" % (
                    agent.state.current_rollout,
                    agent.state.current_vstep,
                    ep_rew_mean,
                    ep_value_mean,
                    ep_is_success_mean,
                    value_loss.item(),
                    policy_loss.item(),
                    distill_loss.item()
                ))

            agent.state.current_rollout += 1
            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent)
            # print("TRAIN TIME: %.2f" % (time.time() - tstart))

    finally:
        common.maybe_save(0, 10e9, args, agent)
        if "envs" in locals():
            envs.close()

    # Needed by PBT to save model after iteration ends
    # XXX: limit returned mean reward to only the rollouts in this iteration
    # XXX: but no more than the last 300 rollouts (esp. if training vs BattleAI)
    ret_rew = common.safe_mean(list(agent.state.rollout_rew_queue_1000)[-min(300, agent.state.current_rollout):])
    ret_value = common.safe_mean(list(agent.state.rollout_net_value_queue_1000)[-min(300, agent.state.current_rollout):])

    wandb_log({
        "trial/ep_rew_mean": ret_rew,
        "trial/ep_value_mean": ret_value,
        "trial/num_rollouts": agent.state.current_rollout,
    }, commit=True)  # commit on final log line

    return (agent, ret_rew, ret_value)


def debug_args():
    return Args(
        run_id="mppo_dna-test",
        group_id="mppo_dna-test",
        run_name=None,
        loglevel=logging.DEBUG,
        trial_id=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="/Users/simo/Projects/vcmi-gym/agent-T4.pt",
        agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_mapchange=0,
        rollouts_per_log=1,
        rollouts_per_table_log=100000,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="defender",
        save_every=2000000000,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=0,
        out_dir_template="data/mppo_dna-test/mppo_dna-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.001),
        lr_schedule_value=ScheduleArgs(mode="const", start=0),
        lr_schedule_policy=ScheduleArgs(mode="const", start=0),
        lr_schedule_distill=ScheduleArgs(mode="const", start=0),
        num_envs=1,
        num_steps=256,
        num_minibatches=4,
        update_epochs=2,
        gae_lambda=0.9,
        gamma=0.8,
        gae_lambda_policy=0,
        gae_lambda_value=0,
        num_minibatches_value=0,
        num_minibatches_policy=0,
        num_minibatches_distill=0,
        update_epochs_value=0,
        update_epochs_policy=0,
        update_epochs_distill=0,
        norm_adv=True,
        clip_coef=0.3,
        clip_vloss=True,
        ent_coef=0.01,
        max_grad_norm=0.5,
        distill_beta=1.0,
        target_kl=None,
        logparams={},
        cfg_file=None,
        seed=42,
        skip_wandb_init=False,
        skip_wandb_log_code=False,
        # envmaps=["gym/A1.vmap"],
        envmaps=["gym/generated/4096/4x1024.vmap"],
        env=EnvArgs(
            max_steps=500,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            random_heroes=1,
            random_obstacles=0,
            random_terrain_chance=0,
            tight_formation_chance=0,
            town_chance=0,
            random_stack_chance=0,
            warmachine_chance=0,
            mana_min=0,
            mana_max=0,
            reward_step_fixed=-0.01,
            reward_dmg_mult=0.01,
            reward_term_mult=0.01,
            reward_relval_mult=0.01,
            swap_sides=0,
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
        ),
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        env_wrappers=[
            dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyObservationSpaceWrapper"),
            dict(module="gymnasium.wrappers", cls="RecordEpisodeStatistics")
        ],
        env_version=12,
        network={
            "encoder_other": [
                # => (B, 26)
                {"t": "LazyLinear", "out_features": 32},
                {"t": "LeakyReLU"},
                # => (B, 64)
            ],
            "encoder_hexes": [
                # => (B, 165*H)
                dict(t="Unflatten", dim=1, unflattened_size=[165, 170]),
                # => (B, 165, H)

                # #
                # # HexConv (variant A: classic conv)
                # #
                # {"t": "HexConv", "out_channels": 64},
                # {"t": "LeakyReLU"},
                # # => (B, 165, 64)
                # {"t": "HexConv", "out_channels": 64},
                # {"t": "LeakyReLU"},
                # # => (B, 165, 16)

                #
                # HexConv (variant B: residual conv)
                #
                {"t": "HexConvResBlock", "channels": 170, "depth": 3},

                #
                # HexConv COMMON
                #
                {"t": "LazyLinear", "out_features": 32},
                {"t": "LeakyReLU"},
                # => (B, 165, 16)
            ],
            "encoder_merged": [
                {"t": "LazyLinear", "out_features": 1024},
                {"t": "LeakyReLU"},
                # => (B, 1024)
            ],
            "actor": {"t": "LazyLinear", "out_features": len(MainAction)+165+165},
            "critic": {"t": "LazyLinear", "out_features": 1}
        }
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
