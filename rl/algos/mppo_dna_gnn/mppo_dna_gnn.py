import os
import sys
import random
import logging
import json
import string
import argparse
import threading
import contextlib
import gymnasium as gym
import enum
import copy
import importlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence as kld
from torch_geometric.data import HeteroData, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn as gnn


from rl.algos.common import CategoricalMasked
from rl.world.util.structured_logger import StructuredLogger
from rl.world.util.persistence import load_checkpoint, save_checkpoint
from rl.world.util.wandb import setup_wandb
from rl.world.util.timer import Timer
from rl.world.util.misc import dig, safe_mean, timer_stats

from vcmi_gym.envs.v13.vcmi_env import VcmiEnv
from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper

from vcmi_gym.envs.v13.pyconnector import (
    STATE_SIZE,
    STATE_SIZE_ONE_HEX,
    STATE_SIZE_HEXES,
    N_ACTIONS,
    N_HEX_ACTIONS,
    N_NONHEX_ACTIONS,
    GLOBAL_ATTR_MAP,
    GLOBAL_ACT_MAP,
    HEX_ATTR_MAP,
    HEX_ACT_MAP,
    LINK_TYPES,
)


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

if os.getenv("PYDEBUG", None) == "1":
    def excepthook(exc_type, exc_value, tb):
        import ipdb
        ipdb.post_mortem(tb)

    sys.excepthook = excepthook


@dataclass
class State:
    seed: int = -1
    resumes: int = 0
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0
    current_rollout: int = 0
    global_second: int = 0
    current_second: int = 0
    global_episode: int = 0
    current_episode: int = 0

    ep_rew_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_net_value_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_is_success_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    def to_json(self):
        j = {}
        for k, v in asdict(self).items():
            j[k] = list(v) if isinstance(v, deque) else v
        return json.dumps(j, indent=4, sort_keys=False)

    def from_json(self, j):
        for k, v in json.loads(j).items():
            attr = getattr(self, k)
            v = deque(v, maxlen=attr.maxlen) if isinstance(attr, deque) else v
            setattr(self, k, v)


def to_hdata(obs, done, links):
    device = obs.device
    res = HeteroData()
    res.obs = obs.unsqueeze(0)
    res.done = done.unsqueeze(0).float()
    res.value = torch.tensor(0., device=device)
    res.action = torch.tensor(0, device=device)
    res.reward = torch.tensor(0., device=device)
    res.logprob = torch.tensor(0., device=device)
    res.advantage = torch.tensor(0., device=device)
    res.ep_return = torch.tensor(0., device=device)

    res["hex"].x = obs[:STATE_SIZE_HEXES].view(165, STATE_SIZE_ONE_HEX)
    for lt in LINK_TYPES.keys():
        res["hex", lt, "hex"].edge_index = torch.as_tensor(links[lt]["index"], device=device)
        res["hex", lt, "hex"].edge_attr = torch.as_tensor(links[lt]["attrs"], device=device)

    return res


# b_obs: torch.tensor of shape (B, STATE_SIZE)
# tuple_links: tuple of B dicts, where each dict is a single obs's "links"
def to_hdata_batch(b_obs, b_done, tuple_links):
    b_hdatas = []
    for obs, done, links in zip(b_obs, b_done, tuple_links):
        b_hdatas.append(to_hdata(obs, done, links))
    # XXX: this concatenates along the first dim
    # i.e. stacking two (165, STATE_SIZE_ONE_HEX)
    #       gives  (330, STATE_SIZE_ONE_HEX)
    #       not sure if that's required for GNN to work?
    #       but it breaks my encode() which uses torch.split()
    return Batch.from_data_list(b_hdatas)


class Storage:
    def __init__(self, venv, num_vsteps, device):
        v = venv.num_envs
        self.rollout_buffer = []  # contains Batch() objects
        self.v_next_hdata = to_hdata_batch(
            torch.as_tensor(venv.reset()[0], device=device),
            torch.zeros(v, device=device),
            venv.call("links"),
        )

        # Needed for the GAE computation (to prevent spaghetti)
        # and for explained_var computation
        self.bv_dones = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_values = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_rewards = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_advantages = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_returns = torch.zeros((num_vsteps, venv.num_envs), device=device)


@dataclass
class TrainStats:
    value_loss: float
    policy_loss: float
    entropy_loss: float
    distill_loss: float
    approx_kl: float
    clipfrac: float
    explained_var: float


@dataclass
class SampleStats:
    ep_rew_mean: float = 0.0
    ep_len_mean: float = 0.0
    ep_value_mean: float = 0.0
    ep_is_success_mean: float = 0.0
    num_episodes: int = 0


# Aggregated version of SampleStats with a handle
# to the individual SampleStats variants.
@dataclass
class MultiStats(SampleStats):
    variants: dict = field(default_factory=dict)

    def add(self, name, stats):
        self.variants[name] = stats

        if stats.num_episodes == 0:
            print("WARNING: adding SampleStats with num_episodes=0")

        # Don't let "empty" samples influence the mean values EXCEPT for num_episodes
        self.num_episodes = safe_mean([v.num_episodes for v in self.variants.values()])
        self.ep_rew_mean = safe_mean([v.ep_rew_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_len_mean = safe_mean([v.ep_len_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_value_mean = safe_mean([v.ep_value_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_is_success_mean = safe_mean([v.ep_is_success_mean for v in self.variants.values() if v.num_episodes > 0])


class MainAction(enum.IntEnum):
    WAIT = 0
    MOVE = enum.auto()
    AMOVE = enum.auto()
    SHOOT = enum.auto()


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


class Model(nn.Module):
    @staticmethod
    def build_action_tables():
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
        calcind = lambda y, x: y*15 + x if (y in range(0, 11) and x in range(0, 15)) else -1

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

        return action_table, inverse_table, amove_hexes

    def __init__(self, config):
        super().__init__()

        self.dim_other = STATE_SIZE - STATE_SIZE_HEXES
        self.dim_hexes = STATE_SIZE_HEXES

        action_table, inverse_table, amove_hexes = self.__class__.build_action_tables()

        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0))
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1)
        self.register_buffer("action_table", action_table)
        self.register_buffer("inverse_table", inverse_table)

        link_types = [
            "ADJACENT",
            "REACH",
            "RANGED_MOD",
            "ACTS_BEFORE",
            "MELEE_DMG_REL",
            "RETAL_DMG_REL",
            "RANGED_DMG_REL"
        ]

        gatconv_kwargs = dict(
            in_channels=(-1, -1),
            out_channels=config["gnn_z_size"],
            heads=config["gnn_heads"],
            add_self_loops=True
        )

        # XXX: todo: over-arching global node connected to all hexes
        #           (for non-hex data)

        self.layers = nn.ModuleList()
        for _ in range(config["gnn_layers"]):
            layer = dict()
            for lt in link_types:
                layer[("hex", lt, "hex")] = gnn.GATConv(**gatconv_kwargs)
                # XXX: a leaky_relu is applied after each GATConv, see encode()
            self.layers.append(gnn.HeteroConv(layer))

        self.encoder_merged = nn.Sequential(
            nn.LazyLinear(config["z_size_merged"]),
            nn.LeakyReLU()
        )

        self.actor = nn.LazyLinear(len(MainAction)+165+165)
        self.critic = nn.LazyLinear(1)

        # Init lazy layers (must be before weight/bias init)
        with torch.no_grad():
            obs = torch.randn([2, STATE_SIZE])
            done = torch.zeros(2)
            links = 2 * [VcmiEnv.OBSERVATION_SPACE["links"].sample()]
            hdata = to_hdata_batch(obs, done, links)
            z = self.encode(hdata)
            self._get_actdata_eval(z, obs)
            self._get_value(z)

        def kaiming_init(linlayer):
            # Assume LeakyReLU's negative slope is the default
            a = torch.nn.LeakyReLU().negative_slope
            nn.init.kaiming_uniform_(linlayer.weight, nonlinearity='leaky_relu', a=a)
            nn.init.zeros_(linlayer.bias)

        def xavier_init(linlayer):
            nn.init.xavier_uniform_(linlayer.weight)
            nn.init.zeros_(linlayer.bias)

        # For layers followed by ReLU or LeakyReLU, use Kaiming (He).
        kaiming_init(self.encoder_merged[0])

        # For other layers, use Xavier.
        xavier_init(self.actor)
        xavier_init(self.critic)

    def encode(self, hdata):
        x_dict = hdata.x_dict

        for layer in self.layers:
            x_dict = layer(x_dict, hdata.edge_index_dict, edge_attr_dict=hdata.edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        zhex, hmask = to_dense_batch(x_dict["hex"], hdata["hex"].batch)
        # zhex is (B, Nmax, Z)
        # hmask is (B, Nmax)
        # where Nmax is 165 (all graphs have N=165 nodes in this case)
        # => hmask will be all-true (no padded nodes)
        # Note that this would not be the case if e.g. units were also nodes.
        assert torch.all(hmask)

        return self.encoder_merged(zhex.flatten(start_dim=1))

    def _get_value(self, z):
        return self.critic(z)

    def _get_actdata_train(self, z_merged, obs, action):
        B = obs.shape[0]
        b_inds = torch.arange(B, device=obs.device)

        act0, hex1, hex2 = self.inverse_table[action].unbind(1)

        action_logits, hex1_logits, hex2_logits = self.actor(z_merged).split([len(MainAction), 165, 165], dim=-1)

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
        mask_action = torch.zeros(B, 4, dtype=torch.bool, device=obs.device)

        # 0=WAIT
        mask_action[:, 0] = obs[:, GLOBAL_ATTR_MAP["ACTION_MASK"][1] + GLOBAL_ACT_MAP["WAIT"]]

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_action[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1)

        # Next, we sample:
        #
        # 1. Sample MAIN ACTION
        dist_act0 = CategoricalMasked(logits=action_logits, mask=mask_action)

        # 2. Sample HEX1 (with mask corresponding to the main action)
        dist_hex1 = CategoricalMasked(logits=hex1_logits, mask=mask_hex1[b_inds, act0])

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        dist_hex2 = CategoricalMasked(logits=hex2_logits, mask=mask_hex2[b_inds, act0, hex1])

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    def _get_actdata_eval(self, z_merged, obs):
        B = obs.shape[0]
        b_inds = torch.arange(B, device=obs.device)

        action_logits, hex1_logits, hex2_logits = self.actor(z_merged).split([len(MainAction), 165, 165], dim=-1)

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
        mask_action = torch.zeros(B, 4, dtype=torch.bool, device=obs.device)

        # 0=WAIT
        mask_action[:, 0] = obs[:, GLOBAL_ATTR_MAP["ACTION_MASK"][1] + GLOBAL_ACT_MAP["WAIT"]]

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_action[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1)

        # Next, we sample:
        #
        # 1. Sample MAIN ACTION
        dist_act0 = CategoricalMasked(logits=action_logits, mask=mask_action)
        act0 = dist_act0.sample()

        # 2. Sample HEX1 (with mask corresponding to the main action)
        dist_hex1 = CategoricalMasked(logits=hex1_logits, mask=mask_hex1[b_inds, act0])
        hex1 = dist_hex1.sample()

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        dist_hex2 = CategoricalMasked(logits=hex2_logits, mask=mask_hex2[b_inds, act0, hex1])
        hex2 = dist_hex2.sample()

        action = self.action_table[act0, hex1, hex2]

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    def get_actdata_train(self, hdata):
        z_merged = self.encode(hdata)
        return self._get_actdata_train(z_merged, hdata.obs, hdata.action)

    def get_actdata_eval(self, hdata):
        z_merged = self.encode(hdata)
        return self._get_actdata_eval(z_merged, hdata.obs)

    def get_value(self, hdata):
        z_merged = self.encode(hdata)
        return self._get_value(z_merged)


class DNAModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model_policy = Model(config)
        self.model_value = Model(config)
        self.device = device
        self.to(device)


def create_venv(env_kwargs, num_envs, sync=True):
    # AsyncVectorEnv creates a dummy_env() in the main process just to
    # extract metadata, which causes VCMI init pid error afterwards
    pid = os.getpid()
    dummy_env = SimpleNamespace(
        metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
        render_mode='ansi',
        action_space=VcmiEnv.ACTION_SPACE,
        observation_space=VcmiEnv.OBSERVATION_SPACE["observation"],
        close=lambda: None
    )

    def env_creator(i):
        if os.getpid() == pid and not sync:
            return dummy_env

        env = VcmiEnv(**env_kwargs)
        env = LegacyObservationSpaceWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    funcs = [partial(env_creator, i) for i in range(num_envs)]

    if sync:
        vec_env = gym.vector.SyncVectorEnv(funcs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        vec_env = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    vec_env.reset()

    return vec_env


def collect_samples(logger, model, venv, num_vsteps, storage):
    assert not torch.is_inference_mode_enabled()  # causes issues during training
    assert not torch.is_grad_enabled()

    stats = SampleStats()
    device = model.device

    storage.rollout_buffer.clear()

    for vstep in range(num_vsteps):
        logger.debug("(train) vstep: %d" % vstep)

        v_hdata = storage.v_next_hdata

        v_actdata = model.model_policy.get_actdata_eval(v_hdata)
        v_value = model.model_value.get_value(v_hdata)

        v_hdata.action[:] = v_actdata.action
        v_hdata.logprob[:] = v_actdata.logprob
        v_hdata.value[:] = v_value.flatten()

        v_obs, v_rew, v_term, v_trunc, v_info = venv.step(v_actdata.action.cpu().numpy())

        v_hdata.reward[:] = torch.as_tensor(v_rew, device=device)

        storage.bv_dones[vstep] = v_hdata.done
        storage.bv_values[vstep] = v_hdata.value
        storage.bv_rewards[vstep] = v_hdata.reward
        storage.v_next_hdata = to_hdata_batch(
            torch.as_tensor(v_obs, device=device),
            torch.as_tensor(np.logical_or(v_term, v_trunc), device=device),
            venv.call("links")
        )

        storage.rollout_buffer.append(v_hdata)

        # See notes/gym_vector.txt
        if "_final_info" in v_info:
            v_done_id = np.flatnonzero(v_info["_final_info"])
            v_final_info = v_info["final_info"]
            stats.ep_rew_mean += sum(v_final_info["episode"]["r"][v_done_id])
            stats.ep_len_mean += sum(v_final_info["episode"]["l"][v_done_id])
            stats.ep_value_mean += sum(v_final_info["net_value"][v_done_id])
            stats.ep_is_success_mean += sum(v_final_info["is_success"][v_done_id])
            stats.num_episodes += len(v_done_id)

    assert len(storage.rollout_buffer) == num_vsteps

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

    # bootstrap value if not done
    v_next_value = model.model_value.get_value(storage.v_next_hdata).flatten()

    storage.v_next_hdata.value[:] = v_next_value

    return stats


def eval_model(logger, model, venv, num_vsteps):
    assert torch.is_inference_mode_enabled()

    stats = SampleStats()
    device = model.device

    t = lambda x: torch.as_tensor(x, device=model.device)

    v_obs, _ = venv.reset()
    v_done = torch.zeros(venv.num_envs, dtype=torch.bool, device=device)

    for vstep in range(0, num_vsteps):
        logger.debug("(eval) vstep: %d" % vstep)

        v_hdata = to_hdata_batch(t(v_obs), t(v_done), venv.call("links"))
        v_actdata = model.model_policy.get_actdata_eval(v_hdata)
        v_obs, v_rew, v_term, v_trunc, v_info = venv.step(v_actdata.action.cpu().numpy())
        v_done = np.logical_or(v_term, v_trunc)

        # See notes/gym_vector.txt
        if "_final_info" in v_info:
            v_done_id = np.flatnonzero(v_info["_final_info"])
            v_final_info = v_info["final_info"]
            stats.ep_rew_mean += sum(v_final_info["episode"]["r"][v_done_id])
            stats.ep_len_mean += sum(v_final_info["episode"]["l"][v_done_id])
            stats.ep_value_mean += sum(v_final_info["net_value"][v_done_id])
            stats.ep_is_success_mean += sum(v_final_info["is_success"][v_done_id])
            stats.num_episodes += len(v_done_id)

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

    return stats


def train_model(
    logger,
    model,
    optimizer_policy,
    optimizer_value,
    optimizer_distill,
    autocast_ctx,
    scaler,
    storage,
    train_config
):
    assert torch.is_grad_enabled()

    num_vsteps = train_config["num_vsteps"]
    num_envs = train_config["env"]["num_envs"]

    # # compute advantages
    with torch.no_grad():
        lastgaelam = 0

        for t in reversed(range(num_vsteps)):
            if t == num_vsteps - 1:
                nextnonterminal = 1.0 - storage.v_next_hdata.done
                nextvalues = storage.v_next_hdata.value
            else:
                nextnonterminal = 1.0 - storage.bv_dones[t + 1]
                nextvalues = storage.bv_values[t + 1]
            delta = storage.bv_rewards[t] + train_config["gamma"] * nextvalues * nextnonterminal - storage.bv_values[t]
            storage.bv_advantages[t] = lastgaelam = delta + train_config["gamma"] * train_config["gae_lambda"] * nextnonterminal * lastgaelam
        storage.bv_returns[:] = storage.bv_advantages + storage.bv_values

        for b in range(num_vsteps):
            v_hdata = storage.rollout_buffer[b]
            v_hdata.advantage[:] = storage.bv_advantages[b]
            v_hdata.ep_return[:] = storage.bv_returns[b]

    batch_size = num_vsteps * num_envs
    minibatch_size = int(batch_size // train_config["num_minibatches"])

    # Explode buffer into individual hdatas (dataloader forms a single, large batch)
    # TODO: maybe clone obs here to prevent inference_mode error?
    dataloader = DataLoader(
        [hdata for batch in storage.rollout_buffer for hdata in batch.to_data_list()],
        batch_size=minibatch_size,
        shuffle=True
    )

    clipfracs = []

    policy_losses = torch.zeros(train_config["num_minibatches"])
    entropy_losses = torch.zeros(train_config["num_minibatches"])
    value_losses = torch.zeros(train_config["num_minibatches"])
    distill_losses = torch.zeros(train_config["num_minibatches"])

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.policy) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.policy) minibatch: %d" % i)

            newactdata = model.model_policy.get_actdata_train(mb)

            logratio = newactdata.logprob - mb.logprob
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item()]

            if train_config["norm_adv"]:
                mb_advantages = (mb.advantage - mb.advantage.mean()) / (mb.advantage.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_config["clip_coef"], 1 + train_config["clip_coef"])
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = newactdata.entropy.mean()

            policy_losses[i] = policy_loss.detach()
            entropy_losses[i] = entropy_loss.detach()

            action_loss = policy_loss - entropy_loss * train_config["ent_coef"]

            with autocast_ctx(False):
                scaler.scale(action_loss).backward()
                scaler.unscale_(optimizer_policy)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_policy.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_policy)
                scaler.update()
                optimizer_policy.zero_grad()

        if train_config["target_kl"] is not None and approx_kl > train_config["target_kl"]:
            break

    # Value network optimization
    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.value) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.value) minibatch: %d" % i)

            newvalue = model.model_value.get_value(mb)

            # Value loss
            newvalue = newvalue.view(-1)
            if train_config["clip_vloss"]:
                v_loss_unclipped = (newvalue - mb.ep_return) ** 2
                v_clipped = mb.value + torch.clamp(
                    newvalue - mb.value,
                    -train_config["clip_coef"],
                    train_config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - mb.ep_return) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()
            else:
                # XXX: SIMO: SB3 does not multiply by 0.5 here
                value_loss = 0.5 * ((newvalue - mb.ep_return) ** 2).mean()

            value_losses[i] = value_loss.detach()

            with autocast_ctx(False):
                scaler.scale(value_loss).backward()
                scaler.unscale_(optimizer_value)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_value.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_value)
                scaler.update()
                optimizer_value.zero_grad()

    # Value network to policy network distillation
    model.model_policy.zero_grad(True)  # don't clone gradients
    old_model_policy = copy.deepcopy(model.model_policy).to(model.device)
    old_model_policy.eval()
    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.distill) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.distill) minibatch: %d" % i)

            # Compute policy and value targets
            with torch.no_grad():
                old_actdata = old_model_policy.get_actdata_eval(mb)
                value_target = model.model_value.get_value(mb)

            # XXX: must pass action=<old_action> to ensure masks for hex1 and hex2 are the same
            #     (if actions differ, masks will differ and KLD will become NaN)
            new_z = model.model_policy.encode(mb)
            new_actdata = model.model_policy._get_actdata_train(new_z, mb.obs, mb.action)
            new_value = model.model_policy._get_value(new_z)

            # Distillation loss
            distill_actloss = (
                kld(old_actdata.act0_dist, new_actdata.act0_dist)
                + kld(old_actdata.hex1_dist, new_actdata.hex1_dist)
                + kld(old_actdata.hex2_dist, new_actdata.hex2_dist)
            ).mean()

            distill_vloss = 0.5 * (new_value.view(-1) - value_target).square().mean()
            distill_loss = distill_vloss + train_config["distill_beta"] * distill_actloss

            distill_losses[i] = distill_loss.detach()

            with autocast_ctx(False):
                scaler.scale(distill_loss).backward()
                scaler.unscale_(optimizer_distill)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_policy.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_distill)
                scaler.update()
                optimizer_distill.zero_grad()

    y_pred, y_true = storage.bv_values.cpu().numpy(), storage.bv_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return TrainStats(
        value_loss=value_losses.mean().item(),
        policy_loss=policy_losses.mean().item(),
        entropy_loss=entropy_losses.mean().item(),
        distill_loss=distill_losses.mean().item(),
        approx_kl=approx_kl.item(),
        clipfrac=float(np.mean(clipfracs)),
        explained_var=float(explained_var),
    )


def prepare_wandb_log(
    model,
    optimizer,
    state,
    train_stats,
    train_sample_stats,
    eval_multistats,
):
    wlog = {}

    if eval_multistats.num_episodes > 0:
        wlog.update({
            "eval/ep_rew_mean": eval_multistats.ep_rew_mean,
            "eval/ep_value_mean": eval_multistats.ep_value_mean,
            "eval/ep_len_mean": eval_multistats.ep_len_mean,
            "eval/ep_success_rate": eval_multistats.ep_is_success_mean,
            "eval/ep_count": eval_multistats.num_episodes,
        })

    for name, eval_sample_stats in eval_multistats.variants.items():
        wlog.update({
            f"eval/{name}/ep_rew_mean": eval_sample_stats.ep_rew_mean,
            f"eval/{name}/ep_value_mean": eval_sample_stats.ep_value_mean,
            f"eval/{name}/ep_len_mean": eval_sample_stats.ep_len_mean,
            f"eval/{name}/ep_success_rate": eval_sample_stats.ep_is_success_mean,
            f"eval/{name}/ep_count": eval_sample_stats.num_episodes,
        })

    if train_sample_stats.num_episodes > 0:
        state.rollout_rew_queue_100.append(train_sample_stats.ep_rew_mean)
        state.rollout_rew_queue_1000.append(train_sample_stats.ep_rew_mean)
        state.rollout_net_value_queue_100.append(train_sample_stats.ep_value_mean)
        state.rollout_net_value_queue_1000.append(train_sample_stats.ep_value_mean)
        state.rollout_is_success_queue_100.append(train_sample_stats.ep_is_success_mean)
        state.rollout_is_success_queue_1000.append(train_sample_stats.ep_is_success_mean)
        wlog.update({
            "train/ep_rew_mean": train_sample_stats.ep_rew_mean,
            "train/ep_value_mean": train_sample_stats.ep_value_mean,
            "train/ep_len_mean": train_sample_stats.ep_len_mean,
            "train/ep_success_rate": train_sample_stats.ep_is_success_mean,
            "train/ep_count": train_sample_stats.num_episodes,
        })

    wlog.update({
        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train/value_loss": train_stats.value_loss,
        "train/policy_loss": train_stats.policy_loss,
        "train/entropy_loss": train_stats.entropy_loss,
        "train/distill_loss": train_stats.distill_loss,
        "train/approx_kl": train_stats.approx_kl,
        "train/clipfrac": train_stats.clipfrac,
        "train/explained_var": train_stats.explained_var,
        "train/ep_value_mean_100": safe_mean(state.rollout_net_value_queue_100),
        "train/ep_value_mean_1000": safe_mean(state.rollout_net_value_queue_1000),
        "train/ep_rew_mean_100": safe_mean(state.rollout_rew_queue_100),
        "train/ep_rew_mean_1000": safe_mean(state.rollout_rew_queue_1000),
        "train/ep_success_rate_100": safe_mean(state.rollout_is_success_queue_100),
        "train/ep_success_rate_1000": safe_mean(state.rollout_is_success_queue_1000),
        "global/global_num_timesteps": state.global_timestep,
        "global/global_num_seconds": state.global_second,
        "global/num_rollouts": state.current_rollout,
        "global/num_timesteps": state.current_timestep,
        "global/num_seconds": state.current_second,
        "global/num_episode": state.current_episode,
    })

    return wlog


def main(config, loglevel, dry_run, no_wandb, seconds_total=float("inf"), save_on_exit=True):
    run_id = config["run"]["id"]
    resumed_config = config["run"]["resumed_config"]

    os.makedirs(config["run"]["out_dir"], exist_ok=True)
    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        msg = f"Saving new config to: {f.name}"
        if dry_run:
            print(f"{msg} (--dry-run)")
        else:
            print(msg)
            json.dump(config, f, indent=4)

    # assert config["checkpoint"]["interval_s"] > config["eval"]["interval_s"]
    assert config["checkpoint"]["permanent_interval_s"] > config["eval"]["interval_s"]
    assert config["train"]["env"]["kwargs"]["user_timeout"] >= 2 * config["eval"]["interval_s"]

    checkpoint_config = dig(config, "checkpoint")
    train_config = dig(config, "train")
    eval_config = dig(config, "eval")

    logger = StructuredLogger(level=getattr(logging, loglevel), filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"), context=dict(run_id=run_id))
    logger.info(dict(config=config))

    learning_rate = config["train"]["learning_rate"]

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    if train_config.get("torch_detect_anomaly", None):
        torch.autograd.set_detect_anomaly(True)  # debug

    if train_config.get("torch_cuda_matmul", None):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    train_venv = create_venv(train_config["env"]["kwargs"], train_config["env"]["num_envs"], sync=train_config["env"].get("sync", False))
    logger.info("Initialized %d train envs" % train_venv.num_envs)

    eval_venv_variants = {}
    for name, envcfg in eval_config["env_variants"].items():
        eval_venv_variants[name] = create_venv(envcfg["kwargs"], envcfg["num_envs"], sync=envcfg.get("sync", False))
        logger.info("Initialized %d eval envs for variant: %s" % (envcfg["num_envs"], name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = train_config["env"]["num_envs"]
    num_steps = train_config["num_vsteps"] * num_envs
    batch_size = int(num_steps)
    assert batch_size % train_config["num_minibatches"] == 0, f"{batch_size} % {train_config['num_minibatches']} == 0"
    storage = Storage(train_venv, train_config["num_vsteps"], device)
    state = State()

    model = DNAModel(config=config["model"], device=device)

    optimizer_policy = torch.optim.Adam(model.model_policy.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(model.model_value.parameters(), lr=learning_rate)
    optimizer_distill = torch.optim.Adam(model.model_policy.parameters(), lr=learning_rate)

    optimizer_policy.param_groups[0].setdefault("initial_lr", learning_rate)
    optimizer_value.param_groups[0].setdefault("initial_lr", learning_rate)
    optimizer_distill.param_groups[0].setdefault("initial_lr", learning_rate)

    if train_config["torch_autocast"]:
        autocast_ctx = lambda enabled: torch.autocast(device.type, enabled=enabled)
        scaler = torch.GradScaler(device.type, enabled=True)
    else:
        # No-op autocast and scaler
        autocast_ctx = contextlib.nullcontext
        scaler = torch.GradScaler(device.type, enabled=False)

    logger.debug("Initialized models and optimizers (autocast=%s)" % train_config["torch_autocast"])

    if resumed_config:
        load_checkpoint(
            logger=logger,
            dry_run=dry_run,
            models={"dna": model},
            optimizers={
                "policy": optimizer_policy,
                "value": optimizer_value,
                "distill": optimizer_distill,
            },
            scalers={"default": scaler},
            states={"default": state},
            out_dir=config["run"]["out_dir"],
            run_id=run_id,
            optimize_local_storage=checkpoint_config["optimize_local_storage"],
            s3_config=checkpoint_config["s3"],
            device=device,
        )

        state.current_rollout = 0
        state.current_timestep = 0
        state.current_second = 0
        state.current_episode = 0
        state.current_vstep = 0

        # lr is lost after loading weights
        optimizer_policy.param_groups[0]["lr"] = learning_rate
        optimizer_value.param_groups[0]["lr"] = learning_rate
        optimizer_distill.param_groups[0]["lr"] = learning_rate

        state.resumes += 1
        logger.info("Resumes: %d" % state.resumes)

    if no_wandb:
        from unittest.mock import Mock
        wandb = Mock()
    else:
        wandb = setup_wandb(config, model, __file__)

    accumulated_logs = {}

    def accumulate_logs(data):
        for k, v in data.items():
            if k not in accumulated_logs:
                accumulated_logs[k] = [v]
            else:
                accumulated_logs[k].append(v)

    def aggregate_logs():
        agg_data = {k: safe_mean(v) for k, v in accumulated_logs.items()}
        accumulated_logs.clear()
        return agg_data

    wandb.log({
        "global/resumes": state.resumes,
        "train_config/num_envs": num_envs,
        "train_config/num_vsteps": train_config["num_vsteps"],
        "train_config/num_minibatches": train_config["num_minibatches"],
        "train_config/update_epochs": train_config["update_epochs"],
        "train_config/gamma": train_config["gamma"],
        "train_config/gae_lambda": train_config["gae_lambda"],
        "train_config/ent_coef": train_config["ent_coef"],
        "train_config/clip_coef": train_config["clip_coef"],
        "train_config/learning_rate": train_config["learning_rate"],  # also logged during training
        "train_config/norm_adv": int(train_config["norm_adv"]),
        "train_config/clip_vloss": int(train_config["clip_vloss"]),
        "train_config/max_grad_norm": train_config["max_grad_norm"],
        "train_config/distill_beta": train_config["distill_beta"],
    }, commit=False)

    # during training, we simply check if the event is set and optionally skip the upload
    # Non-bloking, but uploads may be skipped (checkpoint uploads)
    uploading_event = threading.Event()

    timers = {
        "all": Timer(),
        "sample": Timer(),
        "train": Timer(),
        "eval": Timer(),
    }

    # For benchmark
    cumulative_timer_values = {k: 0 for k in timers.keys()}

    timers["all"].start()
    eval_net_value_best = None

    permanent_checkpoint_timer = Timer()
    permanent_checkpoint_timer.start()
    wandb_log_commit_timer = Timer()
    wandb_log_commit_timer.start()
    wandb_log_commit_timer._started_at = 0  # force first trigger
    eval_timer = Timer()
    eval_timer.start()
    if config["eval"]["at_script_start"]:
        eval_timer._started_at = 0  # force first trigger

    lr_schedule_timer = Timer()
    lr_schedule_timer.start()

    if train_config["lr_scheduler_mod"]:
        lr_scheduler_mod = importlib.import_module(train_config["lr_scheduler_mod"])
        lr_scheduler_cls = getattr(lr_scheduler_mod, train_config["lr_scheduler_cls"])

        lr_schedule_policy = lr_scheduler_cls(optimizer_policy, **train_config["lr_scheduler_kwargs"])
        lr_schedule_value = lr_scheduler_cls(optimizer_value, **train_config["lr_scheduler_kwargs"])
        lr_schedule_distill = lr_scheduler_cls(optimizer_distill, **train_config["lr_scheduler_kwargs"])
    else:
        lr_schedule_policy = torch.optim.lr_scheduler.LambdaLR(optimizer_policy, lr_lambda=lambda _: 1)
        lr_schedule_value = torch.optim.lr_scheduler.LambdaLR(optimizer_value, lr_lambda=lambda _: 1)
        lr_schedule_distill = torch.optim.lr_scheduler.LambdaLR(optimizer_distill, lr_lambda=lambda _: 1)

    # TODO: torch LR schedulers are very buggy and cannot be resumed reliably
    # (they perform just 1 step for StepLR; they change the step size for LinearLR, ...etc)
    # Also, advancing manually like this raises warning for not calling optimizer.step()
    # Also, calling .step(N) raises deprecation warning...
    for _ in range(state.global_second // train_config["lr_scheduler_interval_s"]):
        lr_schedule_policy.step()
        lr_schedule_value.step()
        lr_schedule_distill.step()

    global_second_start = state.global_second

    save_fn = partial(
        save_checkpoint,
        logger=logger,
        dry_run=dry_run,
        models={"dna": model},
        optimizers={
            "policy": optimizer_policy,
            "value": optimizer_value,
            "distill": optimizer_distill,
        },
        scalers={"default": scaler},
        states={"default": state},
        out_dir=config["run"]["out_dir"],
        run_id=run_id,
        optimize_local_storage=checkpoint_config["optimize_local_storage"],
        s3_config=None,
        config=config,
        uploading_event=threading.Event(),  # never skip this upload
        timestamped=True,
    )

    try:
        while True:
            state.global_second = global_second_start + int(cumulative_timer_values["all"])
            state.current_second = int(cumulative_timer_values["all"])

            if state.current_second >= seconds_total:
                break

            [v.reset(start=(k == "all")) for k, v in timers.items()]

            logger.debug("learning_rate: %s" % optimizer_policy.param_groups[0]['lr'])
            if lr_schedule_timer.peek() > train_config["lr_scheduler_interval_s"]:
                lr_schedule_timer.reset(start=True)
                lr_schedule_policy.step()
                lr_schedule_value.step()
                lr_schedule_distill.step()
                logger.info("New learning_rate: %s" % optimizer_policy.param_groups[0]['lr'])

            # Evaluate first (for a baseline when resuming with modified params)
            eval_multistats = MultiStats()

            if eval_timer.peek() > eval_config["interval_s"]:
                logger.info("Time for eval")
                eval_timer.reset(start=True)

                with timers["eval"]:
                    model.eval()

                    def eval_worker_fn(name, venv, vsteps):
                        sublogger = logger.sublogger(dict(variant=name))
                        with torch.inference_mode():
                            sublogger.info("Start evaluating env variant: %s" % name)
                            stats = eval_model(logger=sublogger, model=model, venv=venv, num_vsteps=vsteps)
                            sublogger.info("Done evaluating env variant: %s" % name)
                            return name, stats

                    with ThreadPoolExecutor(max_workers=100) as ex:
                        futures = [
                            ex.submit(eval_worker_fn, name, venv, eval_config["num_vsteps"])
                            for name, venv in eval_venv_variants.items()
                        ]

                        for fut in as_completed(futures):
                            eval_multistats.add(*fut.result())

            with timers["sample"], torch.no_grad(), autocast_ctx(True):
                model.eval()
                train_sample_stats = collect_samples(
                    logger=logger,
                    model=model,
                    venv=train_venv,
                    num_vsteps=train_config["num_vsteps"],
                    storage=storage,
                )

            state.current_vstep += train_config["num_vsteps"]
            state.current_timestep += train_config["num_vsteps"] * num_envs
            state.global_timestep += train_config["num_vsteps"] * num_envs
            state.current_episode += train_sample_stats.num_episodes
            state.global_episode += train_sample_stats.num_episodes

            model.train()
            with timers["train"], autocast_ctx(True):
                train_stats = train_model(
                    logger=logger,
                    model=model,
                    optimizer_policy=optimizer_policy,
                    optimizer_value=optimizer_value,
                    optimizer_distill=optimizer_distill,
                    autocast_ctx=autocast_ctx,
                    scaler=scaler,
                    storage=storage,
                    train_config=train_config,
                )

            if eval_multistats.num_episodes > 0:
                eval_net_value = eval_multistats.ep_value_mean

                if eval_net_value_best is None:
                    # Initial baseline for resumed configs
                    eval_net_value_best = eval_net_value
                    logger.info("No baseline for checkpoint yet (eval_net_value=%f, eval_net_value_best=None), setting it now" % eval_net_value)
                elif eval_net_value < eval_net_value_best:
                    logger.info("Bad checkpoint (eval_net_value=%f, eval_net_value_best=%f), will skip it" % (eval_net_value, eval_net_value_best))
                else:
                    logger.info("Good checkpoint (eval_net_value=%f, eval_net_value_best=%f), will save it" % (eval_net_value, eval_net_value_best))
                    eval_net_value_best = eval_net_value
                    thread = threading.Thread(target=save_fn, kwargs=dict(uploading_event=uploading_event))
                    thread.start()

            if permanent_checkpoint_timer.peek() > config["checkpoint"]["permanent_interval_s"]:
                permanent_checkpoint_timer.reset(start=True)
                logger.info("Time for a permanent checkpoint")
                thread = threading.Thread(target=save_fn, kwargs=dict(timestamped=True, s3_config=checkpoint_config["s3"]))
                thread.start()

            wlog = prepare_wandb_log(
                model=model.model_policy,
                optimizer=optimizer_policy,
                state=state,
                train_stats=train_stats,
                train_sample_stats=train_sample_stats,
                eval_multistats=eval_multistats,
            )

            accumulate_logs(wlog)

            if wandb_log_commit_timer.peek() > config["wandb_log_interval_s"]:
                # logger.info("Time for wandb log")
                wandb_log_commit_timer.reset(start=True)
                wlog.update(aggregate_logs())
                tstats = timer_stats(timers)
                wlog.update(tstats)
                wlog["train_config/learning_rate"] = optimizer_policy.param_groups[0]['lr']
                wandb.log(wlog, commit=True)

            logger.info(wlog)

            for k in timers.keys():
                cumulative_timer_values[k] += timers[k].peek()

            state.current_rollout += 1

        ret_rew = safe_mean(list(state.rollout_rew_queue_1000)[-min(300, state.current_rollout):])
        ret_value = safe_mean(list(state.rollout_net_value_queue_1000)[-min(300, state.current_rollout):])

        return ret_rew, ret_value, save_fn
    finally:
        if save_on_exit:
            save_fn(timestamped=True)
        if os.getenv("VASTAI_INSTANCE_ID") and not dry_run:
            import vastai_sdk
            vastai_sdk.VastAI().label_instance(id=int(os.environ["VASTAI_INSTANCE_ID"]), label="idle")


# This is in a separate function to prevent vars from being global
def init_config(args):
    if args.dry_run:
        args.no_wandb = True

    if args.f:
        with open(args.f, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)
        config["run"]["resumed_config"] = args.f
    else:
        from .config import config
        run_id = ''.join(random.choices(string.ascii_lowercase, k=8))
        config["run"] = dict(
            id=run_id,
            name=config["name_template"].format(id=run_id, datetime=datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            out_dir=os.path.abspath(config["out_dir_template"].format(id=run_id)),
            resumed_config=None,
        )

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
    parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument("--loglevel", metavar="LOGLEVEL", default="INFO", help="DEBUG | INFO | WARN | ERROR")
    args = parser.parse_args()
    config = init_config(args)

    main(
        config=config,
        loglevel=args.loglevel,
        dry_run=args.dry_run,
        no_wandb=args.no_wandb,
        # seconds_total=10
    )
