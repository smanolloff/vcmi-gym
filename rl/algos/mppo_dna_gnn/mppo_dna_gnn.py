import os
import sys
import random
import logging
import json
import string
import argparse
import threading
import contextlib
import enum
import copy
import importlib
import math

from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import kl_divergence as kld
from torch_geometric.data import Batch
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

from .dual_vec_env import DualVecEnv, to_hdata_list


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
    global_rollout: int = 0
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


class BattleResult(enum.IntEnum):
    LOSS = 0
    WIN = 1
    NA = 2


class Storage:
    def __init__(self, venv, num_vsteps, device):
        v = venv.num_envs
        self.rollout_buffer = []  # contains Batch() objects
        self.v_next_hdata_list = to_hdata_list(
            torch.as_tensor(venv.reset()[0], device=device),
            torch.zeros(v, device=device),
            torch.zeros(v, dtype=torch.int64, device=device),
            venv.call("links"),
        )

        # Needed for the GAE computation (to prevent spaghetti)
        # and for explained_var computation
        self.bv_dones = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_values = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_rewards = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_advantages = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_returns = torch.zeros((num_vsteps, venv.num_envs), device=device)

        # Categorical torch storage for BattleResult
        self.bv_ep_results = torch.zeros((num_vsteps, venv.num_envs), dtype=torch.int64, device=device)


@dataclass
class TrainStats:
    result_loss: float
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
    def __init__(
        self,
        act0,
        act0_logits,
        act0_dist,
        hex1,
        hex1_logits,
        hex1_dist,
        hex2,
        hex2_logits,
        hex2_dist,
        action
    ):
        self.act0 = act0
        self.act0_dist = act0_dist
        self.act0_logits = act0_logits
        self.act0_logprob = act0_dist.log_prob(act0)
        self.act0_entropy = act0_dist.entropy()

        self.hex1 = hex1
        self.hex1_dist = hex1_dist
        self.hex1_logits = hex1_logits
        self.hex1_logprob = hex1_dist.log_prob(hex1)
        self.hex1_entropy = hex1_dist.entropy()

        self.hex2 = hex2
        self.hex2_dist = hex2_dist
        self.hex2_logits = hex2_logits
        self.hex2_logprob = hex2_dist.log_prob(hex2)
        self.hex2_entropy = hex2_dist.entropy()

        self.action = action
        self.logprob = self.act0_logprob + self.hex1_logprob + self.hex2_logprob
        self.entropy = self.act0_entropy + self.hex1_entropy + self.hex2_entropy

    def cpu(self):
        for t in ["act0", "hex1", "hex2"]:
            setattr(self, t, getattr(self, t).cpu())
            # setattr(self, f"{t}_dist", getattr(self, f"{t}_dist").cpu())
            setattr(self, f"{t}_logits", getattr(self, f"{t}_logits").cpu())
            setattr(self, f"{t}_logprob", getattr(self, f"{t}_logprob").cpu())
            setattr(self, f"{t}_entropy", getattr(self, f"{t}_entropy").cpu())

        self.action = self.action.cpu()
        self.logprob = self.logprob.cpu()
        self.entropy = self.entropy.cpu()
        return self


class NonGNNLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x_dict):
        return {k: self.fn(x) for k, x in x_dict.items()}


class GNNBlock(nn.Module):
    # XXX: in_channels must be a tuple of (size_a, size_b) in case of
    #       bipartite graphs.
    def __init__(
        self,
        num_layers,
        in_channels,
        hidden_channels,
        out_channels,
        link_types=LINK_TYPES.keys(),
        node_type="hex",
        edge_dim=1
    ):
        super().__init__()

        # at least 1 "hidden" layer is required for shapes to match
        assert num_layers >= 2

        kwargs = dict(edge_dim=edge_dim, add_self_loops=True)

        # NOTE: the code below will likely fail if len(node_types) > 1 unless
        #       all they all have the same shape
        def make_hetero_dict(inchan, outchan):
            return {
                (node_type, lt, node_type): gnn.GENConv(**kwargs, in_channels=inchan, out_channels=outchan)
                for lt in link_types
            }

        layers = []

        for i in range(num_layers - 1):
            ch_in = in_channels if i == 0 else hidden_channels
            hetero_dict = make_hetero_dict(ch_in, hidden_channels)
            layers.append((gnn.HeteroConv(hetero_dict), "x_dict, edge_index_dict, edge_attr_dict -> x_dict"))
            layers.append((NonGNNLayer(nn.LeakyReLU()), "x_dict -> x_dict"))

        # No activation after last layer
        hetero_dict = make_hetero_dict(hidden_channels, out_channels)
        layers.append((gnn.HeteroConv(hetero_dict), "x_dict, edge_index_dict, edge_attr_dict -> x_dict"))

        self.layers = gnn.Sequential("x_dict, edge_index_dict, edge_attr_dict", layers)

    def forward(self, hdata):
        return self.layers(hdata.x_dict, hdata.edge_index_dict, hdata.edge_attr_dict)


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

        action_table, inverse_table, amove_hexes = self.__class__.build_action_tables()

        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0))
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1)
        self.register_buffer("action_table", action_table)
        self.register_buffer("inverse_table", inverse_table)

        # XXX: todo: over-arching global node connected to all hexes
        #           (for non-hex data)

        self.encoder_hexes = GNNBlock(
            config["gnn_num_layers"],
            STATE_SIZE_ONE_HEX,
            config["gnn_hidden_channels"],
            config["gnn_out_channels"],
        )

        d = config["gnn_out_channels"]

        self.encoder_other = nn.Sequential(
            nn.Linear(self.dim_other, d),
            nn.LeakyReLU()
        )

        self.act0_head = nn.Linear(d, len(MainAction))
        self.emb_act0 = nn.Embedding(len(MainAction), d)
        self.Wk_hex1 = nn.Linear(d, d, bias=False)
        self.Wk_hex2 = nn.Linear(d, d, bias=False)
        self.Wq_hex1 = nn.Linear(2*d, d)
        self.Wq_hex2 = nn.Linear(2*d, d)

        self.critic = nn.Sequential(
            # nn.LayerNorm(d), helps?
            nn.Linear(d, config["critic_hidden_features"]),
            nn.LeakyReLU(),
            nn.Linear(config["critic_hidden_features"], 1)
        )

        # self.result_predictor = nn.Sequential(
        #     nn.Linear(d, config["result_predictor_hidden_features"]),
        #     nn.LeakyReLU(),
        #     nn.Linear(config["result_predictor_hidden_features"], 2)  # 0=loss, 2=win
        # )

        # Init lazy layers (must be before weight/bias init)
        with torch.no_grad():
            obs = torch.randn([2, STATE_SIZE])
            done = torch.zeros(2, dtype=torch.bool)
            result = torch.zeros(2, dtype=torch.int64)
            links = 2 * [VcmiEnv.OBSERVATION_SPACE["links"].sample()]
            hdata = Batch.from_data_list(to_hdata_list(obs, done, result, links))
            z_hexes, z_global = self.encode(hdata)
            self._get_actdata_eval(z_hexes, z_global, obs)
            self._get_value(z_global)

        def kaiming_init(linlayer):
            # Assume LeakyReLU's negative slope is the default
            a = torch.nn.LeakyReLU().negative_slope
            nn.init.kaiming_uniform_(linlayer.weight, nonlinearity='leaky_relu', a=a)
            if linlayer.bias is not None:
                nn.init.zeros_(linlayer.bias)

        def xavier_init(linlayer):
            nn.init.xavier_uniform_(linlayer.weight)
            if linlayer.bias is not None:
                nn.init.zeros_(linlayer.bias)

        # For layers followed by ReLU or LeakyReLU, use Kaiming (He).
        kaiming_init(self.encoder_other[0])

        # For other layers, use Xavier.
        xavier_init(self.act0_head)
        xavier_init(self.Wk_hex1)
        xavier_init(self.Wk_hex2)
        xavier_init(self.Wq_hex1)
        xavier_init(self.Wq_hex2)
        kaiming_init(self.critic[0])
        xavier_init(self.critic[2])

    def encode(self, hdata):
        z_hexes_dict = self.encoder_hexes(hdata)
        z_hexes, hmask = to_dense_batch(z_hexes_dict["hex"], hdata["hex"].batch)

        # zhex is (B, Nmax, Z)
        # hmask is (B, Nmax)
        # where Nmax is 165 (all graphs have N=165 nodes in this case)
        # => hmask will be all-true (no padded nodes)
        # Note that this would not be the case if e.g. units were also nodes.
        assert torch.all(hmask)

        z_other = self.encoder_other(hdata.obs[:, :self.dim_other])
        z_global = z_other + z_hexes.mean(1)

        return z_hexes, z_global

    def _get_result_prediction(self, z_global):
        return self.result_predictor(z_global)

    def _get_value(self, z_global):
        return self.critic(z_global)

    def _get_actdata_train(self, z_hexes, z_global, obs, action):
        B = obs.shape[0]
        b_inds = torch.arange(B, device=obs.device)

        act0, hex1, hex2 = self.inverse_table[action].unbind(1)

        act0_logits = self.act0_head(z_global)

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
        dist_act0 = CategoricalMasked(logits=act0_logits, mask=mask_action)

        # 2. Sample HEX1 (with mask corresponding to the main action)
        act0_emb = self.emb_act0(act0)
        d = act0_emb.size(-1)
        q_hex1 = self.Wq_hex1(torch.cat([z_global, act0_emb], -1))              # (B, d)
        k_hex1 = self.Wk_hex1(z_hexes)                                          # (B, 165, d)
        hex1_logits = (k_hex1 @ q_hex1.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        dist_hex1 = CategoricalMasked(logits=hex1_logits, mask=mask_hex1[b_inds, act0])

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        z_hex1 = z_hexes[b_inds, hex1, :]                                       # (B, d)
        q_hex2 = self.Wq_hex2(torch.cat([z_global, z_hex1], -1))                # (B, d)
        k_hex2 = self.Wk_hex2(z_hexes)                                         # (B, 165, d)
        hex2_logits = (k_hex2 @ q_hex2.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        dist_hex2 = CategoricalMasked(logits=hex2_logits, mask=mask_hex2[b_inds, act0, hex1])

        return ActionData(
            act0=act0, act0_logits=act0_logits, act0_dist=dist_act0,
            hex1=hex1, hex1_logits=hex1_logits, hex1_dist=dist_hex1,
            hex2=hex2, hex2_logits=hex2_logits, hex2_dist=dist_hex2,
            action=action,
        )

    def _get_actdata_eval(self, z_hexes, z_global, obs, deterministic=False):
        B = obs.shape[0]
        b_inds = torch.arange(B, device=obs.device)

        act0_logits = self.act0_head(z_global)

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
        dist_act0 = CategoricalMasked(logits=act0_logits, mask=mask_action)
        act0 = dist_act0.probs.argmax(dim=1) if deterministic else dist_act0.sample()  # for testing vs. executorch

        # 2. Sample HEX1 (with mask corresponding to the main action)
        act0_emb = self.emb_act0(act0)
        d = act0_emb.size(-1)
        q_hex1 = self.Wq_hex1(torch.cat([z_global, act0_emb], -1))              # (B, d)
        k_hex1 = self.Wk_hex1(z_hexes)                                          # (B, 165, d)
        hex1_logits = (k_hex1 @ q_hex1.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        dist_hex1 = CategoricalMasked(logits=hex1_logits, mask=mask_hex1[b_inds, act0])
        hex1 = dist_hex1.probs.argmax(dim=1) if deterministic else dist_hex1.sample()

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        z_hex1 = z_hexes[b_inds, hex1, :]                                       # (B, d)
        q_hex2 = self.Wq_hex2(torch.cat([z_global, z_hex1], -1))                # (B, d)
        k_hex2 = self.Wk_hex2(z_hexes)                                          # (B, 165, d)
        hex2_logits = (k_hex2 @ q_hex2.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        dist_hex2 = CategoricalMasked(logits=hex2_logits, mask=mask_hex2[b_inds, act0, hex1])
        hex2 = dist_hex2.probs.argmax(dim=1) if deterministic else dist_hex2.sample()  # for testing vs. executorch

        action = self.action_table[act0, hex1, hex2]

        return ActionData(
            act0=act0, act0_logits=act0_logits, act0_dist=dist_act0,
            hex1=hex1, hex1_logits=hex1_logits, hex1_dist=dist_hex1,
            hex2=hex2, hex2_logits=hex2_logits, hex2_dist=dist_hex2,
            action=action,
        )

    def get_actdata_train(self, hdata):
        z_hexes, z_global = self.encode(hdata)
        return self._get_actdata_train(z_hexes, z_global, hdata.obs, hdata.action)

    def get_actdata_eval(self, hdata, deterministic=False):
        z_hexes, z_global = self.encode(hdata)
        return self._get_actdata_eval(z_hexes, z_global, hdata.obs, deterministic)

    def get_value(self, hdata):
        _, z_global = self.encode(hdata)
        return self._get_value(z_global)

    def get_result_prediction(self, hdata):
        _, z_global = self.encode(hdata)
        return self._get_result_prediction(z_global)


class DNAModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model_policy = Model(config)
        self.model_value = Model(config)
        self.device = device
        self.to(device)


def collect_samples(logger, model, venv, num_vsteps, storage):
    assert not torch.is_inference_mode_enabled()  # causes issues during training
    assert not torch.is_grad_enabled()

    stats = SampleStats()

    storage.rollout_buffer.clear()

    for vstep in range(num_vsteps):
        logger.debug("(train) vstep: %d" % vstep)

        v_hdata_list = storage.v_next_hdata_list
        v_hdata_batch = Batch.from_data_list(v_hdata_list).to(model.device)

        v_actdata = model.model_policy.get_actdata_eval(v_hdata_batch).cpu()
        v_value = model.model_value.get_value(v_hdata_batch).flatten().cpu()
        v_obs, v_rew, v_term, v_trunc, v_info = venv.step(v_actdata.action.numpy())
        v_rew = torch.as_tensor(v_rew)

        for i, hdata in enumerate(v_hdata_list):
            hdata.action = v_actdata.action[i]
            hdata.logprob = v_actdata.logprob[i]
            hdata.value = v_value[i]
            hdata.reward = v_rew[i]

            storage.bv_dones[vstep, i] = hdata.done
            storage.bv_values[vstep, i] = hdata.value
            storage.bv_rewards[vstep, i] = hdata.reward
            storage.bv_ep_results[vstep, i] = hdata.ep_result

        # See notes/gym_vector.txt
        if "_final_info" in v_info:
            v_done_id = np.flatnonzero(v_info["_final_info"])
            v_final_info = v_info["final_info"]
            stats.ep_rew_mean += sum(v_final_info["episode"]["r"][v_done_id])
            stats.ep_len_mean += sum(v_final_info["episode"]["l"][v_done_id])
            stats.ep_value_mean += sum(v_final_info["net_value"][v_done_id])
            stats.ep_is_success_mean += sum(v_final_info["is_success"][v_done_id])
            stats.num_episodes += len(v_done_id)

            print(f"FINAL _INFO? (vstep={vstep}): {v_info['_final_info']}")
            print(f"FINAL INFO RAW (vstep={vstep}): {v_info['final_info']['is_success']}")

            # v_ep_results:
            # Shape: (num_envs), dtype: int64
            # 0=loss, 1=win, 2=NA

            # XXX: gymnasium's vec env returns inconsistent dtype for boolean arrays in info dicts:
            # - if all values in the array are False, gymnaisium (correctly) returns a dtype=bool dict
            # - if at least 1 value is True, gymnasium returns a dtype=object dict where False is None instead.
            # => must convert excplicitly to bool before converting to int
            v_ep_results = torch.as_tensor(np.where(
                v_info["_final_info"],
                v_info["final_info"]["is_success"].astype(bool).astype(np.int64),
                BattleResult.NA
            ))
            # => (num_envs) with dtype=int64 and values: 0=loss, 1=win, 2=NA
        else:
            v_ep_results = torch.full((venv.num_envs,), BattleResult.NA.value, dtype=torch.int64)

        storage.v_next_hdata_list = to_hdata_list(
            torch.as_tensor(v_obs),
            torch.as_tensor(np.logical_or(v_term, v_trunc)),
            v_ep_results,
            venv.call("links")
        )
        storage.rollout_buffer.extend(v_hdata_list)

    assert len(storage.rollout_buffer) == num_vsteps * venv.num_envs

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

    # bootstrap value if not done
    v_next_hdata_batch = Batch.from_data_list(storage.v_next_hdata_list).to(model.device)
    v_next_value = model.model_value.get_value(v_next_hdata_batch).flatten().cpu()

    for i, hdata in enumerate(storage.v_next_hdata_list):
        hdata.value = v_next_value[i]

    return stats


def eval_model(logger, model, venv, num_vsteps):
    assert torch.is_inference_mode_enabled()

    stats = SampleStats()
    v_obs, _ = venv.reset()

    bv_dones = torch.zeros((num_vsteps, venv.num_envs), dtype=torch.bool)
    bv_ep_results = torch.zeros((num_vsteps, venv.num_envs), dtype=torch.int64)
    bv_result_preds = torch.zeros((num_vsteps, venv.num_envs, 2), dtype=torch.int64)

    v_next_hdata_list = to_hdata_list(
        torch.as_tensor(v_obs),
        torch.zeros(venv.num_envs, dtype=torch.bool),   # v_done
        torch.zeros(venv.num_envs, dtype=torch.int64),  # v_result
        venv.call("links")
    )

    for vstep in range(0, num_vsteps):
        logger.debug("(eval) vstep: %d" % vstep)

        v_hdata_batch = Batch.from_data_list(v_next_hdata_list).to(model.device)

        bv_dones[vstep] = v_hdata_batch.done.cpu()
        bv_ep_results[vstep] = v_hdata_batch.ep_result.cpu()
        bv_result_preds[vstep] = model.model_value.get_result_prediction(v_hdata_batch).cpu()

        v_actdata = model.model_policy.get_actdata_eval(v_hdata_batch)
        v_obs, v_rew, v_term, v_trunc, v_info = venv.step(v_actdata.action.cpu().numpy())

        # See notes/gym_vector.txt
        if "_final_info" in v_info:
            v_done_id = np.flatnonzero(v_info["_final_info"])
            v_final_info = v_info["final_info"]
            stats.ep_rew_mean += sum(v_final_info["episode"]["r"][v_done_id])
            stats.ep_len_mean += sum(v_final_info["episode"]["l"][v_done_id])
            stats.ep_value_mean += sum(v_final_info["net_value"][v_done_id])
            stats.ep_is_success_mean += sum(v_final_info["is_success"][v_done_id])
            stats.num_episodes += len(v_done_id)

            v_ep_results = torch.as_tensor(np.where(
                v_info["_final_info"],
                v_info["final_info"]["is_success"].astype(bool).astype(np.int64),
                BattleResult.NA
            ))
            # => (num_envs) with dtype=int64 and values: 0=loss, 1=win, 2=NA
        else:
            v_ep_results = torch.full((venv.num_envs,), BattleResult.NA.value, dtype=torch.int64)

        v_next_hdata_list = to_hdata_list(
            torch.as_tensor(v_obs),
            torch.as_tensor(np.logical_or(v_term, v_trunc)),
            v_ep_results,
            venv.call("links")
        )

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

        # ep_result is set for the *first* step of the *next* episode
        # We want is to have it for *all* steps of the *relevant* episode
        v_next_hdata_batch = Batch.from_data_list(v_next_hdata_list).to(model.device)
        bv_ep_results_new = torch.full((num_vsteps, venv.num_envs), BattleResult.NA)
        bv_ep_results_new[-1] = v_next_hdata_batch.ep_result

        for t in reversed(range(num_vsteps - 1)):
            bv_ep_results_new[t] = torch.where(
                bv_dones[t + 1].bool(),
                bv_ep_results[t + 1],
                bv_ep_results_new[t + 1]
            )

        bv_ep_results[:] = bv_ep_results_new

        import ipdb; ipdb.set_trace()  # noqa

    return stats


def train_model(
    logger,
    model,
    old_model_policy,
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
    num_envs = sum(train_config["env"]["num_envs_per_opponent"].values())
    v_next_hdata_batch = Batch.from_data_list(storage.v_next_hdata_list)

    # compute advantages
    with torch.no_grad():
        lastgaelam = torch.zeros_like(storage.bv_advantages[0])

        nextnonterminal = 1.0 - v_next_hdata_batch.done
        nextvalues = v_next_hdata_batch.value

        # ep_result is set for the *first* step of the *next* episode
        # We want is to have it for *all* steps of the *relevant* episode
        bv_ep_results_new = torch.full_like(storage.bv_ep_results, BattleResult.NA)
        bv_ep_results_new[-1] = v_next_hdata_batch.ep_result

        for t in reversed(range(num_vsteps - 1)):
            nextnonterminal = 1.0 - storage.bv_dones[t + 1]
            nextvalues = storage.bv_values[t + 1]
            delta = storage.bv_rewards[t] + train_config["gamma"] * nextvalues * nextnonterminal - storage.bv_values[t]
            storage.bv_advantages[t] = lastgaelam = delta + train_config["gamma"] * train_config["gae_lambda"] * nextnonterminal * lastgaelam

            # Update results on terminal steps, otherwise keep last result
            bv_ep_results_new[t] = torch.where(
                storage.bv_dones[t + 1].bool(),
                storage.bv_ep_results[t + 1],
                bv_ep_results_new[t + 1]
            )

        storage.bv_returns[:] = storage.bv_advantages + storage.bv_values
        storage.bv_ep_results[:] = bv_ep_results_new

        for b in range(num_vsteps):
            for v in range(num_envs):
                v_hdata = storage.rollout_buffer[b*num_envs + v]
                v_hdata.advantage = storage.bv_advantages[b, v]
                v_hdata.ep_return = storage.bv_returns[b, v]
                v_hdata.ep_result = storage.bv_ep_results[b, v]

    batch_size = num_vsteps * num_envs
    minibatch_size = int(batch_size // train_config["num_minibatches"])

    dataloader = DataLoader(
        storage.rollout_buffer,
        batch_size=minibatch_size,
        shuffle=True,
        pin_memory=True,
    )

    clipfracs = []

    policy_losses = torch.zeros(train_config["num_minibatches"])
    entropy_losses = torch.zeros(train_config["num_minibatches"])
    value_losses = torch.zeros(train_config["num_minibatches"])
    result_losses = torch.zeros(train_config["num_minibatches"])
    distill_losses = torch.zeros(train_config["num_minibatches"])

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.policy) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.policy) minibatch: %d" % i)
            mb = mb.to(model.device, non_blocking=True)

            newactdata = model.model_policy.get_actdata_train(mb)

            logratio = newactdata.logprob - mb.logprob
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item()]

            mb_advantages = mb.advantage.dtype
            if train_config["norm_adv"]:
                # The 1e-8 is not numerically safe under autocast
                # mb_advantages = (mb.advantage - mb.advantage.mean()) / (mb.advantage.std() + 1e-8)
                with autocast_ctx(False):
                    adv32 = mb.advantage.float()
                    mean = adv32.mean()
                    var = adv32.var(unbiased=False)
                    norm = (adv32 - mean) * torch.rsqrt(var + 1e-8)
                mb_advantages = norm.to(mb.advantage.dtype)

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
            mb = mb.to(model.device, non_blocking=True)

            newvalue = model.model_value.get_value(mb)
            newrespred = model.model_value.get_result_prediction(mb)

            # Value loss
            # XXX: this overflows under autocast since ep_returns values are around ~1000
            #       (e.g. value loss becomes 484841.34375, too big for float16)
            #       => wrapped it in autocast(false), but that does NOT fix the NaN (why?)
            with autocast_ctx(False):
                newvalue = newvalue.view(-1).float()
                mb.value = mb.value.float()
                mb.ep_return = mb.ep_return.float()

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

                # Result prediction loss
                # ipdb> mb.ep_result
                # tensor([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 0, 1,
                #         1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 2, 1,
                #         1, 1])

                assert BattleResult.LOSS == 0
                assert BattleResult.WIN == 1
                assert BattleResult.NA == 2
                result_loss = F.cross_entropy(newrespred, mb.ep_result, ignore_index=2)

            value_losses[i] = value_loss.detach()
            result_losses[i] = result_loss.detach()

            with autocast_ctx(False):
                scaler.scale(value_loss + result_loss).backward()
                scaler.unscale_(optimizer_value)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_value.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_value)
                scaler.update()
                optimizer_value.zero_grad()

    # Value network to policy network distillation
    old_model_policy.load_state_dict(model.model_policy.state_dict(), strict=True)
    old_model_policy.eval()

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.distill) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.distill) minibatch: %d" % i)
            mb = mb.to(model.device, non_blocking=True)

            # Compute policy and value targets
            with torch.no_grad():
                old_actdata = old_model_policy.get_actdata_eval(mb)
                value_target = model.model_value.get_value(mb)

            # XXX: must pass action=<old_action> to ensure masks for hex1 and hex2 are the same
            #     (if actions differ, masks will differ and KLD will become NaN)
            new_z_hexes, new_z_global = model.model_policy.encode(mb)
            new_actdata = model.model_policy._get_actdata_train(new_z_hexes, new_z_global, mb.obs, old_actdata.action)
            new_value = model.model_policy._get_value(new_z_global)

            # Distillation loss
            distill_actloss = (
                CategoricalMasked.kld(old_actdata.act0_dist, new_actdata.act0_dist)
                + CategoricalMasked.kld(old_actdata.hex1_dist, new_actdata.hex1_dist)
                + CategoricalMasked.kld(old_actdata.hex2_dist, new_actdata.hex2_dist)
            ).mean()

            distill_vloss = 0.5 * (new_value.view(-1) - value_target).square().mean()
            distill_loss = distill_vloss + train_config["distill_beta"] * distill_actloss

            distill_losses[i] = distill_loss.detach()

            if not torch.isfinite(distill_loss).all():
                optimizer_distill.zero_grad()
                print("WARNING: nan/inf values vound in distill_loss! Skipping backprop")
                continue

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
        result_loss=result_losses.mean().item(),
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
        "train/nan/value_loss": math.isnan(train_stats.value_loss) or math.isinf(train_stats.value_loss),
        "train/nan/policy_loss": math.isnan(train_stats.policy_loss) or math.isinf(train_stats.policy_loss),
        "train/nan/entropy_loss": math.isnan(train_stats.entropy_loss) or math.isinf(train_stats.entropy_loss),
        "train/nan/distill_loss": math.isnan(train_stats.distill_loss) or math.isinf(train_stats.distill_loss),
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
        "global/global_num_rollouts": state.global_rollout,
        "global/num_rollouts": state.current_rollout,
        "global/num_timesteps": state.current_timestep,
        "global/num_seconds": state.current_second,
        "global/num_episode": state.current_episode,
    })

    return wlog


def main(config, loglevel, dry_run, no_wandb, seconds_total=float("inf"), save_on_exit=True):
    run_id = config["run"]["id"]
    resumed_config = config["run"]["resumed_config"]

    fcfg = os.path.join(config["run"]["out_dir"], f"{run_id}-config.json")
    msg = f"Saving new config to: {fcfg}"

    if dry_run:
        print(f"{msg} (--dry-run)")
    else:
        os.makedirs(config["run"]["out_dir"], exist_ok=True)
        with open(fcfg, "w") as f:
            print(msg)
            json.dump(config, f, indent=4)

    # assert config["checkpoint"]["interval_s"] > config["eval"]["interval_s"]
    assert config["checkpoint"]["permanent_interval_s"] > config["eval"]["interval_s"]
    assert config["train"]["env"]["kwargs"]["user_timeout"] >= 2 * config["eval"]["interval_s"]

    checkpoint_config = dig(config, "checkpoint")
    train_config = dig(config, "train")
    eval_config = dig(config, "eval")

    logfilename = None if dry_run else os.path.join(config["run"]["out_dir"], f"{run_id}.log")
    logger = StructuredLogger(level=getattr(logging, loglevel), filename=logfilename, context=dict(run_id=run_id))

    logger.info(dict(config=config))

    learning_rate = config["train"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    if train_config.get("torch_detect_anomaly", None):
        torch.autograd.set_detect_anomaly(True)  # debug

    if train_config.get("torch_cuda_matmul", None):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    def create_bot_model_factory(config_file, weights_file):
        with open(config_file, "r") as f:
            loaded_cfg = json.load(f)

        # Bot models must be trained on the opposite side
        train_role = train_config["env"]["kwargs"]["role"]
        bot_role = loaded_cfg["train"]["env"]["kwargs"]["role"]
        assert train_role != bot_role, f"{train_role} != {bot_role}"

        device_type = device.type

        # This function will be called from within another process,
        # (it must not reference external objects which are non-serializable).
        def model_factory():
            weights = torch.load(weights_file, weights_only=True, map_location=device_type)
            model = DNAModel(loaded_cfg["model"], torch.device(device_type)).eval()
            model.load_state_dict(weights, strict=True)
            return model

        return model_factory

    if train_config["env"]["num_envs_per_opponent"]["model"] > 0:
        train_model_factory = create_bot_model_factory(
            train_config["env"]["model"]["config_file"],
            train_config["env"]["model"]["weights_file"],
        )
    else:
        train_model_factory = None

    train_venv = DualVecEnv(
        train_config["env"]["kwargs"],
        train_config["env"]["num_envs_per_opponent"]["StupidAI"],
        train_config["env"]["num_envs_per_opponent"]["BattleAI"],
        train_config["env"]["num_envs_per_opponent"]["model"],
        train_model_factory,
        logprefix="train-",
        e_max=3300
    )

    logger.info("Initialized %d train envs (%s)" % (train_venv.num_envs, train_config["env"]["num_envs_per_opponent"]))

    eval_venv_variants = {}
    for name, envcfg in eval_config["env_variants"].items():
        if envcfg["num_envs_per_opponent"]["model"] > 0:
            eval_model_factory = create_bot_model_factory(
                envcfg["model"]["config_file"],
                envcfg["model"]["weights_file"],
            )
        else:
            eval_model_factory = None

        eval_venv_variants[name] = DualVecEnv(
            envcfg["kwargs"],
            envcfg["num_envs_per_opponent"]["StupidAI"],
            envcfg["num_envs_per_opponent"]["BattleAI"],
            envcfg["num_envs_per_opponent"]["model"],
            eval_model_factory,
            logprefix=f"eval/{name}-",
            e_max=3300
        )

        logger.info("Initialized %d eval envs (variant '%s', %s)" % (eval_venv_variants[name].num_envs, name, envcfg["num_envs_per_opponent"]))

    num_envs = train_venv.num_envs
    num_steps = train_config["num_vsteps"] * num_envs
    batch_size = int(num_steps)
    assert batch_size % train_config["num_minibatches"] == 0, f"{batch_size} % {train_config['num_minibatches']} == 0"
    storage = Storage(train_venv, train_config["num_vsteps"], torch.device("cpu"))  # force storage on cpu
    state = State()

    model = DNAModel(config=config["model"], device=device)
    old_model_policy = copy.deepcopy(model.model_policy).to(device).eval()
    for p in old_model_policy.parameters():
        p.requires_grad = False

    if train_config["torch_compile"]:
        model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
        # XXX: compiling this causes load_state_dict to fail
        # old_model_policy = torch.compile(old_model_policy, mode="max-autotune", fullgraph=True, dynamic=True)

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
        timestamped=False,
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
                    old_model_policy=old_model_policy,
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
            state.global_rollout += 1

        ret_rew = safe_mean(list(state.rollout_rew_queue_1000)[-min(300, state.current_rollout):])
        ret_value = safe_mean(list(state.rollout_net_value_queue_1000)[-min(300, state.current_rollout):])

        return ret_rew, ret_value, save_fn
    finally:
        if save_on_exit:
            save_fn(timestamped=True)
        if os.getenv("VASTAI_INSTANCE_ID") and not dry_run:
            import vastai_sdk
            vastai_sdk.VastAI().label_instance(id=int(os.environ["VASTAI_INSTANCE_ID"]), label="IDLE")


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
