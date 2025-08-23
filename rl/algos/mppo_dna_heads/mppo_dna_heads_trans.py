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

from rl.algos.common import CategoricalMasked
from rl.world.util.structured_logger import StructuredLogger
from rl.world.util.persistence import load_checkpoint, save_checkpoint
from rl.world.util.wandb import setup_wandb
from rl.world.util.timer import Timer
from rl.world.util.misc import dig, safe_mean, timer_stats

from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper

from vcmi_gym.envs.v12.pyconnector import (
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


class TensorStorage:
    def __init__(self, venv, num_vsteps, device):
        s = num_vsteps
        e = venv.num_envs
        ospace = venv.single_observation_space
        aspace = venv.single_action_space

        self.obs = torch.zeros((s, e) + ospace.shape, device=device)
        self.logprobs = torch.zeros(s, e, device=device)
        self.actions = torch.zeros((s, e) + aspace.shape, device=device)

        self.rewards = torch.zeros((s, e), device=device)
        self.dones = torch.zeros((s, e), device=device)
        self.values = torch.zeros((s, e), device=device)
        self.next_obs = torch.as_tensor(venv.reset()[0], device=device)
        self.next_done = torch.zeros(e, device=device)

        self.next_value = torch.zeros(e, device=device)  # needed for GAE
        self.advantages = torch.zeros(s, e, device=device)
        self.returns = torch.zeros(s, e, device=device)

        self.device = device


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
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        self.dim_other = STATE_SIZE - STATE_SIZE_HEXES
        self.dim_hexes = STATE_SIZE_HEXES

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

        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0), persistent=False)
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1, persistent=False)
        self.register_buffer("action_table", action_table, persistent=False)
        self.register_buffer("inverse_table", inverse_table, persistent=False)

        dmodel = config["d_model"]

        self.encoder_other = nn.Sequential(nn.Linear(self.dim_other, dmodel), nn.LayerNorm(dmodel))
        self.encoder_hexes = nn.Sequential(nn.Linear(STATE_SIZE_ONE_HEX, dmodel), nn.LayerNorm(dmodel))
        self.pos_hex = nn.Parameter(torch.zeros(165, dmodel))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dmodel, nhead=8, batch_first=True),
            num_layers=config["num_layers"]
        )

        self.Wq_critic = nn.Linear(dmodel, dmodel, bias=False)   # query from global
        self.Wk_critic = nn.Linear(dmodel, dmodel, bias=False)   # keys from hexes
        self.Wv_critic = nn.Linear(dmodel, dmodel, bias=False)   # values from hexes
        self.head_value = nn.Sequential(
            nn.Linear(4*dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, 1)
        )

        self.head_act0 = nn.Sequential(
            nn.Linear(2*dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, 1)
        )

        self.proj_g = nn.Linear(dmodel, dmodel)
        self.emb_act0 = nn.Embedding(len(MainAction), dmodel)
        self.nheads_attn_act0 = 8
        assert dmodel % 8 == 0, f"{dmodel} % 8 == 0"
        self.Wq_act0 = nn.Linear(dmodel, dmodel, bias=False)
        self.Wk_act0 = nn.Linear(dmodel, dmodel, bias=False)
        self.Wv_act0 = nn.Linear(dmodel, dmodel, bias=False)

        # hex1 | act0 : bilinear score s_i = <Wq e_a, Wk h_i>
        self.Wq_hex1 = nn.Linear(dmodel, dmodel, bias=False)
        self.Wk_hex1 = nn.Linear(dmodel, dmodel, bias=False)

        # hex2 | (act0, hex1) : cross-attention
        # q = Wq2 [e_a ; h_i]  (concat), K = Wk2 H, logits_j = q · K_j
        self.Wq_hex2 = nn.Linear(2 * dmodel, dmodel, bias=False)
        self.Wk_hex2 = nn.Linear(dmodel, dmodel, bias=False)

        # # Init lazy layers (must be before weight/bias init)
        # with torch.no_grad():
        #     self.get_actdata_and_value(torch.randn([2, STATE_SIZE]))

        def xavier_init(linlayer):
            nn.init.xavier_uniform_(linlayer.weight)
            if linlayer.bias:
                nn.init.zeros_(linlayer.bias)

        # torch initializes Linear layers with kaiming by default
        # => (which is OK for layers followed by ReLU)
        nn.init.xavier_uniform_(self.encoder_other[0].weight)
        nn.init.xavier_uniform_(self.encoder_hexes[0].weight)
        nn.init.xavier_uniform_(self.head_value[2].weight)
        nn.init.xavier_uniform_(self.head_act0[2].weight)
        nn.init.xavier_uniform_(self.proj_g.weight)
        nn.init.xavier_uniform_(self.Wq_act0.weight)
        nn.init.xavier_uniform_(self.Wk_act0.weight)
        nn.init.xavier_uniform_(self.Wv_act0.weight)
        nn.init.xavier_uniform_(self.Wq_hex1.weight)
        nn.init.xavier_uniform_(self.Wk_hex1.weight)
        nn.init.xavier_uniform_(self.Wq_hex2.weight)
        nn.init.xavier_uniform_(self.Wk_hex2.weight)

        self.idx_cache = {}

    def _idx(self, name, size):
        if name not in self.idx_cache:
            self.idx_cache[name] = {}

        if size not in self.idx_cache[name]:
            self.idx_cache[name][size] = torch.arange(size, dtype=torch.long, device=self.device)

        return self.idx_cache[name][size]

    def encode(self, x):
        other, hexes = torch.split(x, [self.dim_other, self.dim_hexes], dim=1)
        hexes = hexes.reshape(-1, 165, STATE_SIZE_ONE_HEX)
        z_other = self.encoder_other(other)                         # (B, d)
        z_hexes = self.encoder_hexes(hexes) + self.pos_hex          # (B, 165, d)
        tr_in = torch.cat((z_other.unsqueeze(1), z_hexes), dim=1)   # (B, 166, d)
        return self.transformer(tr_in)                              # (B, 166, d)

    def _get_value(self, z):
        z_g = z[:, 0]
        z_h = z[:, 1:]

        q = self.Wq_critic(z_g).unsqueeze(1)                            # (B,1,d)
        k = self.Wk_critic(z_h)                                         # (B,N,d)
        v = self.Wv_critic(z_h)                                         # (B,N,d)
        ctx = F.scaled_dot_product_attention(q, k, v).squeeze(1)        # (B,d)
        feat = torch.cat([z_g, z_h.mean(1), z_h.amax(1), ctx], dim=-1)  # (B,4d)
        return self.head_value(feat)                                    # (B,1)

    def _get_actdata_eval(self, z, obs):
        z_g = z[:, 0]
        z_h = z[:, 1:]
        b_inds = self._idx("B", obs.size(0))

        act0, hex1, hex2 = None, None, None
        mask_act0, mask_hex1, mask_hex2 = self._build_masks(obs)

        # act0
        logits_act0 = self._act0_logits(z_g, z_h, mask_hex1)
        dist_act0 = CategoricalMasked(logits=logits_act0, mask=mask_act0)
        act0 = dist_act0.sample()
        act0_emb = self.emb_act0(act0)

        # hex1 | act0
        logits_hex1 = self._hex1_logits(z_h, act0_emb)
        dist_hex1 = CategoricalMasked(logits=logits_hex1, mask=mask_hex1[b_inds, act0])
        hex1 = dist_hex1.sample()

        # hex2 | (act0, hex1)
        logits_hex2 = self._hex2_logits(z_h, act0_emb, hex1, b_inds)
        dist_hex2 = CategoricalMasked(logits=logits_hex2, mask=mask_hex2[b_inds, act0, hex1])
        hex2 = dist_hex2.sample()

        action = self.action_table[act0, hex1, hex2]

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    # torch compile-friendly (no python conditionals) get_actdata for training
    def _get_actdata_train(self, z, obs, action):
        z_g = z[:, 0]
        z_h = z[:, 1:]
        b_inds = self._idx("B", obs.size(0))

        act0, hex1, hex2 = self.inverse_table[action].unbind(1)
        mask_act0, mask_hex1, mask_hex2 = self._build_masks(obs)

        # act0
        logits_act0 = self._act0_logits(z_g, z_h, mask_hex1)
        dist_act0 = CategoricalMasked(logits=logits_act0, mask=mask_act0)
        act0_emb = self.emb_act0(act0)

        # hex1 | act0
        logits_hex1 = self._hex1_logits(z_h, act0_emb)
        dist_hex1 = CategoricalMasked(logits=logits_hex1, mask=mask_hex1[b_inds, act0])

        # hex2 | (act0, hex1)
        logits_hex2 = self._hex2_logits(z_h, act0_emb, hex1, b_inds)
        dist_hex2 = CategoricalMasked(logits=logits_hex2, mask=mask_hex2[b_inds, act0, hex1])

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    def _act0_logits(self, z_g, z_h, mask_hex1):
        B, _, d = z_h.shape
        A0 = len(MainAction)

        # Queries per action, conditioned on global
        g_proj = self.proj_g(z_g).unsqueeze(1)                              # (B, 1, d)
        q = self.emb_act0.weight.unsqueeze(0).expand(B, A0, d) + g_proj     # (B, A0, d)
        k = self.Wk_act0(z_h)   # (B,N,d)
        v = self.Wv_act0(z_h)   # (B,N,d)

        nhead = self.nheads_attn_act0
        dh = d // nhead
        q = q.view(B, A0, nhead, dh).transpose(1, 2)    # (B,nhead,A0,dh)
        k = k.view(B, 165, nhead, dh).transpose(1, 2)   # (B,nhead,165,dh)
        v = v.view(B, 165, nhead, dh).transpose(1, 2)   # (B,nhead,165,dh)

        # Cross-attend actions (queries) over hexes (keys/values)
        # Attn mask: True = disallowed attention position
        # PyTorch requires a 3D attn_mask of shape (B*num_heads, Tq, Tk)
        attn_mask = ~mask_hex1.unsqueeze(1)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B,H,A0,dh)
        ctx = ctx.transpose(1, 2).reshape(B, A0, d)                         # (B,A0,d)
        ctx = ctx * mask_hex1.any(-1, keepdim=True)

        # Final per-action features and logits
        feat = torch.cat([g_proj.expand_as(ctx), ctx], dim=-1)      # (B, A0, 2d)
        return self.head_act0(feat).squeeze(-1)                     # (B, A0)

    def _hex1_logits(self, z_h, act0_emb):
        d = z_h.size(2)
        qin = act0_emb
        q_hex1 = self.Wq_hex1(qin).unsqueeze(1)         # (B, 1, d)
        k_hex1 = self.Wk_hex1(z_h)                      # (B, 165, d)
        # XXX: the einsum is the same as matmul with the 2nd matrix transposed:
        # return torch.einsum("bd,bnd->bn", q_hex1, k_hex1) / (d ** 0.5)  # (B,N)
        return (q_hex1 @ k_hex1.transpose(1, 2)).squeeze(1) / (d ** 0.5)

    def _hex2_logits(self, z_h, act0_emb, hex1, b_inds):
        b, _, d = z_h.shape
        z_hex1 = z_h[b_inds, hex1]                      # (B, d)
        qin = torch.cat([act0_emb, z_hex1], dim=-1)     # (B, 1, 2d)
        q_hex2 = self.Wq_hex2(qin).unsqueeze(1)         # (B, 1, d)
        k_hex2 = self.Wk_hex2(z_h)                      # (B, N, d)
        return (q_hex2 @ k_hex2.transpose(1, 2)).squeeze(1) / (d ** 0.5)

    def _build_masks(self, obs):
        B = obs.size(0)
        A0 = len(MainAction)

        # 1. MASK_HEX1 - ie. allowed hex#1 for each action
        mask_hex1 = torch.zeros(B, A0, 165, dtype=torch.bool, device=obs.device)
        hexobs = obs[:, -STATE_SIZE_HEXES:].view([-1, 165, STATE_SIZE_ONE_HEX])

        # 1.1 for 0=WAIT: nothing to do (all zeros)
        # 1.2 for 1=MOVE: Take MOVE bit from obs's action mask
        movemask = hexobs[:, :, HEX_ATTR_MAP["ACTION_MASK"][1] + HEX_ACT_MAP["MOVE"]]
        mask_hex1[:, 1, :] = movemask

        # 1.3 for 2=AMOVE: Take any(AMOVEX) bits from obs's action mask
        amovemask = hexobs[:, :, self._idx("amove", 12) + HEX_ATTR_MAP["ACTION_MASK"][1]].bool()
        mask_hex1[:, 2, :] = amovemask.any(dim=-1)

        # 1.4 for 3=SHOOT: Take SHOOT bit from obs's action mask
        shootmask = hexobs[:, :, HEX_ATTR_MAP["ACTION_MASK"][1] + HEX_ACT_MAP["SHOOT"]]
        mask_hex1[:, 3, :] = shootmask

        # 2. MASK_HEX2 - ie. allowed hex2 for each (action, hex1) combo
        mask_hex2 = torch.zeros([B, A0, 165, 165], dtype=torch.bool, device=obs.device)

        # 2.1 for 0=WAIT: nothing to do (all zeros)
        # 2.2 for 1=MOVE: nothing to do (all zeros)
        # 2.3 for 2=AMOVE: For each SRC hex, create a DST hex mask of allowed hexes
        dest = self.amove_hexes.expand(B, -1, -1)
        valid = amovemask & self.amove_hexes_valid.expand_as(dest)
        b_idx = self._idx("B", B).view(B, 1, 1).expand_as(dest)
        s_idx = self._idx("srchex", 165).view(1, 165, 1).expand_as(dest)

        # Select only valid triples and write
        b_sel = b_idx[valid]
        s_sel = s_idx[valid]
        t_sel = dest[valid]

        mask_hex2[b_sel, 2, s_sel, t_sel] = True

        # 2.4 for 3=SHOOT: nothing to do (all zeros)

        # 3. MASK_ACT0 - ie. allowed main action mask
        mask_act0 = torch.zeros(B, A0, dtype=torch.bool, device=obs.device)

        # 0=WAIT
        mask_act0[:, 0] = obs[:, GLOBAL_ATTR_MAP["ACTION_MASK"][1] + GLOBAL_ACT_MAP["WAIT"]]

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_act0[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1)

        return mask_act0, mask_hex1, mask_hex2

    def get_actdata_train(self, obs, action):
        z = self.encode(obs)
        return self._get_actdata_train(z, obs, action)

    def get_actdata_eval(self, obs):
        z = self.encode(obs)
        return self._get_actdata_eval(z, obs)

    def get_value(self, obs):
        z = self.encode(obs)
        return self._get_value(z)


class DNAModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model_policy = Model(config, device)
        self.model_value = Model(config, device)
        self.device = device
        self.to(device)

    def get_actdata(self, obs):
        return self.model_policy.get_actdata_eval(obs)

    def get_value(self, obs):
        return self.model_value.get_value(obs)


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
    assert not torch.is_grad_enabled()

    stats = SampleStats()
    device = model.device

    assert num_vsteps == storage.obs.size(0)
    num_envs = storage.obs.size(1)
    assert num_envs == venv.num_envs

    for vstep in range(0, num_vsteps):
        logger.debug("(train) vstep: %d" % vstep)
        storage.obs[vstep] = storage.next_obs
        storage.dones[vstep] = storage.next_done

        actdata = model.get_actdata(storage.next_obs)
        value = model.get_value(storage.next_obs)

        storage.values[vstep] = value.flatten()
        storage.actions[vstep] = actdata.action
        storage.logprobs[vstep] = actdata.logprob

        next_obs, reward, terminations, truncations, infos = venv.step(actdata.action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        storage.rewards[vstep] = torch.as_tensor(reward, device=device).flatten()
        storage.next_obs = torch.as_tensor(next_obs, device=device)
        storage.next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)
        storage.next_mask = torch.as_tensor(np.array(venv.call("action_mask")), device=device)

        # See notes/gym_vector.txt
        if "_final_info" in infos:
            done_ids = np.flatnonzero(infos["_final_info"])
            final_infos = infos["final_info"]
            stats.ep_rew_mean += sum(final_infos["episode"]["r"][done_ids])
            stats.ep_len_mean += sum(final_infos["episode"]["l"][done_ids])
            stats.ep_value_mean += sum(final_infos["net_value"][done_ids])
            stats.ep_is_success_mean += sum(final_infos["is_success"][done_ids])
            stats.num_episodes += len(done_ids)

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

    # bootstrap value if not done
    next_value = model.get_value(storage.next_obs).reshape(1, -1)
    storage.next_value = next_value.reshape(1, -1)

    return stats


def eval_model(logger, model, venv, num_vsteps):
    assert not torch.is_grad_enabled()

    stats = SampleStats()

    t = lambda x: torch.as_tensor(x, device=model.device)

    obs, _ = venv.reset()

    for vstep in range(0, num_vsteps):
        logger.debug("(eval) vstep: %d" % vstep)
        obs = t(obs)
        actdata = model.get_actdata(obs)

        obs, rew, term, trunc, info = venv.step(actdata.action.cpu().numpy())

        # See notes/gym_vector.txt
        if "_final_info" in info:
            done_ids = np.flatnonzero(info["_final_info"])
            final_info = info["final_info"]
            stats.ep_rew_mean += sum(final_info["episode"]["r"][done_ids])
            stats.ep_len_mean += sum(final_info["episode"]["l"][done_ids])
            stats.ep_value_mean += sum(final_info["net_value"][done_ids])
            stats.ep_is_success_mean += sum(final_info["is_success"][done_ids])
            stats.num_episodes += len(done_ids)

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

    # XXX: this always returns False for CPU https://github.com/pytorch/pytorch/issues/110966
    # assert torch.is_autocast_enabled()

    # compute advantages
    with torch.no_grad():
        lastgaelam = 0
        num_vsteps = train_config["num_vsteps"]
        assert storage.obs.size(0) == num_vsteps

        for t in reversed(range(num_vsteps)):
            if t == num_vsteps - 1:
                nextnonterminal = 1.0 - storage.next_done
                nextvalues = storage.next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = storage.rewards[t] + train_config["gamma"] * nextvalues * nextnonterminal - storage.values[t]
            storage.advantages[t] = lastgaelam = delta + train_config["gamma"] * train_config["gae_lambda"] * nextnonterminal * lastgaelam

        storage.returns[:] = storage.advantages + storage.values

    # flatten the batch (num_envs, env_samples, *) => (num_steps, *)
    b_obs = storage.obs.flatten(end_dim=1)
    b_logprobs = storage.logprobs.flatten(end_dim=1)
    b_actions = storage.actions.flatten(end_dim=1).long()
    b_advantages = storage.advantages.flatten(end_dim=1)
    b_returns = storage.returns.flatten(end_dim=1)
    b_values = storage.values.flatten(end_dim=1)

    batch_size = b_obs.size(0)
    minibatch_size = int(batch_size // train_config["num_minibatches"])
    b_inds = np.arange(batch_size)
    clipfracs = []

    policy_losses = torch.zeros(train_config["num_minibatches"])
    entropy_losses = torch.zeros(train_config["num_minibatches"])
    value_losses = torch.zeros(train_config["num_minibatches"])
    distill_losses = torch.zeros(train_config["num_minibatches"])

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.policy) epoch: %d" % epoch)
        np.random.shuffle(b_inds)
        for i, start in enumerate(range(0, batch_size, minibatch_size)):
            logger.debug("(train.policy) minibatch: %d" % i)
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]

            newactdata = model.model_policy.get_actdata_train(mb_obs, mb_actions)

            logratio = newactdata.logprob - mb_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if train_config["norm_adv"]:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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
        np.random.shuffle(b_inds)
        for i, start in enumerate(range(0, batch_size, minibatch_size)):
            logger.debug("(train.value) minibatch: %d" % i)
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]

            newvalue = model.model_value.get_value(mb_obs)

            # Value loss
            newvalue = newvalue.view(-1)
            if train_config["clip_vloss"]:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -train_config["clip_coef"],
                    train_config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()
            else:
                # XXX: SIMO: SB3 does not multiply by 0.5 here
                value_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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
        np.random.shuffle(b_inds)
        for i, start in enumerate(range(0, batch_size, minibatch_size)):
            logger.debug("(train.distill) minibatch: %d" % i)
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]

            # Compute policy and value targets
            with torch.no_grad():
                old_actdata = old_model_policy.get_actdata_eval(mb_obs)
                value_target = model.model_value.get_value(mb_obs)

            # XXX: must pass action=<old_action> to ensure masks for hex1 and hex2 are the same
            #     (if actions differ, masks will differ and KLD will become NaN)
            z = model.model_policy.encode(mb_obs)
            new_actdata = model.model_policy._get_actdata_train(z, mb_obs, old_actdata.action)
            new_value = model.model_policy._get_value(z)

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

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
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

    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

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
    storage = TensorStorage(train_venv, train_config["num_vsteps"], device)
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
        "train_config/num_vsteps": train_config["num_vsteps"],
        "train_config/num_minibatches": train_config["num_minibatches"],
        "train_config/update_epochs": train_config["update_epochs"],
        "train_config/gamma": train_config["gamma"],
        "train_config/gae_lambda": train_config["gae_lambda"],
        "train_config/ent_coef": train_config["ent_coef"],
        "train_config/clip_coef": train_config["clip_coef"],
        "train_config/learning_rate": train_config["learning_rate"],
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

    checkpoint_timer = Timer()
    checkpoint_timer.start()
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

            with timers["sample"], torch.inference_mode(), autocast_ctx(True):
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

            # Checkpoint only if we have eval stats
            if checkpoint_timer.peek() > config["checkpoint"]["interval_s"] and eval_multistats.num_episodes > 0:
                logger.info("Time for a checkpoint")
                checkpoint_timer.reset(start=True)
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
                    thread = threading.Thread(target=save_fn, kwargs=dict(uploading_event=uploading_event, config=None))  # no need to save config here
                    thread.start()

            if permanent_checkpoint_timer.peek() > config["checkpoint"]["permanent_interval_s"]:
                permanent_checkpoint_timer.reset(start=True)
                logger.info("Time for a permanent checkpoint")
                thread = threading.Thread(target=save_fn, kwargs=dict(timestamped=True))
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
                wlog["train/learning_rate"] = optimizer_policy.param_groups[0]['lr']
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
