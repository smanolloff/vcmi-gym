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
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import warnings

from .. import common

ENVS = []  # debug


def render():
    print(ENVS[0].render())


@dataclass
class ScheduleArgs:
    # const / lin_decay / exp_decay
    mode: str = "const"
    start: float = 2.5e-4
    end: float = 0
    rate: float = 10


@dataclass
class EnvArgs:
    encoding_type: str = ""  # DEPRECATED
    max_steps: int = 500
    reward_dmg_factor: int = 5
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    sparse_info: bool = True
    step_reward_fixed: int = 0
    step_reward_frac: float = 0
    step_reward_mult: int = 1
    term_reward_mult: int = 0
    consecutive_error_reward_factor: Optional[int] = None  # DEPRECATED
    user_timeout: int = 30
    vcmi_timeout: int = 30
    boot_timeout: int = 30
    conntype: str = "proc"
    random_heroes: int = 1
    random_obstacles: int = 1
    town_chance: int = 0
    warmachine_chance: int = 0
    random_terrain_chance: int = 0
    tight_formation_chance: int = 0
    battlefield_pattern: str = ""
    mana_min: int = 0
    mana_max: int = 0
    swap_sides: int = 0
    reward_clip_tanh_army_frac: int = 1
    reward_army_value_ref: int = 0
    reward_dynamic_scaling: bool = False
    true_rng: bool = True  # DEPRECATED
    deprecated_args: list[dict] = field(default_factory=lambda: [
        "encoding_type",
        "consecutive_error_reward_factor",
        "sparse_info",
        "true_rng"
    ])

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class NetworkArgs:
    attention: dict = field(default_factory=dict)
    encoder: list[dict] = field(default_factory=list)
    actor: list[dict] = field(default_factory=list)
    critic: list[dict] = field(default_factory=list)


@dataclass
class State:
    seed: int = -1
    resumes: int = 0
    map_swaps: int = 0  # DEPRECATED
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0
    global_rollout: int = 0
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

        padded_offsets0 = [-17, -16, -1, 0, 1, 17, 18]
        padded_offsets1 = [-18, -17, -1, 0, 1, 16, 17]
        self.padded_convinds = torch.zeros(11, 15, 7, dtype=int)
        for y in range(1, 12):
            for x in range(1, 16):
                padded_hexind = y * 17 + x
                if y % 2 == 0:
                    padded_offsets = torch.tensor(padded_offsets0)
                else:
                    padded_offsets = torch.tensor(padded_offsets1)
                self.padded_convinds[y-1, x-1] = padded_offsets + padded_hexind

        self.fc = nn.LazyLinear(out_features=out_channels)

    def forward(self, x):
        x = x.reshape(x.shape[0], 11, 15, -1)
        padded_shape = torch.tensor(x.shape) + torch.tensor([0, 2, 2, 0])
        padded_x = torch.zeros(*padded_shape)
        padded_x[:, 1:12, 1:16, :] = x
        padded_x = padded_x.flatten(start_dim=1, end_dim=2)
        fc_input = padded_x[:, self.padded_convinds.flatten(), :].reshape(x.shape[0], 165, -1)
        return self.fc(fc_input)


class HexConvResBlock(nn.Module):
    def __init__(self, channels, act={"t": "LeakyReLU"}):
        super().__init__()

        self.conv1 = HexConv(channels)
        self.conv2 = HexConv(channels)
        self.act = AgentNN.build_layer(act)

    def forward(self, x):
        x = x.reshape(x.shape[0], 165, -1)
        identity = x
        out = self.conv2(self.act(self.conv1(x)))
        return out + identity


class SelfAttention(nn.Module):
    def __init__(self, edim, num_heads=1):
        assert edim % num_heads == 0, f"{edim} % {num_heads} == 0"
        super().__init__()
        self.edim = edim
        self.mha = nn.MultiheadAttention(embed_dim=edim, num_heads=num_heads, batch_first=True)

    def forward(self, b_obs, b_masks=None):
        assert len(b_obs.shape) == 3
        assert b_obs.shape[2] == self.edim

        if b_masks is None:
            res, _ = self.mha(b_obs, b_obs, b_obs, need_weights=False)
            return res
        else:
            assert len(b_masks.shape) == 3
            assert b_masks.shape[0] == b_obs.shape[0], f"{b_masks.shape[0]} == {b_obs.shape[0]}"
            assert b_masks.shape[1] == b_obs.shape[1], f"{b_masks.shape[1]} == {b_obs.shape[1]}"
            assert b_masks.shape[1] == b_masks.shape[2], f"{b_masks.shape[1]} == {b_masks.shape[2]}"
            b_obs = b_obs.flatten(start_dim=1, end_dim=2)
            # => (B, 165, e)
            res, _ = self.mha(b_obs, b_obs, b_obs, attn_mask=b_masks, need_weights=False)
            return res


class AgentNN(nn.Module):
    @staticmethod
    def build_layer(spec):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")
        layer_cls = getattr(torch.nn, t, None) or globals()[t]
        return layer_cls(**kwargs)

    def __init__(self, network, obs_dim, n_actions):
        super().__init__()

        assert n_actions == 1 + 165*8

        self.n_nonhex_actions = 1  # WAIT
        self.n_hex_actions = 8  # MOVE, SHOOT, 6xAMOVE
        self.embedded_actions = torch.nn.functional.one_hot(torch.arange(1 + self.n_hex_actions))

        self.attention = None
        self.obs_dim = obs_dim
        self.encoder = torch.nn.Sequential()
        for spec in network.encoder:
            layer = AgentNN.build_layer(spec)
            self.encoder.append(layer)

        self.actor = torch.nn.Sequential()
        for spec in network.actor:
            layer = AgentNN.build_layer(spec)
            self.actor.append(layer)

        self.critic = torch.nn.Sequential()
        for spec in network.critic:
            layer = AgentNN.build_layer(spec)
            self.critic.append(layer)

        # Init lazy layers
        with torch.no_grad():
            self.get_action_and_value(torch.randn([3, obs_dim]), torch.ones([1, n_actions], dtype=torch.bool))

        # common.layer_init(self.encoder)
        common.layer_init(self.actor, gain=0.01)
        common.layer_init(self.critic, gain=1.0)

        self.temperature = nn.Parameter(torch.ones(1))

    # Notation used:
    #   B = batch
    #   HE = hex embedding dim (1 hex)
    #   AE = hexaction embedding dim (1 hexaction)
    #   N = n_hex_actions

    def get_action_logits(self, b_hex_embs, b_mask):
        b = b_hex_embs.shape[0]
        #
        # Non-hex action
        #
        nonhex_action_embs = self.embedded_actions[0:1]
        # => (1, AE)

        b_nonhex_action_emb_expanded = nonhex_action_embs.unsqueeze(0).expand(b, *nonhex_action_embs.shape)
        # => (B, 1, AE)

        b_hex_embs_mean = b_hex_embs.mean(dim=1, keepdim=True)
        # => (B, 1, HE)

        b_nonhex_action_emb_combined = torch.cat([b_nonhex_action_emb_expanded, b_hex_embs_mean], dim=-1)
        # => (B, 1, HE+AE)

        #
        # Hex actions
        #

        hex_action_embs = self.embedded_actions[1:]
        # => (N, AE)

        b_hex_action_embs_expanded = hex_action_embs.unsqueeze(0).unsqueeze(0).expand(b, 165, *hex_action_embs.shape)
        # => (B, 165, N, AE)

        # b_hex_embs:
        # => (B, 165, HE)

        b_hex_embs_expanded = b_hex_embs.unsqueeze(2).expand(-1, -1, self.n_hex_actions, -1)
        # => (B, 165, N, HE)

        b_hex_embs_combined0 = torch.cat([b_hex_embs_expanded, b_hex_action_embs_expanded], dim=-1)
        # => (B, 165, N, HE+AE)

        b_hex_embs_combined = b_hex_embs_combined0.flatten(start_dim=1, end_dim=2)
        # => (B, 165*N, HE+AE)

        b_combined = torch.cat((b_nonhex_action_emb_combined, b_hex_embs_combined), dim=1)
        # => (B, 1+165*N, HE+AE)

        # Get logits for all hex-action pairs
        b_logits = self.actor(b_combined)
        # => (B, 1+165*N)

        # Apply temperature and mask
        # b_logits = b_logits / self.temperature
        b_logits = b_logits.masked_fill(~b_mask, -1e9)

        # => (B, 1+165*N)
        return b_logits

    def get_action(self, b_obs, b_mask, b_action=None, deterministic=False):
        b_encobs = self.encoder(b_obs)
        b_action_logits = self.get_action_logits(b_encobs, b_mask)
        dist = torch.distributions.Categorical(logits=b_action_logits)
        if b_action is None:
            if deterministic:
                b_action = torch.argmax(dist.probs, dim=1)
            else:
                b_action = dist.sample()

        return b_action, dist.log_prob(b_action), dist.entropy(), dist

    def get_value(self, b_obs):
        return self.critic(self.encoder(b_obs))

    def get_action_and_value(self, b_obs, b_mask, b_action=None, deterministic=False):
        b_encobs = self.encoder(b_obs)
        b_action_logits = self.get_action_logits(b_encobs, b_mask)
        dist = torch.distributions.Categorical(logits=b_action_logits)
        if b_action is None:
            if deterministic:
                b_action = torch.argmax(dist.probs, dim=1)
            else:
                b_action = dist.sample()

        b_value = self.critic(b_encobs)
        return b_action, dist.log_prob(b_action), dist.entropy(), dist, b_value

    # Inference (deterministic)
    def predict(self, b_obs, b_mask):
        with torch.no_grad():
            b_obs = torch.as_tensor(b_obs)
            b_mask = torch.as_tensor(b_mask)

            # Return unbatched action if input was unbatched
            if len(b_mask.shape) == 1:
                b_obs = b_obs.unsqueeze(dim=0)
                b_mask = b_mask.unsqueeze(dim=0)
                b_env_action, _, _, _ = self.get_action_and_value(b_obs, b_mask, deterministic=True)
                return b_env_action[0].cpu().item()
            else:
                b_env_action, _, _, _ = self.get_action_and_value(b_obs, b_mask, deterministic=True)
                return b_env_action.cpu().numpy()


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

        attrs = ["args", "obs_dim", "n_actions", "state"]
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
        attrs = ["args", "obs_dim", "n_actions", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN_value.load_state_dict(agent.NN_value.state_dict(), strict=True)
        clean_agent.NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)
        clean_agent.optimizer_value.load_state_dict(agent.optimizer_value.state_dict())
        clean_agent.optimizer_policy.load_state_dict(agent.optimizer_policy.state_dict())
        clean_agent.optimizer_distill.load_state_dict(agent.optimizer_distill.state_dict())
        jagent = JitAgent()
        jagent.env_version = clean_agent.env_version
        jagent.obs_dim = clean_agent.obs_dim
        jagent.n_actions = clean_agent.n_actions
        jagent.actor = clean_agent.NN_policy.actor
        jagent.critic = clean_agent.NN_value.critic
        jagent_optimized = optimize_for_mobile(torch.jit.script(jagent), preserved_methods=["get_version", "predict", "get_value"])
        jagent_optimized._save_for_lite_interpreter(jagent_file)

    @staticmethod
    def load(agent_file, device="cpu"):
        print("Loading agent from %s (device: %s)" % (agent_file, device))
        return torch.load(agent_file, map_location=device, weights_only=False)

    def __init__(self, args, obs_dim, n_actions, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self._optimizer_state = None  # needed for save/load
        self.obs_dim = obs_dim  # needed for save/load
        self.n_actions = n_actions  # needed for save/load
        self.NN_value = AgentNN(args.network, obs_dim, n_actions)
        self.NN_policy = AgentNN(args.network, obs_dim, n_actions)
        self.NN_value.to(device)
        self.NN_policy.to(device)
        self.optimizer_value = torch.optim.AdamW(self.NN_value.parameters(), eps=1e-5)
        self.optimizer_policy = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.optimizer_distill = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.predict = self.NN_policy.predict
        self.state = state or State()


class JitAgent(nn.Module):
    """ TorchScript version of Agent (inference only) """

    # XXXXXXXXX: JitAgent must be fixed to handle the embeddings

    def __init__(self):
        super().__init__()
        # XXX: these are overwritten after object is initialized
        self.encoder = nn.Identity()
        self.actor_net = nn.Identity()
        self.value_net = nn.Identity()
        self.env_version = 0

    # Inference
    @torch.jit.export
    def predict(self, obs, mask, deterministic: bool = False) -> int:
        b_obs = obs.unsqueeze(dim=0)
        b_mask = mask.unsqueeze(dim=0)

        # v1, v2
        # features = self.features_extractor(b_obs)

        # v3+
        split = self.obs_splitter(b_obs)  # cannot use destructuring assignment
        features = self.features_extractor2_policy(torch.cat(dim=1, tensors=(
            self.features_extractor1_policy_misc(split[0]),
            self.features_extractor1_policy_stacks(split[1]),
            self.features_extractor1_policy_hexes(split[2])
        )))

        action_logits = self.actor(features)
        probs = self.categorical_masked(logits0=action_logits, mask=b_mask)

        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = self.sample(probs, action_logits)

        return action.int().item()

    @torch.jit.export
    def forward(self, obs) -> torch.Tensor:
        b_obs = obs.unsqueeze(dim=0)
        split = self.obs_splitter(b_obs)  # cannot use destructuring assignment
        features = self.features_extractor2_policy(torch.cat(dim=1, tensors=(
            self.features_extractor1_policy_misc(split[0]),
            self.features_extractor1_policy_stacks(split[1]),
            self.features_extractor1_policy_hexes(split[2])
        )))

        return torch.cat(dim=1, tensors=(self.actor(features), self.critic(features)))

    @torch.jit.export
    def get_value(self, obs) -> float:
        b_obs = obs.unsqueeze(dim=0)
        split = self.obs_splitter(b_obs)  # cannot use destructuring assignment
        features = self.features_extractor2_value(torch.cat(dim=1, tensors=(
            self.features_extractor1_value_misc(split[0]),
            self.features_extractor1_value_stacks(split[1]),
            self.features_extractor1_value_hexes(split[2])
        )))

        value = self.critic(features)
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


def main(args):
    LOG = logging.getLogger("mppo_dna")
    LOG.setLevel(args.loglevel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert isinstance(args, Args)

    agent, args = common.maybe_resume(Agent, args, device=device)

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

    if args.env_version == 7:
        from vcmi_gym import VcmiEnv_v7 as VcmiEnv
    else:
        raise Exception("Unsupported env version: %d" % args.env_version)

    obs_space = VcmiEnv.OBSERVATION_SPACE
    act_space = VcmiEnv.ACTION_SPACE

    if agent is None:
        obs_dim = VcmiEnv.STATE_SIZE
        agent = Agent(args, obs_dim, act_space.n - 1, device=device)

    # TRY NOT TO MODIFY: seeding
    LOG.info("RNG master seed: %s" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    try:
        seeds = [np.random.randint(2**31) for i in range(args.num_envs)]
        envs = common.create_venv(VcmiEnv, args, seeds)
        [ENVS.append(e) for e in envs.unwrapped.envs]  # DEBUG

        agent.state.seed = seed

        # these are used by gym's RecordEpisodeStatistics wrapper
        envs.return_queue = agent.state.ep_rew_queue
        envs.length_queue = agent.state.ep_length_queue
        if args.wandb_project:
            import wandb
            common.setup_wandb(args, agent, __file__, watch=False)

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

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, num_envs) + obs_space["observation"].shape).to(device)
        actions = torch.zeros((args.num_steps, num_envs) + act_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, num_envs)).to(device)
        dones = torch.zeros((args.num_steps, num_envs)).to(device)
        values = torch.zeros((args.num_steps, num_envs)).to(device)

        masks = torch.zeros((args.num_steps, num_envs, act_space.n - 1), dtype=torch.bool).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(next_obs, device=device)
        next_done = torch.zeros(num_envs, device=device)
        next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask")), device=device)

        progress = 0
        map_rollouts = 0
        start_time = time.time()
        global_start_second = agent.state.global_second

        while progress < 1:
            if args.vsteps_total:
                progress = agent.state.current_vstep / args.vsteps_total
            elif args.seconds_total:
                progress = agent.state.current_second / args.seconds_total
            else:
                progress = 0

            agent.optimizer_value.param_groups[0]["lr"] = lr_schedule_fn_value(progress)
            agent.optimizer_policy.param_groups[0]["lr"] = lr_schedule_fn_policy(progress)
            agent.optimizer_distill.param_groups[0]["lr"] = lr_schedule_fn_distill(progress)

            # XXX: eval during experience collection
            agent.eval()

            t0 = time.time()
            # print("Collecting experiences...")
            for step in range(0, args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                masks[step] = next_mask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, _ = agent.NN_policy.get_action(next_obs, next_mask)

                    value = agent.NN_value.get_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs, device=device)
                next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask")), device=device)

                # XXX SIMO: SB3 does bootstrapping for truncated episodes here
                # https://github.com/DLR-RM/stable-baselines3/pull/658

                # See notes/gym_vector.txt
                for final_info, has_final_info in zip(infos.get("final_info", []), infos.get("_final_info", [])):
                    # "final_info" must be None if "has_final_info" is False
                    if has_final_info:
                        assert final_info is not None, "has_final_info=True, but final_info=None"
                        agent.state.ep_net_value_queue.append(final_info["net_value"])
                        agent.state.ep_is_success_queue.append(final_info["is_success"])
                        agent.state.current_episode += 1
                        agent.state.global_episode += 1

                agent.state.current_vstep += 1
                agent.state.current_timestep += num_envs
                agent.state.global_timestep += num_envs
                agent.state.current_second = int(time.time() - start_time)
                agent.state.global_second = global_start_second + agent.state.current_second

            t1_sample = time.time()

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.NN_value.get_value(next_obs).reshape(1, -1)

                advantages, _ = compute_advantages(
                    rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_policy
                )
                _, returns = compute_advantages(rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_value)

            # flatten the batch
            b_obs = obs.flatten(end_dim=1)
            b_logprobs = logprobs.flatten(end_dim=1)
            b_actions = actions.flatten(end_dim=1)
            b_masks = masks.flatten(end_dim=1)
            b_advantages = advantages.flatten(end_dim=1)
            b_returns = returns.flatten(end_dim=1)
            b_values = values.flatten(end_dim=1)

            # Policy network optimization
            b_inds = np.arange(batch_size_policy)
            clipfracs = []

            # print("Training (1)...")
            agent.train()
            for epoch in range(args.update_epochs_policy):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_policy, minibatch_size_policy):
                    end = start + minibatch_size_policy
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, _ = agent.NN_policy.get_action(b_obs[mb_inds], b_masks[mb_inds], b_action=b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
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

                    entropy_loss = entropy.mean()
                    policy_loss = pg_loss - args.ent_coef * entropy_loss

                    agent.optimizer_policy.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(agent.NN_policy.parameters(), args.max_grad_norm)
                    agent.optimizer_policy.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # print("Training (2)...")
            # Value network optimization
            for epoch in range(args.update_epochs_value):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_value, minibatch_size_value):
                    end = start + minibatch_size_value
                    mb_inds = b_inds[start:end]

                    newvalue = agent.NN_value.get_value(b_obs[mb_inds])
                    newvalue = newvalue.view(-1)

                    # Value loss
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    agent.optimizer_value.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(agent.NN_value.parameters(), args.max_grad_norm)
                    agent.optimizer_value.step()

            # Value network to policy network distillation
            # print("Training (3)...")
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
                        _, _, _, old_action_dist = old_NN_policy.get_action(b_obs[mb_inds], b_masks[mb_inds])
                        value_target = agent.NN_value.get_value(b_obs[mb_inds])

                    _, _, _, new_action_dist, new_value = agent.NN_policy.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds])

                    # Distillation loss
                    policy_kl_loss = torch.distributions.kl_divergence(old_action_dist, new_action_dist).mean()
                    value_loss = 0.5 * (new_value.view(-1) - value_target).square().mean()
                    distill_loss = value_loss + args.distill_beta * policy_kl_loss

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

            t2_train = time.time()

            if envs.episode_count > 0:
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
            wandb_log({"rollout/ep_count": envs.episode_count})
            wandb_log({"rollout/ep_len_mean": common.safe_mean(envs.length_queue)})

            if envs.episode_count > 0:
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

            envs.episode_count = 0

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

            wandb_log({
                "global/global_num_timesteps": agent.state.global_timestep,
                "global/global_num_seconds": agent.state.global_second,
                "global/global_num_rollouts": agent.state.global_rollout,
                "global/global_num_episodes": agent.state.global_episode,
            }, commit=agent.state.current_rollout > 0 and agent.state.current_rollout % args.rollouts_per_log == 0)

            LOG.debug("rollout=%d vstep=%d rew=%.2f net_value=%.2f is_success=%.2f losses=%.1f|%.1f|%.1f dur=%.1fs|%.1fs|%.1fs" % (
                agent.state.current_rollout,
                agent.state.current_vstep,
                ep_rew_mean,
                ep_value_mean,
                ep_is_success_mean,
                value_loss.item(),
                policy_loss.item(),
                distill_loss.item(),
                t1_sample - t0,
                t2_train - t1_sample,
                t2_train - t0
            ))

            agent.state.current_rollout += 1
            agent.state.global_rollout += 1
            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent, out_dir)
            # print("TRAIN TIME: %.2f" % (time.time() - tstart))

    finally:
        common.maybe_save(0, 10e9, args, agent, out_dir)
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
        "mppo_dna-test",
        "mppo_dna-test",
        loglevel=logging.DEBUG,
        run_name=None,
        trial_id=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="data/PBT-DNA-split-emb-20250106_033328/3bef1_00000/checkpoint_000004/agent.pt",
        agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_mapchange=0,
        rollouts_per_log=1,
        rollouts_per_table_log=100000,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="attacker",
        save_every=2000000000,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=1,
        out_dir_template="data/mppo_dna-test/mppo_dna-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.001),
        lr_schedule_value=ScheduleArgs(mode="const", start=0.001),
        lr_schedule_policy=ScheduleArgs(mode="const", start=0.001),
        lr_schedule_distill=ScheduleArgs(mode="const", start=0.001),
        num_envs=1,
        num_steps=512,
        gamma=0.8,
        gae_lambda=0.9,
        gae_lambda_policy=0.95,
        gae_lambda_value=0.95,
        num_minibatches=4,
        num_minibatches_value=4,
        num_minibatches_policy=4,
        num_minibatches_distill=4,
        update_epochs=4,
        update_epochs_value=4,
        update_epochs_policy=4,
        update_epochs_distill=4,
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
        envmaps=["gym/generated/4096/4096-mixstack-100K-01.vmap"],
        env=EnvArgs(
            max_steps=500,
            reward_dmg_factor=5,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            consecutive_error_reward_factor=-1,
            sparse_info=True,
            step_reward_fixed=-100,
            step_reward_mult=1,
            term_reward_mult=0,
            random_heroes=0,
            random_obstacles=0,
            town_chance=0,
            warmachine_chance=0,
            mana_min=0,
            mana_max=0,
            swap_sides=0,
            reward_clip_tanh_army_frac=1,
            reward_army_value_ref=0,
            reward_dynamic_scaling=False,
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
            conntype="thread"
        ),
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        env_wrappers=[
            dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyObservationSpaceWrapper"),
            dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyActionSpaceWrapper")
        ],
        env_version=7,
        network=dict(
            attention=None,
            encoder=[
                # => (B, 165*E)
                dict(t="HexConv", out_channels=64),
                dict(t="LeakyReLU"),
                # => (B, 165, 64)
                dict(t="HexConv", out_channels=64),
                dict(t="LeakyReLU"),
                # => (B, 165, 64)
            ],
            actor=[
                # HE=64 (encoder out)
                # AE=9 (onehot for 1 nonhex + 8 hex actions)
                # => (B, 1+165*N, HE+AE)
                dict(t="LazyLinear", out_features=64+9),
                dict(t="LeakyReLU"),
                # => (B, 1+165*N, 64)
                dict(t="LazyLinear", out_features=32),
                dict(t="LeakyReLU"),
                # => (B, 1+165*N, 32)
                dict(t="LazyLinear", out_features=1),
                # => (B, 1+165*N, 1)
                dict(t="Flatten")
                # => (B, 1+165*N)
            ],
            critic=[
                # HE=64 (encoder out)
                # => (B, 165, HE)
                dict(t="LazyLinear", out_features=32),
                dict(t="LeakyReLU"),
                # => (B, 165, 32)
                dict(t="Flatten"),
                # => (B, 5280)
                dict(t="LazyLinear", out_features=1024),
                dict(t="LeakyReLU"),
                # => (B, 1024)
                dict(t="LazyLinear", out_features=1)
            ]
        )
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())

    # test_args = Args(**{'run_id': '3bef1_00003', 'group_id': 'PBT-DNA-split-emb-20250106_033328', 'run_name': None, 'trial_id': '3bef1_00003', 'wandb_project': 'vcmi-gym', 'resume': False, 'overwrite': [], 'notes': '', 'tags': ['BattleAI', 'obstacles-random', 'v4'], 'loglevel': logging.DEBUG, 'agent_load_file': '/Users/simo/Projects/vcmi-gym/rl/raytune/../../data/PBT-DNA-split-emb-20250106_033328/3bef1_00003/checkpoint_000006/agent.pt', 'vsteps_total': 0, 'seconds_total': 3600, 'rollouts_per_mapchange': 0, 'rollouts_per_log': 50, 'rollouts_per_table_log': 0, 'success_rate_target': None, 'ep_rew_mean_target': None, 'quit_on_target': False, 'mapside': 'attacker', 'mapmask': '', 'randomize_maps': False, 'permasave_every': 0, 'save_every': 0, 'max_saves': 3, 'out_dir_template': 'data/{group_id}/{run_id}', 'opponent_load_file': None, 'opponent_sbm_probs': [1, 0, 0], 'lr_schedule': {'mode': 'const', 'start': 0.0001064459013883141, 'end': 0, 'rate': 10}, 'lr_schedule_value': {'mode': 'const', 'start': 0.0001064459013883141, 'end': 0, 'rate': 10}, 'lr_schedule_policy': {'mode': 'const', 'start': 0.0001064459013883141, 'end': 0, 'rate': 10}, 'lr_schedule_distill': {'mode': 'const', 'start': 0.0001064459013883141, 'end': 0, 'rate': 10}, 'clip_coef': 0.6, 'clip_vloss': False, 'distill_beta': 1.0, 'ent_coef': 0.04, 'vf_coef': 1.2875, 'gae_lambda': 0.6805263157894736, 'gae_lambda_policy': 0.6805263157894736, 'gae_lambda_value': 0.6805263157894736, 'gamma': 0.69975, 'max_grad_norm': 4.3, 'norm_adv': 0, 'num_envs': 1, 'envmaps': ['gym/generated/4096/4096-mixstack-100K-01.vmap'], 'num_minibatches': 4, 'num_minibatches_distill': 4, 'num_minibatches_policy': 4, 'num_minibatches_value': 4, 'num_steps': 256, 'stats_buffer_size': 100, 'update_epochs': 4, 'update_epochs_distill': 4, 'update_epochs_policy': 4, 'update_epochs_value': 4, 'weight_decay': 0, 'target_kl': None, 'logparams': {'params/ent_coef': 'ent_coef', 'params/gamma': 'gamma', 'params/max_grad_norm': 'max_grad_norm', 'params/num_steps': 'num_steps', 'params/lr_schedule.start': 'lr_schedule.start', 'params/gae_lambda': 'gae_lambda', 'params/num_minibatches': 'num_minibatches', 'params/update_epochs': 'update_epochs', 'params/vf_coef': 'vf_coef', 'params/clip_vloss': 'clip_vloss', 'params/clip_coef': 'clip_coef', 'params/norm_adv': 'norm_adv'}, 'cfg_file': None, 'seed': 0, 'skip_wandb_init': True, 'skip_wandb_log_code': True, 'env': {'encoding_type': '', 'max_steps': 500, 'reward_dmg_factor': 5, 'vcmi_loglevel_global': 'error', 'vcmi_loglevel_ai': 'error', 'vcmienv_loglevel': 'WARN', 'sparse_info': True, 'step_reward_fixed': 0, 'step_reward_frac': -0.001, 'step_reward_mult': 1, 'term_reward_mult': 0, 'consecutive_error_reward_factor': None, 'user_timeout': 600, 'vcmi_timeout': 600, 'boot_timeout': 300, 'conntype': 'thread', 'random_heroes': 1, 'random_obstacles': 1, 'town_chance': 10, 'warmachine_chance': 40, 'random_terrain_chance': 100, 'tight_formation_chance': 0, 'battlefield_pattern': '', 'mana_min': 0, 'mana_max': 0, 'swap_sides': 0, 'reward_clip_tanh_army_frac': 1, 'reward_army_value_ref': 500, 'reward_dynamic_scaling': False, 'true_rng': True, 'deprecated_args': ['encoding_type', 'consecutive_error_reward_factor', 'sparse_info', 'true_rng']}, 'env_version': 6, 'env_wrappers': [{'module': 'vcmi_gym.envs.util.wrappers', 'cls': 'LegacyObservationSpaceWrapper'}], 'network': {'action_embedding_dim': 128, 'encoder': {'type': 'split', 'kwargs': {'z_dims': {'misc': 4, '1stack': 8, '1hex': 8, 'merged': 512, 'out': 256}, 'obs_dims': {'misc': 4, 'stacks': 4580, 'hexes': 9570}}}, 'body': [{'t': 'LazyLinear', 'out_features': 256}, {'t': 'LeakyReLU'}, {'t': 'LazyLinear', 'out_features': 256}, {'t': 'LeakyReLU'}]}})
    # main(test_args)
