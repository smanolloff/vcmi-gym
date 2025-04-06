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
import ray

from .. import common
from .sampler import Sampler

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
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
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
    reward_step_fixed: int = -1
    reward_dmg_mult: int = 1
    reward_term_mult: int = 1
    swap_sides: int = 0
    true_rng: bool = True  # DEPRECATED
    deprecated_args: list[dict] = field(default_factory=lambda: [
        "encoding_type",
        "consecutive_error_reward_factor",
        "true_rng"
    ])

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class NetworkArgs:
    encoders: dict = field(default_factory=dict)
    heads: dict = field(default_factory=dict)


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
    rollouts_per_log: int = 1
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
    num_samplers: int = 1
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


class HexConvResBlock(nn.Module):
    def __init__(self, channels, depth=1, act={"t": "LeakyReLU"}):
        super().__init__()

        self.layers = []
        for _ in range(depth):
            self.layers.append((
                HexConv(channels),
                AgentNN.build_layer(act),
                HexConv(channels),
            ))

    def forward(self, x):
        assert x.is_contiguous
        for conv1, act, conv2 in self.layers:
            x = act(conv2(act(conv1(x))).add_(x))
        return x


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

    def __init__(self, network, dim_other, dim_hexes, n_actions, device=torch.device("cpu")):
        super().__init__()

        self.dims = {
            "other": dim_other,
            "hexes": dim_hexes,
            "obs": dim_other + dim_hexes,
            "hex": dim_hexes // 165,
            "merged": network.encoders["merged"]["size"]
        }

        self.n_actions = n_actions
        self.device = device

        self.encoders_other = nn.Sequential()
        self.encoders_hex = nn.Sequential()
        self.encoders_merged = nn.Sequential()

        for k in ["other", "hex", "merged"]:
            blocks = network.encoders[k]["blocks"]
            size = network.encoders[k]["size"]

            for _ in range(blocks):
                getattr(self, f"encoders_{k}").append(nn.Sequential(
                    nn.LazyLinear(size),
                    nn.LazyBatchNorm1d(),
                    nn.LeakyReLU(),
                    nn.LazyLinear(self.dims[k]),
                ))

        self.encoder_premerge = nn.LazyLinear(network.encoders["merged"]["size"])
        self.actor = nn.LazyLinear(network.heads["actor"]["size"])
        self.critic = nn.LazyLinear(network.heads["critic"]["size"])

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            self.get_action_and_value(torch.randn([2, self.dims["obs"]]), torch.ones([2, n_actions], dtype=torch.bool, device=device))

        common.layer_init(self.actor, gain=0.01)
        common.layer_init(self.critic, gain=1.0)

    def encode(self, x):
        other, hexes = torch.split(x, [self.dims["other"], self.dims["hexes"]], dim=1)

        z_other = other
        for block in self.encoders_other:
            z_other = block(z_other) + z_other

        z_hexes = hexes.unflatten(dim=1, sizes=[165, self.dims["hex"]])
        for block in self.encoders_hex:
            z_hexes = block(z_hexes) + z_hexes

        z_merged = torch.cat((z_other, z_hexes.flatten(start_dim=1)), dim=-1)
        z_merged = self.encoder_premerge(z_merged)
        for block in self.encoders_merged:
            z_merged = block(z_merged) + z_merged

        return z_merged

    def get_value(self, x, attn_mask=None):
        return self.critic(self.encode(x))

    def get_action(self, x, mask, attn_mask=None, action=None, deterministic=False):
        encoded = self.encode(x)
        action_logits = self.actor(encoded)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), dist

    def get_action_and_value(self, x, mask, attn_mask=None, action=None, deterministic=False):
        encoded = self.encode(x)
        value = self.critic(encoded)
        action_logits = self.actor(encoded)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), dist, value

    # Inference (deterministic)
    def predict(self, b_obs, b_mask):
        with torch.no_grad():
            b_obs = torch.as_tensor(b_obs)
            b_mask = torch.as_tensor(b_mask)

            # Return unbatched action if input was unbatched
            if len(b_mask.shape) == 1:
                b_obs = b_obs.unsqueeze(dim=0)
                b_mask = b_mask.unsqueeze(dim=0)
                b_env_action, _, _, _ = self.get_action(b_obs, b_mask, deterministic=True)
                return b_env_action[0].cpu().item()
            else:
                b_env_action, _, _, _ = self.get_action(b_obs, b_mask, deterministic=True)
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

        attrs = ["args", "dim_other", "dim_hexes", "n_actions", "state"]
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

    def __init__(self, args, dim_other, dim_hexes, n_actions, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self._optimizer_state = None  # needed for save/load
        self.dim_other = dim_other  # needed for save/load
        self.dim_hexes = dim_hexes  # needed for save/load
        self.n_actions = n_actions  # needed for save/load
        self.NN_value = AgentNN(args.network, dim_other, dim_hexes, n_actions, device)
        self.NN_policy = AgentNN(args.network, dim_other, dim_hexes, n_actions, device)
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
    def predict(self, obs, mask, deterministic: bool = False) -> int:
        b_obs = obs.unsqueeze(dim=0)
        b_mask = mask.unsqueeze(dim=0)
        encoded = self.encoder(b_obs)
        action_logits = self.actor(encoded)
        probs = self.categorical_masked(logits0=action_logits, mask=b_mask)

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

    if args.env_version == 10:
        from vcmi_gym import VcmiEnv_v10 as VcmiEnv
    else:
        raise Exception("Unsupported env version: %d" % args.env_version)

    obs_space = VcmiEnv.OBSERVATION_SPACE
    act_space = VcmiEnv.ACTION_SPACE

    if agent is None:
        # TODO: robust mechanism ensuring these don't get mixed up
        dim_other = VcmiEnv.STATE_SIZE_GLOBAL + 2*VcmiEnv.STATE_SIZE_ONE_PLAYER
        dim_hexes = VcmiEnv.STATE_SIZE_HEXES
        n_actions = VcmiEnv.ACTION_SPACE.n
        assert VcmiEnv.STATE_SIZE == dim_other + dim_hexes
        agent = Agent(args, dim_other, dim_hexes, n_actions, device=device)

    # TRY NOT TO MODIFY: seeding
    LOG.info("RNG master seed: %s" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    try:
        if args.wandb_project:
            import wandb
            common.setup_wandb(args, agent, __file__, watch=True)

            # For wandb.log, commit=True by default
            # for wandb_log, commit=False by default
            def wandb_log(data, commit=False):
                loglevel = "info" if commit else "debug"
                getattr(LOG, loglevel)(data)
                wandb.log(data, commit=commit)
        else:
            def wandb_log(data, commit=False):
                loglevel = "info" if commit else "debug"
                getattr(LOG, loglevel)(data)

        common.log_params(args, wandb_log)

        if args.resume:
            agent.state.resumes += 1
            wandb_log({"global/resumes": agent.state.resumes})

        # print("Agent state: %s" % asdict(agent.state))

        assert act_space.shape == ()

        # ALGO Logic: Storage setup
        b_obs = torch.zeros((args.num_steps,) + obs_space["observation"].shape).to(device)
        b_logprobs = torch.zeros(args.num_steps).to(device)
        b_actions = torch.zeros((args.num_steps,) + act_space.shape).to(device)
        b_masks = torch.zeros((args.num_steps, act_space.n), dtype=torch.bool).to(device)
        b_advantages = torch.zeros(args.num_steps).to(device)
        b_returns = torch.zeros(args.num_steps).to(device)
        b_values = torch.zeros(args.num_steps).to(device)

        progress = 0
        map_rollouts = 0
        start_time = time.time()
        global_start_second = agent.state.global_second

        # XXXXX:
        # this is better achieved with Dataset and workers (also async)
        RemoteSampler = ray.remote(Sampler)
        sampler_steps = args.num_steps // args.num_samplers

        def NN_creator():
            return AgentNN(args.network, dim_other, dim_hexes, n_actions, device=torch.device("cpu"))

        def venv_creator():
            return common.create_venv(VcmiEnv, args)

        def sampler_creator(sampler_id):
            return RemoteSampler.remote(0, NN_creator, venv_creator, sampler_steps, args.gamma, args.gae_lambda_policy, args.gae_lambda_value, torch.device("cpu"))

        LOG.info("[main] init %d samplers" % args.num_samplers)
        samplers = [sampler_creator(i) for i in range(args.num_samplers)]

        timer_all = common.Timer()
        timer_sample = common.Timer()
        timer_train = common.Timer()
        timer_value_optimization = common.Timer()
        timer_policy_distillation = common.Timer()

        while progress < 1:
            timer_all.reset()
            timer_sample.reset()
            timer_train.reset()
            timer_value_optimization.reset()
            timer_policy_distillation.reset()

            timer_all.start()

            if args.vsteps_total:
                progress = agent.state.current_vstep / args.vsteps_total
            elif args.seconds_total:
                progress = agent.state.current_second / args.seconds_total
            else:
                progress = 0

            agent.optimizer_value.param_groups[0]["lr"] = lr_schedule_fn_value(progress)
            agent.optimizer_policy.param_groups[0]["lr"] = lr_schedule_fn_policy(progress)
            agent.optimizer_distill.param_groups[0]["lr"] = lr_schedule_fn_distill(progress)

            ep_count = 0

            # LOG.debug("Set weights...")
            ray.get([s.set_weights.remote(agent.NN_value.state_dict(), agent.NN_policy.state_dict()) for s in samplers])

            # LOG.debug("Call samplers...")
            futures = [s.sample.remote() for s in samplers]

            with timer_sample:
                # LOG.debug("Gather results...")
                for i in range(len(futures)):
                    done, futures = ray.wait(futures, num_returns=1)
                    res, stats = ray.get(done[0])

                    # t = timesteps
                    (
                        s_obs,          # => (t, STATE_SIZE)
                        s_logprobs,     # => (t)
                        s_actions,      # => (t)
                        s_masks,        # => (t, N_ACTIONS)
                        s_advantages,   # => (t)
                        s_returns,      # => (t)
                        s_values        # => (t)
                    ) = res

                    (
                        s_seconds,
                        s_episodes,
                        s_ep_net_value,
                        s_ep_is_success,
                        s_ep_return,
                        s_ep_length
                    ) = stats

                    assert all(x.shape[0] == sampler_steps for x in res), [x.shape[0] for x in res]

                    start = sampler_steps * i
                    end = sampler_steps * i + sampler_steps

                    b_obs[start:end] = s_obs
                    b_logprobs[start:end] = s_logprobs
                    b_actions[start:end] = s_actions
                    b_masks[start:end] = s_masks
                    b_advantages[start:end] = s_advantages
                    b_returns[start:end] = s_returns
                    b_values[start:end] = s_values

                    agent.state.ep_net_value_queue.append(s_ep_net_value)
                    agent.state.ep_is_success_queue.append(s_ep_is_success)
                    agent.state.ep_rew_queue.append(s_ep_return)
                    agent.state.ep_length_queue.append(s_ep_length)
                    agent.state.current_episode += s_episodes
                    agent.state.global_episode += s_episodes
                    ep_count += s_episodes

                    agent.state.current_vstep += sampler_steps
                    agent.state.current_timestep += sampler_steps
                    agent.state.global_timestep += sampler_steps
                    agent.state.current_second = int(time.time() - start_time)
                    agent.state.global_second = global_start_second + agent.state.current_second

            b_inds = np.arange(batch_size_policy)
            clipfracs = []

            # Policy network optimization
            with timer_train:
                agent.train()
                for epoch in range(args.update_epochs_policy):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size_policy, minibatch_size_policy):
                        end = start + minibatch_size_policy
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, _ = agent.NN_policy.get_action(
                            b_obs[mb_inds],
                            b_masks[mb_inds],
                            action=b_actions[mb_inds],
                        )
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

            # Value network optimization
            with timer_value_optimization:
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
            with timer_policy_distillation:
                agent.NN_policy.zero_grad(True)  # don't clone gradients

                # deepcopy vs load_tate_dict have basically the same performance
                # For CUDA models, however, state_dict could be more efficient?
                # old_NN_policy = copy.deepcopy(agent.NN_policy).to(device)
                old_NN_policy = AgentNN(args.network, dim_other, dim_hexes, n_actions, device)
                old_NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)

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

                        _, _, _, new_action_dist, new_value = agent.NN_policy.get_action_and_value(
                            b_obs[mb_inds],
                            b_masks[mb_inds],
                        )

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
            ep_len_mean = common.safe_mean(agent.state.ep_length_queue)
            ep_value_mean = common.safe_mean(agent.state.ep_net_value_queue)
            ep_is_success_mean = common.safe_mean(agent.state.ep_is_success_queue)

            if ep_count > 0:
                assert ep_rew_mean is not np.nan
                assert ep_value_mean is not np.nan
                assert ep_is_success_mean is not np.nan
                agent.state.rollout_rew_queue_100.append(ep_rew_mean)
                agent.state.rollout_rew_queue_1000.append(ep_rew_mean)
                agent.state.rollout_net_value_queue_100.append(ep_value_mean)
                agent.state.rollout_net_value_queue_1000.append(ep_value_mean)
                agent.state.rollout_is_success_queue_100.append(ep_is_success_mean)
                agent.state.rollout_is_success_queue_1000.append(ep_is_success_mean)

            tall = timer_all.peek()

            wlog = {
                "params/policy_learning_rate": agent.optimizer_policy.param_groups[0]["lr"],
                "params/value_learning_rate": agent.optimizer_value.param_groups[0]["lr"],
                "params/distill_learning_rate": agent.optimizer_distill.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/distill_loss": distill_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "rollout/ep_count": ep_count,
                "rollout/ep_len_mean": ep_len_mean,
                "rollout100/ep_value_mean": common.safe_mean(agent.state.rollout_net_value_queue_100),
                "rollout1000/ep_value_mean": common.safe_mean(agent.state.rollout_net_value_queue_1000),
                "rollout100/ep_rew_mean": common.safe_mean(agent.state.rollout_rew_queue_100),
                "rollout1000/ep_rew_mean": common.safe_mean(agent.state.rollout_rew_queue_1000),
                "rollout100/ep_success_rate": common.safe_mean(agent.state.rollout_is_success_queue_100),
                "rollout1000/ep_success_rate": common.safe_mean(agent.state.rollout_is_success_queue_1000),
                "global/num_rollouts": agent.state.current_rollout,
                "global/num_timesteps": agent.state.current_timestep,
                "global/num_seconds": agent.state.current_second,
                "global/num_episode": agent.state.current_episode,

                "timer/sample": timer_sample.peek() / tall,
                "timer/train": timer_train.peek() / tall,
                "timer/value_optimization": timer_value_optimization.peek() / tall,
                "timer/policy_distillation": timer_policy_distillation.peek() / tall,
                "timer/other": tall - (timer_sample.peek() + timer_train.peek() + timer_value_optimization.peek() + timer_policy_distillation.peek()),
            }

            if rollouts_total:
                wlog["global/progress"] = progress

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
                wlog["global/global_num_timesteps"] = agent.state.global_timestep
                wlog["global/global_num_seconds"] = agent.state.global_second
                wandb_log(wlog, commit=True)

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
            else:
                wandb_log(wlog, commit=False)

            agent.state.current_rollout += 1
            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent, out_dir)
            # print("TRAIN TIME: %.2f" % (time.time() - tstart))

    finally:
        common.maybe_save(0, 10e9, args, agent, out_dir)
        if "samplers" in locals():
            ray.get([s.shutdown.remote() for s in samplers])

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
        "mppo_dna_ray",
        "MDR",
        loglevel=logging.DEBUG,
        run_name=None,
        trial_id=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="/var/folders/m3/8p3yhh9171sbnhc7j_2xpk880000gn/T/x.pt",
        agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_log=1,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="attacker",
        save_every=2000000000,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=0,
        out_dir_template="data/mppo_dna-test/mppo_dna-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.0001),
        lr_schedule_value=ScheduleArgs(mode="const", start=0.0001),
        lr_schedule_policy=ScheduleArgs(mode="const", start=0.0001),
        lr_schedule_distill=ScheduleArgs(mode="const", start=0.0001),
        num_envs=1,  # always 1 (use num_samplers for parallel sampling)
        num_steps=1000,
        num_samplers=5,  # (num_steps // num_samplers) timesteps per sampler
        gamma=0.85,
        gae_lambda=0.9,
        gae_lambda_policy=0.95,
        gae_lambda_value=0.95,
        num_minibatches=2,
        num_minibatches_value=2,
        num_minibatches_policy=2,
        num_minibatches_distill=2,
        update_epochs=2,
        update_epochs_value=2,
        update_epochs_policy=2,
        update_epochs_distill=2,
        norm_adv=True,
        clip_coef=0.5,
        clip_vloss=True,
        ent_coef=0.05,
        max_grad_norm=1,
        distill_beta=1.0,
        target_kl=None,
        logparams={},
        cfg_file=None,
        seed=42,
        skip_wandb_init=False,
        skip_wandb_log_code=False,
        envmaps=["gym/generated/4096/4x1024.vmap"],
        env=EnvArgs(
            random_terrain_chance=100,
            tight_formation_chance=0,
            max_steps=500,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            random_heroes=1,
            random_obstacles=1,
            town_chance=10,
            warmachine_chance=40,
            mana_min=0,
            mana_max=0,
            reward_step_fixed=-1,
            reward_dmg_mult=1,
            reward_term_mult=1,
            swap_sides=0,
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
            conntype="thread"
        ),
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        env_wrappers=[dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyObservationSpaceWrapper")],
        env_version=10,
        network={
            "encoders": {
                "other": {"blocks": 3, "size": 10},
                "hex": {"blocks": 3, "size": 10},
                "merged": {"blocks": 3, "size": 10},
            },
            "heads": {
                "actor": {"size": 2312},
                "critic": {"size": 1}
            }
        }
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
