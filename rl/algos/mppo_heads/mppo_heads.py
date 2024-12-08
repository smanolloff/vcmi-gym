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
# This file contains a modified version of CleanRL's PPO implementation:
# https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py
import os
import sys
import random
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import warnings

from .. import common

from vcmi_gym import VcmiEnv_v5


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
    max_steps: int = 500
    reward_dmg_factor: int = 5
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    step_reward_fixed: int = 0
    step_reward_frac: float = 0
    step_reward_mult: int = 1
    term_reward_mult: int = 0
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
        "true_rng"
    ])

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class NetworkArgs:
    attention: dict = field(default_factory=dict)
    features_extractor1_misc: list[dict] = field(default_factory=list)
    features_extractor1_stacks: list[dict] = field(default_factory=list)
    features_extractor1_hexes: list[dict] = field(default_factory=list)
    features_extractor2: list[dict] = field(default_factory=list)
    actor_head1: dict = field(default_factory=dict)
    actor_head2: dict = field(default_factory=dict)
    critic: dict = field(default_factory=dict)


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
    mapside: str = "attacker"
    randomize_maps: bool = False
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    lr_schedule: ScheduleArgs = field(default_factory=lambda: ScheduleArgs())
    weight_decay: float = 0.0
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float = None
    num_minibatches: int = 4
    num_steps: int = 128
    stats_buffer_size: int = 100
    update_epochs: int = 4
    num_envs: int = 1
    envmaps: list = field(default_factory=lambda: ["gym/generated/4096/4x1024.vmap"])

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
        if not isinstance(self.network, NetworkArgs):
            self.network = NetworkArgs(**self.network)

        common.coerce_dataclass_ints(self)


class ChanFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class Split(nn.Module):
    def __init__(self, split_size, dim):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.split_size, self.dim)


class ResBlock(nn.Module):
    def __init__(self, channels, activation="LeakyReLU"):
        super().__init__()
        self.block = nn.Sequential(
            common.layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)),
            getattr(nn, activation)(),
            common.layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, edim, num_heads=1):
        assert edim % num_heads == 0, f"{edim} % {num_heads} == 0"
        super().__init__()
        self.edim = edim
        self.mha = nn.MultiheadAttention(embed_dim=edim, num_heads=1, batch_first=True)

    def forward(self, b_obs, b_masks=None):
        assert len(b_obs.shape) == 3
        assert b_obs.shape[2] == self.edim, f"wrong obs shape: {b_obs.shape} (edim={self.edim})"

        if b_masks is None:
            res, _ = self.mha(b_obs, b_obs, b_obs, need_weights=False)
            return res
        else:
            assert b_masks.shape == (b_masks.shape[0], 165, 165), f"wrong b_masks shape: {b_masks.shape} != ({b_masks.shape[0]}, 165, 165)"
            b_obs = b_obs.flatten(start_dim=1, end_dim=2)
            # => (B, 165, e)
            res, _ = self.mha(b_obs, b_obs, b_obs, attn_mask=b_masks, need_weights=False)
            return res


class AgentNN(nn.Module):
    @staticmethod
    def build_layer(spec, obs_dims):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")

        for k, v in kwargs.items():
            if v == "_M_":
                kwargs[k] = obs_dims["misc"]
            if v == "_H_":
                assert obs_dims["hexes"] % 165 == 0
                kwargs[k] = obs_dims["hexes"] // 165
            if v == "_S_":
                assert obs_dims["stacks"] % 20 == 0
                kwargs[k] = obs_dims["stacks"] // 20

        layer_cls = getattr(torch.nn, t, None) or globals()[t]
        return layer_cls(**kwargs)

    def __init__(self, network, action_space, observation_space, obs_dims):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dims = obs_dims

        assert isinstance(obs_dims, dict)
        assert len(obs_dims) == 3
        assert list(obs_dims.keys()) == ["misc", "stacks", "hexes"]  # order is important

        # 2 nonhex actions (RETREAT, WAIT) + 165 hexes*14 actions each
        # Commented assert due to false positives on legacy models
        # assert action_space.n == 2 + (165*14)

        if network.attention:
            layer = AgentNN.build_layer(network.attention, obs_dims)
            self.attention = common.layer_init(layer)
        else:
            self.attention = None

        # XXX: no lazy option for SelfAttention
        with torch.no_grad():
            dummy_outputs = []

        self.obs_splitter = Split(list(obs_dims.values()), dim=1)

        self.features_extractor1_misc = torch.nn.Sequential()
        for spec in network.features_extractor1_misc:
            layer = AgentNN.build_layer(spec, obs_dims)
            self.features_extractor1_misc.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_misc(torch.randn([1, obs_dims["misc"]])))

        for layer in self.features_extractor1_misc:
            common.layer_init(layer)

        self.features_extractor1_stacks = torch.nn.Sequential(
            torch.nn.Unflatten(dim=1, unflattened_size=[20, obs_dims["stacks"] // 20])
        )

        for spec in network.features_extractor1_stacks:
            layer = AgentNN.build_layer(spec, obs_dims)
            self.features_extractor1_stacks.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_stacks(torch.randn([1, obs_dims["stacks"]])))

        for layer in self.features_extractor1_stacks:
            common.layer_init(layer)

        self.features_extractor1_hexes = torch.nn.Sequential(
            torch.nn.Unflatten(dim=1, unflattened_size=[165, obs_dims["hexes"] // 165])
        )

        for spec in network.features_extractor1_hexes:
            layer = AgentNN.build_layer(spec, obs_dims)
            self.features_extractor1_hexes.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            dummy_outputs.append(self.features_extractor1_hexes(torch.randn([1, obs_dims["hexes"]])))

        for layer in self.features_extractor1_hexes:
            common.layer_init(layer)

        self.features_extractor2 = torch.nn.Sequential()
        for spec in network.features_extractor2:
            layer = AgentNN.build_layer(spec, obs_dims)
            self.features_extractor2.append(layer)

        # dummy input to initialize lazy modules
        with torch.no_grad():
            self.features_extractor2(torch.cat(tuple(dummy_outputs), dim=1))

        for layer in self.features_extractor2:
            common.layer_init(layer)

        self.actor_head1 = common.layer_init(AgentNN.build_layer(network.actor_head1, obs_dims), gain=0.01)
        self.actor_head2 = common.layer_init(AgentNN.build_layer(network.actor_head2, obs_dims), gain=0.01)
        self.critic = common.layer_init(AgentNN.build_layer(network.critic, obs_dims), gain=1.0)

    def extract_features(self, x):
        misc, stacks, hexes = self.obs_splitter(x)
        fmisc = self.features_extractor1_misc(misc)
        fstacks = self.features_extractor1_stacks(stacks)
        fhexes = self.features_extractor1_hexes(hexes)
        fcat = torch.cat((fmisc, fstacks, fhexes), dim=1)
        return self.features_extractor2(fcat)

    def get_value(self, x, attn_mask=None):
        if self.attention:
            x = self.attention(x, attn_mask)
        return self.critic(self.extract_features(x))

    def get_action_and_value(self, b_obs, b_mask1, b_mask2, b_action1=None, b_action2=None, attn_mask=None, deterministic=False):
        assert not b_mask1.requires_grad and not b_mask2.requires_grad

        b = b_obs.shape[0]
        b_features = self.extract_features(b_obs)
        value = self.critic(b_features)

        #
        # Head 1 (primary action)
        #

        b_head1_in = b_features
        b_head1_out = self.actor_head1(b_head1_in)
        b_head1_dist = common.CategoricalMasked(logits=b_head1_out, mask=b_mask1)

        if b_action1 is None:
            assert b_action2 is None
            if deterministic:
                b_action1 = torch.argmax(b_head1_dist.probs, dim=1)
            else:
                b_action1 = b_head1_dist.sample()

        b_logprob1 = b_head1_dist.log_prob(b_action1)
        b_entropy1 = b_head1_dist.entropy()

        #
        # Head 2 (hex)
        #

        # XXX (heads): b_action1 is already detached (sampling is non-differentiable)
        # => action2 won't affect gradients of head1

        # XXX (heads): try detaching the b_feature to prevent action2
        # from updating affecting the encoder gradients
        # Not sure it makes too much sense:
        # in case of MOVE, head2 is generally more important
        # in case of AMOVE, head2 is generally less important

        b_head2_in = torch.cat((b_features, b_action1.unsqueeze(-1)), dim=1)
        b_head2_out = self.actor_head2(b_head2_in)
        b_head2_mask = b_mask2[torch.arange(b), b_action1]
        b_head2_dist = common.CategoricalMasked(logits=b_head2_out, mask=b_head2_mask)

        if b_action2 is None:
            if deterministic:
                b_action2 = torch.argmax(b_head2_dist.probs, dim=1)
            else:
                b_action2 = b_head2_dist.sample()

        b_logprob2 = b_head2_dist.log_prob(b_action2)
        b_entropy2 = b_head2_dist.entropy()

        # XXX: fully masked distributions causes numerical instabilities
        #
        # b_mask1 is (B, 13)
        # b_mask2 is (B, 13, 165)
        # b_head2_mask is (B, 165)  (b_mask2, but only for the actions from head1)
        #
        # results:
        # b_action2 logits are (B, 165)     // the 165 hexes for each action1
        # b_logprob2 is (B)
        # b_entropy2 is (B)
        #
        # => zero-out results where the distribution was fully masked
        #    (analogy: imagine plots from LayerNorm, BathNorm, ... site
        #              imagine (B,A,H) as (Z,X,Y)
        #              We have B actions => select the masks for them only
        #              => any(dim=1) returns (Z,Y) aka. (B,A)
        b_results_to_keep = b_head2_mask.any(dim=1)

        #
        # <OR>
        #
        # => zero-out first N masks (where primary actions dont require hex)
        #    (RETREAT is *always* at index 0 => they are always at the start)
        # b_results_to_keep = b_action1 >= 2

        # XXX (heads):
        # DEBUG: compare two methods for calculating valid_mask_2
        # TODO: remove
        if not torch.equal(b_results_to_keep, b_action1 >= 2):
            import ipdb; ipdb.set_trace()  # noqa
            print(1)

        b_logprob2 = b_logprob2.where(b_results_to_keep, 0)
        b_entropy2 = b_entropy2.where(b_results_to_keep, 0)
        # b_action2 = b_action2.where(b_results_to_keep, -1)  # for debugging purposes ONLY

        return (
            (b_action1, b_logprob1, b_entropy1),
            (b_action2, b_logprob2, b_entropy2),
            value
        )

    # Inference (deterministic)
    def predict(self, obs, mask):
        with torch.no_grad():
            obs = torch.as_tensor(obs)
            mask = torch.as_tensor(mask)

            # XXX: assuming input is NOT batched
            assert obs.shape == self.observation_space.shape, f"{obs.shape} == {self.observation_space.shape}"
            b_env_action, _, _, _ = self.get_action_and_value(obs, mask, deterministic=True)
            return b_env_action[0].cpu().item()


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

        attrs = ["args", "observation_space", "action_space", "obs_dims", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
        torch.save(clean_agent, agent_file)

    @staticmethod
    def jsave(agent, jagent_file):
        print("Saving JIT agent to %s" % jagent_file)
        attrs = ["args", "observation_space", "action_space", "obs_dims", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
        jagent = JitAgent()
        jagent.env_version = clean_agent.env_version
        jagent.obs_dims = clean_agent.obs_dims

        # v2-
        # jagent.features_extractor = clean_agent.NN.features_extractor

        # v3+
        jagent.obs_splitter = clean_agent.NN.obs_splitter
        jagent.features_extractor1_misc = clean_agent.NN.features_extractor1_misc
        jagent.features_extractor1_stacks = clean_agent.NN.features_extractor1_stacks
        jagent.features_extractor1_hexes = clean_agent.NN.features_extractor1_hexes
        jagent.features_extractor2 = clean_agent.NN.features_extractor2

        # common
        jagent.actor = clean_agent.NN.actor
        jagent.critic = clean_agent.NN.critic
        torch.jit.save(torch.jit.script(jagent), jagent_file)

    @staticmethod
    def load(agent_file, device="cpu"):
        print("Loading agent from %s (device: %s)" % (agent_file, device))
        return torch.load(agent_file, map_location=device, weights_only=False)

    def __init__(self, args, observation_space, action_space, obs_dims, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self.observation_space = observation_space  # needed for save/load
        self.action_space = action_space  # needed for save/load
        self.obs_dims = obs_dims  # needed for save/load
        self.NN = AgentNN(args.network, action_space, observation_space, obs_dims)
        self.NN.to(device)
        self.optimizer = torch.optim.AdamW(self.NN.parameters(), eps=1e-5)
        self.predict = self.NN.predict
        self.state = state or State()


class JitAgent(nn.Module):
    """ TorchScript version of Agent (inference only) """

    def __init__(self):
        super().__init__()
        # XXX: these are overwritten after object is initialized
        self.obs_splitter = nn.Identity()
        self.features_extractor1_misc = nn.Identity()
        self.features_extractor1_stacks = nn.Identity()
        self.features_extractor1_hexes = nn.Identity()
        self.features_extractor2 = nn.Identity()
        self.actor = nn.Identity()
        self.critic = nn.Identity()
        self.env_version = 0

    # Inference (deterministic)
    # XXX: attention is not handled here
    @torch.jit.export
    def predict(self, obs, mask) -> int:
        with torch.no_grad():
            b_obs = obs.unsqueeze(dim=0)
            b_mask = mask.unsqueeze(dim=0)

            # v2-
            # features = self.features_extractor(b_obs)

            # v3+
            misc, stacks, hexes = self.obs_splitter(b_obs)
            fmisc = self.features_extractor1_misc(misc)
            fstacks = self.features_extractor1_stacks(stacks)
            fhexes = self.features_extractor1_hexes(hexes)
            fcat = torch.cat((fmisc, fstacks, fhexes), dim=1)
            features = self.features_extractor2(fcat)

            action_logits = self.actor(features)
            dist = common.SerializableCategoricalMasked(logits=action_logits, mask=b_mask)
            action = torch.argmax(dist.probs, dim=1)
            return action.int().item()

    @torch.jit.export
    def get_value(self, obs) -> float:
        with torch.no_grad():
            b_obs = obs.unsqueeze(dim=0)
            misc, stacks, hexes = self.obs_splitter(b_obs)
            fmisc = self.features_extractor1_misc(misc)
            fstacks = self.features_extractor1_stacks(stacks)
            fhexes = self.features_extractor1_hexes(hexes)
            fcat = torch.cat((fmisc, fstacks, fhexes), dim=1)
            features = self.features_extractor2(fcat)
            value = self.critic(features)
            return value.float().item()

    @torch.jit.export
    def get_version(self) -> int:
        return self.env_version


def main(args, agent_cls=Agent):
    LOG = logging.getLogger("mppo")
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

    lr_schedule_fn = common.schedule_fn(args.lr_schedule)

    batch_size = int(num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

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

    # if args.env_version == 1:
    #     from vcmi_gym import VcmiEnv_v1 as VcmiEnv
    # elif args.env_version == 2:
    #     from vcmi_gym import VcmiEnv_v2 as VcmiEnv
    # elif args.env_version == 3:
    #     from vcmi_gym import VcmiEnv_v3 as VcmiEnv
    # elif args.env_version == 4:
    #     from vcmi_gym import VcmiEnv_v4 as VcmiEnv
    # else:
    #     raise Exception("Unsupported env version: %d" % args.env_version)

    assert args.env_version == 5, "multi-head PPO supports VCMI Env v5 only"
    VcmiEnv = VcmiEnv_v5

    act_space = VcmiEnv.ACTION_SPACE

    # TODO: robust mechanism ensuring these don't get mixed up
    assert VcmiEnv.STATE_SEQUENCE == ["misc", "stacks", "hexes"]
    obs_dims = dict(
        misc=VcmiEnv.STATE_SIZE_MISC,
        stacks=VcmiEnv.STATE_SIZE_STACKS,
        hexes=VcmiEnv.STATE_SIZE_HEXES,
    )

    # # A dummy env needs to be created to infer the action and obs space
    # # (use `create_venv` to enure the same wrappers are in place)
    # dummy_args = copy.deepcopy(args)
    # dummy_args.num_envs = 1
    # dummy_args.env.mapname = "gym/A1.vmap"
    # dummy_args.env.conntype = "proc"
    # dummy_venv = common.create_venv(VcmiEnv, dummy_args, [1])
    # seeds = [np.random.randint(2**31) for i in range(args.num_envs)]
    # envs = common.create_venv(VcmiEnv, args, seeds)
    # # Do not use env_cls.OBSERVATION_SPACE
    # # (obs space is changed by a wrapper)
    # act_space = dummy_venv.envs[0].action_space
    # obs_space = dummy_venv.envs[0].observation_space
    # dummy_venv.close()
    # del dummy_venv
    act_space = VcmiEnv.ACTION_SPACE
    obs_space = VcmiEnv.OBSERVATION_SPACE

    if agent is None:
        agent = Agent(args, obs_space, act_space, obs_dims, device=device)

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

        # attn = agent.NN.attention is not None
        attn = False

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, num_envs) + obs_space["observation"].shape).to(device)
        actions1 = torch.zeros((args.num_steps, num_envs) + act_space["action_1"].shape).to(device)
        actions2 = torch.zeros((args.num_steps, num_envs) + act_space["action_2"].shape).to(device)
        logprobs1 = torch.zeros((args.num_steps, num_envs)).to(device)
        logprobs2 = torch.zeros((args.num_steps, num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, num_envs)).to(device)
        dones = torch.zeros((args.num_steps, num_envs)).to(device)
        values = torch.zeros((args.num_steps, num_envs)).to(device)

        masks1 = torch.zeros(((args.num_steps, num_envs) + obs_space["action_mask_1"].shape), dtype=torch.bool).to(device)
        masks2 = torch.zeros(((args.num_steps, num_envs) + obs_space["action_mask_2"].shape), dtype=torch.bool).to(device)
        attnmasks = torch.zeros((args.num_steps, num_envs, 165, 165)).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs_dict, _ = envs.reset(seed=agent.state.seed)  # XXX: seed has no effect here
        next_obs = torch.Tensor(next_obs_dict["observation"]).to(device)
        next_done = torch.zeros(num_envs).to(device)
        next_mask1 = torch.as_tensor(next_obs_dict["action_mask_1"]).to(device)
        next_mask2 = torch.as_tensor(next_obs_dict["action_mask_2"]).to(device)

        if attn:
            next_attnmask = torch.as_tensor(np.array(envs.unwrapped.call("attn_mask")), device=device)

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

            agent.optimizer.param_groups[0]["lr"] = lr_schedule_fn(progress)

            # XXX: eval during experience collection
            agent.eval()
            for step in range(0, args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                masks1[step] = next_mask1
                masks2[step] = next_mask2

                if attn:
                    attnmasks[step] = next_attnmask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    head1, head2, value = agent.NN.get_action_and_value(
                        next_obs,
                        next_mask1,
                        next_mask2,
                        attn_mask=next_attnmask if attn else None
                    )
                    values[step] = value.flatten()

                action1, logprob1, _ = head1
                action2, logprob2, _ = head2

                actions1[step] = action1
                actions2[step] = action2
                logprobs1[step] = logprob1
                logprobs2[step] = logprob2

                # TRY NOT TO MODIFY: execute the game and log data.
                env_action = {"action_1": action1.cpu().numpy(), "action_2": action2.cpu().numpy()}
                next_obs_dict, reward, terminations, truncations, infos = envs.step(env_action)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs_dict["observation"], device=device)
                next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)
                next_mask1 = torch.as_tensor(next_obs_dict["action_mask_1"]).to(device)
                next_mask2 = torch.as_tensor(next_obs_dict["action_mask_2"]).to(device)

                if attn:
                    next_attnmask = torch.as_tensor(np.array(envs.unwrapped.call("attn_mask")), device=device)

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

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.NN.get_value(
                    next_obs,
                    attn_mask=next_attnmask if attn else None
                ).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.flatten(end_dim=1)
            b_logprobs1 = logprobs1.flatten(end_dim=1)
            b_logprobs2 = logprobs2.flatten(end_dim=1)
            b_actions1 = actions1.flatten(end_dim=1)
            b_actions2 = actions2.flatten(end_dim=1)
            b_masks1 = masks1.flatten(end_dim=1)
            b_masks2 = masks2.flatten(end_dim=1)
            b_advantages = advantages.flatten(end_dim=1)
            b_returns = returns.flatten(end_dim=1)
            b_values = values.flatten(end_dim=1)

            if attn:
                b_attn_masks = attnmasks.flatten(end_dim=1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs1 = []
            clipfracs2 = []

            # XXX: train during optimization
            agent.train()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    head1, head2, newvalue = agent.NN.get_action_and_value(
                        b_obs[mb_inds],
                        b_masks1[mb_inds],
                        b_masks2[mb_inds],
                        attn_mask=b_attn_masks[mb_inds] if attn else None,
                        b_action1=b_actions1.long()[mb_inds],
                        b_action2=b_actions2.long()[mb_inds]
                    )

                    _, newlogprob1, entropy1 = head1
                    _, newlogprob2, entropy2 = head2

                    logratio1 = newlogprob1 - b_logprobs1[mb_inds]
                    logratio2 = newlogprob2 - b_logprobs2[mb_inds]
                    ratio1 = logratio1.exp()
                    ratio2 = logratio2.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl1 = (-logratio1).mean()
                        old_approx_kl2 = (-logratio2).mean()
                        approx_kl1 = ((ratio1 - 1) - logratio1).mean()
                        approx_kl2 = ((ratio2 - 1) - logratio2).mean()
                        clipfracs1 += [((ratio1 - 1.0).abs() > args.clip_coef).float().mean().item()]
                        clipfracs2 += [((ratio2 - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss11 = -mb_advantages * ratio1
                    pg_loss12 = -mb_advantages * ratio2
                    pg_loss21 = -mb_advantages * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss22 = -mb_advantages * torch.clamp(ratio2, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss1 = torch.max(pg_loss11, pg_loss12).mean()
                    pg_loss2 = torch.max(pg_loss21, pg_loss22).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
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

                    entropy_loss1 = entropy1.mean()
                    entropy_loss2 = entropy2.mean()

                    loss1 = pg_loss1 - args.ent_coef * entropy_loss1  # + v_loss * args.vf_coef
                    loss2 = pg_loss2 - args.ent_coef * entropy_loss2  # + v_loss * args.vf_coef

                    # XXX (heads): zero-out loss2 where action2 is -1
                    # XXX (heads): Redundant?
                    #     entropy and logprobs are already zeroed
                    #     Even though 0.exp()=1 which is used in loss calc,
                    #     the grads after backprop are still 0.
                    # loss2 = loss2.where(action2 == -1, 0)

                    # XXX (heads): maybe do weighted sum instead?
                    loss = loss1 + loss2 + v_loss * args.vf_coef

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.NN.parameters(), args.max_grad_norm)
                    agent.optimizer.step()

                # XXX (heads): no sense in summing approx_kl (unless its weighted)
                approx_kl = (approx_kl1 + approx_kl2).mean()
                if args.target_kl is not None and approx_kl.mean() > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            ep_rew_mean = common.safe_mean(agent.state.ep_rew_queue)
            ep_value_mean = common.safe_mean(agent.state.ep_net_value_queue)
            ep_is_success_mean = common.safe_mean(agent.state.ep_is_success_queue)

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

            wandb_log({"params/learning_rate": agent.optimizer.param_groups[0]["lr"]})
            wandb_log({"losses/total_loss": loss.item()})
            wandb_log({"losses/value_loss": v_loss.item()})
            wandb_log({"losses/policy_loss1": pg_loss1.item()})
            wandb_log({"losses/policy_loss2": pg_loss2.item()})
            wandb_log({"losses/entropy1": entropy_loss1.item()})
            wandb_log({"losses/entropy2": entropy_loss2.item()})
            wandb_log({"losses/old_approx_kl1": old_approx_kl1.item()})
            wandb_log({"losses/old_approx_kl2": old_approx_kl2.item()})
            wandb_log({"losses/approx_kl": approx_kl.item()})
            wandb_log({"losses/clipfrac1": np.mean(clipfracs1)})
            wandb_log({"losses/clipfrac2": np.mean(clipfracs2)})
            wandb_log({"losses/explained_variance": explained_var})
            wandb_log({"rollout/ep_count": envs.episode_count})
            wandb_log({"rollout/ep_len_mean": common.safe_mean(envs.length_queue)})

            if envs.episode_count > 0:
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

            if agent.state.current_rollout > 0 and agent.state.current_rollout % args.rollouts_per_log == 0:
                wandb_log({
                    "global/global_num_timesteps": agent.state.global_timestep,
                    "global/global_num_seconds": agent.state.global_second,
                    "global/global_num_episodes": agent.state.global_episode,
                }, commit=True)  # commit on final log line

                LOG.debug("rollout=%d vstep=%d rew=%.2f net_value=%.2f is_success=%.2f loss=%.2f" % (
                    agent.state.current_rollout,
                    agent.state.current_vstep,
                    ep_rew_mean,
                    ep_value_mean,
                    ep_is_success_mean,
                    loss.item()
                ))

            agent.state.current_rollout += 1
            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent, out_dir)

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
        "mppo-test",
        "mppo-test",
        loglevel=logging.DEBUG,
        run_name=None,
        trial_id=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="/Users/simo/Projects/vcmi-gym/data/mppo-test/mppo-test/agent-1733620580.pt",
        # agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_mapchange=0,
        rollouts_per_log=1,
        rollouts_per_table_log=100000,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="defender",
        save_every=10,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=1,
        out_dir_template="data/mppo-test/mppo-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.001),
        # envmaps=["gym/generated/4096/4x1024.vmap"],
        envmaps=["gym/A1.vmap"],
        # num_steps=64,
        num_steps=4,
        gamma=0.8,
        gae_lambda=0.8,
        num_minibatches=2,
        # num_minibatches=16,
        # update_epochs=2,
        update_epochs=2,
        norm_adv=True,
        clip_coef=0.3,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=1.2,
        max_grad_norm=0.5,
        target_kl=None,
        logparams={},
        cfg_file=None,
        seed=42,
        skip_wandb_init=False,
        skip_wandb_log_code=False,
        env=EnvArgs(
            max_steps=500,
            reward_dmg_factor=5,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
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
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
            conntype="proc"
        ),
        env_wrappers=[],
        env_version=5,
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        network=dict(
            attention=None,
            features_extractor1_misc=[
                # => (B, M)
                dict(t="LazyLinear", out_features=4),
                dict(t="LeakyReLU"),
                # => (B, 2)
            ],
            features_extractor1_stacks=[
                # => (B, 20, S)
                dict(t="LazyLinear", out_features=64),
                dict(t="LeakyReLU"),
                dict(t="Linear", in_features=64, out_features=16),
                # => (B, 20, 16)

                dict(t="Flatten"),
                # => (B, 320)
            ],
            features_extractor1_hexes=[
                # => (B, 165, H)
                dict(t="LazyLinear", out_features=32),
                dict(t="LeakyReLU"),
                dict(t="Linear", in_features=32, out_features=16),
                # => (B, 165, 16)

                dict(t="Flatten"),
                # => (B, 2640)
            ],
            features_extractor2=[
                # => (B, 2964)
                dict(t="LeakyReLU"),
                dict(t="LazyLinear", out_features=512),
                dict(t="LeakyReLU"),
            ],
            actor_head1=dict(t="Linear", in_features=512, out_features=13),
            actor_head2=dict(t="Linear", in_features=513, out_features=165),
            critic=dict(t="Linear", in_features=512, out_features=1)
        )
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
