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

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
# import tyro

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
    step_reward_mult: int = 1
    term_reward_mult: int = 0
    consecutive_error_reward_factor: Optional[int] = None  # DEPRECATED
    user_timeout: int = 30
    vcmi_timeout: int = 30
    boot_timeout: int = 30
    random_heroes: int = 1
    random_obstacles: int = 1
    town_chance: int = 0
    warmachine_chance: int = 0
    mana_min: int = 0
    mana_max: int = 0
    swap_sides: int = 0
    reward_clip_tanh_army_frac: int = 1
    reward_army_value_ref: int = 0
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
    attention: dict = field(default_factory=dict)
    features_extractor1_stacks: list[dict] = field(default_factory=list)
    features_extractor1_hexes: list[dict] = field(default_factory=list)
    features_extractor2: list[dict] = field(default_factory=list)
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
    mapside: str = "attacker"
    mapmask: str = ""  # DEPRECATED
    randomize_maps: bool = False  # DEPRECATED
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    lr_schedule: ScheduleArgs = field(default_factory=lambda: ScheduleArgs())
    weight_decay: float = 0.0
    num_envs: int = 0  # DEPRECATED (envmaps determines number of envs now)
    envmaps: list = field(default_factory=lambda: ["gym/generated/4096/4096-mixstack-100K-01.vmap"])
    num_steps: int = 128
    gamma: float = 0.99
    stats_buffer_size: int = 100
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
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
        if not isinstance(self.network, NetworkArgs):
            self.network = NetworkArgs(**self.network)

        common.coerce_dataclass_ints(self)


class ChanFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


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
    def __init__(self, edim):
        super().__init__()
        self.edim = edim
        self.mha = nn.MultiheadAttention(embed_dim=edim, num_heads=1, batch_first=True)

    def forward(self, b_obs, b_masks=None):
        assert b_obs.shape == (b_obs.shape[0], 11, 15, self.edim), f"wrong obs shape: {b_obs.shape} != ({b_obs.shape[0]}, 11, 15, {self.edim})"
        if b_masks is None:
            b_obs = b_obs.flatten(start_dim=1, end_dim=2)
            # => (B, 165, e)
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
    def build_layer(spec):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")
        layer_cls = getattr(torch.nn, t, None) or globals()[t]
        return layer_cls(**kwargs)

    def __init__(self, network, action_space, observation_space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        # 2 nonhex actions (RETREAT, WAIT) + 165 hexes*14 actions each
        # Commented assert due to false positives on legacy models
        # assert action_space.n == 2 + (165*14)

        if network.attention:
            layer = AgentNN.build_layer(network.attention)
            self.attention = common.layer_init(layer)
        else:
            self.attention = None

        self.features_extractor1_stacks = torch.nn.Sequential()
        for spec in network.features_extractor1_stacks:
            layer = AgentNN.build_layer(spec)
            self.features_extractor1_stacks.append(common.layer_init(layer))

        self.features_extractor1_hexes = torch.nn.Sequential()
        for spec in network.features_extractor1_hexes:
            layer = AgentNN.build_layer(spec)
            self.features_extractor1_hexes.append(common.layer_init(layer))

        self.features_extractor2 = torch.nn.Sequential()
        for spec in network.features_extractor2:
            layer = AgentNN.build_layer(spec)
            self.features_extractor2.append(common.layer_init(layer))

        self.actor = common.layer_init(AgentNN.build_layer(network.actor), gain=0.01)
        self.critic = common.layer_init(AgentNN.build_layer(network.critic), gain=1.0)

    def extract_features(self, x):
        stacks, hexes = x.split([1960, 10725], dim=1)
        fstacks = self.features_extractor1_stacks(stacks)
        fhexes = self.features_extractor1_hexes(hexes)
        fcat = torch.cat((fstacks, fhexes), dim=1)
        return self.features_extractor2(fcat)

    def get_value(self, x, attn_mask=None):
        if self.attention:
            x = self.attention(x, attn_mask)
        return self.critic(self.extract_features(x))

    def get_action_and_value(self, x, mask, attn_mask=None, action=None, deterministic=False):
        if self.attention:
            x = self.attention(x, attn_mask)
        features = self.extract_features(x)
        value = self.critic(features)
        action_logits = self.actor(features)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    # Inference (deterministic)
    # XXX: attention is not handled here
    def predict(self, b_obs, b_mask):
        with torch.no_grad():
            b_obs = torch.as_tensor(b_obs, device='cpu')
            b_mask = torch.as_tensor(b_mask, device='cpu')

            # Return unbatched action if input was unbatched
            if b_obs.shape == self.observation_space.shape:
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
        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
        torch.save(clean_agent, agent_file)

    @staticmethod
    def jsave(agent, jagent_file):
        print("Saving JIT agent to %s" % jagent_file)
        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN.load_state_dict(agent.NN.state_dict(), strict=True)
        clean_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
        jagent = JitAgent()
        jagent.env_version = clean_agent.env_version

        # v1, v2
        # jagent.features_extractor = clean_agent.NN.features_extractor

        # v3
        jagent.features_extractor1_stacks = clean_agent.NN.features_extractor1_stacks
        jagent.features_extractor1_hexes = clean_agent.NN.features_extractor1_hexes
        jagent.features_extractor2 = clean_agent.NN.features_extractor2

        # common
        jagent.actor = clean_agent.NN.actor
        jagent.critic = clean_agent.NN.critic
        torch.jit.save(torch.jit.script(jagent), jagent_file)

    @staticmethod
    def load(agent_file, device="cpu"):
        print("Loading agent from %s" % agent_file)
        return torch.load(agent_file, map_location=device)

    def __init__(self, args, observation_space, action_space, state=None, device="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self.observation_space = observation_space  # needed for save/load
        self.action_space = action_space  # needed for save/load
        self.NN = AgentNN(args.network, action_space, observation_space)
        self.NN.to(device)
        self.optimizer = torch.optim.AdamW(self.NN.parameters(), eps=1e-5)
        self.predict = self.NN.predict
        self.state = state or State()


class JitAgent(nn.Module):
    """ TorchScript version of Agent (inference only) """

    def __init__(self):
        super().__init__()
        # XXX: these are overwritten after object is initialized
        self.features_extractor = nn.Identity()
        self.actor = nn.Identity()
        self.critic = nn.Identity()
        self.env_version = 0

    # Inference (deterministic)
    # XXX: attention is not handled here
    @torch.jit.export
    def predict(self, obs, mask) -> int:
        with torch.no_grad():
            b_obs = torch.as_tensor(obs).cpu().unsqueeze(dim=0)
            b_mask = torch.as_tensor(mask).cpu().unsqueeze(dim=0)

            # v1, v2
            # features = self.features_extractor(b_obs)

            # v3
            stacks, hexes = b_obs.split([1960, 10725], dim=1)
            fstacks = self.features_extractor1_stacks(stacks)
            fhexes = self.features_extractor1_hexes(hexes)
            fcat = torch.cat((fstacks, fhexes), dim=1)
            features = self.features_extractor2(fcat)

            action_logits = self.actor(features)
            dist = common.SerializableCategoricalMasked(logits=action_logits, mask=b_mask)
            action = torch.argmax(dist.probs, dim=1)
            return action.int().item()

    @torch.jit.export
    def get_value(self, obs) -> float:
        with torch.no_grad():
            b_obs = torch.as_tensor(obs).cpu().unsqueeze(dim=0)
            features = self.features_extractor(b_obs)
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

    lr_schedule_fn = common.schedule_fn(args.lr_schedule)

    num_envs = len(args.envmaps)
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

    if args.seed >= 0:
        seed = args.seed
    elif agent and agent.state.seed >= 0:
        seed = agent.state.seed
    else:
        seed = np.random.randint(2**31)

    wrappers = args.env_wrappers

    if args.env_version == 1:
        from vcmi_gym import VcmiEnv_v1 as VcmiEnv
    elif args.env_version == 2:
        from vcmi_gym import VcmiEnv_v2 as VcmiEnv
    elif args.env_version == 3:
        from vcmi_gym import VcmiEnv_v3 as VcmiEnv
    else:
        raise Exception("Unsupported env version: %d" % args.env_version)

    obs_space = VcmiEnv.OBSERVATION_SPACE
    act_space = VcmiEnv.ACTION_SPACE

    if agent is None:
        agent = Agent(args, obs_space, act_space, device=device)

    # Legacy models with offset actions
    if agent.NN.actor.out_features == 2311:
        print("Using legacy model with 2311 actions")
        wrappers.append(dict(module="vcmi_gym", cls="LegacyActionSpaceWrapper"))
        n_actions = 2311
    else:
        print("Using new model with 2312 actions")
        n_actions = 2312

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    try:
        envs = common.create_venv(VcmiEnv, args, seed=np.random.randint(2**31))
        [ENVS.append(e) for e in envs.unwrapped.envs]  # DEBUG

        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent.state.seed = seed

        # these are used by gym's RecordEpisodeStatistics wrapper
        envs.return_queue = agent.state.ep_rew_queue
        envs.length_queue = agent.state.ep_length_queue

        assert act_space.shape == ()

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
        obs = torch.zeros((args.num_steps, num_envs) + obs_space.shape).to(device)
        actions = torch.zeros((args.num_steps, num_envs) + act_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, num_envs)).to(device)
        dones = torch.zeros((args.num_steps, num_envs)).to(device)
        values = torch.zeros((args.num_steps, num_envs)).to(device)

        masks = torch.zeros((args.num_steps, num_envs, n_actions), dtype=torch.bool).to(device)
        attnmasks = torch.zeros((args.num_steps, num_envs, 165, 165)).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=agent.state.seed)  # XXX: seed has no effect here
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(num_envs).to(device)
        next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask"))).to(device)

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
                masks[step] = next_mask

                if attn:
                    attnmasks[step] = next_attnmask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.NN.get_action_and_value(
                        next_obs,
                        next_mask,
                        attn_mask=next_attnmask if attn else None
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs, device=device), torch.Tensor(next_done, device=device)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask")), device=device)

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
            b_obs = obs.reshape((-1,) + obs_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + act_space.shape)
            b_masks = masks.reshape((-1,) + (n_actions,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            if attn:
                b_attn_masks = attnmasks.reshape((-1,) + (165, 165))

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []

            # XXX: train during optimization
            agent.train()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.NN.get_action_and_value(
                        b_obs[mb_inds],
                        b_masks[mb_inds],
                        attn_mask=b_attn_masks[mb_inds] if attn else None,
                        action=b_actions.long()[mb_inds]
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

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.NN.parameters(), args.max_grad_norm)
                    agent.optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
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
            wandb_log({"losses/policy_loss": pg_loss.item()})
            wandb_log({"losses/entropy": entropy_loss.item()})
            wandb_log({"losses/old_approx_kl": old_approx_kl.item()})
            wandb_log({"losses/approx_kl": approx_kl.item()})
            wandb_log({"losses/clipfrac": np.mean(clipfracs)})
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
    return agent, common.safe_mean(list(agent.state.rollout_rew_queue_1000)[-agent.state.current_rollout:])


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
        # agent_load_file="data/mppo-test/mppo-test/agent-1718752596.pt",
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
        max_saves=1,
        out_dir_template="data/mppo-test/mppo-test",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.001),
        envmaps=["gym/A1.vmap"],
        # num_steps=64,
        num_steps=256,
        gamma=0.8,
        gae_lambda=0.8,
        num_minibatches=2,
        # num_minibatches=16,
        # update_epochs=2,
        update_epochs=10,
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
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
        ),
        env_wrappers=[],
        env_version=3,
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        network=dict(
            attention=None,
            features_extractor1_stacks=[
                # => (B, 1960)
                dict(t="Unflatten", dim=1, unflattened_size=[1, 20*98]),
                dict(t="Conv1d", in_channels=1, out_channels=8, kernel_size=98, stride=98),
                dict(t="Flatten"),
                dict(t="LeakyReLU"),
                # => (B, 160)
            ],
            features_extractor1_hexes=[
                # => (B, 10725)
                dict(t="Unflatten", dim=1, unflattened_size=[1, 165*65]),
                # => (B, 1, 10725)
                dict(t="Conv1d", in_channels=1, out_channels=4, kernel_size=65, stride=65),
                dict(t="Flatten"),
                dict(t="LeakyReLU"),
                # => (B, 660)
            ],
            features_extractor2=[
                # => (B, 820)
                dict(t="Linear", in_features=820, out_features=512),
                dict(t="LeakyReLU"),
            ],
            actor=dict(t="Linear", in_features=512, out_features=2312),
            critic=dict(t="Linear", in_features=512, out_features=1)
        )
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
