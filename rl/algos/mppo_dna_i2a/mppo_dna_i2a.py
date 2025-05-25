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
import json
import string
import argparse
import copy
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
# import tyro
import warnings
import ray

from .. import common
from ..mppo_dna_ray.sampler import Sampler
from ...world.world import I2A

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
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    user_timeout: int = 30
    vcmi_timeout: int = 30
    boot_timeout: int = 30
    random_heroes: int = 1
    random_obstacles: int = 1
    town_chance: int = 0
    warmachine_chance: int = 0
    random_stack_chance: int = 0
    random_terrain_chance: int = 0
    tight_formation_chance: int = 0
    battlefield_pattern: str = ""
    mana_min: int = 0
    mana_max: int = 0
    reward_step_fixed: int = -1
    reward_dmg_mult: int = 1
    reward_term_mult: int = 1
    swap_sides: int = 0
    deprecated_args: list[dict] = field(default_factory=lambda: [])

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class I2AKwargs:
    i2a_fc_units: int = 16
    num_trajectories: int = 5
    rollout_dim: int = 16
    rollout_policy_fc_units: int = 16
    horizon: int = 3
    obs_processor_output_size: int = 16
    transition_model_file: str = "hauzybxn-model.pt"
    action_prediction_model_file: str = "ogyesvkb-model.pt"


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
    run_name_template: Optional[str] = None
    trial_id: Optional[str] = None
    wandb_project: Optional[str] = None
    resume: bool = False
    overwrite: list = field(default_factory=list)
    notes: Optional[str] = None
    tags: Optional[list] = field(default_factory=list)
    loglevel: str = "DEBUG"

    agent_load_file: Optional[str] = None
    vsteps_total: int = 0
    seconds_total: int = 0
    rollouts_per_log: int = 1
    success_rate_target: Optional[float] = None
    ep_rew_mean_target: Optional[float] = None
    quit_on_target: bool = False
    mapside: str = "both"
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_old_saves: int = 1
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
    envmaps: list = field(default_factory=lambda: ["gym/generated/4096/4x1024.vmap"])

    num_minibatches: int = 4  # used if nn-specific num_minibatches is 0
    num_minibatches_distill: int = 0
    num_minibatches_policy: int = 0
    num_minibatches_value: int = 0
    num_steps_per_sampler: int = 128
    num_samplers: int = 1
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
    i2a_kwargs: I2AKwargs = field(default_factory=lambda: I2AKwargs())

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
        if not isinstance(self.i2a_kwargs, I2AKwargs):
            self.i2a_kwargs = I2AKwargs(**self.i2a_kwargs)

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


class AgentNN(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super().__init__()

        # I2A's p10n worls only as defender
        assert args.mapside == "defender"

        self.i2a = I2A(
            i2a_fc_units=args.i2a_kwargs.i2a_fc_units,
            num_trajectories=args.i2a_kwargs.num_trajectories,
            rollout_dim=args.i2a_kwargs.rollout_dim,
            rollout_policy_fc_units=args.i2a_kwargs.rollout_policy_fc_units,
            horizon=args.i2a_kwargs.horizon,
            obs_processor_output_size=args.i2a_kwargs.obs_processor_output_size,
            side=(args.mapside == "defender"),
            reward_step_fixed=args.env.reward_step_fixed,
            reward_dmg_mult=args.env.reward_dmg_mult,
            reward_term_mult=args.env.reward_term_mult,
            transition_model_file=args.i2a_kwargs.transition_model_file,
            action_prediction_model_file=args.i2a_kwargs.action_prediction_model_file,
            device=torch.device("cpu"),
        )

        self.device = device
        self.to(device)

    def forward(self, obs, mask):
        return self.i2a(obs, mask)

    def get_value(self, obs, mask):
        _, value = self.i2a(obs, mask)
        return value

    def get_action(self, obs, mask, action=None, deterministic=False):
        action_logits, value = self.i2a(obs, mask)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_action_and_value(self, obs, mask, action=None, deterministic=False):
        action_logits, value = self.i2a(obs, mask)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def predict(self, obs, mask):
        raise NotImplementedError()


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

        attrs = ["args", "device_name", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN_value.load_state_dict(agent.NN_value.state_dict(), strict=True)
        clean_agent.NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)
        clean_agent.optimizer_value.load_state_dict(agent.optimizer_value.state_dict())
        clean_agent.optimizer_policy.load_state_dict(agent.optimizer_policy.state_dict())
        clean_agent.optimizer_distill.load_state_dict(agent.optimizer_distill.state_dict())
        torch.save(clean_agent, agent_file)

    @staticmethod
    def load(agent_file, device_name="cpu"):
        print("Loading agent from %s (device_name: %s)" % (agent_file, device_name))
        return torch.load(agent_file, map_location=torch.device(device_name), weights_only=False)

    def __init__(self, args, state=None, device_name="cpu"):
        super().__init__()
        self.args = args
        self.env_version = args.env_version
        self._optimizer_state = None  # needed for save/load
        self.device_name = device_name
        self.NN_value = AgentNN(args, torch.device(device_name))
        self.NN_policy = AgentNN(args, torch.device(device_name))
        self.optimizer_value = torch.optim.AdamW(self.NN_value.parameters(), eps=1e-5)
        self.optimizer_policy = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.optimizer_distill = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.predict = self.NN_policy.predict
        self.state = state or State()


def compute_advantages(rewards, dones, values, next_done, next_value, gamma, gae_lambda_policy, gae_lambda_value):
    # Must be time-major
    # (num_workers, num_wsteps) => (num_wsteps, num_workers)
    rewards = rewards.swapaxes(0, 1)
    dones = dones.swapaxes(0, 1)
    values = values.swapaxes(0, 1)

    total_steps = len(rewards)

    # XXX: zeros_like keeps the memory layout of `rewards`
    #      `rewards` is currently non-contigouos due to the swapaxes opereation
    #      `advantages` has the same mem layout and is also non-contiguous now
    #      ...but the returned value (after one more swap) is contiguous :)
    advantages_policy = torch.zeros_like(rewards)
    advantages_value = torch.zeros_like(rewards)
    lastgaelam_policy = 0
    lastgaelam_value = 0
    for t in reversed(range(total_steps)):
        if t == total_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages_policy[t] = lastgaelam_policy = delta + gamma * gae_lambda_policy * nextnonterminal * lastgaelam_policy
        advantages_value[t] = lastgaelam_value = delta + gamma * gae_lambda_value * nextnonterminal * lastgaelam_value
    returns_value = advantages_value + values
    return advantages_policy.swapaxes(0, 1), returns_value.swapaxes(0, 1)


def main(args):
    LOG = logging.getLogger("mppo_dna")
    LOG.setLevel(getattr(logging, args.loglevel))

    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    assert isinstance(args, Args)

    agent, args = common.maybe_resume(Agent, args, device_name=device_name)
    num_steps = args.num_steps_per_sampler * args.num_samplers

    if args.seconds_total:
        assert not args.vsteps_total, "cannot have both vsteps_total and seconds_total"
        rollouts_total = 0
    else:
        rollouts_total = args.vsteps_total // num_steps

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
    LOG.info("Out dir: %s" % args.out_dir)

    lr_schedule_fn_value = common.schedule_fn(args.lr_schedule_value)
    lr_schedule_fn_policy = common.schedule_fn(args.lr_schedule_policy)
    lr_schedule_fn_distill = common.schedule_fn(args.lr_schedule_distill)

    batch_size_policy = int(num_steps)
    batch_size_value = int(num_steps)
    batch_size_distill = int(num_steps)

    assert batch_size_policy % args.num_minibatches_policy == 0, f"{batch_size_policy} % {args.num_minibatches_policy} == 0"
    assert batch_size_value % args.num_minibatches_value == 0, f"{batch_size_value} % {args.num_minibatches_value} == 0"
    assert batch_size_distill % args.num_minibatches_distill == 0, f"{batch_size_distill} % {args.num_minibatches_distill} == 0"

    minibatch_size_policy = int(batch_size_policy // args.num_minibatches_policy)
    minibatch_size_value = int(batch_size_value // args.num_minibatches_value)
    minibatch_size_distill = int(batch_size_distill // args.num_minibatches_distill)

    save_ts = None
    permasave_ts = None

    if args.agent_load_file and not agent:
        f = args.agent_load_file
        agent = Agent.load(f, device_name=device_name)
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

    if args.env_version == 12:
        from vcmi_gym import VcmiEnv_v12 as VcmiEnv
    else:
        raise Exception("Unsupported env version: %d" % args.env_version)

    if agent is None:
        agent = Agent(args, device_name=device_name)

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
                logfn = getattr(LOG, "info" if commit else "debug")
                logfn(data)
                wandb.log(data, commit=commit)
        else:
            def wandb_log(data, commit=False):
                logfn = getattr(LOG, "info" if commit else "debug")
                logfn(data)

        if not args.logparams:
            args.logparams = {
                "params/gamma": "gamma",
                "params/gae_lambda": "gae_lambda",
                "params/ent_coef": "ent_coef",
                "params/clip_coef": "clip_coef",
                # "params/lr_schedule": "lr_schedule",
                "params/norm_adv": "norm_adv",
                "params/clip_vloss": "clip_vloss",
                "params/max_grad_norm": "max_grad_norm",
                "params/weight_decay": "weight_decay",
                "params/distill_beta": "distill_beta",
            }

        common.log_params(args, wandb_log)

        if args.resume:
            agent.state.resumes += 1
            wandb_log({"global/resumes": agent.state.resumes})

        # print("Agent state: %s" % asdict(agent.state))

        obs_space = VcmiEnv.OBSERVATION_SPACE
        act_space = VcmiEnv.ACTION_SPACE

        # ALGO Logic: Storage setup
        device = torch.device(device_name)
        obs = torch.zeros((args.num_samplers, args.num_steps_per_sampler) + obs_space["observation"].shape, device=device)
        logprobs = torch.zeros(args.num_samplers, args.num_steps_per_sampler, device=device)
        actions = torch.zeros((args.num_samplers, args.num_steps_per_sampler) + act_space.shape, device=device)
        masks = torch.zeros((args.num_samplers, args.num_steps_per_sampler, act_space.n), dtype=torch.bool, device=device)  # XXX: must use torch.bool (for CategoricalMasked)
        advantages = torch.zeros(args.num_samplers, args.num_steps_per_sampler, device=device)
        returns = torch.zeros(args.num_samplers, args.num_steps_per_sampler, device=device)

        rewards = torch.zeros((args.num_samplers, args.num_steps_per_sampler), device=device)
        dones = torch.zeros((args.num_samplers, args.num_steps_per_sampler), device=device)
        next_obs = torch.zeros((args.num_samplers,) + obs_space["observation"].shape, device=device)
        next_mask = torch.zeros((args.num_samplers, act_space.n), device=device)
        next_done = torch.zeros(args.num_samplers, device=device)

        progress = 0
        map_rollouts = 0
        start_time = time.time()
        global_start_second = agent.state.global_second

        sampler_device_name = device_name
        sampler_device = torch.device(sampler_device_name)

        ray.init(num_cpus=1)

        if sampler_device_name == "cuda":
            RemoteSampler = ray.remote(num_cpus=0.01, num_gpus=0.01)(Sampler)
        else:
            RemoteSampler = ray.remote(num_cpus=0.01)(Sampler)

        def NN_creator(device):
            return AgentNN(args, device=device)

        def venv_creator():
            return common.create_venv(VcmiEnv, args)

        def remote_sampler_creator(sampler_id):
            return RemoteSampler.remote(
                sampler_id,
                NN_creator,
                venv_creator,
                args.num_steps_per_sampler,
                sampler_device_name
            )

        # s = Sampler(
        #     0,
        #     NN_creator,
        #     venv_creator,
        #     args.num_steps_per_sampler,
        #     args.gamma,
        #     args.gae_lambda_policy,
        #     args.gae_lambda_value,
        #     sampler_device_name
        # )

        # s.sample()
        # import ipdb; ipdb.set_trace()  # noqa

        LOG.info("[main] init %d samplers" % args.num_samplers)
        samplers = [remote_sampler_creator(i) for i in range(args.num_samplers)]

        timers = {
            "all": common.Timer(),
            "sample": common.Timer(),
            "set_weights": common.Timer(),
            "train": common.Timer(),
            "value_optimization": common.Timer(),
            "policy_distillation": common.Timer(),
            "save": common.Timer(),

        }

        while progress < 1:
            [t.reset() for t in timers.values()]
            timers["all"].start()

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
            with timers["set_weights"]:
                ref = ray.put({k: v.to(sampler_device) for k, v in agent.NN_policy.state_dict().items()})
                ray.get([s.set_weights.remote(ref) for s in samplers])

            with timers["sample"]:
                # LOG.debug("Call samplers...")
                futures = [s.sample.remote() for s in samplers]

                # LOG.debug("Gather results...")
                for i in range(len(futures)):
                    done, futures = ray.wait(futures, num_returns=1)
                    res, stats = ray.get(done[0])

                    # tw = timesteps per worker
                    (
                        obs[i],          # => (tw, STATE_SIZE)
                        logprobs[i],     # => (tw)
                        actions[i],      # => (tw)
                        masks[i],        # => (tw, N_ACTIONS)
                        rewards[i],      # => (tw)
                        dones[i],        # => (tw)
                        next_obs[i],     # => (STATE_SIZE)
                        next_mask[i],    # => (N_ACTIONS)
                        next_done[i],    # => (1)
                    ) = res

                    (
                        s_seconds,
                        s_episodes,
                        s_ep_net_value,
                        s_ep_is_success,
                        s_ep_return,
                        s_ep_length
                    ) = stats

                    agent.state.ep_net_value_queue.append(s_ep_net_value)
                    agent.state.ep_is_success_queue.append(s_ep_is_success)
                    agent.state.ep_rew_queue.append(s_ep_return)
                    agent.state.ep_length_queue.append(s_ep_length)
                    agent.state.current_episode += s_episodes
                    agent.state.global_episode += s_episodes
                    ep_count += s_episodes

                    agent.state.current_vstep += args.num_steps_per_sampler
                    agent.state.current_timestep += args.num_steps_per_sampler
                    agent.state.global_timestep += args.num_steps_per_sampler
                    agent.state.current_second = int(time.time() - start_time)
                    agent.state.global_second = global_start_second + agent.state.current_second

            with torch.no_grad():
                agent.eval()
                values = agent.NN_value.get_value(obs.flatten(end_dim=1), masks.flatten(end_dim=1)).reshape(args.num_samplers, -1)
                # bootstrap value if not done
                next_value = agent.NN_value.get_value(next_obs, next_mask).flatten()

                advantages, returns = compute_advantages(
                    rewards,
                    dones,
                    values,
                    next_done,
                    next_value,
                    args.gamma,
                    args.gae_lambda_policy,
                    args.gae_lambda_value,
                )

            # flatten the batch (workers, worker_samples, *) => (num_steps, *)
            b_obs = obs.flatten(end_dim=1)
            b_logprobs = logprobs.flatten(end_dim=1)
            b_actions = actions.flatten(end_dim=1)
            b_masks = masks.flatten(end_dim=1)
            b_advantages = advantages.flatten(end_dim=1)
            b_returns = returns.flatten(end_dim=1)
            b_values = values.flatten(end_dim=1)

            b_inds = np.arange(batch_size_policy)
            clipfracs = []

            # Policy network optimization
            with timers["train"]:
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
            with timers["value_optimization"]:
                for epoch in range(args.update_epochs_value):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size_value, minibatch_size_value):
                        end = start + minibatch_size_value
                        mb_inds = b_inds[start:end]

                        newvalue = agent.NN_value.get_value(b_obs[mb_inds], b_masks[mb_inds])
                        newvalue = newvalue.view(-1)

                        # Value loss
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        agent.optimizer_value.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.NN_value.parameters(), args.max_grad_norm)
                        agent.optimizer_value.step()

            # Value network to policy network distillation
            with timers["policy_distillation"]:
                agent.NN_policy.zero_grad(True)  # don't clone gradients

                old_NN_policy = copy.deepcopy(agent.NN_policy).to(device)
                # old_NN_policy = AgentNN(args.network, agent.dim_other, agent.dim_hexes, agent.n_actions, agent.NN_policy.device)
                # old_NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)

                old_NN_policy.eval()
                for epoch in range(args.update_epochs_distill):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size_distill, minibatch_size_distill):
                        end = start + minibatch_size_distill
                        mb_inds = b_inds[start:end]
                        # Compute policy and value targets
                        with torch.no_grad():
                            _, _, _, old_action_dist = old_NN_policy.get_action(b_obs[mb_inds], b_masks[mb_inds])
                            value_target = agent.NN_value.get_value(b_obs[mb_inds], b_masks[mb_inds])

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

            wlog = {}

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
                wlog["rollout/ep_rew_mean"] = ep_rew_mean
                wlog["rollout/ep_value_mean"] = ep_value_mean
                wlog["rollout/ep_success_rate"] = ep_is_success_mean

            wlog = dict(wlog, **{
                "params/policy_learning_rate": agent.optimizer_policy.param_groups[0]["lr"],
                "params/value_learning_rate": agent.optimizer_value.param_groups[0]["lr"],
                "params/distill_learning_rate": agent.optimizer_distill.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/distill_loss": distill_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": float(np.mean(clipfracs)),
                "losses/explained_variance": float(explained_var),
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
            })

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

            wandb_commit = False
            if agent.state.current_rollout > 0 and agent.state.current_rollout % args.rollouts_per_log == 0:
                wandb_commit = True
                wlog["global/global_num_timesteps"] = agent.state.global_timestep
                wlog["global/global_num_seconds"] = agent.state.global_second

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

            with timers["save"]:
                save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent)

            t_all = timers["all"].peek()
            for k, v in timers.items():
                wlog[f"timer/{k}"] = v.peek()
                if k != "all":
                    wlog[f"timer_rel/{k}"] = v.peek() / t_all

            wlog["timer/other"] = t_all - sum(v.peek() for k, v in timers.items() if k != "all")
            wlog["timer_rel/other"] = wlog["timer/other"] / t_all

            wandb_log(wlog, commit=wandb_commit)
            # print("TRAIN TIME: %.2f" % (time.time() - tstart))

    finally:
        common.maybe_save(0, 10e9, args, agent)
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


def debug_config():
    return dict(
        run_id="mppo-dna-i2a-debug",
        group_id="mppo-dna-i2a-debug",
        loglevel="DEBUG",
        run_name_template="{datetime}-{id}",
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
        mapside="defender",
        save_every=2000000000,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_old_saves=0,
        out_dir_template="data/{group_id}",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.0001),
        lr_schedule_value=ScheduleArgs(mode="const", start=0.0001),
        lr_schedule_policy=ScheduleArgs(mode="const", start=0.0001),
        lr_schedule_distill=ScheduleArgs(mode="const", start=0.0001),
        num_steps_per_sampler=6,
        num_samplers=1,
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
        envmaps=["gym/A1.vmap"],
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
            user_timeout=600,
            vcmi_timeout=600,
            boot_timeout=300,
        ),
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        env_wrappers=[dict(module="vcmi_gym.envs.util.wrappers", cls="LegacyObservationSpaceWrapper")],
        env_version=12,
        i2a_kwargs=dict(
            i2a_fc_units=16,
            num_trajectories=5,
            rollout_dim=16,
            rollout_policy_fc_units=16,
            horizon=3,
            obs_processor_output_size=16,
            transition_model_file="hauzybxn-model.pt",
            action_prediction_model_file="ogyesvkb-model.pt",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", metavar="FILE", help="json file to resume")
    parser.add_argument("--debug", action="store_true", help="use hardcoded debug args")
    args = parser.parse_args()

    if args.resume:
        assert not args.debug, "mutually exclusive: --debug and --resume"
        with open(args.resume, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)
            config["resume"] = True
    else:
        if args.debug:
            config = debug_config()
            run_id = config["run_id"]
        else:
            from .config import config
            run_id = ''.join(random.choices(string.ascii_lowercase, k=8))

        config["run_id"] = run_id
        config["run_name"] = config["run_name_template"].format(id=run_id, datetime=datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        config["out_dir"] = config["out_dir_template"].format(seed=config["seed"], group_id=config["group_id"], run_id=config["run_id"])

    config["out_dir_abs"] = os.path.abspath(config["out_dir"])

    main(Args(**config))
