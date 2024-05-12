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
# This file contains a modified version of CleanRL's DQN implementation:
# https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/dqn.py
# with a quantile-regression enhancement inspired by SB3's QR-DQN implementation:
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/8dca5d5e22/sb3_contrib/qrdqn/qrdqn.py

import sys
import os
import random
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import psutil
import warnings
import logging
from typing import NamedTuple

from vcmi_gym import VcmiEnv

from .. import common

ENVS = []  # debug
LOG = None


def render():
    print(ENVS[0].render())


class ReplayBufferSample(NamedTuple):
    b_observation: torch.Tensor
    b_action: torch.Tensor
    b_action_mask: torch.Tensor
    b_done: torch.Tensor
    b_reward: torch.Tensor
    b_next_observation: torch.Tensor
    b_next_action_mask: torch.Tensor


class ReplayBuffer():
    def __init__(self, n_envs, size_vsteps, observation_space, action_space):
        self.size_vsteps = size_vsteps
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape
        self.pos = 0
        self.full = False
        self.n_envs = n_envs
        self.size_vsteps = max(size_vsteps // n_envs, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observations = np.zeros((self.size_vsteps, self.n_envs, *observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((self.size_vsteps, self.n_envs), dtype=int)
        self.action_masks = np.zeros((self.size_vsteps, self.n_envs, action_space.n), dtype=bool)
        self.rewards = np.zeros((self.size_vsteps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.size_vsteps, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.size_vsteps, self.n_envs), dtype=np.float32)

        # Check that the replay buffer can fit into the memory
        mem_available = psutil.virtual_memory().available
        total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        if total_memory_usage > mem_available:
            total_memory_usage /= 1e9
            mem_available /= 1e9
            warnings.warn(
                "This system does not have apparently enough memory to store the complete "
                f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    # Inputs are numpy arrays
    def add(self, b_obs, b_action, b_action_mask, b_reward, b_done, b_next_obs, b_next_action_mask) -> None:
        self.observations[self.pos] = b_obs  # np.array(obs).copy()
        self.actions[self.pos] = b_action  # np.array(action).copy()
        self.action_masks[self.pos] = b_action_mask  # np.array(action).copy()
        self.rewards[self.pos] = b_reward  # np.array(reward).copy()
        self.dones[self.pos] = b_done  # np.array(done).copy()

        special_pos = (self.pos + 1) % self.size_vsteps
        self.observations[special_pos] = b_next_obs  # np.array(next_obs).copy()
        self.action_masks[special_pos] = b_next_action_mask  # np.array(b_next_action_mask).copy()

        self.pos += 1
        if self.pos == self.size_vsteps:
            self.full = True
            self.pos = 0

    def to_tensor(self, ary, dtype=torch.float32):
        return torch.as_tensor(ary, dtype=dtype, device=self.device)

    def sample(self, batch_size):
        if self.full:
            batch_inds = (np.random.randint(1, self.size_vsteps, size=batch_size) + self.pos) % self.size_vsteps
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        special_inds = (batch_inds + 1) % self.size_vsteps

        return ReplayBufferSample(
            b_observation=self.to_tensor(self.observations[batch_inds, env_inds]),
            b_action=self.to_tensor(self.actions[batch_inds, env_inds], int),
            b_action_mask=self.to_tensor(self.action_masks[batch_inds, env_inds], bool),
            b_reward=self.to_tensor(self.rewards[batch_inds, env_inds]),
            b_done=self.to_tensor(self.dones[batch_inds, env_inds]),
            b_next_observation=self.to_tensor(self.observations[special_inds, env_inds]),
            b_next_action_mask=self.to_tensor(self.action_masks[special_inds, env_inds], bool),
        )


@dataclass
class ScheduleArgs:
    # const / lin_decay / exp_decay
    mode: str = "const"
    start: float = 2.5e-4
    end: float = 0
    # for lin_decay, rate=1 means reach end value at 100%
    #                rate=10 means end value = 10% progress (until the end)
    # for exp_decay, rate=1 never goes close to end (v=37% @ 100% progress)
    #                rate=10 is good (50%@7%, 25%@14%, 10%@23%, 1%@46% progress)
    rate: float = 10


@dataclass
class EnvArgs:
    encoding_type: str = "default"
    max_steps: int = 500
    reward_dmg_factor: int = 5
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    sparse_info: bool = True
    step_reward_fixed: int = 0
    step_reward_mult: int = 1
    term_reward_mult: int = 0
    consecutive_error_reward_factor: Optional[int] = None
    vcmi_timeout: int = 5
    random_heroes: int = 1
    random_obstacles: int = 1
    reward_clip_tanh_army_frac: int = 1
    reward_army_value_ref: int = 0

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class State:
    resumes: int = 0
    map_swaps: int = 0
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0

    def __post_init__(self):
        common.coerce_dataclass_ints(self)


@dataclass
class Args:
    run_id: str
    group_id: str
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    resume: bool = False
    overwrite: list = field(default_factory=list)
    notes: Optional[str] = None
    tags: Optional[list] = field(default_factory=list)

    agent_load_file: Optional[str] = None
    vsteps_total: int = 0
    # vsteps_per_mapchange: int = 0
    trains_per_log: int = 100
    # vsteps_per_table_log: int = 40_000
    success_rate_target: Optional[float] = None
    ep_rew_mean_target: Optional[float] = None
    quit_on_target: bool = False
    mapside: str = "both"
    mapmask: str = "ai/generated/B*.vmap"
    randomize_maps: bool = False
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    lr_schedule: ScheduleArgs = ScheduleArgs()
    weight_decay: float = 0.0
    num_envs: int = 4
    stats_buffer_size: int = 100

    buffer_size_vsteps: int = 100_000
    eps_schedule: ScheduleArgs = ScheduleArgs()
    vsteps_for_warmup: int = 100_000
    random_warmup: bool = True
    batch_size: int = 32
    tau: int = 1.0
    train_iterations: int = 1
    vsteps_per_train: int = 4
    trains_per_target_update: int = 1000
    gamma: float = 0.99
    max_grad_norm: float = 0.5

    logparams: dict = field(default_factory=dict)
    cfg_file: Optional[str] = None
    seed: int = 42
    skip_wandb_init: bool = False

    env: EnvArgs = EnvArgs()
    env_wrappers: list = field(default_factory=list)
    network: list[dict] = field(default_factory=list)
    n_quantiles: int = 5

    def __post_init__(self):
        if not isinstance(self.env, EnvArgs):
            self.env = EnvArgs(**self.env)
        if not isinstance(self.lr_schedule, ScheduleArgs):
            self.lr_schedule = ScheduleArgs(**self.lr_schedule)
        if not isinstance(self.eps_schedule, ScheduleArgs):
            self.eps_schedule = ScheduleArgs(**self.eps_schedule)

        common.coerce_dataclass_ints(self)


class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        # TODO: attn_mask
        res, _weights = super().forward(x, x, x, need_weights=False, attn_mask=None)
        return res


class AgentNN(nn.Module):
    @staticmethod
    def build_layer(spec):
        kwargs = dict(spec)  # copy
        t = kwargs.pop("t")
        layer_cls = getattr(torch.nn, t, None) or globals()[t]
        return layer_cls(**kwargs)

    def __init__(self, network_spec, n_quantiles, action_space, observation_space):
        super().__init__()

        self.n_quantiles = n_quantiles
        self.observation_space = observation_space
        self.action_space = action_space

        # 1 nonhex action (RETREAT) + 165 hexes*14 actions each
        assert action_space.n == 1 + (165*14)

        self.network = torch.nn.Sequential()
        self.target_network = torch.nn.Sequential()

        for spec in network_spec:
            layer = AgentNN.build_layer(spec)
            self.network.append(common.layer_init(layer))
            self.target_network.append(common.layer_init(layer))

        # add action head
        with torch.no_grad():
            obs = torch.as_tensor(observation_space.sample(), dtype=torch.float32)
            out = self.network(obs.unsqueeze(0)).squeeze(0)

        assert len(out.shape) == 1

        self.network.append(common.layer_init(nn.Linear(out.shape[0], n_quantiles * action_space.n), gain=0.01))
        self.target_network.append(common.layer_init(nn.Linear(out.shape[0], n_quantiles * action_space.n), gain=0.01))

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.train(False)

        self.mask_value = torch.tensor(torch.finfo(torch.float32).min).float()

    def b_q_logits(self, b_obs: torch.Tensor, use_target=False):
        net = self.target_network if use_target else self.network

        b_logits = net(b_obs)
        # => (B, n_quantiles*n_actions)
        b_q_logits = b_logits.view(-1, self.n_quantiles, self.action_space.n)
        # => (B, n_quantiles, n_actions)
        return b_q_logits

    def b_action(self, b_q_logits: torch.Tensor, b_mask: torch.Tensor):
        # => (B, n_quantiles, n_actions)
        b_value = b_q_logits.mean(dim=1)
        # => (B, n_actions)
        b_action = b_value.where(b_mask, self.mask_value).argmax(dim=1)
        # => (B)
        return b_action

    def predict(self, b_obs: torch.Tensor, b_mask: torch.Tensor):
        b_q_logits = self.b_q_logits(b_obs, use_target=False)
        return self.b_action(b_q_logits, b_mask)


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space, optimizer=None, state=None):
        super().__init__()

        self.args = args
        self.observation_space = observation_space  # needed for save/load
        self.action_space = action_space  # needed for save/load

        self.NN = AgentNN(args.network, args.n_quantiles, action_space, observation_space)
        self.predict = self.NN.predict
        self.optimizer = optimizer or torch.optim.AdamW(self.NN.network.parameters(), eps=1e-5)
        self.state = state or State()

    # XXX: This is a method => it will work after pytorch.load if the saved model did not have it
    # XXX: NN must not be included here
    def save_attrs(self):
        return ["args", "observation_space", "action_space", "optimizer", "state"]


def quantile_huber_loss(value, target, batch_size, n_quantiles):
    assert value.shape == (batch_size, n_quantiles), value.shape
    assert target.shape == (batch_size, n_quantiles), target.shape

    cum_prob = (torch.arange(n_quantiles, device=value.device, dtype=torch.float32) + 0.5) / n_quantiles
    cum_prob = cum_prob.unsqueeze(-1)
    # => (B, n_quantiles, 1)

    # target: (B, n_quantiles) -> (B, 1, n_quantiles)
    # value: (B, n_quantiles) -> (B, n_quantiles, 1)
    # pairwise_delta: (B, n_quantiles, n_quantiles)
    pairwise_delta = target.unsqueeze(-2) - value.unsqueeze(-1)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    res = loss.sum(dim=-2).mean()
    return res


def main(args):
    LOG = logging.getLogger("mqrdqn")
    LOG.setLevel(logging.INFO)

    assert isinstance(args, Args)

    agent, args = common.maybe_resume(args)

    # Re-initialize to prevent errors from newly introduced args when loading/resuming
    # TODO: handle removed args
    args = Args(**vars(args))
    printargs = asdict(args).copy()

    # Logger
    formatter = logging.Formatter(f"-- %(asctime)s %(levelname)s [{args.run_id}] %(message)s")
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = None
    loghandler = logging.StreamHandler()
    loghandler.setFormatter(formatter)
    LOG.addHandler(loghandler)

    LOG.info("Args: %s" % printargs)

    out_dir = args.out_dir_template.format(seed=args.seed, group_id=args.group_id, run_id=args.run_id)
    LOG.info("Out dir: %s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    lr_schedule_fn = common.schedule_fn(args.lr_schedule)
    eps_schedule_fn = common.schedule_fn(args.eps_schedule)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_ts = None
    permasave_ts = None
    agent = None
    start_map_swaps = 0
    trains = 0

    if args.agent_load_file:
        f = args.agent_load_file
        LOG.info("Loading agent from %s" % f)
        agent = torch.load(f)
        agent.args = args
        start_map_swaps = agent.state.map_swaps
        agent.state.current_timestep = 0
        agent.state.current_vstep = 0

        backup = "%s/loaded-%s" % (os.path.dirname(f), os.path.basename(f))
        with open(f, 'rb') as fsrc:
            with open(backup, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
                LOG.info("Wrote backup %s" % backup)

    common.validate_tags(args.tags)

    try:
        envs, _ = common.create_venv(VcmiEnv, args, start_map_swaps)
        [ENVS.append(e) for e in envs.unwrapped.envs]  # DEBUG

        obs_space = envs.unwrapped.single_observation_space
        act_space = envs.unwrapped.single_action_space

        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

        replay_buffer = ReplayBuffer(
            n_envs=args.num_envs,
            size_vsteps=args.buffer_size_vsteps,
            observation_space=obs_space,
            action_space=act_space
        )

        if agent is None:
            agent = Agent(args, obs_space, act_space)

        agent = agent.to(device)

        # XXX: the start=0 requirement is needed for SB3 compat
        assert act_space.start == 0
        assert act_space.shape == ()

        if args.vsteps_total:
            assert agent.state.current_vstep < args.vsteps_total
            assert args.vsteps_for_warmup < args.vsteps_total

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

        ep_net_value_queue = deque(maxlen=envs.return_queue.maxlen)
        ep_is_success_queue = deque(maxlen=envs.return_queue.maxlen)

        rollout_rew_queue_100 = deque(maxlen=100)
        rollout_rew_queue_1000 = deque(maxlen=1000)
        rollout_net_value_queue_100 = deque(maxlen=100)
        rollout_net_value_queue_1000 = deque(maxlen=1000)
        rollout_is_success_queue_100 = deque(maxlen=100)
        rollout_is_success_queue_1000 = deque(maxlen=1000)

        warmup_progress_next_log = 0
        warming_up = False
        if agent.state.current_vstep < args.vsteps_for_warmup:
            LOG.info("Warming up... (mode: %s)" % ("random" if args.random_warmup else "non-random"))
            warming_up = True

        observations, _ = envs.reset(seed=args.seed)
        observations = observations.astype(np.float32)

        end_vstep = args.vsteps_total or 10e9

        while agent.state.current_vstep < end_vstep:
            if args.vsteps_total and not warming_up:
                progress = (agent.state.current_vstep - args.vsteps_for_warmup) / (args.vsteps_total - args.vsteps_for_warmup)
            else:
                progress = 0

            agent.optimizer.param_groups[0]["lr"] = lr_schedule_fn(progress)
            eps = eps_schedule_fn(progress)

            masks = np.array(envs.unwrapped.call("action_mask"), dtype=bool)

            # XXX: eval during experience collection
            agent.eval()

            if random.random() < eps or (warming_up and args.random_warmup):
                actions = np.array([random.choice(np.where(m)[0]) for m in masks])
            else:
                with torch.no_grad():
                    # need original non-tensor observations for insertion in replay buffer
                    t_observations = torch.as_tensor(observations, device=device)
                    t_masks = torch.as_tensor(masks, device=device)
                    actions = agent.predict(t_observations, t_masks).cpu().numpy()

            next_observations, rewards, terminations, truncations, infos = envs.step(actions)

            # preemptively cast to float32 (as used in both NN and replay buffer)
            next_observations = next_observations.astype(np.float32)
            next_masks = np.array(envs.unwrapped.call("action_mask"), dtype=bool)

            # See notes/gym_vector.txt
            for final_info, has_final_info in zip(infos.get("final_info", []), infos.get("_final_info", [])):
                # "final_info" must be None if "has_final_info" is False
                if has_final_info:
                    assert final_info is not None, "has_final_info=True, but final_info=None"
                    ep_net_value_queue.append(final_info["net_value"])
                    ep_is_success_queue.append(final_info["is_success"])

            # XXX SIMO: SB3 does bootstrapping for truncated episodes here
            # https://github.com/DLR-RM/stable-baselines3/pull/658

            replay_buffer.add(
                b_obs=observations,
                b_action=actions,
                b_action_mask=masks,
                b_reward=rewards,
                b_done=terminations,
                b_next_obs=next_observations,
                b_next_action_mask=next_masks
            )

            observations = next_observations
            masks = next_masks
            agent.state.current_vstep += 1
            agent.state.current_timestep += args.num_envs
            agent.state.global_timestep += args.num_envs

            if warming_up:
                warmup_progress = agent.state.current_vstep / args.vsteps_for_warmup
                if warmup_progress >= warmup_progress_next_log:
                    warmup_progress_next_log += 0.1
                    LOG.info("Warmup: %d/%d (%d%%)" % (agent.state.current_vstep, args.vsteps_for_warmup, 100*warmup_progress))

                warming_up = warmup_progress < 1
                continue

            if agent.state.current_vstep % args.vsteps_per_train != 0:
                continue

            # XXX: train during optimization
            agent.train()
            trains += 1

            for _ in range(0, args.train_iterations):
                sample = replay_buffer.sample(args.batch_size)

                with torch.no_grad():
                    b_q_target_logits = agent.NN.b_q_logits(sample.b_next_observation, use_target=True)
                    # => (B, n_quantiles, n_actions)
                    b_target_action = agent.NN.b_action(b_q_target_logits, sample.b_next_action_mask)
                    # => (B)
                    # Broadcast action to required shape:
                    b_q_target_action = b_target_action.unsqueeze(-1).unsqueeze(-1).expand(args.batch_size, agent.NN.n_quantiles, 1)
                    # => (B, n_quantiles, 1)
                    # Take the corresponding logit value in each quantile and remove last dim:
                    b_q_target_value = b_q_target_logits.gather(dim=2, index=b_q_target_action).squeeze(dim=2)
                    # => (B, n_quantiles)
                    # 1-step TD target:
                    b_q_target = sample.b_reward.unsqueeze(-1) + (1 - sample.b_done.unsqueeze(-1)) * args.gamma * b_q_target_value
                    # => (B, n_quantiles)

                # Same as target, except:
                # 1. We extract logits from the current instead of the target network
                # 2. We extract actions from the replay buffer instead of the logits
                b_q_logits = agent.NN.b_q_logits(sample.b_observation, use_target=False)
                b_action = sample.b_action
                b_q_action = b_action.unsqueeze(-1).unsqueeze(-1).expand(args.batch_size, agent.NN.n_quantiles, 1)
                b_q_value = torch.gather(b_q_logits, dim=2, index=b_q_action).squeeze(dim=2)

                loss = quantile_huber_loss(b_q_value, b_q_target, args.batch_size, agent.NN.n_quantiles)

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.NN.network.parameters(), args.max_grad_norm)
                agent.optimizer.step()

            # update target network (polyak)
            if trains % args.trains_per_target_update == 0:
                with torch.no_grad():
                    LOG.info("Updating target network (tau=%.2f)" % args.tau)
                    for param, target_param in zip(agent.NN.network.parameters(), agent.NN.target_network.parameters()):
                        target_param.data.mul_(1 - args.tau)
                        torch.add(target_param.data, param.data, alpha=args.tau, out=target_param.data)

            gs = agent.state.global_timestep

            ep_rew_mean = common.safe_mean(envs.return_queue)
            rollout_rew_queue_100.append(ep_rew_mean)
            rollout_rew_queue_1000.append(ep_rew_mean)

            ep_value_mean = common.safe_mean(ep_net_value_queue)
            rollout_net_value_queue_100.append(ep_value_mean)
            rollout_net_value_queue_1000.append(ep_value_mean)

            ep_is_success_mean = common.safe_mean(ep_is_success_queue)
            rollout_is_success_queue_100.append(ep_is_success_mean)
            rollout_is_success_queue_1000.append(ep_is_success_mean)

            wandb_log({"params/learning_rate": agent.optimizer.param_groups[0]["lr"]})
            wandb_log({"params/exploration_rate": eps})
            wandb_log({"losses/loss": loss.item()})
            wandb_log({"rollout/ep_rew_mean": ep_rew_mean})
            wandb_log({"rollout100/ep_rew_mean": common.safe_mean(rollout_rew_queue_100)})
            wandb_log({"rollout1000/ep_rew_mean": common.safe_mean(rollout_rew_queue_1000)})
            wandb_log({"rollout/ep_value_mean": ep_value_mean})
            wandb_log({"rollout100/ep_value_mean": common.safe_mean(rollout_net_value_queue_100)})
            wandb_log({"rollout1000/ep_value_mean": common.safe_mean(rollout_net_value_queue_1000)})
            wandb_log({"rollout/ep_success_rate": ep_is_success_mean})
            wandb_log({"rollout100/ep_success_rate": common.safe_mean(rollout_is_success_queue_100)})
            wandb_log({"rollout1000/ep_success_rate": common.safe_mean(rollout_is_success_queue_1000)})
            wandb_log({"rollout/ep_len_mean": common.safe_mean(envs.length_queue)})
            wandb_log({"rollout/ep_count": envs.episode_count})
            wandb_log({"global/current_vstep": agent.state.current_vstep})
            wandb_log({"global/current_timestep": agent.state.current_timestep})

            envs.episode_count = 0

            if args.vsteps_total:
                wandb_log({"global/progress": progress})

            LOG.debug("vstep=%d, rew=%.2f, net_value=%.2f, is_success=%.2f loss=%.2f" % (
                agent.state.current_vstep,
                ep_rew_mean,
                ep_value_mean,
                ep_is_success_mean,
                loss.item()
            ))

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

            # XXX: disabled because this would introduce inconsistent s,a,s' in replay buffer
            #      Need to somehow implement a logic that changes env only after all envs finish an episode
            # if args.vsteps_per_mapchange and agent.state.current_vstep % args.vsteps_per_mapchange == 0:
            #     agent.state.map_swaps += 1
            #     wandb_log({"global/map_swaps": agent.state.map_swaps})
            #     envs.close()
            #     envs, _ = common.create_venv(VcmiEnv, args, agent.state.map_swaps)
            #     next_obs, _ = envs.reset(seed=args.seed)
            #     next_obs = torch.Tensor(next_obs).to(device)
            #     next_done = torch.zeros(args.num_envs).to(device)
            #     next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_masks"))).to(device)

            if trains % args.trains_per_log == 0:
                wandb_log({"global/num_timesteps": gs}, commit=True)  # commit on final log line
                LOG.info("train=%d vstep=%d rew=%.2f net_value=%.2f is_success=%.2f loss=%.2f" % (
                    trains,
                    agent.state.current_vstep,
                    ep_rew_mean,
                    ep_value_mean,
                    ep_is_success_mean,
                    loss.item()
                ))

            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent, out_dir)

    finally:
        common.maybe_save(0, 10e9, args, agent, out_dir)
        envs.close()


# if __name__ == "__main__":
#     args = tyro.cli(Args)
#     main(args)


def debug_args():
    return Args(
        "debug-crl",
        "debug-crl",
        run_name=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="data/heads/heads-simple-A1/agent-1710806916.zip",
        agent_load_file=None,
        vsteps_total=0,
        # vsteps_per_mapchange=0,
        trains_per_log=100000,
        # vsteps_per_table_log=100000,
        success_rate_target=None,
        ep_rew_mean_target=None,
        quit_on_target=False,
        mapside="both",
        mapmask="gym/A1.vmap",
        randomize_maps=False,
        save_every=2000000000,  # greater than time.time()
        permasave_every=2000000000,  # greater than time.time()
        max_saves=0,
        out_dir_template="data/debug-crl/debug-crl",
        opponent_load_file=None,
        opponent_sbm_probs=[1, 0, 0],
        weight_decay=0.05,
        lr_schedule=ScheduleArgs(mode="const", start=0.001),
        num_envs=1,

        buffer_size_vsteps=10,
        eps_schedule=ScheduleArgs(mode="const", start=0.1),
        vsteps_for_warmup=0,
        random_warmup=True,
        batch_size=4,
        tau=1.0,
        train_iterations=1,
        vsteps_per_train=2,
        trains_per_target_update=10,
        gamma=0.99,
        max_grad_norm=0.5,

        logparams={},
        cfg_file=None,
        seed=42,
        skip_wandb_init=False,
        env=EnvArgs(
            encoding_type="float",
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
            reward_clip_tanh_army_frac=1,
            reward_army_value_ref=0,

        ),
        env_wrappers=[],
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
        n_quantiles=3,
        network=[
            dict(t="Flatten", start_dim=2),
            dict(t="Unflatten", dim=1, unflattened_size=[1, 11]),
            dict(t="Conv2d", in_channels=1, out_channels=32, kernel_size=[1, 574], stride=[1, 574], padding=0),
            dict(t="LeakyReLU"),
            dict(t="Flatten"),
            dict(t="Linear", in_features=5280, out_features=1024),
        ],
    )


# if __name__ == "__main__":
#     args = tyro.cli(Args)
#     main(args)

if __name__ == "__main__":
    # Run from vcmi-gym root:
    # $ python -m rl.algos.mqrdqn
    main(debug_args())
