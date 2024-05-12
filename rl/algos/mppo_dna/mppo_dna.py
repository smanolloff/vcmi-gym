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
import sys
import random
import logging
import time
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
# import tyro

from vcmi_gym import VcmiEnv
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
class NetworkArgs:
    features_extractor: list[dict] = field(default_factory=list)
    actor: dict = field(default_factory=dict)
    critic: dict = field(default_factory=dict)


@dataclass
class State:
    resumes: int = 0
    map_swaps: int = 0
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0
    current_rollout: int = 0
    global_second: int = 0
    current_second: int = 0

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
    mapmask: str = "gym/A1.vmap"
    randomize_maps: bool = False
    permasave_every: int = 7200  # seconds; no retention
    save_every: int = 3600  # seconds; retention (see max_saves)
    max_saves: int = 3
    out_dir_template: str = "data/{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    lr_schedule_value: ScheduleArgs = ScheduleArgs()
    lr_schedule_policy: ScheduleArgs = ScheduleArgs()
    lr_schedule_distill: ScheduleArgs = ScheduleArgs()
    clip_coef: float = 0.2
    clip_vloss: bool = False
    distill_beta: float = 1.0
    ent_coef: float = 0.01
    gae_lambda_policy: float = 0.95
    gae_lambda_value: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    num_envs: int = 1
    num_minibatches_distill: int = 4
    num_minibatches_policy: int = 4
    num_minibatches_value: int = 4
    num_steps: int = 128
    stats_buffer_size: int = 100
    switch_train_eval: bool = True
    update_epochs_distill: int = 4
    update_epochs_policy: int = 4
    update_epochs_value: int = 4
    vf_coef: float = 0.5
    weight_decay: float = 0.0
    target_kl: float = None

    logparams: dict = field(default_factory=dict)
    cfg_file: Optional[str] = None
    seed: int = 42
    skip_wandb_init: bool = False
    skip_wandb_log_code: bool = False

    env: EnvArgs = EnvArgs()
    env_wrappers: list = field(default_factory=list)
    network: NetworkArgs = NetworkArgs()

    def __post_init__(self):
        if not isinstance(self.env, EnvArgs):
            self.env = EnvArgs(**self.env)
        if not isinstance(self.lr_schedule_value, ScheduleArgs):
            self.lr_schedule_value = ScheduleArgs(**self.lr_schedule_value)
        if not isinstance(self.lr_schedule_policy, ScheduleArgs):
            self.lr_schedule_policy = ScheduleArgs(**self.lr_schedule_policy)
        if not isinstance(self.lr_schedule_distill, ScheduleArgs):
            self.lr_schedule_distill = ScheduleArgs(**self.lr_schedule_distill)
        if not isinstance(self.network, NetworkArgs):
            self.network = NetworkArgs(**self.network)

        common.coerce_dataclass_ints(self)


class ChanFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        # TODO: attn_mask
        res, _weights = super().forward(x, x, x, need_weights=False, attn_mask=None)
        return res


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

        # 1 nonhex action (RETREAT) + 165 hexes*14 actions each
        assert action_space.n == 1 + (165*14)

        self.features_extractor = torch.nn.Sequential()
        for spec in network.features_extractor:
            layer = AgentNN.build_layer(spec)
            self.features_extractor.append(common.layer_init(layer))

        self.actor = common.layer_init(AgentNN.build_layer(network.actor), gain=0.01)
        self.critic = common.layer_init(AgentNN.build_layer(network.critic), gain=1.0)

    def get_value(self, x):
        return self.critic(self.features_extractor(x))

    def get_action_and_value(self, x, mask, action=None, deterministic=False):
        features = self.features_extractor(x)
        value = self.critic(features)
        action_logits = self.actor(features)
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
            b_obs = torch.as_tensor(b_obs).cpu()
            b_mask = torch.as_tensor(b_mask).cpu()

            # Return unbatched action if input was unbatched
            if b_obs.shape == self.observation_space.shape:
                b_obs = b_obs.unsqueeze(dim=0)
                b_mask = b_mask.unsqueeze(dim=0)
                b_env_action, _, _, _, _ = self.get_action_and_value(b_obs, b_mask, deterministic=True)
                return b_env_action[0].cpu().item()
            else:
                b_env_action, _, _, _, _ = self.get_action_and_value(b_obs, b_mask, deterministic=True)
                return b_env_action.cpu().numpy()


class Agent(nn.Module):
    """
    Store a "clean" version of the agent: create a fresh one and copy the attrs
    manually (for nn's and optimizers - copy their states).
    This prevents issues if the agent contains wandb hooks at save time.
    """
    @staticmethod
    def save(agent, agent_file, nn_file=None):
        print("Saving agent to %s" % agent_file)
        attrs = ["args", "observation_space", "action_space", "state"]
        data = {k: agent.__dict__[k] for k in attrs}
        clean_agent = agent.__class__(**data)
        clean_agent.NN_value.load_state_dict(agent.NN_value.state_dict(), strict=True)
        clean_agent.NN_policy.load_state_dict(agent.NN_policy.state_dict(), strict=True)
        clean_agent.optimizer_value.load_state_dict(agent.optimizer_value.state_dict())
        clean_agent.optimizer_policy.load_state_dict(agent.optimizer_policy.state_dict())
        clean_agent.optimizer_distill.load_state_dict(agent.optimizer_distill.state_dict())
        torch.save(clean_agent, agent_file)

    @staticmethod
    def load(agent_file):
        print("Loading agent from %s" % agent_file)
        return torch.load(agent_file)

    def __init__(self, args, observation_space, action_space, state=None):
        super().__init__()
        self.args = args
        self.observation_space = observation_space  # needed for save/load
        self.action_space = action_space  # needed for save/load
        self._optimizer_state = None  # needed for save/load
        self.NN_value = AgentNN(args.network, action_space, observation_space)
        self.NN_policy = AgentNN(args.network, action_space, observation_space)
        self.optimizer_value = torch.optim.AdamW(self.NN_value.parameters(), eps=1e-5)
        self.optimizer_policy = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.optimizer_distill = torch.optim.AdamW(self.NN_policy.parameters(), eps=1e-5)
        self.predict = self.NN_policy.predict
        self.state = state or State()


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
    LOG = logging.getLogger("mppo_conv")
    LOG.setLevel(args.loglevel)

    assert isinstance(args, Args)

    agent, args = common.maybe_resume(Agent, args)

    if args.seconds_total:
        assert not args.vsteps_total, "cannot have both vsteps_total and seconds_total"
        rollouts_total = 0
    else:
        rollouts_total = args.vsteps_total // args.num_steps

    # Re-initialize to prevent errors from newly introduced args when loading/resuming
    # TODO: handle removed args
    args = Args(**vars(args))
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

    lr_schedule_fn_value = common.schedule_fn(args.lr_schedule_value)
    lr_schedule_fn_policy = common.schedule_fn(args.lr_schedule_policy)
    lr_schedule_fn_distill = common.schedule_fn(args.lr_schedule_distill)

    batch_size_policy = int(args.num_envs * args.num_steps)
    batch_size_value = int(args.num_envs * args.num_steps)
    batch_size_distill = int(args.num_envs * args.num_steps)
    minibatch_size_policy = int(batch_size_policy // args.num_minibatches_policy)
    minibatch_size_value = int(batch_size_value // args.num_minibatches_value)
    minibatch_size_distill = int(batch_size_distill // args.num_minibatches_distill)

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

    if args.agent_load_file and not agent:
        f = args.agent_load_file
        agent = Agent.load(f)
        agent.args = args
        start_map_swaps = agent.state.map_swaps
        agent.state.current_timestep = 0
        agent.state.current_vstep = 0
        agent.state.current_rollout = 0
        agent.state.current_second = 0

        # backup = "%s/loaded-%s" % (os.path.dirname(f), os.path.basename(f))
        # with open(f, 'rb') as fsrc:
        #     with open(backup, 'wb') as fdst:
        #         shutil.copyfileobj(fsrc, fdst)
        #         LOG.info("Wrote backup %s" % backup)

    common.validate_tags(args.tags)

    try:
        envs, _ = common.create_venv(VcmiEnv, args, start_map_swaps)
        [ENVS.append(e) for e in envs.unwrapped.envs]  # DEBUG

        obs_space = envs.unwrapped.single_observation_space
        act_space = envs.unwrapped.single_action_space

        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

        if agent is None:
            agent = Agent(args, obs_space, act_space)

        agent = agent.to(device)

        # these are used by gym's RecordEpisodeStatistics wrapper
        envs.return_queue = agent.state.ep_rew_queue
        envs.length_queue = agent.state.ep_length_queue

        # XXX: the start=0 requirement is needed for SB3 compat
        assert act_space.start == 0
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

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # XXX: the start=0 requirement is needed for SB3 compat
        assert act_space.start == 0
        masks = torch.zeros((args.num_steps, args.num_envs, act_space.n), dtype=torch.bool).to(device)

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask"))).to(device)

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

            if args.switch_train_eval:
                agent.eval()

            for step in range(0, args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                masks[step] = next_mask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, _, _ = agent.NN_policy.get_action_and_value(next_obs, next_mask)
                    value = agent.NN_value.get_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask"))).to(device)

                # XXX SIMO: SB3 does bootstrapping for truncated episodes here
                # https://github.com/DLR-RM/stable-baselines3/pull/658

                # See notes/gym_vector.txt
                for final_info, has_final_info in zip(infos.get("final_info", []), infos.get("_final_info", [])):
                    # "final_info" must be None if "has_final_info" is False
                    if has_final_info:
                        assert final_info is not None, "has_final_info=True, but final_info=None"
                        agent.state.ep_net_value_queue.append(final_info["net_value"])
                        agent.state.ep_is_success_queue.append(final_info["is_success"])

                agent.state.current_vstep += 1
                agent.state.current_timestep += args.num_envs
                agent.state.global_timestep += args.num_envs
                agent.state.current_second = int(time.time() - start_time)
                agent.state.global_second = global_start_second + agent.state.current_second

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.NN_value.get_value(next_obs).reshape(1, -1)
                advantages, _ = compute_advantages(
                    rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_policy
                )
                _, returns = compute_advantages(rewards, dones, values, next_done, next_value, args.gamma, args.gae_lambda_value)

            # flatten the batch
            b_obs = obs.reshape((-1,) + obs_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + act_space.shape)
            b_masks = masks.reshape((-1,) + (act_space.n,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Policy network optimization
            b_inds = np.arange(batch_size_policy)
            clipfracs = []

            if args.switch_train_eval:
                agent.train()

            for epoch in range(args.update_epochs_policy):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_policy, minibatch_size_policy):
                    end = start + minibatch_size_policy
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, _, _ = agent.NN_policy.get_action_and_value(
                        b_obs[mb_inds],
                        b_masks[mb_inds],
                        b_actions.long()[mb_inds]
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
            agent.NN_policy.zero_grad(True)  # don't clone gradients
            old_NN_policy = copy.deepcopy(agent.NN_policy)
            old_NN_policy.eval()
            for epoch in range(args.update_epochs_distill):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size_distill, minibatch_size_distill):
                    end = start + minibatch_size_distill
                    mb_inds = b_inds[start:end]

                    # Compute policy and value targets
                    with torch.no_grad():
                        _, _, _, old_action_dist, _ = old_NN_policy.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds])
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

            wandb_log({"params/value_learning_rate": agent.optimizer_value.param_groups[0]["lr"]})
            wandb_log({"params/policy_learning_rate": agent.optimizer_policy.param_groups[0]["lr"]})
            wandb_log({"params/distill_learning_rate": agent.optimizer_distill.param_groups[0]["lr"]})

            wandb_log({"losses/value_loss": v_loss.item()})
            wandb_log({"losses/policy_loss": pg_loss.item()})
            wandb_log({"losses/distill_loss": pg_loss.item()})
            wandb_log({"losses/entropy": distill_loss.item()})
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

            if args.rollouts_per_mapchange and map_rollouts % args.rollouts_per_mapchange == 0:
                map_rollouts = 0
                agent.state.map_swaps += 1
                wandb_log({"global/map_swaps": agent.state.map_swaps})
                envs.close()
                envs, _ = common.create_venv(VcmiEnv, args, agent.state.map_swaps)
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_mask"))).to(device)

            if agent.state.current_rollout > 0 and agent.state.current_rollout % args.rollouts_per_log == 0:
                wandb_log({
                    "global/global_num_timesteps": agent.state.global_timestep,
                    "global/global_num_seconds": agent.state.global_second
                }, commit=True)  # commit on final log line

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
            save_ts, permasave_ts = common.maybe_save(save_ts, permasave_ts, args, agent, out_dir)

    finally:
        common.maybe_save(0, 10e9, args, agent, out_dir)
        envs.close()

    # Needed by PBT to save model after iteration ends
    return agent, common.safe_mean(agent.state.rollout_rew_queue_1000)


def debug_args():
    return Args(
        "debug-crl",
        "debug-crl",
        loglevel=logging.DEBUG,
        run_name=None,
        wandb_project=None,
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="data/heads/heads-simple-A1/agent-1710806916.zip",
        agent_load_file=None,
        vsteps_total=0,
        seconds_total=0,
        rollouts_per_mapchange=0,
        rollouts_per_log=1,
        rollouts_per_table_log=100000,
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
        lr_schedule_value=ScheduleArgs(mode="const", start=0.001),
        lr_schedule_policy=ScheduleArgs(mode="const", start=0.001),
        lr_schedule_distill=ScheduleArgs(mode="const", start=0.001),
        num_envs=1,
        num_steps=8,
        gamma=0.8,
        gae_lambda_policy=0.95,
        gae_lambda_value=0.95,
        num_minibatches_value=4,
        num_minibatches_policy=4,
        num_minibatches_distill=4,
        update_epochs_value=2,
        update_epochs_policy=2,
        update_epochs_distill=2,
        norm_adv=True,
        clip_coef=0.3,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=1.2,
        max_grad_norm=0.5,
        distill_beta=1.0,
        switch_train_eval=True,
        target_kl=None,
        logparams={},
        cfg_file=None,
        seed=42,
        skip_wandb_init=False,
        skip_wandb_log_code=False,
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
        network=dict(
            features_extractor=[
                dict(t="Flatten", start_dim=2),
                dict(t="Unflatten", dim=1, unflattened_size=[1, 11]),
                dict(t="Conv2d", in_channels=1, out_channels=32, kernel_size=[1, 86], stride=[1, 86], padding=0),
                dict(t="LeakyReLU"),
                dict(t="Flatten"),
                dict(t="Linear", in_features=5280, out_features=1024),
            ],
            actor=dict(t="Linear", in_features=1024, out_features=2311),
            critic=dict(t="Linear", in_features=1024, out_features=1)
        )
    )


if __name__ == "__main__":
    # To run from vcmi-gym root:
    # $ python -m rl.algos.mppo
    main(debug_args())
