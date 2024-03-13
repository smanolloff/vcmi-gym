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
import random
import time
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
# import tyro
from torch.utils.tensorboard import SummaryWriter

from vcmi_gym import VcmiEnv

from . import common


@dataclass
class EnvArgs:
    max_steps: int = 500
    reward_dmg_factor: int = 5
    vcmi_loglevel_global: str = "error"
    vcmi_loglevel_ai: str = "error"
    vcmienv_loglevel: str = "WARN"
    sparse_info: bool = True
    step_reward_mult: int = 1
    term_reward_mult: int = 0
    reward_clip_mod: Optional[int] = None
    consecutive_error_reward_factor: Optional[int] = None


@dataclass
class State:
    resumes: int = 0
    map_swaps: int = 0
    global_step: int = 0
    global_rollout: int = 0
    optimizer_state_dict: Optional[dict] = None


@dataclass
class Args:
    run_id: str
    group_id: str
    resume: bool = False
    overwrite: list = field(default_factory=list)
    notes: Optional[str] = None

    agent_load_file: Optional[str] = None
    rollouts_total: int = 10000
    rollouts_per_mapchange: int = 20
    rollouts_per_log: int = 1
    success_rate_target: Optional[float] = None
    mapmask: str = "ai/generated/B*.vmap"
    randomize_maps: bool = False
    save_every: int = 3600  # seconds
    max_saves: int = 3
    out_dir_template: str = "data/CRL_MPPO-{group_id}/{run_id}"

    opponent_load_file: Optional[str] = None
    opponent_sbm_probs: list = field(default_factory=lambda: [1, 0, 0])
    weight_decay: float = 0.0
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
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
    wandb: bool = True
    seed: int = 42

    env: EnvArgs = EnvArgs()
    env_wrappers: list = field(default_factory=list)
    state: State = State()

    def __post_init__(self):
        if not isinstance(self.env, EnvArgs):
            self.env = EnvArgs(**self.env)
        if not isinstance(self.state, State):
            self.state = State(**self.state)


class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        # TODO: attn_mask
        res, _weights = super().forward(x, x, x, need_weights=False, attn_mask=None)
        return res


class Agent(nn.Module):
    def __init__(self, observation_space, action_space, state):
        super().__init__()

        self.state = state

        assert observation_space.shape[0] == 1
        assert observation_space.shape[1] == 11
        assert observation_space.shape[2] / 56 == 15

        self.features_extractor = common.layer_init(nn.Sequential(
            # loaded from file
        ))

        self.actor = common.layer_init(nn.Linear(165, action_space.n), gain=0.01)
        self.critic = common.layer_init(nn.Linear(165, 1), gain=1.0)

    def get_value(self, x):
        return self.critic(self.features_extractor(x))

    def get_action_and_value(self, x, mask, action=None):
        with torch.no_grad():
            old_features = self.features_extractor[:-2](x)
        features = self.features_extractor[-2:](old_features)
        value = self.critic(features)
        action_logits = self.actor(features)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    # Inference (deterministic)
    def predict(self, x, mask):
        with torch.no_grad():
            logits = self.actor(self.features_extractor(x))
            dist = common.CategoricalMasked(logits=logits, mask=mask)
            return torch.argmax(dist.probs, dim=1).cpu().numpy()


def main(args):
    assert isinstance(args, Args)

    args = common.maybe_resume_args(args)

    # XXX: handle newly introduced options
    args = Args(**vars(args))
    # TODO: handle removed options

    print("Args: %s" % (asdict(args)))
    common.maybe_setup_wandb(args, __file__)
    out_dir = args.out_dir_template.format(seed=args.seed, group_id=args.group_id, run_id=args.run_id)
    print("Out dir: %s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    writer = SummaryWriter(out_dir)
    common.log_params(args, writer)

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    save_ts = None
    agent = None
    optimizer = None
    start_map_swaps = args.state.map_swaps

    if args.agent_load_file:
        f = args.agent_load_file
        print("Loading agent from %s" % f)
        agent = torch.load(f)
        start_map_swaps = agent.state.map_swaps

        backup = "%s/loaded-%s" % (os.path.dirname(f), os.path.basename(f))
        with open(f, 'rb') as fsrc:
            with open(backup, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
                print("Wrote backup %s" % backup)

    try:
        envs = common.create_venv(VcmiEnv, args, writer, start_map_swaps)  # noqa: E501
        obs_space = envs.unwrapped.single_observation_space
        act_space = envs.unwrapped.single_action_space

        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

        if agent is None:
            agent = Agent(obs_space, act_space, args.state).to(device)

        if args.resume:
            agent.state.resumes += 1
            writer.add_scalar("global/resumes", agent.state.resumes)

        # print("Agent state: %s" % asdict(agent.state))

        optimizer = common.init_optimizer(args, agent, optimizer)
        ep_net_value_queue = deque(maxlen=envs.return_queue.maxlen)
        ep_is_success_queue = deque(maxlen=envs.return_queue.maxlen)

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
        next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_masks"))).to(device)

        start_rollout = agent.state.global_rollout + 1
        assert start_rollout < args.rollouts_total

        for rollout in range(start_rollout, args.rollouts_total + 1):
            agent.state.global_rollout = rollout
            rollout_start_time = time.time()
            rollout_start_step = agent.state.global_step

            # XXX: eval during experience collection
            agent.eval()
            for step in range(0, args.num_steps):
                agent.state.global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                masks[step] = next_mask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs, next_mask)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_masks"))).to(device)

                # XXX SIMO: SB3 does bootstrapping for truncated episodes here
                # https://github.com/DLR-RM/stable-baselines3/pull/658

                # See notes/gym_vector.txt
                for final_info, has_final_info in zip(infos.get("final_info", []), infos.get("_final_info", [])):
                    # "final_info" must be None if "has_final_info" is False
                    if has_final_info:
                        assert final_info is not None, "has_final_info=True, but final_info=None"
                        ep_net_value_queue.append(final_info["net_value"])
                        ep_is_success_queue.append(final_info["is_success"])

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
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
            b_masks = masks.reshape((-1,) + (act_space.n,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

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

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
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

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            # writer.add_scalar("params/learning_rate", optimizer.param_groups[0]["lr"], agent.state.global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), agent.state.global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), agent.state.global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), agent.state.global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), agent.state.global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), agent.state.global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), agent.state.global_step)
            writer.add_scalar("losses/explained_variance", explained_var, agent.state.global_step)
            writer.add_scalar("time/rollout_duration", time.time() - rollout_start_time)
            writer.add_scalar("time/steps_per_second", (agent.state.global_step - rollout_start_step) / (time.time() - rollout_start_time))  # noqa: E501
            writer.add_scalar("rollout/ep_rew_mean", common.safe_mean(envs.return_queue))
            writer.add_scalar("rollout/ep_len_mean", common.safe_mean(envs.length_queue))
            writer.add_scalar("rollout/ep_value_mean", common.safe_mean(ep_net_value_queue))
            writer.add_scalar("rollout/ep_success_rate", common.safe_mean(ep_is_success_queue))
            writer.add_scalar("rollout/ep_count", envs.episode_count)
            writer.add_scalar("global/num_timesteps", agent.state.global_step)
            writer.add_scalar("global/num_rollouts", agent.state.global_rollout)
            writer.add_scalar("global/progress", agent.state.global_rollout / args.rollouts_total)

            print(f"global_step={agent.state.global_step}, rollout/ep_rew_mean={common.safe_mean(envs.return_queue)}")

            if args.success_rate_target and common.safe_mean(ep_is_success_queue) >= args.success_rate_target:
                writer.flush()
                print("Early stopping after %d rollouts due to: success rate > %.2f (%.2f)" % (
                    rollout % args.rollouts_per_mapchange,
                    args.success_rate_target,
                    common.safe_mean(ep_is_success_queue)
                ))

            if rollout > start_rollout and rollout % args.rollouts_per_log == 0:
                writer.flush()
                # reset per-rollout stats (affects only logging)
                envs.return_queue.clear()
                envs.length_queue.clear()
                # envs.time_queue.clear()  # irrelevant
                ep_net_value_queue.clear()
                ep_is_success_queue.clear()
                envs.episode_count = 0

            if rollout > start_rollout and rollout % args.rollouts_per_mapchange == 0:
                agent.state.map_swaps += 1
                writer.add_scalar("global/map_swaps", agent.state.map_swaps)
                envs.close()
                envs = common.create_venv(VcmiEnv, args, writer, agent.state.map_swaps)  # noqa: E501
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_masks"))).to(device)

            save_ts = common.maybe_save(save_ts, args, agent, optimizer, out_dir)

    finally:
        common.maybe_save(0, args, agent, optimizer, out_dir)
        envs.close()
        writer.close()


# if __name__ == "__main__":
#     args = tyro.cli(Args)
#     main(args)


if __name__ == "__main__":
    args = Args(
        "debug-crl",
        "debug-crl",
        resume=False,
        overwrite=[],
        notes=None,
        # agent_load_file="debugging/crl-agent.pt",
        rollouts_total=1000000,
        rollouts_per_mapchange=1000,
        rollouts_per_log=1000,
        opponent_load_file=None,
        success_rate_target=None,
        mapmask="ai/P1.vmap",
        randomize_maps=False,
        save_every=2000000000,  # greater than time.time()
        max_saves=0,
        out_dir_template="data/debug-crl/debug-crl",
        weight_decay=0.0,
        learning_rate=0.001,
        num_envs=2,
        num_steps=4,
        gamma=0.8,
        gae_lambda=0.8,
        num_minibatches=2,
        update_epochs=2,
        norm_adv=True,
        clip_coef=0.4,
        clip_vloss=False,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        logparams={},
        cfg_file=None,
        wandb=False,
        seed=42,
        env=EnvArgs(
            max_steps=500,
            reward_clip_mod=None,
            reward_dmg_factor=5,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            consecutive_error_reward_factor=-1,
            sparse_info=True,
            step_reward_mult=1,
            term_reward_mult=0,
        ),
        env_wrappers=[],
        # env_wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
    )

    main(args)
