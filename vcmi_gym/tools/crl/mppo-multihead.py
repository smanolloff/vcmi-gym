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
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
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
    opponent_load_file: Optional[str] = None
    success_rate_target: Optional[float] = None
    mapmask: str = "ai/generated/B*.vmap"
    randomize_maps: bool = False
    save_every: int = 3600  # seconds
    max_saves: int = 3
    out_dir_template: str = "data/CRL_MPPO-{group_id}/{run_id}"

    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    cfg_file: str = 'path/to/cfg.yml'
    wandb: bool = True
    seed: int = 42

    env: EnvArgs = EnvArgs()
    state: State = State()

    def __post_init__(self):
        if not isinstance(self.env, EnvArgs):
            self.env = EnvArgs(**self.env)
        if not isinstance(self.state, State):
            self.state = State(**self.state)


class Bx1x11x15N_to_Bx165xN(nn.Module):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def forward(self, x):
        # (B, 1, 11, 15*N) -> (B, N, 11, 15, 1) -> (B, 165, N)
        return x.unflatten(-1, (15, self.n)) \
                .permute(0, 4, 2, 3, 1) \
                .flatten(start_dim=-3) \
                .permute(0, 2, 1)


# See notes/reshape_Bx165xE_to_BxEx11x15.py
class Bx165xE_to_Ex11x15(nn.Module):
    def __init__(self, e):
        self.e = e
        super().__init__()

    def forward(self, x):
        # (B, 165, E) -> (B, E, 11, 15)
        return x.unflatten(1, (11, 15)).permute(0, 3, 1, 2)


class Agent(nn.Module):
    def __init__(self, observation_space, action_space, state):
        super().__init__()

        self.state = state

        # 1 nonhex action (RETREAT) + 165 hexex*14 actions each
        assert action_space.n == 1 + (165*14)

        assert observation_space.shape[0] == 1
        assert observation_space.shape[1] == 11
        assert observation_space.shape[2] / 56 == 15

        # Produces hex embeddings
        self.hex_embedder = nn.Sequential(
            # => (B, 1, 11, 840)
            Bx1x11x15N_to_Bx165xN(n=56),
            # => (B, 165, 56)
            common.layer_init(nn.Linear(56, 128)),
            nn.LeakyReLU(),
            # => (B, 165, 128)
            Bx165xE_to_Ex11x15(e=128),
            # => (B, 128, 11, 15)
            common.layer_init(nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # => (B, 32, 11, 15)
            common.layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # => (B, 64, 11, 15)
        )

        # Produces a summary embedding
        self.summary_embedder = nn.Sequential(
            # => (B, 64, 11, 15)
            nn.Flatten(),
            # => (B, 10560)
            common.layer_init(nn.Linear(10560, 256)),
            nn.LeakyReLU(),
            # => (B, 256)
        )

        #
        # Value head:
        #
        # => (B, 256)
        self.value_net = common.layer_init(nn.Linear(256, 1), std=1.0)
        # => (B, 1)

        #
        # Action Head #1: one of 5 actions: WAIT, DEFEND, SHOOT, MOVE, AMOVE
        #
        # => (B, 256)
        self.action1_net = common.layer_init(nn.Linear(256, 5), std=0.01)
        # => (B, 5)

        #
        # Action Head #2: one of 165 hexes given hex_embeddings + action
        #
        self.action2_net = nn.Sequential(
            # => (B, 65, 11, 15)
            common.layer_init(nn.Conv2d(65, 1, kernel_size=1, stride=1, padding=0)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            # => (B, 1, 11, 15)
            nn.Flatten(),
            # => (B, 1, 165)
            common.layer_init(nn.Linear(165, 1), std=0.01)
        )

        #
        # Action Head #3: one of 165 hexes given hex_embeddings + action + hex1
        #
        self.action3_net = nn.Sequential(
            # => (B, 66, 11, 15)
            common.layer_init(nn.Conv2d(66, 1, kernel_size=1, stride=1, padding=0)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            # => (B, 1, 11, 15)
            nn.Flatten(),
            # => (B, 1, 165)
            common.layer_init(nn.Linear(165, 1), std=0.01)
        )

    # see notes/concat_B_and_BxNx11x15.py
    def concat_B_and_BxNx11x15(self, b_action, b_hex_embeddings):
        assert len(b_action.shape) == 1
        broadcasted = b_action. \
            unsqueeze(-1). \
            unsqueeze(-1). \
            unsqueeze(-1). \
            broadcast_to(b_action.shape[0], 1, 11, 15)

        return torch.cat(broadcasted, b_hex_embeddings)

    def get_value(self, x):
        return self.value_net(self.summary_embedder(self.hex_embedder(x)))

    # XXX: b_mask and b_action are batched, e.g. for 2 batches:
    #   b_mask = [[<2311 bools>], [<2311 bools]]
    #   b_action = [<int>, <int>]
    def get_action_and_value(self, b_obs: np.ndarray, b_mask: np.ndarray, b_action=Optional[torch.Tensor]):
        b_size = len(b_obs)

        # Per-hex action mask (see notes/mask.txt)
        def amove_mask(m):
            # Split the 2311-element mask into 165 chunks (1 chunk per hex)
            return [any(hexmask[1:]) for hexmask in np.split(m[1:], 165)]

        # must go through np.array first
        b_mask_shoot = [m[14::14] for m in b_mask]
        b_mask_move = [m[1::14] for m in b_mask]
        b_mask_amove = [amove_mask(m) for m in b_mask]

        # When collecting experiences, b_action is None
        # When updating the policy, b_action is provided
        need_action = b_action is None

        #
        # Head 1
        #

        b_hex_embeddings = self.hex_embedder(torch.as_tensor(b_obs))
        b_summary_embedding = self.summary_embedder(b_hex_embeddings)
        b_value = self.value_net(b_summary_embedding)

        def mask_for_action1(i):
            return [
                b_mask[i][0],            # WAIT
                True,                    # DEFEND
                any(b_mask_shoot[i]),    # SHOOT
                any(b_mask_move[i]),     # MOVE
                any(b_mask_amove[i]),    # AMOVE
            ]

        b_action1_masks = torch.as_tensor([mask_for_action1(i) for i in range(b_size)])
        b_action1_logits = self.action1_net(b_summary_embedding)
        b_action1_dist = common.CategoricalMasked(logits=b_action1_logits, mask=b_action1_masks)

        if need_action:
            b_action1 = b_action1_dist.sample()

        b_action1_log_prob = b_action1_dist.log_prob(b_action1)
        b_action1_entropy = b_action1_dist.entropy()

        #
        # Head 2
        #

        b_hex_embeddings2 = self.concat_B_and_BxNx11x15(b_action1, b_hex_embeddings)

        # action2 is the target hex
        def mask_for_action2(i):
            action1 = b_action1[i]
            return b_mask_shoot[i] if action1 == 0 \
                else b_mask_move[i] if action1 == 1 \
                else b_mask_amove[i] if action1 == 2 \
                else Exception("not supposed to be here")

        b_action2_masks = torch.as_tensor(np.array([mask_for_action2(i) for i in range(b_size)]))
        b_action2_logits = self.action2_net(b_hex_embeddings2)
        b_action2_dist = common.CategoricalMasked(logits=b_action2_logits, mask=b_action2_masks)

        if need_action:
            b_action2 = b_action2_dist.sample()

        b_action2_log_prob = b_action2_dist.log_prob(b_action2)
        b_action2_entropy = b_action2_dist.entropy()

        #
        # Head 3
        #

        b_hex_embeddings3 = self.concat_B_and_BxNx11x15(b_action2, b_hex_embeddings2)

        # Hex offsets for attacking at direction 0..12
        # Depends if the row is even (offset0) or odd (offset1)
        # see notes/masks.txt
        # XXX: indexes MUST correspond to the directions in hexaction.h
        offsets0 = [-15, -14, 1, 16, 15, -1, 14, -2, -16, -13, 2, 17]
        offsets1 = [-16, -15, 1, 15, 14, -1, 13, -2, -17, -14, 2, 16]

        # calculate the offsets for each action2 (hex) in the batch:
        # e.g. b_offsets = [offsets1, offsets0]  (if batch=2)
        b_offsets = [[
            offsets0 if (b_action2[i]//15) % 2 == 0 else offsets1
        ] for i in range(b_size)]

        # action3 is the target hex
        def mask_for_action3(i: int, action2: int):
            mask = np.zeros(165, dtype=bool)
            for offset in b_offsets[i]:
                hex_target = action2 + offset
                if hex_target >= 0 and hex_target < len(mask):
                    mask[hex_target] = True

            return mask

        b_action3_masks = torch.as_tensor(np.array([mask_for_action3(i) for i in range(b_size)]))
        b_action3_logits = self.action3_net(b_hex_embeddings3)
        b_action3_dist = common.CategoricalMasked(logits=b_action3_logits, mask=b_action3_masks)

        if need_action:
            b_action3 = b_action3_dist.sample()

        b_action3_log_prob = b_action3_dist.log_prob(b_action3)
        b_action3_entropy = b_action3_dist.entropy()

        #
        # Result
        #

        def calc_action(i):
            a1, a2, a3 = b_actions[i]

            if a1 == 0:
                # WAIT
                return 0
            elif a1 == 1:
                # DEFEND
                # The 19th HexAttribute is IsActive
                # (see Battlefield::exportState() in battlefield.cpp)
                hex = np.where(b_obs[i].flatten()[19::56] == 1)
                # Defend is just a no-op MOVE (move to the active hex)
                return 1 + hex*14
            elif a1 == 2:
                # SHOOT
                return 1 + a2*14 + 13
            elif a1 == 3:
                # MOVE
                return 1 + a2*14
            elif a1 == 4:
                # AMOVE
                # find the index of a3 (the target hex) amongst a2's neughbours
                # this is the "direction" of attack from a2 (the source hex)
                a2_neighbours = b_offsets[i] + a2
                direction = np.where(a2_neighbours == a3)[0][0]

                # 0 is MOVE, 1 corresponds to direction=0 (see hexaction.h)
                return 1 + a2*14 + direction + 1

            raise Exception("Should not be here: a1 = %s" % a1)

        if need_action:
            # b_actions is a List<List<int, 3>, B>
            # b_action is a List<int, B> (1 int represents a gym discrete action)
            # Need to reconstruct the integer from the 3 elements

            # 1. Reshape
            #    [[1, 10], [2, 20], [3, 30]] => [[1, 2, 3], [10, 20, 30]]
            b_actions = np.transpose([b_action1, b_action2, b_action3])
            # 2. Combine the 3-head actions into a single integer
            #    [[1,2,3], [10,20,30]] => [124, 1631]
            b_action = torch.as_tensor([calc_action(i) for i in range(b_size)])

        # b_log_probs and b_entropies contain values for each action head
        # Example:
        # [
        #   [b1_a1_logprob, b1_a2_logprob, b1_a3_logprob],  # batch 1
        #   [b2_a1_logprob, b2_a2_logprob, b2_a3_logprob],  # batch 2
        # ]
        # Same applies for b_entropies
        b_log_probs = np.transpose([b_action1_log_prob, b_action2_log_prob, b_action3_log_prob])
        b_entropies = np.transpose([b_action1_entropy, b_action2_entropy, b_action3_entropy])

        #
        # Example return (for batch=2):
        # b_action = [214, 1631]
        # b_log_probs = [[-0.41, -0.23, -0.058], [-0.64, -0.01, -0.093]]
        # b_entropies = [[0.081, 0.31, 0.11], [0.123, 0.714, 0.4]]
        # b_value = [5.125, 67.11]

        return b_action, b_log_probs, b_entropies, b_value

    # Inference (deterministic)
    def predict(self, x, mask):
        with torch.no_grad():
            logits = self.actor(self.features_extractor(x))
            dist = common.CategoricalMasked(logits=logits, mask=mask)
            return torch.argmax(dist.probs, dim=1).cpu().numpy()


def main(args):
    assert isinstance(args, Args)

    args = common.maybe_resume_args(args)
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
        print("Loading agent from %s" % args.agent_load_file)
        agent = torch.load(args.agent_load_file)
        start_map_swaps = agent.state.map_swaps

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


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
