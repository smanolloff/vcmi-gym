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

import os
import random
import time
import re
import glob
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque
import concurrent.futures
import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from vcmi_gym import VcmiEnv


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


# https://boring-guy.sh/posts/masking-rl/
# combined with
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.2.1/sb3_contrib/common/maskable/distributions.py#L18
class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        assert mask is not None
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        self.mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype)
        logits = torch.where(self.mask, logits, self.mask_value)
        super().__init__(logits=logits)

    def entropy(self):
        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
        return -p_log_p.sum(-1)


def create_venv(env_cls, args, writer, map_swaps):
    mappath = "/Users/simo/Library/Application Support/vcmi/Maps"
    all_maps = glob.glob("%s/%s" % (mappath, args.mapmask))
    all_maps = [m.replace("%s/" % mappath, "") for m in all_maps]
    all_maps.sort()

    if args.num_envs == 1:
        n_maps = 1
    else:
        assert args.num_envs % 2 == 0
        assert args.num_envs <= len(all_maps) * 2
        n_maps = args.num_envs // 2

    if args.randomize_maps:
        maps = random.sample(all_maps, n_maps)
    else:
        i = (n_maps * map_swaps) % len(all_maps)
        new_i = (i + n_maps) % len(all_maps)
        # wandb.log({"map_offset": i}, commit=False)
        writer.add_scalar("global/map_offset", i)

        if new_i > i:
            maps = all_maps[i:new_i]
        else:
            maps = all_maps[i:] + all_maps[:new_i]

        assert len(maps) == n_maps

    pairs = [[("attacker", m), ("defender", m)] for m in maps]
    pairs = [x for y in pairs for x in y]  # aka. pairs.flatten(1)...
    state = {"n": 0}
    lock = threading.RLock()

    def env_creator():
        with lock:
            assert state["n"] < args.num_envs
            role, mapname = pairs[state["n"]]
            # logfile = f"/tmp/{run_id}-env{state['n']}-actions.log"
            logfile = None

            env_kwargs = dict(
                asdict(args.env),
                mapname=mapname,
                attacker="MMAI_MODEL" if args.opponent_load_file else "StupidAI",
                defender="MMAI_MODEL" if args.opponent_load_file else "StupidAI",
                attacker_model=args.opponent_load_file,
                defender_model=args.opponent_load_file,
                actions_log_file=logfile
            )

            env_kwargs[role] = "MMAI_USER"
            print("Env kwargs (env.%d): %s" % (state["n"], env_kwargs))
            state["n"] += 1

        return env_cls(**env_kwargs)

    if args.num_envs > 1:
        # I don't remember anymore, but there were issues if max_workers>8
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.num_envs, 8)) as executor:
            futures = [executor.submit(env_creator) for _ in range(args.num_envs)]
            results = [future.result() for future in futures]

        funcs = [lambda x=x: x for x in results]
        vec_env = gym.vector.SyncVectorEnv(funcs)
    else:
        vec_env = gym.vector.SyncVectorEnv([env_creator])

    vec_env = gym.wrappers.RecordEpisodeStatistics(vec_env)
    return vec_env


def maybe_save(t, args, agent, optimizer, out_dir):
    now = time.time()

    if t is None:
        return now

    if t + args.save_every > now:
        return t

    os.makedirs(out_dir, exist_ok=True)
    agent_file = os.path.join(out_dir, "agent-%d.pt" % now)
    agent.state.optimizer_state_dict = optimizer.state_dict()
    torch.save(agent, agent_file)
    print("Saved agent to %s" % agent_file)

    args_file = os.path.join(out_dir, "args-%d.pt" % now)
    torch.save(args, args_file)
    print("Saved args to %s" % args_file)

    # save file retention (keep latest N saves)
    files = sorted(
        glob.glob(os.path.join(out_dir, "agent-[0-9]*.pt")),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
        reverse=True
    )

    for file in files[args.max_saves:]:
        print("Deleting %s" % file)
        os.remove(file)
        argfile = "%s/args-%s" % (os.path.dirname(file), os.path.basename(file).removeprefix("agent-"))
        if os.path.isfile(argfile):
            print("Deleting %s" % argfile)
            os.remove(argfile)

    return now


def find_latest_save(group_id, run_id):
    threshold = datetime.datetime.now() - datetime.timedelta(hours=3)
    pattern = f"data/{group_id}/{run_id}/agent-[0-9]*.pt"
    files = glob.glob(pattern)
    assert len(files) > 0, f"No files found for: {pattern}"
    filtered = [f for f in files if datetime.datetime.fromtimestamp(os.path.getmtime(f)) > threshold]
    agent_file = max(filtered, key=os.path.getmtime)

    args_file = "%s/args-%s" % (os.path.dirname(agent_file), os.path.basename(agent_file).removeprefix("agent-"))
    assert os.path.isfile(args_file)

    return args_file, agent_file


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def safe_mean(array_like) -> float:
    return np.nan if len(array_like) == 0 else float(np.mean(array_like))


class Agent(nn.Module):
    def __init__(self, observation_space, action_space, state):
        super().__init__()

        self.state = state

        assert observation_space.shape[0] == 1
        assert observation_space.shape[1] == 11
        assert observation_space.shape[2] / 56 == 15

        self.features_extractor = nn.Sequential(
            # => (B, 1, 11, 840)
            layer_init(nn.Conv2d(1, 32, kernel_size=(1, 56), stride=(1, 56))),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # => (B, 32, 11, 15)
            nn.Flatten(),
            # => (B, 5280)
            layer_init(nn.Linear(5280, 1024)),
            nn.LeakyReLU(),
            # => (B, 1024)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(1024, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(1024, action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(self.features_extractor(x))

    def get_action_and_value(self, x, mask, action=None):
        logits = self.actor(self.features_extractor(x))
        dist = CategoricalMasked(logits=logits, mask=mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(x)

    # Inference (deterministic)
    def predict(self, x, mask):
        with torch.no_grad():
            logits = self.actor(self.features_extractor(x))
            dist = CategoricalMasked(logits=logits, mask=mask)
            return torch.argmax(dist.probs, dim=1).cpu().numpy()


def main(args):
    assert isinstance(args, Args)

    # XXX: resume will overwrite all input args except run_id & group_id
    if args.resume:
        args_load_file, agent_load_file = find_latest_save(args.group_id, args.run_id)
        loaded_args = torch.load(args_load_file)
        assert loaded_args.group_id == args.group_id
        assert loaded_args.run_id == args.run_id

        # List of arg names to overwrite after loading
        # (some args (incl. overwrite itself) must always be overwritten)
        loaded_args.overwrite = args.overwrite
        loaded_args.cfg_file = args.cfg_file

        for argname in args.overwrite:
            parts = argname.split(".")
            if len(parts) == 1:
                print("Overwrite %s: %s -> %s" % (argname, getattr(loaded_args, argname), getattr(args, argname)))
                setattr(loaded_args, argname, getattr(args, argname))
            else:
                assert len(parts) == 2
                sub_loaded = getattr(args, parts[0])
                sub_arg = getattr(args, parts[0])
                print("Overwrite %s: %s -> %s" % (argname, getattr(sub_loaded, parts[1]), getattr(sub_arg, args[1])))
                setattr(sub_loaded, parts[1], getattr(sub_arg, parts[1]))

        args = loaded_args
        args.resume = True
        args.agent_load_file = agent_load_file

        print("Resuming run %s/%s" % (args.group_id, args.run_id))
        print("Loaded args from %s" % args_load_file)

    else:
        print("Starting new run %s/%s" % (args.group_id, args.run_id))

    print("Args: %s" % (asdict(args)))

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

    if args.wandb:
        import wandb

        wandb.init(
            project="vcmi-gym",
            group=args.group_id,
            name=args.run_id,
            id=args.run_id,
            notes=args.notes,
            resume="must" if args.resume else "never",
            # resume="allow",  # XXX: reuse id for insta-failed runs
            config=asdict(args),
            sync_tensorboard=True,
            save_code=False,  # code saved manually below
            allow_val_change=args.resume,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
        )

        # https://docs.wandb.ai/ref/python/run#log_code
        # XXX: "path" is relative to THIS dir
        #      but args.cfg_file is relative to vcmi-gym ROOT dir
        def code_include_fn(path):
            res = (
                path.endswith(os.path.basename(__file__)) or
                path.endswith(os.path.basename(args.cfg_file)) or
                path.endswith("requirements.txt") or
                path.endswith("requirements.lock")
            )

            print("Should include %s: %s" % (path, res))
            return res

        wandb.run.log_code(root=os.path.dirname(__file__), include_fn=code_include_fn)

    out_dir = args.out_dir_template.format(seed=args.seed, group_id=args.group_id, run_id=args.run_id)
    print("Out dir: %s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    writer = SummaryWriter(out_dir)

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
        envs = create_venv(VcmiEnv, args, writer, start_map_swaps)  # noqa: E501
        obs_space = envs.unwrapped.single_observation_space
        act_space = envs.unwrapped.single_action_space

        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

        if agent is None:
            agent = Agent(obs_space, act_space, args.state).to(device)

        if args.resume:
            agent.state.resumes += 1
            writer.add_scalar("global/resumes", agent.state.resumes)

        print("Agent state: %s" % asdict(agent.state))

        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        if agent.state.optimizer_state_dict:
            optimizer.load_state_dict(agent.state.optimizer_state_dict)
            # Need to explicitly set lr after loading state
            optimizer.param_groups[0]["lr"] = args.learning_rate

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

                # For a vectorized environments the output will be in the form of::
                #     >>> infos = {
                #     ...     "final_observation": "<array<obs> of length num-envs>",
                #     ...     "_final_observation": "<array<bool> of length num-envs>",
                #     ...     "final_info": "<array<hash> of length num-envs>",
                #     ...     "_final_info": "<array<bool> of length num-envs>",
                #     ...     "episode": {
                #     ...         "r": "<array<float> of cumulative reward>",
                #     ...         "l": "<array<int> of episode length>",
                #     ...         "t": "<array<float> of elapsed time since beginning of episode>"
                #     ...     },
                #     ...     "_episode": "<boolean array of length num-envs>"
                #     ... }
                #
                # My notes:
                #   "episode" and "_episode" is added by RecordEpisodeStatistics wrapper
                #   gym's vec env *automatically* collects episode returns and lengths
                #   in envs.return_queue and envs.length_queue
                #   (eg. [-1024.2, 333.6, ...] and [34, 36, 41, ...]) - each element is a full episode
                #
                #  "final_info" and "_final_info" are NOT present at all if no env was done
                #   If at least 1 env was done, both are present, with info about all envs
                #   (this applies for all info keys)
                #
                #   Note that rewards are accessed as infos["episode"]["r"][i]
                #   ... but env's info is accessed as infos["final_info"][i][key]
                #
                # See
                #   https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/vector/sync_vector_env.py#L142-L157
                #   https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/vector/vector_env.py#L275-L300
                #   https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/wrappers/record_episode_statistics.py#L102-L124
                #
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
            writer.add_scalar("rollout/ep_rew_mean", safe_mean(envs.return_queue))
            writer.add_scalar("rollout/ep_len_mean", safe_mean(envs.length_queue))
            writer.add_scalar("rollout/ep_value_mean", safe_mean(ep_net_value_queue))
            writer.add_scalar("rollout/ep_success_rate", safe_mean(ep_is_success_queue))
            writer.add_scalar("rollout/ep_count", envs.episode_count)
            writer.add_scalar("global/num_timesteps", agent.state.global_step)
            writer.add_scalar("global/num_rollouts", agent.state.global_rollout)
            writer.add_scalar("global/progress", agent.state.global_rollout / args.rollouts_total)

            print(f"global_step={agent.state.global_step}, rollout/ep_rew_mean={safe_mean(envs.return_queue)}")

            if args.success_rate_target and safe_mean(ep_is_success_queue) >= args.success_rate_target:
                writer.flush()
                print("Early stopping after %d rollouts due to: success rate > %.2f (%.2f)" % (
                    rollout % args.rollouts_per_mapchange,
                    args.success_rate_target,
                    safe_mean(ep_is_success_queue)
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
                envs = create_venv(VcmiEnv, args, writer, agent.state.map_swaps)  # noqa: E501
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                next_mask = torch.as_tensor(np.array(envs.unwrapped.call("action_masks"))).to(device)

            save_ts = maybe_save(save_ts, args, agent, optimizer, out_dir)

    finally:
        maybe_save(0, args, agent, optimizer, out_dir)
        envs.close()
        writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
