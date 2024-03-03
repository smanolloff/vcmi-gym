# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import random
import glob
import threading
from dataclasses import dataclass
from typing import Optional
from collections import deque
import concurrent.futures

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from vcmi_gym import VcmiEnv


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

# python clean-rl/mppo.py \
#     --exp-name cleanrl-test \
#     --track \
#     --wandb-id test-(date +%s) \
#     --wandb-group cleanrl \
#     --learning-rate 0.00126  \
#     --num-envs 1 \
#     --num-steps 128 \
#     --num-minibatches 4 \
#     --update-epochs 4 \
#     --clip-coef 0.4 \
#     --clip-vloss \
#     --ent-coef 0.007 \
#     --vf-coef 0.6 \
#     --max-grad-norm 2.5 \
#     --gamma 0.8425 \
#     --gae-lambda 0.8

@dataclass
class EnvArgs:
  max_steps: int = 500
  reward_dmg_factor: int = 5
  vcmi_loglevel_global: str = "error"
  vcmi_loglevel_ai: str = "error"
  vcmienv_loglevel: str = "WARN"
  consecutive_error_reward_factor: int = -1
  sparse_info: bool = True
  step_reward_mult: int = 1
  term_reward_mult: int = 0
  reward_clip_mod: Optional[int] = None


@dataclass
class Args:
    run_id: str
    group_id: str
    agent_load_file: Optional[str] = None
    map_cycle: int = 0
    rollouts_total: int = 10000
    rollouts_per_mapchange: int = 20
    rollouts_per_log: int = 1
    opponent_load_file: Optional[str] = None
    success_rate_target: Optional[float] = None
    mapmask: str = "ai/generated/B*.vmap"
    randomize_maps: bool = False
    save_every: 7200  # seconds
    max_saves: 3
    out_dir: str = "data/default"

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

    wandb: bool = True
    seed: int = 42

    env: EnvArgs = EnvArgs()


def make_vec_env_parallel(j, env_creator, n_envs):
    def initenv():
        env = env_creator()
        return env

    if n_envs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=j) as executor:
            futures = [executor.submit(initenv) for _ in range(n_envs)]
            results = [future.result() for future in futures]

        funcs = [lambda x=x: x for x in results]
        vec_env = DummyVecEnv(funcs)
    else:
        vec_env = DummyVecEnv([initenv])

    vec_env.seed()
    return vec_env


def create_venv(n_envs, env_cls, env_kwargs, mapmask, randomize, run_id, writer, map_cycle):
    mappath = "/Users/simo/Library/Application Support/vcmi/Maps"
    all_maps = glob.glob("%s/%s" % (mappath, mapmask))
    all_maps = [m.replace("%s/" % mappath, "") for m in all_maps]
    all_maps.sort()

    if n_envs == 1:
        n_maps = 1
    else:
        assert n_envs % 2 == 0
        assert n_envs <= len(all_maps) * 2
        n_maps = n_envs // 2

    if randomize:
        maps = random.sample(all_maps, n_maps)
    else:
        i = (n_maps * map_cycle) % len(all_maps)
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
            assert state["n"] < n_envs
            role, mapname = pairs[state["n"]]
            # logfile = f"/tmp/{run_id}-env{state['n']}-actions.log"
            logfile = None

            env_kwargs2 = dict(
                env_kwargs,
                mapname=mapname,
                attacker="MMAI_MODEL" if args.opponent_load_file else "StupidAI",
                defender="MMAI_MODEL" if args.opponent_load_file else "StupidAI",
                attacker_model=args.opponent_load_file,
                defender_model=args.opponent_load_file,
                actions_log_file=logfile
            )

            env_kwargs2[role] = "MMAI_USER"
            print("Env kwargs (env.%d): %s" % (state["n"], env_kwargs2))
            state["n"] += 1

        return env_cls(**env_kwargs2)

    # I don't remember anymore, but there were some issues with more envs
    if n_envs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_envs, 8)) as executor:
            futures = [executor.submit(env_creator) for _ in range(n_envs)]
            results = [future.result() for future in futures]

        funcs = [lambda x=x: x for x in results]
        vec_env = gym.vector.SyncVectorEnv(funcs)
    else:
        vec_env = gym.vector.SyncVectorEnv([env_creator])

    vec_env = gym.wrappers.RecordEpisodeStatistics(vec_env)
    return vec_env


def maybe_save(t, agent, out_dir, save_every, max_saves):
    now = time.time()

    if t is None:
        return now

    if t + save_every > now:
        return t

    os.makedirs(out_dir, exist_ok=True)
    agent_file = os.path.join(out_dir, "agent-%d.pt" % now)
    torch.save(agent, agent_file)
    print("Saved agent to %s" % agent_file)

    # save file retention (keep latest N saves)
    files = sorted(
        glob.glob(os.path.join(out_dir, "agent-[0-9]*.zip")),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()),
        reverse=True
    )

    for file in files[max_saves:]:
        print("Deleting %s" % file)
        os.remove(file)

    return now


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def safe_mean(array_like) -> float:
    return np.nan if len(array_like) == 0 else float(np.mean(array_like))


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

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
        probs = CategoricalMasked(logits=logits, mask=mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


def main(**kwargs):
    args = Args(**kwargs)  # will blow up if any invalid kwargs are given
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

    if args.wandb:
        import wandb

        wandb.init(
            project="vcmi-gym",
            group=args.group_id,
            name=args.run_id,
            id=args.run_id,
            resume="never",
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
            allow_val_change=False,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
        )

    writer = SummaryWriter(args.out_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # args.torch_deterministic

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    iteration = args.iteration
    t = None

    try:
        envs = create_venv(args.num_envs, VcmiEnv, vars(args.env), args.mapmask, args.randomize_maps, args.run_id, writer, iteration)
        net_value_queue = deque(maxlen=envs.return_queue.maxlen)

        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs.single_observation_space, envs.single_action_space).to(device)
        if args.agent_load_file:
            print("Loading agent from %s" % args.agent_load_file)
            agent.load_state_dict(torch.load(args.agent_load_file), strict=True)

        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # XXX: the start=0 requirement is needed for SB3 compat
        assert envs.single_action_space.start == 0
        masks = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), dtype=torch.bool).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        next_mask = torch.as_tensor(np.array(envs.call("action_masks"))).to(device)

        start_rollout = args.iteration * args.rollouts_per_mapchange
        assert start_rollout < args.rollouts_total

        for rollout in range(start_rollout, args.rollouts_total):
            rollout_start_time = time.time()
            rollout_start_step = global_step

            for step in range(0, args.num_steps):
                global_step += args.num_envs
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
                next_mask = torch.as_tensor(np.array(envs.call("action_masks"))).to(device)

                # For a vectorized environments the output will be in the form of::
                #     >>> infos = {
                #     ...     "final_observation": "<array of length num-envs>",
                #     ...     "_final_observation": "<boolean array of length num-envs>",
                #     ...     "final_info": "<array of length num-envs>",
                #     ...     "_final_info": "<boolean array of length num-envs>",
                #     ...     "episode": {
                #     ...         "r": "<array of cumulative reward>",
                #     ...         "l": "<array of episode length>",
                #     ...         "t": "<array of elapsed time since beginning of episode>"
                #     ...     },
                #     ...     "_episode": "<boolean array of length num-envs>"
                #     ... }
                # 
                # My notes:
                #   "episode" and "_episode" is added by RecordEpisodeStatistics wrapper
                #   gym's vec env *automatically* collects episode returns and lengths in envs.return_queue and envs.length_queue
                #   (eg. [-1024.2, 333.6, ...] and [34, 36, 41, ...]) - each element is a full episode
                #
                # See https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/vector/sync_vector_env.py#L142-L157
                #     https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/vector/vector_env.py#L275-L300
                #     https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/wrappers/record_episode_statistics.py#L102-L124
                #
                for final_info, has_final_info in infos.get("final_info", []).zip(infos.get("_final_info", [])):
                    if has_final_info:
                        ep_net_value_queue.append(safe_mean(final_info["net_value"]))
                        ep_is_success_queue.append(safe_mean(final_info["is_success"]))

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
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_masks = masks.reshape((-1,) + (envs.single_action_space.n,))
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

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds], b_actions.long()[mb_inds])
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

            iteration += 1
            assert iteration == rollout // args.rollouts_per_mapchange

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("params/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("time/rollout_duration", time.time() - rollout_start_time)
            writer.add_scalar("time/steps_per_second", (global_step - rollout_start_step) / (time.time() - rollout_start_time))
            writer.add_scalar("rollout/ep_rew_mean", safe_mean(envs.return_queue))
            writer.add_scalar("rollout/ep_len_mean", safe_mean(envs.length_queue))
            writer.add_scalar("rollout/ep_value_mean", safe_mean(ep_net_value_queue))
            writer.add_scalar("rollout/ep_success_rate", safe_mean(ep_is_success_queue))
            writer.add_scalar("rollout/ep_count", envs.episode_count)
            writer.add_scalar("global/num_timesteps", global_step)
            writer.add_scalar("global/num_rollouts", rollout)
            writer.add_scalar("global/iterations", iteration)
            writer.add_scalar("global/progress", rollout / args.rollouts_total)

            print(f"global_step={global_step}, rollout/ep_rew_mean={safe_mean(envs.return_queue)}")

            if args.success_rate_target and safe_mean(ep_is_success_queue) >= args.success_rate_target:
                writer.flush()
                print("Early stopping after %d rollouts due to: success rate > %.2f (%.2f)" % (self.this_env_rollouts, self.success_rate_target, success_rate))

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
                envs.close()
                envs = create_venv(args.num_envs, VcmiEnv, vars(args.env), args.mapmask, args.randomize_maps, args.run_id, writer, iteration)
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                next_mask = torch.as_tensor(np.array(envs.call("action_masks"))).to(device)

        t = maybe_save(t, agent, out_dir, save_every=3600, max_saves=3)

    finally:
        envs.close()
        writer.close()
        agent_file = os.path.join(out_dir, "agent.pt")
        torch.save(agent, agent_file)
        print("Saved agent to %s" % agent_file)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(**vars(args))

# C:\simo> pscp.exe -P 2222 -i id_rsa-alextmp clean-rl/mppo.py simo@151.251.227.203:/Users/simo/Projects/vcmi-gym/cleanrl-mppo.py
 