import os
import sys
import random
import logging
import json
import string
import argparse
import threading
import contextlib
import gymnasium as gym
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F

from rl.world.util.structured_logger import StructuredLogger
from rl.world.util.persistence import load_checkpoint, save_checkpoint
from rl.world.util.wandb import setup_wandb
from rl.world.util.timer import Timer
from rl.world.util.misc import dig, timer_stats, safe_mean

from .. import common
# from ...world.i2a import I2A


if os.getenv("PYDEBUG", None) == "1":
    def excepthook(exc_type, exc_value, tb):
        import ipdb
        ipdb.post_mortem(tb)

    sys.excepthook = excepthook


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

    ep_rew_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_rew_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_net_value_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_net_value_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    ep_is_success_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_100: deque = field(default_factory=lambda: deque(maxlen=100))
    rollout_is_success_queue_1000: deque = field(default_factory=lambda: deque(maxlen=1000))

    def to_json(self):
        j = {}
        for k, v in asdict(self).items():
            j[k] = list(v) if isinstance(v, deque) else v
        return json.dumps(j, indent=4, sort_keys=False)

    def from_json(self, j):
        for k, v in json.loads(j).items():
            attr = getattr(self, k)
            self.k = deque(v, maxlen=attr.maxlen) if isinstance(attr, deque) else v


class TensorStorage:
    def __init__(self, venv, num_vsteps, device):
        s = num_vsteps
        e = venv.num_envs
        ospace = venv.single_observation_space
        aspace = venv.single_action_space

        self.obs = torch.zeros((s, e) + ospace.shape, device=device)
        self.logprobs = torch.zeros(s, e, device=device)
        self.actions = torch.zeros((s, e) + aspace.shape, device=device)
        self.masks = torch.zeros((s, e, aspace.n), dtype=torch.bool, device=device)

        self.rewards = torch.zeros((s, e), device=device)
        self.dones = torch.zeros((s, e), device=device)
        self.values = torch.zeros((s, e), device=device)
        self.next_obs = torch.as_tensor(venv.reset()[0], device=device)
        self.next_mask = torch.as_tensor(np.array(venv.call("action_mask")), device=device)
        self.next_done = torch.zeros(e, device=device)

        self.next_value = torch.zeros(e, device=device)  # needed for GAE
        self.advantages = torch.zeros(s, e, device=device)
        self.returns = torch.zeros(s, e, device=device)

        self.device = device


# XXX: DEBUG: TEST SIMPLE MODEL FREE ONLY
from rl.world.util.hexconv import HexConvResBlock
from rl.world.util.constants_v12 import STATE_SIZE_GLOBAL, STATE_SIZE_ONE_PLAYER, STATE_SIZE_ONE_HEX, N_ACTIONS
HEXES_OFFSET = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER

class SimpleModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.device = torch.device("cpu")

        self.z_size_other = 64
        self.z_size_hex = 16
        self.output_size = output_size

        self.encoder_other = nn.Sequential(
            nn.LazyLinear(self.z_size_other),
            nn.LeakyReLU()
            # => (B, Z_OTHER)
        )

        self.encoder_hexes = nn.Sequential(
            # => (B, 165*H)
            nn.Unflatten(dim=1, unflattened_size=[165, STATE_SIZE_ONE_HEX]),
            # => (B, 165, H)
            HexConvResBlock(channels=STATE_SIZE_ONE_HEX, depth=3),
            # => (B, 165, H)
            nn.LazyLinear(out_features=self.z_size_hex),
            nn.LeakyReLU(),
            # => (B, 165, Z_HEX)
            nn.Flatten(),
            # => (B, 165*Z_HEX)
        )

        self.encoder_merged = nn.Sequential(
            # => (B, Z_OTHER + 165*Z_HEX)
            nn.LazyLinear(out_features=self.output_size),
            nn.LeakyReLU()
            # => (B, OUTPUT_SIZE)
        )

        self.action_head = nn.LazyLinear(N_ACTIONS)
        self.value_head = nn.LazyLinear(1)

    def forward(self, obs, _mask, debug=False):
        other, hexes = torch.split(obs, [HEXES_OFFSET, 165*STATE_SIZE_ONE_HEX], dim=1)
        z_other = self.encoder_other(other)
        z_hexes = self.encoder_hexes(hexes)
        merged = torch.cat((z_other, z_hexes), dim=1)
        action_logits = self.action_head(merged)
        value = self.value_head(merged)
        return action_logits, value


@dataclass
class TrainStats:
    value_loss: float
    policy_loss: float
    entropy_loss: float
    distill_loss: float
    loss: float
    approx_kl: float
    clipfrac: float
    explained_var: float


@dataclass
class SampleStats:
    ep_rew_mean: float = 0.0
    ep_len_mean: float = 0.0
    ep_value_mean: float = 0.0
    ep_is_success_mean: float = 0.0
    num_episodes: int = 0
    num_transition_truncations: int = 0


def create_venv(env_kwargs, num_envs):
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper

    # AsyncVectorEnv creates a dummy_env() in the main process just to
    # extract metadata, which causes VCMI init pid error afterwards
    pid = os.getpid()
    dummy_env = SimpleNamespace(
        metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
        render_mode='ansi',
        action_space=VcmiEnv.ACTION_SPACE,
        observation_space=VcmiEnv.OBSERVATION_SPACE["observation"],
        close=lambda: None
    )

    def env_creator(i):
        if os.getpid() == pid:
            return dummy_env

        env = VcmiEnv(**env_kwargs)
        env = LegacyObservationSpaceWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    funcs = [partial(env_creator, i) for i in range(num_envs)]

    # SyncVectorEnv won't work when both train and eval env are started in main process
    # if num_envs > 1:
    #     vec_env = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    # else:
    #     vec_env = gym.vector.SyncVectorEnv(funcs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    vec_env = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    vec_env.reset()

    return vec_env


def collect_samples(logger, model, venv, num_vsteps, storage):
    assert not torch.is_grad_enabled()

    stats = SampleStats()

    device = storage.device

    # model_ic = model.imagination_aggregator.rollout_encoder.imagination_core
    # truncs_start = model_ic.num_truncations

    assert num_vsteps == storage.obs.size(0)
    num_envs = storage.obs.size(1)
    assert num_envs == venv.num_envs

    for vstep in range(0, num_vsteps):
        logger.debug("(train) vstep: %d" % vstep)
        storage.obs[vstep] = storage.next_obs
        storage.dones[vstep] = storage.next_done
        storage.masks[vstep] = storage.next_mask

        action_logits, value = model(storage.next_obs, storage.next_mask, debug=False)
        dist = common.CategoricalMasked(logits=action_logits, mask=storage.next_mask)
        action = dist.sample()

        storage.values[vstep] = value.flatten()
        storage.actions[vstep] = action
        storage.logprobs[vstep] = dist.log_prob(action)

        next_obs, reward, terminations, truncations, infos = venv.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        storage.rewards[vstep] = torch.as_tensor(reward, device=device).flatten()
        storage.next_obs = torch.as_tensor(next_obs, device=device)
        storage.next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)
        storage.next_mask = torch.as_tensor(np.array(venv.call("action_mask")), device=device)

        # See notes/gym_vector.txt
        if "_final_info" in infos:
            done_ids = np.flatnonzero(infos["_final_info"])
            final_infos = infos["final_info"]
            stats.ep_rew_mean += sum(final_infos["episode"]["r"][done_ids])
            stats.ep_len_mean += sum(final_infos["episode"]["l"][done_ids])
            stats.ep_value_mean += sum(final_infos["net_value"][done_ids])
            stats.ep_is_success_mean += sum(final_infos["is_success"][done_ids])
            stats.num_episodes += len(done_ids)

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

    # XXX: DEBUG: TEST SIMPLE MODEL FREE ONLY
    # truncs = model_ic.num_truncations
    # stats.num_transition_truncations = truncs - truncs_start

    # bootstrap value if not done
    _, next_value = model(storage.next_obs, storage.next_mask)
    storage.next_value = next_value.reshape(1, -1)

    return stats


def eval_model(logger, model, venv, num_vsteps):
    assert not torch.is_grad_enabled()

    stats = SampleStats()
    # model_ic = model.imagination_aggregator.rollout_encoder.imagination_core
    # truncs_start = model_ic.num_truncations

    t = lambda x: torch.as_tensor(x, device=model.device)

    obs, _ = venv.reset()

    for vstep in range(0, num_vsteps):
        logger.debug("(eval) vstep: %d" % vstep)
        obs = t(obs)
        mask = t(np.array(venv.call("action_mask")))

        action_logits, value = model(obs, mask)
        dist = common.CategoricalMasked(logits=action_logits, mask=mask)
        action = dist.sample()

        obs, rew, term, trunc, info = venv.step(action.cpu().numpy())

        # See notes/gym_vector.txt
        if "_final_info" in info:
            done_ids = np.flatnonzero(info["_final_info"])
            final_info = info["final_info"]
            stats.ep_rew_mean += sum(final_info["episode"]["r"][done_ids])
            stats.ep_len_mean += sum(final_info["episode"]["l"][done_ids])
            stats.ep_value_mean += sum(final_info["net_value"][done_ids])
            stats.ep_is_success_mean += sum(final_info["is_success"][done_ids])
            stats.num_episodes += len(done_ids)

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes

    # XXX: DEBUG: TEST SIMPLE MODEL FREE ONLY
    # truncs = model_ic.num_truncations
    # stats.num_transition_truncations = truncs - truncs_start

    return stats


def train_model(logger, model, optimizer, autocast_ctx, scaler, storage, train_config):
    assert torch.is_grad_enabled()

    # XXX: this always returns False for CPU https://github.com/pytorch/pytorch/issues/110966
    # assert torch.is_autocast_enabled()

    # compute advantages
    with torch.no_grad():
        lastgaelam = 0
        num_vsteps = train_config["num_vsteps"]
        assert storage.obs.size(0) == num_vsteps

        for t in reversed(range(num_vsteps)):
            if t == num_vsteps - 1:
                nextnonterminal = 1.0 - storage.next_done
                nextvalues = storage.next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = storage.rewards[t] + train_config["gamma"] * nextvalues * nextnonterminal - storage.values[t]
            storage.advantages[t] = lastgaelam = delta + train_config["gamma"] * train_config["gae_lambda"] * nextnonterminal * lastgaelam

        storage.returns[:] = storage.advantages + storage.values

    # flatten the batch (num_envs, env_samples, *) => (num_steps, *)
    b_obs = storage.obs.flatten(end_dim=1)
    b_logprobs = storage.logprobs.flatten(end_dim=1)
    b_actions = storage.actions.flatten(end_dim=1).long()
    b_masks = storage.masks.flatten(end_dim=1)
    b_advantages = storage.advantages.flatten(end_dim=1)
    b_returns = storage.returns.flatten(end_dim=1)
    b_values = storage.values.flatten(end_dim=1)

    batch_size = b_obs.size(0)
    minibatch_size = int(batch_size // train_config["num_minibatches"])
    b_inds = np.arange(batch_size)
    clipfracs = []

    policy_losses = torch.zeros(train_config["num_minibatches"])
    entropy_losses = torch.zeros(train_config["num_minibatches"])
    value_losses = torch.zeros(train_config["num_minibatches"])
    distill_losses = torch.zeros(train_config["num_minibatches"])

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train) epoch: %d" % epoch)
        np.random.shuffle(b_inds)
        for i, start in enumerate(range(0, batch_size, minibatch_size)):
            logger.debug("(train) minibatch: %d" % i)
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_masks = b_masks[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]

            newlogits, newvalue = model(mb_obs, mb_masks)
            newdist = common.CategoricalMasked(logits=newlogits, mask=mb_masks)

            logratio = newdist.log_prob(mb_actions) - mb_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if train_config["norm_adv"]:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_config["clip_coef"], 1 + train_config["clip_coef"])
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if train_config["clip_vloss"]:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -train_config["clip_coef"],
                    train_config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()
            else:
                # XXX: SIMO: SB3 does not multiply by 0.5 here
                #            (ie. SB3's vf_coef is essentially x2)
                value_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = newdist.entropy().mean()

            # """
            # (I2A paper, Supplementary matierial, Section A)
            # [...] we do not backpropagate gradients of ldist wrt. to the
            # parameters of the rollout policy through the behavioral
            # policy [...]
            # // => detach `logit`
            # """
            # rp_logits = model.imagination_aggregator.rollout_encoder.rollout_policy(b_obs[mb_inds])
            # rp_dist = common.CategoricalMasked(logits=rp_logits, mask=mb_masks)
            # rp_log_probs = rp_dist.logits.log_softmax(dim=-1)
            # teacher_log_probs = newdist.logits.log_softmax(dim=-1)
            # teacher_probs = teacher_log_probs.exp().detach()
            # distill_loss = F.kl_div(rp_log_probs, teacher_probs, reduction='batchmean')
            # XXX: DEBUG: TEST SIMPLE MODEL FREE ONLY
            distill_loss = torch.zeros_like(entropy_loss)

            policy_losses[i] = policy_loss.detach()
            entropy_losses[i] = entropy_loss.detach()
            value_losses[i] = value_loss.detach()
            distill_losses[i] = distill_loss.detach()

            loss = (
                policy_loss
                - (entropy_loss * train_config["ent_coef"])
                + (value_loss * train_config["vf_coef"])
                + (distill_loss * train_config["distill_lambda"])
            )

            with autocast_ctx(False):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

        if train_config["target_kl"] is not None and approx_kl > train_config["target_kl"]:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return TrainStats(
        value_loss=value_losses.mean().item(),
        policy_loss=policy_losses.mean().item(),
        entropy_loss=entropy_losses.mean().item(),
        distill_loss=distill_losses.mean().item(),
        loss=loss.item(),
        approx_kl=approx_kl.item(),
        clipfrac=float(np.mean(clipfracs)),
        explained_var=float(explained_var),
    )


def prepare_wandb_log(
    model,
    optimizer,
    state,
    train_stats,
    train_sample_stats,
    eval_sample_stats,
):
    wlog = {}

    if eval_sample_stats.num_episodes > 0:
        wlog.update({
            "eval/ep_rew_mean": eval_sample_stats.ep_rew_mean,
            "eval/ep_value_mean": eval_sample_stats.ep_value_mean,
            "eval/ep_len_mean": eval_sample_stats.ep_len_mean,
            "eval/ep_success_rate": eval_sample_stats.ep_is_success_mean,
            "eval/ep_count": eval_sample_stats.num_episodes,
            "eval/transition_truncations": eval_sample_stats.num_transition_truncations,
        })

    if train_sample_stats.num_episodes > 0:
        state.rollout_rew_queue_100.append(train_sample_stats.ep_rew_mean)
        state.rollout_rew_queue_1000.append(train_sample_stats.ep_rew_mean)
        state.rollout_net_value_queue_100.append(train_sample_stats.ep_value_mean)
        state.rollout_net_value_queue_1000.append(train_sample_stats.ep_value_mean)
        state.rollout_is_success_queue_100.append(train_sample_stats.ep_is_success_mean)
        state.rollout_is_success_queue_1000.append(train_sample_stats.ep_is_success_mean)
        wlog.update({
            "train/ep_rew_mean": train_sample_stats.ep_rew_mean,
            "train/ep_value_mean": train_sample_stats.ep_value_mean,
            "train/ep_len_mean": train_sample_stats.ep_len_mean,
            "train/ep_success_rate": train_sample_stats.ep_is_success_mean,
            "train/ep_count": train_sample_stats.num_episodes,
            "train/transition_truncations": train_sample_stats.num_transition_truncations,
        })

    wlog.update({
        "train/learning_rate": optimizer.param_groups[0]["lr"],
        "train/value_loss": train_stats.value_loss,
        "train/policy_loss": train_stats.policy_loss,
        "train/entropy_loss": train_stats.entropy_loss,
        "train/distill_loss": train_stats.distill_loss,
        "train/approx_kl": train_stats.approx_kl,
        "train/clipfrac": train_stats.clipfrac,
        "train/explained_var": train_stats.explained_var,
        "train/ep_value_mean_100": safe_mean(state.rollout_net_value_queue_100),
        "train/ep_value_mean_1000": safe_mean(state.rollout_net_value_queue_1000),
        "train/ep_rew_mean_100": safe_mean(state.rollout_rew_queue_100),
        "train/ep_rew_mean_1000": safe_mean(state.rollout_rew_queue_1000),
        "train/ep_success_rate_100": safe_mean(state.rollout_is_success_queue_100),
        "train/ep_success_rate_1000": safe_mean(state.rollout_is_success_queue_1000),
        "global/global_num_timesteps": state.global_timestep,
        "global/global_num_seconds": state.global_second,
        "global/num_rollouts": state.current_rollout,
        "global/num_timesteps": state.current_timestep,
        "global/num_seconds": state.current_second,
        "global/num_episode": state.current_episode,
    })

    return wlog


def main(config, resume_config, loglevel, dry_run, no_wandb):
    if resume_config:
        with open(resume_config, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)

        run_id = config["run"]["id"]
        config["run"]["resumed_config"] = resume_config
    else:
        run_id = ''.join(random.choices(string.ascii_lowercase, k=8))
        config["run"] = dict(
            id=run_id,
            name=config["name_template"].format(id=run_id, datetime=datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            out_dir=os.path.abspath(config["out_dir_template"].format(id=run_id)),
            resumed_config=None,
        )

    os.makedirs(config["run"]["out_dir"], exist_ok=True)
    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        print(f"Saving new config to: {f.name}")
        json.dump(config, f, indent=4)

    # assert config["checkpoint"]["interval_s"] > config["eval"]["interval_s"]
    assert config["checkpoint"]["permanent_interval_s"] > config["eval"]["interval_s"]
    assert config["train"]["env"]["kwargs"]["user_timeout"] >= 2 * config["eval"]["interval_s"]

    checkpoint_config = dig(config, "checkpoint")
    train_config = dig(config, "train")
    eval_config = dig(config, "eval")

    logger = StructuredLogger(level=getattr(logging, loglevel), filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"), context=dict(run_id=run_id))
    logger.info(dict(config=config))

    learning_rate = config["train"]["learning_rate"]

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    train_venv = create_venv(train_config["env"]["kwargs"], train_config["env"]["num_envs"])
    logger.debug("Initialized %d train envs" % train_venv.num_envs)
    eval_venv = create_venv(eval_config["env"]["kwargs"], eval_config["env"]["num_envs"])
    logger.debug("Initialized %d eval envs" % eval_venv.num_envs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = train_config["env"]["num_envs"]
    num_steps = train_config["num_vsteps"] * num_envs
    batch_size = int(num_steps)
    assert batch_size % train_config["num_minibatches"] == 0, f"{batch_size} % {train_config['num_minibatches']} == 0"
    storage = TensorStorage(train_venv, train_config["num_vsteps"], device)
    state = State()

    model_config = dig(config, "model")
    # model = I2A(
    #     i2a_fc_units=model_config["i2a_fc_units"],
    #     num_trajectories=model_config["num_trajectories"],
    #     rollout_dim=model_config["rollout_dim"],
    #     rollout_policy_fc_units=model_config["rollout_policy_fc_units"],
    #     horizon=model_config["horizon"],
    #     obs_processor_output_size=model_config["obs_processor_output_size"],
    #     side=(train_config["env"]["kwargs"]["role"] == "defender"),
    #     reward_step_fixed=train_config["env"]["kwargs"]["reward_step_fixed"],
    #     reward_dmg_mult=train_config["env"]["kwargs"]["reward_dmg_mult"],
    #     reward_term_mult=train_config["env"]["kwargs"]["reward_term_mult"],
    #     max_transitions=model_config["max_transitions"],
    #     transition_model_file=model_config["transition_model_file"],
    #     action_prediction_model_file=model_config["action_prediction_model_file"],
    #     reward_prediction_model_file=model_config["reward_prediction_model_file"],
    #     device=device,
    #     debug_render=False,
    # )
    model = SimpleModel(1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.param_groups[0].setdefault("initial_lr", learning_rate)

    if train_config["torch_autocast"]:
        autocast_ctx = lambda enabled: torch.autocast(device.type, enabled=enabled)
        scaler = torch.GradScaler(device.type, enabled=True)
    else:
        # No-op autocast and scaler
        autocast_ctx = contextlib.nullcontext
        scaler = torch.GradScaler(device.type, enabled=False)

    logger.debug("Initialized I2A model and optimizer (autocast=%s)" % train_config["torch_autocast"])

    """
    XXX: debug code block for torch autocast+no_grad transformer matmul error
    with autocast_ctx(enabled=True):
        # TEST
        from ...world.t10n import t10n
        from ...world.p10n import p10n
        from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
        env = VcmiEnv()
        obs_ = env.reset()[0]
        test_obs = torch.as_tensor(obs_["observation"]).unsqueeze(0)
        test_mask = torch.as_tensor(obs_["action_mask"]).unsqueeze(0)

        print("================ TESTS START ============")
        test_action = torch.tensor([1]).long()

        # fail:
        # res = model(test_obs, test_mask)

        # fail:
        res = model.imagination_aggregator(test_obs, test_mask, t10n.Reconstruction.GREEDY, p10n.Prediction.GREEDY)

        # res = model.imagination_aggregator.rollout_encoder(test_obs, test_action, t10n.Reconstruction.GREEDY, p10n.Prediction.GREEDY)

        # ok:
        # res = model.imagination_aggregator.rollout_encoder.imagination_core.transition_model(test_obs, test_action)

        # test_transformer_input = torch.randn([1, 165, 512])
        # model.imagination_aggregator.rollout_encoder.imagination_core.transition_model.transformer_hex(test_transformer_input)

        import ipdb; ipdb.set_trace()  # noqa
        pass
    """

    if resume_config:
        load_checkpoint(
            logger=logger,
            dry_run=dry_run,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            out_dir=config["run"]["out_dir"],
            run_id=run_id,
            optimize_local_storage=checkpoint_config["optimize_local_storage"],
            s3_config=checkpoint_config["s3"],
            device=device,
            state=state,
        )

        # lr is lost after loading weights
        optimizer.param_groups[0]["lr"] = learning_rate

        state.resumes += 1
        logger.info("Resumes: %d" % state.resumes)

    if no_wandb:
        from unittest.mock import Mock
        wandb = Mock()
    else:
        wandb = setup_wandb(config, model, __file__)

    accumulated_logs = {}

    def accumulate_logs(data):
        for k, v in data.items():
            if k not in accumulated_logs:
                accumulated_logs[k] = [v]
            else:
                accumulated_logs[k].append(v)

    def aggregate_logs():
        agg_data = {k: safe_mean(v) for k, v in accumulated_logs.items()}
        accumulated_logs.clear()
        return agg_data

    wandb.log({
        "global/resumes": state.resumes,
        "train_config/num_vsteps": train_config["num_vsteps"],
        "train_config/num_minibatches": train_config["num_minibatches"],
        "train_config/update_epochs": train_config["update_epochs"],
        "train_config/gamma": train_config["gamma"],
        "train_config/gae_lambda": train_config["gae_lambda"],
        "train_config/ent_coef": train_config["ent_coef"],
        "train_config/clip_coef": train_config["clip_coef"],
        "train_config/learning_rate": train_config["learning_rate"],
        "train_config/norm_adv": int(train_config["norm_adv"]),
        "train_config/clip_vloss": int(train_config["clip_vloss"]),
        "train_config/max_grad_norm": train_config["max_grad_norm"],
        "train_config/distill_lambda": train_config["distill_lambda"],
    }, commit=False)

    # during training, we simply check if the event is set and optionally skip the upload
    # Non-bloking, but uploads may be skipped (checkpoint uploads)
    uploading_event = threading.Event()

    timers = {
        "all": Timer(),
        "sample": Timer(),
        "train": Timer(),
        "eval": Timer(),
    }

    timers["all"].start()
    eval_net_value_best = None

    checkpoint_timer = Timer()
    checkpoint_timer.start()
    permanent_checkpoint_timer = Timer()
    permanent_checkpoint_timer.start()
    wandb_log_commit_timer = Timer()
    wandb_log_commit_timer.start()
    wandb_log_commit_timer._started_at = 0  # force first trigger
    eval_timer = Timer()
    eval_timer.start()
    if config["eval"]["at_script_start"]:
        eval_timer._started_at = 0  # force first trigger

    lr_schedule_timer = Timer()
    lr_schedule_timer.start()

    assert train_config["lr_scheduler_min_value"] < train_config["learning_rate"]

    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=train_config["lr_scheduler_step_mult"],
        last_epoch=max(state.global_second//train_config["lr_scheduler_interval_s"] - 1, -1)
    )

    clamp_lr = lambda: max(optimizer.param_groups[0]["lr"], train_config["lr_scheduler_min_value"])
    optimizer.param_groups[0]["lr"] = clamp_lr()

    try:
        while True:
            [v.reset(start=(k == "all")) for k, v in timers.items()]

            if lr_schedule_timer.peek() > train_config["lr_scheduler_interval_s"]:
                lr_schedule_timer.reset(start=True)
                lr_schedule.step()
                optimizer.param_groups[0]["lr"] = clamp_lr()
                logger.info("New learning_rate: %s" % optimizer.param_groups[0]['lr'])

            # Evaluate first (for a baseline when resuming with modified params)
            if eval_timer.peek() > eval_config["interval_s"]:
                logger.info("Time for eval")
                eval_timer.reset(start=True)

                with timers["eval"]:
                    model.eval()
                    with torch.no_grad():
                        eval_sample_stats = eval_model(
                            logger=logger,
                            model=model,
                            venv=eval_venv,
                            num_vsteps=eval_config["num_vsteps"],
                        )
            else:
                eval_sample_stats = SampleStats()

            model.eval()
            with timers["sample"], torch.no_grad(), autocast_ctx(True):
                train_sample_stats = collect_samples(
                    logger=logger,
                    model=model,
                    venv=train_venv,
                    num_vsteps=train_config["num_vsteps"],
                    storage=storage,
                )

            state.current_vstep += train_config["num_vsteps"]
            state.current_timestep += train_config["num_vsteps"] * num_envs
            state.global_timestep += train_config["num_vsteps"] * num_envs
            state.current_episode += train_sample_stats.num_episodes
            state.global_episode += train_sample_stats.num_episodes
            state.current_second = int(timers["sample"].peek())
            state.global_second += int(timers["all"].peek())

            model.train()
            with timers["train"], autocast_ctx(True):
                train_stats = train_model(
                    logger=logger,
                    model=model,
                    optimizer=optimizer,
                    autocast_ctx=autocast_ctx,
                    scaler=scaler,
                    storage=storage,
                    train_config=train_config,
                )

            # Checkpoint only if we have eval stats
            if checkpoint_timer.peek() > config["checkpoint"]["interval_s"] and eval_sample_stats.num_episodes > 0:
                logger.info("Time for a checkpoint")
                checkpoint_timer.reset(start=True)
                eval_net_value = eval_sample_stats.ep_value_mean

                if eval_net_value_best is None:
                    # Initial baseline for resumed configs
                    eval_net_value_best = eval_net_value
                    logger.info("No baseline for checkpoint yet (eval_net_value=%f, eval_net_value_best=None), setting it now" % eval_net_value)
                elif eval_net_value < eval_net_value_best:
                    logger.info("Bad checkpoint (eval_net_value=%f, eval_net_value_best=%f), will skip it" % (eval_net_value, eval_net_value_best))
                else:
                    logger.info("Good checkpoint (eval_net_value=%f, eval_net_value_best=%f), will save it" % (eval_net_value, eval_net_value_best))
                    eval_net_value_best = eval_net_value
                    thread = threading.Thread(target=save_checkpoint, kwargs=dict(
                        logger=logger,
                        dry_run=dry_run,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        state=state,
                        out_dir=config["run"]["out_dir"],
                        run_id=run_id,
                        optimize_local_storage=checkpoint_config["optimize_local_storage"],
                        s3_config=checkpoint_config["s3"],
                        uploading_event=uploading_event
                    ))
                    thread.start()

            if permanent_checkpoint_timer.peek() > config["checkpoint"]["permanent_interval_s"]:
                permanent_checkpoint_timer.reset(start=True)
                logger.info("Time for a permanent checkpoint")
                thread = threading.Thread(target=save_checkpoint, kwargs=dict(
                    logger=logger,
                    dry_run=dry_run,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    state=state,
                    out_dir=config["run"]["out_dir"],
                    run_id=run_id,
                    optimize_local_storage=checkpoint_config["optimize_local_storage"],
                    s3_config=checkpoint_config["s3"],
                    uploading_event=threading.Event(),  # never skip this upload
                    permanent=True,
                    config=config,
                ))
                thread.start()

            wlog = prepare_wandb_log(
                model=model,
                optimizer=optimizer,
                state=state,
                train_stats=train_stats,
                train_sample_stats=train_sample_stats,
                eval_sample_stats=eval_sample_stats,
            )

            accumulate_logs(wlog)

            if wandb_log_commit_timer.peek() > config["wandb_log_interval_s"]:
                # logger.info("Time for wandb log")
                wandb_log_commit_timer.reset(start=True)
                wlog.update(aggregate_logs())
                wlog.update(timer_stats(timers))
                wlog["train/learning_rate"] = optimizer.param_groups[0]['lr']
                wandb.log(wlog, commit=True)

            logger.info(wlog)
            state.current_rollout += 1

        ret_rew = safe_mean(list(state.rollout_rew_queue_1000)[-min(300, state.current_rollout):])
        ret_value = safe_mean(list(state.rollout_net_value_queue_1000)[-min(300, state.current_rollout):])

        return ret_rew, ret_value
    finally:
        save_checkpoint(
            logger=logger,
            dry_run=dry_run,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            out_dir=config["run"]["out_dir"],
            run_id=run_id,
            optimize_local_storage=checkpoint_config["optimize_local_storage"],
            s3_config=None,
            uploading_event=threading.Event(),  # never skip this upload
            permanent=True,
            config=config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
    parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument("--loglevel", metavar="LOGLEVEL", default="INFO", help="DEBUG | INFO | WARN | ERROR")
    args = parser.parse_args()

    if args.dry_run:
        args.no_wandb = True

    from .config import config

    main(
        config=config,
        resume_config=args.f,
        loglevel=args.loglevel,
        dry_run=args.dry_run,
        no_wandb=args.no_wandb,
    )
