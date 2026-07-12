import os
import re
import sys
import time
import random
import logging
import json
import string
import argparse
import threading
import contextlib
import importlib
import math
import traceback
import copy

from dataclasses import dataclass, field, asdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import datetime as dt
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as gnn_softmax
from torch_scatter import scatter_sum

from .util.structured_logger import StructuredLogger
from .util.persistence import load_checkpoint, save_checkpoint, download_latest_model
from .util.wandb import setup_wandb
from .util.timer import Timer
from .util.misc import dig, safe_mean, timer_stats

from .dual_vec_env import DualVecEnv, AbstractModelLoader, VcmiEnv
from .gnn_model import GNNModel, to_hdata_list, add_action_active_local_ids

if os.getenv("PYDEBUG", None) == "1":
    def excepthook(exc_type, exc_value, tb):
        import ipdb
        print("\n".join(traceback.format_exception(exc_value)))
        ipdb.post_mortem(tb)

    sys.excepthook = excepthook


@dataclass
class State:
    seed: int = -1
    resumes: int = 0
    global_timestep: int = 0
    current_timestep: int = 0
    current_vstep: int = 0
    global_rollout: int = 0
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

    volatile_checkpoint_counter: int = 0

    permanent_checkpoint_timer_elapsed: float = 0.0
    volatile_checkpoint_timer_elapsed: float = 0.0
    eval_timer_elapsed: float = 0.0

    def to_json(self):
        j = {}
        for k, v in asdict(self).items():
            j[k] = list(v) if isinstance(v, deque) else v
        return json.dumps(j, indent=4, sort_keys=False)

    def from_json(self, j):
        for k, v in json.loads(j).items():
            attr = getattr(self, k)
            v = deque(v, maxlen=attr.maxlen) if isinstance(attr, deque) else v
            setattr(self, k, v)


class Storage:
    def __init__(self, venv, num_vsteps, device):
        v = venv.num_envs
        self.rollout_buffer = []  # contains Batch() objects
        self.v_next_hdata_list = to_hdata_list(
            venv.call("graph_obs"),
            torch.zeros(v, device=device),
        )

        # Needed for the GAE computation (to prevent spaghetti)
        # and for explained_var computation
        self.bv_dones = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_values = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_rewards = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_advantages = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_returns = torch.zeros((num_vsteps, venv.num_envs), device=device)


@dataclass
class TrainStats:
    value_loss: float
    policy_loss: float
    entropy_loss: float
    distill_loss: float
    approx_kl: float
    clipfrac: float
    explained_var: float


@dataclass
class SampleStats:
    ep_rew_mean: float = 0.0
    ep_len_mean: float = 0.0
    ep_value_mean: float = 0.0
    ep_rounds_mean: float = 0.0
    ep_is_success_mean: float = 0.0
    ep_rew_step_fixed_mean: float = 0.0
    ep_rew_dmg_mult_mean: float = 0.0
    ep_rew_term_mult_mean: float = 0.0
    ep_rew_relval_mult_mean: float = 0.0
    ep_rew_prog_mean: float = 0.0
    num_episodes: int = 0
    num_truncations: int = 0


# Aggregated version of SampleStats with a handle
# to the individual SampleStats variants.
@dataclass
class MultiStats(SampleStats):
    variants: dict = field(default_factory=dict)

    def add(self, name, stats):
        self.variants[name] = stats

        if stats.num_episodes == 0:
            print("WARNING: adding SampleStats with num_episodes=0")

        # Don't let "empty" samples influence the mean values EXCEPT for num_episodes
        self.num_episodes = safe_mean([v.num_episodes for v in self.variants.values()])
        self.num_truncations = safe_mean([v.num_truncations for v in self.variants.values()])
        self.ep_rew_mean = safe_mean([v.ep_rew_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_len_mean = safe_mean([v.ep_len_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_value_mean = safe_mean([v.ep_value_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_rounds_mean = safe_mean([v.ep_rounds_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_is_success_mean = safe_mean([v.ep_is_success_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_rew_step_fixed_mean = safe_mean([v.ep_rew_step_fixed_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_rew_dmg_mult_mean = safe_mean([v.ep_rew_dmg_mult_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_rew_term_mult_mean = safe_mean([v.ep_rew_term_mult_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_rew_relval_mult_mean = safe_mean([v.ep_rew_relval_mult_mean for v in self.variants.values() if v.num_episodes > 0])
        self.ep_rew_prog_mean = safe_mean([v.ep_rew_prog_mean for v in self.variants.values() if v.num_episodes > 0])


class DNAModel(nn.Module):
    def __init__(self, node_types, edge_types, config, device):
        super().__init__()
        self.model_policy = GNNModel(node_types, edge_types, config)
        self.model_value = GNNModel(node_types, edge_types, config)
        self.device = device
        self.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelLoader(AbstractModelLoader):
    def __init__(self, device_type, role, loglevel="INFO"):
        assert role in ["attacker", "defender"]
        self.device_type = device_type
        self.role = role
        self.logger = StructuredLogger(level=getattr(logging, loglevel), context=dict(name="model_loader"))
        self.model = None

    def configure(self, config_file):
        assert not self.model, "Cannot call .configure() after .load()"
        self.config_file = config_file
        self.logger.debug(f"Loading model config from file: {config_file}")
        with open(config_file, "r") as f:
            self.config = json.load(f)

        loaded_role = self.config["train"]["env"]["kwargs"]["role"]
        assert loaded_role == self.role, f"{loaded_role} == {self.role}"

    # This function will be called from within another process
    # (referenced non-local objects must be serializable for IPC).
    def load(self, weights_file):
        self.logger.info(f"Loading model weights from {weights_file}")
        weights = torch.load(weights_file, weights_only=True, map_location=self.device_type)

        if not self.model:
            self.model = DNAModel(
                node_types=VcmiEnv.node_types(),
                edge_types=VcmiEnv.filtered_edge_types(self.config["train"]["env"]["kwargs"]["ignored_edges"]),
                config=self.config["model"],
                device=torch.device(self.device_type)
            ).eval()

        self.model.load_state_dict(weights, strict=True)

    def get_model(self):
        return self.model


@dataclass
class ModelLoaderInfo():
    # model_loader => (run_id, reload_at, run_id, weights_file)
    model_loader: ModelLoader
    model_ts: dt.datetime
    loaded_at: dt.datetime
    reload_interval_s: int
    run_id: str
    config_file: str
    weights_file: str


def collect_samples(logger, model, venv, num_vsteps, storage):
    assert not torch.is_inference_mode_enabled()  # causes issues during training
    assert not torch.is_grad_enabled()

    stats = SampleStats()

    storage.rollout_buffer.clear()

    for vstep in range(num_vsteps):
        logger.debug("(train) vstep: %d" % vstep)
        v_hdata_list = storage.v_next_hdata_list
        v_hdata_batch = Batch.from_data_list(v_hdata_list).to(model.device)
        add_action_active_local_ids(v_hdata_batch)

        v_action, v_logprob, v_entropy = model.model_policy.forward_policy(v_hdata_batch)
        v_value = model.model_value.forward_value(v_hdata_batch).flatten()
        _, v_rew, v_term, v_trunc, v_info = venv.step(v_action.cpu().numpy())
        v_rew = torch.as_tensor(v_rew)

        for i, hdata in enumerate(v_hdata_list):
            hdata.action = v_action[i].detach().cpu()
            hdata.logprob = v_logprob[i].detach().cpu()
            hdata.value = v_value[i].detach().cpu()
            hdata.reward = v_rew[i]

            storage.bv_dones[vstep, i] = hdata.done
            storage.bv_values[vstep, i] = hdata.value
            storage.bv_rewards[vstep, i] = hdata.reward

        storage.v_next_hdata_list = to_hdata_list(
            venv.call("graph_obs"),
            torch.as_tensor(np.logical_or(v_term, v_trunc)),
        )

        storage.rollout_buffer.extend(v_hdata_list)

        # See notes/gym_vector.txt
        if "_final_info" in v_info:
            v_done_id = np.flatnonzero(v_info["_final_info"])
            v_final_info = v_info["final_info"]
            stats.ep_rew_mean += sum(v_final_info["episode"]["r"][v_done_id])
            stats.ep_len_mean += sum(v_final_info["episode"]["l"][v_done_id])
            stats.ep_value_mean += sum(v_final_info["net_value"][v_done_id])
            stats.ep_rounds_mean += sum(v_final_info["round"][v_done_id])
            stats.ep_is_success_mean += sum(v_final_info["is_success"][v_done_id])
            assert len(v_done_id) == int(np.logical_or(v_term, v_trunc).sum())
            stats.num_episodes += len(v_done_id)
            stats.num_truncations += int(v_trunc.sum())
            stats.ep_rew_step_fixed_mean += sum(v_final_info["reward_step_fixed"][v_done_id])
            stats.ep_rew_prog_mean += sum(v_final_info["reward_prog"][v_done_id])
            stats.ep_rew_dmg_mult_mean += sum(v_final_info["reward_dmg_mult"][v_done_id])
            stats.ep_rew_term_mult_mean += sum(v_final_info["reward_term_mult"][v_done_id])
            stats.ep_rew_relval_mult_mean += sum(v_final_info["reward_relval_mult"][v_done_id])

    assert len(storage.rollout_buffer) == num_vsteps * venv.num_envs

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_rounds_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes
        stats.ep_rew_step_fixed_mean /= stats.num_episodes
        stats.ep_rew_prog_mean /= stats.num_episodes
        stats.ep_rew_dmg_mult_mean /= stats.num_episodes
        stats.ep_rew_term_mult_mean /= stats.num_episodes
        stats.ep_rew_relval_mult_mean /= stats.num_episodes

    # bootstrap value if not done
    v_next_hdata_batch = Batch.from_data_list(storage.v_next_hdata_list).to(model.device)
    add_action_active_local_ids(v_next_hdata_batch)
    v_next_value = model.model_value.forward_value(v_next_hdata_batch).flatten().cpu()

    for i, hdata in enumerate(storage.v_next_hdata_list):
        hdata.value = v_next_value[i]

    return stats


def eval_model(logger, model, venv, num_vsteps):
    assert torch.is_inference_mode_enabled()

    stats = SampleStats()
    v_obs, _ = venv.reset()
    v_done = torch.zeros(venv.num_envs, dtype=torch.bool)

    for vstep in range(0, num_vsteps):
        # logger.debug("(eval) vstep: %d" % vstep)
        # print(venv.render()[0])

        v_hdata_list = to_hdata_list(venv.call("graph_obs"), v_done)
        v_hdata_batch = Batch.from_data_list(v_hdata_list).to(model.device)
        add_action_active_local_ids(v_hdata_batch)

        v_action, v_logprob, v_entropy = model.model_policy.forward_policy(v_hdata_batch, deterministic=True)
        _, v_rew, v_term, v_trunc, v_info = venv.step(v_action.cpu().numpy())
        v_done = torch.as_tensor(np.logical_or(v_term, v_trunc), dtype=torch.bool)

        # See notes/gym_vector.txt
        if "_final_info" in v_info:
            v_done_id = np.flatnonzero(v_info["_final_info"])
            v_final_info = v_info["final_info"]
            stats.ep_rew_mean += sum(v_final_info["episode"]["r"][v_done_id])
            stats.ep_len_mean += sum(v_final_info["episode"]["l"][v_done_id])
            stats.ep_value_mean += sum(v_final_info["net_value"][v_done_id])
            stats.ep_rounds_mean += sum(v_final_info["round"][v_done_id])
            stats.ep_is_success_mean += sum(v_final_info["is_success"][v_done_id])
            stats.ep_rew_step_fixed_mean += sum(v_final_info["reward_step_fixed"][v_done_id])
            stats.ep_rew_dmg_mult_mean += sum(v_final_info["reward_dmg_mult"][v_done_id])
            stats.ep_rew_term_mult_mean += sum(v_final_info["reward_term_mult"][v_done_id])
            stats.ep_rew_relval_mult_mean += sum(v_final_info["reward_relval_mult"][v_done_id])
            stats.ep_rew_prog_mean += sum(v_final_info["reward_prog"][v_done_id])

            assert len(v_done_id) == int(np.logical_or(v_term, v_trunc).sum())
            stats.num_episodes += len(v_done_id)
            stats.num_truncations += int(v_trunc.sum())

    if stats.num_episodes > 0:
        stats.ep_rew_mean /= stats.num_episodes
        stats.ep_len_mean /= stats.num_episodes
        stats.ep_value_mean /= stats.num_episodes
        stats.ep_rounds_mean /= stats.num_episodes
        stats.ep_is_success_mean /= stats.num_episodes
        stats.ep_rew_step_fixed_mean /= stats.num_episodes
        stats.ep_rew_dmg_mult_mean /= stats.num_episodes
        stats.ep_rew_term_mult_mean /= stats.num_episodes
        stats.ep_rew_relval_mult_mean /= stats.num_episodes
        stats.ep_rew_prog_mean /= stats.num_episodes

    return stats


def train_model(
    logger,
    model: DNAModel,
    old_model_policy: GNNModel,
    optimizer_policy,
    optimizer_value,
    optimizer_distill,
    autocast_ctx,
    scaler,
    storage,
    train_config
):
    assert torch.is_grad_enabled()

    num_vsteps = train_config["num_vsteps"]
    num_envs = sum(v["num"] for v in train_config["env"]["envs_per_opponent"].values())
    v_next_hdata_batch = Batch.from_data_list(storage.v_next_hdata_list)
    add_action_active_local_ids(v_next_hdata_batch)

    # compute advantages
    with torch.no_grad():
        lastgaelam = torch.zeros_like(storage.bv_advantages[0])

        for t in reversed(range(num_vsteps)):
            if t == num_vsteps - 1:
                nextnonterminal = 1.0 - v_next_hdata_batch.done
                nextvalues = v_next_hdata_batch.value
            else:
                nextnonterminal = 1.0 - storage.bv_dones[t + 1]
                nextvalues = storage.bv_values[t + 1]
            delta = storage.bv_rewards[t] + train_config["gamma"] * nextvalues * nextnonterminal - storage.bv_values[t]
            storage.bv_advantages[t] = lastgaelam = delta + train_config["gamma"] * train_config["gae_lambda"] * nextnonterminal * lastgaelam
        storage.bv_returns[:] = storage.bv_advantages + storage.bv_values

        for b in range(num_vsteps):
            for v in range(num_envs):
                v_hdata = storage.rollout_buffer[b*num_envs + v]
                v_hdata.advantage = storage.bv_advantages[b, v]
                v_hdata.ep_return = storage.bv_returns[b, v]

    batch_size = num_vsteps * num_envs
    minibatch_size = int(batch_size // train_config["num_minibatches"])

    dataloader = DataLoader(
        storage.rollout_buffer,
        batch_size=minibatch_size,
        shuffle=True,
        pin_memory=True,
    )

    approx_kls = []
    clipfracs = []
    value_losses = []
    policy_losses = []
    entropy_losses = []
    distill_losses = []
    kl_exceeded = False

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.policy) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.policy) minibatch: %d" % i)
            mb = mb.to(model.device, non_blocking=True)
            add_action_active_local_ids(mb)

            with autocast_ctx(True):
                _newaction, newlogprob, newentropy = model.model_policy.forward_policy(mb, b_action=mb.action)

            oldlogprob = mb.logprob.float()
            newlogprob = newlogprob.float()

            logratio = newlogprob - oldlogprob
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                approx_kls.append(approx_kl.detach().item())
                clipfracs.append(((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item())

            if train_config["target_kl"] is not None and approx_kl > train_config["target_kl"]:
                kl_exceeded = True
                break

            advantages = mb.advantage.float()
            if train_config["norm_adv"]:
                mean = advantages.mean()
                var = advantages.var(unbiased=False)
                advantages = (advantages - mean) * torch.rsqrt(var + 1e-8)

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - train_config["clip_coef"], 1 + train_config["clip_coef"])
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = newentropy.float().mean()

            loss = policy_loss - entropy_loss * train_config["ent_coef"]

            policy_losses.append(policy_loss.detach().item())
            entropy_losses.append(entropy_loss.detach().item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_policy)  # needed for clip_grad_norm
            nn.utils.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])
            scaler.step(optimizer_policy)
            scaler.update()
            optimizer_policy.zero_grad(set_to_none=True)

        if kl_exceeded:
            break

    # Value network optimization
    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.value) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.value) minibatch: %d" % i)
            mb = mb.to(model.device, non_blocking=True)
            add_action_active_local_ids(mb)

            with autocast_ctx(True):
                newvalue = model.model_value.forward_value(mb)

            newvalue = newvalue.view(-1).float()
            oldvalue = mb.value.float()
            returns = mb.ep_return.float()

            if train_config["clip_vloss"]:
                v_loss_unclipped = (newvalue - returns) ** 2
                v_clipped = oldvalue + torch.clamp(
                    newvalue - oldvalue,
                    -train_config["clip_coef"],
                    train_config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((newvalue - returns) ** 2).mean()

            value_losses.append(value_loss.detach().item())

            scaler.scale(value_loss).backward()
            scaler.unscale_(optimizer_value)  # needed for clip_grad_norm
            nn.utils.clip_grad_norm_(model.model_value.parameters(), train_config["max_grad_norm"])
            scaler.step(optimizer_value)
            scaler.update()
            optimizer_value.zero_grad(set_to_none=True)

    # Value network to policy network distillation
    old_model_policy.load_state_dict(model.model_policy.state_dict(), strict=True)
    old_model_policy.eval()

    for epoch in range(train_config["update_epochs"]):
        logger.debug("(train.distill) epoch: %d" % epoch)
        for i, mb in enumerate(dataloader):
            logger.debug("(train.distill) minibatch: %d" % i)
            mb = mb.to(model.device, non_blocking=True)
            add_action_active_local_ids(mb)

            batch_size = mb.num_graphs

            with torch.no_grad(), autocast_ctx(True):
                # Compute policy and value targets
                old_gnn_out = old_model_policy.gnn(mb)
                (
                    old_active_logits,
                    old_active_batch_index,
                    old_active_local_action_ids,
                    old_batch_size,
                ) = old_model_policy._get_active_logits(old_gnn_out, mb)
                value_target = model.model_value.forward_value(mb)

            with autocast_ctx(True):
                new_gnn_out = model.model_policy.gnn(mb)
                (
                    new_active_logits,
                    new_active_batch_index,
                    new_active_local_action_ids,
                    new_batch_size,
                ) = model.model_policy._get_active_logits(new_gnn_out, mb)
                new_value = model.model_policy._forward_value(new_gnn_out)

            assert old_batch_size == new_batch_size
            assert batch_size == old_batch_size
            assert torch.equal(old_active_batch_index, new_active_batch_index)
            assert torch.equal(old_active_local_action_ids, new_active_local_action_ids)

            old_log_probs = torch.log(gnn_softmax(old_active_logits.float(), old_active_batch_index).clamp_min(1e-12))
            new_log_probs = torch.log(gnn_softmax(new_active_logits.float(), new_active_batch_index).clamp_min(1e-12))
            per_action_kl = old_log_probs.exp() * (old_log_probs - new_log_probs)
            b_kl = scatter_sum(per_action_kl, old_active_batch_index, dim=0, dim_size=batch_size)
            distill_actloss = b_kl.mean()

            distill_vloss = 0.5 * (new_value.view(-1) - value_target.view(-1)).square().mean()
            distill_loss = distill_vloss + train_config["distill_beta"] * distill_actloss

            distill_losses.append(distill_loss.detach().item())

            scaler.scale(distill_loss).backward()
            scaler.unscale_(optimizer_distill)  # needed for clip_grad_norm
            nn.utils.clip_grad_norm_(model.model_policy.parameters(), train_config["max_grad_norm"])
            scaler.step(optimizer_distill)
            scaler.update()
            optimizer_distill.zero_grad(set_to_none=True)

    y_pred, y_true = storage.bv_values.cpu().numpy(), storage.bv_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return TrainStats(
        value_loss=safe_mean(value_losses),
        policy_loss=safe_mean(policy_losses),
        entropy_loss=safe_mean(entropy_losses),
        distill_loss=safe_mean(distill_losses),
        approx_kl=safe_mean(approx_kls),
        clipfrac=safe_mean(clipfracs),
        explained_var=float(explained_var),
    )


def prepare_wandb_log(
    model,
    optimizer,
    state,
    train_stats,
    train_sample_stats,
    eval_multistats,
):
    wlog = {}

    if eval_multistats.num_episodes > 0:
        reward_abs_tot = (
            abs(eval_multistats.ep_rew_step_fixed_mean)
            + abs(eval_multistats.ep_rew_dmg_mult_mean)
            + abs(eval_multistats.ep_rew_term_mult_mean)
            + abs(eval_multistats.ep_rew_relval_mult_mean)
            + abs(eval_multistats.ep_rew_prog_mean)
        )

        assert reward_abs_tot > 0

        wlog.update({
            "eval/ep_rew_mean": eval_multistats.ep_rew_mean,
            "eval/ep_value_mean": eval_multistats.ep_value_mean,
            "eval/ep_len_mean": eval_multistats.ep_len_mean,
            "eval/ep_rounds_mean": eval_multistats.ep_rounds_mean,
            "eval/ep_success_rate": eval_multistats.ep_is_success_mean,
            "eval/ep_count": eval_multistats.num_episodes,
            "eval/ep_trunc_count": eval_multistats.num_truncations,
            "eval/reward/step_fixed": eval_multistats.ep_rew_step_fixed_mean,
            "eval/reward/dmg_mult": eval_multistats.ep_rew_dmg_mult_mean,
            "eval/reward/term_mult": eval_multistats.ep_rew_term_mult_mean,
            "eval/reward/relval_mult": eval_multistats.ep_rew_relval_mult_mean,
            "eval/reward/prog": eval_multistats.ep_rew_prog_mean,
            "eval/reward_rel/step_fixed": abs(eval_multistats.ep_rew_step_fixed_mean) / reward_abs_tot,
            "eval/reward_rel/dmg_mult": abs(eval_multistats.ep_rew_dmg_mult_mean) / reward_abs_tot,
            "eval/reward_rel/term_mult": abs(eval_multistats.ep_rew_term_mult_mean) / reward_abs_tot,
            "eval/reward_rel/relval_mult": abs(eval_multistats.ep_rew_relval_mult_mean) / reward_abs_tot,
            "eval/reward_rel/prog": abs(eval_multistats.ep_rew_prog_mean) / reward_abs_tot,
        })

    for name, eval_sample_stats in eval_multistats.variants.items():
        reward_abs_tot = (
            abs(eval_sample_stats.ep_rew_step_fixed_mean)
            + abs(eval_sample_stats.ep_rew_dmg_mult_mean)
            + abs(eval_sample_stats.ep_rew_term_mult_mean)
            + abs(eval_sample_stats.ep_rew_relval_mult_mean)
            + abs(eval_sample_stats.ep_rew_prog_mean)
        )

        wlog.update({
            f"eval/{name}/ep_rew_mean": eval_sample_stats.ep_rew_mean,
            f"eval/{name}/ep_value_mean": eval_sample_stats.ep_value_mean,
            f"eval/{name}/ep_len_mean": eval_sample_stats.ep_len_mean,
            f"eval/{name}/ep_rounds_mean": eval_sample_stats.ep_rounds_mean,
            f"eval/{name}/ep_success_rate": eval_sample_stats.ep_is_success_mean,
            f"eval/{name}/ep_count": eval_sample_stats.num_episodes,
            f"eval/{name}/ep_trunc_count": eval_sample_stats.num_truncations,
            f"eval/{name}/reward/step_fixed_mean": eval_sample_stats.ep_rew_step_fixed_mean,
            f"eval/{name}/reward/dmg_mult_mean": eval_sample_stats.ep_rew_dmg_mult_mean,
            f"eval/{name}/reward/term_mult_mean": eval_sample_stats.ep_rew_term_mult_mean,
            f"eval/{name}/reward/relval_mult_mean": eval_sample_stats.ep_rew_relval_mult_mean,
            f"eval/{name}/reward/prog_mean": eval_sample_stats.ep_rew_prog_mean,
            f"eval/{name}/reward_rel/step_fixed": abs(eval_sample_stats.ep_rew_step_fixed_mean) / reward_abs_tot,
            f"eval/{name}/reward_rel/dmg_mult": abs(eval_sample_stats.ep_rew_dmg_mult_mean) / reward_abs_tot,
            f"eval/{name}/reward_rel/term_mult": abs(eval_sample_stats.ep_rew_term_mult_mean) / reward_abs_tot,
            f"eval/{name}/reward_rel/relval_mult": abs(eval_sample_stats.ep_rew_relval_mult_mean) / reward_abs_tot,
            f"eval/{name}/reward_rel/prog": abs(eval_sample_stats.ep_rew_prog_mean) / reward_abs_tot,
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
            "train/ep_rounds_mean": train_sample_stats.ep_rounds_mean,
            "train/ep_success_rate": train_sample_stats.ep_is_success_mean,
            "train/ep_count": train_sample_stats.num_episodes,
            "train/ep_trunc_count": train_sample_stats.num_truncations,
        })

    wlog.update({
        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train/value_loss": train_stats.value_loss,
        "train/policy_loss": train_stats.policy_loss,
        "train/entropy_loss": train_stats.entropy_loss,
        "train/distill_loss": train_stats.distill_loss,
        "train/nan/value_loss": math.isnan(train_stats.value_loss) or math.isinf(train_stats.value_loss),
        "train/nan/policy_loss": math.isnan(train_stats.policy_loss) or math.isinf(train_stats.policy_loss),
        "train/nan/entropy_loss": math.isnan(train_stats.entropy_loss) or math.isinf(train_stats.entropy_loss),
        "train/nan/distill_loss": math.isnan(train_stats.distill_loss) or math.isinf(train_stats.distill_loss),
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
        "global/global_num_rollouts": state.global_rollout,
        "global/num_rollouts": state.current_rollout,
        "global/num_timesteps": state.current_timestep,
        "global/num_seconds": state.current_second,
        "global/num_episode": state.current_episode,
    })

    return wlog


def init_model_loader(env_config, checkpoint_config, out_dir, logger, dry_run, device):
    if env_config["envs_per_opponent"]["model"]["num"] == 0:
        return None

    # Wanted bot role based on train role
    bot_roles = dict(defender="attacker", attacker="defender")
    bot_role = bot_roles[env_config["kwargs"]["role"]]
    model_loader = ModelLoader(device.type, role=bot_role)

    modelcfg = env_config["model"]
    assert modelcfg, str(modelcfg)
    assert modelcfg["type"] in ["static", "dynamic"]

    if modelcfg["type"] == "static":
        # For "static" models, config and weights are hard-coded in config
        config_file = modelcfg["config_file"]
        weights_file = modelcfg["weights_file"]
        logger.info(f"Loading static model config from {config_file}")
        model_loader.configure(config_file)
        logger.info(f"Loading static model weights from {weights_file}")
        model_loader.load(weights_file)
        # These models will never be reloaded => set a huge reload interval
        latest_ts = None
        reload_interval_s = 1e9
    else:
        # # BEGIN: DEBUG
        # print("*** DEBUG: force local model ***")
        # latest_ts = dt.datetime(2000, 1, 1).astimezone(dt.timezone.utc)
        # config_file = [f for f in os.listdir(".") if "nkjrmrsq" in f and f.endswith(".json")][0]
        # weights_file = [f for f in os.listdir(".") if "nkjrmrsq" in f and f.endswith(".pt")][0]
        # # EOF: DEBUG

        # For "dynamic" models, config and weights may change on each download
        latest_ts, config_file, weights_file = download_latest_model(
            logger,
            dry_run=dry_run,
            out_dir=out_dir,
            algo="dna",
            run_id=modelcfg["run_id"],
            optimize_local_storage=checkpoint_config["optimize_local_storage"],
            s3_config=checkpoint_config["s3"],
            # Force download via a timestamp in the distant past
            timestamp=dt.datetime(2000, 1, 1).astimezone(dt.timezone.utc)
        )

        logger.info(f"Loading dynamic model config from {config_file} / {latest_ts.isoformat()}")
        model_loader.configure(config_file)
        logger.info(f"Loading dynamic model weights from {weights_file}")
        model_loader.load(weights_file)
        reload_interval_s = modelcfg["reload_interval_s"]

    with open(config_file, "r") as f:
        run_id = json.load(f)["run"]["id"]

    return ModelLoaderInfo(
        model_loader=model_loader,
        model_ts=latest_ts,
        loaded_at=dt.datetime.now().astimezone(dt.timezone.utc),
        reload_interval_s=reload_interval_s,
        run_id=run_id,
        config_file=config_file,
        weights_file=weights_file
    )


def main(config, loglevel, dry_run, no_wandb, seconds_total=float("inf"), skip_eval=False, max_rollouts=float("inf"), save_on_exit=True):
    run_id = config["run"]["id"]
    resumed_config = config["run"]["resumed_config"]

    fcfg = os.path.join(config["run"]["out_dir"], f"{run_id}-config.json")
    msg = f"Saving new config to: {fcfg}"

    if dry_run:
        print(f"{msg} (--dry-run)")
    else:
        os.makedirs(config["run"]["out_dir"], exist_ok=True)
        with open(fcfg, "w") as f:
            print(msg)
            json.dump(config, f, indent=4)

    # assert config["checkpoint"]["interval_s"] > config["eval"]["interval_s"]
    assert config["checkpoint"]["permanent_interval_s"] > config["eval"]["interval_s"]

    # A blind guess for the time it takes to complete an eval cycle
    # i.e. the slowest env to finish eval_model().
    # Real value depends on num_steps, num_envs, opponent & hardware
    eval_duration_s_guess = 600
    assert config["train"]["env"]["kwargs"]["user_timeout"] >= eval_duration_s_guess

    checkpoint_config = dig(config, "checkpoint")
    train_config = dig(config, "train")
    eval_config = dig(config, "eval")

    logfilename = None if dry_run else os.path.join(config["run"]["out_dir"], f"{run_id}.log")
    logger = StructuredLogger(level=getattr(logging, loglevel), filename=logfilename, context=dict(run_id=run_id))

    logger.info(dict(config=config))

    learning_rate = config["train"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    if train_config.get("torch_detect_anomaly", None):
        torch.autograd.set_detect_anomaly(True)  # debug

    if train_config.get("torch_cuda_matmul", None):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    loader_infos = []

    train_loader_info = init_model_loader(train_config["env"], checkpoint_config, config["run"]["out_dir"], logger, dry_run, device)

    if train_loader_info:
        loader_infos.append(train_loader_info)
        train_model_loader = train_loader_info.model_loader
    else:
        train_model_loader = None

    train_venv_seed = random.randint(0, (2**30)-1)  # leave some room to add i (see DualVecEnv)
    train_venv = DualVecEnv(
        env_kwargs=dict(train_config["env"]["kwargs"], seed=train_venv_seed),
        envs_stupidai=train_config["env"]["envs_per_opponent"]["StupidAI"],
        envs_battleai=train_config["env"]["envs_per_opponent"]["BattleAI"],
        envs_mmai_battleai=train_config["env"]["envs_per_opponent"]["MMAI_BATTLEAI"],
        envs_model=train_config["env"]["envs_per_opponent"]["model"],
        model_loader=train_model_loader,
        logprefix="train-",
    )

    logger.info("Initialized %d train envs (%s)" % (train_venv.num_envs, {k: v["num"] for k, v in train_config["env"]["envs_per_opponent"].items()}))

    eval_venv_variants = {}
    for name, envcfg in eval_config["env_variants"].items():
        # Blind guess for the time ot tales tp complete 1 training cycle
        # i.e. one cycle of collect_samples() + train_model()
        # Real value depends on num_steps, num_envs, opponent & hardware
        assert envcfg["kwargs"]["user_timeout"] >= config["eval"]["interval_s"] + 300
        eval_loader_info = init_model_loader(envcfg, checkpoint_config, config["run"]["out_dir"], logger, dry_run, device)
        if eval_loader_info:
            loader_infos.append(eval_loader_info)
            eval_model_loader = eval_loader_info.model_loader
        else:
            eval_model_loader = None

        eval_venv_seed = random.randint(0, (2**30)-1)  # leave some room to add i (see DualVecEnv)
        eval_venv_variants[name] = DualVecEnv(
            env_kwargs=dict(envcfg["kwargs"], seed=eval_venv_seed),
            envs_stupidai=envcfg["envs_per_opponent"]["StupidAI"],
            envs_battleai=envcfg["envs_per_opponent"]["BattleAI"],
            envs_mmai_battleai=envcfg["envs_per_opponent"]["MMAI_BATTLEAI"],
            envs_model=envcfg["envs_per_opponent"]["model"],
            model_loader=eval_model_loader,
            logprefix=f"eval/{name}-",
        )

        logger.info("Initialized %d eval envs (variant '%s')" % (sum(v["num"] for v in envcfg["envs_per_opponent"].values()), name))

    num_envs = train_venv.num_envs
    num_steps = train_config["num_vsteps"] * num_envs
    batch_size = int(num_steps)
    assert batch_size % train_config["num_minibatches"] == 0, f"{batch_size} % {train_config['num_minibatches']} == 0"
    storage = Storage(train_venv, train_config["num_vsteps"], torch.device("cpu"))  # force storage on cpu
    state = State()

    model = DNAModel(
        node_types=VcmiEnv.node_types(),
        edge_types=VcmiEnv.filtered_edge_types(config["train"]["env"]["kwargs"]["ignored_edges"]),
        config=config["model"],
        device=device
    )

    old_model_policy = copy.deepcopy(model.model_policy).to(device).eval()
    for p in old_model_policy.parameters():
        p.requires_grad = False

    optimizer_policy = torch.optim.Adam(model.model_policy.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(model.model_value.parameters(), lr=learning_rate)
    optimizer_distill = torch.optim.Adam(model.model_policy.parameters(), lr=learning_rate)

    optimizer_policy.param_groups[0].setdefault("initial_lr", learning_rate)
    optimizer_value.param_groups[0].setdefault("initial_lr", learning_rate)
    optimizer_distill.param_groups[0].setdefault("initial_lr", learning_rate)

    if train_config["torch_autocast"]:
        autocast_ctx = lambda enabled: torch.autocast(device.type, enabled=enabled)
        scaler = torch.GradScaler(device.type, enabled=True)
    else:
        # No-op autocast and scaler
        autocast_ctx = contextlib.nullcontext
        scaler = torch.GradScaler(device.type, enabled=False)

    logger.debug("Initialized models and optimizers (autocast=%s)" % train_config["torch_autocast"])

    if resumed_config:
        load_checkpoint(
            logger=logger,
            dry_run=dry_run,
            models={"dna": model},
            optimizers={
                "policy": optimizer_policy,
                "value": optimizer_value,
                "distill": optimizer_distill,
            },
            scalers={"default": scaler},
            states={"default": state},
            out_dir=config["run"]["out_dir"],
            run_id=run_id,
            optimize_local_storage=checkpoint_config["optimize_local_storage"],
            s3_config=checkpoint_config["s3"],
            device=device,
        )

        state.current_rollout = 0
        state.current_timestep = 0
        state.current_second = 0
        state.current_episode = 0
        state.current_vstep = 0

        # lr is lost after loading weights
        optimizer_policy.param_groups[0]["lr"] = learning_rate
        optimizer_value.param_groups[0]["lr"] = learning_rate
        optimizer_distill.param_groups[0]["lr"] = learning_rate

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
        "global/instance_id": os.getenv("VASTAI_INSTANCE_ID", 0),
        "train_config/num_envs": num_envs,
        "train_config/num_vsteps": train_config["num_vsteps"],
        "train_config/num_minibatches": train_config["num_minibatches"],
        "train_config/update_epochs": train_config["update_epochs"],
        "train_config/gamma": train_config["gamma"],
        "train_config/gae_lambda": train_config["gae_lambda"],
        "train_config/ent_coef": train_config["ent_coef"],
        "train_config/clip_coef": train_config["clip_coef"],
        "train_config/learning_rate": train_config["learning_rate"],  # also logged during training
        "train_config/norm_adv": int(train_config["norm_adv"]),
        "train_config/clip_vloss": int(train_config["clip_vloss"]),
        "train_config/max_grad_norm": train_config["max_grad_norm"],
        "train_config/distill_beta": train_config["distill_beta"],
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

    # For benchmark
    cumulative_timer_values = {k: 0 for k in timers.keys()}

    timers["all"].start()
    eval_net_value_best = None

    permanent_checkpoint_timer = Timer(elapsed=state.permanent_checkpoint_timer_elapsed)
    permanent_checkpoint_timer.start()
    volatile_checkpoint_timer = Timer(elapsed=state.volatile_checkpoint_timer_elapsed)
    volatile_checkpoint_timer.start()
    wandb_log_commit_timer = Timer()
    wandb_log_commit_timer.start()
    eval_timer = Timer(elapsed=state.eval_timer_elapsed)
    eval_timer.start()

    if skip_eval:
        eval_timer.reset(start=True)

    lr_schedule_timer = Timer()
    lr_schedule_timer.start()

    if train_config["lr_scheduler_mod"]:
        lr_scheduler_mod = importlib.import_module(train_config["lr_scheduler_mod"])
        lr_scheduler_cls = getattr(lr_scheduler_mod, train_config["lr_scheduler_cls"])

        lr_schedule_policy = lr_scheduler_cls(optimizer_policy, **train_config["lr_scheduler_kwargs"])
        lr_schedule_value = lr_scheduler_cls(optimizer_value, **train_config["lr_scheduler_kwargs"])
        lr_schedule_distill = lr_scheduler_cls(optimizer_distill, **train_config["lr_scheduler_kwargs"])
    else:
        lr_schedule_policy = torch.optim.lr_scheduler.LambdaLR(optimizer_policy, lr_lambda=lambda _: 1)
        lr_schedule_value = torch.optim.lr_scheduler.LambdaLR(optimizer_value, lr_lambda=lambda _: 1)
        lr_schedule_distill = torch.optim.lr_scheduler.LambdaLR(optimizer_distill, lr_lambda=lambda _: 1)

    # TODO: torch LR schedulers are very buggy and cannot be resumed reliably
    # (they perform just 1 step for StepLR; they change the step size for LinearLR, ...etc)
    # Also, advancing manually like this raises warning for not calling optimizer.step()
    # Also, calling .step(N) raises deprecation warning...
    for _ in range(state.global_second // train_config["lr_scheduler_interval_s"]):
        lr_schedule_policy.step()
        lr_schedule_value.step()
        lr_schedule_distill.step()

    global_second_start = state.global_second

    save_fn = partial(
        save_checkpoint,
        logger=logger,
        dry_run=dry_run,
        models={"dna": model},
        optimizers={
            "policy": optimizer_policy,
            "value": optimizer_value,
            "distill": optimizer_distill,
        },
        scalers={"default": scaler},
        states={"default": state},
        out_dir=config["run"]["out_dir"],
        run_id=run_id,
        optimize_local_storage=checkpoint_config["optimize_local_storage"],
        s3_config=None,
        config=config,
        # We upload only timestamped (unique) checkpoints
        # => no risk of overwriting a file that is still being uploaded
        # => always pass a new, unset event
        uploading_event=uploading_event,
        async_upload=True,
    )

    try:
        while state.current_rollout < max_rollouts:
            state.global_second = global_second_start + int(cumulative_timer_values["all"])
            state.current_second = int(cumulative_timer_values["all"])

            if state.current_second >= seconds_total:
                break

            state.permanent_checkpoint_timer_elapsed = permanent_checkpoint_timer.peek()
            state.volatile_checkpoint_timer_elapsed = volatile_checkpoint_timer.peek()
            state.eval_timer_elapsed = eval_timer.peek()

            [v.reset(start=(k == "all")) for k, v in timers.items()]

            logger.debug("learning_rate: %s" % optimizer_policy.param_groups[0]['lr'])
            if lr_schedule_timer.peek() > train_config["lr_scheduler_interval_s"]:
                lr_schedule_timer.reset(start=True)
                lr_schedule_policy.step()
                lr_schedule_value.step()
                lr_schedule_distill.step()
                logger.info("New learning_rate: %s" % optimizer_policy.param_groups[0]['lr'])

            now = dt.datetime.now().astimezone(dt.timezone.utc)
            for loader_info in loader_infos:
                next_check_at = loader_info.loaded_at + dt.timedelta(seconds=loader_info.reload_interval_s)
                if now < next_check_at:
                    continue

                loader_info.loaded_at = now
                logger.info(f"Check if newer model exists for {loader_info.run_id} / {loader_info.model_ts.isoformat()}")
                try:
                    latest_ts, _, weights_file = download_latest_model(
                        logger=logger,
                        dry_run=dry_run,
                        algo="dna",
                        out_dir=config["run"]["out_dir"],
                        run_id=loader_info.run_id,
                        optimize_local_storage=checkpoint_config["optimize_local_storage"],
                        s3_config=checkpoint_config["s3"],
                        timestamp=loader_info.model_ts
                    )

                    if latest_ts is None or latest_ts <= loader_info.model_ts:
                        logger.info(f"No newer model found for {loader_info.run_id}")
                        continue

                    loader_info.model_ts = latest_ts

                    # Remove old weights
                    if weights_file != loader_info.weights_file and os.path.exists(loader_info.weights_file):
                        msg = f"Removing old weights at {loader_info.weights_file}"
                        if dry_run:
                            logger.info(msg + " (--dry-run)")
                        else:
                            logger.info(msg)
                            os.unlink(loader_info.weights_file)

                    loader_info.weights_file = weights_file
                    loader_info.model_loader.load(weights_file)
                    loader_info.loaded_at = now
                except Exception as e:
                    logger.error("Error while trying to update model %s: %s\n%s" % (
                        loader_info.run_id, e, "\n".join(traceback.format_exception(e))
                    ))

            # Evaluate first (for a baseline when resuming with modified params)
            eval_multistats = MultiStats()

            if eval_timer.peek() > eval_config["interval_s"]:
                logger.info("Time for eval")
                eval_timer.reset(start=True)

                with timers["eval"]:
                    model.eval()

                    def eval_worker_fn(name, venv, vsteps):
                        sublogger = logger.sublogger(dict(variant=name))
                        with torch.inference_mode():
                            sublogger.info("Start evaluating env variant: %s" % name)
                            stats = eval_model(logger=sublogger, model=model, venv=venv, num_vsteps=vsteps)
                            sublogger.info("Done evaluating env variant: %s" % name)
                            return name, stats

                    with ThreadPoolExecutor(max_workers=100) as ex:
                        futures = [
                            ex.submit(eval_worker_fn, name, venv, eval_config["num_vsteps"])
                            for name, venv in eval_venv_variants.items()
                        ]

                        for fut in as_completed(futures):
                            eval_multistats.add(*fut.result())

            if eval_multistats.num_episodes > 0:
                eval_net_value = eval_multistats.ep_value_mean

                if eval_net_value_best is None:
                    # Initial baseline for resumed configs
                    eval_net_value_best = eval_net_value
                    logger.info("No baseline for checkpoint yet (eval_net_value=%f, eval_net_value_best=None), setting it now" % eval_net_value)
                elif eval_net_value < eval_net_value_best:
                    logger.info("Bad checkpoint (eval_net_value=%f, eval_net_value_best=%f), will skip it" % (eval_net_value, eval_net_value_best))
                else:
                    logger.info("Good checkpoint (eval_net_value=%f, eval_net_value_best=%f), will save it" % (eval_net_value, eval_net_value_best))
                    if uploading_event.is_set():
                        logger.info("Still uploading previous 'best' checkpoint -- will not overwrite it")
                    else:
                        eval_net_value_best = eval_net_value
                        # Add resumes to filename to prevent overwriting pre-crash best results
                        # (functions in init.sh expect  alphanumeric tags => no separator)
                        save_fn(s3_config=None, tag=f"best{state.resumes}")

            with timers["sample"], torch.no_grad(), autocast_ctx(True):
                model.eval()
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

            model.train()
            with timers["train"]:
                train_stats = train_model(
                    logger=logger,
                    model=model,
                    old_model_policy=old_model_policy,
                    optimizer_policy=optimizer_policy,
                    optimizer_value=optimizer_value,
                    optimizer_distill=optimizer_distill,
                    autocast_ctx=autocast_ctx,
                    scaler=scaler,
                    storage=storage,
                    train_config=train_config,
                )

            if permanent_checkpoint_timer.peek() > checkpoint_config["permanent_interval_s"]:
                permanent_checkpoint_timer.reset(start=True)
                volatile_checkpoint_timer.reset(start=True)
                logger.info("Time for a permanent checkpoint")
                save_fn(s3_config=checkpoint_config["s3"], tag=f"{time.time():.0f}")
            elif volatile_checkpoint_timer.peek() > config["checkpoint"].get("volatile_interval_s", float("inf")):
                volatile_checkpoint_timer.reset(start=True)
                volatile_id = state.volatile_checkpoint_counter % checkpoint_config.get("volatile_num_tags", 2)
                save_fn(s3_config=checkpoint_config["s3"], tag=f"volatile{volatile_id}")
                state.volatile_checkpoint_counter += 1

            wlog = prepare_wandb_log(
                model=model.model_policy,
                optimizer=optimizer_policy,
                state=state,
                train_stats=train_stats,
                train_sample_stats=train_sample_stats,
                eval_multistats=eval_multistats,
            )

            accumulate_logs(wlog)

            if wandb_log_commit_timer.peek() > config["wandb_log_interval_s"]:
                # logger.info("Time for wandb log")
                wandb_log_commit_timer.reset(start=True)
                wlog.update(aggregate_logs())
                tstats = timer_stats(timers)
                wlog.update(tstats)
                wlog["train_config/learning_rate"] = optimizer_policy.param_groups[0]['lr']
                wandb.log(wlog, commit=True)

            logger.info(wlog)

            for k in timers.keys():
                cumulative_timer_values[k] += timers[k].peek()

            state.current_rollout += 1
            state.global_rollout += 1

        ret_rew = safe_mean(list(state.rollout_rew_queue_1000)[-min(300, state.current_rollout):])
        ret_value = safe_mean(list(state.rollout_net_value_queue_1000)[-min(300, state.current_rollout):])

        return ret_rew, ret_value, save_fn, cumulative_timer_values["all"]
    finally:
        if save_on_exit:
            save_fn(s3_config=None, tag=f"{time.time():.0f}")
        if os.getenv("VASTAI_INSTANCE_ID") and not dry_run:
            import vastai_sdk
            vastai_sdk.VastAI().label_instance(id=int(os.environ["VASTAI_INSTANCE_ID"]), label="IDLE")
        logger.warn(dict(event="finish", timers=cumulative_timer_values))


# This is in a separate function to prevent vars from being global
def init_config(args):
    if args.dry_run:
        args.no_wandb = True

    if args.f:
        assert args.run_id is None, "Cannot pass both --run-id and -f"
        # assert args.suffix is None, "Cannot pass both --suffix and -f"
        with open(args.f, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)
        config["run"]["resumed_config"] = args.f
    else:
        if args.run_id:
            assert re.match(r"^[0-9a-z]{8}$", args.run_id), f"bad run_id: {args.run_id}"
            run_id = args.run_id
        else:
            run_id = ''.join(random.choices(string.ascii_lowercase, k=8))

        from .config import config

        config["run"] = dict(
            id=run_id,
            name=config["name_template"].format(
                id=run_id,
                datetime=dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                suffix="v15" if args.suffix is None else args.suffix
            ),
            out_dir=os.path.abspath(config["out_dir_template"].format(id=run_id)),
            resumed_config=None,
        )

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", metavar="RUN_ID", help="run id to use (incompatible with -f)")
    parser.add_argument("--suffix", metavar="SUFFIX", help="wandb run name template suffix value")
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
    parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument("--loglevel", metavar="LOGLEVEL", default="INFO", help="DEBUG | INFO | WARN | ERROR")
    parser.add_argument("--skip-eval", action="store_true", help="do not eval at script start")
    parser.add_argument("--max-rollouts", metavar="N", type=int, default=0, help="exit after N rollouts, printing iterationsing info")
    args = parser.parse_args()

    config = init_config(args)

    *_, t = main(
        config=config,
        loglevel=args.loglevel,
        dry_run=args.dry_run,
        no_wandb=args.no_wandb,
        skip_eval=args.skip_eval,
        max_rollouts=args.max_rollouts or float("inf"),
        # seconds_total=10
    )
