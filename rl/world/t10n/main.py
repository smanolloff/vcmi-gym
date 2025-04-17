import os
import torch
import torch.nn as nn
import random
import string
import json
import botocore.exceptions
import threading
import logging
import argparse
import math
import time
import numpy as np
import enum
import contextlib
from functools import partial
from datetime import datetime

from torch.nn.functional import (
    mse_loss,
    binary_cross_entropy_with_logits,
    cross_entropy,
)

from ..util.dataset_vcmi import DatasetVCMI, Data, Context
from ..util.dataset_s3 import DatasetS3
from ..util.buffer_base import BufferBase
from ..util.obs_index import ObsIndex
from ..util.persistence import load_local_or_s3_checkpoint, save_checkpoint, save_buffer_async
from ..util.structured_logger import StructuredLogger
from ..util.wandb import setup_wandb
from ..util.timer import Timer
from ..util.constants_v11 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


class Other(enum.IntEnum):
    CAN_WAIT = 0
    DONE = enum.auto()

    _count = enum.auto()


# Skips last transition and carries reward over to the new transition
#
# (also see doc in dataset_vcmi.py)
#
# R(s0t0)=NaN is OK (episode start => no prev step, no reward)
# R(s1t0)=NaN is NOT OK => use R(s0t2) from prev step
# A(s0t2)=-1  is NOT OK => use A(s1t0) from next step
# A(s3t2)=-1  is OK (this is the terminal obs => no aciton)
#
# We `yield` a total of 9 samples:
#
#  | obs            | reward         | action         |
# -|----------------|----------------|----------------|
#  | O(s0t0)        | R(s0t0)=NaN    | A(s0t0)        | t=0 s=0
#  | O(s0t1)        | R(s0t1)        | A(s0t1)        | t=1
#  |                |                |                | t=2
# -|----------------|----------------|----------------|
#  | O(s1t0)        | R(s0t2) <- !!! | A(s1t0)        | t=0 s=1
#  | O(s1t1)        | R(s1t1)        | A(s1t1)        | t=1
#  |                |                |                | t=2
# -|----------------|----------------|----------------|
#  | O(s2t0)        | R(s1t2) <- !!! | A(s2t0)        | t=0 s=2
#  | O(s2t1)        | R(s2t1)        | A(s2t1)        | t=1
#  |                |                |                | t=2
# -|----------------|----------------|----------------|
#  | O(s3t0)        | R(s2t2) <- !!! | A(s3t0)        | t=0 s=3
#  | O(s3t1)        | R(s3t1)        | A(s3t1)        | t=1
#  | O(s3t2)        | R(s3t2)        | A(s1t0)=-1     | t=2 !!!
#
# =============================================================
#
# => for t=2 (the final transition):
#   - we carry its reward to next step's t=0
#   - we `yield` it only if s=3 (last step)
#

def vcmi_dataloader_functor():
    state = {"reward_carry": 0}

    def mw(data: Data, ctx: Context):
        if ctx.transition_id == ctx.num_transitions - 1:
            state["reward_carry"] = data.reward
            if not data.done:
                return None
        if ctx.transition_id == 0 and ctx.ep_steps > 0:
            return data._replace(reward=state["reward_carry"])
        return data

    return mw


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    for mod in list(layer.modules())[1:]:
        layer_init(mod, gain, bias_const)
    return layer


class Buffer(BufferBase):
    def _valid_indices(self):
        max_index = self.capacity if self.full else self.index
        # Valid are indices of samples where done=False and cutoff=False
        # (i.e. to ensure obs,next_obs is valid)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        ok_samples = ~self.containers["done"][:max_index - 1].bool()
        ok_samples[self.worker_cutoffs] = False
        return torch.nonzero(ok_samples, as_tuple=True)[0]

    def sample(self, batch_size):
        inds = self._valid_indices()
        sampled_indices = inds[torch.randint(len(inds), (batch_size,), device=self.device)]

        obs = self.containers["obs"][sampled_indices]
        # action_mask = self.containers["mask"][sampled_indices]
        action = self.containers["action"][sampled_indices]
        next_obs = self.containers["obs"][sampled_indices + 1]
        next_mask = self.containers["mask"][sampled_indices + 1]
        next_reward = self.containers["reward"][sampled_indices + 1]
        next_done = self.containers["done"][sampled_indices + 1]

        return obs, action, next_obs, next_mask, next_reward, next_done

    def sample_iter(self, batch_size):
        valid_indices = self._valid_indices()
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)]

        # The valid indices are < than all indices by `short`
        short = self.capacity - len(shuffled_indices)
        if short:
            filler_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)][:short]
            shuffled_indices = torch.cat((shuffled_indices, filler_indices))

        assert len(shuffled_indices) == self.capacity

        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            yield (
                self.containers["obs"][batch_indices],
                self.containers["action"][batch_indices],
                self.containers["obs"][batch_indices + 1],
                self.containers["mask"][batch_indices + 1],
                self.containers["reward"][batch_indices + 1],
                self.containers["done"][batch_indices + 1]
            )


class TransitionModel(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        obsind = ObsIndex(device)

        self.abs_index = obsind.abs_index
        self.rel_index_global = obsind.rel_index_global
        self.rel_index_player = obsind.rel_index_player
        self.rel_index_hex = obsind.rel_index_hex

        #
        # ChatGPT notes regarding encoders:
        #
        # Continuous ([0,1]):
        # Keep as-is (no normalization needed).
        # No linear layer or activation required.
        #
        # Binary (0/1):
        # Apply a linear layer to project to a small vector (e.g., Linear(1, d)).
        # No activation needed before concatenation.
        #
        # Categorical:
        # Use nn.Embedding(num_classes, d) for each feature.
        #
        # Concatenate all per-element features → final vector of shape (model_dim,).
        #
        # Pass to Transformer:
        # Optionally use a linear layer after concatenation to unify dimensions
        # (Linear(total_dim, model_dim)), especially if feature-specific dimensions vary.
        #

        #
        # Further details:
        #
        # Continuous data:
        # If your continuous inputs are already scaled to [0, 1], and you
        # cannot compute global normalization, it is acceptable to use them
        # without further normalization.
        #
        # Binary data:
        # To process binary inputs, apply nn.Linear(1, d) to each feature if
        # treating them separately, or nn.Linear(n, d) to the whole binary
        # vector if treating them jointly.
        #   binary_input = torch.tensor([[0., 1., 0., ..., 1.]])  # shape: (B, 30)
        #   linear = nn.Linear(30, d)  # d = desired output dimension
        #   output = linear(binary_input)  # shape: (B, d)

        emb_calc = lambda n: math.ceil(math.sqrt(n))

        self.encoder_action = nn.Embedding(N_ACTIONS, emb_calc(N_ACTIONS))

        #
        # Global encoders
        #

        # Continuous:
        # (B, n)
        self.encoder_global_continuous = nn.Identity()

        # Binaries:
        # (B, n)
        self.encoder_global_binary = nn.Identity()
        if self.abs_index["global"]["binary"].numel():
            n_binary_features = len(self.abs_index["global"]["binary"])
            self.encoder_global_binary = nn.LazyLinear(n_binary_features)
            # No nonlinearity needed

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        self.encoders_global_categoricals = nn.ModuleList([])
        for ind in self.rel_index_global["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_global_categoricals.append(cat_emb_size)

        # Merge
        z_size_global = 128
        self.encoder_merged_global = nn.Sequential(
            # => (B, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS)
            nn.LazyLinear(z_size_global),
            nn.LeakyReLU(),
        )
        # => (B, Z_GLOBAL)

        #
        # Player encoders
        #

        # Continuous per player:
        # (B, n)
        self.encoder_player_continuous = nn.Identity()

        # Binaries per player:
        # (B, n)
        self.encoder_player_binary = nn.Identity()
        if self.abs_index["player"]["binary"].numel():
            n_binary_features = len(self.abs_index["player"]["binary"][0])
            self.encoder_player_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        self.encoders_player_categoricals = nn.ModuleList([])
        for ind in self.rel_index_player["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_player_categoricals.append(cat_emb_size)

        # Merge per player
        z_size_player = 128
        self.encoder_merged_player = nn.Sequential(
            # => (B, 2, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS)
            nn.LazyLinear(z_size_player),
            nn.LeakyReLU(),
        )
        # => (B, 2, Z_PLAYER)

        #
        # Hex encoders
        #

        # Continuous per hex:
        # (B, n)
        self.encoder_hex_continuous = nn.Identity()

        # Binaries per hex:
        # (B, n)
        if self.abs_index["hex"]["binary"].numel():
            n_binary_features = len(self.abs_index["hex"]["binary"][0])
            self.encoder_hex_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per hex:
        # [(B, C1), (B, C2), ...]
        self.encoders_hex_categoricals = nn.ModuleList([])
        for ind in self.rel_index_hex["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_hex_categoricals.append(cat_emb_size)

        # Merge per hex
        z_size_hex = 512
        self.encoder_merged_hex = nn.Sequential(
            # => (B, 165, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS)
            nn.LazyLinear(z_size_hex),
            nn.LeakyReLU(),
        )
        # => (B, 165, Z_HEX)

        # Transformer (hexes only)
        self.transformer_hex = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=z_size_hex, nhead=8, batch_first=True),
            num_layers=6
        )
        # => (B, 165, Z_HEX)

        #
        # Aggregator
        #

        # (B, Z_GLOBAL + AVG(2*Z_PLAYER) + AVG(165*Z_HEX))
        self.aggregator = nn.LazyLinear(2048)
        # => (B, Z_AGG)

        #
        # Heads
        #

        # => (B, Z_AGG)
        self.head_global = nn.LazyLinear(STATE_SIZE_GLOBAL)

        # => (B, 2, Z_AGG + Z_PLAYER)
        self.head_player = nn.LazyLinear(STATE_SIZE_ONE_PLAYER)

        # => (B, 165, Z_AGG + Z_HEX)
        self.head_hex = nn.LazyLinear(STATE_SIZE_ONE_HEX)

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            obs = self.reconstruct(torch.randn([2, DIM_OBS], device=device))
            action = torch.tensor([1, 1], device=device)
            self.forward(obs, action)

        layer_init(self)

    def forward(self, obs, action):
        assert obs.device.type == self.device.type, f"{obs.device.type} == {self.device.type}"

        action_z = self.encoder_action(action)

        global_continuous_in = obs[:, self.abs_index["global"]["continuous"]]
        global_binary_in = obs[:, self.abs_index["global"]["binary"]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_binary_z = self.encoder_global_binary(global_binary_in)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_merged = torch.cat((action_z, global_continuous_z, global_binary_z, global_categorical_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_continuous_in = obs[:, self.abs_index["player"]["continuous"]]
        player_binary_in = obs[:, self.abs_index["player"]["binary"]]
        player_categorical_ins = [obs[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_binary_z = self.encoder_player_binary(player_binary_in)
        player_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 2, -1), player_continuous_z, player_binary_z, player_categorical_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_continuous_in = obs[:, self.abs_index["hex"]["continuous"]]
        hex_binary_in = obs[:, self.abs_index["hex"]["binary"]]
        hex_categorical_ins = [obs[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
        hex_continuous_z = self.encoder_hex_continuous(hex_continuous_in)
        hex_binary_z = self.encoder_hex_binary(hex_binary_in)
        hex_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)], dim=-1)
        hex_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 165, -1), hex_continuous_z, hex_binary_z, hex_categorical_z), dim=-1)
        z_hex = self.encoder_merged_hex(hex_merged)
        z_hex = self.transformer_hex(z_hex)
        # => (B, 165, Z_HEX)

        mean_z_player = z_player.mean(dim=1)
        mean_z_hex = z_hex.mean(dim=1)
        z_agg = self.aggregator(torch.cat([z_global, mean_z_player, mean_z_hex], dim=-1))
        # => (B, Z_AGG)

        #
        # Outputs
        #

        global_out = self.head_global(z_agg)
        # => (B, STATE_SIZE_GLOBAL)

        player_out = self.head_player(torch.cat([z_agg.unsqueeze(1).expand(-1, 2, -1), z_player], dim=-1))
        # => (B, 2, STATE_SIZE_ONE_PLAYER)

        hex_out = self.head_hex(torch.cat([z_agg.unsqueeze(1).expand(-1, 165, -1), z_hex], dim=-1))
        # => (B, 165, STATE_SIZE_ONE_HEX)

        obs_out = torch.cat((global_out, player_out.flatten(start_dim=1), hex_out.flatten(start_dim=1)), dim=1)

        # obs, rew, can_wait
        return obs_out

    def reconstruct(self, obs_out):
        global_continuous_out = obs_out[:, self.abs_index["global"]["continuous"]]
        global_binary_out = obs_out[:, self.abs_index["global"]["binary"]]
        global_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        player_continuous_out = obs_out[:, self.abs_index["player"]["continuous"]]
        player_binary_out = obs_out[:, self.abs_index["player"]["binary"]]
        player_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        hex_continuous_out = obs_out[:, self.abs_index["hex"]["continuous"]]
        hex_binary_out = obs_out[:, self.abs_index["hex"]["binary"]]
        hex_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
        next_obs = torch.zeros_like(obs_out)

        next_obs[:, self.abs_index["global"]["continuous"]] = torch.clamp(global_continuous_out, 0, 1)
        next_obs[:, self.abs_index["global"]["binary"]] = torch.sigmoid(global_binary_out).round()
        for ind, out in zip(self.abs_index["global"]["categoricals"], global_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.abs_index["player"]["continuous"]] = torch.clamp(player_continuous_out, 0, 1)
        next_obs[:, self.abs_index["player"]["binary"]] = torch.sigmoid(player_binary_out).round()
        for ind, out in zip(self.abs_index["player"]["categoricals"], player_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.abs_index["hex"]["continuous"]] = torch.clamp(hex_continuous_out, 0, 1)
        next_obs[:, self.abs_index["hex"]["binary"]] = torch.sigmoid(hex_binary_out).round()
        for ind, out in zip(self.abs_index["hex"]["categoricals"], hex_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        return next_obs

    # Predict next obs
    def predict(self, obs, action):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(0)

            was_training = self.training
            self.eval()
            try:
                obs_pred_logits = self.forward(obs, action)
                obs_pred = self.reconstruct(obs_pred_logits)[0].numpy()
            finally:
                self.train(was_training)

            return obs_pred


class Stats:
    def __init__(self, model, device):
        # Store [mean, var] for each continuous feature
        # Shape: (N_CONT_FEATURES, 2)
        self.continuous = {
            "global": torch.zeros(*model.rel_index_global["continuous"].shape, 2, device=device),
            "player": torch.zeros(*model.rel_index_player["continuous"].shape, 2, device=device),
            "hex": torch.zeros(*model.rel_index_hex["continuous"].shape, 2, device=device),
        }

        # Store [n_ones, n] for each binary feature
        # Shape: (N_BIN_FEATURES, 2)
        self.binary = {
            "global": torch.zeros(*model.rel_index_global["binary"].shape, 2, dtype=torch.int64, device=device),
            "player": torch.zeros(*model.rel_index_player["binary"].shape, 2, dtype=torch.int64, device=device),
            "hex": torch.zeros(*model.rel_index_hex["binary"].shape, 2, dtype=torch.int64, device=device),
        }

        # Store [n_ones_class0, n_ones_class1, ...]  for each categorical feature
        # Python list with N_CAT_FEATURES elements
        # Each element has shape: (N_CLASSES, 2), where N_CLASSES varies
        # e.g.
        # [
        #  [n_ones_F1_class0, n_ones_F1_class1, n_ones_F1_class2],
        #  [n_ones_F2_class0, n_ones_F2_class2, n_ones_F2_class3, n_ones_F2_class4],
        #  ...etc
        # ]
        self.categoricals = {
            "global": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_global["categoricals"]],
            "player": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_player["categoricals"]],
            "hex": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_hex["categoricals"]],
        }

        # Simple counter. 1 sample = 1 obs
        # i.e. the counter for each hex feature will be 165*num_samples
        self.num_samples = 0

        # Number of updates, should correspond to training iteration
        self.iteration = 0

    def export_data(self):
        return {
            "iteration": self.iteration,
            "num_samples": self.num_samples,
            "continuous": self.continuous,
            "binary": self.binary,
            "categoricals": self.categoricals
        }

    def load_state_dict(self, data):
        self.continuous = data["continuous"]
        self.binary = data["binary"]
        self.categoricals = data["categoricals"]

    def update(self, buffer, model):
        with torch.no_grad():
            self._update(buffer, model)

    def _update(self, buffer, model):
        self.num_samples += buffer.capacity

        # self.continuous["global"][:, 0] = model.encoder_global_continuous[0].running_mean
        # self.continuous["global"][:, 1] = model.encoder_global_continuous[0].running_var
        # self.continuous["player"][:, 0] = model.encoder_player_continuous[1].running_mean
        # self.continuous["player"][:, 1] = model.encoder_player_continuous[1].running_var
        # self.continuous["hex"][:, 0] = model.encoder_hex_continuous[1].running_mean
        # self.continuous["hex"][:, 1] = model.encoder_hex_continuous[1].running_var

        obs = buffer.obs_buffer

        # stat.add_(obs[:, ind].sum(0).round().long())
        values_global = obs[:, model.abs_index["global"]["binary"]].round().long()
        self.binary["global"][:, 0] += values_global.sum(0)
        self.binary["global"][:, 1] += np.prod(values_global.shape)

        values_player = obs[:, model.abs_index["player"]["binary"]].flatten(end_dim=1).round().long()
        self.binary["player"][:, 0] += values_player.sum(0)
        self.binary["player"][:, 1] += np.prod(values_player.shape)

        values_hex = obs[:, model.abs_index["hex"]["binary"]].flatten(end_dim=1).round().long()
        self.binary["hex"][:, 0] += values_hex.sum(0)
        self.binary["hex"][:, 1] += np.prod(values_hex.shape)

        for ind, stat in zip(model.abs_index["global"]["categoricals"], self.categoricals["global"]):
            stat.add_(obs[:, ind].round().long().sum(0))

        for ind, stat in zip(model.abs_index["player"]["categoricals"], self.categoricals["player"]):
            stat.add_(obs[:, ind].flatten(end_dim=1).round().long().sum(0))

        for ind, stat in zip(model.abs_index["hex"]["categoricals"], self.categoricals["hex"]):
            stat.add_(obs[:, ind].flatten(end_dim=1).round().long().sum(0))


def compute_losses(logger, obs_index, loss_weights, next_obs, pred_obs):
    logits_global_continuous = pred_obs[:, obs_index["global"]["continuous"]]
    logits_global_binary = pred_obs[:, obs_index["global"]["binary"]]
    logits_global_categoricals = [pred_obs[:, ind] for ind in obs_index["global"]["categoricals"]]
    logits_player_continuous = pred_obs[:, obs_index["player"]["continuous"]]
    logits_player_binary = pred_obs[:, obs_index["player"]["binary"]]
    logits_player_categoricals = [pred_obs[:, ind] for ind in obs_index["player"]["categoricals"]]
    logits_hex_continuous = pred_obs[:, obs_index["hex"]["continuous"]]
    logits_hex_binary = pred_obs[:, obs_index["hex"]["binary"]]
    logits_hex_categoricals = [pred_obs[:, ind] for ind in obs_index["hex"]["categoricals"]]

    loss_continuous = 0
    loss_binary = 0
    loss_categorical = 0

    # Global

    if logits_global_continuous.numel():
        target_global_continuous = next_obs[:, obs_index["global"]["continuous"]]
        loss_continuous += mse_loss(logits_global_continuous, target_global_continuous)

    if logits_global_binary.numel():
        target_global_binary = next_obs[:, obs_index["global"]["binary"]]
        # weight_global_binary = loss_weights["binary"]["global"]
        # loss_binary += binary_cross_entropy_with_logits(logits_global_binary, target_global_binary, pos_weight=weight_global_binary)
        loss_binary += binary_cross_entropy_with_logits(logits_global_binary, target_global_binary)

    if logits_global_categoricals:
        target_global_categoricals = [next_obs[:, index] for index in obs_index["global"]["categoricals"]]
        # weight_global_categoricals = loss_weights["categoricals"]["global"]
        # for logits, target, weight in zip(logits_global_categoricals, target_global_categoricals, weight_global_categoricals):
        #     loss_categorical += cross_entropy(logits, target, weight=weight)
        for logits, target in zip(logits_global_categoricals, target_global_categoricals):
            loss_categorical += cross_entropy(logits, target)

    # Player (2x)

    if logits_player_continuous.numel():
        target_player_continuous = next_obs[:, obs_index["player"]["continuous"]]
        loss_continuous += mse_loss(logits_player_continuous, target_player_continuous)

    if logits_player_binary.numel():
        target_player_binary = next_obs[:, obs_index["player"]["binary"]]
        # weight_player_binary = loss_weights["binary"]["player"]
        # loss_binary += binary_cross_entropy_with_logits(logits_player_binary, target_player_binary, pos_weight=weight_player_binary)
        loss_binary += binary_cross_entropy_with_logits(logits_player_binary, target_player_binary)

    # XXX: CrossEntropyLoss expects (B, C, *) input where C=num_classes
    #      => transpose (B, 2, C) => (B, C, 2)
    #      (not needed for BCE or MSE)
    # See difference:
    # [cross_entropy(logits, target).item(), cross_entropy(logits.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1)).item(), cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2)).item()]

    if logits_player_categoricals:
        target_player_categoricals = [next_obs[:, index] for index in obs_index["player"]["categoricals"]]
        # weight_player_categoricals = loss_weights["categoricals"]["player"]
        # for logits, target, weight in zip(logits_player_categoricals, target_player_categoricals, weight_player_categoricals):
        #     loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2), weight=weight)
        for logits, target in zip(logits_player_categoricals, target_player_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2))

    # Hex (165x)

    if logits_hex_continuous.numel():
        target_hex_continuous = next_obs[:, obs_index["hex"]["continuous"]]
        loss_continuous += mse_loss(logits_hex_continuous, target_hex_continuous)

    if logits_hex_binary.numel():
        target_hex_binary = next_obs[:, obs_index["hex"]["binary"]]
        # weight_hex_binary = loss_weights["binary"]["hex"]
        # loss_binary += binary_cross_entropy_with_logits(logits_hex_binary, target_hex_binary, pos_weight=weight_hex_binary)
        loss_binary += binary_cross_entropy_with_logits(logits_hex_binary, target_hex_binary)

    if logits_hex_categoricals:
        target_hex_categoricals = [next_obs[:, index] for index in obs_index["hex"]["categoricals"]]
        # weight_hex_categoricals = loss_weights["categoricals"]["hex"]
        # for logits, target, weight in zip(logits_hex_categoricals, target_hex_categoricals, weight_hex_categoricals):
        #     loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2), weight=weight)
        for logits, target in zip(logits_hex_categoricals, target_hex_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2))

    return loss_binary, loss_continuous, loss_categorical


def compute_loss_weights(stats, device):
    weights = {
        "binary": {
            "global": torch.tensor(0., device=device),
            "player": torch.tensor(0., device=device),
            "hex": torch.tensor(0., device=device)
        },
        "categoricals": {
            "global": [],
            "player": [],
            "hex": []
        },
    }

    # NOTE: Clamping weights to prevent huge weights for binaries
    # which are never positive (e.g. SLEEPING stack flag)
    for type in weights["binary"].keys():
        s = stats.binary[type]
        if len(s) == 0:
            continue
        num_positives = s[:, 0]
        num_negatives = s[:, 1] - num_positives
        pos_weights = num_negatives / num_positives
        weights["binary"][type] = pos_weights.clamp(max=100)

    # NOTE: Computing weights only for labels that have appeared
    # to prevent huge weights for labels which never occur
    # (e.g. hex.IS_REAR) from making the other weights very small
    for type, cat_weights in weights["categoricals"].items():
        for cat_stats in stats.categoricals[type]:
            w = torch.zeros(cat_stats.shape, dtype=torch.float32, device=device)
            mask = cat_stats > 0
            masked_stats = cat_stats[mask].float()
            w[mask] = masked_stats.mean() / masked_stats
            cat_weights.append(w.clamp(max=100))

    return weights


def train_model(
    logger,
    model,
    optimizer,
    scaler,
    buffer,
    stats,
    loss_weights,
    epochs,
    batch_size,
    accumulate_grad,
):
    model.train()
    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    total_losses = []
    timer = Timer()

    maybe_autocast = torch.amp.autocast(model.device.type) if scaler else contextlib.nullcontext()

    assert buffer.capacity % batch_size == 0, f"{buffer.capacity} % {batch_size} == 0"
    grad_steps = buffer.capacity // batch_size
    # grad_step = 0
    assert grad_steps > 0

    for epoch in range(epochs):
        timer.start()
        for batch in buffer.sample_iter(batch_size):
            timer.stop()
            obs, action, next_obs, next_mask, next_rew, next_done = batch

            with maybe_autocast:
                pred_obs = model(obs, action)
                loss_cont, loss_bin, loss_cat = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
                loss_tot = loss_cont + loss_bin + loss_cat

            continuous_losses.append(loss_cont.item())
            binary_losses.append(loss_bin.item())
            categorical_losses.append(loss_cat.item())
            total_losses.append(loss_tot.item())

            if accumulate_grad:
                if scaler:
                    scaler.scale(loss_tot / grad_steps).backward()
                else:
                    (loss_tot / grad_steps).backward()
            else:
                if scaler:
                    scaler.scale(loss_tot).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_tot.backward()
                    optimizer.step()
                optimizer.zero_grad()

            timer.start()
        timer.stop()

        if accumulate_grad:
            # assert grad_step == 0, "Sample waste: %d sample batches"
            # Update once after the entire buffer is exhausted
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        continuous_loss = sum(continuous_losses) / len(continuous_losses)
        binary_loss = sum(binary_losses) / len(binary_losses)
        categorical_loss = sum(categorical_losses) / len(categorical_losses)
        total_loss = sum(total_losses) / len(total_losses)
        total_wait = timer.peek()

        return (
            continuous_loss,
            binary_loss,
            categorical_loss,
            total_loss,
            total_wait,
        )


def eval_model(logger, model, loss_weights, buffer, batch_size):
    model.eval()

    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    total_losses = []
    timer = Timer()

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, action, next_obs, next_mask, next_rew, next_done = batch

        with torch.no_grad():
            pred_obs = model(obs, action)

        loss_cont, loss_bin, loss_cat = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
        loss_tot = loss_cont + loss_bin + loss_cat

        continuous_losses.append(loss_cont.item())
        binary_losses.append(loss_bin.item())
        categorical_losses.append(loss_cat.item())
        total_losses.append(loss_tot.item())
        timer.start()
    timer.stop()

    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    binary_loss = sum(binary_losses) / len(binary_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()

    return (
        continuous_loss,
        binary_loss,
        categorical_loss,
        total_loss,
        total_wait
    )


def dig(data, *keys):
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


def aggregate_metrics(queue):
    total = 0
    count = 0

    while not queue.empty():
        item = queue.get()
        total += item
        count += 1

    return total / count if count else None


def timer_stats(timers):
    res = {}
    t_all = timers["all"].peek()
    for k, v in timers.items():
        res[f"timer/{k}"] = v.peek()
        if k != "all":
            res[f"timer_rel/{k}"] = v.peek() / t_all

    res["timer/other"] = t_all - sum(v.peek() for k, v in timers.items() if k != "all")
    res["timer_rel/other"] = res["timer/other"] / t_all
    return res


def train(resume_config, loglevel, dry_run, no_wandb, sample_only):
    if resume_config:
        with open(resume_config, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)

        run_id = config["run"]["id"]
        config["run"]["resumed_config"] = resume_config
    else:
        from .config import config
        run_id = ''.join(random.choices(string.ascii_lowercase, k=8))
        config["run"] = dict(
            id=run_id,
            name=config["name_template"].format(id=run_id, datetime=datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            out_dir=os.path.abspath("data/t10n"),
            resumed_config=None,
        )

    checkpoint_s3_config = dig(config, "s3", "checkpoint")
    train_s3_config = dig(config, "s3", "data", "train")
    eval_s3_config = dig(config, "s3", "data", "eval")
    train_env_config = dig(config, "env", "train")
    eval_env_config = dig(config, "env", "eval")

    train_sample_from_env = train_env_config is not None
    eval_sample_from_env = eval_env_config is not None

    train_sample_from_s3 = (not train_sample_from_env) and train_s3_config is not None
    eval_sample_from_s3 = (not eval_sample_from_env) and eval_s3_config is not None

    train_save_samples = train_sample_from_env and train_s3_config is not None
    eval_save_samples = eval_sample_from_env and eval_s3_config is not None

    train_batch_size = config["train"]["batch_size"]
    eval_batch_size = config["eval"]["batch_size"]

    if train_env_config:
        # Prevent guaranteed waiting time for each batch during training
        assert train_batch_size <= (train_env_config["num_workers"] * train_env_config["batch_size"])
        # Samples would be lost otherwise (batched_iter uses loop with step=batch_size)
        assert (train_env_config["num_workers"] * train_env_config["batch_size"]) % train_batch_size == 0
    else:
        assert train_batch_size <= (train_s3_config["num_workers"] * train_s3_config["batch_size"])
        assert (train_s3_config["num_workers"] * train_s3_config["batch_size"]) % train_batch_size == 0

    if eval_env_config:
        # Samples would be lost otherwise (batched_iter uses loop with step=batch_size)
        assert eval_batch_size <= (eval_env_config["num_workers"] * eval_env_config["batch_size"])
        assert (eval_env_config["num_workers"] * eval_env_config["batch_size"]) % eval_batch_size == 0
    else:
        assert eval_batch_size <= (eval_s3_config["num_workers"] * eval_s3_config["batch_size"])
        assert (eval_s3_config["num_workers"] * eval_s3_config["batch_size"]) % eval_batch_size == 0

    assert config["checkpoint_interval_s"] > config["eval"]["interval_s"]

    os.makedirs(config["run"]["out_dir"], exist_ok=True)

    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        print(f"Saving new config to: {f.name}")
        json.dump(config, f, indent=4)

    logger = StructuredLogger(level=getattr(logging, loglevel), filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"), context=dict(run_id=run_id))
    logger.info(dict(config=config))

    learning_rate = config["train"]["learning_rate"]
    train_epochs = config["train"]["epochs"]

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitionModel(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    optimize_local_storage = config.get("s3", {}).get("optimize_local_storage")

    def make_vcmi_dataloader(cfg, mq):
        return torch.utils.data.DataLoader(
            DatasetVCMI(logger=logger, env_kwargs=cfg["kwargs"], metric_queue=mq, mw_functor=vcmi_dataloader_functor),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
            # persistent_workers=True,  # no effect here
        )

    def make_s3_dataloader(cfg, mq, split_ratio=None, split_side=None):
        return torch.utils.data.DataLoader(
            DatasetS3(
                logger=logger,
                bucket_name=cfg["bucket_name"],
                s3_dir=cfg["s3_dir"],
                cache_dir=cfg["cache_dir"],
                cached_files_max=cfg["cached_files_max"],
                shuffle=cfg["shuffle"],
                split_ratio=split_ratio,
                split_side=split_side,
                aws_access_key=os.environ["AWS_ACCESS_KEY"],
                aws_secret_key=os.environ["AWS_SECRET_KEY"],
                metric_queue=mq,
                # mw_functor=
            ),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
            pin_memory=cfg["pin_memory"]
        )

    train_metric_queue = torch.multiprocessing.Queue()
    eval_metric_queue = torch.multiprocessing.Queue()

    if train_sample_from_env:
        dataloader_obj = make_vcmi_dataloader(train_env_config, train_metric_queue)
    if eval_sample_from_env:
        eval_dataloader_obj = make_vcmi_dataloader(eval_env_config, eval_metric_queue)
    if train_sample_from_s3:
        dataloader_obj = make_s3_dataloader(train_s3_config, train_metric_queue, 0.98, 0)
    if eval_sample_from_s3:
        # eval_dataloader_obj = make_s3_dataloader(eval_s3_config, eval_metric_queue)
        eval_dataloader_obj = make_s3_dataloader(dict(eval_s3_config, s3_dir=train_s3_config["s3_dir"]), eval_metric_queue, 0.98, 1)

    def make_buffer(dloader):
        return Buffer(logger=logger, dataloader=dloader, dim_obs=DIM_OBS, n_actions=N_ACTIONS, device=device)

    buffer = make_buffer(dataloader_obj)
    dataloader = iter(dataloader_obj)
    eval_buffer = make_buffer(eval_dataloader_obj)
    eval_dataloader = iter(eval_dataloader_obj)
    stats = Stats(model, device=device)

    if resume_config:
        load_checkpoint = partial(
            load_local_or_s3_checkpoint,
            logger,
            dry_run,
            checkpoint_s3_config,
            optimize_local_storage,
            device,
            config["run"]["out_dir"],
            run_id,
        )

        load_checkpoint("model", model, strict=True)
        load_checkpoint("optimizer", optimizer)

        if scaler:
            try:
                load_checkpoint("scaler", scaler)
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    logger.warn("WARNING: scaler weights not found (maybe the model was trained on CPU only?)")
                else:
                    raise

    if no_wandb:
        def wandb_log(data, commit=False):
            logger.info(data)
    else:
        wandb = setup_wandb(config, model, __file__)

        def wandb_log(data, commit=False):
            wandb.log(data, commit=commit)
            logger.info(data)

    wandb_log({
        "train/learning_rate": learning_rate,
        "train/buffer_capacity": buffer.capacity,
        "train/epochs": train_epochs,
        "train/batch_size": train_batch_size,
        "eval/buffer_capacity": eval_buffer.capacity,
        "eval/batch_size": eval_batch_size,
    })

    last_checkpoint_at = time.time()
    last_evaluation_at = 0

    # during training, we simply check if the event is set and optionally skip the upload
    # Non-bloking, but uploads may be skipped (checkpoint uploads)
    uploading_event = threading.Event()
    train_uploading_event_buf = threading.Event()
    eval_uploading_event_buf = threading.Event()

    # during sample collection, we use a cond lock to prevent more than 1 upload at a time
    # Blocking, but all uploads are processed (buffer uploads)
    train_uploading_cond = threading.Condition()
    eval_uploading_cond = threading.Condition()

    timers = {
        "all": Timer(),
        "sample": Timer(),
        "train": Timer(),
        "eval": Timer(),
    }

    eval_loss_best = None

    while True:
        timers["sample"].reset()
        timers["train"].reset()
        timers["eval"].reset()

        timers["all"].reset()
        timers["all"].start()

        now = time.time()
        with timers["sample"]:
            buffer.load_samples(dataloader)

        logger.info("Samples loaded: %d" % buffer.capacity)

        assert buffer.full and not buffer.index

        if train_save_samples:
            save_buffer_async(
                run_id=run_id,
                logger=logger,
                dry_run=dry_run,
                buffer=buffer,
                env_config=train_env_config,
                s3_config=train_s3_config,
                allow_skip=not sample_only,
                uploading_cond=train_uploading_cond,
                uploading_event_buf=train_uploading_event_buf,
                optimize_local_storage=optimize_local_storage
            )

        if sample_only:
            stats.iteration += 1
            continue

        loss_weights = compute_loss_weights(stats, device=device)

        wlog = {"iteration": stats.iteration}

        # Evaluate first (for a baseline when resuming with modified params)
        if now - last_evaluation_at > config["eval"]["interval_s"]:
            last_evaluation_at = now

            with timers["sample"]:
                eval_buffer.load_samples(eval_dataloader)

            with timers["eval"]:
                (
                    eval_continuous_loss,
                    eval_binary_loss,
                    eval_categorical_loss,
                    eval_loss,
                    eval_wait
                ) = eval_model(
                    logger=logger,
                    model=model,
                    loss_weights=loss_weights,
                    buffer=eval_buffer,
                    batch_size=eval_batch_size,
                )

            wlog["eval_loss/continuous"] = eval_continuous_loss
            wlog["eval_loss/binary"] = eval_binary_loss
            wlog["eval_loss/categorical"] = eval_categorical_loss
            wlog["eval_loss/total"] = eval_loss
            wlog["eval_dataset/wait_time_s"] = eval_wait

            train_dataset_metrics = aggregate_metrics(train_metric_queue)
            if train_dataset_metrics:
                wlog["train_dataset/avg_worker_utilization"] = train_dataset_metrics

            eval_dataset_metrics = aggregate_metrics(eval_metric_queue)
            if eval_dataset_metrics:
                wlog["eval_dataset/avg_worker_utilization"] = eval_dataset_metrics

            if eval_save_samples:
                save_buffer_async(
                    run_id=run_id,
                    logger=logger,
                    dry_run=dry_run,
                    buffer=eval_buffer,
                    env_config=eval_env_config,
                    s3_config=eval_s3_config,
                    allow_skip=not sample_only,
                    uploading_cond=eval_uploading_cond,
                    uploading_event_buf=eval_uploading_event_buf,
                    optimize_local_storage=optimize_local_storage
                )

            if now - last_checkpoint_at > config["checkpoint_interval_s"]:
                last_checkpoint_at = now

                if eval_loss_best is None:
                    # Initial baseline
                    eval_loss_best = eval_loss
                    logger.info("No baseline for checkpoint yet (eval_loss=%f, eval_loss_best=None), setting it now" % (eval_loss))
                elif eval_loss >= eval_loss_best:
                    logger.info("Bad checkpoint (eval_loss=%f, eval_loss_best=%f), will skip it" % (eval_loss, eval_loss_best))
                else:
                    logger.info("Good checkpoint (eval_loss=%f, eval_loss_best=%f), will save it" % (eval_loss, eval_loss_best))
                    eval_loss_best = eval_loss
                    thread = threading.Thread(target=save_checkpoint, kwargs=dict(
                        logger=logger,
                        dry_run=dry_run,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        out_dir=config["run"]["out_dir"],
                        run_id=run_id,
                        optimize_local_storage=optimize_local_storage,
                        s3_config=config.get("s3", {}).get("checkpoint"),
                        uploading_event=uploading_event
                    ))
                    thread.start()

        with timers["train"]:
            (
                train_continuous_loss,
                train_binary_loss,
                train_categorical_loss,
                train_loss,
                train_wait,
            ) = train_model(
                logger=logger,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                buffer=buffer,
                stats=stats,
                loss_weights=loss_weights,
                epochs=train_epochs,
                batch_size=train_batch_size,
                accumulate_grad=config["train"]["accumulate_grad"],
            )

        wlog["train_loss/continuous"] = train_continuous_loss
        wlog["train_loss/binary"] = train_binary_loss
        wlog["train_loss/categorical"] = train_categorical_loss
        wlog["train_loss/total"] = train_loss
        wlog["train_dataset/wait_time_s"] = train_wait

        if "eval_loss/total" in wlog:
            wlog = dict(wlog, **timer_stats(timers))
            wandb_log(wlog, commit=True)
        else:
            logger.info(wlog)

        # XXX: must log timers here (some may have been skipped)
        stats.iteration += 1


def test(cfg_file):
    from vcmi_gym.envs.v11.vcmi_env import VcmiEnv

    run_id = os.path.basename(cfg_file).removesuffix("-config.json")
    weights_file = f"data/t10n/{run_id}-model.pt"
    model = load_for_test(weights_file)
    env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap", conntype="thread", random_heroes=1, swap_sides=1)
    do_test(model, env)


def load_for_test(file):
    model = TransitionModel()
    model.eval()
    print(f"Loading {file}")
    weights = torch.load(file, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(weights, strict=True)
    return model


def do_test(model, env):
    from vcmi_gym.envs.v11.decoder.decoder import Decoder

    env.reset()
    for _ in range(10):
        print("=" * 100)
        if env.terminated or env.truncated:
            env.reset()
        act = env.random_action()
        obs, rew, term, trunc, _info = env.step(act)

        # [(obs, act, real_obs), (obs, act, real_obs), ...]
        dream = [(obs["transitions"]["observations"][0], obs["transitions"]["actions"][0], None)]

        for i in range(1, len(obs["transitions"]["observations"])):
            obs_prev = obs["transitions"]["observations"][i-1]
            act_prev = obs["transitions"]["actions"][i-1]
            obs_next = obs["transitions"]["observations"][i]
            # mask_next = obs["transitions"]["action_masks"][i]
            # rew_next = obs["transitions"]["rewards"][i]
            # done_next = (term or trunc) and i == len(obs["transitions"]["observations"]) - 1

            obs_pred_raw = model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(act_prev).unsqueeze(0))
            obs_pred_raw = obs_pred_raw[0]
            obs_pred = model.predict(obs_prev, act_prev)
            dream.append((model.predict(*dream[i-1][:2]), obs["transitions"]["actions"][i], obs_next))

            def prepare(state, action, reward, headline):
                import re
                bf = Decoder.decode(state)
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                rewtxt = "" if reward is None else "Reward: %s" % round(reward, 2)
                render = {}
                render["bf_lines"] = bf.render_battlefield()[0][:-1]
                render["bf_len"] = [len(l) for l in render["bf_lines"]]
                render["bf_printlen"] = [len(ansi_escape.sub('', l)) for l in render["bf_lines"]]
                render["bf_maxlen"] = max(render["bf_len"])
                render["bf_maxprintlen"] = max(render["bf_printlen"])
                render["bf_lines"].insert(0, rewtxt.ljust(render["bf_maxprintlen"]))
                render["bf_printlen"].insert(0, len(render["bf_lines"][0]))
                render["bf_lines"].insert(0, headline)
                render["bf_printlen"].insert(0, len(render["bf_lines"][0]))
                render["bf_lines"] = [l + " "*(render["bf_maxprintlen"] - pl) for l, pl in zip(render["bf_lines"], render["bf_printlen"])]
                render["bf_lines"].append(env.__class__.action_text(action, bf=bf).rjust(render["bf_maxprintlen"]))
                return render["bf_lines"]

            lines_prev = prepare(obs_prev, act_prev, None, "Start:")
            lines_real = prepare(obs_next, -1, None, "Real:")
            lines_pred = prepare(obs_pred, -1, None, "Predicted:")

            losses = compute_losses(
                logger=None,
                obs_index=model.abs_index,
                loss_weights=None,
                next_obs=torch.as_tensor(obs_next).unsqueeze(0),
                pred_obs=obs_pred_raw.unsqueeze(0),
            )

            print("Losses | Obs: binary=%.4f, cont=%.4f, categorical=%.4f" % losses)

            # print(Decoder.decode(obs_prev).render(0))
            # for i in range(len(bfields)):
            print("")
            print("\n".join([(" ".join(rowlines)) for rowlines in zip(lines_prev, lines_real, lines_pred)]))
            print("")

            # bf_next = Decoder.decode(obs_next)
            # bf_pred = Decoder.decode(obs_pred)

            # print(env.render_transitions())
            # print("Predicted:")
            # print(bf_pred.render(0))
            # print("Real:")
            # print(bf_next.render(0))

            # hex20_pred.stack.QUEUE.raw

            # def action_str(obs, a):
            #     if a > 1:
            #         bf = Decoder.decode(obs)
            #         hex = bf.get_hex((a - 2) // len(HEX_ACT_MAP))
            #         act = list(HEX_ACT_MAP)[(a - 2) % len(HEX_ACT_MAP)]
            #         return "%s (y=%s x=%s)" % (act, hex.Y_COORD.v, hex.X_COORD.v)
            #     else:
            #         assert a == 1
            #         return "Wait"

        if len(dream) > 2:
            print(" ******** SEQUENCE: ********** ")
            print(env.render_transitions(add_regular_render=False))
            print(" ******** DREAM: ********** ")
            rcfg = env.reward_cfg._replace(step_fixed=0)
            for i, (obs, act, obs_real) in enumerate(dream):
                print("*" * 10)
                if i == 0:
                    print("Start:")
                    print(Decoder.decode(obs).render(act))
                else:
                    bf_real = Decoder.decode(obs_real)
                    bf = Decoder.decode(obs)
                    print(f"Real step #{i}:")
                    print(bf_real.render(act))
                    print("")
                    print(f"Dream step #{i}:")
                    print(bf.render(act))
                    print(f"Real / Dream rewards: {env.calc_reward(0, bf_real, rcfg)} / {env.calc_reward(0, bf, rcfg)}:")

    # print(env.render_transitions())

    # print("Pred:")
    # print(Decoder.decode(obs_pred))
    # print("Real:")
    # print(Decoder.decode(obs_real))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
    parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument("--loglevel", metavar="LOGLEVEL", default="INFO", help="DEBUG | INFO | WARN | ERROR")
    parser.add_argument('action', metavar="ACTION", type=str, help="train | test | sample")
    args = parser.parse_args()

    if args.dry_run:
        args.no_wandb = True

    if args.action == "test":
        test(args.f)
    elif args.action == "train":
        train(args.f, args.loglevel, args.dry_run, args.no_wandb, False)
    elif args.action == "sample":
        train(args.f, args.loglevel, args.dry_run, args.no_wandb, True)
