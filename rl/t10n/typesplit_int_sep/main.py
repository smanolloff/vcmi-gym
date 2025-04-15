import os
import torch
import torch.nn as nn
import random
import string
import json
import boto3
import botocore.exceptions
import threading
import logging
import shutil
import argparse
import math
import time
import numpy as np
import pathlib
import enum
import contextlib
from datetime import datetime
from functools import partial
from boto3.s3.transfer import TransferConfig

from torch.nn.functional import mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy

from ..constants_v10 import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    STATE_HEXES_INDEX_START,
    HEX_MASK_INDEX_START,
    HEX_MASK_INDEX_END,
)

from ..util.vcmidataset import VCMIDataset
from ..util.s3dataset import S3Dataset
from ..util.timer import Timer

DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


class Other(enum.IntEnum):
    CAN_WAIT = 0
    DONE = enum.auto()

    _count = enum.auto()


def wandb_log(*args, **kwargs):
    pass


def setup_wandb(config, model, src_file):
    import wandb

    resumed = config["run"]["resumed_config"] is not None

    wandb.init(
        project="vcmi-gym",
        group="transition-model",
        name=config["run"]["name"],
        id=config["run"]["id"],
        resume="must" if resumed else "never",
        # resume="allow",  # XXX: reuse id for insta-failed runs
        config=config,
        sync_tensorboard=False,
        save_code=False,  # code saved manually below
        allow_val_change=resumed,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )

    # https://docs.wandb.ai/ref/python/run#log_code
    # XXX: "path" is relative to `root`
    #      but args.cfg_file is relative to vcmi-gym ROOT dir
    src_file = pathlib.Path(src_file)

    def code_include_fn(path):
        p = pathlib.Path(path).absolute()
        return p.samefile(src_file)

    wandb.run.log_code(root=src_file.parent, include_fn=code_include_fn)
    wandb.watch(model, log="all", log_graph=True, log_freq=1000)
    return wandb


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    for mod in list(layer.modules())[1:]:
        layer_init(mod, gain, bias_const)
    return layer


class Buffer:
    def __init__(self, capacity, dim_obs, n_actions, worker_cutoffs=[], device=torch.device("cpu")):
        self.capacity = capacity
        self.device = device
        self.worker_cutoffs = worker_cutoffs

        self.containers = {
            "obs": torch.zeros((capacity, dim_obs), dtype=torch.float32, device=device),
            "mask": torch.zeros((capacity, n_actions), dtype=torch.float32, device=device),
            "reward": torch.zeros((capacity,), dtype=torch.float32, device=device),
            "done": torch.zeros((capacity,), dtype=torch.float32, device=device),
            "action": torch.zeros((capacity,), dtype=torch.int64, device=device)
        }

        self.index = 0
        self.full = False

    # Using compact version with single obs and mask buffers
    def add(self, obs, mask, reward, done, action):
        self.containers["obs"][self.index] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.containers["mask"][self.index] = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        self.containers["reward"][self.index] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.containers["done"][self.index] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.containers["action"][self.index] = torch.as_tensor(action, dtype=torch.int64, device=self.device)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def add_batch(self, obs, mask, reward, done, action):
        batch_size = obs.shape[0]
        start = self.index
        end = self.index + batch_size

        assert end <= self.capacity, f"{end} <= {self.capacity}"
        assert self.index % batch_size == 0, f"{self.index} % {batch_size} == 0"
        assert self.capacity % batch_size == 0, f"{self.capacity} % {batch_size} == 0"

        self.containers["obs"][start:end] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.containers["mask"][start:end] = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        self.containers["reward"][start:end] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.containers["done"][start:end] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.containers["action"][start:end] = torch.as_tensor(action, dtype=torch.int64, device=self.device)

        self.index = end
        if self.index == self.capacity:
            self.index = 0
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Valid are indices of samples where done=False and cutoff=False
        # (i.e. to ensure obs,next_obs is valid)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        ok_samples = ~self.containers["done"][:max_index - 1].bool()
        ok_samples[self.worker_cutoffs] = False
        valid_indices = torch.nonzero(ok_samples, as_tuple=True)[0]
        sampled_indices = valid_indices[torch.randint(len(valid_indices), (batch_size,), device=self.device)]

        obs = self.containers["obs"][sampled_indices]
        # action_mask = self.containers["mask"][sampled_indices]
        action = self.containers["action"][sampled_indices]
        next_obs = self.containers["obs"][sampled_indices + 1]
        next_mask = self.containers["mask"][sampled_indices + 1]
        next_reward = self.containers["reward"][sampled_indices + 1]
        next_done = self.containers["done"][sampled_indices + 1]

        return obs, action, next_obs, next_mask, next_reward, next_done

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # See note in .sample()
        ok_samples = ~self.containers["done"][:max_index - 1].bool()
        ok_samples[self.worker_cutoffs] = False
        valid_indices = torch.nonzero(ok_samples, as_tuple=True)[0]
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)]

        # The valid indices are < than all indices by `short`
        short = self.capacity - len(shuffled_indices)
        if short:
            shuffled_indices = torch.cat((shuffled_indices, valid_indices[torch.randperm(len(valid_indices), device=self.device)][:short]))

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


class Swap(nn.Module):
    def __init__(self, axis1, axis2):
        super().__init__()
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, x):
        return x.swapaxes(self.axis1, self.axis2)


class TransitionModel(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        self._build_indices()

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
        if self.obs_index["global"]["binary"].numel():
            n_binary_features = len(self.obs_index["global"]["binary"])
            self.encoder_global_binary = nn.LazyLinear(n_binary_features)
            # No nonlinearity needed

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        self.encoders_global_categoricals = nn.ModuleList([])
        for ind in self.global_index["categoricals"]:
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
        if self.obs_index["player"]["binary"].numel():
            n_binary_features = len(self.obs_index["player"]["binary"][0])
            self.encoder_player_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        self.encoders_player_categoricals = nn.ModuleList([])
        for ind in self.player_index["categoricals"]:
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
        if self.obs_index["hex"]["binary"].numel():
            n_binary_features = len(self.obs_index["hex"]["binary"][0])
            self.encoder_hex_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per hex:
        # [(B, C1), (B, C2), ...]
        self.encoders_hex_categoricals = nn.ModuleList([])
        for ind in self.hex_index["categoricals"]:
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

        # => (B, Z_AGG)
        self.head_other = nn.LazyLinear(Other._count)

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

        global_continuous_in = obs[:, self.obs_index["global"]["continuous"]]
        global_binary_in = obs[:, self.obs_index["global"]["binary"]]
        global_categorical_ins = [obs[:, ind] for ind in self.obs_index["global"]["categoricals"]]
        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_binary_z = self.encoder_global_binary(global_binary_in)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_merged = torch.cat((action_z, global_continuous_z, global_binary_z, global_categorical_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_continuous_in = obs[:, self.obs_index["player"]["continuous"]]
        player_binary_in = obs[:, self.obs_index["player"]["binary"]]
        player_categorical_ins = [obs[:, ind] for ind in self.obs_index["player"]["categoricals"]]
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_binary_z = self.encoder_player_binary(player_binary_in)
        player_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 2, -1), player_continuous_z, player_binary_z, player_categorical_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_continuous_in = obs[:, self.obs_index["hex"]["continuous"]]
        hex_binary_in = obs[:, self.obs_index["hex"]["binary"]]
        hex_categorical_ins = [obs[:, ind] for ind in self.obs_index["hex"]["categoricals"]]
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

        other_out = self.head_other(z_agg)
        # => (B, Other._count)

        # obs, rew, can_wait
        return obs_out, other_out

    def reconstruct(self, obs_out):
        global_continuous_out = obs_out[:, self.obs_index["global"]["continuous"]]
        global_binary_out = obs_out[:, self.obs_index["global"]["binary"]]
        global_categorical_outs = [obs_out[:, ind] for ind in self.obs_index["global"]["categoricals"]]
        player_continuous_out = obs_out[:, self.obs_index["player"]["continuous"]]
        player_binary_out = obs_out[:, self.obs_index["player"]["binary"]]
        player_categorical_outs = [obs_out[:, ind] for ind in self.obs_index["player"]["categoricals"]]
        hex_continuous_out = obs_out[:, self.obs_index["hex"]["continuous"]]
        hex_binary_out = obs_out[:, self.obs_index["hex"]["binary"]]
        hex_categorical_outs = [obs_out[:, ind] for ind in self.obs_index["hex"]["categoricals"]]
        next_obs = torch.zeros_like(obs_out)

        next_obs[:, self.obs_index["global"]["continuous"]] = torch.clamp(global_continuous_out, 0, 1)
        next_obs[:, self.obs_index["global"]["binary"]] = torch.sigmoid(global_binary_out).round()
        for ind, out in zip(self.obs_index["global"]["categoricals"], global_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.obs_index["player"]["continuous"]] = torch.clamp(player_continuous_out, 0, 1)
        next_obs[:, self.obs_index["player"]["binary"]] = torch.sigmoid(player_binary_out).round()
        for ind, out in zip(self.obs_index["player"]["categoricals"], player_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.obs_index["hex"]["continuous"]] = torch.clamp(hex_continuous_out, 0, 1)
        next_obs[:, self.obs_index["hex"]["binary"]] = torch.sigmoid(hex_binary_out).round()
        for ind, out in zip(self.obs_index["hex"]["categoricals"], hex_categorical_outs):
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
                obs_pred_logits, other_pred_logits = self.forward(obs, action)
                obs_pred = self.reconstruct(obs_pred_logits)[0].numpy()
            finally:
                self.train(was_training)

            mask_pred = torch.zeros(N_ACTIONS, dtype=bool)
            mask_pred[1] = torch.sigmoid(other_pred_logits[0, Other.CAN_WAIT])
            mask_pred[2:] = torch.as_tensor(obs_pred[STATE_HEXES_INDEX_START:].reshape(165, STATE_SIZE_ONE_HEX)[:, HEX_MASK_INDEX_START:HEX_MASK_INDEX_END].flatten())

            done_pred = torch.sigmoid(other_pred_logits[0, Other.DONE]).item()

            return obs_pred, mask_pred, done_pred

    def _build_indices(self):
        self.global_index = {"continuous": [], "binary": [], "categoricals": []}
        self.player_index = {"continuous": [], "binary": [], "categoricals": []}
        self.hex_index = {"continuous": [], "binary": [], "categoricals": []}

        self._add_indices(GLOBAL_ATTR_MAP, self.global_index)
        self._add_indices(PLAYER_ATTR_MAP, self.player_index)
        self._add_indices(HEX_ATTR_MAP, self.hex_index)

        for index in [self.global_index, self.player_index, self.hex_index]:
            for type in ["continuous", "binary"]:
                index[type] = torch.tensor(index[type], device=self.device)

            index["categoricals"] = [torch.tensor(ind, device=self.device) for ind in index["categoricals"]]

        self._build_obs_indices()

    def _add_indices(self, attr_map, index):
        i = 0

        for attr, (enctype, offset, n, vmax) in attr_map.items():
            length = n
            if enctype.endswith("EXPLICIT_NULL"):
                if not enctype.startswith("CATEGORICAL"):
                    index["binary"].append(i)
                    i += 1
                    length -= 1
            elif enctype.endswith("IMPLICIT_NULL"):
                raise Exception("IMPLICIT_NULL is not supported")
            elif enctype.endswith("MASKING_NULL"):
                raise Exception("MASKING_NULL is not supported")
            elif enctype.endswith("STRICT_NULL"):
                pass
            elif enctype.endswith("ZERO_NULL"):
                pass
            else:
                raise Exception("Unexpected enctype: %s" % enctype)

            t = None
            if enctype.startswith("ACCUMULATING"):
                t = "binary"
            elif enctype.startswith("BINARY"):
                t = "binary"
            elif enctype.startswith("CATEGORICAL"):
                t = "categorical"
            elif enctype.startswith("EXPNORM"):
                t = "continuous"
            elif enctype.startswith("LINNORM"):
                t = "continuous"
            else:
                raise Exception("Unexpected enctype: %s" % enctype)

            if t == "categorical":
                ind = []
                index["categoricals"].append(ind)
                for _ in range(length):
                    ind.append(i)
                    i += 1
            else:
                for _ in range(length):
                    index[t].append(i)
                    i += 1

    # Index for extracting values from (batched) observation
    # This is different than the other indexes:
    # - self.hex_index contains *relative* indexes for 1 hex
    # - self.obs_index["hex"] contains *absolute* indexes for all 165 hexes
    def _build_obs_indices(self):
        t = lambda ary: torch.tensor(ary, dtype=torch.int64, device=self.device)

        # XXX: Discrete (or "noncontinuous") is a combination of binary + categoricals
        #      where for direct extraction from obs
        self.obs_index = {
            "global": {"continuous": t([]), "binary": t([]), "categoricals": [], "categorical": t([]), "discrete": t([])},
            "player": {"continuous": t([]), "binary": t([]), "categoricals": [], "categorical": t([]), "discrete": t([])},
            "hex": {"continuous": t([]), "binary": t([]), "categoricals": [], "categorical": t([]), "discrete": t([])},
        }

        # Global

        if self.global_index["continuous"].numel():
            self.obs_index["global"]["continuous"] = self.global_index["continuous"]

        if self.global_index["binary"].numel():
            self.obs_index["global"]["binary"] = self.global_index["binary"]

        if self.global_index["categoricals"]:
            self.obs_index["global"]["categoricals"] = self.global_index["categoricals"]

        self.obs_index["global"]["categorical"] = torch.cat(tuple(self.obs_index["global"]["categoricals"]), dim=0)

        global_discrete = torch.zeros(0, dtype=torch.int64, device=self.device)
        global_discrete = torch.cat((global_discrete, self.obs_index["global"]["binary"]), dim=0)
        global_discrete = torch.cat((global_discrete, *self.obs_index["global"]["categoricals"]), dim=0)
        self.obs_index["global"]["discrete"] = global_discrete

        # Helper function to reduce code duplication
        # Essentially replaces this:
        # if len(model.player_index["binary"]):
        #     ind = torch.zeros([2, len(model.player_index["binary"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.player_index["binary"] + offset
        #     obs_index["player"]["binary"] = ind
        # if len(model.player_index["continuous"]):
        #     ind = torch.zeros([2, len(model.player_index["continuous"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.player_index["continuous"] + offset
        #     obs_index["player"]["continuous"] = ind
        # if len(model.player_index["categoricals"]):
        #     for cat_ind in model.player_index["categoricals"]:
        #         ind = torch.zeros([2, len(cat_ind)], dtype=torch.int64)
        #         for i in range(2):
        #             offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #             ind[i, :] = cat_ind + offset
        #         obs_index["player"]["categoricals"].append(cat_ind)
        # ...
        # - `indexes` is an array of *relative* indexes for 1 element (e.g. hex)
        def repeating_index(n, base_offset, repeating_offset, indexes):
            if indexes.numel() == 0:
                return torch.zeros([n, 0], dtype=torch.int64, device=self.device)
            ind = torch.zeros([n, len(indexes)], dtype=torch.int64, device=self.device)
            for i in range(n):
                offset = base_offset + i*repeating_offset
                ind[i, :] = indexes + offset

            return ind

        # Players (2)
        repind_players = partial(
            repeating_index,
            2,
            STATE_SIZE_GLOBAL,
            STATE_SIZE_ONE_PLAYER
        )

        self.obs_index["player"]["continuous"] = repind_players(self.player_index["continuous"])
        self.obs_index["player"]["binary"] = repind_players(self.player_index["binary"])
        for cat_ind in self.player_index["categoricals"]:
            self.obs_index["player"]["categoricals"].append(repind_players(cat_ind))

        self.obs_index["player"]["categorical"] = torch.cat(tuple(self.obs_index["player"]["categoricals"]), dim=1)

        player_discrete = torch.zeros([2, 0], dtype=torch.int64, device=self.device)
        player_discrete = torch.cat((player_discrete, self.obs_index["player"]["binary"]), dim=1)
        player_discrete = torch.cat((player_discrete, *self.obs_index["player"]["categoricals"]), dim=1)
        self.obs_index["player"]["discrete"] = player_discrete

        # Hexes (165)
        repind_hexes = partial(
            repeating_index,
            165,
            STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER,
            STATE_SIZE_ONE_HEX
        )

        self.obs_index["hex"]["continuous"] = repind_hexes(self.hex_index["continuous"])
        self.obs_index["hex"]["binary"] = repind_hexes(self.hex_index["binary"])
        for cat_ind in self.hex_index["categoricals"]:
            self.obs_index["hex"]["categoricals"].append(repind_hexes(cat_ind))
        self.obs_index["hex"]["categorical"] = torch.cat(tuple(self.obs_index["hex"]["categoricals"]), dim=1)

        hex_discrete = torch.zeros([165, 0], dtype=torch.int64, device=self.device)
        hex_discrete = torch.cat((hex_discrete, self.obs_index["hex"]["binary"]), dim=1)
        hex_discrete = torch.cat((hex_discrete, *self.obs_index["hex"]["categoricals"]), dim=1)
        self.obs_index["hex"]["discrete"] = hex_discrete


class StructuredLogger:
    def __init__(self, level, filename=None, context={}):
        self.level = level
        self.filename = filename
        self.context = context
        self.info(dict(filename=filename))

        assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        self.level = level

    def log(self, obj):
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds')
        thread_id = np.base_repr(threading.current_thread().ident, 36).lower()
        log_obj = dict(timestamp=timestamp, thread_id=thread_id, **dict(self.context, message=obj))
        # print(yaml.dump(log_obj, sort_keys=False))
        print(json.dumps(log_obj, sort_keys=False))

        if self.filename:
            with open(self.filename, "a+") as f:
                f.write(json.dumps(log_obj) + "\n")

    def debug(self, obj):
        self._level_log(obj, logging.DEBUG, "DEBUG")

    def info(self, obj):
        self._level_log(obj, logging.INFO, "INFO")

    def warn(self, obj):
        self._level_log(obj, logging.WARN, "WARN")

    def warning(self, obj):
        self._level_log(obj, logging.WARNING, "WARNING")

    def error(self, obj):
        self._level_log(obj, logging.ERROR, "ERROR")

    def _level_log(self, obj, level, levelname):
        if self.level > level:
            return
        if isinstance(obj, dict):
            self.log(dict(obj))
        else:
            self.log(dict(string=obj))


def load_samples(logger, dataloader, buffer):
    logger.debug("Loading observations...")

    # This is technically not needed, but is easier to benchmark
    # when the batch sizes for adding and iterating are the same
    assert buffer.index == 0, f"{buffer.index} == 0"

    buffer.full = False
    while not buffer.full:
        buffer.add_batch(*next(dataloader))

    assert buffer.index == 0, f"{buffer.index} == 0"
    logger.debug(f"Loaded {buffer.capacity} observations")


class Stats:
    def __init__(self, model, device):
        # Store [mean, var] for each continuous feature
        # Shape: (N_CONT_FEATURES, 2)
        self.continuous = {
            "global": torch.zeros(*model.global_index["continuous"].shape, 2, device=device),
            "player": torch.zeros(*model.player_index["continuous"].shape, 2, device=device),
            "hex": torch.zeros(*model.hex_index["continuous"].shape, 2, device=device),
        }

        # Store [n_ones, n] for each binary feature
        # Shape: (N_BIN_FEATURES, 2)
        self.binary = {
            "global": torch.zeros(*model.global_index["binary"].shape, 2, dtype=torch.int64, device=device),
            "player": torch.zeros(*model.player_index["binary"].shape, 2, dtype=torch.int64, device=device),
            "hex": torch.zeros(*model.hex_index["binary"].shape, 2, dtype=torch.int64, device=device),
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
            "global": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.global_index["categoricals"]],
            "player": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.player_index["categoricals"]],
            "hex": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.hex_index["categoricals"]],
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
        values_global = obs[:, model.obs_index["global"]["binary"]].round().long()
        self.binary["global"][:, 0] += values_global.sum(0)
        self.binary["global"][:, 1] += np.prod(values_global.shape)

        values_player = obs[:, model.obs_index["player"]["binary"]].flatten(end_dim=1).round().long()
        self.binary["player"][:, 0] += values_player.sum(0)
        self.binary["player"][:, 1] += np.prod(values_player.shape)

        values_hex = obs[:, model.obs_index["hex"]["binary"]].flatten(end_dim=1).round().long()
        self.binary["hex"][:, 0] += values_hex.sum(0)
        self.binary["hex"][:, 1] += np.prod(values_hex.shape)

        for ind, stat in zip(model.obs_index["global"]["categoricals"], self.categoricals["global"]):
            stat.add_(obs[:, ind].round().long().sum(0))

        for ind, stat in zip(model.obs_index["player"]["categoricals"], self.categoricals["player"]):
            stat.add_(obs[:, ind].flatten(end_dim=1).round().long().sum(0))

        for ind, stat in zip(model.obs_index["hex"]["categoricals"], self.categoricals["hex"]):
            stat.add_(obs[:, ind].flatten(end_dim=1).round().long().sum(0))


def compute_other_losses(canwait_pred, canwait_target, done_pred, done_target):
    return (
        binary_cross_entropy_with_logits(canwait_pred, canwait_target),
        binary_cross_entropy_with_logits(done_pred, done_target),
    )


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
    batch_size
):
    model.train()
    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    canwait_losses = []
    done_losses = []
    total_losses = []
    timer = Timer()

    maybe_autocast = torch.amp.autocast(model.device.type) if scaler else contextlib.nullcontext()

    for epoch in range(epochs):
        timer.start()
        for batch in buffer.sample_iter(batch_size):
            timer.stop()
            obs, action, next_obs, next_mask, next_rew, next_done = batch

            with maybe_autocast:
                pred_obs, pred_other = model(obs, action)
                loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, pred_obs)
                loss_canwait, loss_done = compute_other_losses(
                    pred_other[:, Other.CAN_WAIT], next_mask[:, 1],
                    pred_other[:, Other.DONE], next_done
                )
                loss_tot = loss_cont + loss_bin + loss_cat + loss_canwait + loss_done

            continuous_losses.append(loss_cont.item())
            binary_losses.append(loss_bin.item())
            categorical_losses.append(loss_cat.item())
            canwait_losses.append(loss_canwait.item())
            done_losses.append(loss_done.item())
            total_losses.append(loss_tot.item())

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss_tot).backward()
                # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # No clipping, just measuring
                # max_norm = 1.0  # Adjust as needed
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_tot.backward()
                # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # No clipping, just measuring
                # max_norm = 1.0  # Adjust as needed
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            timer.start()
        timer.stop()

        continuous_loss = sum(continuous_losses) / len(continuous_losses)
        binary_loss = sum(binary_losses) / len(binary_losses)
        categorical_loss = sum(categorical_losses) / len(categorical_losses)
        canwait_loss = sum(canwait_losses) / len(canwait_losses)
        done_loss = sum(done_losses) / len(done_losses)
        total_loss = sum(total_losses) / len(total_losses)
        total_wait = timer.peek()

        return (
            continuous_loss,
            binary_loss,
            categorical_loss,
            canwait_loss,
            done_loss,
            total_loss,
            total_wait
        )


def eval_model(logger, model, loss_weights, buffer, batch_size):
    model.eval()

    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    canwait_losses = []
    done_losses = []
    total_losses = []
    timer = Timer()

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, action, next_obs, next_mask, next_rew, next_done = batch
        with torch.no_grad():
            pred_obs, pred_other = model(obs, action)

        loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, pred_obs)
        loss_canwait, loss_done = compute_other_losses(
            pred_other[:, Other.CAN_WAIT], next_mask[:, 1],
            pred_other[:, Other.DONE], next_done
        )

        loss_tot = loss_cont + loss_bin + loss_cat + loss_canwait + loss_done

        continuous_losses.append(loss_cont.item())
        binary_losses.append(loss_bin.item())
        categorical_losses.append(loss_cat.item())
        canwait_losses.append(loss_canwait.item())
        done_losses.append(loss_done.item())
        total_losses.append(loss_tot.item())
        timer.start()
    timer.stop()

    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    binary_loss = sum(binary_losses) / len(binary_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    total_loss = sum(total_losses) / len(total_losses)
    canwait_losses = sum(canwait_losses) / len(canwait_losses)
    done_losses = sum(done_losses) / len(done_losses)
    total_wait = timer.peek()

    return (
        continuous_loss,
        binary_loss,
        categorical_loss,
        canwait_losses,
        done_losses,
        total_loss,
        total_wait
    )


def init_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        region_name="eu-north-1",
        config=botocore.config.Config(connect_timeout=10, read_timeout=30)
    )


def save_checkpoint(logger, dry_run, model, optimizer, scaler, out_dir, run_id, optimize_local_storage, s3_config, uploading_event):
    f_model = os.path.join(out_dir, f"{run_id}-model.pt")
    f_optimizer = os.path.join(out_dir, f"{run_id}-optimizer.pt")
    f_scaler = os.path.join(out_dir, f"{run_id}-scaler.pt")
    msg = dict(
        event="Saving checkpoint...",
        model=f_model,
        optimizer=f_optimizer,
        scaler=f_scaler,
    )

    files = [f_model, f_optimizer]
    if scaler:
        files.append(f_scaler)

    if uploading_event.is_set():
        logger.warn("Still uploading previous checkpoint, will not save this one locally or to S3")
        return

    if dry_run:
        msg["event"] += " (--dry-run)"
        logger.info(msg)
    else:
        logger.info(msg)
        # Prevent corrupted checkpoints if terminated during torch.save

        if optimize_local_storage:
            # Use "...~" as a lockfile
            # While the lockfile exists, the original file is corrupted
            # (i.e. save() was interrupted => S3 download is needed to load())

            # NOTE: bulk create and remove lockfiles to prevent mixing up
            #       different checkpoints when only 1 or 2 files get saved

            pathlib.Path(f"{f_model}~").touch()
            pathlib.Path(f"{f_optimizer}~").touch()
            if scaler:
                pathlib.Path(f"{f_scaler}~").touch()

            torch.save(model.state_dict(), f_model)
            torch.save(optimizer.state_dict(), f_optimizer)
            if scaler:
                torch.save(scaler.state_dict(), f_scaler)

            os.unlink(f"{f_model}~")
            os.unlink(f"{f_optimizer}~")
            if scaler:
                os.unlink(f"{f_scaler}~")
        else:
            # Use temporary files to ensure the original one is always good
            # even if the .save is interrupted
            # NOTE: first save all, then move all, to prevent mixing up
            #       different checkpoints when only 1 or 2 files get saved
            torch.save(model.state_dict(), f"{f_model}.tmp")
            torch.save(optimizer.state_dict(), f"{f_optimizer}.tmp")
            if scaler:
                torch.save(scaler.state_dict(), f"{f_scaler}.tmp")

            shutil.move(f"{f_model}.tmp", f_model)
            shutil.move(f"{f_optimizer}.tmp", f_optimizer)
            if scaler:
                shutil.move(f"{f_scaler}.tmp", f_scaler)

    if not s3_config:
        return

    if uploading_event.is_set():
        logger.warn("Still uploading previous checkpoint, will not upload this one to S3")
        return

    uploading_event.set()
    logger.debug("uploading_event: set")

    bucket = s3_config["bucket_name"]
    s3_dir = s3_config["s3_dir"]
    s3 = init_s3_client()

    files.insert(0, os.path.join(out_dir, f"{run_id}-config.json"))

    try:
        for f in files:
            key = f"{s3_dir}/{os.path.basename(f)}"
            msg = f"Uploading to s3://{bucket}/{key} ..."

            if dry_run:
                logger.info(f"{msg} (--dry-run)")
            else:
                logger.info(msg)
                size_mb = os.path.getsize(f) / 1e6

                if size_mb < 100:
                    logger.debug("Uploading as single chunk")
                    s3.upload_file(f, bucket, key)
                elif size_mb < 1000:  # 1GB
                    logger.debug("Uploding on chunks of 50MB")
                    tc = TransferConfig(multipart_threshold=50 * 1024 * 1024, use_threads=True)
                    s3.upload_file(f, bucket, key, Config=tc)
                else:
                    logger.debug("Uploding on chunks of 500MB")
                    tc = TransferConfig(multipart_threshold=500 * 1024 * 1024, use_threads=True)
                    s3.upload_file(f, bucket, key, Config=tc)

                logger.info(f"Uploaded: s3://{bucket}/{key}")

    finally:
        uploading_event.clear()
        logger.debug("uploading_event: cleared")


# NOTE: this assumes no old observations are left in the buffer
def _save_buffer(
    logger,
    dry_run,
    buffer,
    run_id,
    env_config,
    s3_config,
    uploading_cond,
    uploading_event,
    optimize_local_storage,
    allow_skip=True
):
    # XXX: this is a sub-thread
    # Parent thread has released waits for us to notify via the cond that we have
    # saved the buffer to files, so it can start filling the buffer with new
    # while we are uploading.
    # However, it won't be able to start a new upload until this one finishes.

    # XXX: Saving to tempdir (+deleting afterwards) to prevent disk space issues
    # bufdir = os.path.join(out_dir, "samples", "%s-%d" % (run_id, time.time()))
    # msg = f"Saving buffer to {bufdir}"
    # if dry_run:
    #     logger.info(f"{msg} (--dry-run)")
    # else:
    #     logger.info(msg)

    cache_dir = s3_config["cache_dir"]
    s3_dir = s3_config["s3_dir"]
    bucket = s3_config["bucket_name"]

    # No need to store temp files if we can bail early
    if allow_skip and uploading_event.is_set():
        logger.warn("Still uploading previous buffer, will not upload this one to S3")
        # We must still unblock the main thread
        with uploading_cond:
            logger.debug("Obtained lock (sub-thread); notify_all() ...")
            uploading_cond.notify_all()
        return

    now = time.time_ns() / 1000
    fname = f"transitions-{buffer.containers['obs'].shape[0]}-{now:.0f}.npz"
    s3_path = f"{s3_dir}/{fname}"
    local_path = f"{cache_dir}/{s3_path}"
    msg = f"Saving buffer to {local_path}"
    to_save = {k: v.cpu().numpy() for k, v in buffer.containers.items()}
    to_save["md"] = {"env_config": env_config, "s3_config": s3_config}

    if dry_run:
        logger.info(f"{msg} (--dry-run)")
    else:
        logger.info(msg)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        np.savez_compressed(local_path, **to_save)

    def do_upload():
        s3 = init_s3_client()
        msg = f"Uploading to s3://{bucket}/{s3_path} ..."

        if dry_run:
            logger.info(f"{msg} (--dry-run + sleep(10))")
            time.sleep(10)
        else:
            logger.info(msg)
            s3.upload_file(local_path, bucket, s3_path)

        logger.info(f"Uploaded: s3://{bucket}/{s3_path}")

        if optimize_local_storage and os.path.exists(local_path):
            logger.info(f"Remove {local_path}")
            os.unlink(local_path)

    # Buffer saved to local disk =>
    # Notify parent thread so it can now proceed with collecting new obs in it
    # XXX: this must happen AFTER the buffer is fully dumped to local disk
    logger.debug("Trying to obtain lock for notify (sub-thread)...")
    with uploading_cond:
        logger.debug("Obtained lock (sub-thread); notify_all() ...")
        uploading_cond.notify_all()

    if allow_skip:
        # We will simply skip the upload if another one is still in progress
        # (useful if training while also collecting samples)
        if uploading_event.is_set():
            logger.warn("Still uploading previous buffer, will not upload this one to S3")
            return
        uploading_event.set()
        logger.debug("uploading_event: set")
        do_upload()
        uploading_event.clear()
        logger.debug("uploading_event: cleared")
    else:
        # We will hold the cond lock until we are done with the upload
        # so parent will have to wait before starting us again
        # (useful if collecting samples only)
        logger.debug("Trying to obtain lock for upload (sub-thread)...")
        with uploading_cond:
            logger.debug("Obtained lock; Proceeding with upload (sub-thread) ...")
            do_upload()
            logger.info("Successfully uploaded buffer to s3; releasing lock ...")


def save_buffer_async(
    run_id,
    logger,
    dry_run,
    buffer,
    env_config,
    s3_config,
    allow_skip,
    uploading_cond,
    uploading_event_buf,
    optimize_local_storage
):
    # If a previous upload is still in progress, block here until it finishes
    logger.debug("Trying to obtain lock (main thread)...")
    with uploading_cond:
        logger.debug("Obtained lock (main thread); starting sub-thread...")

        thread = threading.Thread(target=_save_buffer, kwargs=dict(
            logger=logger,
            dry_run=dry_run,
            buffer=buffer,
            # out_dir=config["run"]["out_dir"],
            run_id=run_id,
            env_config=env_config,
            s3_config=s3_config,
            uploading_cond=uploading_cond,
            uploading_event=uploading_event_buf,
            optimize_local_storage=optimize_local_storage,
            allow_skip=allow_skip,
        ))
        thread.start()
        # sub-thread should save the buffer to temp dir and notify us
        logger.debug("Waiting on cond (main thread) ...")
        if not uploading_cond.wait(timeout=10):
            logger.error("Thread for buffer upload did not start properly")
        logger.debug("Notified; releasing lock (main thread) ...")
        uploading_cond.notify_all()


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
            VCMIDataset(logger=logger, env_kwargs=cfg["kwargs"], metric_queue=mq),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
            # persistent_workers=True,  # no effect here
        )

    def make_s3_dataloader(cfg, mq, split_ratio=None, split_side=None):
        return torch.utils.data.DataLoader(
            S3Dataset(
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
        return Buffer(
            capacity=dloader.num_workers * dloader.batch_size,
            # XXX: dirty hack to prevents (obs, obs_next) from different workers
            #       1. assumes dataloader fetches `batch_size` samples from 1 worker
            #           (instead of e.g. round robin worker for each sample)
            #       2. assumes buffer.capacity % dataloader.batch_size == 0
            worker_cutoffs=[i * dloader.batch_size - 1 for i in range(1, dloader.num_workers)],
            dim_obs=DIM_OBS,
            n_actions=N_ACTIONS,
            device=device
        )

    buffer = make_buffer(dataloader_obj)
    dataloader = iter(dataloader_obj)
    eval_buffer = make_buffer(eval_dataloader_obj)
    eval_dataloader = iter(eval_dataloader_obj)
    stats = Stats(model, device=device)

    if resume_config:
        def load_local_or_s3_checkpoint(what, torch_obj, **load_kwargs):
            filename = "%s/%s-%s.pt" % (config["run"]["out_dir"], run_id, what)
            logger.info(f"Load {what} from {filename}")

            if os.path.exists(f"{filename}~"):
                if os.path.exists(filename):
                    msg = f"Lockfile for {filename} still exists => deleting local (corrupted) file"
                    if dry_run:
                        logger.warn(f"{msg} (--dry-run)")
                    else:
                        logger.warn(msg)
                        os.unlink(filename)
                if not dry_run:
                    os.unlink(f"{filename}~")

            # Download is OK even if --dry-run is given (nothing overwritten)
            if checkpoint_s3_config and not os.path.exists(filename):
                logger.debug("Local file does not exist, try S3")

                s3_filename = f"{checkpoint_s3_config['s3_dir']}/{os.path.basename(filename)}"
                logger.info(f"Download s3://{checkpoint_s3_config['bucket_name']}/{s3_filename} ...")

                if os.path.exists(f"{filename}.tmp"):
                    os.unlink(f"{filename}.tmp")
                init_s3_client().download_file(checkpoint_s3_config["bucket_name"], s3_filename, f"{filename}.tmp")
                shutil.move(f"{filename}.tmp", filename)
            torch_obj.load_state_dict(torch.load(filename, weights_only=True, map_location=device), **load_kwargs)

            if not dry_run and not optimize_local_storage:
                backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
                logger.debug(f"Backup resumed model weights as {backname}")
                shutil.copy2(filename, backname)

        load_local_or_s3_checkpoint("model", model, strict=True)
        load_local_or_s3_checkpoint("optimizer", optimizer)

        if scaler:
            try:
                load_local_or_s3_checkpoint("scaler", scaler)
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    logger.warn("WARNING: scaler weights not found (maybe the model was trained on CPU only?)")
                else:
                    raise

    global wandb_log

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

    # Skip saving if eval_loss gets worse
    eval_loss_best = 1e9
    eval_loss = eval_loss_best

    while True:
        timers["sample"].reset()
        timers["train"].reset()
        timers["eval"].reset()

        timers["all"].reset()
        timers["all"].start()

        now = time.time()
        with timers["sample"]:
            load_samples(logger=logger, dataloader=dataloader, buffer=buffer)

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

        with timers["train"]:
            (
                train_continuous_loss,
                train_binary_loss,
                train_categorical_loss,
                train_canwait_loss,
                train_done_loss,
                train_loss,
                train_wait
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
            )

        if now - last_evaluation_at > config["eval"]["interval_s"]:
            last_evaluation_at = now

            with timers["sample"]:
                load_samples(logger=logger, dataloader=eval_dataloader, buffer=eval_buffer)

            with timers["eval"]:
                (
                    eval_continuous_loss,
                    eval_binary_loss,
                    eval_categorical_loss,
                    eval_canwait_loss,
                    eval_done_loss,
                    eval_loss,
                    eval_wait
                ) = eval_model(
                    logger=logger,
                    model=model,
                    loss_weights=loss_weights,
                    buffer=eval_buffer,
                    batch_size=eval_batch_size,
                )

            should_log_to_wandb = True
            wlog = {
                "iteration": stats.iteration,
                "train_loss/continuous": train_continuous_loss,
                "train_loss/binary": train_binary_loss,
                "train_loss/categorical": train_categorical_loss,
                "train_loss/canwait": train_canwait_loss,
                "train_loss/done": train_done_loss,
                "train_loss/total": train_loss,
                "train_dataset/wait_time_s": train_wait,
                "eval_loss/continuous": eval_continuous_loss,
                "eval_loss/binary": eval_binary_loss,
                "eval_loss/categorical": eval_categorical_loss,
                "eval_loss/canwait": eval_canwait_loss,
                "eval_loss/done": eval_done_loss,
                "eval_loss/total": eval_loss,
                "eval_dataset/wait_time_s": eval_wait,
            }

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
                if eval_loss >= eval_loss_best:
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
        else:
            logger.info({
                "iteration": stats.iteration,
                "train_loss/continuous": train_continuous_loss,
                "train_loss/binary": train_binary_loss,
                "train_loss/categorical": train_categorical_loss,
                "train_loss/canwait": train_canwait_loss,
                "train_loss/done": train_done_loss,
                "train_loss/total": train_loss,
                "train_dataset/wait_time_s": train_wait,
            })

        if should_log_to_wandb:
            should_log_to_wandb = False
            wlog = dict(wlog, **timer_stats(timers))
            wandb_log(wlog, commit=True)

        # XXX: must log timers here (some may have been skipped)
        stats.iteration += 1


def test(cfg_file):
    from vcmi_gym.envs.v10.vcmi_env import VcmiEnv

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
    from vcmi_gym.envs.v10.decoder.decoder import Decoder

    for _ in range(10):
        print("=" * 100)
        if env.terminated or env.truncated:
            env.reset()
        action = env.random_action()
        obs, rew, term, trunc, _info = env.step(action)

        for i in range(1, len(obs["transitions"]["observations"])):
            obs_prev = obs["transitions"]["observations"][i-1]
            obs_next = obs["transitions"]["observations"][i]
            mask_next = obs["transitions"]["action_masks"][i]
            # rew_next = obs["transitions"]["rewards"][i]
            done_next = (term or trunc) and i == len(obs["transitions"]["observations"]) - 1

            obs_pred_raw, other_pred_raw = model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(action).unsqueeze(0))
            obs_pred_raw = obs_pred_raw[0]
            other_pred_raw = other_pred_raw[0]
            obs_pred, canwait_pred, done_pred = model.predict(obs_prev, action)
            canwait_pred_raw = other_pred_raw[Other.CAN_WAIT]
            done_pred_raw = other_pred_raw[Other.DONE]

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

            lines_prev = prepare(obs_prev, action, None, "Start:")
            lines_real = prepare(obs_next, -1, None, "Real:")
            lines_pred = prepare(obs_pred, -1, None, "Predicted:")

            losses = compute_losses(None, model.obs_index, None, torch.as_tensor(obs_next).unsqueeze(0), obs_pred_raw.unsqueeze(0))
            other_losses = compute_other_losses(canwait_pred_raw, torch.as_tensor(mask_next[1], dtype=torch.float32), done_pred_raw, torch.as_tensor(done_next, dtype=torch.float32))
            losses += other_losses

            print("Losses | Obs: binary=%.4f, cont=%.4f, categorical=%.4f | CanWait: %.4f | Done: %.4f" % losses)

            # print(Decoder.decode(obs_prev).render(0))
            # for i in range(len(bfields)):
            print("")
            print("\n".join([(" ".join(rowlines)) for rowlines in zip(lines_prev, lines_real, lines_pred)]))
            print("")
            # import ipdb; ipdb.set_trace()  # noqa

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

    # # XXXX: TEST
    # device = torch.device("cpu")
    # model = TransitionModel(device=device)
    # next_obs = model.reconstruct(torch.randn([2, DIM_OBS], device=device))
    # pred_obs = model.forward(next_obs, torch.tensor([1, 1], device=device))
    # compute_losses(
    #     None,
    #     model.obs_index,
    #     compute_loss_weights(Stats(model, device=device), device=device),
    #     next_obs,
    #     pred_obs
    # )
    # assert 0
    # # XXXXX: EOF: TEST

    if args.dry_run:
        args.no_wandb = True

    if args.action == "test":
        test(args.f)
    elif args.action == "train":
        train(args.f, args.loglevel, args.dry_run, args.no_wandb, False)
    elif args.action == "sample":
        train(args.f, args.loglevel, args.dry_run, args.no_wandb, True)
