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
import tempfile
import argparse
import math
import time
import numpy as np
import pathlib
from datetime import datetime
from functools import partial

from torch.nn.functional import mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from torch.nn.functional import one_hot

from ..constants_v10 import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
)

from ..util.s3dataset import S3Dataset

DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


def wandb_log(*args, **kwargs):
    pass


def setup_wandb(config, model, src_file):
    import wandb

    resumed = config["run"]["resumed_config"] is not None

    wandb.init(
        project="vcmi-gym",
        group="transition-model",
        name="%s-%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), config["run"]["id"]),
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
    def __init__(self, capacity, dim_obs, n_actions, device=torch.device("cpu")):
        self.capacity = capacity
        self.device = device

        self.obs_buffer = torch.empty((capacity, dim_obs), dtype=torch.float32, device=device)
        # self.mask_buffer = torch.empty((capacity, n_actions), dtype=torch.float32, device=device)
        self.done_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.action_buffer = torch.empty((capacity,), dtype=torch.int64, device=device)
        # self.reward_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)

        self.index = 0
        self.full = False

    # Using compact version with single obs and mask buffers
    # def add(self, obs, action_mask, done, action, reward, next_obs, next_action_mask, next_done):
    def add(self, obs, action_mask, done, action):
        self.obs_buffer[self.index] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # self.mask_buffer[self.index] = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        self.done_buffer[self.index] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.action_buffer[self.index] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        # self.reward_buffer[self.index] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    # def add_batch(self, obs, mask, done, action, reward):
    def add_batch(self, obs, action, done):
        batch_size = obs.shape[0]
        start = self.index
        end = self.index + batch_size

        assert end <= self.capacity, f"{end} <= {self.capacity}"
        assert self.index % batch_size == 0, f"{self.index} % {batch_size} == 0"
        assert self.capacity % batch_size == 0, f"{self.capacity} % {batch_size} == 0"

        self.obs_buffer[start:end] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # self.mask_buffer[start:end] = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        self.done_buffer[start:end] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.action_buffer[start:end] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        # self.reward_buffer[start:end] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        self.index = end
        if self.index == self.capacity:
            self.index = 0
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False (episode not ended)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1].bool(), as_tuple=True)[0]
        sampled_indices = valid_indices[torch.randint(len(valid_indices), (batch_size,), device=self.device)]

        obs = self.obs_buffer[sampled_indices]
        # action_mask = self.mask_buffer[sampled_indices]
        action = self.action_buffer[sampled_indices]
        # reward = self.reward_buffer[sampled_indices]
        next_obs = self.obs_buffer[sampled_indices + 1]
        # next_action_mask = self.mask_buffer[sampled_indices + 1]
        # next_done = self.done_buffer[sampled_indices + 1]

        # return obs, action, reward, next_obs, next_action_mask, next_done
        return obs, action, next_obs

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1].bool(), as_tuple=True)[0]
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)]

        # The valid indices are than all indices
        short = self.capacity - len(shuffled_indices)
        if short:
            shuffled_indices = torch.cat((shuffled_indices, valid_indices[torch.randperm(len(valid_indices), device=self.device)][:short]))

        assert len(shuffled_indices) == self.capacity

        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            yield (
                self.obs_buffer[batch_indices],
                self.action_buffer[batch_indices],
                # self.reward_buffer[batch_indices],
                self.obs_buffer[batch_indices + 1],
                # self.mask_buffer[batch_indices + 1],
                # self.done_buffer[batch_indices + 1]
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
        self.encoder_global_continuous = nn.Identity()

        # Binaries per player:
        # (B, n)
        self.encoder_player_binary = nn.Identity()
        if self.obs_index["player"]["binary"].numel():
            n_binary_features = len(self.obs_index["player"]["binary"][0])
            self.encoder_player_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        self.encoder_player_categoricals = nn.ModuleList([])
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
        self.encoder_hex_categoricals = nn.ModuleList([])
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
            nn.TransformerEncoderLayer(d_model=z_size_hex, nhead=4, batch_first=True),
            num_layers=3
        )
        # => (B, 165, Z_HEX)

        #
        # Aggregator
        #

        # (B, Z_GLOBAL + AVG(2*Z_PLAYER) + AVG(165*Z_HEX))
        self.aggregator = nn.Linear(512)
        # => (B, Z_AGG)

        #
        # Heads
        #

        # => (B, Z_AGG)
        self.head_global = nn.Linear(STATE_SIZE_GLOBAL)

        # => (B, 2, Z_AGG + Z_PLAYER)
        self.head_player = nn.Linear(STATE_SIZE_ONE_PLAYER)

        # => (B, 165, Z_AGG + Z_HEX)
        self.head_hex = nn.Linear(STATE_SIZE_ONE_HEX)

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            self(torch.randn([2, DIM_OBS], device=device), torch.tensor([1, 1], device=device))

        layer_init(self)

    def forward(self, obs, action):
        assert obs.device.type == self.device.type, f"{obs.device.type} == {self.device.type}"

        action_z = self.encoder_action(action)

        global_continuous_in = obs[:, self.obs_index["global"]["continuous"]]
        global_binary_in = obs[:, self.obs_index["global"]["binary"]]
        global_categorical_ins = [obs[:, ind] for ind in self.obs_index["global"]["categoricals"]]
        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_binary_z = self.encoder_global_binary(global_binary_in)
        global_categorical_z = torch.cat([enc(x) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)])
        global_merged = torch.cat((action_z, global_continuous_z, global_binary_z, global_categorical_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_continuous_in = obs[:, self.obs_index["player"]["continuous"]]
        player_binary_in = obs[:, self.obs_index["player"]["binary"]]
        player_categorical_ins = [obs[:, ind] for ind in self.obs_index["player"]["categoricals"]]
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_binary_z = self.encoder_player_binary(player_binary_in)
        player_categorical_z = torch.cat([enc(x) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)])
        player_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 2, -1), player_continuous_z, player_binary_z, player_categorical_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_continuous_in = obs[:, self.obs_index["hex"]["continuous"]]
        hex_binary_in = obs[:, self.obs_index["hex"]["binary"]]
        hex_categorical_ins = [obs[:, ind] for ind in self.obs_index["hex"]["categorical"]]
        hex_continuous_z = self.encoder_hex_continuous(hex_continuous_in)
        hex_binary_z = self.encoder_hex_binary(hex_binary_in)
        hex_categorical_z = torch.cat([enc(x) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)])
        hex_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 165, -1), hex_continuous_z, hex_binary_z, hex_categorical_z), dim=-1)
        z_hex = self.encoder_merged_hex(hex_merged)
        z_hex = self.transformer_hex(z_hex)
        # => (B, 165, Z_HEX)

        mean_z_player = z_player.mean(dim=1)
        mean_z_hex = z_hex.mean(dim=1)
        z_agg = self.aggregator(z_global + mean_z_player + mean_z_hex)
        # => (B, Z_AGG)

        #
        # Outputs
        #

        global_out = self.head_global(z_agg)
        # => (B, STATE_SIZE_GLOBAL)

        player_out = self.head_player(z_agg.unsqueeze(-1).expand(-1, 2, -1), z_player)
        # => (B, 2, STATE_SIZE_ONE_PLAYER)

        hex_out = self.head_hex(z_agg.unsqueeze(-1).expand(-1, 165, -1), z_hex)
        # => (B, 165, STATE_SIZE_ONE_HEX)

        obs_out = torch.cat((global_out, player_out.flatten(start_dim=1), hex_out.flatten(start_dim=1)), dim=1)

        return obs_out

    # Predict next obs
    def predict(self, obs, action):
        with torch.no_grad():
            return self._predict(obs, action)

    # private

    def _predict(self, obs, action):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(0)

        (
            global_continuous_out,
            global_binary_out,
            global_categorical_outs,
            player_continuous_out,
            player_binary_out,
            player_categorical_outs,
            hex_continuous_out,
            hex_binary_out,
            hex_categorical_outs,
        ) = self.forward(obs, action)

        next_obs = torch.zeros_like(obs)

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

        return next_obs[0].numpy()

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
    def __init__(self, level, filename):
        self.level = level
        self.filename = filename
        self.info(dict(filename=filename))

        assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        self.level = level

    def log(self, obj):
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds')
        thread_id = np.base_repr(threading.current_thread().ident, 36).lower()
        log_obj = dict(timestamp=timestamp, thread_id=thread_id, message=obj)
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
            self.log(dict(message=dict(string=obj)))


# progress_report_steps=0 => quiet
# progress_report_steps=1 => report 100%
# progress_report_steps=2 => report 50%, 100%
# progress_report_steps=3 => report 33%, 67%, 100%
# ...

def collect_observations(logger, env, buffer, n, progress_report_steps=0):
    if progress_report_steps > 0:
        progress_report_step = 1 / progress_report_steps
    else:
        progress_report_step = float("inf")

    next_progress_report_at = 0
    progress = 0
    terms = 0
    truncs = 0
    term = env.terminated
    trunc = env.truncated
    dict_obs = env.obs
    buffer_index_start = buffer.index
    i = 0

    while i < n:
        # Ensure logging on final obs
        progress = round(i / n, 3)
        if progress >= next_progress_report_at:
            next_progress_report_at += progress_report_step
            logger.debug(dict(observations_collected=i, progress=progress*100, terms=terms, truncs=truncs))

        tr = dict_obs["transitions"]
        for obs, mask, action in zip(tr["observations"], tr["action_masks"], tr["actions"]):
            buffer.add(obs, mask, False, action)
            i += 1

        next_action = env.random_action()
        if next_action is None:
            assert term or trunc

            # The current obs is typically oldest one in the next obs's `transitions`
            # However, the env must be reset here, i.e. the obs's transitions will be blank
            # => add it explicitly

            # terms are OK, but truncs are not predictable
            if term:
                buffer.add(dict_obs["observation"], dict_obs["action_mask"], True, -1)
                i += 1

            terms += term
            truncs += trunc
            term = False
            trunc = False
            dict_obs, _info = env.reset()
        else:
            dict_obs, _rew, term, trunc, _info = env.step(next_action)

    if n == buffer.capacity and buffer_index_start == 0:
        # There may be a few extra samples added due to intermediate states
        buffer.index = 0

    logger.debug(dict(observations_collected=i, progress=100, terms=terms, truncs=truncs))


def load_observations(logger, dataloader, buffer):
    logger.debug("Loading observations...")
    buffer.add_batch(*next(dataloader))
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

    def load_data(self, data):
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


def compute_losses(logger, obs_index, loss_weights, next_obs, pred_obs):
    logits_global_continuous = pred_obs[:, obs_index["global"]["continuous"]]
    logits_global_binary = pred_obs[:, obs_index["global"]["binary"]]
    logits_global_categoricals = pred_obs[:, obs_index["global"]["categorical"]]
    logits_player_continuous = pred_obs[:, obs_index["player"]["continuous"]]
    logits_player_binary = pred_obs[:, obs_index["player"]["binary"]]
    logits_player_categoricals = pred_obs[:, obs_index["player"]["categorical"]]
    logits_hex_continuous = pred_obs[:, obs_index["hex"]["continuous"]]
    logits_hex_binary = pred_obs[:, obs_index["hex"]["binary"]]
    logits_hex_categoricals = pred_obs[:, obs_index["hex"]["categorical"]]

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

    for epoch in range(epochs):
        continuous_losses = []
        binary_losses = []
        categorical_losses = []
        total_losses = []

        for batch in buffer.sample_iter(batch_size):
            # obs, action, next_rew, next_obs, next_mask, next_done = batch
            obs, action, next_obs = batch

            if scaler:
                with torch.amp.autocast(model.device.type):
                    pred_obs = model(obs, action)
                    loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, pred_obs)
                    loss_tot = loss_cont + loss_bin + loss_cat

            else:
                pred_obs = model(obs, action)
                loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, pred_obs)
                loss_tot = loss_cont + loss_bin + loss_cat

            continuous_losses.append(loss_cont.item())
            binary_losses.append(loss_bin.item())
            categorical_losses.append(loss_cat.item())
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

        continuous_loss = sum(continuous_losses) / len(continuous_losses)
        binary_loss = sum(binary_losses) / len(binary_losses)
        categorical_loss = sum(categorical_losses) / len(categorical_losses)
        total_loss = sum(total_losses) / len(total_losses)

        # logger.log(dict(
        #     train_epoch=epoch,
        #     continuous_loss=round(continuous_loss, 6),
        #     binary_loss=round(binary_loss, 6),
        #     categorical_loss=round(categorical_loss, 6),
        #     total_loss=round(total_loss, 6),
        #     gradient_norm=round(total_norm.item(), 6),
        # ))
        return continuous_loss, binary_loss, categorical_loss, total_loss


def eval_model(logger, model, loss_weights, buffer, batch_size):
    model.eval()

    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    total_losses = []

    for batch in buffer.sample_iter(batch_size):
        # obs, action, next_rew, next_obs, next_mask, next_done = batch
        obs, action, next_obs = batch
        with torch.no_grad():
            pred_obs = model(obs, action)

        loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, pred_obs)
        loss_tot = loss_cont + loss_bin + loss_cat

        continuous_losses.append(loss_cont.item())
        binary_losses.append(loss_bin.item())
        categorical_losses.append(loss_cat.item())
        total_losses.append(loss_tot.item())

    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    binary_loss = sum(binary_losses) / len(binary_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    total_loss = sum(total_losses) / len(total_losses)

    return continuous_loss, binary_loss, categorical_loss, total_loss


def init_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        region_name="eu-north-1"
    )


def save_checkpoint(logger, dry_run, model, optimizer, scaler, stats, out_dir, run_id, s3_config, uploading_event):
    f_model = os.path.join(out_dir, f"{run_id}-model.pt")
    f_optimizer = os.path.join(out_dir, f"{run_id}-optimizer.pt")
    f_stats = os.path.join(out_dir, f"{run_id}-stats.pt")
    f_scaler = os.path.join(out_dir, f"{run_id}-scaler.pt")
    msg = dict(
        event="Saving checkpoint...",
        model=f_model,
        optimizer=f_optimizer,
        scaler=f_scaler,
        stats=f_stats,
    )

    # Bail here, before saving to local disk
    # (otherwise we may overwrite files which are currently being uploaded)
    files = [f_model, f_optimizer, f_stats]

    if uploading_event.is_set():
        logger.warn("Still uploading previous checkpoint, will not save this one locally or to S3")
        return

    if scaler:
        files.append(f_scaler)

    if dry_run:
        msg["event"] += " (--dry-run)"
        logger.info(msg)
    else:
        logger.info(msg)
        # Prevent corrupted checkpoints if terminated during torch.save
        for f in files:
            if os.path.exists(f):
                shutil.copy2(f, f"{f}~")

        torch.save(model.state_dict(), f_model)
        torch.save(optimizer.state_dict(), f_optimizer)
        torch.save(stats.export_data(), f_stats)
        if scaler:
            torch.save(scaler.state_dict(), f_scaler)

    if not s3_config:
        return

    uploading_event.set()
    logger.debug("uploading_event: set")

    bucket = s3_config["bucket_name"]
    s3_dir = s3_config["s3_dir"]
    s3 = init_s3_client()

    files.insert(0, os.path.join(out_dir, f"{run_id}-config.json"))

    for f in files:
        key = f"{s3_dir}/{os.path.basename(f)}"
        msg = f"Uploading to s3://{bucket}/{key} ..."

        if dry_run:
            logger.info(f"{msg} (--dry-run)")
        else:
            logger.info(msg)
            try:
                s3.head_object(Bucket=bucket, Key=key)
                s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=f"{key}.bak")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Reraise if it's not a 404 (file not found) error

            s3.upload_file(f, bucket, key)
            logger.debug(f"Upload finished: s3://{bucket}/{key}")

    uploading_event.clear()
    logger.debug("uploading_event: cleared")


# NOTE: this assumes no old observations are left in the buffer
def save_buffer(logger, dry_run, buffer, run_id, s3_config, uploading_cond, uploading_event, allow_skip=True):
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

    s3_dir = s3_config["s3_dir"]
    bucket = s3_config["bucket_name"]

    # [(local_path, s3_path), ...)]
    paths = []

    # No need to store temp files if we can bail early
    if allow_skip and uploading_event.is_set():
        logger.warn("Still uploading previous buffer, will not upload this one to S3")
        # We must still unblock the main thread
        with uploading_cond:
            logger.debug("Obtained lock (sub-thread); notify_all() ...")
            uploading_cond.notify_all()
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        for type in ["obs", "done", "action"]:
            fname = f"{type}-{run_id}-{time.time():.0f}.npz"
            buf = getattr(buffer, f"{type}_buffer")
            local_path = f"{temp_dir}/{fname}"
            msg = f"Saving buffer to {local_path}"
            if dry_run:
                logger.info(f"{msg} (--dry-run)")
            else:
                logger.info(msg)
                np.savez_compressed(local_path, buf)
            s3_path = f"{s3_dir}/{fname}"
            paths.append((local_path, s3_path))

        def do_upload():
            s3 = init_s3_client()

            for local_path, s3_path in paths:
                msg = f"Uploading buffer to s3://{bucket}/{s3_path} ..."

                if dry_run:
                    logger.info(f"{msg} (--dry-run + sleep(10))")
                    time.sleep(10)
                else:
                    logger.info(msg)
                    s3.upload_file(local_path, bucket, s3_path)

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
            out_dir=os.path.abspath("data/t10n"),
            resumed_config=None,
        )

    sample_from_env = config["env"] is not None
    sample_from_s3 = config["env"] is None and config["s3"]["data"] is not None
    save_samples = config["env"] is not None and config["s3"]["data"] is not None

    assert config.get("env") or config.get("s3", {}).get("data")

    os.makedirs(config["run"]["out_dir"], exist_ok=True)

    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        print(f"Saving new config to: {f.name}")
        json.dump(config, f, indent=4)

    logger = StructuredLogger(level=getattr(logging, loglevel), filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"))
    logger.info(dict(config=config))

    lr_start = config["train"]["lr_start"]
    lr_min = config["train"]["lr_min"]
    lr_step_size = config["train"]["lr_step_size"]
    lr_gamma = config["train"]["lr_gamma"]
    buffer_capacity = config["train"]["buffer_capacity"]
    train_epochs = config["train"]["epochs"]
    train_batch_size = config["train"]["batch_size"]

    eval_buffer_capacity = config["eval"]["buffer_capacity"]
    eval_batch_size = config["eval"]["batch_size"]

    assert buffer_capacity % train_batch_size == 0  # needed for train_steps

    if sample_from_env:
        from vcmi_gym.envs.v10.vcmi_env import VcmiEnv
        env = VcmiEnv(**config["env"])

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitionModel(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    buffer = Buffer(capacity=buffer_capacity, dim_obs=DIM_OBS, n_actions=N_ACTIONS, device=device)
    eval_buffer = Buffer(capacity=eval_buffer_capacity, dim_obs=DIM_OBS, n_actions=N_ACTIONS, device=device)

    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    data_split_ratio = 0.9  # train / test

    if sample_from_s3:
        dataloader = iter(torch.utils.data.DataLoader(
            S3Dataset(
                bucket_name=config["s3"]["data"]["bucket_name"],
                s3_dir=config["s3"]["data"]["s3_dir"],
                cache_dir=config["s3"]["data"]["cache_dir"],
                cached_files_max=config["s3"]["data"]["cached_files_max"],
                shuffle=config["s3"]["data"]["shuffle"],
                # Don't store keys in config (will appear in clear text in config.json)
                aws_access_key=os.environ["AWS_ACCESS_KEY"],
                aws_secret_key=os.environ["AWS_SECRET_KEY"],
                split_ratio=data_split_ratio,
                split_side=0
            ),
            batch_size=buffer.capacity,
            num_workers=config["s3"]["data"]["num_workers"],
            prefetch_factor=config["s3"]["data"]["prefetch_factor"],
            pin_memory=config["s3"]["data"]["pin_memory"]
        ))

        if not sample_only:
            eval_dataloader = iter(torch.utils.data.DataLoader(
                S3Dataset(
                    bucket_name=config["s3"]["data"]["bucket_name"],
                    s3_dir=config["s3"]["data"]["s3_dir"],
                    cache_dir=config["s3"]["data"]["cache_dir"],
                    cached_files_max=config["s3"]["data"]["cached_files_max"],
                    shuffle=config["s3"]["data"]["shuffle"],
                    # Don't store keys in config (will appear in clear text in config.json)
                    aws_access_key=os.environ["AWS_ACCESS_KEY"],
                    aws_secret_key=os.environ["AWS_SECRET_KEY"],
                    split_ratio=data_split_ratio,
                    split_side=1
                ),
                batch_size=eval_buffer.capacity,
                num_workers=1,
                prefetch_factor=1,
                pin_memory=config["s3"]["data"]["pin_memory"]
            ))

    stats = Stats(model, device=device)

    if resume_config:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], run_id)
        logger.info(f"Load model weights from {filename}")
        if not os.path.exists(filename):
            logger.debug("Local file does not exist, try S3")
            s3_config = config["s3"]["checkpoint"]
            s3_filename = f"{s3_config['s3_dir']}/{os.path.basename(filename)}"
            logger.info(f"Download s3://{s3_config['bucket_name']}/{s3_filename} ...")
            init_s3_client().download_file(s3_config["bucket_name"], s3_filename, filename)
        model.load_state_dict(torch.load(filename, weights_only=True, map_location=device), strict=True)

        if not dry_run:
            backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
            logger.debug(f"Backup resumed model weights as {backname}")
            shutil.copy2(filename, backname)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], run_id)
        logger.info(f"Load optimizer weights from {filename}")
        if not os.path.exists(filename):
            logger.debug("Local file does not exist, try S3")
            s3_config = config["s3"]["checkpoint"]
            s3_filename = f"{s3_config['s3_dir']}/{os.path.basename(filename)}"
            logger.info(f"Download s3://{s3_config['bucket_name']}/{s3_filename} ...")
            init_s3_client().download_file(s3_config["bucket_name"], s3_filename, filename)
        optimizer.load_state_dict(torch.load(filename, weights_only=True, map_location=device))
        if not dry_run:
            backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
            logger.debug(f"Backup optimizer weights as {backname}")
            shutil.copy2(filename, backname)

        if scaler:
            filename = "%s/%s-scaler.pt" % (config["run"]["out_dir"], run_id)
            if not os.path.exists(filename):
                logger.debug("Local file does not exist, try S3")
                s3_config = config["s3"]["checkpoint"]
                s3_filename = f"{s3_config['s3_dir']}/{os.path.basename(filename)}"
                logger.info(f"Download s3://{s3_config['bucket_name']}/{s3_filename} ...")
                try:
                    init_s3_client().download_file(s3_config["bucket_name"], s3_filename, filename)
                except botocore.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] != "404":
                        logger.debug(f"File does not exist in s3: {s3_config['bucket_name']}/{s3_filename} ...")
                        raise

            if os.path.exists(filename):
                logger.info(f"Load scaler weights from {filename}")
                scaler.load_state_dict(torch.load(filename, weights_only=True, map_location=device))
                if not dry_run:
                    backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
                    logger.debug(f"Backup scaler weights as {backname}")
                    shutil.copy2(filename, backname)
            else:
                logger.warn(f"WARNING: scaler weights not found: {filename}")

        filename = "%s/%s-stats.pt" % (config["run"]["out_dir"], run_id)
        logger.info(f"Load training stats from {filename}")
        if not os.path.exists(filename):
            logger.debug("Local file does not exist, try S3")
            s3_config = config["s3"]["checkpoint"]
            s3_filename = f"{s3_config['s3_dir']}/{os.path.basename(filename)}"
            logger.info(f"Download s3://{s3_config['bucket_name']}/{s3_filename} ...")
            init_s3_client().download_file(s3_config["bucket_name"], s3_filename, filename)
        stats.load_data(torch.load(filename, weights_only=True))
        if not dry_run:
            backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
            logger.debug(f"Backup training stats as {backname}")
            shutil.copy2(filename, backname)

    global wandb_log

    if no_wandb:
        def wandb_log(data, commit=False):
            logger.info(data)
    else:
        wandb = setup_wandb(config, model, __file__)

        def wandb_log(data, commit=False):
            wandb.log(data, commit=commit)
            logger.info(data)

    for _ in range(stats.iteration):
        if scheduler.get_last_lr()[0] <= lr_min:
            break
        scheduler.step()

    wandb_log({
        "train/buffer_capacity": buffer_capacity,
        "train/epochs": train_epochs,
        "train/batch_size": train_batch_size,
        "eval/buffer_capacity": eval_buffer_capacity,
        "eval/batch_size": eval_batch_size,
    })

    last_checkpoint_at = time.time()
    last_evaluation_at = 0

    # during training, we simply check if the event is set and optionally skip the upload
    # Non-bloking, but uploads may be skipped (checkpoint uploads)
    uploading_event = threading.Event()
    uploading_event_buf = threading.Event()

    # during sample collection, we use a cond lock to prevent more than 1 upload at a time
    # Blocking, but all uploads are processed (buffer uploads)
    uploading_cond = threading.Condition()

    while True:
        now = time.time()
        if sample_from_env:
            collect_observations(logger=logger, env=env, buffer=buffer, n=buffer.capacity, progress_report_steps=0)
        elif sample_from_s3:
            load_observations(logger=logger, dataloader=dataloader, buffer=buffer)

        assert buffer.full and not buffer.index

        stats.update(buffer, model)

        if save_samples:
            # If a previous upload is still in progress, block here until it finishes
            logger.debug("Trying to obtain lock (main thread)...")
            with uploading_cond:
                logger.debug("Obtained lock (main thread); starting sub-thread...")

                thread = threading.Thread(target=save_buffer, kwargs=dict(
                    logger=logger,
                    dry_run=dry_run,
                    buffer=buffer,
                    # out_dir=config["run"]["out_dir"],
                    run_id=run_id,
                    s3_config=config.get("s3", {}).get("data"),
                    uploading_cond=uploading_cond,
                    uploading_event=uploading_event_buf,
                    allow_skip=not sample_only
                ))
                thread.start()
                # sub-thread should save the buffer to temp dir and notify us
                logger.debug("Waiting on cond (main thread) ...")
                if not uploading_cond.wait(timeout=10):
                    logger.error("Thread for buffer upload did not start properly")
                logger.debug("Notified; releasing lock (main thread) ...")
                uploading_cond.notify_all()

        if sample_only:
            stats.iteration += 1
            continue

        loss_weights = compute_loss_weights(stats, device=device)
        train_continuous_loss, train_binary_loss, train_categorical_loss, train_total_loss = train_model(
            logger=logger,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            buffer=buffer,
            stats=stats,
            loss_weights=loss_weights,
            epochs=train_epochs,
            batch_size=train_batch_size
        )

        if now - last_evaluation_at > config["eval"]["interval_s"]:
            last_evaluation_at = now

            if sample_from_env:
                collect_observations(logger=logger, env=env, buffer=eval_buffer, n=eval_buffer.capacity, progress_report_steps=0)
            elif sample_from_s3:
                load_observations(logger=logger, dataloader=eval_dataloader, buffer=eval_buffer)

            eval_continuous_loss, eval_binary_loss, eval_categorical_loss, eval_total_loss = eval_model(
                logger=logger,
                model=model,
                loss_weights=loss_weights,
                buffer=eval_buffer,
                batch_size=eval_batch_size,
            )

            wandb_log({
                "iteration": stats.iteration,
                "train_loss/continuous": train_continuous_loss,
                "train_loss/binary": train_binary_loss,
                "train_loss/categorical": train_categorical_loss,
                "train_loss/total": train_total_loss,
                "eval_loss/continuous": eval_continuous_loss,
                "eval_loss/binary": eval_binary_loss,
                "eval_loss/categorical": eval_categorical_loss,
                "eval_loss/total": eval_total_loss,
            }, commit=True)
        else:
            logger.info({
                "iteration": stats.iteration,
                "train_loss/continuous": train_continuous_loss,
                "train_loss/binary": train_binary_loss,
                "train_loss/categorical": train_categorical_loss,
                "train_loss/total": train_total_loss,
                "eval_loss/continuous": eval_continuous_loss,
                "eval_loss/binary": eval_binary_loss,
                "eval_loss/categorical": eval_categorical_loss,
                "eval_loss/total": eval_total_loss,
            })

        if now - last_checkpoint_at > config["s3"]["checkpoint"]["interval_s"]:
            last_checkpoint_at = now
            thread = threading.Thread(target=save_checkpoint, kwargs=dict(
                logger=logger,
                dry_run=dry_run,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                stats=stats,
                out_dir=config["run"]["out_dir"],
                run_id=run_id,
                s3_config=config.get("s3", {}).get("checkpoint"),
                uploading_event=uploading_event
            ))
            thread.start()

        stats.iteration += 1
        if scheduler.get_last_lr()[0] > lr_min:
            scheduler.step()


def test(cfg_file):
    from vcmi_gym.envs.v10.vcmi_env import VcmiEnv
    from vcmi_gym.envs.v10.decoder.decoder import Decoder, pyconnector

    run_id = os.path.basename(cfg_file).removesuffix("-config.json")
    model = TransitionModel()
    weights_file = f"data/t10n/{run_id}-model.pt"
    print(f"Loading {weights_file}")
    weights = torch.load(weights_file, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(weights, strict=True)
    model.eval()

    env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap", conntype="thread")
    obs_prev = env.result.state.copy()
    bf = Decoder.decode(1, obs_prev)
    action = bf.hexes[4][13].action(pyconnector.HEX_ACT_MAP["MOVE"]).item()
    bf = Decoder.decode(action, obs_prev)

    obs_pred = torch.as_tensor(model.predict(obs_prev, action))
    env.step(action)
    obs_real = env.result.intstates[1]
    obs_dirty = obs_pred.clone()

    # print("*** Before preprocessing: ***")
    # print("Loss: %s" % torch.nn.functional.mse_loss(torch.as_tensor(obs_pred), torch.as_tensor(obs_next)))
    # print(Decoder.decode(obs_pred).render())

    model._build_indices()
    obs_pred[model.obs_index["global"]["binary"]] = (obs_pred[model.obs_index["global"]["binary"]] > 0.5).float()
    obs_pred[model.obs_index["global"]["continuous"]] = torch.clamp(obs_pred[model.obs_index["global"]["continuous"]], 0, 1)
    for ind in model.obs_index["global"]["categoricals"]:
        out = obs_pred[ind]
        one_hot = torch.zeros_like(out)
        one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
        obs_pred[ind] = one_hot
    obs_pred[model.obs_index["player"]["binary"]] = (obs_pred[model.obs_index["player"]["binary"]] > 0.5).float()
    obs_pred[model.obs_index["player"]["continuous"]] = torch.clamp(obs_pred[model.obs_index["player"]["continuous"]], 0, 1)
    for ind in model.obs_index["player"]["categoricals"]:
        out = obs_pred[ind]
        one_hot = torch.zeros_like(out)
        one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
        obs_pred[ind] = one_hot
    obs_pred[model.obs_index["hex"]["binary"]] = (obs_pred[model.obs_index["hex"]["binary"]] > 0.5).float()
    obs_pred[model.obs_index["hex"]["continuous"]] = torch.clamp(obs_pred[model.obs_index["hex"]["continuous"]], 0, 1)
    for ind in model.obs_index["hex"]["categoricals"]:
        out = obs_pred[ind]
        one_hot = torch.zeros_like(out)
        one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
        obs_pred[ind] = one_hot

    render = {"dirty": {}, "prev": {}, "pred": {}, "real": {}, "combined": {}}

    def prepare(action, obs, headline):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        render = {}
        render["bf_lines"] = Decoder.decode(action, obs).render_battlefield()[0][:-1]
        render["bf_len"] = [len(l) for l in render["bf_lines"]]
        render["bf_printlen"] = [len(ansi_escape.sub('', l)) for l in render["bf_lines"]]
        render["bf_maxlen"] = max(render["bf_len"])
        render["bf_maxprintlen"] = max(render["bf_printlen"])
        render["bf_lines"].insert(0, headline.rjust(render["bf_maxprintlen"]))
        render["bf_printlen"].insert(0, len(render["bf_lines"][0]))
        render["bf_lines"] = [l + " "*(render["bf_maxprintlen"] - pl) for l, pl in zip(render["bf_lines"], render["bf_printlen"])]
        return render

    # bfields = [prepare(action, state, f"Action: {action}") for action, state in zip(self.result.intactions, self.result.intstates)]

    render["dirty"] = prepare(action, obs_dirty.numpy(), "Dirty:")
    render["prev"] = prepare(action, obs_prev, "Previous:")
    render["real"] = prepare(action, obs_real, "Real:")
    render["pred"] = prepare(action, obs_pred.numpy(), "Predicted:")

    render["combined"]["bf"] = "\n".join("%s → %s%s" % (l1, l2, l3) for l1, l2, l3 in zip(render['prev']['bf_lines'], render['real']['bf_lines'], render['pred']['bf_lines']))
    print(render["combined"]["bf"])

    # print("Dirty (all):")
    # print(render["dirty"]["raw"])
    print("Pred (all):")
    print(render["pred"]["raw"])
    print("Real (all):")
    print(render["real"]["raw"])


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
