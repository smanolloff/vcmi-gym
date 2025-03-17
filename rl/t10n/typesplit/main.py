import os
import torch
import torch.nn as nn
import random
import string
import json
import yaml
import shutil
import argparse
import time
import numpy as np
import pathlib
from datetime import datetime
from functools import partial

from torch.nn.functional import mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from torch.nn.functional import one_hot

from vcmi_gym.envs.v8.vcmi_env import VcmiEnv
from vcmi_gym.envs.v8.pyprocconnector import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
    STATE_SIZE,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
)

from ..util.s3dataset import S3Dataset


def wandb_log(*args, **kwargs):
    pass


def setup_wandb(logger, config, model, src_file):
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

        self.obs_buffer = torch.empty((capacity, dim_obs), dtype=torch.float32, device=self.device)
        self.mask_buffer = torch.empty((capacity, n_actions), dtype=torch.float32, device=self.device)
        self.done_buffer = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.empty((capacity,), dtype=torch.int64, device=self.device)
        self.reward_buffer = torch.empty((capacity,), dtype=torch.float32, device=self.device)

        self.index = 0
        self.full = False

    # Using compact version with single obs and mask buffers
    # def add(self, obs, action_mask, done, action, reward, next_obs, next_action_mask, next_done):
    def add(self, obs, action_mask, done, action, reward):
        self.obs_buffer[self.index] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.mask_buffer[self.index] = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        self.done_buffer[self.index] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.action_buffer[self.index] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        self.reward_buffer[self.index] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def add_batch(self, obs, mask, done, action, reward):
        batch_size = obs.shape[0]
        start = self.index
        end = self.index + batch_size

        assert end <= self.capacity, f"{end} <= {self.capacity}"
        assert self.index % batch_size == 0, f"{self.index} % {batch_size} == 0"
        assert self.capacity % batch_size == 0, f"{self.capacity} % {batch_size} == 0"

        self.obs_buffer[start:end] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.mask_buffer[start:end] = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        self.done_buffer[start:end] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.action_buffer[start:end] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        self.reward_buffer[start:end] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

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
        reward = self.reward_buffer[sampled_indices]
        next_obs = self.obs_buffer[sampled_indices + 1]
        next_action_mask = self.mask_buffer[sampled_indices + 1]
        next_done = self.done_buffer[sampled_indices + 1]

        return obs, action, reward, next_obs, next_action_mask, next_done

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1].bool(), as_tuple=True)[0]
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)]

        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            yield (
                self.obs_buffer[batch_indices],
                self.action_buffer[batch_indices],
                self.reward_buffer[batch_indices],
                self.obs_buffer[batch_indices + 1],
                self.mask_buffer[batch_indices + 1],
                self.done_buffer[batch_indices + 1]
            )

    def save(self, out_dir, metadata):
        if os.path.exists(out_dir):
            print(f"WARNINNG: dir {out_dir} already exists, will NOT save this buffer")
            return False

        os.makedirs(out_dir, exist_ok=True)

        md = dict(metadata)
        md["shapes"] = dict(
            created_at=int(time.time()),
            capacity=self.capacity,
            shapes={}
        )

        for type in ["obs", "mask", "done", "action", "reward"]:
            fname = os.path.join(out_dir, f"{type}.npz")
            buf = getattr(self, f"{type}_buffer")
            np.savez_compressed(fname, buf)
            md["shapes"][type] = list(buf.shape)

        with open(os.path.join(out_dir, "metadata.json"), "w") as mdfile:
            json.dump(md, mdfile)

        return True


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
        # Global
        #

        if self.obs_index["global"]["continuous"].numel():
            self.encoder_global_continuous = nn.Sequential(
                # (B, C)
                # nn.LazyBatchNorm1d(),  # XXX: also uncomment running_mean calc in Stats
                nn.LazyLinear(32),
                nn.LeakyReLU(),
            )
        else:
            self.encoder_global_continuous = nn.Identity()

        if self.obs_index["global"]["discrete"].numel():
            self.encoder_global_discrete = nn.Sequential(
                nn.LazyLinear(32),
                nn.LeakyReLU(),
            )
        else:
            self.encoder_global_discrete = nn.Identity()

        #
        # Player
        #

        if self.obs_index["player"]["continuous"].numel():
            self.encoder_player_continuous = nn.Sequential(
                # Swap(1, 2),  # (B, 2, C) -> (B, C, 2)
                # nn.LazyBatchNorm1d(),
                # Swap(1, 2),  # (B, C, 2) -> (B, 2, C)
                nn.LazyLinear(32),
                nn.LeakyReLU(),
            )
        else:
            self.encoder_player_continuous = nn.Identity()

        if self.obs_index["player"]["discrete"].numel():
            self.encoder_player_discrete = nn.Sequential(
                nn.LazyLinear(32),
                nn.LeakyReLU(),
            )
        else:
            self.encoder_player_discrete = nn.Identity()

        #
        # Hex
        #

        if self.hex_index["continuous"].numel():
            self.encoder_hex_continuous = nn.Sequential(
                # Swap(1, 2),  # (B, 165, C) -> (B, C, 165)
                # nn.LazyBatchNorm1d(),
                # Swap(1, 2),  # (B, C, 165) -> (B, 165, C)
                nn.LazyLinear(128),
                nn.LeakyReLU(),
                nn.LazyLinear(256),
                nn.LeakyReLU(),
                nn.LazyLinear(64),
                nn.LeakyReLU(),
            )
        else:
            self.encoder_hex_continuous = nn.Identity()

        if self.obs_index["hex"]["discrete"].numel():
            self.encoder_hex_discrete = nn.Sequential(
                nn.LazyLinear(128),
                nn.LeakyReLU(),
                nn.LazyLinear(256),
                nn.LeakyReLU(),
                nn.LazyLinear(64),
                nn.LeakyReLU(),
            )
        else:
            self.encoder_hex_discrete = nn.Identity()

        self.encoder_merged = nn.Sequential(
            # => (B, N_ACTIONS + 32 + 2*32 + 165*64)
            nn.LazyLinear(1024),
            nn.LeakyReLU(),
            nn.LazyLinear(1024),
        )

        # Global heads

        if self.obs_index["global"]["continuous"].numel():
            self.global_continuous_head = nn.LazyLinear(len(self.global_index["continuous"]))

        if self.global_index["binary"].numel():
            self.global_binary_head = nn.LazyLinear(len(self.global_index["binary"]))

        if self.global_index["categoricals"]:
            self.global_categorical_heads = nn.ModuleList([nn.LazyLinear(len(ind)) for ind in self.global_index["categoricals"]])

        # Player heads

        if self.player_index["continuous"].numel():
            self.player_continuous_head = nn.LazyLinear(len(self.player_index["continuous"]))

        if self.player_index["binary"].numel():
            self.player_binary_head = nn.LazyLinear(len(self.player_index["binary"]))

        if self.player_index["categoricals"]:
            self.player_categorical_heads = nn.ModuleList([nn.LazyLinear(len(ind)) for ind in self.player_index["categoricals"]])

        # Hex heads

        if self.hex_index["continuous"].numel():
            self.hex_continuous_head = nn.LazyLinear(len(self.hex_index["continuous"]))

        if self.hex_index["binary"].numel():
            self.hex_binary_head = nn.LazyLinear(len(self.hex_index["binary"]))

        if self.hex_index["categoricals"]:
            self.hex_categorical_heads = nn.ModuleList([nn.LazyLinear(len(ind)) for ind in self.hex_index["categoricals"]])

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            self(torch.randn([2, STATE_SIZE], device=device), torch.tensor([1, 1], device=device))

        layer_init(self)

    def forward(self, obs, action):
        action_in = one_hot(torch.as_tensor(action), num_classes=N_ACTIONS).to(self.device)

        assert obs.device.type == self.device.type, f"{obs.device.type} == {self.device.type}"

        global_continuous_in = obs[:, self.obs_index["global"]["continuous"]]
        global_discrete_in = obs[:, self.obs_index["global"]["discrete"]]
        player_continuous_in = obs[:, self.obs_index["player"]["continuous"]]
        player_discrete_in = obs[:, self.obs_index["player"]["discrete"]]
        hex_continuous_in = obs[:, self.obs_index["hex"]["continuous"]]
        hex_discrete_in = obs[:, self.obs_index["hex"]["discrete"]]

        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_discrete_z = self.encoder_global_discrete(global_discrete_in)
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_discrete_z = self.encoder_player_discrete(player_discrete_in)
        hex_continuous_z = self.encoder_hex_continuous(hex_continuous_in)
        hex_discrete_z = self.encoder_hex_discrete(hex_discrete_in)

        global_z = torch.cat((global_continuous_z, global_discrete_z), dim=-1)
        # => (B, Zg)
        player_z = torch.cat((player_continuous_z, player_discrete_z), dim=-1)
        # => (B, 2, Zp)
        hex_z = torch.cat((hex_continuous_z, hex_discrete_z), dim=-1)
        # => (B, 165, Zh)

        merged_z = torch.cat((action_in, global_z, player_z.flatten(1), hex_z.flatten(1)), dim=-1)
        z = self.encoder_merged(merged_z)
        # => (B, Z)

        b = obs.shape[0]
        global_continuous_out = torch.zeros([b, 0], device=self.device)
        global_binary_out = torch.zeros([b, 0], device=self.device)
        global_categorical_outs = []
        player_continuous_out = torch.zeros([b, 0], device=self.device)
        player_binary_out = torch.zeros([b, 0], device=self.device)
        player_categorical_outs = []
        hex_continuous_out = torch.zeros([b, 0], device=self.device)
        hex_binary_out = torch.zeros([b, 0], device=self.device)
        hex_categorical_outs = []

        #
        # Global
        #

        global_input = torch.cat((global_z, z), dim=1)

        if self.global_index["continuous"].numel():
            global_continuous_out = self.global_continuous_head(global_input)

        if self.global_index["binary"].numel():
            global_binary_out = self.global_binary_head(global_input)

        if self.global_index["categoricals"]:
            global_categorical_outs = [head(global_input) for head in self.global_categorical_heads]

        #
        # Player
        #

        # Expand "z" for the two players
        z_expanded_for_players = z.unsqueeze(1).expand(z.shape[0], 2, z.shape[1])
        # => (B, 2, Z)

        player_inputs = torch.cat((z_expanded_for_players, player_z), dim=2)
        # => (B, 2, Z+ZP)

        if self.player_index["continuous"].numel():
            player_continuous_out = self.player_continuous_head(player_inputs)
            # => (B, 2, PCAT)

        if self.player_index["binary"].numel():
            player_binary_out = self.player_binary_head(player_inputs)
            # => (B, 2, PBIN)

        if self.player_index["categoricals"]:
            player_categorical_outs = [head(player_inputs) for head in self.player_categorical_heads]
            # => [N, (B, 2, *)]

        #
        # Hex
        #

        # Expand "z" for the 165 hexes
        z_expanded_for_hexes = z.unsqueeze(1).expand(z.shape[0], 165, z.shape[1])
        # => (B, 165, Z)

        hex_inputs = torch.cat((z_expanded_for_hexes, hex_z), dim=2)
        # => (B, 165, Z+ZH)

        if self.hex_index["continuous"].numel():
            hex_continuous_out = self.hex_continuous_head(hex_inputs)
            # => (B, 165, HCAT)

        if self.hex_index["binary"].numel():
            hex_binary_out = self.hex_binary_head(hex_inputs)
            # => (B, 165, HBIN)

        if self.hex_index["categoricals"]:
            hex_categorical_outs = [head(hex_inputs) for head in self.hex_categorical_heads]
            # => [N, (B, 165, C*)] where C* is num_classes (may differ)

        return (
            global_continuous_out,
            global_binary_out,
            global_categorical_outs,
            player_continuous_out,
            player_binary_out,
            player_categorical_outs,
            hex_continuous_out,
            hex_binary_out,
            hex_categorical_outs,
        )

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
            "global": {"continuous": t([]), "binary": t([]), "categoricals": [], "discrete": t([])},
            "player": {"continuous": t([]), "binary": t([]), "categoricals": [], "discrete": t([])},
            "hex": {"continuous": t([]), "binary": t([]), "categoricals": [], "discrete": t([])},
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

        hex_discrete = torch.zeros([165, 0], dtype=torch.int64, device=self.device)
        hex_discrete = torch.cat((hex_discrete, self.obs_index["hex"]["binary"]), dim=1)
        hex_discrete = torch.cat((hex_discrete, *self.obs_index["hex"]["categoricals"]), dim=1)
        self.obs_index["hex"]["discrete"] = hex_discrete


class StructuredLogger:
    def __init__(self, filename):
        self.filename = filename
        self.log(dict(filename=filename))

    def log(self, obj):
        timestamp = datetime.utcnow().isoformat(timespec="milliseconds")
        if isinstance(obj, dict):
            log_obj = dict(timestamp=timestamp, message=obj)
        else:
            log_obj = dict(timestamp=timestamp, message=dict(string=obj))

        print(yaml.dump(log_obj, sort_keys=False))
        with open(self.filename, "a+") as f:
            f.write(json.dumps(log_obj) + "\n")


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

    for i in range(n):
        # Ensure logging on final obs
        progress = round(i / n, 3)
        if progress >= next_progress_report_at:
            next_progress_report_at += progress_report_step
            logger.log(dict(observations_collected=i, progress=progress*100, terms=terms, truncs=truncs))

        action = env.random_action()
        if action is None:
            assert term or trunc
            terms += term
            truncs += trunc
            term = False
            trunc = False
            buffer.add(dict_obs["observation"], dict_obs["action_mask"], True, -1, -1)
            dict_obs, _info = env.reset()
        else:
            next_obs, rew, term, trunc, _info = env.step(action)
            buffer.add(dict_obs["observation"], dict_obs["action_mask"], False, action, rew)
            dict_obs = next_obs

    logger.log(dict(observations_collected=n, progress=100, terms=terms, truncs=truncs))


def load_observations(logger, dataloader, buffer):
    logger.log("Loading observations...")
    buffer.add_batch(*next(dataloader))
    logger.log(f"Loaded {buffer.capacity} observations")


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


def compute_losses(logger, obs_index, loss_weights, obs, logits):
    (
        logits_global_continuous,
        logits_global_binary,
        logits_global_categoricals,
        logits_player_continuous,
        logits_player_binary,
        logits_player_categoricals,
        logits_hex_continuous,
        logits_hex_binary,
        logits_hex_categoricals,
    ) = logits

    loss_continuous = 0
    loss_binary = 0
    loss_categorical = 0

    # Global

    if logits_global_continuous.numel():
        target_global_continuous = obs[:, obs_index["global"]["continuous"]]
        loss_continuous += mse_loss(logits_global_continuous, target_global_continuous)

    if logits_global_binary.numel():
        target_global_binary = obs[:, obs_index["global"]["binary"]]
        weight_global_binary = loss_weights["binary"]["global"]
        loss_binary += binary_cross_entropy_with_logits(logits_global_binary, target_global_binary, pos_weight=weight_global_binary)
        loss_binary += binary_cross_entropy_with_logits(logits_global_binary, target_global_binary)

    if logits_global_categoricals:
        target_global_categoricals = [obs[:, index] for index in obs_index["global"]["categoricals"]]
        weight_global_categoricals = loss_weights["categoricals"]["global"]
        for logits, target, weight in zip(logits_global_categoricals, target_global_categoricals, weight_global_categoricals):
            loss_categorical += cross_entropy(logits, target, weight=weight)
        for logits, target in zip(logits_global_categoricals, target_global_categoricals):
            loss_categorical += cross_entropy(logits, target)

    # Player (2x)

    if logits_player_continuous.numel():
        target_player_continuous = obs[:, obs_index["player"]["continuous"]]
        loss_continuous += mse_loss(logits_player_continuous, target_player_continuous)

    if logits_player_binary.numel():
        target_player_binary = obs[:, obs_index["player"]["binary"]]
        weight_player_binary = loss_weights["binary"]["player"]
        loss_binary += binary_cross_entropy_with_logits(logits_player_binary, target_player_binary, pos_weight=weight_player_binary)
        loss_binary += binary_cross_entropy_with_logits(logits_player_binary, target_player_binary)

    # XXX: CrossEntropyLoss expects (B, C, *) input where C=num_classes
    #      => transpose (B, 2, C) => (B, C, 2)
    #      (not needed for BCE or MSE)
    # See difference:
    # [cross_entropy(logits, target).item(), cross_entropy(logits.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1)).item(), cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2)).item()]

    if logits_player_categoricals:
        target_player_categoricals = [obs[:, index] for index in obs_index["player"]["categoricals"]]
        weight_player_categoricals = loss_weights["categoricals"]["player"]
        for logits, target, weight in zip(logits_player_categoricals, target_player_categoricals, weight_player_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2), weight=weight)
        for logits, target in zip(logits_player_categoricals, target_player_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2))

    # Hex (165x)

    if logits_hex_continuous.numel():
        target_hex_continuous = obs[:, obs_index["hex"]["continuous"]]
        loss_continuous += mse_loss(logits_hex_continuous, target_hex_continuous)

    if logits_hex_binary.numel():
        target_hex_binary = obs[:, obs_index["hex"]["binary"]]
        weight_hex_binary = loss_weights["binary"]["hex"]
        loss_binary += binary_cross_entropy_with_logits(logits_hex_binary, target_hex_binary, pos_weight=weight_hex_binary)
        loss_binary += binary_cross_entropy_with_logits(logits_hex_binary, target_hex_binary)

    if logits_hex_categoricals:
        target_hex_categoricals = [obs[:, index] for index in obs_index["hex"]["categoricals"]]
        weight_hex_categoricals = loss_weights["categoricals"]["hex"]
        for logits, target, weight in zip(logits_hex_categoricals, target_hex_categoricals, weight_hex_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2), weight=weight)
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
    buffer,
    stats,
    loss_weights,
    train_epochs,
    train_batch_size
):
    model.train()

    for epoch in range(train_epochs):
        continuous_losses = []
        binary_losses = []
        categorical_losses = []
        total_losses = []

        for batch in buffer.sample_iter(train_batch_size):
            obs, action, next_rew, next_obs, next_mask, next_done = batch
            logits = model(obs, action)
            loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, logits)

            loss_tot = loss_cont + loss_bin + loss_cat
            continuous_losses.append(loss_cont.item())
            binary_losses.append(loss_bin.item())
            categorical_losses.append(loss_cat.item())
            total_losses.append(loss_tot.item())

            optimizer.zero_grad()
            loss_tot.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # No clipping, just measuring

            max_norm = 1.0  # Adjust as needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        continuous_loss = sum(continuous_losses) / len(continuous_losses)
        binary_loss = sum(binary_losses) / len(binary_losses)
        categorical_loss = sum(categorical_losses) / len(categorical_losses)
        total_loss = sum(total_losses) / len(total_losses)

        logger.log(dict(
            train_epoch=epoch,
            continuous_loss=round(continuous_loss, 6),
            binary_loss=round(binary_loss, 6),
            categorical_loss=round(categorical_loss, 6),
            total_loss=round(total_loss, 6),
            gradient_norm=round(total_norm.item(), 6),
        ))


def eval_model(logger, model, buffer, loss_weights, eval_env_steps):
    model.eval()
    batch_size = eval_env_steps // 10

    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    total_losses = []

    for batch in buffer.sample_iter(batch_size):
        obs, action, next_rew, next_obs, next_mask, next_done = batch
        logits = model(obs, action)
        loss_cont, loss_bin, loss_cat = compute_losses(logger, model.obs_index, loss_weights, next_obs, logits)
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


def train(resume_config, dry_run, no_wandb, sample_only):
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
    sample_from_s3 = config["env"] is None and config["s3data"] is not None
    save_samples = config["env"] is not None and config["s3data"] is not None

    assert config.get("env") or config.get("s3data")

    os.makedirs(config["run"]["out_dir"], exist_ok=True)

    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        print(f"Saving new config to: {f.name}")
        json.dump(config, f, indent=4)

    logger = StructuredLogger(filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"))
    logger.log(dict(config=config))

    lr_start = config["train"]["lr_start"]
    lr_min = config["train"]["lr_min"]
    lr_step_size = config["train"]["lr_step_size"]
    lr_gamma = config["train"]["lr_gamma"]
    buffer_capacity = config["train"]["buffer_capacity"]
    train_epochs = config["train"]["train_epochs"]
    train_batch_size = config["train"]["train_batch_size"]
    eval_env_steps = config["train"]["eval_env_steps"]

    assert buffer_capacity % train_batch_size == 0  # needed for train_steps
    assert eval_env_steps % 10 == 0  # needed for eval batch_size

    if sample_from_env:
        env = VcmiEnv(**config["env"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitionModel(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    buffer = Buffer(capacity=buffer_capacity, dim_obs=STATE_SIZE, n_actions=N_ACTIONS, device=device)

    if sample_from_s3:
        dataloader = torch.utils.data.DataLoader(
            S3Dataset(
                bucket_name=config["s3data"]["bucket_name"],
                s3_prefix=config["s3data"]["s3_prefix"],
                cache_dir=config["s3data"]["cache_dir"],
                aws_access_key=os.environ["AWS_ACCESS_KEY"],
                aws_secret_key=os.environ["AWS_SECRET_KEY"],
                region_name=config["s3data"]["region_name"],
                shuffle=config["s3data"]["shuffle"],
            ),
            batch_size=buffer.capacity,
            num_workers=config["s3data"]["num_workers"],
            prefetch_factor=config["s3data"]["prefetch_factor"],
        )
        dataloader = iter(dataloader)

    stats = Stats(model, device=device)

    if resume_config:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], run_id)
        logger.log(f"Load model weights from {filename}")
        model.load_state_dict(torch.load(filename, weights_only=True), strict=True)
        if not dry_run:
            backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
            logger.log(f"Backup resumed model weights as {backname}")
            shutil.copy2(filename, backname)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], run_id)
        logger.log(f"Load optimizer weights from {filename}")
        optimizer.load_state_dict(torch.load(filename, weights_only=True))
        if not dry_run:
            backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
            logger.log(f"Backup optimizer weights as {backname}")
            shutil.copy2(filename, backname)

        filename = "%s/%s-stats.pt" % (config["run"]["out_dir"], run_id)
        logger.log(f"Load training stats from {filename}")
        stats.load_data(torch.load(filename, weights_only=True))
        if not dry_run:
            backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
            logger.log(f"Backup training stats as {backname}")
            shutil.copy2(filename, backname)

    global wandb_log

    if no_wandb:
        def wandb_log(data, commit=False):
            logger.log(data)
    else:
        wandb = setup_wandb(logger, config, model, __file__)

        def wandb_log(data, commit=False):
            wandb.log(data, commit=commit)
            logger.log(data)

    for _ in range(stats.iteration):
        if scheduler.get_last_lr()[0] <= lr_min:
            break
        scheduler.step()

    wandb_log({
        "params/buffer_capacity": buffer_capacity,
        "params/train_epochs": train_epochs,
        "params/train_batch_size": train_batch_size,
        "params/eval_env_steps": eval_env_steps,
    })

    while True:
        if sample_from_env:
            collect_observations(
                logger=logger,
                env=env,
                buffer=buffer,
                n=buffer.capacity,
                progress_report_steps=0
            )
        elif sample_from_s3:
            load_observations(logger=logger, dataloader=dataloader, buffer=buffer)

        assert buffer.full and not buffer.index

        stats.update(buffer, model)

        if save_samples:
            # NOTE: this assumes no old observations are left in the buffer
            bufdir = os.path.join(config["run"]["out_dir"], "samples", f"v{env.ENV_VERSION}", "%s-%05d" % (run_id, stats.iteration))
            msg = f"Saving samples to {bufdir}"
            if dry_run:
                logger.log(f"{msg} (--dry-run)")
            else:
                logger.log(msg)
                buffer.save(bufdir, dict(run_id=run_id, iteration=stats.iteration))

        if sample_only:
            stats.iteration += 1
            continue

        loss_weights = compute_loss_weights(stats, device=device)

        continuous_loss, binary_loss, categorical_loss, total_loss = eval_model(
            logger=logger,
            model=model,
            buffer=buffer,
            loss_weights=loss_weights,
            eval_env_steps=eval_env_steps,
        )

        wandb_log({
            "iteration": stats.iteration,
            "params/learning_rate": scheduler.get_last_lr()[0],
            "stats/num_samples": stats.num_samples,
            "loss/continuous": continuous_loss,
            "loss/binary": binary_loss,
            "loss/categorical": categorical_loss,
            "loss/total": total_loss,
            # "loss/weights": ???
        }, commit=True)

        train_model(
            logger=logger,
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            stats=stats,
            loss_weights=loss_weights,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size
        )

        f_base = os.path.join(config["run"]["out_dir"], run_id)
        f_model = f"{f_base}-model.pt"
        f_optimizer = f"{f_base}-optimizer.pt"
        f_stats = f"{f_base}-stats.pt"

        msg = dict(
            event="Saving checkpoint...",
            model=f_model,
            optimizer=f_optimizer,
            stats=f_stats
        )

        if dry_run:
            msg["event"] += " (--dry-run)"
            logger.log(msg)
        else:
            logger.log(msg)
            # Prevent corrupted checkpoints if terminated during torch.save
            for f in [f_model, f_optimizer, f_stats]:
                if os.path.exists(f):
                    shutil.copy2(f, f"{f}~")

            torch.save(model.state_dict(), f_model)
            torch.save(optimizer.state_dict(), f_optimizer)
            torch.save(stats.export_data(), f_stats)

        stats.iteration += 1
        if scheduler.get_last_lr()[0] > lr_min:
            scheduler.step()


def test(weights_path):
    from vcmi_gym.envs.v8.decoder.decoder import Decoder, pyconnector

    model = TransitionModel()
    weights = torch.load(weights_path, weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()

    env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap")
    obs_prev = env.result.state.copy()
    bf = Decoder.decode(obs_prev)
    action = bf.hexes[4][13].action(pyconnector.HEX_ACT_MAP["MOVE"]).item()

    obs_pred = model.predict(obs_prev, action)
    obs_real = env.step(action)[0]["observation"]

    # print("*** Predicted: ***")
    print("Loss: %s" % torch.nn.functional.mse_loss(torch.as_tensor(obs_pred), torch.as_tensor(obs_real)))
    render = {"prev": {}, "pred": {}, "real": {}, "combined": {}}

    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def prepare_bf(obs, name, headline):
        render[name] = {}
        render[name]["raw"] = Decoder.decode(obs).render()
        render[name]["lines"] = render[name]["raw"].split("\n")
        render[name]["bf_lines"] = render[name]["lines"][:15]
        render[name]["bf_lines"].insert(0, headline)
        render[name]["bf_len"] = [len(l) for l in render[name]["bf_lines"]]
        render[name]["bf_printlen"] = [len(ansi_escape.sub("", l)) for l in render[name]["bf_lines"]]
        render[name]["bf_maxlen"] = max(render[name]["bf_len"])
        render[name]["bf_maxprintlen"] = max(render[name]["bf_printlen"])
        render[name]["bf_lines"] = [l + " "*(render[name]["bf_maxprintlen"] - pl) for l, pl in zip(render[name]["bf_lines"], render[name]["bf_printlen"])]

    prepare_bf(obs_prev, "prev", "Previous:")
    prepare_bf(obs_real, "real", "Real:")
    prepare_bf(obs_pred.numpy(), "pred", "Predicted:")

    render["combined"]["bf"] = "\n".join("%s → %s%s" % (l1, l2, l3) for l1, l2, l3 in zip(render["prev"]["bf_lines"], render["real"]["bf_lines"], render["pred"]["bf_lines"]))
    print(render["combined"]["bf"])

    print("Pred (all):")
    print(render["pred"]["raw"])
    print("Real (all):")
    print(render["real"]["raw"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
    parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument('action', metavar="ACTION", type=str, help="train | test | sample")
    args = parser.parse_args()

    if args.dry_run:
        args.no_wandb = True

    if args.action == "test":
        test(args.test)
    elif args.action == "train":
        train(args.f, args.dry_run, args.no_wandb, False)
    elif args.action == "sample":
        train(args.f, args.dry_run, args.no_wandb, True)
