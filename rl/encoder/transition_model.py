# from vcmi_gym.envs.v8.vcmi_env import VcmiEnv; env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap")
# from rl.encoder.transition_model import TransitionModel; import torch; weights = torch.load("data/autoencoder/bfyevnzy-model.pt", weights_only=True); model = TransitionModel(); model.load_state_dict(weights, strict=True)
# from vcmi_gym.envs.v8.decoder.decoder import Decoder, pyconnector
# obs = env.result.state.copy()
# action = bf.hexes[4][13].action(pyconnector.HEX_ACT_MAP["MOVE"]).item()
# obs_pred = model.predict(obs, action)
# obs_next, * = env.step(action)
# obs_next = obs_next["observation"]
# torch.nn.functional.mse_loss(torch.as_tensor(obs_next), torch.as_tensor(obs))
# # print(Decoder.decode(obs_pred).render())
# # print(Decoder.decode(obs_next).render())

import os
import torch
import torch.nn as nn
import random
import string
import json
import yaml
import argparse
import numpy as np
from functools import partial

from torch.nn.functional import mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from torch.nn.functional import one_hot
from datetime import datetime

from vcmi_gym.envs.v8.vcmi_env import VcmiEnv
from vcmi_gym.envs.v8.pyprocconnector import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
    STATE_SIZE,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    STATE_SEQUENCE,
    N_ACTIONS,
)


def to_tensor(dict_obs):
    return torch.as_tensor(dict_obs["observation"])


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    for mod in list(layer.modules())[1:]:
        layer_init(mod, gain, bias_const)
    return layer


class Buffer:
    def __init__(self, capacity, dim_obs, n_actions, device="cpu"):
        self.capacity = capacity
        self.device = device

        self.obs_buffer = torch.empty((capacity, dim_obs), dtype=torch.float32, device=device)
        self.mask_buffer = torch.empty((capacity, n_actions), dtype=torch.float32, device=device)
        self.done_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.action_buffer = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.reward_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)

        self.index = 0
        self.full = False

    # Using compact version with single obs and mask buffers
    # def add(self, obs, action_mask, done, action, reward, next_obs, next_action_mask, next_done):
    def add(self, obs, action_mask, done, action, reward):
        self.obs_buffer[self.index] = torch.as_tensor(obs, dtype=torch.float32)
        self.mask_buffer[self.index] = torch.as_tensor(action_mask, dtype=torch.float32)
        self.done_buffer[self.index] = torch.as_tensor(done, dtype=torch.float32)
        self.action_buffer[self.index] = torch.as_tensor(action, dtype=torch.int64)
        self.reward_buffer[self.index] = torch.as_tensor(reward, dtype=torch.float32)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False (episode not ended)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1].bool(), as_tuple=True)[0]
        sampled_indices = valid_indices[torch.randint(len(valid_indices), (batch_size,))]

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
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices))]

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


class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._build_indices()

        self.encoder_global = nn.Sequential(
            nn.LazyLinear(32),
            nn.LeakyReLU(),
        )

        self.encoder_player = nn.Sequential(
            nn.LazyLinear(32),
            nn.LeakyReLU(),
        )

        self.encoder_hex = nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(64),
            nn.LeakyReLU(),
        )

        self.encoder_merged = nn.Sequential(
            # => (B, N_ACTIONS + 32 + 64 + 10560)
            nn.LazyLinear(1024),
            nn.LeakyReLU(),
        )

        # Global heads

        if len(self.global_index["binary"]):
            self.global_binary_head = nn.LazyLinear(len(self.global_index["binary"]))

        if len(self.global_index["continuous"]):
            self.global_continuous_head = nn.LazyLinear(len(self.global_index["continuous"]))

        if len(self.global_index["categoricals"]):
            self.global_categorical_heads = nn.ModuleList([nn.LazyLinear(len(ind)) for ind in self.global_index["categoricals"]])

        # Player heads

        if len(self.player_index["binary"]):
            self.player_binary_head = nn.LazyLinear(len(self.player_index["binary"]))

        if len(self.player_index["continuous"]):
            self.player_continuous_head = nn.LazyLinear(len(self.player_index["continuous"]))

        if len(self.player_index["categoricals"]):
            self.player_categorical_heads = nn.ModuleList([nn.LazyLinear(len(ind)) for ind in self.player_index["categoricals"]])

        # Hex heads

        if len(self.hex_index["binary"]):
            self.hex_binary_head = nn.LazyLinear(len(self.hex_index["binary"]))

        if len(self.hex_index["continuous"]):
            self.hex_continuous_head = nn.LazyLinear(len(self.hex_index["continuous"]))

        if len(self.hex_index["categoricals"]):
            self.hex_categorical_heads = nn.ModuleList([nn.LazyLinear(len(ind)) for ind in self.hex_index["categoricals"]])

        # Init lazy layers
        with torch.no_grad():
            self(torch.randn([1, STATE_SIZE]), torch.tensor([1]))

        layer_init(self)

    def forward(self, obs, action):
        assert STATE_SEQUENCE == ["global", "player", "player", "hexes"]
        delims = [STATE_SIZE_GLOBAL, STATE_SIZE_ONE_PLAYER * 2, STATE_SIZE_ONE_HEX * 165]
        obs_global, obs_players, obs_hexes = torch.split(obs, delims, dim=1)
        obs_players = obs_players.unflatten(1, [2, STATE_SIZE_ONE_PLAYER])
        obs_hexes = obs_hexes.unflatten(1, [165, STATE_SIZE_ONE_HEX])

        oh_action = one_hot(torch.as_tensor(action), num_classes=N_ACTIONS)
        z_global = self.encoder_global(obs_global)
        z_players = self.encoder_player(obs_players)
        z_hexes = self.encoder_hex(obs_hexes)

        merged = torch.cat((oh_action, z_global, z_players.flatten(1), z_hexes.flatten(1)), dim=-1)

        z = self.encoder_merged(merged)
        # => (B, Z)

        global_binary_out = torch.tensor([])
        global_continuous_out = torch.tensor([])
        global_categorical_outs = torch.tensor([])
        player_binary_out = torch.tensor([])
        player_continuous_out = torch.tensor([])
        player_categorical_outs = torch.tensor([])
        hex_binary_out = torch.tensor([])
        hex_continuous_out = torch.tensor([])
        hex_categorical_outs = torch.tensor([])

        #
        # Global
        #

        global_input = torch.cat((z_global, z), dim=1)

        if len(self.global_index["binary"]):
            global_binary_out = self.global_binary_head(global_input)

        if len(self.global_index["continuous"]):
            global_continuous_out = self.global_continuous_head(global_input)

        if len(self.global_index["categoricals"]):
            global_categorical_outs = [head(global_input) for head in self.global_categorical_heads]

        #
        # Player
        #

        # Expand "z" for the two players
        z_expanded_for_players = z.unsqueeze(1).expand(z.shape[0], 2, z.shape[1])
        # => (B, 2, Z)

        player_inputs = torch.cat((z_expanded_for_players, z_players), dim=2)
        # => (B, 2, Z+ZP)

        if len(self.player_index["binary"]):
            player_binary_out = self.player_binary_head(player_inputs)
            # => (B, 2, PBIN)

        if len(self.player_index["continuous"]):
            player_continuous_out = self.player_continuous_head(player_inputs)
            # => (B, 2, PCAT)

        if len(self.player_index["categoricals"]):
            player_categorical_outs = [head(player_inputs) for head in self.player_categorical_heads]
            # => [N, (B, 2, *)]

        #
        # Hex
        #

        # Expand "z" for the 165 hexes
        z_expanded_for_hexes = z.unsqueeze(1).expand(z.shape[0], 165, z.shape[1])
        # => (B, 165, Z)

        hex_inputs = torch.cat((z_expanded_for_hexes, z_hexes), dim=2)
        # => (B, 165, Z+ZH)

        if len(self.hex_index["binary"]):
            hex_binary_out = self.hex_binary_head(hex_inputs)
            # => (B, 165, HBIN)

        if len(self.hex_index["continuous"]):
            hex_continuous_out = self.hex_continuous_head(hex_inputs)
            # => (B, 165, HCAT)

        if len(self.hex_index["categoricals"]):
            hex_categorical_outs = [head(hex_inputs) for head in self.hex_categorical_heads]
            # => [N, (B, 165, C*)] where C* is num_classes (may differ)

        # global_binary_index = self.global_index["binary"].unsqueeze(0).expand(global_binary_out.shape)
        # next_obs.scatter_(1, global_binary_index, global_binary_out)

        return (
            global_binary_out,
            global_continuous_out,
            global_categorical_outs,
            player_binary_out,
            player_continuous_out,
            player_categorical_outs,
            hex_binary_out,
            hex_continuous_out,
            hex_categorical_outs,
        )

    # Predict next obs
    def predict(self, obs, action):
        with torch.no_grad():
            return self._predict(obs, action)

    # private

    def _predict(self, obs, action):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(0)

        (
            global_binary_out,
            global_continuous_out,
            global_categorical_outs,
            player_binary_out,
            player_continuous_out,
            player_categorical_outs,
            hex_binary_out,
            hex_continuous_out,
            hex_categorical_outs,
        ) = self.forward(obs, action)

        next_obs = torch.zeros_like(obs)

        next_obs[:, self.obs_index["global"]["binary"]] = torch.sigmoid(global_binary_out).round()
        next_obs[:, self.obs_index["global"]["continuous"]] = torch.clamp(global_continuous_out, 0, 1)
        for ind, out in zip(self.obs_index["global"]["categoricals"], global_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.obs_index["player"]["binary"]] = torch.sigmoid(player_binary_out).round()
        next_obs[:, self.obs_index["player"]["continuous"]] = torch.clamp(player_continuous_out, 0, 1)
        for ind, out in zip(self.obs_index["player"]["categoricals"], player_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.obs_index["hex"]["binary"]] = torch.sigmoid(hex_binary_out).round()
        next_obs[:, self.obs_index["hex"]["continuous"]] = torch.clamp(hex_continuous_out, 0, 1)
        for ind, out in zip(self.obs_index["hex"]["categoricals"], hex_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        return next_obs[0].numpy()

    def _build_indices(self):
        self.global_index = {"binary": [], "continuous": [], "categoricals": []}
        self.player_index = {"binary": [], "continuous": [], "categoricals": []}
        self.hex_index = {"binary": [], "continuous": [], "categoricals": []}

        self._add_indices(GLOBAL_ATTR_MAP, self.global_index)
        self._add_indices(PLAYER_ATTR_MAP, self.player_index)
        self._add_indices(HEX_ATTR_MAP, self.hex_index)

        for index in [self.global_index, self.player_index, self.hex_index]:
            for type in ["binary", "continuous"]:
                index[type] = torch.tensor(index[type])

            index["categoricals"] = [torch.tensor(ind) for ind in index["categoricals"]]

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
        self.obs_index = {
            "global": {"binary": [], "continuous": [], "categoricals": []},
            "player": {"binary": [], "continuous": [], "categoricals": []},
            "hex": {"binary": [], "continuous": [], "categoricals": []},
        }

        # Global

        if len(self.global_index["binary"]):
            self.obs_index["global"]["binary"] = self.global_index["binary"]

        if len(self.global_index["continuous"]):
            self.obs_index["global"]["continuous"] = self.global_index["continuous"]

        if len(self.global_index["continuous"]):
            self.obs_index["global"]["categoricals"] = self.global_index["categoricals"]

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
            if len(indexes) == 0:
                return []
            ind = torch.zeros([n, len(indexes)], dtype=torch.int64)
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

        self.obs_index["player"]["binary"] = repind_players(self.player_index["binary"])
        self.obs_index["player"]["continuous"] = repind_players(self.player_index["continuous"])
        for cat_ind in self.player_index["categoricals"]:
            self.obs_index["player"]["categoricals"].append(repind_players(cat_ind))

        # Hexes (165)
        repind_hexes = partial(
            repeating_index,
            165,
            STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER,
            STATE_SIZE_ONE_HEX
        )

        self.obs_index["hex"]["binary"] = repind_hexes(self.hex_index["binary"])
        self.obs_index["hex"]["continuous"] = repind_hexes(self.hex_index["continuous"])
        for cat_ind in self.hex_index["categoricals"]:
            self.obs_index["hex"]["categoricals"].append(repind_hexes(cat_ind))


class StructuredLogger:
    def __init__(self, filename):
        self.filename = filename
        self.log(dict(filename=filename))

    def log(self, obj):
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds')
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


def compute_losses(logger, obs_index, obs, logits):
    (
        logits_global_binary,
        logits_global_continuous,
        logits_global_categoricals,
        logits_player_binary,
        logits_player_continuous,
        logits_player_categoricals,
        logits_hex_binary,
        logits_hex_continuous,
        logits_hex_categoricals,
    ) = logits

    loss_binary = 0
    loss_continuous = 0
    loss_categorical = 0

    # Global

    if len(logits_global_binary):
        target_global_binary = obs[:, obs_index["global"]["binary"]]
        loss_binary += binary_cross_entropy_with_logits(logits_global_binary, target_global_binary)

    if len(logits_global_continuous):
        target_global_continuous = obs[:, obs_index["global"]["continuous"]]
        loss_continuous += mse_loss(logits_global_continuous, target_global_continuous)

    if len(logits_global_categoricals):
        target_global_categoricals = [obs[:, index] for index in obs_index["global"]["categoricals"]]
        for logits, target in zip(logits_global_categoricals, target_global_categoricals):
            loss_categorical += cross_entropy(logits, target)

    # Player (2x)

    if len(logits_player_binary):
        target_player_binary = obs[:, obs_index["player"]["binary"]]
        loss_binary += binary_cross_entropy_with_logits(logits_player_binary, target_player_binary)

    if len(logits_player_continuous):
        target_player_continuous = obs[:, obs_index["player"]["continuous"]]
        loss_continuous += mse_loss(logits_player_continuous, target_player_continuous)

    # XXX: CrossEntropyLoss expects (B, C, *) input where C=num_classes
    #      => transpose (B, 2, C) => (B, C, 2)
    #      (not needed for BCE or MSE)
    # See difference:
    # [cross_entropy(logits, target).item(), cross_entropy(logits.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1)).item(), cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2)).item()]

    if len(logits_player_categoricals):
        target_player_categoricals = [obs[:, index] for index in obs_index["player"]["categoricals"]]
        for logits, target in zip(logits_player_categoricals, target_player_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2))

    # Hex (165x)

    if len(logits_hex_binary):
        target_hex_binary = obs[:, obs_index["hex"]["binary"]]
        loss_binary += binary_cross_entropy_with_logits(logits_hex_binary, target_hex_binary)

    if len(logits_hex_continuous):
        target_hex_continuous = obs[:, obs_index["hex"]["continuous"]]
        loss_continuous += mse_loss(logits_hex_continuous, target_hex_continuous)

    if len(logits_hex_categoricals):
        target_hex_categoricals = [obs[:, index] for index in obs_index["hex"]["categoricals"]]
        for logits, target in zip(logits_hex_categoricals, target_hex_categoricals):
            loss_categorical += cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2))

    return loss_binary, loss_continuous, loss_categorical


def train_model(
    logger,
    model,
    optimizer,
    buffer,
    train_epochs,
    train_batch_size
):
    model.train()

    for epoch in range(train_epochs):
        binary_losses = []
        continuous_losses = []
        categorical_losses = []
        total_losses = []

        for batch in buffer.sample_iter(train_batch_size):
            obs, action, next_rew, next_obs, next_mask, next_done = batch
            logits = model(obs, action)
            loss_bin, loss_cont, loss_cat = compute_losses(logger, model.obs_index, next_obs, logits)

            loss_tot = loss_bin + loss_cont + loss_cat
            binary_losses.append(loss_bin.item())
            continuous_losses.append(loss_cont.item())
            categorical_losses.append(loss_cat.item())
            total_losses.append(loss_tot.item())

            optimizer.zero_grad()
            loss_tot.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))  # No clipping, just measuring
            print(f"Gradient norm before clipping: {total_norm}")

            max_norm = 1.0  # Adjust as needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        binary_loss = sum(binary_losses) / len(binary_losses)
        continuous_loss = sum(continuous_losses) / len(continuous_losses)
        categorical_loss = sum(categorical_losses) / len(categorical_losses)
        total_loss = sum(total_losses) / len(total_losses)

        logger.log(dict(
            train_epoch=epoch,
            binary_loss=round(binary_loss, 6),
            continuous_loss=round(continuous_loss, 6),
            categorical_loss=round(categorical_loss, 6),
            total_loss=round(total_loss, 6),
        ))


def eval_model(logger, model, buffer, eval_env_steps):
    model.eval()
    batch_size = eval_env_steps // 10

    binary_losses = []
    continuous_losses = []
    categorical_losses = []
    total_losses = []

    for batch in buffer.sample_iter(batch_size):
        obs, action, next_rew, next_obs, next_mask, next_done = batch
        logits = model(obs, action)
        loss_bin, loss_cont, loss_cat = compute_losses(logger, model.obs_index, next_obs, logits)
        loss_tot = loss_bin + loss_cont + loss_cat

        binary_losses.append(loss_bin.item())
        continuous_losses.append(loss_cont.item())
        categorical_losses.append(loss_cat.item())
        total_losses.append(loss_tot.item())

    binary_loss = sum(binary_losses) / len(binary_losses)
    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    total_loss = sum(total_losses) / len(total_losses)

    return binary_loss, continuous_loss, categorical_loss, total_loss


def train(resume_config, dry_run):
    run_id = ''.join(random.choices(string.ascii_lowercase, k=8))

    # Usage:
    # python -m rl.encoder.autoencoder [path/to/config.json]

    if resume_config:
        with open(resume_config, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)

        resumed_run_id = config["run"]["id"]
        config["run"]["id"] = run_id
        config["run"]["resumed_config"] = resume_config
    else:
        config = dict(
            run=dict(
                id=run_id,
                out_dir=os.path.abspath("data/autoencoder"),
                resumed_config=None,
            ),
            env=dict(
                # opponent="BattleAI",  # BROKEN in develop1.6 from 2025-01-31
                opponent="StupidAI",
                mapname="gym/generated/4096/4x1024.vmap",
                max_steps=1000,
                random_heroes=1,
                random_obstacles=1,
                town_chance=30,
                warmachine_chance=40,
                random_terrain_chance=100,
                tight_formation_chance=20,
                allow_invalid_actions=True,
                user_timeout=3600,
                vcmi_timeout=3600,
                boot_timeout=300,
                conntype="thread",
                # vcmi_loglevel_global="trace",
                # vcmi_loglevel_ai="trace",
            ),
            train=dict(
                # TODO: consider torch.optim.lr_scheduler.StepLR
                learning_rate=1e-3,

                buffer_capacity=10_000,
                train_epochs=3,
                train_batch_size=1000,
                eval_env_steps=10_000,

                # Debug
                # buffer_capacity=100,
                # train_epochs=2,
                # train_batch_size=10,
                # eval_env_steps=100,
            )
        )

    os.makedirs(config["run"]["out_dir"], exist_ok=True)

    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        print(f"Saving new config to: {f.name}")
        json.dump(config, f, indent=4)

    logger = StructuredLogger(filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"))
    logger.log(dict(config=config))

    learning_rate = config["train"]["learning_rate"]
    buffer_capacity = config["train"]["buffer_capacity"]
    train_epochs = config["train"]["train_epochs"]
    train_batch_size = config["train"]["train_batch_size"]
    eval_env_steps = config["train"]["eval_env_steps"]

    assert buffer_capacity % train_batch_size == 0  # needed for train_steps
    assert eval_env_steps % 10 == 0  # needed for eval batch_size

    # Initialize environment, buffer, and model
    env = VcmiEnv(**config["env"])
    model = TransitionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if resume_config:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], resumed_run_id)
        logger.log(f"Loading model weights from {filename}")
        model.load_state_dict(torch.load(filename, weights_only=True), strict=True)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], resumed_run_id)
        logger.log(f"Loading optimizer weights from {filename}")
        optimizer.load_state_dict(torch.load(filename, weights_only=True))

    buffer = Buffer(capacity=buffer_capacity, dim_obs=STATE_SIZE, n_actions=N_ACTIONS, device="cpu")

    iteration = 0
    while True:
        collect_observations(
            logger=logger,
            env=env,
            buffer=buffer,
            n=buffer.capacity,
            progress_report_steps=0
        )
        assert buffer.full and not buffer.index

        binary_loss, continuous_loss, categorical_loss, total_loss = eval_model(
            logger=logger,
            model=model,
            buffer=buffer,
            eval_env_steps=eval_env_steps,
        )

        logger.log(dict(
            iteration=iteration,
            binary_loss=round(binary_loss, 6),
            continuous_loss=round(continuous_loss, 6),
            categorical_loss=round(categorical_loss, 6),
            total_loss=round(total_loss, 6),
        ))

        train_model(
            logger=logger,
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size
        )

        if not dry_run:
            filename = os.path.join(config["run"]["out_dir"], f"{run_id}-model.pt")
            logger.log(f"Saving model weights to {filename}")
            torch.save(model.state_dict(), filename)

            filename = os.path.join(config["run"]["out_dir"], f"{run_id}-optimizer.pt")
            logger.log(f"Saving optimizer weights to {filename}")
            torch.save(optimizer.state_dict(), filename)

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar="FILE", help="config file to resume or test")
    parser.add_argument('--dry-run', action="store_true", help="do not save model")
    args = parser.parse_args()

    train(args.f, args.dry_run)
