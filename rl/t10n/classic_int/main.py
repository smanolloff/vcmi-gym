# An (updated) version of `cgpefvda`

import os
import torch
import torch.nn as nn
import random
import string
import json
import time
import numpy as np
import pathlib
import argparse
import shutil
import botocore.exceptions
import botocore.config
import boto3
from boto3.s3.transfer import TransferConfig
import threading
import logging
import tempfile


from functools import partial

from torch.nn.functional import mse_loss
from datetime import datetime


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

        # XXX: TEST: REMOVE
        # end_of_episode_indices = torch.nonzero(self.done_buffer[:max_index - 1].bool(), as_tuple=True)[0]
        # assert not any(self.obs_buffer[end_of_episode_indices, 2])
        # EOF: TEST

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


class TransitionModel(nn.Module):
    def __init__(self, dim_other, dim_hexes, n_actions, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        assert dim_hexes % 165 == 0
        self.dim_other = dim_other
        self.dim_hexes = dim_hexes
        self.dim_obs = dim_other + dim_hexes
        self.d1hex = dim_hexes // 165

        # TODO: try flat obs+action (no per-hex)

        self.encoder1_other = nn.Sequential(
            nn.LazyLinear(100),
            nn.LeakyReLU(),
            nn.LazyLinear(self.dim_other),
        )

        self.encoder2_other = nn.Sequential(
            nn.LazyLinear(100),
            # nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(self.dim_other),
        )

        self.encoder3_other = nn.Sequential(
            nn.LazyLinear(100),
            # nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(self.dim_other),
        )

        self.encoder1_hex = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=[165, self.d1hex]),
            nn.LazyLinear(500),
            # nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(self.d1hex),
        )

        self.encoder2_hex = nn.Sequential(
            nn.LazyLinear(500),
            # nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(self.d1hex),
        )

        self.encoder3_hex = nn.Sequential(
            nn.LazyLinear(500),
            # nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(self.d1hex),
        )

        self.encoder1_pre = nn.Sequential(
            nn.LazyLinear(10000)
        )

        self.encoder1_merged = nn.Sequential(
            nn.LazyLinear(10000),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(10000),
        )

        self.encoder2_merged = nn.Sequential(
            nn.LazyLinear(10000),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(10000),
        )

        self.encoder3_merged = nn.Sequential(
            nn.LazyLinear(10000),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(10000),
        )

        self.head_obs = nn.LazyLinear(self.dim_obs)
        # self.head_mask = nn.LazyLinear(n_actions)
        # self.head_rew = nn.Sequential(nn.LazyLinear(1), nn.Flatten(0))
        # self.head_done = nn.Sequential(nn.LazyLinear(1), nn.Flatten(0))

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            self(torch.randn([2, DIM_OBS], device=device), torch.tensor([1, 1], device=device))

    def forward(self, obs, action):
        other, hexes = torch.split(obs, [self.dim_other, self.dim_hexes], dim=1)

        zother1 = self.encoder1_other(other) + other
        zother2 = self.encoder2_other(zother1) + zother1
        zother3 = self.encoder3_other(zother2) + zother2

        zhexes1 = self.encoder1_hex(hexes) + hexes.unflatten(dim=1, sizes=[165, self.d1hex])
        zhexes2 = self.encoder2_hex(zhexes1) + zhexes1
        zhexes3 = self.encoder3_hex(zhexes2) + zhexes2

        merged = torch.cat((nn.functional.one_hot(action, N_ACTIONS), zother3, zhexes3.flatten(start_dim=1)), dim=-1)
        merged = torch.cat((nn.functional.one_hot(action, N_ACTIONS), zother1, zhexes1.flatten(start_dim=1)), dim=-1)

        zmerged_pre = self.encoder1_pre(merged)
        zmerged1 = self.encoder1_merged(zmerged_pre) + zmerged_pre
        zmerged2 = self.encoder2_merged(zmerged1) + zmerged1
        zmerged3 = self.encoder3_merged(zmerged2) + zmerged2

        next_obs = self.head_obs(zmerged3)

        return next_obs

    def predict(self, obs, action):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(0)
            return self(obs, action)[0].numpy()

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

        if len(self.global_index["continuous"]):
            self.obs_index["global"]["continuous"] = self.global_index["continuous"]

        if len(self.global_index["binary"]):
            self.obs_index["global"]["binary"] = self.global_index["binary"]

        if len(self.global_index["categoricals"]):
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
            if len(indexes) == 0:
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


def train_model(
    logger,
    model,
    optimizer,
    scaler,
    buffer,
    epochs,
    batch_size
):
    model.train()

    for epoch in range(epochs):
        obs_losses = []
        # rew_losses = []
        # mask_losses = []
        # done_losses = []
        # total_losses = []

        for batch in buffer.sample_iter(batch_size):
            # obs, action, next_rew, next_obs, next_mask, next_done = batch
            obs, action, next_obs = batch
            # next_obs_pred, next_rew_pred, next_mask_pred, next_done_pred = model(obs, action)
            if scaler:
                with torch.amp.autocast(model.device.type):
                    next_obs_pred = model(obs, action)
                    obs_loss = mse_loss(next_obs_pred, next_obs)
                    # rew_loss = 0.1 * mse_loss(next_rew_pred, next_rew)
                    # mask_loss = binary_cross_entropy_with_logits(next_mask_pred, next_mask)
                    # done_loss = binary_cross_entropy_with_logits(next_done_pred, next_done)
                    # total_loss = obs_loss + rew_loss + mask_loss + done_loss
            else:
                next_obs_pred = model(obs, action)
                obs_loss = mse_loss(next_obs_pred, next_obs)

            obs_losses.append(obs_loss.item())
            # rew_losses.append(rew_loss.item())
            # mask_losses.append(mask_loss.item())
            # done_losses.append(done_loss.item())
            # total_losses.append(total_loss.item())

            optimizer.zero_grad()
            if scaler:
                scaler.scale(obs_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                obs_loss.backward()
                optimizer.step()

        obs_loss = sum(obs_losses) / len(obs_losses)
        # rew_loss = sum(rew_losses) / len(rew_losses)
        # mask_loss = sum(mask_losses) / len(mask_losses)
        # done_loss = sum(done_losses) / len(done_losses)
        # total_loss = sum(total_losses) / len(total_losses)

        return obs_loss


def eval_model(logger, model, buffer, batch_size):
    model.eval()
    obs_losses = []
    # rew_losses = []
    # mask_losses = []
    # done_losses = []
    # total_losses = []

    for batch in buffer.sample_iter(batch_size):
        # obs, action, next_rew, next_obs, next_mask, next_done = batch
        obs, action, next_obs = batch
        with torch.no_grad():
            # next_obs_pred, next_rew_pred, next_mask_pred, next_done_pred = model(obs, action)
            next_obs_pred = model(obs, action)

        obs_loss = mse_loss(next_obs_pred, next_obs)
        # rew_loss = 0.1 * mse_loss(next_rew_pred, next_rew)
        # mask_loss = binary_cross_entropy_with_logits(next_mask_pred, next_mask)
        # done_loss = binary_cross_entropy_with_logits(next_done_pred, next_done)
        # total_loss = obs_loss + rew_loss + mask_loss + done_loss

        obs_losses.append(obs_loss.item())
        # rew_losses.append(rew_loss.item())
        # mask_losses.append(mask_loss.item())
        # done_losses.append(done_loss.item())
        # total_losses.append(total_loss.item())

    obs_loss = sum(obs_losses) / len(obs_losses)
    # rew_loss = sum(rew_losses) / len(rew_losses)
    # mask_loss = sum(mask_losses) / len(mask_losses)
    # done_loss = sum(done_losses) / len(done_losses)
    # total_loss = sum(total_losses) / len(total_losses)

    # return obs_loss, rew_loss, mask_loss, done_loss, total_loss
    return obs_loss


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

            shutil.mv(f"{f_model}.tmp", f_model)
            shutil.mv(f"{f_optimizer}.tmp", f_optimizer)
            if scaler:
                shutil.mv(f"{f_scaler}.tmp", f_scaler)

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

    now = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        for type in ["obs", "done", "action"]:
            fname = f"{type}-{run_id}-{now:.0f}.npz"
            buf = getattr(buffer, f"{type}_buffer")
            local_path = f"{temp_dir}/{fname}"
            msg = f"Saving buffer to {local_path}"
            if dry_run:
                logger.info(f"{msg} (--dry-run)")
            else:
                logger.info(msg)
                np.savez_compressed(f"{local_path}.tmp", buf)
                shutil.move(f"{local_path}.tmp", local_path)
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

    learning_rate = config["train"]["learning_rate"]
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
    model = TransitionModel(DIM_OTHER, DIM_HEXES, N_ACTIONS, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    buffer = Buffer(capacity=buffer_capacity, dim_obs=DIM_OBS, n_actions=N_ACTIONS, device=device)
    eval_buffer = Buffer(capacity=eval_buffer_capacity, dim_obs=DIM_OBS, n_actions=N_ACTIONS, device=device)

    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    data_split_ratio = 0.98  # train / test
    optimize_local_storage = config.get("s3", {}).get("optimize_local_storage")

    if sample_from_s3:
        dataloader = iter(torch.utils.data.DataLoader(
            S3Dataset(
                logger=logger,
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
                    logger=logger,
                    bucket_name=config["s3"]["data"]["bucket_name"],
                    s3_dir=config["s3"]["data"]["s3_dir"],
                    cache_dir=config["s3"]["data"]["cache_dir"],
                    cached_files_max=config["s3"]["data"]["cached_files_max"],
                    shuffle=False,  # False needed for the save space hack where split is 90/1 (not 90/10)
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
            if not os.path.exists(filename):
                logger.debug("Local file does not exist, try S3")

                s3_config = config["s3"]["checkpoint"]
                s3_filename = f"{s3_config['s3_dir']}/{os.path.basename(filename)}"
                logger.info(f"Download s3://{s3_config['bucket_name']}/{s3_filename} ...")

                if os.path.exists(f"{filename}.tmp"):
                    os.unlink(f"{filename}.tmp")
                init_s3_client().download_file(s3_config["bucket_name"], s3_filename, f"{filename}.tmp")
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
        "train/buffer_capacity": buffer_capacity,
        "train/epochs": train_epochs,
        "train/batch_size": train_batch_size,
        "eval/buffer_capacity": eval_buffer_capacity,
        "eval/batch_size": eval_batch_size,
    })

    iteration = 0
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
            continue

        train_loss = train_model(
            logger=logger,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            buffer=buffer,
            epochs=train_epochs,
            batch_size=train_batch_size
        )

        if now - last_evaluation_at > config["eval"]["interval_s"]:
            last_evaluation_at = now

            if sample_from_env:
                collect_observations(logger=logger, env=env, buffer=eval_buffer, n=eval_buffer.capacity, progress_report_steps=0)
            elif sample_from_s3:
                load_observations(logger=logger, dataloader=eval_dataloader, buffer=eval_buffer)

            eval_loss = eval_model(
                logger=logger,
                model=model,
                buffer=eval_buffer,
                batch_size=eval_batch_size,
            )

            wandb_log({
                "iteration": iteration,
                "train_loss/total": train_loss,
                "eval_loss/total": eval_loss,
            }, commit=True)
        else:
            logger.info({
                "iteration": iteration,
                "train_loss/total": train_loss,
            })

        if now - last_checkpoint_at > config["s3"]["checkpoint"]["interval_s"]:
            last_checkpoint_at = now
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

        iteration += 1


def test(cfg_file):
    from vcmi_gym.envs.v10.vcmi_env import VcmiEnv

    run_id = os.path.basename(cfg_file).removesuffix("-config.json")
    weights_file = f"data/t10n/{run_id}-model.pt"

    model = load_for_test(weights_file)
    env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap", conntype="thread")
    do_test(model, env)


def load_for_test(file):
    dim_other = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
    dim_hexes = 165*STATE_SIZE_ONE_HEX
    model = TransitionModel(dim_other, dim_hexes, N_ACTIONS)
    model.eval()
    print(f"Loading {file}")
    weights = torch.load(file, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(weights, strict=True)
    return model


def do_test(model, env):
    from vcmi_gym.envs.v10.decoder.decoder import Decoder  # , pyconnector

    obs_prev = env.result.state.copy()
    print(env.render())
    # bf = Decoder.decode(1, obs_prev)
    # action = bf.hexes[4][13].action(pyconnector.HEX_ACT_MAP["MOVE"]).item()
    action = env.random_action()
    # bf = Decoder.decode(action, obs_prev)

    obs_pred = torch.as_tensor(model.predict(obs_prev, action))
    env.step(action)
    obs_real = env.result.intstates[1] if len(env.result.intstates) > 1 else env.result.state
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

    print(Decoder.decode(action, obs_pred.numpy()).render())
    print(Decoder.decode(action, obs_real).render())

    # print("Dirty (all):")
    # print(render["dirty"]["raw"])
    # print("Pred (all):")
    # print(render["pred"]["raw"])
    # print("Real (all):")
    # print(render["real"]["raw"])


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

