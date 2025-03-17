# An (updated) version of `cgpefvda`

import os
import torch
import torch.nn as nn
import random
import string
import json
import yaml
import time
import numpy as np
import pathlib
import argparse
from functools import partial

from torch.nn.functional import mse_loss
from datetime import datetime
from vcmi_gym.envs.v8.pyprocconnector import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
)

from ..util.s3dataset import S3Dataset
from vcmi_gym.envs.v8.vcmi_env import VcmiEnv


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


class Buffer:
    def __init__(self, capacity, dim_obs, n_actions, device=torch.device("cpu")):
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
        self.obs_buffer[self.index] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.mask_buffer[self.index] = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        self.done_buffer[self.index] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.action_buffer[self.index] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        self.reward_buffer[self.index] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False (episode not ended)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1].bool(), as_tuple=True,)[0]
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


class TransitionModel(nn.Module):
    def __init__(self, dim_other, dim_hexes, n_actions, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        assert dim_hexes % 165 == 0
        self.dim_other = dim_other
        self.dim_hexes = dim_hexes
        self.dim_obs = dim_other + dim_hexes
        d1hex = dim_hexes // 165

        # TODO: try flat obs+action (no per-hex)

        self.encoder_other = nn.Sequential(
            nn.LazyLinear(64),
            nn.LeakyReLU(),
        )

        self.encoder_hex = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=[165, d1hex]),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(256),
            nn.LeakyReLU(),
            nn.LazyLinear(64),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.encoder_merged = nn.Sequential(
            # => (B, 64 + 10560 + 1)  // +1 action
            nn.LazyLinear(1024),
            nn.LeakyReLU(),
            nn.LazyLinear(1024),
        )

        self.head_obs = nn.LazyLinear(self.dim_obs)
        # self.head_mask = nn.LazyLinear(n_actions)
        # self.head_rew = nn.Sequential(nn.LazyLinear(1), nn.Flatten(0))
        # self.head_done = nn.Sequential(nn.LazyLinear(1), nn.Flatten(0))

        self.to(device)

    def forward(self, obs, action):
        other, hexes = torch.split(obs, [self.dim_other, self.dim_hexes], dim=1)

        zother = self.encoder_other(other)
        zhexes = self.encoder_hex(hexes)
        merged = torch.cat((nn.functional.one_hot(action, VcmiEnv.ACTION_SPACE.n), zother, zhexes), dim=-1)
        z = self.encoder_merged(merged)
        next_obs = self.head_obs(z)
        # next_rew = self.head_rew(z)
        # next_mask = self.head_mask(z)
        # next_done = self.head_done(z)
        # return next_obs, next_rew, next_mask, next_done
        return next_obs

    def predict(self, obs, action):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.int64, device=self.device)
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


def load_observations(logger, dataloader, buffer):
    logger.log("Loading observations...")
    buffer.add_batch(*next(dataloader))
    logger.log(f"Loaded {buffer.capacity} observations")


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
        obs_losses = []
        rew_losses = []
        mask_losses = []
        done_losses = []
        total_losses = []

        for batch in buffer.sample_iter(train_batch_size):
            obs, action, next_rew, next_obs, next_mask, next_done = batch
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

            optimizer.zero_grad()
            # total_loss.backward()
            obs_loss.backward()
            optimizer.step()

        obs_loss = sum(obs_losses) / len(obs_losses)
        # rew_loss = sum(rew_losses) / len(rew_losses)
        # mask_loss = sum(mask_losses) / len(mask_losses)
        # done_loss = sum(done_losses) / len(done_losses)
        # total_loss = sum(total_losses) / len(total_losses)
        logger.log(dict(
            train_epoch=epoch,
            obs_loss=obs_loss,
            # rew_loss=rew_loss,
            # mask_loss=mask_loss,
            # done_loss=done_loss,
            # total_loss=total_loss,
        ))


def eval_model(logger, model, buffer, eval_env_steps):
    model.eval()
    batch_size = eval_env_steps // 10
    obs_losses = []
    rew_losses = []
    mask_losses = []
    done_losses = []
    total_losses = []

    for batch in buffer.sample_iter(batch_size):
        obs, action, next_rew, next_obs, next_mask, next_done = batch
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


def train(resume_config, dry_run, no_wandb, sample_only):
    # Usage:
    # python -m rl.encoder.autoencoder [path/to/config.json]

    if resume_config:
        with open(resume_config, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)

        run_id = config["run"]["id"]
        config["run"]["resumed_config"] = resume_config
    else:
        run_id = ''.join(random.choices(string.ascii_lowercase, k=8))

        #
        # Depending on the `env` and `s3data` configs, the behaviour is:
        #
        # env=no    s3data=no   => ERROR
        # env=yes   s3data=no   => sample from env
        # env=no    s3data=yes  => load samples from S3
        # env=yes   s3data=yes  => sample from env + save to S3
        #
        # NOTE: saving to S3 in this script is not implemented
        #       Files are saved to local disk only (see `bufdir`)
        #       and can be later uploaded via s3uploader.py
        #
        config = dict(
            run=dict(
                id=run_id,
                out_dir=os.path.abspath("data/autoencoder"),
                resumed_config=None,
            ),

            # env=None,
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

            # s3data=None,
            s3data=dict(
                bucket_name="vcmi-gym",
                s3_prefix="v8",
                # Must not be part of the config (clear-text in metadata.json)
                # aws_access_key=os.environ["AWS_ACCESS_KEY"],
                # aws_secret_key=os.environ["AWS_SECRET_KEY"],
                region_name="eu-north-1",
                cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache")),
                num_workers=1,
                prefetch_factor=1,
                shuffle=True,
            ),

            train={
                # TODO: consider torch.optim.lr_scheduler.StepLR
                "learning_rate": 1e-4,

                "buffer_capacity": 10_000,
                "train_epochs": 1,
                "train_batch_size": 1000,
                "eval_env_steps": 10_000,

                # !!! DEBUG (linter warning is OK) !!!
                # "buffer_capacity": 100,
                # "train_epochs": 1,
                # "train_batch_size": 10,
                # "eval_env_steps": 100,
            }
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

    learning_rate = config["train"]["learning_rate"]
    buffer_capacity = config["train"]["buffer_capacity"]
    train_epochs = config["train"]["train_epochs"]
    train_batch_size = config["train"]["train_batch_size"]
    eval_env_steps = config["train"]["eval_env_steps"]

    assert buffer_capacity % train_batch_size == 0  # needed for train_steps
    assert eval_env_steps % 10 == 0  # needed for eval batch_size

    if sample_from_env:
        env = VcmiEnv(**config["env"])

    dict_obs, _ = env.reset()

    dim_other = VcmiEnv.STATE_SIZE_GLOBAL + 2*VcmiEnv.STATE_SIZE_ONE_PLAYER
    dim_hexes = VcmiEnv.STATE_SIZE_HEXES
    dim_obs = dim_other + dim_hexes
    n_actions = VcmiEnv.ACTION_SPACE.n

    model = TransitionModel(dim_other, dim_hexes, n_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    buffer = Buffer(capacity=buffer_capacity, dim_obs=dim_obs, n_actions=n_actions, device=device)

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

    if resume_config:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], run_id)
        logger.log(f"Loading model weights from {filename}")
        model.load_state_dict(torch.load(filename, weights_only=True), strict=True)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], run_id)
        logger.log(f"Loading optimizer weights from {filename}")
        optimizer.load_state_dict(torch.load(filename, weights_only=True))

    global wandb_log

    if no_wandb:
        def wandb_log(data, commit=False):
            logger.log(data)
    else:
        wandb = setup_wandb(logger, config, model, __file__)

        def wandb_log(data, commit=False):
            wandb.log(data, commit=commit)
            logger.log(data)

    wandb_log({
        "params/learning_rate": learning_rate,
        "params/buffer_capacity": buffer_capacity,
        "params/train_epochs": train_epochs,
        "params/train_batch_size": train_batch_size,
        "params/eval_env_steps": eval_env_steps,
    })

    iteration = 0
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

        if save_samples:
            # NOTE: this assumes no old observations are left in the buffer
            bufdir = os.path.join(config["run"]["out_dir"], "samples", "%s-%d" % (run_id, time.time()))
            msg = f"Saving buffer to {bufdir}"

            if dry_run:
                logger.log(f"{msg} (--dry-run)")
            else:
                logger.log(msg)
                buffer.save(bufdir, dict(run_id=run_id))

        if sample_only:
            continue

        # obs_loss, rew_loss, mask_loss, done_loss, total_loss = eval_model(
        obs_loss = eval_model(
            logger=logger,
            model=model,
            buffer=buffer,
            eval_env_steps=eval_env_steps,
        )

        wandb_log({
            "iteration": iteration,
            "loss/total": obs_loss
        }, commit=True)

        logger.log(dict(
            iteration=iteration,
            obs_loss=round(obs_loss, 6),
            # rew_loss=round(rew_loss, 6),
            # mask_loss=round(mask_loss, 6),
            # done_loss=round(done_loss, 6),
            # total_loss=round(total_loss, 6)
        ))

        train_model(
            logger=logger,
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size
        )

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-model.pt")
        msg = f"Saving model weights to {filename}"
        if dry_run:
            logger.log(f"{msg} (--dry-run)")
        else:
            logger.log(msg)
            torch.save(model.state_dict(), filename)

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-optimizer.pt")
        msg = f"Saving optimizer weights to {filename}"
        if dry_run:
            logger.log(f"{msg} (--dry-run)")
        else:
            logger.log(msg)
            torch.save(optimizer.state_dict(), filename)

        iteration += 1


def test():
    from vcmi_gym.envs.v8.decoder.decoder import Decoder, pyconnector

    run_id = os.path.basename(__file__).removesuffix(".py")
    dim_other = VcmiEnv.STATE_SIZE_GLOBAL + 2*VcmiEnv.STATE_SIZE_ONE_PLAYER
    dim_hexes = VcmiEnv.STATE_SIZE_HEXES
    n_actions = VcmiEnv.ACTION_SPACE.n
    model = TransitionModel(dim_other, dim_hexes, n_actions)
    weights = torch.load(f"data/autoencoder/{run_id}-model.pt", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()

    env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap")
    obs_prev = env.result.state.copy()
    bf = Decoder.decode(obs_prev)
    action = bf.hexes[4][13].action(pyconnector.HEX_ACT_MAP["MOVE"]).item()

    obs_pred = torch.as_tensor(model.predict(obs_prev, action))
    obs_real = env.step(action)[0]["observation"]
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

    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def prepare(obs, name, headline):
        render[name] = {}
        render[name]["raw"] = Decoder.decode(obs).render()
        render[name]["lines"] = render[name]["raw"].split("\n")
        render[name]["bf_lines"] = render[name]["lines"][:15]
        render[name]["bf_lines"].insert(0, headline)
        render[name]["bf_len"] = [len(l) for l in render[name]["bf_lines"]]
        render[name]["bf_printlen"] = [len(ansi_escape.sub('', l)) for l in render[name]["bf_lines"]]
        render[name]["bf_maxlen"] = max(render[name]["bf_len"])
        render[name]["bf_maxprintlen"] = max(render[name]["bf_printlen"])
        render[name]["bf_lines"] = [l + " "*(render[name]["bf_maxprintlen"] - pl) for l, pl in zip(render[name]["bf_lines"], render[name]["bf_printlen"])]

    prepare(obs_prev, "prev", "Previous:")
    prepare(obs_real, "real", "Real:")
    prepare(obs_pred.numpy(), "pred", "Predicted:")
    prepare(obs_dirty, "dirty", "Dirty:")

    render["combined"]["bf"] = "\n".join("%s → %s%s" % (l1, l2, l3) for l1, l2, l3 in zip(render["prev"]["bf_lines"], render["real"]["bf_lines"], render["pred"]["bf_lines"]))
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
