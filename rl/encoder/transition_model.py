import os
import torch
import torch.nn as nn
import random
import string
import json
import yaml
import argparse

from torch.nn.functional import mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from datetime import datetime

from vcmi_gym.envs.v8.vcmi_env import VcmiEnv


def to_tensor(dict_obs):
    return torch.as_tensor(dict_obs["observation"])


class Buffer:
    def __init__(self, capacity, dim_obs, n_actions, device="cpu"):
        self.capacity = capacity
        self.device = device

        self.obs_buffer = torch.empty((capacity, dim_obs), dtype=torch.float32, device=device)
        self.mask_buffer = torch.empty((capacity, n_actions), dtype=torch.float32, device=device)
        self.done_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.action_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.reward_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)

        self.index = 0
        self.full = False

    # Using compact version with single obs and mask buffers
    # def add(self, obs, action_mask, done, action, reward, next_obs, next_action_mask, next_done):
    def add(self, obs, action_mask, done, action, reward):
        self.obs_buffer[self.index] = torch.as_tensor(obs, dtype=torch.float32)
        self.mask_buffer[self.index] = torch.as_tensor(action_mask, dtype=torch.float32)
        self.done_buffer[self.index] = torch.as_tensor(done, dtype=torch.float32)
        self.action_buffer[self.index] = torch.as_tensor(action, dtype=torch.float32)
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
    def __init__(self, dim_other, dim_hexes, n_actions):
        super().__init__()

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
        self.head_mask = nn.LazyLinear(n_actions)
        self.head_rew = nn.Sequential(nn.LazyLinear(1), nn.Flatten(0))
        self.head_done = nn.Sequential(nn.LazyLinear(1), nn.Flatten(0))

    def forward(self, obs, action):
        other, hexes = torch.split(obs, [self.dim_other, self.dim_hexes], dim=1)

        zother = self.encoder_other(other)
        zhexes = self.encoder_hex(hexes)
        merged = torch.cat((action.unsqueeze(-1), zother, zhexes), dim=-1)
        z = self.encoder_merged(merged)
        next_obs = self.head_obs(z)
        next_rew = self.head_rew(z)
        next_mask = self.head_mask(z)
        next_done = self.head_done(z)
        return next_obs, next_rew, next_mask, next_done


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
            next_obs_pred, next_rew_pred, next_mask_pred, next_done_pred = model(obs, action)

            obs_loss = mse_loss(next_obs_pred, next_obs)
            rew_loss = 0.1 * mse_loss(next_rew_pred, next_rew)
            mask_loss = binary_cross_entropy_with_logits(next_mask_pred, next_mask)
            done_loss = binary_cross_entropy_with_logits(next_done_pred, next_done)
            total_loss = obs_loss + rew_loss + mask_loss + done_loss

            obs_losses.append(obs_loss.item())
            rew_losses.append(rew_loss.item())
            mask_losses.append(mask_loss.item())
            done_losses.append(done_loss.item())
            total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        obs_loss = sum(obs_losses) / len(obs_losses)
        rew_loss = sum(rew_losses) / len(rew_losses)
        mask_loss = sum(mask_losses) / len(mask_losses)
        done_loss = sum(done_losses) / len(done_losses)
        total_loss = sum(total_losses) / len(total_losses)
        logger.log(dict(
            train_epoch=epoch,
            obs_loss=obs_loss,
            rew_loss=rew_loss,
            mask_loss=mask_loss,
            done_loss=done_loss,
            total_loss=total_loss,
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
        next_obs_pred, next_rew_pred, next_mask_pred, next_done_pred = model(obs, action)

        obs_loss = mse_loss(next_obs_pred, next_obs)
        rew_loss = 0.1 * mse_loss(next_rew_pred, next_rew)
        mask_loss = binary_cross_entropy_with_logits(next_mask_pred, next_mask)
        done_loss = binary_cross_entropy_with_logits(next_done_pred, next_done)
        total_loss = obs_loss + rew_loss + mask_loss + done_loss

        obs_losses.append(obs_loss.item())
        rew_losses.append(rew_loss.item())
        mask_losses.append(mask_loss.item())
        done_losses.append(done_loss.item())
        total_losses.append(total_loss.item())

    obs_loss = sum(obs_losses) / len(obs_losses)
    rew_loss = sum(rew_losses) / len(rew_losses)
    mask_loss = sum(mask_losses) / len(mask_losses)
    done_loss = sum(done_losses) / len(done_losses)
    total_loss = sum(total_losses) / len(total_losses)

    return obs_loss, rew_loss, mask_loss, done_loss, total_loss


def train(resume_config):
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
                learning_rate=1e-4,

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

    dict_obs, _ = env.reset()

    dim_other = VcmiEnv.STATE_SIZE_GLOBAL + 2*VcmiEnv.STATE_SIZE_ONE_PLAYER
    dim_hexes = VcmiEnv.STATE_SIZE_HEXES
    dim_obs = dim_other + dim_hexes
    n_actions = VcmiEnv.ACTION_SPACE.n

    model = TransitionModel(dim_other, dim_hexes, n_actions)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if resume_config:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], resumed_run_id)
        logger.log(f"Loading model weights from {filename}")
        model.load_state_dict(torch.load(filename, weights_only=True), strict=True)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], resumed_run_id)
        logger.log(f"Loading optimizer weights from {filename}")
        optimizer.load_state_dict(torch.load(filename, weights_only=True))

    buffer = Buffer(capacity=buffer_capacity, dim_obs=dim_obs, n_actions=n_actions, device="cpu")

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

        obs_loss, rew_loss, mask_loss, done_loss, total_loss = eval_model(
            logger=logger,
            model=model,
            buffer=buffer,
            eval_env_steps=eval_env_steps,
        )

        logger.log(dict(
            iteration=iteration,
            obs_loss=round(obs_loss, 6),
            rew_loss=round(rew_loss, 6),
            mask_loss=round(mask_loss, 6),
            done_loss=round(done_loss, 6),
            total_loss=round(total_loss, 6)
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
        logger.log(f"Saving model weights to {filename}")
        torch.save(model.state_dict(), filename)

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-optimizer.pt")
        logger.log(f"Saving optimizer weights to {filename}")
        torch.save(optimizer.state_dict(), filename)

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar="FILE", help="config file to resume or test")
    args = parser.parse_args()

    train(args.f)
