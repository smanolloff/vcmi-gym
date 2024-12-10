import os
import torch
import sys
import torch.nn as nn
import random
import string
import json
import yaml

from torch.nn.functional import mse_loss
from datetime import datetime

from vcmi_gym import VcmiEnv_v5


def to_tensor(dict_obs):
    return torch.concatenate((
        torch.as_tensor(dict_obs["observation"]),
        torch.as_tensor(dict_obs["action_mask_1"]).float(),
        torch.as_tensor(dict_obs["action_mask_2"]).float().flatten(),
    ))


class Buffer:
    def __init__(self, capacity, obs_shape, device="cpu"):
        self.capacity = capacity
        self.buffer = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.index = 0
        self.full = False

    def add(self, obs):
        self.buffer[self.index] = obs
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index
        indices = random.sample(range(max_index), batch_size)
        return self.buffer[indices]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super().__init__()

        layer_sizes = [4096, 1024, 256, 32]
        encoder = nn.Sequential()
        decoder = nn.Sequential()

        tmpdim = input_dim
        for layer_size in layer_sizes:
            encoder.append(nn.Linear(tmpdim, layer_size))
            encoder.append(nn.ReLU())
            decoder.insert(0, nn.ReLU())
            decoder.insert(0, nn.Linear(layer_size, tmpdim))
            tmpdim = layer_size

        # drop activations after last layer
        self.encoder = encoder[:-1]
        self.decoder = decoder[:-1]

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


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

    for i in range(n):
        # Ensure logging on final obs
        progress = round((i+1) / n, 3)
        if progress >= next_progress_report_at:
            next_progress_report_at += progress_report_step
            logger.log(dict(observations_collected=i+1, progress=progress*100, terms=terms, truncs=truncs))

        action = env.random_action()
        if action is None:
            assert term or trunc
            terms += term
            truncs += trunc
            term = False
            trunc = False
            dict_obs, _info = env.reset()
        else:
            dict_obs, _rew, term, trunc, _info = env.step(action)

        obs = to_tensor(dict_obs)
        buffer.add(obs)

    logger.log(dict(observations_collected=n, progress=100, terms=terms, truncs=truncs))


def train_autoencoder(
    logger,
    autoencoder,
    optimizer,
    buffer,
    train_epochs,
    train_batch_size
):
    autoencoder.train()
    train_steps = buffer.capacity // train_batch_size

    for epoch in range(train_epochs):
        loss_sum = 0
        for _ in range(train_steps):
            batch = buffer.sample(train_batch_size)
            reconstructed = autoencoder(batch)
            loss = mse_loss(reconstructed, batch)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss_sum / train_steps
        logger.log(dict(train_epoch=epoch, loss=round(loss, 6)))


def eval_autoencoder(logger, autoencoder, buffer, eval_env_steps):
    autoencoder.eval()
    loss_sum = 0
    batch_size = eval_env_steps // 10
    for _ in range(10):
        batch = buffer.sample(batch_size)
        reconstructed = autoencoder(batch)
        loss_sum += mse_loss(reconstructed, batch).item()

    return loss_sum / 10


def main():
    run_id = ''.join(random.choices(string.ascii_lowercase, k=8))

    # Usage:
    # python -m rl.encoder.autoencoder [path/to/config.json]

    resuming = False
    if len(sys.argv) > 1:
        resuming = True
        with open(sys.argv[1], "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)
    else:
        config = dict(
            resume_config=None,
            # resume_config="/Users/simo/Projects/vcmi-gym/data/autoencoder/uwgxnffd.json",

            # XXX: values below are irrelevant if `resume_config` is given
            run=dict(
                id=run_id,
                out_dir=os.path.abspath("data/autoencoder"),
            ),
            env=dict(
                opponent="MMAI_RANDOM",
                mapname="gym/generated/4096/4x1024.vmap",
                max_steps=1000,
                random_heroes=1,
                random_obstacles=1,
                town_chance=30,
                warmachine_chance=40,
                random_terrain_chance=100,
                tight_formation_chance=20,
                user_timeout=1800,
                vcmi_timeout=1800,
                boot_timeout=300,
                conntype="thread",
                # vcmi_loglevel_global="trace",
                # vcmi_loglevel_ai="trace",
            ),
            train=dict(
                layer_sizes=[4096, 1024, 256, 32],
                learning_rate=1e-5,

                buffer_capacity=100_000,
                train_epochs=10,
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

    if not resuming:
        with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
            print(f"Saving new config to: {f.name}")
            json.dump(config, f)

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
    env = VcmiEnv_v5(**config["env"])

    dict_obs, _ = env.reset()
    obs = to_tensor(dict_obs)
    ae = Autoencoder(input_dim=obs.shape[0], layer_sizes=config["train"]["layer_sizes"])
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate)

    if resuming:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], config["run"]["id"])
        logger.log(f"Loading model weights from {filename}")
        ae.load_state_dict(torch.load(filename, weights_only=True), strict=True)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], config["run"]["id"])
        logger.log(f"Loading optimizer weights from {filename}")
        optimizer.load_state_dict(torch.load(filename, weights_only=True))

    buffer = Buffer(capacity=buffer_capacity, obs_shape=obs.shape, device="cpu")

    iteration = 0
    while True:
        collect_observations(
            logger=logger,
            env=env,
            buffer=buffer,
            n=buffer.capacity,
            progress_report_steps=5
        )
        assert buffer.full and not buffer.index

        loss = eval_autoencoder(
            logger=logger,
            autoencoder=ae,
            buffer=buffer,
            eval_env_steps=eval_env_steps,
        )

        logger.log(dict(iteration=iteration, eval_loss=round(loss, 6)))

        train_autoencoder(
            logger=logger,
            autoencoder=ae,
            optimizer=optimizer,
            buffer=buffer,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size
        )

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-model.pt")
        logger.log(f"Saving model weights to {filename}")
        torch.save(ae.state_dict(), filename)

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-optimizer.pt")
        logger.log(f"Saving optimizer weights to {filename}")
        torch.save(optimizer.state_dict(), filename)

        iteration += 1


if __name__ == "__main__":
    main()
