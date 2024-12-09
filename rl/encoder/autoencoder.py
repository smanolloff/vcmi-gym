import os
import torch
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
    def __init__(self, input_dim, latent_dim):
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


class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.log(dict(filename=filename))

    def log(self, obj):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%.6S")
        log_obj = dict(timestamp=timestamp, message=obj)
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

    next_progress_report_at = progress_report_step
    progress = 0
    terms = 0
    truncs = 0
    term = False
    trunc = False

    for i in range(n):
        action = env.random_action()
        if action is None:
            assert term or trunc
            term = False
            trunc = False
            terms += term
            truncs += trunc
            dict_obs, _info = env.reset()
        else:
            dict_obs, _rew, term, trunc, _info = env.step(action)

        obs = to_tensor(dict_obs)
        buffer.add(obs)

        # Ensure logging on final obs
        progress = 2 if i == n else i / n
        if progress >= next_progress_report_at:
            next_progress_report_at += progress_report_step
            logger.log(dict(observations_collected=i/n, progress=round(progress*100), terms=terms, truncs=truncs))


def train_autoencoder(
    logger,
    autoencoder,
    optimizer,
    buffer,
    batch_size,
    train_epochs,
    train_steps
):
    autoencoder.train()
    for epoch in range(train_epochs):
        loss_sum = 0
        for _ in range(train_steps):
            batch = buffer.sample(batch_size)
            reconstructed = autoencoder(batch)
            loss = mse_loss(reconstructed, batch)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss_sum / train_steps
        logger.log(dict(train_epoch=epoch+1, epochs_total=train_epochs, loss=round(loss, 6)))


def eval_autoencoder(logger, autoencoder, buffer, eval_steps, batch_size):
    autoencoder.eval()
    loss_sum = 0
    for _ in range(eval_steps):
        batch = buffer.sample(batch_size)
        reconstructed = autoencoder(batch)
        loss_sum += mse_loss(reconstructed, batch).item()

    return loss_sum / eval_steps


def main():
    chars = string.ascii_letters + string.digits
    run_id = ''.join(random.choice(chars) for _ in range(8))

    config = dict(
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
            user_timeout=600,
            vcmi_timeout=600,
            boot_timeout=300,
            conntype="thread",
            # vcmi_loglevel_global="trace",
            # vcmi_loglevel_ai="trace",
        ),
        train=dict(
            learning_rate=1e-4,
            buffer_capacity=100_000,
            train_epochs=10,
            train_steps=100,
            eval_steps=10_000,
        )
    )

    os.makedirs(config["run"]["out_dir"], exist_ok=True)
    logger = Logger(filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"))
    logger.log(dict(config=config))

    # Initialize environment, buffer, and model
    env = VcmiEnv_v5(**config["env"])

    learning_rate = config["train"]["learning_rate"]
    buffer_capacity = config["train"]["buffer_capacity"]
    train_epochs = config["train"]["train_epochs"]
    train_steps = config["train"]["train_steps"]
    eval_steps = config["train"]["eval_steps"]

    steps_per_epoch = train_epochs * train_steps
    assert buffer_capacity % steps_per_epoch == 0
    assert buffer_capacity % eval_steps == 0
    batch_size = buffer_capacity // steps_per_epoch
    eval_batch_size = buffer_capacity // eval_steps

    dict_obs, _ = env.reset()
    obs = to_tensor(dict_obs)
    ae = Autoencoder(input_dim=obs.shape[0], latent_dim=32)
    buffer = Buffer(capacity=buffer_capacity, obs_shape=obs.shape, device="cpu")
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate)

    iteration = 1
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
            eval_steps=eval_steps,
            batch_size=eval_batch_size
        )

        logger.log(dict(iteration=iteration, eval_loss=round(loss, 6)))

        train_autoencoder(
            logger=logger,
            autoencoder=ae,
            optimizer=optimizer,
            buffer=buffer,
            batch_size=batch_size,
            train_epochs=train_epochs,
            train_steps=train_steps
        )



if __name__ == "__main__":
    main()
