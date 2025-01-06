import os
import torch
import sys
import torch.nn as nn
import random
import string
import json
import yaml
import argparse
import re
import numpy as np

from torch.nn.functional import mse_loss
from datetime import datetime
from collections import OrderedDict

from vcmi_gym.envs.v6.vcmi_env import VcmiEnv
from vcmi_gym.envs.v6.decoder.decoder import Decoder


def to_tensor(dict_obs):
    # return torch.as_tensor(dict_obs["observation"])
    return torch.concatenate((
        torch.as_tensor(dict_obs["observation"]),
        # torch.as_tensor(dict_obs["action_mask_1"]).float(),
        # torch.as_tensor(dict_obs["action_mask_2"]).float().flatten(),
    ))


def vae_loss(recon_x, x, mu, logvar, beta=1):
    recon_loss = mse_loss(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence / x.size(0)


def schedule_fn(mode="lin_grow", start=0.1, end=1, rate=0.1):
    assert mode in ["const", "lin_decay", "lin_grow", "exp_decay"]

    if mode != "const":
        assert end > 0
        assert rate > 0

    if mode.endswith("_decay"):
        assert start > end

    if mode.endswith("_grow"):
        assert end > start

    if mode == "lin_decay":
        return lambda p: float(np.clip(start - (start - end) * (rate * p), end, start))
    elif mode == "lin_grow":
        return lambda p: float(np.clip(start + (end - start) * rate * p, start, end))
    elif mode == "exp_decay":
        return lambda p: float(end + (start - end) * np.exp(-rate * p))
    elif mode == "const":
        return lambda _: float(start)
    else:
        raise Exception(f"Unknown mode: {mode}")


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

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index
        indices = torch.randperm(max_index)  # Shuffle indices
        for i in range(0, max_index, batch_size):
            yield self.buffer[indices[i:i + batch_size]]


class VAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, obs_dims, z_dims):
            super().__init__()
            assert ["misc", "stacks", "hexes"] == list(obs_dims.keys())
            assert ["misc", "1stack", "1hex", "out"] == list(z_dims.keys())

            self.dmisc = obs_dims["misc"]
            self.dstacks = obs_dims["stacks"]
            self.dhexes = obs_dims["hexes"]
            self.dz = z_dims["out"]
            self.dzmisc = z_dims["misc"]
            self.dz1stack = z_dims["1stack"]
            self.dz1hex = z_dims["1hex"]
            self.dzstacks = 20 * self.dz1stack
            self.dzhexes = 165 * self.dz1hex
            self.d1stack = VcmiEnv.STATE_SIZE_STACKS // 20
            self.d1hex = VcmiEnv.STATE_SIZE_HEXES // 165
            self.dzmerge = self.dzmisc + self.dzstacks + self.dzhexes

            # Networks:
            self.enc_misc = nn.Sequential(
                # => (B, dmisc)
                nn.LazyLinear(self.dzmisc),
                nn.GELU()
                # => (B, dzmisc)
            )
            self.enc_stacks = nn.Sequential(
                # => (B, 20*S)
                nn.Unflatten(1, [20, self.d1stack]),  # => (B, 20, d1stack)
                nn.LazyLinear(self.dz1stack),         # => (B, 20, dz1stack)
                nn.GELU(),
                nn.Flatten()
                # => (B, dzstacks)
            )
            self.enc_hexes = nn.Sequential(
                # => (B, 165*H)
                nn.Unflatten(1, [165, self.d1hex]),  # => (B, 165, d1hex)
                nn.LazyLinear(self.dz1hex),          # => (B, 165, dz1hex)
                nn.GELU(),
                nn.Flatten()
                # => (B, dzhexes)
            )
            self.enc_merged = nn.Sequential(
                # => (B, dzmerge)
                nn.LazyLinear(self.dz),
                nn.GELU(),
                nn.LazyLinear(self.dz)
                # => (B, dz)
            )

        def forward(self, x):
            misc, stacks, hexes = x.split((self.dmisc, self.dstacks, self.dhexes), dim=1)
            return self.enc_merged(torch.cat(dim=1, tensors=(
                self.enc_misc(misc),
                self.enc_stacks(stacks),
                self.enc_hexes(hexes),
            )))

    class Decoder(nn.Module):
        def __init__(self, obs_dims, z_dims):
            super().__init__()
            assert ["misc", "stacks", "hexes"] == list(obs_dims.keys())
            assert ["misc", "1stack", "1hex", "out"] == list(z_dims.keys())

            self.dmisc = obs_dims["misc"]
            self.dstacks = obs_dims["stacks"]
            self.dhexes = obs_dims["hexes"]
            self.dz = z_dims["out"]
            self.dzmisc = z_dims["misc"]
            self.dz1stack = z_dims["1stack"]
            self.dz1hex = z_dims["1hex"]
            self.dzstacks = 20 * self.dz1stack
            self.dzhexes = 165 * self.dz1hex
            self.d1stack = VcmiEnv.STATE_SIZE_STACKS // 20
            self.d1hex = VcmiEnv.STATE_SIZE_HEXES // 165
            self.dzmerge = self.dzmisc + self.dzstacks + self.dzhexes

            # Networks:
            self.dec_misc = nn.Sequential(
                # => (B, dzmisc)
                nn.LazyLinear(self.dmisc),
                nn.GELU()
                # => (B, dmisc)
            )
            self.dec_stacks = nn.Sequential(
                # => (B, dzstacks)
                nn.Unflatten(1, [20, self.dz1stack]),  # => (B, 20, dz1stack)
                nn.LazyLinear(self.d1stack),           # => (B, 20, d1stack)
                nn.GELU(),
                nn.Flatten()
                # => (B, 20*S)
            )
            self.dec_hexes = nn.Sequential(
                # => (B, dzhexes)
                nn.Unflatten(1, [165, self.dz1hex]),  # => (B, 165, dz1hex)
                nn.LazyLinear(self.d1hex),            # => (B, 165, d1hex)
                nn.GELU(),
                nn.Flatten()
                # => (B, dhexes)
            )
            self.dec_merged = nn.Sequential(
                # => (B, dz)
                nn.LazyLinear(self.dz),
                nn.GELU(),
                nn.LazyLinear(self.dzmerge)
                # => (B, dzmerge)
            )

        def forward(self, z):
            zmerge = self.dec_merged(z)
            zmisc, zstacks, zhexes = zmerge.split((self.dzmisc, self.dzstacks, self.dzhexes), dim=1)
            return torch.cat(dim=1, tensors=(
                self.dec_misc(zmisc),
                self.dec_stacks(zstacks),
                self.dec_hexes(zhexes),
            ))

    def __init__(self, obs_dims, z_dims):
        super().__init__()

        ord_obs_dims = OrderedDict({k: obs_dims[k] for k in ["misc", "stacks", "hexes"]})
        ord_z_dims = OrderedDict({k: z_dims[k] for k in ["misc", "1stack", "1hex", "out"]})

        self.encoder = self.Encoder(ord_obs_dims, ord_z_dims)
        self.decoder = self.Decoder(ord_obs_dims, ord_z_dims)

        self.fc_mu = nn.LazyLinear(z_dims["out"])
        self.fc_logvar = nn.LazyLinear(z_dims["out"])

    def vanilla_forward(self, x):
        h = self.encoder(x)
        reconstructed = self.decoder(h)
        return reconstructed

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


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
            dict_obs, _info = env.reset()
        else:
            dict_obs, _rew, term, trunc, _info = env.step(action)

        obs = to_tensor(dict_obs)
        buffer.add(obs)

    logger.log(dict(observations_collected=n, progress=100, terms=terms, truncs=truncs))


def train_vae(
    logger,
    vae,
    optimizer,
    buffer,
    train_epochs,
    train_batch_size,
    beta,
    vanilla
):
    vae.train()

    for epoch in range(train_epochs):
        losses = []
        for batch in buffer.sample_iter(train_batch_size):
            if vanilla:
                reconstructed = vae.vanilla_forward(batch)
                loss = mse_loss(reconstructed, batch)
            else:
                reconstructed, mu, logvar = vae(batch)
                loss = vae_loss(reconstructed, batch, mu, logvar, beta)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = sum(losses) / len(losses)
        logger.log(dict(train_epoch=epoch, loss=round(loss, 6)))


def eval_vae(logger, vae, buffer, eval_env_steps, beta, vanilla):
    vae.eval()
    batch_size = eval_env_steps // 10
    losses = []

    for batch in buffer.sample_iter(batch_size):
        if vanilla:
            reconstructed = vae.vanilla_forward(batch)
            loss = mse_loss(reconstructed, batch)
        else:
            reconstructed, mu, logvar = vae(batch)
            loss = vae_loss(reconstructed, batch, mu, logvar, beta)

        losses.append(loss.item())

    return sum(losses) / len(losses)


def train(resume_config, name, vanilla=False):
    run_id = name + "-" + ''.join(random.choices(string.ascii_lowercase, k=8))

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
                opponent="MMAI_RANDOM",
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
            vae=dict(
                # Order is important
                obs_dims={
                    "misc": VcmiEnv.STATE_SIZE_MISC,
                    "stacks": VcmiEnv.STATE_SIZE_STACKS,
                    "hexes": VcmiEnv.STATE_SIZE_HEXES,
                },
                z_dims={
                    "misc": 4,
                    "1stack": 8,
                    "1hex": 8,
                    "out": 512,
                },
            ),
            train=dict(
                lr_schedule=dict(mode="exp_decay", start=1e-3, end=1e-5, rate=0.05),
                beta_schedule=dict(mode="const", start=0),  # 0 beta => just AE

                buffer_capacity=100_000,
                train_epochs=3,
                train_batch_size=1000,
                eval_env_steps=10_000,

                # Debug
                # buffer_capacity=10,
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

    lr_schedule_fn = schedule_fn(**config["train"]["lr_schedule"])
    buffer_capacity = config["train"]["buffer_capacity"]
    train_epochs = config["train"]["train_epochs"]
    train_batch_size = config["train"]["train_batch_size"]
    eval_env_steps = config["train"]["eval_env_steps"]

    assert buffer_capacity % train_batch_size == 0  # needed for train_steps
    assert eval_env_steps % 10 == 0  # needed for eval batch_size

    # Initialize environment, buffer, and model
    env = VcmiEnv(**config["env"])

    dict_obs, _ = env.reset()
    obs = to_tensor(dict_obs)
    beta_schedule_fn = schedule_fn(**config["train"]["beta_schedule"])
    vae = VAE(**config["vae"])
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr_schedule_fn(0))

    if resume_config:
        filename = "%s/%s-model.pt" % (config["run"]["out_dir"], resumed_run_id)
        logger.log(f"Loading model weights from {filename}")
        vae.load_state_dict(torch.load(filename, weights_only=True), strict=True)

        filename = "%s/%s-optimizer.pt" % (config["run"]["out_dir"], resumed_run_id)
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
            progress_report_steps=0
        )
        assert buffer.full and not buffer.index

        lr = float(lr_schedule_fn(iteration))
        beta = float(beta_schedule_fn(iteration))
        optimizer.param_groups[0]["lr"] = lr

        loss = eval_vae(
            logger=logger,
            vae=vae,
            buffer=buffer,
            eval_env_steps=eval_env_steps,
            beta=beta,
            vanilla=vanilla
        )

        logger.log(dict(iteration=iteration, lr=lr, beta=beta, eval_loss=round(loss, 6)))

        train_vae(
            logger=logger,
            vae=vae,
            optimizer=optimizer,
            buffer=buffer,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size,
            beta=beta,
            vanilla=vanilla
        )

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-model.pt")
        logger.log(f"Saving model weights to {filename}")
        torch.save(vae.state_dict(), filename)

        filename = os.path.join(config["run"]["out_dir"], f"{run_id}-optimizer.pt")
        logger.log(f"Saving optimizer weights to {filename}")
        torch.save(optimizer.state_dict(), filename)

        iteration += 1


def test(cfg_file, lastobs=False, verbose=False):
    with open(cfg_file, "r") as f:
        config = json.load(f)
    print("Env config: %s" % config["env"])
    env = VcmiEnv(**config["env"])
    dict_obs, _ = env.reset()
    vae = VAE(**config["vae"])
    filename = "%s/%s-model.pt" % (config["run"]["out_dir"], config["run"]["id"])
    print(f"Loading model weights from {filename}")
    vae.load_state_dict(torch.load(filename, weights_only=True), strict=True)

    while True:
        action = env.random_action()
        if action is None:
            dict_obs, _info = env.reset()
            done = False
        else:
            dict_obs, _rew, term, trunc, _info = env.step(action)
            done = term or trunc

        obs_only_shape = dict_obs["observation"].shape

        with torch.no_grad():
            if lastobs:
                print("Loading lastobs.tmp", os.getcwd())
                obs1 = torch.load("lastobs.npy", weights_only=True)
            else:
                obs1 = to_tensor(dict_obs)
                print("Saving to lastobs.tmp", os.getcwd())
                torch.save(obs1, "lastobs.npy")

            print(env.render())

            obs2 = vae.encoder(obs1)
            loss = mse_loss(obs2, obs1)
            print("Loss: %s" % loss)
            d1 = Decoder.decode(obs1[:obs_only_shape[0]].numpy(), done, verbose=verbose)

            d2 = Decoder.decode(
                obs2[:obs_only_shape[0]].numpy(),
                done,
                state0=obs1[:obs_only_shape[0]].numpy(),
                precision=0,  # number of digits after "."
                roundfracs=None,   # 5 = round to nearest 0.2 (3.14 => 3.2)
                verbose=verbose
            )

        print(d1.render())
        print(d2.render())

        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help="be more verbose", action="store_true")
    parser.add_argument('-f', metavar="FILE", help="config file to resume or test")
    parser.add_argument('-l', help="test with lastobs.npy", action="store_true")
    parser.add_argument("-n", metavar="NAME", help="experiment name to prepend to ID")
    parser.add_argument("-V", help="train vanilla autoencoder", action="store_true")
    parser.add_argument('action', choices=["train", "test"])
    args = parser.parse_args()

    if args.n:
        assert re.match(r"^[0-9A-Za-z_-]+$", args.n)
        if args.f:
            print("-n and -f are mutually exclusive")
            sys.exit(1)

    if args.action == "train":
        fn = train
        if args.l or args.v:
            print("-l and -v can only be given if action is 'test'")
            sys.exit(1)

        train(args.f, args.n or "vae", vanilla=args.V)
    else:
        assert args.action == "test"
        if args.f:
            print("-f is required if action is 'test'")
            sys.exit(1)

        test(args.f, args.l, args.v)
