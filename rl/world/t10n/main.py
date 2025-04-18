import os
import torch
import torch.nn as nn
import argparse
import math
import enum
import contextlib

from torch.nn.functional import (
    mse_loss,
    binary_cross_entropy_with_logits,
    cross_entropy,
)

from ..util.buffer_base import BufferBase
from ..util.dataset_vcmi import Data, Context
from ..util.misc import layer_init
from ..util.obs_index import ObsIndex
from ..util.timer import Timer
from ..util.train import train

from ..util.constants_v11 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


class Other(enum.IntEnum):
    CAN_WAIT = 0
    DONE = enum.auto()

    _count = enum.auto()


# Skips last transition and carries reward over to the new transition
#
# (also see doc in dataset_vcmi.py)
#
# R(s0t0)=NaN is OK (episode start => no prev step, no reward)
# R(s1t0)=NaN is NOT OK => use R(s0t2) from prev step
# A(s0t2)=-1  is NOT OK => use A(s1t0) from next step
# A(s3t2)=-1  is OK (this is the terminal obs => no aciton)
#
# We `yield` a total of 9 samples:
#
#  | obs            | reward         | action         |
# -|----------------|----------------|----------------|
#  | O(s0t0)        | R(s0t0)=NaN    | A(s0t0)        | t=0 s=0
#  | O(s0t1)        | R(s0t1)        | A(s0t1)        | t=1
#  |                |                |                | t=2
# -|----------------|----------------|----------------|
#  | O(s1t0)        | R(s0t2) <- !!! | A(s1t0)        | t=0 s=1
#  | O(s1t1)        | R(s1t1)        | A(s1t1)        | t=1
#  |                |                |                | t=2
# -|----------------|----------------|----------------|
#  | O(s2t0)        | R(s1t2) <- !!! | A(s2t0)        | t=0 s=2
#  | O(s2t1)        | R(s2t1)        | A(s2t1)        | t=1
#  |                |                |                | t=2
# -|----------------|----------------|----------------|
#  | O(s3t0)        | R(s2t2) <- !!! | A(s3t0)        | t=0 s=3
#  | O(s3t1)        | R(s3t1)        | A(s3t1)        | t=1
#  | O(s3t2)        | R(s3t2)        | A(s1t0)=-1     | t=2 !!!
#
# =============================================================
#
# => for t=2 (the final transition):
#   - we carry its reward to next step's t=0
#   - we `yield` it only if s=3 (last step)
#

def vcmi_dataloader_functor():
    state = {"reward_carry": 0}

    def mw(data: Data, ctx: Context):
        if ctx.transition_id == ctx.num_transitions - 1:
            state["reward_carry"] = data.reward
            if not data.done:
                return None
        if ctx.transition_id == 0 and ctx.ep_steps > 0:
            return data._replace(reward=state["reward_carry"])
        return data

    return mw


class Buffer(BufferBase):
    def _valid_indices(self):
        max_index = self.capacity if self.full else self.index
        # Valid are indices of samples where done=False and cutoff=False
        # (i.e. to ensure obs,next_obs is valid)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        ok_samples = ~self.containers["done"][:max_index - 1].bool()
        ok_samples[self.worker_cutoffs] = False
        return torch.nonzero(ok_samples, as_tuple=True)[0]

    def sample(self, batch_size):
        inds = self._valid_indices()
        sampled_indices = inds[torch.randint(len(inds), (batch_size,), device=self.device)]

        obs = self.containers["obs"][sampled_indices]
        # action_mask = self.containers["mask"][sampled_indices]
        action = self.containers["action"][sampled_indices]
        next_obs = self.containers["obs"][sampled_indices + 1]
        next_mask = self.containers["mask"][sampled_indices + 1]
        next_reward = self.containers["reward"][sampled_indices + 1]
        next_done = self.containers["done"][sampled_indices + 1]

        return obs, action, next_obs, next_mask, next_reward, next_done

    def sample_iter(self, batch_size):
        valid_indices = self._valid_indices()
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)]

        # The valid indices are < than all indices by `short`
        short = self.capacity - len(shuffled_indices)
        if short:
            filler_indices = valid_indices[torch.randperm(len(valid_indices), device=self.device)][:short]
            shuffled_indices = torch.cat((shuffled_indices, filler_indices))

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


class TransitionModel(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        obsind = ObsIndex(device)

        self.abs_index = obsind.abs_index
        self.rel_index_global = obsind.rel_index_global
        self.rel_index_player = obsind.rel_index_player
        self.rel_index_hex = obsind.rel_index_hex

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
        if self.abs_index["global"]["binary"].numel():
            n_binary_features = len(self.abs_index["global"]["binary"])
            self.encoder_global_binary = nn.LazyLinear(n_binary_features)
            # No nonlinearity needed

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        self.encoders_global_categoricals = nn.ModuleList([])
        for ind in self.rel_index_global["categoricals"]:
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
        if self.abs_index["player"]["binary"].numel():
            n_binary_features = len(self.abs_index["player"]["binary"][0])
            self.encoder_player_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        self.encoders_player_categoricals = nn.ModuleList([])
        for ind in self.rel_index_player["categoricals"]:
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
        if self.abs_index["hex"]["binary"].numel():
            n_binary_features = len(self.abs_index["hex"]["binary"][0])
            self.encoder_hex_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per hex:
        # [(B, C1), (B, C2), ...]
        self.encoders_hex_categoricals = nn.ModuleList([])
        for ind in self.rel_index_hex["categoricals"]:
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

        global_continuous_in = obs[:, self.abs_index["global"]["continuous"]]
        global_binary_in = obs[:, self.abs_index["global"]["binary"]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_binary_z = self.encoder_global_binary(global_binary_in)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_merged = torch.cat((action_z, global_continuous_z, global_binary_z, global_categorical_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_continuous_in = obs[:, self.abs_index["player"]["continuous"]]
        player_binary_in = obs[:, self.abs_index["player"]["binary"]]
        player_categorical_ins = [obs[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_binary_z = self.encoder_player_binary(player_binary_in)
        player_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 2, -1), player_continuous_z, player_binary_z, player_categorical_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_continuous_in = obs[:, self.abs_index["hex"]["continuous"]]
        hex_binary_in = obs[:, self.abs_index["hex"]["binary"]]
        hex_categorical_ins = [obs[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
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

        # obs, rew, can_wait
        return obs_out

    def reconstruct(self, obs_out):
        global_continuous_out = obs_out[:, self.abs_index["global"]["continuous"]]
        global_binary_out = obs_out[:, self.abs_index["global"]["binary"]]
        global_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        player_continuous_out = obs_out[:, self.abs_index["player"]["continuous"]]
        player_binary_out = obs_out[:, self.abs_index["player"]["binary"]]
        player_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        hex_continuous_out = obs_out[:, self.abs_index["hex"]["continuous"]]
        hex_binary_out = obs_out[:, self.abs_index["hex"]["binary"]]
        hex_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
        next_obs = torch.zeros_like(obs_out)

        next_obs[:, self.abs_index["global"]["continuous"]] = torch.clamp(global_continuous_out, 0, 1)
        next_obs[:, self.abs_index["global"]["binary"]] = torch.sigmoid(global_binary_out).round()
        for ind, out in zip(self.abs_index["global"]["categoricals"], global_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.abs_index["player"]["continuous"]] = torch.clamp(player_continuous_out, 0, 1)
        next_obs[:, self.abs_index["player"]["binary"]] = torch.sigmoid(player_binary_out).round()
        for ind, out in zip(self.abs_index["player"]["categoricals"], player_categorical_outs):
            one_hot = torch.zeros_like(out)
            one_hot.scatter_(-1, torch.argmax(out, dim=-1, keepdim=True), 1)
            next_obs[:, ind] = one_hot

        next_obs[:, self.abs_index["hex"]["continuous"]] = torch.clamp(hex_continuous_out, 0, 1)
        next_obs[:, self.abs_index["hex"]["binary"]] = torch.sigmoid(hex_binary_out).round()
        for ind, out in zip(self.abs_index["hex"]["categoricals"], hex_categorical_outs):
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
                obs_pred_logits = self.forward(obs, action)
                obs_pred = self.reconstruct(obs_pred_logits)[0].numpy()
            finally:
                self.train(was_training)

            return obs_pred


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


def train_model(
    logger,
    model,
    optimizer,
    scaler,
    buffer,
    stats,
    loss_weights,
    epochs,
    batch_size,
    accumulate_grad,
    wlog
):
    model.train()
    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    total_losses = []
    timer = Timer()

    maybe_autocast = torch.amp.autocast(model.device.type) if scaler else contextlib.nullcontext()

    assert buffer.capacity % batch_size == 0, f"{buffer.capacity} % {batch_size} == 0"

    if accumulate_grad:
        grad_steps = buffer.capacity // batch_size
        assert grad_steps > 0

    for epoch in range(epochs):
        timer.start()
        for batch in buffer.sample_iter(batch_size):
            timer.stop()
            obs, action, next_obs, next_mask, next_rew, next_done = batch

            with maybe_autocast:
                pred_obs = model(obs, action)
                loss_cont, loss_bin, loss_cat = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
                loss_tot = loss_cont + loss_bin + loss_cat

            continuous_losses.append(loss_cont.item())
            binary_losses.append(loss_bin.item())
            categorical_losses.append(loss_cat.item())
            total_losses.append(loss_tot.item())

            if accumulate_grad:
                if scaler:
                    scaler.scale(loss_tot / grad_steps).backward()
                else:
                    (loss_tot / grad_steps).backward()
            else:
                if scaler:
                    scaler.scale(loss_tot).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_tot.backward()
                    optimizer.step()
                optimizer.zero_grad()
            timer.start()
        timer.stop()

        if accumulate_grad:
            # assert grad_step == 0, "Sample waste: %d sample batches"
            # Update once after the entire buffer is exhausted
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    binary_loss = sum(binary_losses) / len(binary_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()

    wlog["train_loss/continuous"] = continuous_loss
    wlog["train_loss/binary"] = binary_loss
    wlog["train_loss/categorical"] = categorical_loss
    wlog["train_loss/total"] = total_loss
    wlog["train_dataset/wait_time_s"] = total_wait

    return total_loss


def eval_model(
    logger,
    model,
    loss_weights,
    buffer,
    batch_size,
    wlog
):
    model.eval()

    continuous_losses = []
    binary_losses = []
    categorical_losses = []
    total_losses = []
    timer = Timer()

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, action, next_obs, next_mask, next_rew, next_done = batch

        with torch.no_grad():
            pred_obs = model(obs, action)

        loss_cont, loss_bin, loss_cat = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
        loss_tot = loss_cont + loss_bin + loss_cat

        continuous_losses.append(loss_cont.item())
        binary_losses.append(loss_bin.item())
        categorical_losses.append(loss_cat.item())
        total_losses.append(loss_tot.item())
        timer.start()
    timer.stop()

    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    binary_loss = sum(binary_losses) / len(binary_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()

    wlog["eval_loss/continuous"] = continuous_loss
    wlog["eval_loss/binary"] = binary_loss
    wlog["eval_loss/categorical"] = categorical_loss
    wlog["eval_loss/total"] = total_loss
    wlog["eval_dataset/wait_time_s"] = total_wait

    return total_loss


def test(cfg_file):
    from vcmi_gym.envs.v11.vcmi_env import VcmiEnv

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
    from vcmi_gym.envs.v11.decoder.decoder import Decoder

    env.reset()
    for _ in range(10):
        print("=" * 100)
        if env.terminated or env.truncated:
            env.reset()
        act = env.random_action()
        obs, rew, term, trunc, _info = env.step(act)

        # [(obs, act, real_obs), (obs, act, real_obs), ...]
        dream = [(obs["transitions"]["observations"][0], obs["transitions"]["actions"][0], None)]

        for i in range(1, len(obs["transitions"]["observations"])):
            obs_prev = obs["transitions"]["observations"][i-1]
            act_prev = obs["transitions"]["actions"][i-1]
            obs_next = obs["transitions"]["observations"][i]
            # mask_next = obs["transitions"]["action_masks"][i]
            # rew_next = obs["transitions"]["rewards"][i]
            # done_next = (term or trunc) and i == len(obs["transitions"]["observations"]) - 1

            obs_pred_raw = model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(act_prev).unsqueeze(0))
            obs_pred_raw = obs_pred_raw[0]
            obs_pred = model.predict(obs_prev, act_prev)
            dream.append((model.predict(*dream[i-1][:2]), obs["transitions"]["actions"][i], obs_next))

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

            lines_prev = prepare(obs_prev, act_prev, None, "Start:")
            lines_real = prepare(obs_next, -1, None, "Real:")
            lines_pred = prepare(obs_pred, -1, None, "Predicted:")

            losses = compute_losses(
                logger=None,
                obs_index=model.abs_index,
                loss_weights=None,
                next_obs=torch.as_tensor(obs_next).unsqueeze(0),
                pred_obs=obs_pred_raw.unsqueeze(0),
            )

            print("Losses | Obs: binary=%.4f, cont=%.4f, categorical=%.4f" % losses)

            # print(Decoder.decode(obs_prev).render(0))
            # for i in range(len(bfields)):
            print("")
            print("\n".join([(" ".join(rowlines)) for rowlines in zip(lines_prev, lines_real, lines_pred)]))
            print("")

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

        if len(dream) > 2:
            print(" ******** SEQUENCE: ********** ")
            print(env.render_transitions(add_regular_render=False))
            print(" ******** DREAM: ********** ")
            rcfg = env.reward_cfg._replace(step_fixed=0)
            for i, (obs, act, obs_real) in enumerate(dream):
                print("*" * 10)
                if i == 0:
                    print("Start:")
                    print(Decoder.decode(obs).render(act))
                else:
                    bf_real = Decoder.decode(obs_real)
                    bf = Decoder.decode(obs)
                    print(f"Real step #{i}:")
                    print(bf_real.render(act))
                    print("")
                    print(f"Dream step #{i}:")
                    print(bf.render(act))
                    print(f"Real / Dream rewards: {env.calc_reward(0, bf_real, rcfg)} / {env.calc_reward(0, bf, rcfg)}:")

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

    if args.dry_run:
        args.no_wandb = True

    if args.action == "test":
        test(args.f)
    else:
        from .config import config
        common_args = dict(
            config=config,
            resume_config=args.f,
            loglevel=args.loglevel,
            dry_run=args.dry_run,
            no_wandb=args.no_wandb,
            # sample_only=False,
            model_creator=TransitionModel,
            buffer_creator=Buffer,
            vcmi_dataloader_functor=vcmi_dataloader_functor,
            s3_dataloader_functor=None,
            eval_model_fn=eval_model,
            train_model_fn=train_model
        )
        if args.action == "train":
            train(**dict(common_args, sample_only=False))
        elif args.action == "sample":
            train(**dict(common_args, sample_only=True))
