import torch
import torch.nn as nn
import math
import enum
import contextlib

import torch.nn.functional as F

from ..util.buffer_base import BufferBase
from ..util.dataset_vcmi import Data, Context
from ..util.misc import layer_init
from ..util.obs_index import ObsIndex
from ..util.timer import Timer

from ..util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    N_ACTIONS,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


class Other(enum.IntEnum):
    CAN_WAIT = 0
    DONE = enum.auto()


class Reconstruction(enum.IntEnum):
    PROBS = 0               # clamp(cont) + softmax(bin) + softmax(cat)
    SAMPLES = enum.auto()   # clamp(cont) + sample(sigmoid(bin)) + sample(softmax(cat))
    GREEDY = enum.auto()    # clamp(cont) + round(sigmoid(bin)) + argmax(cat)


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
            # No nonlinearity needed?

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        self.encoders_global_categoricals = nn.ModuleList([])
        for ind in self.rel_index_global["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_global_categoricals.append(cat_emb_size)

        # Thresholds:
        # [(B, T1), (B, T2), ...]
        self.encoders_global_thresholds = nn.ModuleList([])
        for ind in self.rel_index_global["thresholds"]:
            self.encoders_global_thresholds.append(nn.LazyLinear(len(ind)))
            # No nonlinearity needed?

        # Merge
        z_size_global = 256
        self.encoder_merged_global = nn.Sequential(
            # => (B, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
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

        # Thresholds per player:
        # [(B, T1), (B, T2), ...]
        self.encoders_player_thresholds = nn.ModuleList([])
        for ind in self.rel_index_player["thresholds"]:
            self.encoders_player_thresholds.append(nn.LazyLinear(len(ind)))

        # Merge per player
        z_size_player = 256
        self.encoder_merged_player = nn.Sequential(
            # => (B, 2, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
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

        # Thresholds per hex:
        # [(B, T1), (B, T2), ...]
        self.encoders_hex_thresholds = nn.ModuleList([])
        for ind in self.rel_index_hex["thresholds"]:
            self.encoders_hex_thresholds.append(nn.LazyLinear(len(ind)))

        # Merge per hex
        z_size_hex = 512
        self.encoder_merged_hex = nn.Sequential(
            # => (B, 165, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
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

    def forward_probs(self, obs, action_probs):
        action_z = torch.matmul(action_probs, self.encoder_action.weight)  # shape: [batch_size, embedding_dim]
        return self._forward(obs, action_z)

    def forward(self, obs, action):
        action_z = self.encoder_action(action)
        return self._forward(obs, action_z)

    def _forward(self, obs, action_z):
        assert obs.device.type == self.device.type, f"{obs.device.type} == {self.device.type}"

        global_continuous_in = obs[:, self.abs_index["global"]["continuous"]]
        global_binary_in = obs[:, self.abs_index["global"]["binary"]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        global_threshold_ins = [obs[:, ind] for ind in self.abs_index["global"]["thresholds"]]
        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_binary_z = self.encoder_global_binary(global_binary_in)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_threshold_z = torch.cat([lin(x) for lin, x in zip(self.encoders_global_thresholds, global_threshold_ins)], dim=-1)
        global_merged = torch.cat((action_z, global_continuous_z, global_binary_z, global_categorical_z, global_threshold_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_continuous_in = obs[:, self.abs_index["player"]["continuous"]]
        player_binary_in = obs[:, self.abs_index["player"]["binary"]]
        player_categorical_ins = [obs[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        player_threshold_ins = [obs[:, ind] for ind in self.abs_index["player"]["thresholds"]]
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_binary_z = self.encoder_player_binary(player_binary_in)
        player_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_threshold_z = torch.cat([lin(x) for lin, x in zip(self.encoders_player_thresholds, player_threshold_ins)], dim=-1)
        player_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 2, -1), player_continuous_z, player_binary_z, player_categorical_z, player_threshold_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_continuous_in = obs[:, self.abs_index["hex"]["continuous"]]
        hex_binary_in = obs[:, self.abs_index["hex"]["binary"]]
        hex_categorical_ins = [obs[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
        hex_threshold_ins = [obs[:, ind] for ind in self.abs_index["hex"]["thresholds"]]
        hex_continuous_z = self.encoder_hex_continuous(hex_continuous_in)
        hex_binary_z = self.encoder_hex_binary(hex_binary_in)
        hex_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)], dim=-1)
        hex_threshold_z = torch.cat([lin(x) for lin, x in zip(self.encoders_hex_thresholds, hex_threshold_ins)], dim=-1)
        hex_merged = torch.cat((action_z.unsqueeze(1).expand(-1, 165, -1), hex_continuous_z, hex_binary_z, hex_categorical_z, hex_threshold_z), dim=-1)
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

        return obs_out

    def reconstruct(self, obs_out, strategy=Reconstruction.GREEDY):
        global_continuous_out = obs_out[:, self.abs_index["global"]["continuous"]]
        global_binary_out = obs_out[:, self.abs_index["global"]["binary"]]
        global_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        global_threshold_outs = [obs_out[:, ind] for ind in self.abs_index["global"]["thresholds"]]
        player_continuous_out = obs_out[:, self.abs_index["player"]["continuous"]]
        player_binary_out = obs_out[:, self.abs_index["player"]["binary"]]
        player_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        player_threshold_outs = [obs_out[:, ind] for ind in self.abs_index["player"]["thresholds"]]
        hex_continuous_out = obs_out[:, self.abs_index["hex"]["continuous"]]
        hex_binary_out = obs_out[:, self.abs_index["hex"]["binary"]]
        hex_categorical_outs = [obs_out[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
        hex_threshold_outs = [obs_out[:, ind] for ind in self.abs_index["hex"]["thresholds"]]
        next_obs = torch.zeros_like(obs_out)

        reconstruct_continuous = lambda logits: torch.clamp(logits, 0, 1)

        # PROBS = enum.auto()     # clamp(cont) + sigmoid(bin) + softmax(cat)
        # SAMPLES = enum.auto()   # clamp(cont) + sample(sigmoid(bin)) + sample(softmax(cat))
        # GREEDY = enum.auto()    # clamp(cont) + round(sigmoid(bin)) + argmax(cat)

        if strategy == Reconstruction.PROBS:
            def reconstruct_binary(logits):
                return logits.sigmoid()

            def reconstruct_categorical(logits):
                return logits.softmax(dim=-1)

            def reconstruct_threshold(logits):
                return reconstruct_binary(logits)

        elif strategy == Reconstruction.SAMPLES:
            def reconstruct_binary(logits):
                return torch.bernoulli(logits.sigmoid())

            def reconstruct_categorical(logits):
                num_classes = logits.shape[-1]
                probs_2d = logits.softmax(dim=-1).view(-1, num_classes)
                sampled_classes = torch.multinomial(probs_2d, num_samples=1).view(logits.shape[:-1])
                return F.one_hot(sampled_classes, num_classes=num_classes).float()

            def reconstruct_threshold(logits):
                return reconstruct_binary(logits)

        elif strategy == Reconstruction.GREEDY:
            def reconstruct_binary(logits):
                return logits.sigmoid().round()

            def reconstruct_categorical(logits):
                return F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()

            def reconstruct_threshold(logits):
                return reconstruct_binary(logits)

        next_obs[:, self.abs_index["global"]["continuous"]] = reconstruct_continuous(global_continuous_out)
        next_obs[:, self.abs_index["global"]["binary"]] = reconstruct_binary(global_binary_out)
        for ind, out in zip(self.abs_index["global"]["categoricals"], global_categorical_outs):
            next_obs[:, ind] = reconstruct_categorical(out)
        for ind, out in zip(self.abs_index["global"]["thresholds"], global_threshold_outs):
            next_obs[:, ind] = reconstruct_threshold(out)

        next_obs[:, self.abs_index["player"]["continuous"]] = reconstruct_continuous(player_continuous_out)
        next_obs[:, self.abs_index["player"]["binary"]] = reconstruct_binary(player_binary_out)
        for ind, out in zip(self.abs_index["player"]["categoricals"], player_categorical_outs):
            next_obs[:, ind] = reconstruct_categorical(out)
        for ind, out in zip(self.abs_index["player"]["thresholds"], player_threshold_outs):
            next_obs[:, ind] = reconstruct_threshold(out)

        next_obs[:, self.abs_index["hex"]["continuous"]] = reconstruct_continuous(hex_continuous_out)
        next_obs[:, self.abs_index["hex"]["binary"]] = reconstruct_binary(hex_binary_out)
        for ind, out in zip(self.abs_index["hex"]["categoricals"], hex_categorical_outs):
            next_obs[:, ind] = reconstruct_categorical(out)
        for ind, out in zip(self.abs_index["hex"]["thresholds"], hex_threshold_outs):
            next_obs[:, ind] = reconstruct_threshold(out)

        return next_obs

    def predict_from_probs_(self, obs, action_probs, strategy=Reconstruction.GREEDY):
        logits = self.forward_probs(obs, action_probs)
        return self.reconstruct(logits, strategy=strategy)

    def predict_(self, obs, action, strategy=Reconstruction.GREEDY):
        logits = self.forward(obs, action)
        return self.reconstruct(logits, strategy=strategy)

    def predict(self, obs, action, strategy=Reconstruction.GREEDY):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.as_tensor(action, dtype=torch.int64, device=self.device).unsqueeze(0)
            return self.predict_(obs, action, strategy=strategy)[0].numpy()


def compute_losses(logger, obs_index, loss_weights, next_obs, pred_obs):
    logits_global_continuous = pred_obs[:, obs_index["global"]["continuous"]]
    logits_global_binary = pred_obs[:, obs_index["global"]["binary"]]
    logits_global_categoricals = [pred_obs[:, ind] for ind in obs_index["global"]["categoricals"]]
    logits_global_thresholds = [pred_obs[:, ind] for ind in obs_index["global"]["thresholds"]]
    logits_player_continuous = pred_obs[:, obs_index["player"]["continuous"]]
    logits_player_binary = pred_obs[:, obs_index["player"]["binary"]]
    logits_player_categoricals = [pred_obs[:, ind] for ind in obs_index["player"]["categoricals"]]
    logits_player_thresholds = [pred_obs[:, ind] for ind in obs_index["player"]["thresholds"]]
    logits_hex_continuous = pred_obs[:, obs_index["hex"]["continuous"]]
    logits_hex_binary = pred_obs[:, obs_index["hex"]["binary"]]
    logits_hex_categoricals = [pred_obs[:, ind] for ind in obs_index["hex"]["categoricals"]]
    logits_hex_thresholds = [pred_obs[:, ind] for ind in obs_index["hex"]["thresholds"]]

    loss_continuous = 0
    loss_binary = 0
    loss_categorical = 0
    loss_threshold = 0

    hex_losses_continuous = torch.zeros(pred_obs.shape[0], dtype=torch.float32, device=pred_obs.device)
    hex_losses_binary = torch.zeros(pred_obs.shape[0], dtype=torch.float32, device=pred_obs.device)
    hex_losses_categorical = torch.zeros(pred_obs.shape[0], dtype=torch.float32, device=pred_obs.device)
    hex_losses_threshold = torch.zeros(pred_obs.shape[0], dtype=torch.float32, device=pred_obs.device)

    # Global

    if logits_global_continuous.numel():
        target_global_continuous = next_obs[:, obs_index["global"]["continuous"]]
        loss_continuous += F.mse_loss(logits_global_continuous, target_global_continuous)

    if logits_global_binary.numel():
        target_global_binary = next_obs[:, obs_index["global"]["binary"]]
        # weight_global_binary = loss_weights["binary"]["global"]
        # loss_binary += F.binary_cross_entropy_with_logits(logits_global_binary, target_global_binary, pos_weight=weight_global_binary)
        loss_binary += F.binary_cross_entropy_with_logits(logits_global_binary, target_global_binary)

    if logits_global_categoricals:
        target_global_categoricals = [next_obs[:, index] for index in obs_index["global"]["categoricals"]]
        # weight_global_categoricals = loss_weights["categoricals"]["global"]
        # for logits, target, weight in zip(logits_global_categoricals, target_global_categoricals, weight_global_categoricals):
        #     loss_categorical += F.cross_entropy(logits, target, weight=weight)
        for logits, target in zip(logits_global_categoricals, target_global_categoricals):
            loss_categorical += F.cross_entropy(logits, target)

    if logits_global_thresholds:
        target_global_thresholds = [next_obs[:, index] for index in obs_index["global"]["thresholds"]]
        for logits, target in zip(logits_global_thresholds, target_global_thresholds):
            bce_loss = F.binary_cross_entropy_with_logits(logits, target)
            # Monotonicity regularization:
            probs = torch.sigmoid(logits)
            mono_diff = probs[:, :-1] - probs[:, 1:]
            mono_violation = F.relu(-mono_diff)
            mono_loss = mono_violation.mean()  # * 1.0  (lambda coefficient)
            loss_threshold += (bce_loss + mono_loss)

    # Player (2x)

    if logits_player_continuous.numel():
        target_player_continuous = next_obs[:, obs_index["player"]["continuous"]]
        loss_continuous += F.mse_loss(logits_player_continuous, target_player_continuous)

    if logits_player_binary.numel():
        target_player_binary = next_obs[:, obs_index["player"]["binary"]]
        # weight_player_binary = loss_weights["binary"]["player"]
        # loss_binary += F.binary_cross_entropy_with_logits(logits_player_binary, target_player_binary, pos_weight=weight_player_binary)
        loss_binary += F.binary_cross_entropy_with_logits(logits_player_binary, target_player_binary)

    # XXX: CrossEntropyLoss expects (B, C, *) input where C=num_classes
    #      => transpose (B, 2, C) => (B, C, 2)
    #      (not needed for BCE or MSE)
    # See difference:
    # [F.cross_entropy(logits, target).item(), F.cross_entropy(logits.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1)).item(), F.cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2)).item()]

    if logits_player_categoricals:
        target_player_categoricals = [next_obs[:, index] for index in obs_index["player"]["categoricals"]]
        # weight_player_categoricals = loss_weights["categoricals"]["player"]
        # for logits, target, weight in zip(logits_player_categoricals, target_player_categoricals, weight_player_categoricals):
        #     loss_categorical += F.cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2), weight=weight)
        for logits, target in zip(logits_player_categoricals, target_player_categoricals):
            loss_categorical += F.cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2))

    if logits_player_thresholds:
        target_player_thresholds = [next_obs[:, index] for index in obs_index["player"]["thresholds"]]
        for logits, target in zip(logits_player_thresholds, target_player_thresholds):
            bce_loss = F.binary_cross_entropy_with_logits(logits, target)
            # Monotonicity regularization:
            probs = torch.sigmoid(logits)
            mono_diff = probs[:, :, :-1] - probs[:, :, 1:]
            mono_violation = F.relu(-mono_diff)
            mono_loss = mono_violation.mean()  # * 1.0  (lambda coefficient)
            loss_threshold += (bce_loss + mono_loss)

    # Hex (165x)

    if logits_hex_continuous.numel():
        target_hex_continuous = next_obs[:, obs_index["hex"]["continuous"]]
        hex_losses_continuous += F.mse_loss(
            logits_hex_continuous,
            target_hex_continuous,
            reduction="none",
        ).flatten(start_dim=1).mean(dim=1)
        # => (B) of losses

    if logits_hex_binary.numel():
        target_hex_binary = next_obs[:, obs_index["hex"]["binary"]]
        # weight_hex_binary = loss_weights["binary"]["hex"]
        # loss_binary += F.binary_cross_entropy_with_logits(logits_hex_binary, target_hex_binary, pos_weight=weight_hex_binary)
        hex_losses_binary += F.binary_cross_entropy_with_logits(
            logits_hex_binary,
            target_hex_binary,
            reduction="none"
        ).flatten(start_dim=1).mean(dim=1)

    if logits_hex_categoricals:
        target_hex_categoricals = [next_obs[:, index] for index in obs_index["hex"]["categoricals"]]
        # weight_hex_categoricals = loss_weights["categoricals"]["hex"]
        # for logits, target, weight in zip(logits_hex_categoricals, target_hex_categoricals, weight_hex_categoricals):
        #     loss_categorical += F.cross_entropy(logits.swapaxes(1, 2), target.swapaxes(1, 2), weight=weight)
        for logits, target in zip(logits_hex_categoricals, target_hex_categoricals):
            hex_losses_categorical += F.cross_entropy(
                logits.swapaxes(1, 2),
                target.swapaxes(1, 2),
                reduction="none"
            ).mean(dim=1)

    if logits_hex_thresholds:
        target_hex_thresholds = [next_obs[:, index] for index in obs_index["hex"]["thresholds"]]
        for logits, target in zip(logits_hex_thresholds, target_hex_thresholds):
            bce_loss = F.binary_cross_entropy_with_logits(logits, target)
            # Monotonicity regularization:
            probs = torch.sigmoid(logits)
            mono_diff = probs[:, :, :-1] - probs[:, :, 1:]
            mono_violation = F.relu(-mono_diff)
            mono_loss = mono_violation.mean()  # * 1.0  (lambda coefficient)
            loss_threshold += (bce_loss + mono_loss)

    # Ignore hex for terminal obs
    is_terminal = torch.nonzero(next_obs[:, GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][1]])
    # => (b) of indexes where obs is terminal (b < B)

    hex_losses_continuous[is_terminal] = 0
    hex_losses_binary[is_terminal] = 0
    hex_losses_categorical[is_terminal] = 0
    hex_losses_threshold[is_terminal] = 0

    loss_binary += hex_losses_binary.mean()
    loss_continuous += hex_losses_continuous.mean()
    loss_categorical += hex_losses_categorical.mean()
    loss_threshold += hex_losses_threshold.mean()

    return loss_binary, loss_continuous, loss_categorical, loss_threshold


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
    threshold_losses = []
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
                loss_cont, loss_bin, loss_cat, loss_thr = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
                loss_tot = loss_cont + loss_bin + loss_cat + loss_thr

            continuous_losses.append(loss_cont.item())
            binary_losses.append(loss_bin.item())
            categorical_losses.append(loss_cat.item())
            threshold_losses.append(loss_thr.item())
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
    threshold_loss = sum(threshold_losses) / len(threshold_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()

    wlog["train_loss/continuous"] = continuous_loss
    wlog["train_loss/binary"] = binary_loss
    wlog["train_loss/categorical"] = categorical_loss
    wlog["train_loss/threshold"] = threshold_loss
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
    threshold_losses = []
    total_losses = []
    timer = Timer()

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, action, next_obs, next_mask, next_rew, next_done = batch

        with torch.no_grad():
            pred_obs = model(obs, action)

        loss_cont, loss_bin, loss_cat, loss_thr = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
        loss_tot = loss_cont + loss_bin + loss_cat + loss_thr

        continuous_losses.append(loss_cont.item())
        binary_losses.append(loss_bin.item())
        categorical_losses.append(loss_cat.item())
        threshold_losses.append(loss_thr.item())
        total_losses.append(loss_tot.item())
        timer.start()
    timer.stop()

    continuous_loss = sum(continuous_losses) / len(continuous_losses)
    binary_loss = sum(binary_losses) / len(binary_losses)
    categorical_loss = sum(categorical_losses) / len(categorical_losses)
    threshold_loss = sum(threshold_losses) / len(threshold_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()

    wlog["eval_loss/continuous"] = continuous_loss
    wlog["eval_loss/binary"] = binary_loss
    wlog["eval_loss/categorical"] = categorical_loss
    wlog["eval_loss/threshold"] = threshold_loss
    wlog["eval_loss/total"] = total_loss
    wlog["eval_dataset/wait_time_s"] = total_wait

    return total_loss
