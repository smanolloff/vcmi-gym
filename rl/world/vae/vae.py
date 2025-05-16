import torch
import torch.nn as nn
import math
import enum
import contextlib
import random
import torch.nn.functional as F
import pandas as pd

from ..util.buffer_base import BufferBase
from ..util.dataset_vcmi import Data, Context
from ..util.misc import layer_init
from ..util.obs_index import ObsIndex, Group, ContextGroup, DataGroup
from ..util.timer import Timer
from ..util.misc import TableColumn

from ..util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    N_HEX_ACTIONS,
    HEX_ACT_MAP,
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

#
# Action distribution of random agent vs. StupidAI:
#
#  prob | action
# ------|---------
# 0.021 | AMOVE_TR
# 0.036 | AMOVE_R
# 0.098 | AMOVE_BR
# 0.057 | AMOVE_BL
# 0.029 | AMOVE_L
# 0.016 | AMOVE_TL
# 0.003 | AMOVE_2TR
# 0.006 | AMOVE_2R
# 0.017 | AMOVE_2BR
# 0.007 | AMOVE_2BL
# 0.003 | AMOVE_2L
# 0.021 | AMOVE_2TL
# 0.471 | MOVE
# 0.213 | SHOOT
#
#

def vcmi_dataloader_functor():
    state = {"reward_carry": 0}

    def mw(data: Data, ctx: Context):
        if ctx.transition_id == ctx.num_transitions - 1:
            state["reward_carry"] = data.reward
            if not data.done:
                return None

        if (data.action - 2) % len(HEX_ACT_MAP) == HEX_ACT_MAP["MOVE"]:
            # Skip 50% of MOVEs
            if random.random() < 0.5:
                return None

        if ctx.transition_id == 0 and ctx.ep_steps > 0:
            data = data._replace(reward=state["reward_carry"])

        return data

    return mw


class Buffer(BufferBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_counters = torch.zeros(N_HEX_ACTIONS).long()
        self.tmp_counter = 0

    def _valid_indices(self):
        max_index = self.capacity if self.full else self.index
        # Valid are indices of samples where done=False and cutoff=False
        # (i.e. to ensure obs,next_obs is valid)
        # XXX: float->bool conversion is OK given floats are exactly 1 or 0
        ok_samples = ~self.containers["done"][:max_index - 1].bool()
        ok_samples[self.worker_cutoffs] = False
        return torch.nonzero(ok_samples, as_tuple=True)[0]

    def add_batch(self, data):
        self.action_counters.add_(torch.bincount((data.action - 2) % N_HEX_ACTIONS, minlength=N_HEX_ACTIONS))
        self.tmp_counter += len(data.action)
        if self.tmp_counter > 1_000_000:
            total = self.action_counters.sum()
            self.logger.info("Action dist after %d samples: %s" % (total, (self.action_counters / total).tolist()))
            self.tmp_counter = 0
        super().add_batch(data)

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


class Encoder(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.obs_index = ObsIndex(device)

        self.abs_index = self.obs_index.abs_index
        self.rel_index = self.obs_index.rel_index

        emb_calc = lambda n: math.ceil(math.sqrt(n))

        self.encoder_action = nn.Embedding(N_ACTIONS, emb_calc(N_ACTIONS))

        #
        # Global encoders
        #

        # Continuous:
        # (B, n)
        self.encoder_global_cont_abs = nn.Identity()
        self.encoder_global_cont_rel = nn.Identity()

        # Continuous (nulls):
        # (B, n)
        self.encoder_global_cont_nullbit = nn.Identity()
        global_nullbit_size = len(self.rel_index[Group.GLOBAL][Group.CONT_NULLBIT])
        if global_nullbit_size:
            self.encoder_global_cont_nullbit = nn.LazyLinear(global_nullbit_size)
            # No nonlinearity needed?

        # Binaries:
        # [(B, b1), (B, b2), ...]
        self.encoders_global_binaries = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.BINARIES]:
            self.encoders_global_binaries.append(nn.LazyLinear(len(ind)))
            # No nonlinearity needed?

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        self.encoders_global_categoricals = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.CATEGORICALS]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_global_categoricals.append(cat_emb_size)

        # Thresholds:
        # [(B, T1), (B, T2), ...]
        self.encoders_global_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.THRESHOLDS]:
            self.encoders_global_thresholds.append(nn.LazyLinear(len(ind)))
            # No nonlinearity needed?

        # Merge
        z_size_global = 256
        self.encoder_merged_global = nn.Sequential(
            # => (B, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            nn.LazyLinear(z_size_global),
            nn.LeakyReLU(),
        )
        # => (B, Z_GLOBAL)

        #
        # Player encoders
        #

        # Continuous per player:
        # (B, n)
        self.encoder_player_cont_abs = nn.Identity()
        self.encoder_player_cont_rel = nn.Identity()

        # Continuous (nulls) per player:
        # (B, n)
        self.encoder_player_cont_nullbit = nn.Identity()
        player_nullbit_size = len(self.rel_index[Group.PLAYER][Group.CONT_NULLBIT])
        if player_nullbit_size:
            self.encoder_player_cont_nullbit = nn.LazyLinear(player_nullbit_size)
            # No nonlinearity needed?

        # Binaries per player:
        # [(B, b1), (B, b2), ...]
        self.encoders_player_binaries = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.BINARIES]:
            self.encoders_player_binaries.append(nn.LazyLinear(len(ind)))
            # No nonlinearity needed?

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        self.encoders_player_categoricals = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.CATEGORICALS]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_player_categoricals.append(cat_emb_size)

        # Thresholds per player:
        # [(B, T1), (B, T2), ...]
        self.encoders_player_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.THRESHOLDS]:
            self.encoders_player_thresholds.append(nn.LazyLinear(len(ind)))

        # Merge per player
        z_size_player = 256
        self.encoder_merged_player = nn.Sequential(
            # => (B, 2, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            nn.LazyLinear(z_size_player),
            nn.LeakyReLU(),
        )
        # => (B, 2, Z_PLAYER)

        #
        # Hex encoders
        #

        # Continuous per hex:
        # (B, n)
        self.encoder_hex_cont_abs = nn.Identity()
        self.encoder_hex_cont_rel = nn.Identity()

        # Continuous (nulls) per hex:
        # (B, n)
        self.encoder_hex_cont_nullbit = nn.Identity()
        hex_nullbit_size = len(self.rel_index[Group.HEX][Group.CONT_NULLBIT])
        if hex_nullbit_size:
            self.encoder_hex_cont_nullbit = nn.LazyLinear(hex_nullbit_size)
            # No nonlinearity needed?

        # Binaries per hex:
        # [(B, b1), (B, b2), ...]
        self.encoders_hex_binaries = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.BINARIES]:
            self.encoders_hex_binaries.append(nn.LazyLinear(len(ind)))
            # No nonlinearity needed?

        # Categoricals per hex:
        # [(B, C1), (B, C2), ...]
        self.encoders_hex_categoricals = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.CATEGORICALS]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_hex_categoricals.append(cat_emb_size)

        # Thresholds per hex:
        # [(B, T1), (B, T2), ...]
        self.encoders_hex_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.THRESHOLDS]:
            self.encoders_hex_thresholds.append(nn.LazyLinear(len(ind)))

        # Merge per hex
        z_size_hex = 512
        self.encoder_merged_hex = nn.Sequential(
            # => (B, 165, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            nn.LazyLinear(z_size_hex),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3)
        )
        # => (B, 165, Z_HEX)

        # Transformer (hexes only)
        self.transformer_hex = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=z_size_hex, nhead=8, dropout=0.3, batch_first=True),
            num_layers=6
        )
        # => (B, 165, Z_HEX)

        #
        # Aggregator
        #

        # (B, Z_GLOBAL + AVG(2*Z_PLAYER) + AVG(165*Z_HEX))
        self.aggregator = nn.Sequential(
            nn.LazyLinear(2048),
            nn.LeakyReLU(),
        )
        # => (B, Z_AGG)

        # (B, Z_AGG)
        self.encoder_mu = nn.LazyLinear(1024)
        self.encoder_logvar = nn.LazyLinear(1024)

    def forward(self, obs, action):
        action_z = self.encoder_action(action)
        return self._forward(obs, action_z)

    def _forward(self, obs, action_z):
        assert obs.device.type == self.device.type, f"{obs.device.type} == {self.device.type}"

        # torch.cat which returns empty tensor if tuple is empty
        def torch_cat(tuple_of_tensors, **kwargs):
            if len(tuple_of_tensors) == 0:
                return torch.tensor([], device=self.device)
            return torch.cat(tuple_of_tensors, **kwargs)

        global_cont_abs_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]]
        global_cont_rel_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]]
        global_cont_nullbit_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]]
        global_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.BINARIES]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.CATEGORICALS]]
        global_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.THRESHOLDS]]
        global_cont_abs_z = self.encoder_global_cont_abs(global_cont_abs_in)
        global_cont_rel_z = self.encoder_global_cont_rel(global_cont_rel_in)
        global_cont_nullbit_z = self.encoder_global_cont_nullbit(global_cont_nullbit_in)
        global_binary_z = torch_cat([lin(x) for lin, x in zip(self.encoders_global_binaries, global_binary_ins)], dim=-1)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch_cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_threshold_z = torch_cat([lin(x) for lin, x in zip(self.encoders_global_thresholds, global_threshold_ins)], dim=-1)
        global_merged = torch_cat((global_cont_abs_z, global_cont_rel_z, global_cont_nullbit_z, global_binary_z, global_categorical_z, global_threshold_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_cont_abs_in = obs[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]]
        player_cont_rel_in = obs[:, self.abs_index[Group.PLAYER][Group.CONT_REL]]
        player_cont_nullbit_in = obs[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]]
        player_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.PLAYER][Group.BINARIES]]
        player_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.PLAYER][Group.CATEGORICALS]]
        player_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.PLAYER][Group.THRESHOLDS]]
        player_cont_abs_z = self.encoder_player_cont_abs(player_cont_abs_in)
        player_cont_rel_z = self.encoder_player_cont_rel(player_cont_rel_in)
        player_cont_nullbit_z = self.encoder_player_cont_nullbit(player_cont_nullbit_in)
        player_binary_z = torch_cat([lin(x) for lin, x in zip(self.encoders_player_binaries, player_binary_ins)], dim=-1)
        player_categorical_z = torch_cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_threshold_z = torch_cat([lin(x) for lin, x in zip(self.encoders_player_thresholds, player_threshold_ins)], dim=-1)
        player_merged = torch_cat((player_cont_abs_z, player_cont_rel_z, player_cont_nullbit_z, player_binary_z, player_categorical_z, player_threshold_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_cont_abs_in = obs[:, self.abs_index[Group.HEX][Group.CONT_ABS]]
        hex_cont_rel_in = obs[:, self.abs_index[Group.HEX][Group.CONT_REL]]
        hex_cont_nullbit_in = obs[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]]
        hex_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.HEX][Group.BINARIES]]
        hex_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.HEX][Group.CATEGORICALS]]
        hex_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.HEX][Group.THRESHOLDS]]
        hex_cont_abs_z = self.encoder_hex_cont_abs(hex_cont_abs_in)
        hex_cont_rel_z = self.encoder_hex_cont_rel(hex_cont_rel_in)
        hex_cont_nullbit_z = self.encoder_hex_cont_nullbit(hex_cont_nullbit_in)
        hex_binary_z = torch_cat([lin(x) for lin, x in zip(self.encoders_hex_binaries, hex_binary_ins)], dim=-1)
        hex_categorical_z = torch_cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)], dim=-1)
        hex_threshold_z = torch_cat([lin(x) for lin, x in zip(self.encoders_hex_thresholds, hex_threshold_ins)], dim=-1)
        hex_merged = torch_cat((hex_cont_abs_z, hex_cont_rel_z, hex_cont_nullbit_z, hex_binary_z, hex_categorical_z, hex_threshold_z), dim=-1)
        z_hex = self.encoder_merged_hex(hex_merged)
        z_hex = self.transformer_hex(z_hex)
        # => (B, 165, Z_HEX)

        mean_z_player = z_player.mean(dim=1)
        mean_z_hex = z_hex.mean(dim=1)
        z_agg = self.aggregator(torch.cat([z_global, mean_z_player, mean_z_hex], dim=-1))
        # => (B, Z_AGG)

        mu = self.encoder_mu(z_agg)
        logvar = self.encoder_logvar(z_agg)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        # (B, Z_AGG)
        self.transition = nn.Sequential(
            nn.LazyLinear(2048),
            nn.LeakyReLU(),
        )
        # => (B, Z_TRANS)

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

        # torch.cat which returns empty tensor if tuple is empty
        def torch_cat(tuple_of_tensors, **kwargs):
            if len(tuple_of_tensors) == 0:
                return torch.tensor([], device=self.device)
            return torch.cat(tuple_of_tensors, **kwargs)

        global_cont_abs_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]]
        global_cont_rel_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]]
        global_cont_nullbit_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]]
        global_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.BINARIES]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.CATEGORICALS]]
        global_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.THRESHOLDS]]
        global_cont_abs_z = self.encoder_global_cont_abs(global_cont_abs_in)
        global_cont_rel_z = self.encoder_global_cont_rel(global_cont_rel_in)
        global_cont_nullbit_z = self.encoder_global_cont_nullbit(global_cont_nullbit_in)
        global_binary_z = torch_cat([lin(x) for lin, x in zip(self.encoders_global_binaries, global_binary_ins)], dim=-1)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch_cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_threshold_z = torch_cat([lin(x) for lin, x in zip(self.encoders_global_thresholds, global_threshold_ins)], dim=-1)
        global_merged = torch_cat((global_cont_abs_z, global_cont_rel_z, global_cont_nullbit_z, global_binary_z, global_categorical_z, global_threshold_z), dim=-1)
        # global_merged = torch_cat((action_z, global_cont_abs_z, global_cont_rel_z, global_cont_nullbit_z, global_binary_z, global_categorical_z, global_threshold_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_cont_abs_in = obs[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]]
        player_cont_rel_in = obs[:, self.abs_index[Group.PLAYER][Group.CONT_REL]]
        player_cont_nullbit_in = obs[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]]
        player_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.PLAYER][Group.BINARIES]]
        player_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.PLAYER][Group.CATEGORICALS]]
        player_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.PLAYER][Group.THRESHOLDS]]
        player_cont_abs_z = self.encoder_player_cont_abs(player_cont_abs_in)
        player_cont_rel_z = self.encoder_player_cont_rel(player_cont_rel_in)
        player_cont_nullbit_z = self.encoder_player_cont_nullbit(player_cont_nullbit_in)
        player_binary_z = torch_cat([lin(x) for lin, x in zip(self.encoders_player_binaries, player_binary_ins)], dim=-1)
        player_categorical_z = torch_cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_threshold_z = torch_cat([lin(x) for lin, x in zip(self.encoders_player_thresholds, player_threshold_ins)], dim=-1)
        player_merged = torch_cat((player_cont_abs_z, player_cont_rel_z, player_cont_nullbit_z, player_binary_z, player_categorical_z, player_threshold_z), dim=-1)
        # player_merged = torch_cat((action_z.unsqueeze(1).expand(-1, 2, -1), player_cont_abs_z, player_cont_rel_z, player_cont_nullbit_z, player_binary_z, player_categorical_z, player_threshold_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_cont_abs_in = obs[:, self.abs_index[Group.HEX][Group.CONT_ABS]]
        hex_cont_rel_in = obs[:, self.abs_index[Group.HEX][Group.CONT_REL]]
        hex_cont_nullbit_in = obs[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]]
        hex_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.HEX][Group.BINARIES]]
        hex_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.HEX][Group.CATEGORICALS]]
        hex_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.HEX][Group.THRESHOLDS]]
        hex_cont_abs_z = self.encoder_hex_cont_abs(hex_cont_abs_in)
        hex_cont_rel_z = self.encoder_hex_cont_rel(hex_cont_rel_in)
        hex_cont_nullbit_z = self.encoder_hex_cont_nullbit(hex_cont_nullbit_in)
        hex_binary_z = torch_cat([lin(x) for lin, x in zip(self.encoders_hex_binaries, hex_binary_ins)], dim=-1)
        hex_categorical_z = torch_cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)], dim=-1)
        hex_threshold_z = torch_cat([lin(x) for lin, x in zip(self.encoders_hex_thresholds, hex_threshold_ins)], dim=-1)
        hex_merged = torch_cat((hex_cont_abs_z, hex_cont_rel_z, hex_cont_nullbit_z, hex_binary_z, hex_categorical_z, hex_threshold_z), dim=-1)
        # hex_merged = torch_cat((action_z.unsqueeze(1).expand(-1, 165, -1), hex_cont_abs_z, hex_cont_rel_z, hex_cont_nullbit_z, hex_binary_z, hex_categorical_z, hex_threshold_z), dim=-1)
        z_hex = self.encoder_merged_hex(hex_merged)
        z_hex = self.transformer_hex(z_hex)
        # => (B, 165, Z_HEX)

        mean_z_player = z_player.mean(dim=1)
        mean_z_hex = z_hex.mean(dim=1)
        z_agg = self.aggregator(torch.cat([z_global, mean_z_player, mean_z_hex], dim=-1))
        # => (B, Z_AGG)

        z_trans = self.transition(torch.cat([action_z, z_agg], dim=-1))
        # => (B, Z_TRANS)

        #
        # Outputs
        #

        global_out = self.head_global(z_trans)
        # => (B, STATE_SIZE_GLOBAL)

        player_out = self.head_player(torch.cat([z_trans.unsqueeze(1).expand(-1, 2, -1), z_player], dim=-1))
        # => (B, 2, STATE_SIZE_ONE_PLAYER)

        hex_out = self.head_hex(torch.cat([z_trans.unsqueeze(1).expand(-1, 165, -1), z_hex], dim=-1))
        # => (B, 165, STATE_SIZE_ONE_HEX)

        obs_out = torch.cat((global_out, player_out.flatten(start_dim=1), hex_out.flatten(start_dim=1)), dim=1)

        return obs_out

    def reconstruct(self, obs_out, strategy=Reconstruction.GREEDY):
        global_cont_abs_out = obs_out[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]]
        global_cont_rel_out = obs_out[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]]
        global_cont_nullbit_out = obs_out[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]]
        global_binary_outs = [obs_out[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.BINARIES]]
        global_categorical_outs = [obs_out[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.CATEGORICALS]]
        global_threshold_outs = [obs_out[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.THRESHOLDS]]
        player_cont_abs_out = obs_out[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]]
        player_cont_rel_out = obs_out[:, self.abs_index[Group.PLAYER][Group.CONT_REL]]
        player_cont_nullbit_out = obs_out[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]]
        player_binary_outs = [obs_out[:, ind] for ind in self.abs_index[Group.PLAYER][Group.BINARIES]]
        player_categorical_outs = [obs_out[:, ind] for ind in self.abs_index[Group.PLAYER][Group.CATEGORICALS]]
        player_threshold_outs = [obs_out[:, ind] for ind in self.abs_index[Group.PLAYER][Group.THRESHOLDS]]
        hex_cont_abs_out = obs_out[:, self.abs_index[Group.HEX][Group.CONT_ABS]]
        hex_cont_rel_out = obs_out[:, self.abs_index[Group.HEX][Group.CONT_REL]]
        hex_cont_nullbit_out = obs_out[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]]
        hex_binary_outs = [obs_out[:, ind] for ind in self.abs_index[Group.HEX][Group.BINARIES]]
        hex_categorical_outs = [obs_out[:, ind] for ind in self.abs_index[Group.HEX][Group.CATEGORICALS]]
        hex_threshold_outs = [obs_out[:, ind] for ind in self.abs_index[Group.HEX][Group.THRESHOLDS]]
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

        elif strategy == Reconstruction.SAMPLES:
            def reconstruct_binary(logits):
                return torch.bernoulli(logits.sigmoid())

            def reconstruct_categorical(logits):
                num_classes = logits.shape[-1]
                probs_2d = logits.softmax(dim=-1).view(-1, num_classes)
                sampled_classes = torch.multinomial(probs_2d, num_samples=1).view(logits.shape[:-1])
                return F.one_hot(sampled_classes, num_classes=num_classes).float()

        elif strategy == Reconstruction.GREEDY:
            def reconstruct_binary(logits):
                return logits.sigmoid().round()

            def reconstruct_categorical(logits):
                return F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()

        next_obs[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]] = reconstruct_continuous(global_cont_abs_out)
        next_obs[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]] = reconstruct_continuous(global_cont_rel_out)
        next_obs[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]] = reconstruct_continuous(global_cont_nullbit_out)
        for ind, out in zip(self.abs_index[Group.GLOBAL][Group.BINARIES], global_binary_outs):
            next_obs[:, ind] = reconstruct_binary(out)
        for ind, out in zip(self.abs_index[Group.GLOBAL][Group.CATEGORICALS], global_categorical_outs):
            next_obs[:, ind] = reconstruct_categorical(out)
        for ind, out in zip(self.abs_index[Group.GLOBAL][Group.THRESHOLDS], global_threshold_outs):
            next_obs[:, ind] = reconstruct_binary(out)

        next_obs[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]] = reconstruct_continuous(player_cont_abs_out)
        next_obs[:, self.abs_index[Group.PLAYER][Group.CONT_REL]] = reconstruct_continuous(player_cont_rel_out)
        next_obs[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]] = reconstruct_continuous(player_cont_nullbit_out)
        for ind, out in zip(self.abs_index[Group.PLAYER][Group.BINARIES], player_binary_outs):
            next_obs[:, ind] = reconstruct_binary(out)
        for ind, out in zip(self.abs_index[Group.PLAYER][Group.CATEGORICALS], player_categorical_outs):
            next_obs[:, ind] = reconstruct_categorical(out)
        for ind, out in zip(self.abs_index[Group.PLAYER][Group.THRESHOLDS], player_threshold_outs):
            next_obs[:, ind] = reconstruct_binary(out)

        next_obs[:, self.abs_index[Group.HEX][Group.CONT_ABS]] = reconstruct_continuous(hex_cont_abs_out)
        next_obs[:, self.abs_index[Group.HEX][Group.CONT_REL]] = reconstruct_continuous(hex_cont_rel_out)
        next_obs[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]] = reconstruct_continuous(hex_cont_nullbit_out)
        for ind, out in zip(self.abs_index[Group.HEX][Group.BINARIES], hex_binary_outs):
            next_obs[:, ind] = reconstruct_binary(out)
        for ind, out in zip(self.abs_index[Group.HEX][Group.CATEGORICALS], hex_categorical_outs):
            next_obs[:, ind] = reconstruct_categorical(out)
        for ind, out in zip(self.abs_index[Group.HEX][Group.THRESHOLDS], hex_threshold_outs):
            next_obs[:, ind] = reconstruct_binary(out)

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


def _compute_losses(logits, target, index, weights, device=torch.device("cpu")):
    # Aggregate each feature's loss across players/hexes

    if logits[Group.CONT_ABS].dim() == 3:
        # (B, 165, N_FEATS)
        def sum_repeats(loss):
            return loss.sum(dim=1)
    else:
        # (B, N_FEATS)
        def sum_repeats(loss):
            return loss

    losses = {}

    for dgroup in DataGroup.as_list():
        if not len(logits[dgroup]):
            losses[dgroup] = torch.tensor([], device=device)
            continue

        lgt = logits[dgroup]
        tgt = target[dgroup]

        if dgroup in [Group.CONT_ABS, Group.CONT_REL]:
            # (B, N_CONTABS_FEATS)             when t=Group.GLOBAL
            # (B, 2, N_CONTABS_FEATS)          when t=Group.PLAYER
            # (B, 165, N_CONTABS_FEATS)        when t=Group.HEX
            losses[dgroup] = sum_repeats(F.mse_loss(lgt, tgt, reduction="none")).mean(dim=0)
            # => (N_CONT_FEATS)

        elif dgroup == Group.CONT_NULLBIT:
            # (B, N_EXPLICIT_NULL_CONT_FEATS)      when t=Group.GLOBAL
            # (B, 2, N_EXPLICIT_NULL_CONT_FEATS)   when t=Group.PLAYER
            # (B, 165, N_EXPLICIT_NULL_CONT_FEATS) when t=Group.HEX
            losses[dgroup] = sum_repeats(F.binary_cross_entropy_with_logits(lgt, tgt, reduction="none")).mean(dim=0)
            # => (N_EXPLICIT_NULL_CONT_FEATS)

        elif dgroup == Group.BINARIES:
            loss = torch.zeros(len(lgt), device=device)
            for i, (i_lgt, i_tgt) in enumerate(zip(lgt, tgt)):
                # (B, N_BIN_FEATi_BITS)      when t=Group.GLOBAL
                # (B, 2, N_BIN_FEATi_BITS)   when t=Group.PLAYER
                # (B, 165, N_BIN_FEATi_BITS) when t=Group.HEX

                # XXX:
                # reduction="none" would result in same-as-input shape result
                # ...but having separate losses for each bit would be too much
                # ... If separate weights are needed for each bit, then maybe dont reduce it...
                # => for now, just reduce it to a single loss per feature
                loss[i] = sum_repeats(F.binary_cross_entropy_with_logits(i_lgt, i_tgt, reduction="none")).mean()
                # (1)  # single loss the i'th binary feat
            losses[dgroup] = loss
            # (N_BIN_FEATS)

        elif dgroup == Group.CATEGORICALS:
            loss = torch.zeros(len(lgt), device=device)
            for i, (i_lgt, i_tgt) in enumerate(zip(lgt, tgt)):
                # (B, N_CAT_FEATi_CLASSES)      when t=Group.GLOBAL
                # (B, 2, N_CAT_FEATi_CLASSES)   when t=Group.PLAYER
                # (B, 165, N_CAT_FEATi_CLASSES) when t=Group.HEX

                if i_lgt.dim() == 3:
                    # XXX: CrossEntropyLoss expects (B, C, *) input where C=num_classes
                    #      => transpose (B, 165, C) => (B, C, 165)
                    #      (not needed for BCE or MSE)
                    i_lgt = i_lgt.swapaxes(1, 2)
                    i_tgt = i_tgt.swapaxes(1, 2)

                # XXX: cross_entropy always removes last dim (even with reduction=none)
                loss[i] = sum_repeats(F.cross_entropy(i_lgt, i_tgt, reduction="none")).mean(dim=0)
                # (1)  # single loss for the i'th categorical feature
            losses[dgroup] = loss
            # (N_CAT_FEATS)

        elif dgroup == Group.THRESHOLDS:
            loss = torch.zeros(len(lgt), device=device)

            for i, (i_lgt, i_tgt) in enumerate(zip(lgt, tgt)):
                # (B, N_THR_FEATi_BINS)      when t=Group.GLOBAL
                # (B, 2, N_THR_FEATi_BINS)   when t=Group.PLAYER
                # (B, 165, N_THR_FEATi_BINS) when t=Group.HEX

                bce_loss = sum_repeats(F.binary_cross_entropy_with_logits(i_lgt, i_tgt, reduction="none")).mean()
                # (1)  # single loss for the i'th global threshold feature

                # Monotonicity regularization:
                #   > t1
                #   => tensor([ 0.6004,  1.3230,  1.0605,  0.7150, -0.2482])
                #       ^ raw logits
                #
                #   > probs = torch.sigmoid(t1)
                #   => tensor([0.6458, 0.7897, 0.7428, 0.6715, 0.4383])
                #       (individual prob for each bit)
                #
                #   > violation = probs[..., 1:] - probs[..., :-1]
                #   => tensor([[-0.0460,  0.3775, -0.1651,  0.0951],
                #              [-0.2894,  0.1434, -0.3976,  0.4508]])
                #               ^p1-p0    ^p2-p1  ^p3-p2    ^p4-p3
                #
                #       Values show if the next bit has *higher* prob
                #       (threshold encoding should never have increasing probs)
                #
                #   > loss = torch.relu(violation)
                #   => tensor([[0.0000, 0.3775, 0.0000, 0.0951],
                #              [0.0000, 0.1434, 0.0000, 0.4508]])
                #       (negative violations i.e. decreasing probs are not a loss)
                #
                #  i.e. loss only where tne next is higher prob than current
                probs = torch.sigmoid(i_lgt)
                mono_diff = probs[..., 1:] - probs[..., :-1]
                mono_loss = F.relu(mono_diff).mean()  # * 1.0  (optional lambda coefficient)
                loss[i] = (bce_loss + mono_loss)
                # (1)  # single loss for the i'th global threshold feature
            losses[dgroup] = loss
            # (N_THR_FEATS)
        else:
            raise Exception("unexpected dgroup: %s" % dgroup)

        losses[dgroup] *= weights[dgroup]

    return losses


def compute_losses(logger, abs_index, loss_weights, next_obs, pred_obs):
    # For shapes, see ObsIndex._build_abs_indices()
    extract = lambda t, obs: {
        Group.CONT_ABS: obs[:, abs_index[t][Group.CONT_ABS]],
        Group.CONT_REL: obs[:, abs_index[t][Group.CONT_REL]],
        Group.CONT_NULLBIT: obs[:, abs_index[t][Group.CONT_NULLBIT]],
        Group.BINARIES: [obs[:, ind] for ind in abs_index[t][Group.BINARIES]],
        Group.CATEGORICALS: [obs[:, ind] for ind in abs_index[t][Group.CATEGORICALS]],
        Group.THRESHOLDS: [obs[:, ind] for ind in abs_index[t][Group.THRESHOLDS]],
    }

    losses = {}
    device = next_obs.device
    total_loss = torch.tensor(0., device=pred_obs.device)

    for cgroup in ContextGroup.as_list():
        logits = extract(cgroup, pred_obs)
        target = extract(cgroup, next_obs)
        index = abs_index[cgroup]
        weights = loss_weights[cgroup]
        losses[cgroup] = _compute_losses(logits, target, index, weights=weights, device=device)
        total_loss += sum(subtype_losses.sum() for subtype_losses in losses[cgroup].values())

    return total_loss, losses


def losses_to_rows(losses, obs_index):
    rows = []
    for context, datatype_groups in losses.items():
        # global/player/hex
        for typename, typeloss in datatype_groups.items():
            # continuous/cont_nullbit/binaries/...
            for i in range(typeloss.shape[0]):
                attr_id = obs_index.attr_ids[context][typename][i]
                attr_name = obs_index.attr_names[context][attr_id]
                rows.append({
                    TableColumn.ATTRIBUTE: attr_name,
                    TableColumn.CONTEXT: context,
                    TableColumn.DATATYPE: typename,
                    TableColumn.LOSS: typeloss[i].item()
                })
    return rows


# Aggregate batch losses into a *single* loss per attribute
def rows_to_df(rows):
    return pd.DataFrame(rows).groupby([
        TableColumn.ATTRIBUTE,
        TableColumn.CONTEXT,
        TableColumn.DATATYPE
    ], as_index=False)[TableColumn.LOSS].mean()


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
):
    assert buffer.capacity % batch_size == 0, f"{buffer.capacity} % {batch_size} == 0"

    maybe_autocast = torch.amp.autocast(model.device.type) if scaler else contextlib.nullcontext()

    model.train()
    timer = Timer()
    loss_rows = []

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
                loss_tot, losses = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)

            loss_rows.extend(losses_to_rows(losses, model.obs_index))

            if accumulate_grad:
                if scaler:
                    # XXX: loss_tot / grad_steps should be within autocast
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

    return rows_to_df(loss_rows), timer.peek()


def eval_model(
    logger,
    model,
    loss_weights,
    buffer,
    batch_size,
):
    model.eval()
    timer = Timer()
    loss_rows = []

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, action, next_obs, next_mask, next_rew, next_done = batch

        with torch.no_grad():
            pred_obs = model(obs, action)

        loss_tot, losses = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
        loss_rows.extend(losses_to_rows(losses, model.obs_index))
        timer.start()

    return rows_to_df(loss_rows), timer.peek()
