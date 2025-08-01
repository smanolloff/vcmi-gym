import torch
import torch.nn as nn
import math
import enum
import contextlib
import torch.nn.functional as F
import pandas as pd

from ..util.buffer_base import BufferBase
from ..util.dataset_vcmi import Data, Context, DataInstruction
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
        instruction = DataInstruction.USE

        # Always skip last transition (it is identical to the next first transition)
        if ctx.transition_id == ctx.num_transitions - 1:
            state["reward_carry"] = data.reward
            if not data.done:
                instruction = DataInstruction.SKIP

        if ctx.transition_id == 0 and ctx.ep_steps > 0:
            data = data._replace(reward=state["reward_carry"])

        # XXX:
        # SKIP instructions MUST NOT be used to promote more AMOVE samples here
        # as this results in inconsistent transitions in the buffer.

        return data, instruction

    return mw


class Buffer(BufferBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_counters = torch.zeros(N_HEX_ACTIONS).long()
        self.tmp_counter = 0

    def _valid_indices(self):
        max_index = self.capacity if self.full else self.index
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
        inds = torch.randperm(self.capacity, device=self.device)[:batch_size]
        return self.containers["obs"][inds]

    def sample_iter(self, batch_size):
        inds = torch.randperm(self.capacity, device=self.device)
        for i in range(0, len(inds), batch_size):
            batch_indices = inds[i:i + batch_size]
            yield self.containers["obs"][batch_indices]


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
            cat_emb = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_global_categoricals.append(cat_emb)

        # Thresholds:
        # [(B, T1), (B, T2), ...]
        self.encoders_global_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.THRESHOLDS]:
            self.encoders_global_thresholds.append(nn.LazyLinear(len(ind)))
            # No nonlinearity needed?

        # Merge
        self.z_size_global = 256
        self.encoder_merged_global = nn.Sequential(
            # => (B, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            nn.LazyLinear(self.z_size_global),
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
        self.player_nullbit_size = len(self.rel_index[Group.PLAYER][Group.CONT_NULLBIT])
        if self.player_nullbit_size:
            self.encoder_player_cont_nullbit = nn.LazyLinear(self.player_nullbit_size)
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
            cat_emb = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_player_categoricals.append(cat_emb)

        # Thresholds per player:
        # [(B, T1), (B, T2), ...]
        self.encoders_player_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.THRESHOLDS]:
            self.encoders_player_thresholds.append(nn.LazyLinear(len(ind)))

        # Merge per player
        self.z_size_player = 256
        self.encoder_merged_player = nn.Sequential(
            # => (B, 2, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            nn.LazyLinear(self.z_size_player),
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
        self.hex_nullbit_size = len(self.rel_index[Group.HEX][Group.CONT_NULLBIT])
        if self.hex_nullbit_size:
            self.encoder_hex_cont_nullbit = nn.LazyLinear(self.hex_nullbit_size)
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
            cat_emb = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_hex_categoricals.append(cat_emb)

        # Thresholds per hex:
        # [(B, T1), (B, T2), ...]
        self.encoders_hex_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.THRESHOLDS]:
            self.encoders_hex_thresholds.append(nn.LazyLinear(len(ind)))

        # Merge per hex
        self.z_size_hex = 512
        self.encoder_merged_hex = nn.Sequential(
            # => (B, 165, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            nn.LazyLinear(self.z_size_hex),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3)
        )
        # => (B, 165, Z_HEX)

        # Transformer (hexes only)
        self.transformer_hex = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.z_size_hex, nhead=8, dropout=0.3, batch_first=True),
            num_layers=6,
            norm=nn.LayerNorm(self.z_size_hex)
        )
        # => (B, 165, Z_HEX)

        #
        # Aggregator
        #

        # (B, Z_GLOBAL + AVG(2*Z_PLAYER) + AVG(165*Z_HEX))
        self.z_size_agg = 2048
        self.aggregator = nn.Sequential(
            nn.LazyLinear(self.z_size_agg),
            nn.LeakyReLU(),
        )
        # => (B, Z_AGG)

        #
        # Heads
        #

        # (B, Z_AGG)
        self.z_size = 1024
        self.encoder_mu = nn.LazyLinear(self.z_size)
        self.encoder_logvar = nn.LazyLinear(self.z_size)
        # (B, Z_AGG)

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            obs = torch.randn([2, DIM_OBS], device=device)
            self.forward(obs)

        layer_init(self)

    def forward(self, obs):
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
        z = self.aggregator(torch.cat([z_global, mean_z_player, mean_z_hex], dim=-1))
        # => (B, Z_AGG)

        #
        # Outputs
        #

        mu = self.encoder_mu(z)
        logvar = self.encoder_logvar(z)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.device = encoder.device
        self.abs_index = encoder.abs_index
        self.rel_index = encoder.rel_index

        self.z_size_hex = encoder.z_size_hex
        self.z_size_player = encoder.z_size_player
        self.z_size_global = encoder.z_size_global

        #
        # Aggregator decoder
        #
        self.z_decoder = nn.Sequential(
            nn.LazyLinear(self.z_size_global + self.z_size_player + self.z_size_hex),
            nn.LeakyReLU()
        )
        # => (B, Z_GLOBAL + Z_PLAYER + Z_HEX)

        #
        # Hex decoders
        #

        # learned positional embeddings for 165 hexes
        self.hex_pos_embed = nn.Parameter(torch.randn(1, 165, self.z_size_hex))
        # => (B, 165, Z_HEX)

        self.transformer_decoder_hex = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.z_size_hex,
                nhead=encoder.transformer_hex.layers[0].self_attn.num_heads,
                dropout=encoder.transformer_hex.layers[0].self_attn.dropout,
                batch_first=True
            ),
            num_layers=len(encoder.transformer_hex.layers),
            norm=nn.LayerNorm(self.z_size_hex)
        )
        # => (B, 165, Z_HEX)

        self.z_size_hex_threshold = sum(lin.out_features for lin in encoder.encoders_hex_thresholds)
        self.z_size_hex_categorical = sum(emb.embedding_dim for emb in encoder.encoders_hex_categoricals)
        self.z_size_hex_binary = sum(lin.out_features for lin in encoder.encoders_hex_binaries)
        self.z_size_hex_cont_nullbit = getattr(encoder.encoder_hex_cont_nullbit, "out_features", 0)
        self.z_size_hex_cont_rel = len(self.rel_index[Group.HEX][Group.CONT_REL])
        self.z_size_hex_cont_abs = len(self.rel_index[Group.HEX][Group.CONT_ABS])
        self.z_size_hex_merged = (
            self.z_size_hex_threshold
            + self.z_size_hex_categorical
            + self.z_size_hex_binary
            + self.z_size_hex_cont_nullbit
            + self.z_size_hex_cont_rel
            + self.z_size_hex_cont_abs
        )

        self.decoder_merged_hex = nn.Sequential(
            # => (B, 165, Z_HEX)
            nn.LazyLinear(self.z_size_hex_merged),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3)
        )
        # => (B, 165, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)

        self.decoders_hex_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.THRESHOLDS]:
            self.decoders_hex_thresholds.append(nn.LazyLinear(len(ind)))

        self.decoders_hex_categoricals = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.CATEGORICALS]:
            # Simply use a Linear layer as the inverse of Embedding
            self.decoders_hex_categoricals.append(nn.LazyLinear(len(ind)))

        self.decoders_hex_binaries = nn.ModuleList([])
        for ind in self.rel_index[Group.HEX][Group.BINARIES]:
            self.decoders_hex_binaries.append(nn.LazyLinear(len(ind)))

        self.decoder_hex_cont_nullbit = nn.Identity()
        self.hex_nullbit_size = len(self.rel_index[Group.HEX][Group.CONT_NULLBIT])
        if self.hex_nullbit_size:
            self.decoder_hex_cont_nullbit = nn.LazyLinear(self.hex_nullbit_size)

        self.decoder_hex_cont_abs = nn.Identity()
        self.decoder_hex_cont_rel = nn.Identity()

        #
        # Player decoders
        #

        # learned positional embeddings for 2 players
        self.player_pos_embed = nn.Parameter(torch.randn(1, 2, self.z_size_player))
        # => (B, 2, Z_PLAYER)

        self.z_size_player_threshold = sum(lin.out_features for lin in encoder.encoders_player_thresholds)
        self.z_size_player_categorical = sum(emb.embedding_dim for emb in encoder.encoders_player_categoricals)
        self.z_size_player_binary = sum(lin.out_features for lin in encoder.encoders_player_binaries)
        self.z_size_player_cont_nullbit = getattr(encoder.encoder_player_cont_nullbit, "out_features", 0)
        self.z_size_player_cont_rel = len(self.rel_index[Group.PLAYER][Group.CONT_REL])
        self.z_size_player_cont_abs = len(self.rel_index[Group.PLAYER][Group.CONT_ABS])
        self.z_size_player_merged = (
            self.z_size_player_threshold
            + self.z_size_player_categorical
            + self.z_size_player_binary
            + self.z_size_player_cont_nullbit
            + self.z_size_player_cont_rel
            + self.z_size_player_cont_abs
        )

        self.decoder_merged_player = nn.Sequential(
            # => (B, 2, Z_PLAYER)
            nn.LazyLinear(self.z_size_player_merged),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3)
        )
        # => (B, 2, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)

        self.decoders_player_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.THRESHOLDS]:
            self.decoders_player_thresholds.append(nn.LazyLinear(len(ind)))

        self.decoders_player_categoricals = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.CATEGORICALS]:
            # Simply use a Linear layer as the inverse of Embedding
            self.decoders_player_categoricals.append(nn.LazyLinear(len(ind)))

        self.decoders_player_binaries = nn.ModuleList([])
        for ind in self.rel_index[Group.PLAYER][Group.BINARIES]:
            self.decoders_player_binaries.append(nn.LazyLinear(len(ind)))

        self.decoder_player_cont_nullbit = nn.Identity()
        self.player_nullbit_size = len(self.rel_index[Group.PLAYER][Group.CONT_NULLBIT])
        if self.player_nullbit_size:
            self.decoder_player_cont_nullbit = nn.LazyLinear(self.player_nullbit_size)

        self.decoder_player_cont_abs = nn.Identity()
        self.decoder_player_cont_rel = nn.Identity()

        #
        # Global decoders
        #

        self.z_sizes_global_thresholds = [lin.out_features for lin in encoder.encoders_global_thresholds]
        self.z_sizes_global_categoricals = [emb.embedding_dim for emb in encoder.encoders_global_categoricals]
        self.z_sizes_global_binaries = [lin.out_features for lin in encoder.encoders_global_binaries]
        self.z_size_global_cont_nullbit = getattr(encoder.encoder_global_cont_nullbit, "out_features", 0)
        self.z_size_global_cont_rel = len(self.rel_index[Group.GLOBAL][Group.CONT_REL])
        self.z_size_global_cont_abs = len(self.rel_index[Group.GLOBAL][Group.CONT_ABS])
        self.z_size_global_merged = (
            sum(self.z_sizes_global_thresholds)
            + sum(self.z_sizes_global_categoricals)
            + sum(self.z_sizes_global_binaries)
            + self.z_size_global_cont_nullbit
            + self.z_size_global_cont_rel
            + self.z_size_global_cont_abs
        )

        self.decoder_merged_global = nn.LazyLinear(self.z_size_global_merged)
        # => (B, Z_GLOBAL)

        self.decoders_global_thresholds = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.THRESHOLDS]:
            self.decoders_global_thresholds.append(nn.LazyLinear(len(ind)))

        self.decoders_global_categoricals = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.CATEGORICALS]:
            # Simply use a Linear layer as the inverse of Embedding
            self.decoders_global_categoricals.append(nn.LazyLinear(len(ind)))

        self.decoders_global_binaries = nn.ModuleList([])
        for ind in self.rel_index[Group.GLOBAL][Group.BINARIES]:
            self.decoders_global_binaries.append(nn.LazyLinear(len(ind)))

        self.decoder_global_cont_nullbit = nn.Identity()
        self.global_nullbit_size = len(self.rel_index[Group.GLOBAL][Group.CONT_NULLBIT])
        if self.global_nullbit_size:
            self.decoder_global_cont_nullbit = nn.LazyLinear(self.global_nullbit_size)

        self.decoder_global_cont_abs = nn.Identity()
        self.decoder_global_cont_rel = nn.Identity()

        self.to(encoder.device)

        # Init lazy layers
        with torch.no_grad():
            z_agg = torch.randn([2, encoder.z_size], device=self.device)
            self.forward(z_agg)

        layer_init(self)

    def forward(self, z):
        # torch.cat which returns empty tensor if tuple is empty
        def torch_cat(tuple_of_tensors, dim):
            if len(tuple_of_tensors) == 0:
                return torch.tensor([], device=self.device)
            return torch.cat(tuple_of_tensors, dim=dim)

        b = z.shape[0]
        obs_decoded = torch.zeros([b, DIM_OBS], device=self.device)

        z = self.z_decoder(z)
        mean_z_hex, mean_z_player, z_global = z.split([self.z_size_hex, self.z_size_player, self.z_size_global], dim=1)

        # Hex

        z_hex = mean_z_hex.unsqueeze(1).repeat(1, 165, 1) + self.hex_pos_embed  # => (B, 165, Z_HEX)
        # XXX: we don't have the _original_ z_hex from the encoder to use as "memory"
        # => two options:
        # 1. Zero memory: a tensor of zeros so that cross-attention has no effect
        # 2. z_hex as both tgt and memory: reuse the decoder’s own inputs for both attention passes,
        #   effectively giving the decoder two attention sub-layers without external context.
        # Trying with 2. (TODO: try with 1)
        z_hex = self.transformer_decoder_hex(tgt=z_hex, memory=z_hex)  # => (B, 165, Z_HEX)
        z_hex = self.decoder_merged_hex(z_hex)  # => (B, 165, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
        assert z_hex.shape == torch.Size([b, 165, self.z_size_hex_merged]), f"{z_hex.shape} == {[b, 165, self.z_size_hex_merged]}"
        (
            hex_threshold_z,
            hex_categorical_z,
            hex_binary_z,
            hex_cont_nullbit_z,
            hex_cont_rel_z,
            hex_cont_abs_z
        ) = z_hex.split([
            self.z_size_hex_threshold,
            self.z_size_hex_categorical,
            self.z_size_hex_binary,
            self.z_size_hex_cont_nullbit,
            self.z_size_hex_cont_rel,
            self.z_size_hex_cont_abs,
        ], dim=2)

        emb_calc = lambda n: math.ceil(math.sqrt(n))

        hex_threshold_zs = hex_threshold_z.split([lin.out_features for lin in self.decoders_hex_thresholds], dim=2)
        hex_threshold_outs = [lin(x) for lin, x in zip(self.decoders_hex_thresholds, hex_threshold_zs)]
        hex_categorical_zs = hex_categorical_z.split([emb_calc(lin.out_features) for lin in self.decoders_hex_categoricals], dim=2)
        hex_categorical_outs = [lin(x) for lin, x in zip(self.decoders_hex_categoricals, hex_categorical_zs)]
        hex_binary_zs = hex_binary_z.split([lin.out_features for lin in self.decoders_hex_binaries], dim=2)
        hex_binary_outs = [lin(x) for lin, x in zip(self.decoders_hex_binaries, hex_binary_zs)]
        hex_cont_nullbit_out = self.decoder_hex_cont_nullbit(hex_cont_nullbit_z)
        hex_cont_rel_out = self.decoder_hex_cont_rel(hex_cont_rel_z)
        hex_cont_abs_out = self.decoder_hex_cont_abs(hex_cont_abs_z)

        obs_decoded[:, torch_cat(self.abs_index[Group.HEX][Group.THRESHOLDS], dim=1).to(torch.int64)] = torch_cat(hex_threshold_outs, dim=2).to(obs_decoded.dtype)
        obs_decoded[:, torch_cat(self.abs_index[Group.HEX][Group.CATEGORICALS], dim=1).to(torch.int64)] = torch_cat(hex_categorical_outs, dim=2).to(obs_decoded.dtype)
        obs_decoded[:, torch_cat(self.abs_index[Group.HEX][Group.BINARIES], dim=1).to(torch.int64)] = torch_cat(hex_binary_outs, dim=2).to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]] = hex_cont_nullbit_out.to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.HEX][Group.CONT_REL]] = hex_cont_rel_out.to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.HEX][Group.CONT_ABS]] = hex_cont_abs_out.to(obs_decoded.dtype)

        # Player

        z_player = mean_z_player.unsqueeze(1).repeat(1, 2, 1) + self.player_pos_embed  # => (B, 2, Z_HEX)
        z_player = self.decoder_merged_player(z_player)  # => (B, 2, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
        assert z_player.shape == torch.Size([b, 2, self.z_size_player_merged]), f"{z_player.shape} == {b, 2, self.z_size_player_merged}"
        (
            player_threshold_z,
            player_categorical_z,
            player_binary_z,
            player_cont_nullbit_z,
            player_cont_rel_z,
            player_cont_abs_z
        ) = z_player.split([
            self.z_size_player_threshold,
            self.z_size_player_categorical,
            self.z_size_player_binary,
            self.z_size_player_cont_nullbit,
            self.z_size_player_cont_rel,
            self.z_size_player_cont_abs,
        ], dim=2)
        player_threshold_zs = player_threshold_z.split([lin.out_features for lin in self.decoders_player_thresholds], dim=2)
        player_threshold_outs = [lin(x) for lin, x in zip(self.decoders_player_thresholds, player_threshold_zs)]
        player_categorical_zs = player_categorical_z.split([lin.out_features for lin in self.decoders_player_categoricals], dim=2)
        player_categorical_outs = [lin(x) for lin, x in zip(self.decoders_player_categoricals, player_categorical_zs)]
        player_binary_zs = player_binary_z.split([lin.out_features for lin in self.decoders_player_binaries], dim=2)
        player_binary_outs = [lin(x) for lin, x in zip(self.decoders_player_binaries, player_binary_zs)]
        player_cont_nullbit_out = self.decoder_player_cont_nullbit(player_cont_nullbit_z)
        player_cont_rel_out = self.decoder_player_cont_rel(player_cont_rel_z)
        player_cont_abs_out = self.decoder_player_cont_abs(player_cont_abs_z)

        obs_decoded[:, torch_cat(self.abs_index[Group.PLAYER][Group.THRESHOLDS], dim=1).to(torch.int64)] = torch_cat(player_threshold_outs, dim=2).to(obs_decoded.dtype)
        obs_decoded[:, torch_cat(self.abs_index[Group.PLAYER][Group.CATEGORICALS], dim=1).to(torch.int64)] = torch_cat(player_categorical_outs, dim=2).to(obs_decoded.dtype)
        obs_decoded[:, torch_cat(self.abs_index[Group.PLAYER][Group.BINARIES], dim=1).to(torch.int64)] = torch_cat(player_binary_outs, dim=2).to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]] = player_cont_nullbit_out.to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.PLAYER][Group.CONT_REL]] = player_cont_rel_out.to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]] = player_cont_abs_out.to(obs_decoded.dtype)

        # Global
        z_global = self.decoder_merged_global(z_global)  # => (B, N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
        assert z_global.shape == torch.Size([b, self.z_size_global_merged]), f"{z_global.shape} == {[b, self.z_size_global_merged]}"

        (
            global_threshold_z,
            global_categorical_z,
            global_binary_z,
            global_cont_nullbit_z,
            global_cont_rel_z,
            global_cont_abs_z
        ) = z_global.split([
            sum(self.z_sizes_global_thresholds),
            sum(self.z_sizes_global_categoricals),
            sum(self.z_sizes_global_binaries),
            self.z_size_global_cont_nullbit,
            self.z_size_global_cont_rel,
            self.z_size_global_cont_abs,
        ], dim=1)

        global_threshold_zs = global_threshold_z.split(self.z_sizes_global_thresholds, dim=1)
        global_threshold_outs = [lin(x) for lin, x in zip(self.decoders_global_thresholds, global_threshold_zs)]
        global_categorical_zs = global_categorical_z.split(self.z_sizes_global_categoricals, dim=1)
        global_categorical_outs = [lin(x) for lin, x in zip(self.decoders_global_categoricals, global_categorical_zs)]
        global_binary_zs = global_binary_z.split(self.z_sizes_global_binaries, dim=1)
        global_binary_outs = [lin(x) for lin, x in zip(self.decoders_global_binaries, global_binary_zs)]
        global_cont_nullbit_out = self.decoder_global_cont_nullbit(global_cont_nullbit_z)
        global_cont_rel_out = self.decoder_global_cont_rel(global_cont_rel_z)
        global_cont_abs_out = self.decoder_global_cont_abs(global_cont_abs_z)

        obs_decoded[:, torch_cat(self.abs_index[Group.GLOBAL][Group.THRESHOLDS], dim=0).to(torch.int64)] = torch_cat(global_threshold_outs, dim=1).to(obs_decoded.dtype)
        obs_decoded[:, torch_cat(self.abs_index[Group.GLOBAL][Group.CATEGORICALS], dim=0).to(torch.int64)] = torch_cat(global_categorical_outs, dim=1).to(obs_decoded.dtype)
        obs_decoded[:, torch_cat(self.abs_index[Group.GLOBAL][Group.BINARIES], dim=0).to(torch.int64)] = torch_cat(global_binary_outs, dim=1).to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]] = global_cont_nullbit_out.to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]] = global_cont_rel_out.to(obs_decoded.dtype)
        obs_decoded[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]] = global_cont_abs_out.to(obs_decoded.dtype)

        return obs_decoded

    def reconstruct(self, obs_decoded, strategy=Reconstruction.GREEDY):
        global_cont_abs_out = obs_decoded[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]]
        global_cont_rel_out = obs_decoded[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]]
        global_cont_nullbit_out = obs_decoded[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]]
        global_binary_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.BINARIES]]
        global_categorical_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.CATEGORICALS]]
        global_threshold_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.THRESHOLDS]]
        player_cont_abs_out = obs_decoded[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]]
        player_cont_rel_out = obs_decoded[:, self.abs_index[Group.PLAYER][Group.CONT_REL]]
        player_cont_nullbit_out = obs_decoded[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]]
        player_binary_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.PLAYER][Group.BINARIES]]
        player_categorical_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.PLAYER][Group.CATEGORICALS]]
        player_threshold_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.PLAYER][Group.THRESHOLDS]]
        hex_cont_abs_out = obs_decoded[:, self.abs_index[Group.HEX][Group.CONT_ABS]]
        hex_cont_rel_out = obs_decoded[:, self.abs_index[Group.HEX][Group.CONT_REL]]
        hex_cont_nullbit_out = obs_decoded[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]]
        hex_binary_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.HEX][Group.BINARIES]]
        hex_categorical_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.HEX][Group.CATEGORICALS]]
        hex_threshold_outs = [obs_decoded[:, ind] for ind in self.abs_index[Group.HEX][Group.THRESHOLDS]]
        next_obs = torch.zeros_like(obs_decoded)

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
                return (logits > 0).float()

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


class VAE(nn.Module):
    def __init__(self, deterministic=False, device=torch.device("cpu")):
        super().__init__()
        self.deterministic = deterministic
        self.device = device
        self.obs_index = ObsIndex(device)
        self.abs_index = self.obs_index.abs_index
        self.rel_index = self.obs_index.rel_index

        self.encoder = Encoder(device)
        self.decoder = Decoder(self.encoder)

    def encode(self, obs):
        mu, logvar = self.encoder(obs)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        obs = self.decoder(z)
        return obs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu if self.deterministic else self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return mu, logvar, decoded


def _compute_reconstruction_losses(logits, target, index, weights, device=torch.device("cpu")):
    # Aggregate each feature's loss across players/hexes

    if logits[Group.CONT_ABS].dim() == 3:
        # (B, 165, N_FEATS) (or (B, 165) for categoricals after CE)
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

                # XXX: cross_entropy always removes the "C" dim (even with reduction=none)
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


def compute_decode_losses(logger, abs_index, loss_weights, real_obs, decoded_obs):
    # For shapes, see ObsIndex._build_abs_indices()
    extract = lambda t, obs: {
        Group.CONT_ABS: obs[:, abs_index[t][Group.CONT_ABS]],
        Group.CONT_REL: obs[:, abs_index[t][Group.CONT_REL]],
        Group.CONT_NULLBIT: obs[:, abs_index[t][Group.CONT_NULLBIT]],
        Group.BINARIES: [obs[:, ind] for ind in abs_index[t][Group.BINARIES]],
        Group.CATEGORICALS: [obs[:, ind] for ind in abs_index[t][Group.CATEGORICALS]],
        Group.THRESHOLDS: [obs[:, ind] for ind in abs_index[t][Group.THRESHOLDS]],
    }

    device = real_obs.device

    recon_losses = {}
    total_recon_loss = torch.tensor(0., device=decoded_obs.device)

    for cgroup in ContextGroup.as_list():
        logits = extract(cgroup, decoded_obs)
        target = extract(cgroup, real_obs)
        index = abs_index[cgroup]
        weights = loss_weights[cgroup]
        recon_losses[cgroup] = _compute_reconstruction_losses(logits, target, index, weights=weights, device=device)
        total_recon_loss += sum(subtype_losses.sum() for subtype_losses in recon_losses[cgroup].values())

    return total_recon_loss, recon_losses


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

    maybe_autocast = torch.autocast(model.device.type) if scaler else contextlib.nullcontext()

    model.train()
    timer = Timer()
    loss_rows = []

    if accumulate_grad:
        grad_steps = buffer.capacity // batch_size
        assert grad_steps > 0

    for epoch in range(epochs):
        logger.debug("Epoch: %s" % epoch)
        timer.start()
        for obs in buffer.sample_iter(batch_size):
            logger.debug(f"Obs shape: {obs.shape}")
            timer.stop()

            with maybe_autocast:
                mu, logvar, decoded_obs = model(obs)
                decode_loss_tot, decode_losses = compute_decode_losses(logger, model.abs_index, loss_weights, obs, decoded_obs)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss_tot = decode_loss_tot + kld

            loss_rows.extend(losses_to_rows(decode_losses, model.obs_index))
            loss_rows.append({
                TableColumn.ATTRIBUTE: "kld",
                TableColumn.CONTEXT: "all",
                TableColumn.DATATYPE: "",
                TableColumn.LOSS: kld.item()
            })

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

    return rows_to_df(loss_rows), timer.peek(), {}


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
    for obs in buffer.sample_iter(batch_size):
        timer.stop()

        with torch.no_grad():
            mu, logvar, decoded_obs = model(obs)

        _, decode_losses = compute_decode_losses(logger, model.abs_index, loss_weights, obs, decoded_obs)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss_rows.extend(losses_to_rows(decode_losses, model.obs_index))
        loss_rows.append({
            TableColumn.ATTRIBUTE: "kld",
            TableColumn.CONTEXT: "all",
            TableColumn.DATATYPE: "",
            TableColumn.LOSS: kld.item()
        })

        timer.start()

    return rows_to_df(loss_rows), timer.peek(), {}
