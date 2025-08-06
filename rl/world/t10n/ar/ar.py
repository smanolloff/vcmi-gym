import torch
from torch import nn
import enum
import contextlib
import torch.nn.functional as F
import pandas as pd

from ...util.buffer_base import BufferBase
from ...util.misc import TableColumn
from ...util.timer import Timer
from ...util.dataset_vcmi import Data, Context, DataInstruction
from ...util.obs_index import ObsIndex, Group, ContextGroup, DataGroup
from ...util.constants_v12 import (
    DIM_OBS,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    N_HEX_ACTIONS,
)


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


# Relative hex position bias for self-attention
# See RelPosBias2D in this convo: https://chatgpt.com/c/6892f91d-b898-832f-bba7-0beb540ee27c
# (follow answer-3-branch-2 & answer-4-branch-2)
def init_relpos_table_and_idx(rows=11, cols=15, num_heads=8):
    rows = 11
    cols = 15
    num_dr, num_dc = 2*rows - 1, 2*cols - 1  # 21, 29
    table = nn.Parameter(torch.zeros(num_heads, num_dr * num_dc))
    # nn.init.trunc_normal_(table, std=0.02)

    r = torch.arange(rows)
    c = torch.arange(cols)
    rr, cc = torch.meshgrid(r, c, indexing="ij")         # (11, 15)
    coords = torch.stack([rr, cc], dim=-1).view(-1, 2)   # (165, 2)
    rel = coords[:, None, :] - coords[None, :, :]        # (165, 165, 2)
    dr = rel[..., 0] + (rows - 1)                        # [0..20]
    dc = rel[..., 1] + (cols - 1)                        # [0..28]
    idx = dr * (2*cols - 1) + dc                         # (165, 165)
    # self.register_buffer("idx", idx, persistent=False)

    # return table, idx


    self.G = num_globals
    self.Lt = rows * cols
    self.L = self.G + self.Lt

    self.tile_rpb = RelPosBias2D(rows, cols, num_heads)

    self.learn_global = learn_global
    if learn_global:
        # global->tile, tile->global, global->global
        self.g2t = nn.Parameter(torch.zeros(num_heads, num_globals, self.Lt))
        self.t2g = nn.Parameter(torch.zeros(num_heads, self.Lt, num_globals))
        self.g2g = nn.Parameter(torch.zeros(num_heads, num_globals, num_globals))
        for p in (self.g2t, self.t2g, self.g2g):
            nn.init.trunc_normal_(p, std=0.02)


# TODO: see class RelPosBias2DWithGlobals(nn.Module) here:
#   https://chatgpt.com/s/t_689310a5d178819181048567c4a0cc8f



# def forward(self):
#     return self.table[:, self.idx]       # (num_heads, 165, 165)


class ObsPosEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.pos_global = nn.Parameter(torch.zeros(1, d_model))
        self.pos_player = nn.Parameter(torch.zeros(2, d_model))

        self.pos_hex_x = nn.Embedding(15, d_model)
        self.pos_hex_y = nn.Embedding(11, d_model)

        # Create with B=1 which will be auto-expanded when adding
        # 0,1,2,...,14,0,1,2,...,14
        self.register_buffer("positions_x", torch.arange(15).repeat(11), persistent=False)
        # 0,0,0,...,1,1,1,...,...,11
        self.register_buffer("positions_y", torch.arange(11).repeat_interleave(15), persistent=False)

        nn.init.xavier_uniform_(self.pos_global)
        nn.init.xavier_uniform_(self.pos_player)
        nn.init.xavier_uniform_(self.pos_hex_x.weight)
        nn.init.xavier_uniform_(self.pos_hex_y.weight)

    def forward(self, global_proj, player_proj, hex_proj):
        # The "+" expands RHS to (B, *) automatically
        global_proj += self.pos_global
        player_proj += self.pos_player
        hex_proj += self.pos_hex_x(self.positions_x)
        hex_proj += self.pos_hex_y(self.positions_y)

        return global_proj, player_proj, hex_proj


class ObsProjector(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.splits = [STATE_SIZE_GLOBAL, 2*STATE_SIZE_ONE_PLAYER, 165*STATE_SIZE_ONE_HEX]

        # Input projections
        # TODO: per-datatype projections (binary, categorical, etc.)
        self.proj_global = nn.Linear(STATE_SIZE_GLOBAL, d_model)
        self.proj_player = nn.Linear(STATE_SIZE_ONE_PLAYER, d_model)
        self.proj_hex = nn.Linear(STATE_SIZE_ONE_HEX, d_model)

        # For layers followed by ReLU, use Kaiming (He).
        # TODO: when followed by LeakyRelu, set nonlinearity='leaky_relu', a=0.01
        nn.init.kaiming_uniform_(self.proj_global.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.proj_player.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.proj_hex.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_global.bias)
        nn.init.zeros_(self.proj_player.bias)
        nn.init.zeros_(self.proj_hex.bias)

    def forward(self, obs):
        B = obs.size(0)
        global_in, player_in, hex_in = obs.split(self.splits, dim=1)
        global_in = global_in.view(B, 1, STATE_SIZE_GLOBAL)
        player_in = player_in.view(B, 2, STATE_SIZE_ONE_PLAYER)
        hex_in = hex_in.view(B, 165, STATE_SIZE_ONE_HEX)

        # Project into latent space
        # TODO: per-datatype obs projections (binary, categorical, etc.)
        global_proj = self.proj_global(global_in)   # => (B, 1, d_model)
        player_proj = self.proj_player(player_in)   # => (B, 2, d_model)
        hex_proj = self.proj_hex(hex_in)            # => (B, 165, d_model)
        # => (B, {1,2,165}, d_model)

        return global_proj, player_proj, hex_proj


class Encoder(nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        self.d_model = d_model
        self.device = device

        self.action_embedder = nn.Embedding(N_ACTIONS, self.d_model)
        self.action_pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))

        self.obs_projector = ObsProjector(d_model)
        self.obs_pos_encoder = ObsPosEncoder(d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, nhead=4, dropout=0.3, batch_first=True),
            num_layers=3,
        )

        self.to(device)
        nn.init.xavier_uniform_(self.action_embedder)
        nn.init.xavier_uniform_(self.action_pos_encoder)

    def forward(self, obs, action):
        # obs shape: (B, DIM_OBS)
        # action shape: (B, 1)

        a_emb = self.action_embedder(action)
        a_emb = self.action_pos_encoder(a_emb)

        g_proj, p_proj, h_proj = self.obs_projector(obs)
        g_proj, p_proj, h_proj = self.obs_pos_encoder(g_proj, p_proj, h_proj)

        transformer_in = torch.cat([a_emb, g_proj, p_proj, h_proj], dim=1)
        transformer_out = self.transformer(transformer_in)

        return a_emb, transformer_out


class Decoder(nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        self.d_model = d_model
        self.device = device

        self.obs_projector = ObsProjector(d_model)
        self.obs_pos_encoder = ObsPosEncoder(d_model)
        self.sos_token = nn.Parameter(torch.zeros(1, 1, STATE_SIZE_GLOBAL))

        self.transformer = nn.TransformerDecoder(
            num_layers=3,
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=4,
                dropout=0.3,
                batch_first=True
            )
        )

        # TODO: replace these with complex t10n logic for per-type proj
        self.head_global = nn.LazyLinear(STATE_SIZE_GLOBAL)
        self.head_player = nn.LazyLinear(STATE_SIZE_ONE_PLAYER)
        self.head_hex = nn.LazyLinear(STATE_SIZE_ONE_HEX)

        self.to(device)
        nn.init.xavier_uniform_(self.sos_global)
        nn.init.xavier_uniform_(self.sos_player)
        nn.init.xavier_uniform_(self.sos_hex)
        nn.init.xavier_uniform_(self.head_global)
        nn.init.xavier_uniform_(self.head_player)
        nn.init.xavier_uniform_(self.head_hex)

    def forward_train(self, next_obs, action_emb, memory):
        B = next_obs.size(0)

        # 3) Prepare decoder input via teacher forcing
        # 3a) embed the true next tokens
        g_proj, p_proj, h_proj = self.obs_projector(next_obs)
        obs_proj = torch.cat([g_proj, p_proj, h_proj], dim=1)
        # => (B, 168, d_model)


        # XXXX:
        # o3 suggested non-AR arch which is VERY similar to vanilla t10n
        # I asked for AR, and it gave me simpler arch than o4. It also advised
        # to share input projection layers
        # https://chatgpt.com/c/68914868-d660-832c-bb02-772485a9b5a7









        # 3b) shift right and prepend SOS
        # XXX: Without right-shift it would learn identity: in(i) = out(i)
        #       With shift, it _can_ learn shifted identity: in(i) = out(i+1),
        #       but that would force it to learn to attend to the "memory".
        #       Not sure I am convinced. Convo here:
        #       https://chatgpt.com/c/68910f3c-b56c-832b-885f-cc2bc2e40268
        # Actually, o4 got lost and coult not provide meaningful explanations.
        # I believe these SOS
        sos_seq = torch.cat([
            self.sos_global.expand(B, 1, STATE_SIZE_GLOBAL),
            self.sos_player.expand(B, 2, STATE_SIZE_ONE_PLAYER),
            self.sos_hex.expand(B, 165, STATE_SIZE_ONE_HEX)
        ], dim=1)
        # (B, 168, d_model)

        transformer_in = torch.cat([sos_seq[:, :1, :], obs_proj[:, :-1, :]], dim=1)
        transformer_in = self.pos_enc(dec_in)                                   # (B, 1+N, d_model)

        sos_seq = torch.cat([sos_global, sos_player, sos_hex], dim=1)
        # (B, 168, d_model)


        g_proj, p_proj, h_proj = self.obs_pos_encoder(g_proj, p_proj, h_proj)

        split_sizes = [STATE_SIZE_GLOBAL, 2*STATE_SIZE_ONE_PLAYER, 165*STATE_SIZE_ONE_HEX]
        global_in, player_in, hex_in = next_obs.split(split_sizes, dim=1)
        global_in = global_in.view(B, 1, STATE_SIZE_GLOBAL)
        player_in = player_in.view(B, 2, STATE_SIZE_ONE_PLAYER)
        hex_in = hex_in.view(B, 165, STATE_SIZE_ONE_HEX)

        # Shift sequences right & prepend sos token
        global_in = torch.cat([sos_global, global_in[:, :-1]], dim=1)
        player_in = torch.cat([sos_player, player_in[:, :-1, :]], dim=1)
        hex_in = torch.cat([sos_hex, hex_in[:, :-1, :]], dim=1)

        global_emb = self.embedder_global(global_in)
        player_emb = self.embedder_player(player_in)
        hex_emb = self.embedder_hex(hex_in)
        # => (B, {1,2,165}, d_model)

        emb_seq = self.embedder_pos + torch.cat([action_emb, global_emb, player_emb, hex_emb], dim=1)
        # => (B, 169, d_model)

        causal_mask = torch.triu(torch.ones(169, 169), diagonal=1).bool().to(self.device),
        transformer_out = self.transformer(tgt=emb_seq, tgt_mask=causal_mask, memory=memory)

        (
            _transformer_out_action,
            transformer_out_global,
            transformer_out_player,
            transformer_out_hex,
        ) = transformer_out.split([1, 1, 2, 165], dim=1)

        # Project back to feature space
        global_out = self.head_global(transformer_out_global).flatten(start_dim=1)  # (B, STATE_SIZE_GLOBAL)
        player_out = self.head_player(transformer_out_player).flatten(start_dim=1)  # (B, 2*STATE_SIZE_ONE_PLAYER)
        hex_out = self.head_hex(transformer_out_hex).flatten(start_dim=1)           # (B, 165*STATE_SIZE_ONE_HEX)

        obs_out = torch.cat([global_out, player_out, hex_out], dim=1)

        return obs_out

    def forward_eval(self, memory, action_emb):
        B = memory.size(0)

        # Replace the fullground-truth sequence with only sos tokens
        # Entries beyond the first (i.e. positions 1…N) are dummy embeddings
        # that will be overwritten as each next token is predicted.
        global_in = self.sos_global.expand(B, 1, STATE_SIZE_GLOBAL)
        player_in = self.sos_player.expand(B, 2, STATE_SIZE_ONE_PLAYER)
        hex_in = self.sos_hex.expand(B, 165, STATE_SIZE_ONE_HEX)

        global_emb = self.embedder_global(global_in)
        player_emb = self.embedder_player(player_in)
        hex_emb = self.embedder_hex(hex_in)
        # => (B, {1,2,165}, d_model)

        # Sequense of SOS tokens (to be replaced by pred tokens)
        emb_seq = self.embedder_pos + torch.cat([action_emb, global_emb, player_emb, hex_emb], dim=1)
        # => (B, 169, d_model)

        for i in range(0, emb_seq.shape[1]):
            # compute decoder output up to step i
            transformer_out = self.transformer(
                tgt=emb_seq[:, :i+1, :],
                tgt_mask=torch.triu(torch.ones(i+1, i+1), diagonal=1).bool(),
                memory=memory
            )

            next_token = transformer_out[:, i, :]
            # => (B, d_model)

            if i == 0:
                head, embedder = self.head_global, self.embedder_global
            elif i == 1:
                head, embedder = self.head_player, self.embedder_player
            else:
                head, embedder = self.head_hex, self.embedder_hex



            # ABANDONING
            # in favor of trying learning positional encodings
            # + providing memory to transformer












            #     # global prediction
            #     pred_g = self.global_out(last)   # (B, Fg)
            #     outputs.append(pred_g)
            #     # prepare next input embedding
            #     next_emb = self.global_proj_dec(pred_g.unsqueeze(1))
            # else:
            #     # tile predictions
            #     pred_t = self.tile_out(last)      # (B, Ft)
            #     outputs.append(pred_t)
            #     next_emb = self.tile_proj_dec(pred_t.unsqueeze(1))

            # emb_seq[:, i+1, :] = token

        emb_seq = self.embedder_pos + torch.cat([action_emb, global_emb, player_emb, hex_emb], dim=1)
        # => (B, 169, d_model)

        causal_mask = torch.triu(torch.ones(169, 169), diagonal=1).bool().to(self.device),
        transformer_out = self.transformer(tgt=emb_seq, tgt_mask=causal_mask, memory=memory)

        (
            _transformer_out_action,
            transformer_out_global,
            transformer_out_player,
            transformer_out_hex,
        ) = transformer_out.split([1, 1, 2, 165], dim=1)

        # Project back to feature space
        global_out = self.head_global(transformer_out_global).flatten(start_dim=1)  # (B, STATE_SIZE_GLOBAL)
        player_out = self.head_player(transformer_out_player).flatten(start_dim=1)  # (B, 2*STATE_SIZE_ONE_PLAYER)
        hex_out = self.head_hex(transformer_out_hex).flatten(start_dim=1)           # (B, 165*STATE_SIZE_ONE_HEX)

        obs_out = torch.cat([global_out, player_out, hex_out], dim=1)

        return obs_out


class ARTransitionModel(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()

        self.device = device
        self.obs_index = ObsIndex(device)
        self.abs_index = self.obs_index.abs_index
        self.rel_index = self.obs_index.rel_index

        self.encoder = Encoder(config["d_model"], device)
        self.decoder = Decoder(config["d_model"], device)

    def forward_train(self, obs, action, next_obs):
        memory, action_emb = self.encoder.forward(obs, action)
        pred_obs = self.decoder.forward_train(next_obs, memory, action_emb)
        return pred_obs

    def forward_eval(self, obs, action):
        was_training = self.training
        self.train(False)
        with torch.no_grad():
            memory, action_emb = self.encoder.forward(obs, action)
            pred_obs = self.decoder.forward_eval(memory, action_emb)
        self.train(was_training)
        return pred_obs


def _compute_obs_losses(logits, target, index, weights, device=torch.device("cpu")):
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

    device = next_obs.device

    obs_losses = {}
    total_loss = torch.tensor(0., device=pred_obs.device)

    for cgroup in ContextGroup.as_list():
        logits = extract(cgroup, pred_obs)
        target = extract(cgroup, next_obs)
        index = abs_index[cgroup]
        weights = loss_weights[cgroup]
        obs_losses[cgroup] = _compute_obs_losses(logits, target, index, weights=weights, device=device)
        total_loss += sum(subtype_losses.sum() for subtype_losses in obs_losses[cgroup].values())

    return total_loss, obs_losses


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
    obs_loss_rows = []

    if accumulate_grad:
        grad_steps = buffer.capacity // batch_size
        assert grad_steps > 0

    for epoch in range(epochs):
        timer.start()
        for batch in buffer.sample_iter(batch_size):
            timer.stop()
            obs, action, next_obs, next_mask, next_rew, next_done = batch

            with maybe_autocast:
                pred_obs = model.forward_train(obs, action, next_obs)
                loss_tot, obs_losses = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)

            obs_loss_rows.extend(losses_to_rows(obs_losses, model.obs_index))

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

    return rows_to_df(obs_loss_rows), timer.peek(), {}


def eval_model(
    logger,
    model,
    loss_weights,
    buffer,
    batch_size,
):
    model.eval()
    timer = Timer()
    obs_loss_rows = []

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, action, next_obs, next_mask, next_rew, next_done = batch

        with torch.no_grad():
            # TODO: replace this with regular forward call (true AR, no teacher-forcing)
            pred_obs = model.forward_train(obs, action, next_obs)

        loss_tot, obs_losses = compute_losses(logger, model.abs_index, loss_weights, next_obs, pred_obs)
        obs_loss_rows.extend(losses_to_rows(obs_losses, model.obs_index))
        timer.start()

    return rows_to_df(obs_loss_rows), timer.peek(), {}
