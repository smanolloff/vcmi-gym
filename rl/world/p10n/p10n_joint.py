import torch
import torch.nn as nn
import math
import enum
import pandas as pd
import contextlib

from torch.nn.functional import cross_entropy

from ..util.buffer_base import BufferBase
from ..util.dataset_vcmi import Data, Context, DataInstruction
from ..util.misc import layer_init, safe_mean
from ..util.obs_index import ObsIndex, Group
from ..util.timer import Timer
from ..util.misc import TableColumn

from ..util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    HEX_ATTR_MAP,
    HEX_ACT_MAP,
    N_ACTIONS,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


class Prediction(enum.IntEnum):
    PROBS = 0               # softmax(logits)
    SAMPLE = enum.auto()    # sample(softmax(logits))
    GREEDY = enum.auto()    # argmax(logits)


def vcmi_dataloader_functor():
    def mw(data: Data, ctx: Context):
        instruction = DataInstruction.USE

        if data.done or ctx.transition_id in [0, ctx.num_transitions - 1]:
            instruction = DataInstruction.SKIP

        return data, instruction

    return mw


def s3_dataloader_functor():
    # XXX: must only pass through non-terminal obs which are from the ENEMY
    #      Below is the code from v10, should be *much* simpler in v12
    #      (as there is now a BATTLE_SIDE_ACTIVE_PLAYER global attribute)
    #
    # hexes = obs[STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER:].reshape(165, -1)
    # index_is_active = HEX_ATTR_MAP["STACK_QUEUE"][1]  # 1 if first in queue (zero null => index=offset)
    # # index_is_blue = np.arange(HEX_ATTR_MAP["STACK_SIDE"][1] + HEX_ATTR_MAP["STACK_SIDE"][2]] + 2  # 1 if blue ([null, red, blue] => index=offset+2)
    # extracted = hexes[:, [index_is_active, index_is_blue]]
    # # => (165, 2), where 2 = [is_active, is_blue]
    # mask = extracted[..., 0] != 0
    # # => 165 bools (True for hexes for which is_active=True)
    # idx = np.argmax(mask)  # index of 1st hex with active stack on it
    # is_blue = extracted[idx, 1]  # is_blue for that hex
    # if is_blue:
    #     index_battle_in_progress = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1]  # 1 if winner is N/A (explicit null => index=offset)
    #     if action > 0:
    #         assert obs[index_battle_in_progress] == 1, obs[index_battle_in_progress]
    #         with self.timer_idle:
    #             yield obs, action
    raise NotImplementedError()

    # def mw(data: Data, ctx: Context):
    #     if ctx.transition_id == ctx.num_transitions - 1:
    #         state["reward_carry"] = data.reward
    #         if not data.done:
    #             return None
    #     if ctx.transition_id == 0 and ctx.ep_steps > 0:
    #         return data._replace(reward=state["reward_carry"])
    #     return data

    # return mw


class Buffer(BufferBase):
    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # XXX: ALL indices are valid:
        # (terminal obs has action = -1 which is used for `done` prediction)
        sampled_indices = torch.randint(max_index, (batch_size,), device=self.device)
        obs = self.containers["obs"][sampled_indices]
        mask = self.containers["mask"][sampled_indices]
        reward = self.containers["reward"][sampled_indices]
        done = self.containers["done"][sampled_indices]
        action = self.containers["action"][sampled_indices]

        return obs, mask, reward, done, action

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index

        shuffled_indices = torch.randperm(max_index, device=self.device)

        # will fail if self.full is false
        assert len(shuffled_indices) == self.capacity

        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]

            # NOTE: reward is the result of prev action (see vcmidataset.py)
            #  XXX: reward prediction gets tricky for step0 (no reward!)
            yield (
                self.containers["obs"][batch_indices],
                self.containers["mask"][batch_indices],
                self.containers["reward"][batch_indices],
                self.containers["done"][batch_indices],
                self.containers["action"][batch_indices],
            )


class ActionPredictionModel(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        self.obs_index = ObsIndex(device)

        self.abs_index = self.obs_index.abs_index
        self.rel_index = self.obs_index.rel_index

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
            # => (B, 165, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
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

        #
        # Heads
        #

        # => (B, Z_AGG)
        self.head = nn.LazyLinear(N_ACTIONS)
        # => (B, N_ACTIONS)

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
        z_agg = self.aggregator(torch.cat([z_global, mean_z_player, mean_z_hex], dim=-1))
        # => (B, Z_AGG)

        #
        # Outputs
        #

        out = self.head(z_agg)

        return out

    def _predict_probs(self, obs, logits):
        B = obs.shape[0]

        obs_global, obs_players, obs_hexes = obs.split([STATE_SIZE_GLOBAL, STATE_SIZE_ONE_PLAYER*2, STATE_SIZE_ONE_HEX*165], dim=1)

        mask = torch.zeros([B, N_ACTIONS], device=self.device)
        # => (B, 2) floats where 1=valid, 0=invalid
        # Typically, there are no intermediate values (binary mask)
        # ...but when obs is itself "predicted", then the mask is continuous

        gmask_len = GLOBAL_ATTR_MAP["ACTION_MASK"][2]    # n
        gmask_start = GLOBAL_ATTR_MAP["ACTION_MASK"][1]  # offset
        gmask_end = gmask_start + gmask_len
        mask[:, :gmask_len] = obs_global[:, gmask_start:gmask_end]

        hexmask_len = HEX_ATTR_MAP["ACTION_MASK"][2]    # n
        hexmask_start = HEX_ATTR_MAP["ACTION_MASK"][1]  # offset
        hexmask_end = hexmask_start + hexmask_len
        mask[:, gmask_len:] = obs_hexes.unflatten(-1, [165, -1])[:, :, hexmask_start:hexmask_end].flatten(start_dim=1)

        probs = (logits * mask).softmax(dim=-1)
        # => (B, N_ACTIONS)

        # assert probs.sum(dim=1) == 1, probs.sum()
        return probs

    def _predict_action(self, obs, logits, strategy):
        if strategy == Prediction.GREEDY:
            return logits.argmax(dim=1)
        else:
            num_classes = logits.shape[-1]
            probs_2d = logits.softmax(dim=-1).view(-1, num_classes)
            return torch.multinomial(probs_2d, num_samples=1).view(logits.shape[:-1])

    def predict_(self, obs, logits=None, strategy=Prediction.GREEDY):
        if logits is None:
            logits = self.forward(obs)

        if strategy == Prediction.PROBS:
            return self._predict_probs(obs, logits)
        else:
            return self._predict_action(obs, logits, strategy)

    def predict(self, obs, strategy=Prediction.GREEDY):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.predict_(obs, strategy=strategy)[0].item()


def compute_loss(obs, mask, action, logits):
    assert all(action > 0), "Found invalid actions: %s" % action
    # mask_value = torch.tensor(torch.finfo(logits.dtype).min)
    # masked_logits = logits.masked_fill(~(mask.bool()), mask_value)
    # return cross_entropy(masked_logits, action)
    return cross_entropy(logits, action)


def losses_to_rows(loss):
    return [
        {
            TableColumn.ATTRIBUTE: "joint",
            TableColumn.CONTEXT: "all",
            TableColumn.DATATYPE: "",
            TableColumn.LOSS: loss.item()
        },
    ]


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
    obs_loss_rows = []

    if accumulate_grad:
        grad_steps = buffer.capacity // batch_size
        assert grad_steps > 0

    for epoch in range(epochs):
        timer.start()
        for batch in buffer.sample_iter(batch_size):
            timer.stop()
            obs, mask, rew, done, action = batch

            with maybe_autocast:
                pred_logits = model(obs)
                loss_tot = compute_loss(obs, mask, action, pred_logits)

            obs_loss_rows.extend(losses_to_rows(loss_tot))

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
    match_ratios = []
    obs_loss_rows = []
    timer = Timer()

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, mask, rew, done, action = batch

        with torch.no_grad():
            pred_logits = model(obs)
            pred_action = model.predict_(obs, pred_logits)
            match_ratios.append((pred_action == action).float().mean().item())

        loss_tot = compute_loss(obs, mask, action, pred_logits)
        obs_loss_rows.extend(losses_to_rows(loss_tot))
        timer.start()
    timer.stop()

    return rows_to_df(obs_loss_rows), timer.peek(), {"eval/greedy_match_ratio": safe_mean(match_ratios)}
