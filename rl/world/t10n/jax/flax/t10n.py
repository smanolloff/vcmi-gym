import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import flax.linen as fnn
import math
import enum
from flax.core import freeze, unfreeze

from .obs_index import ObsIndex, Group

from ....util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    DIM_OBS,
)


class Other(enum.IntEnum):
    CAN_WAIT = 0
    DONE = enum.auto()


class Reconstruction(enum.IntEnum):
    PROBS = 0               # clamp(cont) + softmax(bin) + softmax(cat)
    SAMPLES = enum.auto()   # clamp(cont) + sample(sigmoid(bin)) + sample(softmax(cat))
    GREEDY = enum.auto()    # clamp(cont) + round(sigmoid(bin)) + argmax(cat)


class LeakyReLU(fnn.Module):
    negative_slope: float = 0.01

    @fnn.compact
    def __call__(self, x):
        return jnn.leaky_relu(x, self.negative_slope)


class Identity(fnn.Module):
    features: int = 0

    @fnn.compact
    def __call__(self, x):
        return x


class TransformerEncoderLayer(fnn.Module):
    d_model: int
    dim_feedforward: int
    num_heads: int
    dropout_rate: float
    deterministic: bool

    def setup(self):
        self.self_attn = fnn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            use_bias=True,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            broadcast_dropout=False
        )

        self.linear1 = fnn.Dense(self.dim_feedforward)  # 2048=torch default
        self.dropout = fnn.Dropout(self.dropout_rate)
        self.linear2 = fnn.Dense(self.d_model)
        self.norm1 = fnn.LayerNorm(epsilon=1e-5)
        self.norm2 = fnn.LayerNorm(epsilon=1e-5)
        self.dropout1 = fnn.Dropout(self.dropout_rate)
        self.dropout2 = fnn.Dropout(self.dropout_rate)

    def __call__(self, x):
        # Multi-head self-attention block
        residual = x
        x = self.self_attn(x)
        x = self.dropout1(x, deterministic=self.deterministic)
        x = self.norm1(residual + x)

        # Position-wise feed-forward block
        residual = x
        x = self.linear1(x)
        x = fnn.relu(x)
        x = self.dropout(x, deterministic=self.deterministic)
        x = self.linear2(x)
        x = self.dropout2(x, deterministic=self.deterministic)
        x = self.norm2(residual + x)

        return x


class TransformerEncoder(fnn.Module):
    num_layers: int
    d_model: int
    dim_feedforward: int
    num_heads: int
    dropout_rate: float
    deterministic: bool

    def setup(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(TransformerEncoderLayer(
                d_model=self.d_model,
                dim_feedforward=self.dim_feedforward,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=True,
            ))
        self.layers = layers

    def __call__(self, x):
        for mod in self.layers:
            x = mod(x)
        return x


class FlaxTransitionModel(fnn.Module):
    """ Flax translation of the PyTorch TransitionModel. """
    deterministic: bool = False

    def setup(self):
        self.obs_index = ObsIndex()

        self.abs_index = self.obs_index.abs_index
        self.rel_index = self.obs_index.rel_index

        emb_calc = lambda n: math.ceil(math.sqrt(n))

        self.encoder_action = fnn.Embed(N_ACTIONS, emb_calc(N_ACTIONS))

        #
        # Global encoders
        #

        # Continuous:
        # (B, n)
        self.encoder_global_cont_abs = Identity()
        self.encoder_global_cont_rel = Identity()

        # Continuous (nulls):
        # (B, n)
        encoder_global_cont_nullbit = Identity()
        global_nullbit_size = len(self.rel_index[Group.GLOBAL][Group.CONT_NULLBIT])
        if global_nullbit_size:
            encoder_global_cont_nullbit = fnn.Dense(global_nullbit_size)
        self.encoder_global_cont_nullbit = encoder_global_cont_nullbit

        # Binaries:
        # [(B, b1), (B, b2), ...]
        encoders_global_binaries = []
        for ind in self.rel_index[Group.GLOBAL][Group.BINARIES]:
            encoders_global_binaries.append(fnn.Dense(len(ind)))
        self.encoders_global_binaries = encoders_global_binaries

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        encoders_global_categoricals = []
        for ind in self.rel_index[Group.GLOBAL][Group.CATEGORICALS]:
            cat_emb = fnn.Embed(len(ind), emb_calc(len(ind)))
            encoders_global_categoricals.append(cat_emb)
        self.encoders_global_categoricals = encoders_global_categoricals

        # Thresholds:
        # [(B, T1), (B, T2), ...]
        encoders_global_thresholds = []
        for ind in self.rel_index[Group.GLOBAL][Group.THRESHOLDS]:
            encoders_global_thresholds.append(fnn.Dense(len(ind)))
        self.encoders_global_thresholds = encoders_global_thresholds

        # Merge
        z_size_global = 256
        self.encoder_merged_global = fnn.Sequential([
            # => (B, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            fnn.Dense(z_size_global),
            LeakyReLU(),
        ])
        # => (B, Z_GLOBAL)

        #
        # Player encoders
        #

        # Continuous per player:
        # (B, n)
        self.encoder_player_cont_abs = Identity()
        self.encoder_player_cont_rel = Identity()

        # Continuous (nulls) per player:
        # (B, n)
        self.encoder_player_cont_nullbit = Identity()
        player_nullbit_size = len(self.rel_index[Group.PLAYER][Group.CONT_NULLBIT])
        if player_nullbit_size:
            self.encoder_player_cont_nullbit = fnn.Dense(player_nullbit_size)

        # Binaries per player:
        # [(B, b1), (B, b2), ...]
        encoders_player_binaries = []
        for ind in self.rel_index[Group.PLAYER][Group.BINARIES]:
            encoders_player_binaries.append(fnn.Dense(len(ind)))
        self.encoders_player_binaries = encoders_player_binaries

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        encoders_player_categoricals = []
        for ind in self.rel_index[Group.PLAYER][Group.CATEGORICALS]:
            cat_emb = fnn.Embed(len(ind), emb_calc(len(ind)))
            encoders_player_categoricals.append(cat_emb)
        self.encoders_player_categoricals = encoders_player_categoricals

        # Thresholds per player:
        # [(B, T1), (B, T2), ...]
        encoders_player_thresholds = []
        for ind in self.rel_index[Group.PLAYER][Group.THRESHOLDS]:
            encoders_player_thresholds.append(fnn.Dense(len(ind)))
        self.encoders_player_thresholds = encoders_player_thresholds

        # Merge per player
        z_size_player = 256
        self.encoder_merged_player = fnn.Sequential([
            # => (B, 2, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            fnn.Dense(z_size_player),
            LeakyReLU(),
        ])
        # => (B, 2, Z_PLAYER)

        #
        # Hex encoders
        #

        # Continuous per hex:
        # (B, n)
        self.encoder_hex_cont_abs = Identity()
        self.encoder_hex_cont_rel = Identity()

        # Continuous (nulls) per hex:
        # (B, n)
        encoder_hex_cont_nullbit = Identity()
        hex_nullbit_size = len(self.rel_index[Group.HEX][Group.CONT_NULLBIT])
        if hex_nullbit_size:
            encoder_hex_cont_nullbit = fnn.Dense(hex_nullbit_size)
        self.encoder_hex_cont_nullbit = encoder_hex_cont_nullbit

        # Binaries per hex:
        # [(B, b1), (B, b2), ...]
        encoders_hex_binaries = []
        for ind in self.rel_index[Group.HEX][Group.BINARIES]:
            encoders_hex_binaries.append(fnn.Dense(len(ind)))
        self.encoders_hex_binaries = encoders_hex_binaries

        # Categoricals per hex:
        # [(B, C1), (B, C2), ...]
        encoders_hex_categoricals = []
        for ind in self.rel_index[Group.HEX][Group.CATEGORICALS]:
            cat_emb = fnn.Embed(len(ind), emb_calc(len(ind)))
            encoders_hex_categoricals.append(cat_emb)
        self.encoders_hex_categoricals = encoders_hex_categoricals

        # Thresholds per hex:
        # [(B, T1), (B, T2), ...]
        encoders_hex_thresholds = []
        for ind in self.rel_index[Group.HEX][Group.THRESHOLDS]:
            encoders_hex_thresholds.append(fnn.Dense(len(ind)))
        self.encoders_hex_thresholds = encoders_hex_thresholds

        # Merge per hex
        z_size_hex = 512
        self.encoder_merged_hex = fnn.Sequential([
            # => (B, 165, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS + T*N_THR_FEATS)
            fnn.Dense(z_size_hex),
            LeakyReLU(),
            fnn.Dropout(0.3, deterministic=self.deterministic)
        ])
        # => (B, 165, Z_HEX)

        # Transformer (hexes only)
        self.transformer_hex = TransformerEncoder(
            num_layers=6,
            d_model=z_size_hex,
            dim_feedforward=2048,
            num_heads=8,
            dropout_rate=0.3,
            deterministic=self.deterministic
        )
        # => (B, 165, Z_HEX)

        #
        # Aggregator
        #

        z_size_agg = 2048
        self.aggregator = fnn.Sequential([
            # => (B, Z_GLOBAL + AVG(2*Z_PLAYER) + AVG(165*Z_HEX))
            fnn.Dense(z_size_agg),
            LeakyReLU(),
        ])
        # => (B, Z_AGG)

        #
        # Heads
        #

        # => (B, Z_AGG)
        self.head_global = fnn.Dense(STATE_SIZE_GLOBAL)

        # => (B, 2, Z_AGG + Z_PLAYER)
        self.head_player = fnn.Dense(STATE_SIZE_ONE_PLAYER)

        # => (B, 165, Z_AGG + Z_HEX)
        self.head_hex = fnn.Dense(STATE_SIZE_ONE_HEX)

    def __call__(self, obs, action):
        action_z = self.encoder_action(action)

        def jax_cat(arrays, axis=0):
            if sum(len(a) for a in arrays) == 0:
                return jnp.array([], dtype=jnp.float32)
            return jnp.concatenate([a for a in arrays if len(a)], axis=axis)

        def new_axis_broadcast(x, axis, size):
            x_expanded = jnp.expand_dims(x, axis=axis)
            shape = list(x_expanded.shape)
            shape[axis] = size
            return jnp.broadcast_to(x_expanded, tuple(shape))

        global_cont_abs_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]]
        global_cont_rel_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]]
        global_cont_nullbit_in = obs[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]]
        global_binary_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.BINARIES]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.CATEGORICALS]]
        global_threshold_ins = [obs[:, ind] for ind in self.abs_index[Group.GLOBAL][Group.THRESHOLDS]]
        global_cont_abs_z = self.encoder_global_cont_abs(global_cont_abs_in)
        global_cont_rel_z = self.encoder_global_cont_rel(global_cont_rel_in)
        global_cont_nullbit_z = self.encoder_global_cont_nullbit(global_cont_nullbit_in)
        global_binary_z = jax_cat([lin(x) for lin, x in zip(self.encoders_global_binaries, global_binary_ins)], axis=-1)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = jax_cat([enc(x.argmax(axis=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], axis=-1)
        global_threshold_z = jax_cat([lin(x) for lin, x in zip(self.encoders_global_thresholds, global_threshold_ins)], axis=-1)
        global_merged = jax_cat([action_z, global_cont_abs_z, global_cont_rel_z, global_cont_nullbit_z, global_binary_z, global_categorical_z, global_threshold_z], axis=-1)
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
        player_binary_z = jax_cat([lin(x) for lin, x in zip(self.encoders_player_binaries, player_binary_ins)], axis=-1)
        player_categorical_z = jax_cat([enc(x.argmax(axis=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], axis=-1)
        player_threshold_z = jax_cat([lin(x) for lin, x in zip(self.encoders_player_thresholds, player_threshold_ins)], axis=-1)
        player_merged = jax_cat([new_axis_broadcast(action_z, 1, 2), player_cont_abs_z, player_cont_rel_z, player_cont_nullbit_z, player_binary_z, player_categorical_z, player_threshold_z], axis=-1)
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
        hex_binary_z = jax_cat([lin(x) for lin, x in zip(self.encoders_hex_binaries, hex_binary_ins)], axis=-1)
        hex_categorical_z = jax_cat([enc(x.argmax(axis=-1)) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)], axis=-1)
        hex_threshold_z = jax_cat([lin(x) for lin, x in zip(self.encoders_hex_thresholds, hex_threshold_ins)], axis=-1)
        hex_merged = jax_cat([new_axis_broadcast(action_z, 1, 165), hex_cont_abs_z, hex_cont_rel_z, hex_cont_nullbit_z, hex_binary_z, hex_categorical_z, hex_threshold_z], axis=-1)
        z_hex1 = self.encoder_merged_hex(hex_merged)
        z_hex2 = self.transformer_hex(z_hex1)
        # => (B, 165, Z_HEX)

        mean_z_player = z_player.mean(axis=1)
        mean_z_hex = z_hex2.mean(axis=1)
        z_agg = self.aggregator(jax_cat([z_global, mean_z_player, mean_z_hex], axis=-1))
        # => (B, Z_AGG)

        #
        # Outputs
        #

        global_out = self.head_global(z_agg)
        # => (B, STATE_SIZE_GLOBAL)

        player_out = self.head_player(jax_cat([new_axis_broadcast(z_agg, 1, 2), z_player], axis=-1))
        # => (B, 2, STATE_SIZE_ONE_PLAYER)

        hex_out = self.head_hex(jax_cat([new_axis_broadcast(z_agg, 1, 165), z_hex2], axis=-1))
        # => (B, 165, STATE_SIZE_ONE_HEX)

        obs_out = jax_cat([global_out, player_out.reshape(player_out.shape[0], -1), hex_out.reshape(hex_out.shape[0], -1)], axis=1)

        return obs_out

    # XXX: this is not called via __call__ => cannot use to self.make_rng, self.abs_index, etc.
    # unless called like this:
    #   jax_model.apply(
    #       jax_params,
    #       jax_obs_pred_raw,
    #       rngs={"reconstruct": jax.random.PRNGKey(0)},
    #       method=FlaxTransitionModel.reconstruct
    #   )

    def reconstruct(self, obs_out):
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
        next_obs = jnp.zeros_like(obs_out)

        reconstruct_continuous = lambda logits: jnp.clip(logits, a_min=0, a_max=1)

        # PROBS = enum.auto()     # clamp(cont) + sigmoid(bin) + softmax(cat)
        # SAMPLES = enum.auto()   # clamp(cont) + sample(sigmoid(bin)) + sample(softmax(cat))
        # GREEDY = enum.auto()    # clamp(cont) + round(sigmoid(bin)) + argmax(cat)

        def reconstruct_binary(logits):
            return (logits > 0).astype(jnp.float32)

        def reconstruct_categorical(logits):
            return jnn.one_hot(jnp.argmax(logits, axis=-1), num_classes=logits.shape[-1]).astype(jnp.float32)

        next_obs = next_obs.at[:, self.abs_index[Group.GLOBAL][Group.CONT_ABS]].set(reconstruct_continuous(global_cont_abs_out))
        next_obs = next_obs.at[:, self.abs_index[Group.GLOBAL][Group.CONT_REL]].set(reconstruct_continuous(global_cont_rel_out))
        next_obs = next_obs.at[:, self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT]].set(reconstruct_continuous(global_cont_nullbit_out))
        for ind, out in zip(self.abs_index[Group.GLOBAL][Group.BINARIES], global_binary_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_binary(out))
        for ind, out in zip(self.abs_index[Group.GLOBAL][Group.CATEGORICALS], global_categorical_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_categorical(out))
        for ind, out in zip(self.abs_index[Group.GLOBAL][Group.THRESHOLDS], global_threshold_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_binary(out))

        next_obs = next_obs.at[:, self.abs_index[Group.PLAYER][Group.CONT_ABS]].set(reconstruct_continuous(player_cont_abs_out))
        next_obs = next_obs.at[:, self.abs_index[Group.PLAYER][Group.CONT_REL]].set(reconstruct_continuous(player_cont_rel_out))
        next_obs = next_obs.at[:, self.abs_index[Group.PLAYER][Group.CONT_NULLBIT]].set(reconstruct_continuous(player_cont_nullbit_out))
        for ind, out in zip(self.abs_index[Group.PLAYER][Group.BINARIES], player_binary_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_binary(out))
        for ind, out in zip(self.abs_index[Group.PLAYER][Group.CATEGORICALS], player_categorical_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_categorical(out))
        for ind, out in zip(self.abs_index[Group.PLAYER][Group.THRESHOLDS], player_threshold_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_binary(out))

        next_obs = next_obs.at[:, self.abs_index[Group.HEX][Group.CONT_ABS]].set(reconstruct_continuous(hex_cont_abs_out))
        next_obs = next_obs.at[:, self.abs_index[Group.HEX][Group.CONT_REL]].set(reconstruct_continuous(hex_cont_rel_out))
        next_obs = next_obs.at[:, self.abs_index[Group.HEX][Group.CONT_NULLBIT]].set(reconstruct_continuous(hex_cont_nullbit_out))
        for ind, out in zip(self.abs_index[Group.HEX][Group.BINARIES], hex_binary_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_binary(out))
        for ind, out in zip(self.abs_index[Group.HEX][Group.CATEGORICALS], hex_categorical_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_categorical(out))
        for ind, out in zip(self.abs_index[Group.HEX][Group.THRESHOLDS], hex_threshold_outs):
            next_obs = next_obs.at[:, ind].set(reconstruct_binary(out))

        return next_obs

    def predict_batch(self, obs, action):
        logits_pred = self(obs, action)
        return self.reconstruct(logits_pred)

    def predict(self, obs, action):
        obs = jnp.expand_dims(obs, axis=0).astype(jnp.float32)
        action = jnp.array([action])
        return self.predict_batch(obs, action)[0]


if __name__ == "__main__":
    from ...t10n import TransitionModel

    # INIT
    import torch
    torch_model = TransitionModel()
    torch_model.eval()

    jax_model = FlaxTransitionModel(deterministic=True)
    jax_params = jax_model.init(
        rngs={"params": jax.random.PRNGKey(0)},
        obs=jnp.zeros([2, DIM_OBS]),
        action=jnp.array([0, 0])
    )

    # LOAD
    torch_state = torch.load("hauzybxn-model.pt", weights_only=True, map_location="cpu")
    torch_model.load_state_dict(torch_state)

    from .load_utils import load_params_from_torch_state
    jax_params = freeze({
        "params": load_params_from_torch_state(unfreeze(jax_params)["params"], torch_state, head_names=["global", "player", "hex"])
    })

    # TEST

    @jax.jit
    def jit_fwd(params, obs, act):
        return jax_model.apply(jax_params, obs, act)

    @jax.jit
    def jit_reconstruct(params, obs_out, rng_key):
        return jax_model.apply(
            params,
            obs_out,
            rngs={'reconstruct': rng_key},
            method=FlaxTransitionModel.reconstruct,
        )

    @jax.jit
    def jit_predict(params, obs, act, rng_key):
        return jax_model.apply(
            params,
            obs,
            act,
            method=FlaxTransitionModel.predict,
            rngs={'reconstruct': rng_key},
        )

    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    env = VcmiEnv(
        mapname="gym/generated/evaluation/8x512.vmap",
        opponent="BattleAI",
        swap_sides=0,
        random_heroes=1,
        random_obstacles=1,
        town_chance=20,
        warmachine_chance=30,
        random_terrain_chance=100,
        random_stack_chance=70,
        tight_formation_chance=30,
    )

    env.reset()

    with torch.no_grad():
        from vcmi_gym.envs.v12.decoder.decoder import Decoder

        act = env.random_action()
        obs, rew, term, trunc, _info = env.step(act)

        for i in range(1, len(obs["transitions"]["observations"])):
            obs_prev = obs["transitions"]["observations"][i-1]
            act_prev = obs["transitions"]["actions"][i-1]
            obs_next = obs["transitions"]["observations"][i]
            # mask_next = obs["transitions"]["action_masks"][i]
            # rew_next = obs["transitions"]["rewards"][i]
            # done_next = (term or trunc) and i == len(obs["transitions"]["observations"]) - 1

            torch_obs_pred_raw = torch_model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(act_prev).unsqueeze(0))
            jax_obs_pred_raw = jax_model.apply(jax_params, obs_prev.reshape(1, -1), act_prev.reshape(1))
            jit_obs_pred_raw = jit_fwd(jax_params, obs_prev.reshape(1, -1), act_prev.reshape(1))

            torch_recon = torch_model.reconstruct(torch_obs_pred_raw)
            jax_recon = jax_model.apply(jax_params, jax_obs_pred_raw, rngs={"reconstruct": jax.random.PRNGKey(0)}, method=FlaxTransitionModel.reconstruct)
            jit_recon = jit_reconstruct(jax_params, jit_obs_pred_raw, jax.random.PRNGKey(0))

            torch_bf = Decoder.decode(torch_recon[0].numpy())
            jax_bf = Decoder.decode(np.array(jax_recon[0]))
            jit_bf = Decoder.decode(np.array(jit_recon[0]))

            print("TORCH BF:")
            print(torch_bf.render(0))
            print("JAX BF:")
            print(jax_bf.render(0))
            print("JIT BF:")
            print(jit_bf.render(0))

            pred_bf = Decoder.decode(np.asarray(jit_predict(jax_params, obs_prev, act_prev, jax.random.PRNGKey(0))))
            print("PRED BF:")
            print(pred_bf.render(0))

            # BENCHMARKS
            import time

            print("Benchmarking torch (100)...")
            torch_start = time.perf_counter()
            for _ in range(100):
                torch_obs_pred_raw = torch_model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(act_prev).unsqueeze(0))
                torch_recon = torch_model.reconstruct(torch_obs_pred_raw)
                print(".", end="", flush=True)
            torch_end = time.perf_counter()
            print("\ntorch: %.2fs" % (torch_end - torch_start))

            print("Benchmarking jax (5)...")
            jax_start = time.perf_counter()
            for _ in range(5):
                jax_obs_pred_raw = jax_model.apply(jax_params, obs_prev.reshape(1, -1), act_prev.reshape(1))
                jax_recon = jax_model.apply(jax_params, jax_obs_pred_raw, rngs={"reconstruct": jax.random.PRNGKey(0)}, method=FlaxTransitionModel.reconstruct)
                print(".", end="", flush=True)
            jax_end = time.perf_counter()
            print("\njax: %.2fs" % (jax_end - jax_start))

            print("Benchmarking jit (100)...")
            jit_start = time.perf_counter()
            for _ in range(100):
                jit_obs_pred_raw = jit_fwd(jax_params, obs_prev.reshape(1, -1), act_prev.reshape(1))
                jit_recon = jit_reconstruct(jax_params, jit_obs_pred_raw, jax.random.PRNGKey(0))
                print(".", end="", flush=True)
            jit_end = time.perf_counter()
            print("\njit: %.2fs" % (jit_end - jit_start))

            print("Benchmarking pred (100)...")
            jit_start = time.perf_counter()
            for _ in range(100):
                jit_recon = jit_predict(jax_params, obs_prev, act_prev, jax.random.PRNGKey(0))
                print(".", end="", flush=True)
            jit_end = time.perf_counter()
            print("\npred: %.2fs" % (jit_end - jit_start))

            import ipdb; ipdb.set_trace()  # noqa
