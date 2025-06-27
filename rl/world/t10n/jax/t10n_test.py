# flake8: noqa: E241
import torch
import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import math
from flax.core import freeze, unfreeze

from .obs_index import ObsIndex, Group, ContextGroup, DataGroup

from ...util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    N_HEX_ACTIONS,
    DIM_OTHER,
    DIM_HEXES,
    DIM_OBS,
)


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
        print("[JAX] input: %s" % x)
        x = self.self_attn(x)
        print("[JAX] self_attn: %s" % x)
        x = self.dropout1(x, deterministic=self.deterministic)
        x = self.norm1(residual + x)
        print("[JAX] norm1: %s" % x)

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
            print("JAX MID-RES: %s" % x)
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

        # torch.cat which returns empty tensor if tuple is empty
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

        # return obs_out
        return {
            "global_cont_abs_z": global_cont_abs_z,
            "global_cont_rel_z": global_cont_rel_z,
            "global_cont_nullbit_z": global_cont_nullbit_z,
            "global_binary_z": global_binary_z,
            "global_categorical_z": global_categorical_z,
            "global_threshold_z": global_threshold_z,
            "global_merged": global_merged,
            "z_global": z_global,
            "player_cont_abs_z": player_cont_abs_z,
            "player_cont_rel_z": player_cont_rel_z,
            "player_cont_nullbit_z": player_cont_nullbit_z,
            "player_binary_z": player_binary_z,
            "player_categorical_z": player_categorical_z,
            "player_threshold_z": player_threshold_z,
            "player_merged": player_merged,
            "z_player": z_player,
            "hex_cont_abs_z": hex_cont_abs_z,
            "hex_cont_rel_z": hex_cont_rel_z,
            "hex_cont_nullbit_z": hex_cont_nullbit_z,
            "hex_binary_z": hex_binary_z,
            "hex_categorical_z": hex_categorical_z,
            "hex_threshold_z": hex_threshold_z,
            "hex_merged": hex_merged,
            "z_hex1": z_hex1,
            "z_hex2": z_hex2,
            "z_agg": z_agg,
            "global_out": global_out,
            "player_out": player_out,
            "hex_out": hex_out,
            "obs_out": obs_out,
        }


def dig(data, keys):
    for key in keys:
        assert isinstance(data, dict), f"not a dict: {data}"
        assert key in data, f"'{key}' not found in: {data.keys()}"
        data = data[key]
    return data


def load_self_attn(torch_attn_state, jax_attn_params, torch_prefix="", jax_keys=[]):
    jax_attn_params = dig(jax_attn_params, jax_keys)
    in_w = torch_attn_state[f"{torch_prefix}in_proj_weight"]   # (3*D, D)
    in_b = torch_attn_state[f"{torch_prefix}in_proj_bias"]     # (3*D,)
    qkv_size = in_w.shape[0]
    D = qkv_size // 3
    H = jax_attn_params["query"]["bias"].shape[0]
    head_dim = D // H

    # split into query, key, value
    q_w, k_w, v_w = in_w.split(D, dim=0)   # each (D, D)
    q_b, k_b, v_b = in_b.split(D, dim=0)   # each (D,)

    jax_attn_params['query']['kernel']   = q_w.numpy().T.reshape(D, H, head_dim)
    jax_attn_params['query']['bias']     = q_b.numpy().reshape(H, head_dim)
    jax_attn_params['key']['kernel']     = k_w.numpy().T.reshape(D, H, head_dim)
    jax_attn_params['key']['bias']       = k_b.numpy().reshape(H, head_dim)
    jax_attn_params['value']['kernel']   = v_w.numpy().T.reshape(D, H, head_dim)
    jax_attn_params['value']['bias']     = v_b.numpy().reshape(H, head_dim)

    out_w = torch_attn_state[f"{torch_prefix}out_proj.weight"]  # (D, D)
    out_b = torch_attn_state[f"{torch_prefix}out_proj.bias"]    # (D,)
    jax_attn_params['out']['kernel'] = out_w.numpy().T.reshape(H, head_dim, D)
    jax_attn_params['out']['bias']   = out_b.numpy()  # stays (D,)

    return jax_attn_params


def load_generic(torch_state, jax_params, torch_key, jax_keys, transpose=False):
    assert len(jax_keys) > 1
    jax_leaf = dig(jax_params, jax_keys[:-1])
    to_assign = torch_state[torch_key]
    if transpose:
        to_assign = torch_state[torch_key].T

    assert jax_leaf[jax_keys[-1]].shape == tuple(to_assign.shape), f"{jax_keys} == {torch_key}: {jax_leaf[jax_keys[-1]].shape} == {tuple(to_assign.shape)}"
    jax_leaf[jax_keys[-1]] = to_assign.numpy()


def leaf_key_paths(d: dict, parent_path=()):
    paths = []
    for key, value in d.items():
        current_path = parent_path + (key,)
        if isinstance(value, dict) and value:
            paths.extend(leaf_key_paths(value, current_path))
        else:
            paths.append(current_path)
    return paths


if __name__ == "__main__":
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    from ..t10n import TransitionModel
    torch_model = TransitionModel()

    # INIT
    torch_state = torch.load("hauzybxn-model.pt", weights_only=True, map_location="cpu")
    torch_model.load_state_dict(torch_state)
    torch_model.eval()  # NOTE: this changes torch codepath and prevents breakpoints

    jax_model = FlaxTransitionModel(deterministic=True)
    jax_params = unfreeze(jax_model.init(
        rngs={"params": jax.random.PRNGKey(0)},
        obs=jnp.zeros([1, DIM_OBS]),
        action=jnp.array([0])
    ))["params"]

    # LOAD

    # torch keys obtained via `torch_params.keys()`
    # jax keys obtained via `[print(path) for path in leaf_key_paths(jax_params)]`
    # NOTE: transformer handled separately (see below)
    torch_to_jax_mapping = {
        'encoder_action.weight':                    ['encoder_action', 'embedding'],
        'encoders_global_binaries.0.weight':        ['encoders_global_binaries_0', 'kernel'],
        'encoders_global_binaries.0.bias':          ['encoders_global_binaries_0', 'bias'],
        'encoders_global_categoricals.0.weight':    ['encoders_global_categoricals_0', 'embedding'],
        'encoders_global_categoricals.1.weight':    ['encoders_global_categoricals_1', 'embedding'],
        'encoders_global_categoricals.2.weight':    ['encoders_global_categoricals_2', 'embedding'],
        'encoder_merged_global.0.weight':           ['encoder_merged_global', 'layers_0', 'kernel'],
        'encoder_merged_global.0.bias':             ['encoder_merged_global', 'layers_0', 'bias'],
        'encoders_player_categoricals.0.weight':    ['encoders_player_categoricals_0', 'embedding'],
        'encoder_merged_player.0.weight':           ['encoder_merged_player', 'layers_0', 'kernel'],
        'encoder_merged_player.0.bias':             ['encoder_merged_player', 'layers_0', 'bias'],
        'encoders_hex_binaries.0.weight':           ['encoders_hex_binaries_0', 'kernel'],
        'encoders_hex_binaries.0.bias':             ['encoders_hex_binaries_0', 'bias'],
        'encoders_hex_binaries.1.weight':           ['encoders_hex_binaries_1', 'kernel'],
        'encoders_hex_binaries.1.bias':             ['encoders_hex_binaries_1', 'bias'],
        'encoders_hex_binaries.2.weight':           ['encoders_hex_binaries_2', 'kernel'],
        'encoders_hex_binaries.2.bias':             ['encoders_hex_binaries_2', 'bias'],
        'encoders_hex_binaries.3.weight':           ['encoders_hex_binaries_3', 'kernel'],
        'encoders_hex_binaries.3.bias':             ['encoders_hex_binaries_3', 'bias'],
        'encoders_hex_binaries.4.weight':           ['encoders_hex_binaries_4', 'kernel'],
        'encoders_hex_binaries.4.bias':             ['encoders_hex_binaries_4', 'bias'],
        'encoders_hex_categoricals.0.weight':       ['encoders_hex_categoricals_0', 'embedding'],
        'encoders_hex_categoricals.1.weight':       ['encoders_hex_categoricals_1', 'embedding'],
        'encoders_hex_categoricals.2.weight':       ['encoders_hex_categoricals_2', 'embedding'],
        'encoders_hex_categoricals.3.weight':       ['encoders_hex_categoricals_3', 'embedding'],
        'encoders_hex_categoricals.4.weight':       ['encoders_hex_categoricals_4', 'embedding'],
        'encoders_hex_categoricals.5.weight':       ['encoders_hex_categoricals_5', 'embedding'],
        'encoder_merged_hex.0.weight':              ['encoder_merged_hex', 'layers_0', 'kernel'],
        'encoder_merged_hex.0.bias':                ['encoder_merged_hex', 'layers_0', 'bias'],
        'aggregator.0.weight':                      ['aggregator', 'layers_0', 'kernel'],
        'aggregator.0.bias':                        ['aggregator', 'layers_0', 'bias'],
        'head_global.weight':                       ['head_global', 'kernel'],
        'head_global.bias':                         ['head_global', 'bias'],
        'head_player.weight':                       ['head_player', 'kernel'],
        'head_player.bias':                         ['head_player', 'bias'],
        'head_hex.weight':                          ['head_hex', 'kernel'],
        'head_hex.bias':                            ['head_hex', 'bias'],
    }

    for torch_key, jax_keys in torch_to_jax_mapping.items():
        transpose = jax_keys[-1] == "kernel"
        load_generic(torch_state, jax_params, torch_key, jax_keys, transpose)

    for i in range(torch_model.transformer_hex.num_layers):
        torch_common = f"transformer_hex.layers.{i}"
        jax_common = ["transformer_hex", f"layers_{i}"]

        load_self_attn(torch_state, jax_params, torch_prefix=f"{torch_common}.self_attn.", jax_keys=[*jax_common, "self_attn"])
        load_generic(torch_state, jax_params, f"{torch_common}.linear1.weight",  [*jax_common, "linear1", "kernel"], transpose=True)
        load_generic(torch_state, jax_params, f"{torch_common}.linear1.bias",    [*jax_common, "linear1", "bias"])
        load_generic(torch_state, jax_params, f"{torch_common}.linear2.weight",  [*jax_common, "linear2", "kernel"], transpose=True)
        load_generic(torch_state, jax_params, f"{torch_common}.linear2.bias",    [*jax_common, "linear2", "bias"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm1.weight",    [*jax_common, "norm1", "scale"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm1.bias",      [*jax_common, "norm1", "bias"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm2.weight",    [*jax_common, "norm2", "scale"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm2.bias",      [*jax_common, "norm2", "bias"])

    jax_params = freeze({"params": jax_params})

    # env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap", random_heroes=1, swap_sides=1)
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

            from vcmi_gym.envs.v12.decoder.decoder import Decoder
            torch_obs_pred_raw = torch_model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(act_prev).unsqueeze(0))
            jax_obs_pred_raw = jax_model.apply(jax_params, obs_prev.reshape(1,-1), act_prev.reshape(1))

            torch_bf = Decoder.decode(torch_model.reconstruct(torch_obs_pred_raw["obs_out"])[0].numpy())
            jax_bf = Decoder.decode(torch_model.reconstruct(torch.as_tensor(jax_obs_pred_raw["obs_out"]))[0].numpy())

            print(torch_bf.render(0))
            print(jax_bf.render(0))
            import ipdb; ipdb.set_trace()  # noqa
            pass
