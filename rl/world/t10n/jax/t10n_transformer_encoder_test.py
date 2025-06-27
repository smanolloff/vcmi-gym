# flake8: noqa: E241
import torch
import jax
import jax.numpy as jnp
import flax.linen as fnn
from flax.core import freeze, unfreeze


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


if __name__ == "__main__":
    jax_encoder = TransformerEncoder(
        num_layers=2,
        d_model=4,
        dim_feedforward=3,
        num_heads=1,
        dropout_rate=0.5,  # no-op (deterministic=True for jax and .eval() for torch model)
        deterministic=True,
    )

    torch_encoder = torch.nn.TransformerEncoder(
        num_layers=jax_encoder.num_layers,
        encoder_layer=torch.nn.TransformerEncoderLayer(
            d_model=jax_encoder.d_model,
            dim_feedforward=jax_encoder.dim_feedforward,
            nhead=jax_encoder.num_heads,
            dropout=jax_encoder.dropout_rate,
            batch_first=True,
        )
    )
    torch_encoder.eval()

    # INIT
    torch_state = torch_encoder.state_dict()
    jax_params = unfreeze(jax_encoder.init({"params": jax.random.PRNGKey(0)}, jnp.zeros([1, 1, jax_encoder.d_model]))["params"])

    # LOAD
    for i in range(jax_encoder.num_layers):
        load_self_attn(torch_state, jax_params, torch_prefix=f"layers.{i}.self_attn.", jax_keys=[f"layers_{i}", "self_attn"])
        load_generic(torch_state, jax_params, f"layers.{i}.linear1.weight",  [f"layers_{i}", "linear1", "kernel"], transpose=True)
        load_generic(torch_state, jax_params, f"layers.{i}.linear1.bias",    [f"layers_{i}", "linear1", "bias"])
        load_generic(torch_state, jax_params, f"layers.{i}.linear2.weight",  [f"layers_{i}", "linear2", "kernel"], transpose=True)
        load_generic(torch_state, jax_params, f"layers.{i}.linear2.bias",    [f"layers_{i}", "linear2", "bias"])
        load_generic(torch_state, jax_params, f"layers.{i}.norm1.weight",    [f"layers_{i}", "norm1", "scale"])
        load_generic(torch_state, jax_params, f"layers.{i}.norm1.bias",      [f"layers_{i}", "norm1", "bias"])
        load_generic(torch_state, jax_params, f"layers.{i}.norm2.weight",    [f"layers_{i}", "norm2", "scale"])
        load_generic(torch_state, jax_params, f"layers.{i}.norm2.bias",      [f"layers_{i}", "norm2", "bias"])
    jax_params = freeze({"params": jax_params})

    print("========================================")

    torch_in = torch.ones([1, 1, jax_encoder.d_model])
    jax_in = torch_in.numpy()

    # torch_in = torch.tensor([[[1., 1., 1., 1.]]])
    # jax_in = jnp.array([[[0.9999, 0.9999, 0.9999, 0.9999]]])


    # SELF_ATTN_TEST

    torch_out = torch_encoder(torch_in)
    jax_out = jax_encoder.apply(jax_params, jax_in)

    import ipdb; ipdb.set_trace()  # noqa
    pass
