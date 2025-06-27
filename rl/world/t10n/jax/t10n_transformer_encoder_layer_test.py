# flake8: noqa: E241
import torch
import jax
import jax.numpy as jnp
import flax.linen as fnn
from flax.core import freeze, unfreeze


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
    jax_layer = TransformerEncoderLayer(
        d_model=512,
        dim_feedforward=2048,
        num_heads=8,
        dropout_rate=0.0,
        deterministic=True,
    )

    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model=jax_layer.d_model,
        dim_feedforward=jax_layer.dim_feedforward,
        nhead=jax_layer.num_heads,
        dropout=jax_layer.dropout_rate,
        batch_first=True,
    )

    # INIT
    torch_state = torch_layer.state_dict()
    jax_params = unfreeze(jax_layer.init({"params": jax.random.PRNGKey(0)}, jnp.zeros([1, 1, jax_layer.d_model]))["params"])

    # LOAD
    load_self_attn(torch_state, jax_params, torch_prefix="self_attn.", jax_keys=["self_attn"])
    load_generic(torch_state, jax_params, 'linear1.weight',  ['linear1', 'kernel'], transpose=True)
    load_generic(torch_state, jax_params, 'linear1.bias',    ['linear1', 'bias'])
    load_generic(torch_state, jax_params, 'linear2.weight',  ['linear2', 'kernel'], transpose=True)
    load_generic(torch_state, jax_params, 'linear2.bias',    ['linear2', 'bias'])
    load_generic(torch_state, jax_params, 'norm1.weight',    ['norm1', 'scale'])
    load_generic(torch_state, jax_params, 'norm1.bias',      ['norm1', 'bias'])
    load_generic(torch_state, jax_params, 'norm2.weight',    ['norm2', 'scale'])
    load_generic(torch_state, jax_params, 'norm2.bias',      ['norm2', 'bias'])
    jax_params = freeze({"params": jax_params})

    # TEST
    torch_in = torch.ones([1, 1, jax_layer.d_model])
    jax_in = torch_in.numpy()

    torch_out = torch_layer(torch_in)
    jax_out = jax_layer.apply(jax_params, jax_in)

    import ipdb; ipdb.set_trace()  # noqa
    pass
