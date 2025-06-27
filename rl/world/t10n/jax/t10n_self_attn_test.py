# flake8: noqa: E241
import torch
import jax
import jax.numpy as jnp
import flax.linen as fnn
from flax.core import freeze, unfreeze


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


if __name__ == "__main__":
    d_model = 3

    jax_attn = fnn.MultiHeadAttention(
        num_heads=1,
        qkv_features=d_model,
        out_features=d_model,
        use_bias=True,
        dropout_rate=0.0,
        deterministic=True,
        broadcast_dropout=False
    )

    torch_attn = torch.nn.MultiheadAttention(
        jax_attn.qkv_features,
        jax_attn.num_heads,
        dropout=jax_attn.dropout_rate,
        bias=jax_attn.use_bias,
        batch_first=True,
    )

    # INIT
    torch_state = torch_attn.state_dict()
    jax_params = jax_attn.init({"params": jax.random.PRNGKey(0)}, jnp.zeros([1, 1, 512]))
    jax_params = unfreeze(jax_params)["params"]

    # LOAD
    load_self_attn(torch_state, jax_params, torch_prefix="", jax_keys=[])
    jax_params = freeze({"params": jax_params})

    # TEST
    torch_in = torch.ones([1, 1, jax_attn.qkv_features])
    jax_in = torch_in.numpy()

    torch_out = torch_attn(torch_in, torch_in, torch_in)[0]
    jax_out = jax_attn.apply(jax_params, jax_in)

    torch_out2 = torch_attn(torch_out, torch_out, torch_out)[0]
    jax_out2 = jax_attn.apply(jax_params, jax_out)

    import ipdb; ipdb.set_trace()  # noqa
    pass
