

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

    jax_attn_params['query']['kernel'] = q_w.numpy().T.reshape(D, H, head_dim)
    jax_attn_params['key']['kernel'] = k_w.numpy().T.reshape(D, H, head_dim)
    jax_attn_params['value']['kernel'] = v_w.numpy().T.reshape(D, H, head_dim)

    jax_attn_params['query']['bias'] = q_b.numpy().reshape(H, head_dim)
    jax_attn_params['key']['bias'] = k_b.numpy().reshape(H, head_dim)
    jax_attn_params['value']['bias'] = v_b.numpy().reshape(H, head_dim)

    out_w = torch_attn_state[f"{torch_prefix}out_proj.weight"]  # (D, D)
    out_b = torch_attn_state[f"{torch_prefix}out_proj.bias"]    # (D,)
    jax_attn_params['out']['kernel'] = out_w.numpy().T.reshape(H, head_dim, D)
    jax_attn_params['out']['bias'] = out_b.numpy()  # stays (D,)

    return jax_attn_params


def load_generic(torch_state, jax_params, torch_key, jax_keys, transpose=False):
    assert len(jax_keys) > 1
    jax_leaf = dig(jax_params, jax_keys[:-1])
    to_assign = torch_state[torch_key]
    if transpose:
        to_assign = torch_state[torch_key].T

    assert jax_leaf[jax_keys[-1]].shape == tuple(to_assign.shape), f"{jax_keys} == {torch_key}: {jax_leaf[jax_keys[-1]].shape} == {tuple(to_assign.shape)}"
    jax_leaf[jax_keys[-1]] = to_assign.numpy()


def dig(data, keys):
    for key in keys:
        assert isinstance(data, dict), f"not a dict: {data}"
        assert key in data, f"'{key}' not found in: {data.keys()}"
        data = data[key]
    return data


def leaf_key_paths(d: dict, parent_path=()):
    paths = []
    for key, value in d.items():
        current_path = parent_path + (key,)
        if isinstance(value, dict) and value:
            paths.extend(leaf_key_paths(value, current_path))
        else:
            paths.append(current_path)
    return paths


def load_params_from_torch_state(jax_params, torch_state, head_names, action=True):
    torch_to_jax_mapping = {}

    if action:
        torch_to_jax_mapping["encoder_action.weight"] = ['encoder_action', 'embedding']

    # torch keys obtained via `torch_params.keys()`
    # jax keys obtained via `[print(path) for path in leaf_key_paths(jax_params)]`
    # NOTE: transformer handled separately (see below)
    torch_to_jax_mapping = {
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
    }

    for head_name in head_names:
        torch_to_jax_mapping.update({
            f"head_{head_name}.weight":  [f"head_{head_name}", 'kernel'],
            f"head_{head_name}.bias":    [f"head_{head_name}", 'bias'],
        })

    for torch_key, jax_keys in torch_to_jax_mapping.items():
        transpose = jax_keys[-1] == "kernel"
        load_generic(torch_state, jax_params, torch_key, jax_keys, transpose)

    num_layers = sum(1 for k in torch_state.keys() if k.startswith("transformer_hex.layers.") and k.endswith(".in_proj_weight"))

    for i in range(num_layers):
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

    return jax_params
