

def load_self_attn(torch_attn_state, jax_attn_params, torch_prefix="", jax_prefix=""):
    in_w = torch_attn_state[f"{torch_prefix}in_proj_weight"]   # (3*D, D)
    in_b = torch_attn_state[f"{torch_prefix}in_proj_bias"]     # (3*D,)
    qkv_size = in_w.shape[0]
    D = qkv_size // 3

    # split into query, key, value
    q_w, k_w, v_w = in_w.split(D, dim=0)   # each (D, D)
    q_b, k_b, v_b = in_b.split(D, dim=0)   # each (D,)

    assert jax_attn_params[f"{jax_prefix}query"]["w"].shape == q_w.numpy().T.shape
    jax_attn_params[f"{jax_prefix}query"]["w"] = q_w.numpy().T
    jax_attn_params[f"{jax_prefix}key"]["w"] = k_w.numpy().T
    jax_attn_params[f"{jax_prefix}value"]["w"] = v_w.numpy().T

    assert jax_attn_params[f"{jax_prefix}query"]["b"].shape == q_b.numpy().shape
    jax_attn_params[f"{jax_prefix}query"]["b"] = q_b.numpy()
    jax_attn_params[f"{jax_prefix}key"]["b"] = k_b.numpy()
    jax_attn_params[f"{jax_prefix}value"]["b"] = v_b.numpy()

    out_w = torch_attn_state[f"{torch_prefix}out_proj.weight"]  # (D, D)
    out_b = torch_attn_state[f"{torch_prefix}out_proj.bias"]    # (D,)
    assert jax_attn_params[f"{jax_prefix}linear"]["w"].shape == out_w.numpy().shape
    jax_attn_params[f"{jax_prefix}linear"]["w"] = out_w.numpy().T
    jax_attn_params[f"{jax_prefix}linear"]["b"] = out_b.numpy()  # stays (D,)

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
        torch_to_jax_mapping["encoder_action.weight"] = ['encoder_action', 'embeddings']

    # torch keys obtained via `torch_params.keys()`
    # jax keys obtained via `[print(path) for path in leaf_key_paths(jax_params)]`
    # NOTE: transformer handled separately (see below)
    torch_to_jax_mapping = {
        'encoders_global_binaries.0.weight':        ['encoders_global_binaries', 'w'],
        'encoders_global_binaries.0.bias':          ['encoders_global_binaries', 'b'],
        'encoders_global_categoricals.0.weight':    ['encoders_global_categoricals', 'embeddings'],
        'encoders_global_categoricals.1.weight':    ['encoders_global_categoricals_1', 'embeddings'],
        'encoders_global_categoricals.2.weight':    ['encoders_global_categoricals_2', 'embeddings'],
        'encoder_merged_global.0.weight':           ['encoder_merged_global', 'w'],
        'encoder_merged_global.0.bias':             ['encoder_merged_global', 'b'],
        'encoders_player_categoricals.0.weight':    ['encoders_player_categoricals', 'embeddings'],
        'encoder_merged_player.0.weight':           ['encoder_merged_player', 'w'],
        'encoder_merged_player.0.bias':             ['encoder_merged_player', 'b'],
        'encoders_hex_binaries.0.weight':           ['encoders_hex_binaries', 'w'],
        'encoders_hex_binaries.0.bias':             ['encoders_hex_binaries', 'b'],
        'encoders_hex_binaries.1.weight':           ['encoders_hex_binaries_1', 'w'],
        'encoders_hex_binaries.1.bias':             ['encoders_hex_binaries_1', 'b'],
        'encoders_hex_binaries.2.weight':           ['encoders_hex_binaries_2', 'w'],
        'encoders_hex_binaries.2.bias':             ['encoders_hex_binaries_2', 'b'],
        'encoders_hex_binaries.3.weight':           ['encoders_hex_binaries_3', 'w'],
        'encoders_hex_binaries.3.bias':             ['encoders_hex_binaries_3', 'b'],
        'encoders_hex_binaries.4.weight':           ['encoders_hex_binaries_4', 'w'],
        'encoders_hex_binaries.4.bias':             ['encoders_hex_binaries_4', 'b'],
        'encoders_hex_categoricals.0.weight':       ['encoders_hex_categoricals', 'embeddings'],
        'encoders_hex_categoricals.1.weight':       ['encoders_hex_categoricals_1', 'embeddings'],
        'encoders_hex_categoricals.2.weight':       ['encoders_hex_categoricals_2', 'embeddings'],
        'encoders_hex_categoricals.3.weight':       ['encoders_hex_categoricals_3', 'embeddings'],
        'encoders_hex_categoricals.4.weight':       ['encoders_hex_categoricals_4', 'embeddings'],
        'encoders_hex_categoricals.5.weight':       ['encoders_hex_categoricals_5', 'embeddings'],
        'encoder_merged_hex.0.weight':              ['encoder_merged_hex', 'w'],
        'encoder_merged_hex.0.bias':                ['encoder_merged_hex', 'b'],
        'aggregator.0.weight':                      ['aggregator', 'w'],
        'aggregator.0.bias':                        ['aggregator', 'b'],
    }

    for head_name in head_names:
        torch_to_jax_mapping.update({
            f"head_{head_name}.weight":  [f"head_{head_name}", 'w'],
            f"head_{head_name}.bias":    [f"head_{head_name}", 'b'],
        })

    for torch_key, jax_keys in torch_to_jax_mapping.items():
        transpose = jax_keys[-1] == "w"
        jax_keys[0] = f"haiku_transition_model/~/{jax_keys[0]}"
        load_generic(torch_state, jax_params, torch_key, jax_keys, transpose)

    num_layers = sum(1 for k in torch_state.keys() if k.startswith("transformer_hex.layers.") and k.endswith(".in_proj_weight"))

    for i in range(num_layers):
        torch_common = f"transformer_hex.layers.{i}"
        jax_common = f"haiku_transition_model/~/haiku_transformer_encoder/~/layer_{i}/~"

        load_self_attn(torch_state, jax_params, torch_prefix=f"{torch_common}.self_attn.", jax_prefix=f"{jax_common}/self_attn/")

        load_generic(torch_state, jax_params, f"{torch_common}.linear1.weight",  [f"{jax_common}/linear1", "w"], transpose=True)
        load_generic(torch_state, jax_params, f"{torch_common}.linear1.bias",    [f"{jax_common}/linear1", "b"])
        load_generic(torch_state, jax_params, f"{torch_common}.linear2.weight",  [f"{jax_common}/linear2", "w"], transpose=True)
        load_generic(torch_state, jax_params, f"{torch_common}.linear2.bias",    [f"{jax_common}/linear2", "b"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm1.weight",    [f"{jax_common}/norm1", "scale"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm1.bias",      [f"{jax_common}/norm1", "offset"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm2.weight",    [f"{jax_common}/norm2", "scale"])
        load_generic(torch_state, jax_params, f"{torch_common}.norm2.bias",      [f"{jax_common}/norm2", "offset"])

    return jax_params
