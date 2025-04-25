import torch

from .constants_v12 import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
)


def build_feature_weights(model, weights_config):
    obsind = model.obs_index

    attrnames = {
        "global": list(GLOBAL_ATTR_MAP.keys()),
        "player": list(PLAYER_ATTR_MAP.keys()),
        "hex": list(HEX_ATTR_MAP.keys())
    }

    feature_weights = {}

    for group in obsind.var_ids.keys():
        # global/player/hex
        feature_weights[group] = {}
        for subtype in obsind.var_ids[group].keys():
            # continuous/cont_nullbit/binaries/...
            feature_weights[group][subtype] = torch.zeros(len(obsind.var_ids[group][subtype]), device=model.device)
            for i, var_id in enumerate(obsind.var_ids[group][subtype]):
                var_name = attrnames[group][var_id]
                var_weight = weights_config[group][var_name]
                feature_weights[group][subtype][i] = var_weight

    return feature_weights
