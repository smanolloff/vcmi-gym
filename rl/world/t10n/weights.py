import torch
from ..util.constants_v12 import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
)


class FixedKeysDict(dict):
    def __init__(self, default_value, allowed_keys):
        super().__init__({k: default_value for k in allowed_keys})

    def __setitem__(self, key, value):
        assert key in self.keys()
        super().__setitem__(key, value)


def build_feature_weights(model, weights_config):
    obsind = model.obs_index

    attrnames = {
        "global": list(GLOBAL_ATTR_MAP.keys()),
        "player": list(PLAYER_ATTR_MAP.keys()),
        "hex": list(HEX_ATTR_MAP.keys())
    }

    feature_weights = {}

    for group in obsind.attr_ids.keys():
        # global/player/hex
        feature_weights[group] = {}
        for subtype in obsind.attr_ids[group].keys():
            # continuous/cont_nullbit/binaries/...
            feature_weights[group][subtype] = torch.zeros(len(obsind.attr_ids[group][subtype]), device=model.device)
            for i, var_id in enumerate(obsind.attr_ids[group][subtype]):
                var_name = attrnames[group][var_id]
                var_weight = weights_config[group][var_name]
                feature_weights[group][subtype][i] = var_weight

    return feature_weights


weights = {
    "global": FixedKeysDict(1.0, GLOBAL_ATTR_MAP.keys()),
    "player": FixedKeysDict(1.0, PLAYER_ATTR_MAP.keys()),
    "hex": FixedKeysDict(1.0, HEX_ATTR_MAP.keys()),
}

# Fix bug where BATTLE_SIDE results in very bad loss?
weights["global"]["BATTLE_SIDE"] = 0.0
