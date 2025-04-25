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


weights = {
    "global": FixedKeysDict(1.0, GLOBAL_ATTR_MAP.keys()),
    "player": FixedKeysDict(1.0, PLAYER_ATTR_MAP.keys()),
    "hex": FixedKeysDict(1.0, HEX_ATTR_MAP.keys()),
}

# Fix bug where BATTLE_SIDE results in very bad loss?
weights["global"]["BATTLE_SIDE"] = 0.0
