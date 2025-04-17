import torch
from functools import partial

from .constants_v11 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
)


class ObsIndex:
    def __init__(self, device):
        self.device = device

        self.rel_index_global = {"continuous": [], "binary": [], "categoricals": []}
        self.rel_index_player = {"continuous": [], "binary": [], "categoricals": []}
        self.rel_index_hex = {"continuous": [], "binary": [], "categoricals": []}

        self._add_indices(GLOBAL_ATTR_MAP, self.rel_index_global)
        self._add_indices(PLAYER_ATTR_MAP, self.rel_index_player)
        self._add_indices(HEX_ATTR_MAP, self.rel_index_hex)

        for index in [self.rel_index_global, self.rel_index_player, self.rel_index_hex]:
            for type in ["continuous", "binary"]:
                index[type] = torch.tensor(index[type], device=self.device)

            index["categoricals"] = [torch.tensor(ind, device=self.device) for ind in index["categoricals"]]

        self._build_obs_indices()

    def _add_indices(self, attr_map, index):
        i = 0

        for attr, (enctype, offset, n, vmax) in attr_map.items():
            length = n
            if enctype.endswith("EXPLICIT_NULL"):
                if not enctype.startswith("CATEGORICAL"):
                    index["binary"].append(i)
                    i += 1
                    length -= 1
            elif enctype.endswith("IMPLICIT_NULL"):
                raise Exception("IMPLICIT_NULL is not supported")
            elif enctype.endswith("MASKING_NULL"):
                raise Exception("MASKING_NULL is not supported")
            elif enctype.endswith("STRICT_NULL"):
                pass
            elif enctype.endswith("ZERO_NULL"):
                pass
            else:
                raise Exception("Unexpected enctype: %s" % enctype)

            t = None
            if enctype.startswith("ACCUMULATING"):
                t = "binary"
            elif enctype.startswith("BINARY"):
                t = "binary"
            elif enctype.startswith("CATEGORICAL"):
                t = "categorical"
            elif enctype.startswith("EXPNORM"):
                t = "continuous"
            elif enctype.startswith("LINNORM"):
                t = "continuous"
            else:
                raise Exception("Unexpected enctype: %s" % enctype)

            if t == "categorical":
                ind = []
                index["categoricals"].append(ind)
                for _ in range(length):
                    ind.append(i)
                    i += 1
            else:
                for _ in range(length):
                    index[t].append(i)
                    i += 1

    # Index for extracting values from (batched) observation
    # This is different than the other indexes:
    # - self.rel_index_hex contains *relative* indexes for 1 hex
    # - self.abs_index["hex"] contains *absolute* indexes for all 165 hexes
    def _build_obs_indices(self):
        t = lambda ary: torch.tensor(ary, dtype=torch.int64, device=self.device)

        # XXX: Discrete (or "noncontinuous") is a combination of binary + categoricals
        #      where for direct extraction from obs
        self.abs_index = {
            "global": {"continuous": t([]), "binary": t([]), "categoricals": [], "categorical": t([]), "discrete": t([])},
            "player": {"continuous": t([]), "binary": t([]), "categoricals": [], "categorical": t([]), "discrete": t([])},
            "hex": {"continuous": t([]), "binary": t([]), "categoricals": [], "categorical": t([]), "discrete": t([])},
        }

        # Global

        if self.rel_index_global["continuous"].numel():
            self.abs_index["global"]["continuous"] = self.rel_index_global["continuous"]

        if self.rel_index_global["binary"].numel():
            self.abs_index["global"]["binary"] = self.rel_index_global["binary"]

        if self.rel_index_global["categoricals"]:
            self.abs_index["global"]["categoricals"] = self.rel_index_global["categoricals"]

        self.abs_index["global"]["categorical"] = torch.cat(tuple(self.abs_index["global"]["categoricals"]), dim=0)

        global_discrete = torch.zeros(0, dtype=torch.int64, device=self.device)
        global_discrete = torch.cat((global_discrete, self.abs_index["global"]["binary"]), dim=0)
        global_discrete = torch.cat((global_discrete, *self.abs_index["global"]["categoricals"]), dim=0)
        self.abs_index["global"]["discrete"] = global_discrete

        # Helper function to reduce code duplication
        # Essentially replaces this:
        # if len(model.rel_index_player["binary"]):
        #     ind = torch.zeros([2, len(model.rel_index_player["binary"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.rel_index_player["binary"] + offset
        #     obs_index["player"]["binary"] = ind
        # if len(model.rel_index_player["continuous"]):
        #     ind = torch.zeros([2, len(model.rel_index_player["continuous"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.rel_index_player["continuous"] + offset
        #     obs_index["player"]["continuous"] = ind
        # if len(model.rel_index_player["categoricals"]):
        #     for cat_ind in model.rel_index_player["categoricals"]:
        #         ind = torch.zeros([2, len(cat_ind)], dtype=torch.int64)
        #         for i in range(2):
        #             offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #             ind[i, :] = cat_ind + offset
        #         obs_index["player"]["categoricals"].append(cat_ind)
        # ...
        # - `indexes` is an array of *relative* indexes for 1 element (e.g. hex)
        def repeating_index(n, base_offset, repeating_offset, indexes):
            if indexes.numel() == 0:
                return torch.zeros([n, 0], dtype=torch.int64, device=self.device)
            ind = torch.zeros([n, len(indexes)], dtype=torch.int64, device=self.device)
            for i in range(n):
                offset = base_offset + i*repeating_offset
                ind[i, :] = indexes + offset

            return ind

        # Players (2)
        repind_players = partial(
            repeating_index,
            2,
            STATE_SIZE_GLOBAL,
            STATE_SIZE_ONE_PLAYER
        )

        self.abs_index["player"]["continuous"] = repind_players(self.rel_index_player["continuous"])
        self.abs_index["player"]["binary"] = repind_players(self.rel_index_player["binary"])
        for cat_ind in self.rel_index_player["categoricals"]:
            self.abs_index["player"]["categoricals"].append(repind_players(cat_ind))

        self.abs_index["player"]["categorical"] = torch.cat(tuple(self.abs_index["player"]["categoricals"]), dim=1)

        player_discrete = torch.zeros([2, 0], dtype=torch.int64, device=self.device)
        player_discrete = torch.cat((player_discrete, self.abs_index["player"]["binary"]), dim=1)
        player_discrete = torch.cat((player_discrete, *self.abs_index["player"]["categoricals"]), dim=1)
        self.abs_index["player"]["discrete"] = player_discrete

        # Hexes (165)
        repind_hexes = partial(
            repeating_index,
            165,
            STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER,
            STATE_SIZE_ONE_HEX
        )

        self.abs_index["hex"]["continuous"] = repind_hexes(self.rel_index_hex["continuous"])
        self.abs_index["hex"]["binary"] = repind_hexes(self.rel_index_hex["binary"])
        for cat_ind in self.rel_index_hex["categoricals"]:
            self.abs_index["hex"]["categoricals"].append(repind_hexes(cat_ind))
        self.abs_index["hex"]["categorical"] = torch.cat(tuple(self.abs_index["hex"]["categoricals"]), dim=1)

        hex_discrete = torch.zeros([165, 0], dtype=torch.int64, device=self.device)
        hex_discrete = torch.cat((hex_discrete, self.abs_index["hex"]["binary"]), dim=1)
        hex_discrete = torch.cat((hex_discrete, *self.abs_index["hex"]["categoricals"]), dim=1)
        self.abs_index["hex"]["discrete"] = hex_discrete
