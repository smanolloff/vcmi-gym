import torch
from functools import partial

from .constants_v12 import (
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

        # Consider this:
        # constexpr HexEncoding HEX_ENCODING {
        #     E5(HA::Y_COORD,                 CS, 10),
        #     E5(HA::X_COORD,                 CS, 14),
        #     E5(HA::STATE_MASK,              BS, 4),
        #     E5(HA::ACTION_MASK,             BZ, 14),
        #     E5(HA::IS_REAR,                 CE, 1),        // 1=this is the rear hex of a stack
        #     ...
        #
        # =>
        # rel_index["hex"]["categoricals"] = [
        #   [0,1,..,9],     # indexes in state for Y_COORD
        #   [10,..,23],     # indexes in state for X_COORD
        #   [42, 43],       # indexes in state for IS_REAR
        #   ...
        # ]
        #
        # attr_ids["hex"]["categoricals"] = [
        #   0,              # attr_id of Y_COORD in the HEX_ENCODING definitions
        #   1,              # attr_id of X_COORD in the HEX_ENCODING definitions
        #   4,              # attr_id of IS_REAR in the HEX_ENCODING definitions
        #   ...
        # ]
        #
        # ^^^ This allows to create a "weight" matrix like this:
        # weights_hex = [
        #   1.0,            # loss weight for var_0 (Y_COORD)
        #   1.0,            # loss weight for var_1 (X_COORD)
        #   0.5,            # loss weight for var_2
        #   1.0,            # loss weight for var_3
        #   0.5,            # loss weight for var_4 (IS_REAR)
        #   ...
        # ]

        empty_containers = lambda: {
            "continuous": [],
            "cont_nullbit": [],
            "binaries": [],
            "categoricals": [],
            "thresholds": []
        }

        self.rel_index = {
            "global": empty_containers(),
            "player": empty_containers(),
            "hex": empty_containers(),
        }

        self.attr_ids = {
            "global": empty_containers(),
            "player": empty_containers(),
            "hex": empty_containers(),
        }

        self._add_rel_indices(GLOBAL_ATTR_MAP, self.rel_index["global"], self.attr_ids["global"])
        self._add_rel_indices(PLAYER_ATTR_MAP, self.rel_index["player"], self.attr_ids["player"])
        self._add_rel_indices(HEX_ATTR_MAP, self.rel_index["hex"], self.attr_ids["hex"])

        for index in [self.rel_index["global"], self.rel_index["player"], self.rel_index["hex"]]:
            for type in ["continuous", "cont_nullbit"]:
                index[type] = torch.tensor(index[type], device=self.device)

            index["binaries"] = [torch.tensor(ind, device=self.device) for ind in index["binaries"]]
            index["categoricals"] = [torch.tensor(ind, device=self.device) for ind in index["categoricals"]]
            index["thresholds"] = [torch.tensor(ind, device=self.device) for ind in index["thresholds"]]

        self._build_abs_indices()

    def _add_rel_indices(self, attr_map, index, attr_ids):
        i = 0

        for attr_id, (attr, (enctype, offset, n, vmax, _p)) in enumerate(attr_map.items()):
            t = None
            if enctype.startswith("ACCUMULATING"):
                t = "threshold"
            elif enctype.startswith("BINARY"):
                t = "binary"
            elif enctype.startswith("CATEGORICAL"):
                t = "categorical"
            elif enctype.startswith("EXPNORM"):
                t = "continuous"
            elif enctype.startswith("LINNORM"):
                t = "continuous"
            elif enctype.startswith("EXPBIN"):
                t = "continuous"
            elif enctype.startswith("LINBIN"):
                t = "continuous"
            else:
                raise Exception("Unexpected enctype: %s" % enctype)

            length = n

            if enctype.endswith("EXPLICIT_NULL"):
                # NULL is "special" category or bit for CONTINUOUS encodings
                if t == "continuous":
                    index["cont_nullbit"].append(i)
                    attr_ids["cont_nullbit"].append(attr_id)
                    i += 1
                    length -= 1
            elif enctype.endswith("IMPLICIT_NULL"):
                pass
            elif enctype.endswith("MASKING_NULL"):
                # Negative bits would probably mess up CE and BCE loss calculations
                # (CE for sure -- "all bits -1" is not a one-hot categorical encoding)
                raise Exception("MASKING_NULL is not supported")
            elif enctype.endswith("STRICT_NULL"):
                pass
            elif enctype.endswith("ZERO_NULL"):
                pass
            else:
                raise Exception("Unexpected enctype: %s" % enctype)


            if t in ["binary", "categorical", "threshold"]:
                plural = "binaries" if t == "binary" else f"{t}s"
                ind = []
                for _ in range(length):
                    ind.append(i)
                    i += 1
                attr_ids[plural].append(attr_id)
                index[plural].append(ind)
            else:
                for _ in range(length):
                    index[t].append(i)
                    i += 1
                attr_ids[t].append(attr_id)

        # Sanity check
        flattened_attr_ids = [item for sublist in attr_ids.values() for item in sublist]
        assert len(set(flattened_attr_ids)) == len(attr_map)

    # Index for extracting values from (batched) observation
    # This is different than the other indexes:
    # - self.rel_index["hex"] contains *relative* indexes for 1 hex
    # - self.abs_index["hex"] contains *absolute* indexes for all 165 hexes
    def _build_abs_indices(self):
        t = lambda ary: torch.tensor(ary, dtype=torch.int64, device=self.device)

        self.abs_index = {
            "global": {
                "continuous": t([]),    # (N_GLOBAL_CONT_FEATS)
                "cont_nullbit": t([]),  # (N_GLOBAL_EXPLICIT_NULL_CONT_FEATS)
                "binaries": [],         # [(N_GLOBAL_BIN_FEAT0_BITS), (N_GLOBAL_BIN_FEAT1_BITS), ...]
                "categoricals": [],     # [(N_GLOBAL_CAT_FEAT0_CLASSES), (N_GLOBAL_CAT_FEAT1_CLASSES), ...]
                "thresholds": [],       # [(N_GLOBAL_THR_FEAT0_BINS), (N_GLOBAL_THR_FEAT1_BINS), ...]
            },
            "player": {
                "continuous": t([]),    # (2, N_PLAYER_CONT_FEATS)
                "cont_nullbit": t([]),  # (2, N_PLAYER_EXPLICIT_NULL_CONT_FEATS)
                "binaries": [],         # [(2, N_PLAYER_BIN_FEAT0_BITS), (2, N_PLAYER_BIN_FEAT1_BITS), ...]
                "categoricals": [],     # [(2, N_PLAYER_CAT_FEAT0_CLASSES), (2, N_PLAYER_CAT_FEAT1_CLASSES), ...]
                "thresholds": [],       # [(2, N_PLAYER_THR_FEAT0_BINS), (2, N_PLAYER_THR_FEAT1_BINS), ...]
            },
            "hex": {
                "continuous": t([]),    # (165, N_HEX_CONT_FEATS)
                "cont_nullbit": t([]),  # (165, N_HEX_EXPLICIT_NULL_CONT_FEATS)
                "binaries": [],         # [(165, N_HEX_BIN_FEAT0_BITS), (165, N_HEX_BIN_FEAT1_BITS), ...]
                "categoricals": [],     # [(165, N_HEX_CAT_FEAT0_CLASSES), (165, N_HEX_CAT_FEAT1_CLASSES), ...]
                "thresholds": [],       # [(165, N_HEX_THR_FEAT0_BINS), (165, N_HEX_THR_FEAT1_BINS), ...]
            },
        }

        # Global

        if self.rel_index["global"]["continuous"].numel():
            self.abs_index["global"]["continuous"] = self.rel_index["global"]["continuous"]

        if self.rel_index["global"]["cont_nullbit"].numel():
            self.abs_index["global"]["cont_nullbit"] = self.rel_index["global"]["cont_nullbit"]

        if self.rel_index["global"]["binaries"]:
            self.abs_index["global"]["binaries"] = self.rel_index["global"]["binaries"]

        if self.rel_index["global"]["categoricals"]:
            self.abs_index["global"]["categoricals"] = self.rel_index["global"]["categoricals"]

        if self.rel_index["global"]["thresholds"]:
            self.abs_index["global"]["thresholds"] = self.rel_index["global"]["thresholds"]

        # Helper function to reduce code duplication
        # Essentially replaces this:
        # if len(model.rel_index["player"]["binary"]):
        #     ind = torch.zeros([2, len(model.rel_index["player"]["binary"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.rel_index["player"]["binary"] + offset
        #     obs_index["player"]["binary"] = ind
        # if len(model.rel_index["player"]["continuous"]):
        #     ind = torch.zeros([2, len(model.rel_index["player"]["continuous"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.rel_index["player"]["continuous"] + offset
        #     obs_index["player"]["continuous"] = ind
        # if len(model.rel_index["player"]["categoricals"]):
        #     for cat_ind in model.rel_index["player"]["categoricals"]:
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

        self.abs_index["player"]["continuous"] = repind_players(self.rel_index["player"]["continuous"])
        self.abs_index["player"]["cont_nullbit"] = repind_players(self.rel_index["player"]["cont_nullbit"])

        for bin_ind in self.rel_index["player"]["binaries"]:
            self.abs_index["player"]["binaries"].append(repind_players(bin_ind))

        for cat_ind in self.rel_index["player"]["categoricals"]:
            self.abs_index["player"]["categoricals"].append(repind_players(cat_ind))

        for thr_ind in self.rel_index["player"]["thresholds"]:
            self.abs_index["player"]["thresholds"].append(repind_players(thr_ind))

        # Hexes (165)
        repind_hexes = partial(
            repeating_index,
            165,
            STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER,
            STATE_SIZE_ONE_HEX
        )

        self.abs_index["hex"]["continuous"] = repind_hexes(self.rel_index["hex"]["continuous"])
        self.abs_index["hex"]["cont_nullbit"] = repind_hexes(self.rel_index["hex"]["cont_nullbit"])

        for bin_ind in self.rel_index["hex"]["binaries"]:
            self.abs_index["hex"]["binaries"].append(repind_hexes(bin_ind))

        for cat_ind in self.rel_index["hex"]["categoricals"]:
            self.abs_index["hex"]["categoricals"].append(repind_hexes(cat_ind))

        for thr_ind in self.rel_index["hex"]["thresholds"]:
            self.abs_index["hex"]["thresholds"].append(repind_hexes(thr_ind))
