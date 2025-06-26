import jax.numpy as jnp
from functools import partial

from ...util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
)


class ContextGroup:
    GLOBAL = "global"
    PLAYER = "player"
    HEX = "hex"

    @classmethod
    def as_list(cls):
        return [v for k, v in vars(cls).items() if k.isupper() and isinstance(v, str)]


class DataGroup:
    CONT_ABS = "cont_abs"
    CONT_REL = "cont_rel"
    CONT_NULLBIT = "cont_nullbit"
    BINARIES = "binaries"
    CATEGORICALS = "categoricals"
    THRESHOLDS = "thresholds"

    @classmethod
    def as_list(cls):
        return [v for k, v in vars(cls).items() if k.isupper() and isinstance(v, str)]


# Convenience class (e.g. to type Group.HEX instead of ContextGroup.HEX)
class Group(ContextGroup, DataGroup):
    @staticmethod
    def as_list():
        return NotImplementedError()


class ObsIndex:
    def __init__(self):
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
        # rel_index[Group.HEX][Group.CATEGORICALS] = [
        #   [0,1,..,9],     # indexes in state for Y_COORD
        #   [10,..,23],     # indexes in state for X_COORD
        #   [42, 43],       # indexes in state for IS_REAR
        #   ...
        # ]
        #
        # attr_ids[Group.HEX][Group.CATEGORICALS] = [
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

        empty_containers = lambda: {dg: [] for dg in DataGroup.as_list()}

        self.attr_names = {
            ContextGroup.GLOBAL: list(GLOBAL_ATTR_MAP),
            ContextGroup.PLAYER: list(PLAYER_ATTR_MAP),
            ContextGroup.HEX: list(HEX_ATTR_MAP)
        }

        self.rel_index = {cg: empty_containers() for cg in ContextGroup.as_list()}
        self.attr_ids = {cg: empty_containers() for cg in ContextGroup.as_list()}

        self._add_rel_indices(GLOBAL_ATTR_MAP, self.rel_index[Group.GLOBAL], self.attr_ids[Group.GLOBAL])
        self._add_rel_indices(PLAYER_ATTR_MAP, self.rel_index[Group.PLAYER], self.attr_ids[Group.PLAYER])
        self._add_rel_indices(HEX_ATTR_MAP, self.rel_index[Group.HEX], self.attr_ids[Group.HEX])

        for index in [self.rel_index[Group.GLOBAL], self.rel_index[Group.PLAYER], self.rel_index[Group.HEX]]:
            for type in [Group.CONT_ABS, Group.CONT_REL, Group.CONT_NULLBIT]:
                index[type] = jnp.array(index[type])

            index[Group.BINARIES] = [jnp.array(ind) for ind in index[Group.BINARIES]]
            index[Group.CATEGORICALS] = [jnp.array(ind) for ind in index[Group.CATEGORICALS]]
            index[Group.THRESHOLDS] = [jnp.array(ind) for ind in index[Group.THRESHOLDS]]

        self._build_abs_indices()

    def _add_rel_indices(self, attr_map, index, attr_ids):
        i = 0

        for attr_id, (attr, (enctype, offset, n, vmax, _p)) in enumerate(attr_map.items()):
            g = None
            if enctype.startswith("ACCUMULATING"):
                g = Group.THRESHOLDS
            elif enctype.startswith("BINARY"):
                g = Group.BINARIES
            elif enctype.startswith("CATEGORICAL"):
                g = Group.CATEGORICALS
            elif enctype.startswith("EXPNORM"):
                g = Group.CONT_ABS
            elif enctype.startswith("LINNORM"):
                if attr.endswith("REL") or attr.endswith("REL0"):
                    g = Group.CONT_REL
                else:
                    g = Group.CONT_ABS
            elif enctype.startswith("EXPBIN"):
                g = Group.CONT_ABS
            elif enctype.startswith("LINBIN"):
                g = Group.CONT_ABS
            else:
                raise Exception("Unexpected enctype: %s" % enctype)

            length = n

            if enctype.endswith("EXPLICIT_NULL"):
                # NULL is "special" category or bit for CONTINUOUS encodings
                if g in [Group.CONT_ABS, Group.CONT_REL]:
                    index[Group.CONT_NULLBIT].append(i)
                    attr_ids[Group.CONT_NULLBIT].append(attr_id)
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

            if g in [Group.BINARIES, Group.CATEGORICALS, Group.THRESHOLDS]:
                ind = []
                for _ in range(length):
                    ind.append(i)
                    i += 1
                attr_ids[g].append(attr_id)
                index[g].append(ind)
            else:
                for _ in range(length):
                    index[g].append(i)
                    i += 1
                attr_ids[g].append(attr_id)

        # Sanity check
        flattened_attr_ids = [item for sublist in attr_ids.values() for item in sublist]
        assert len(set(flattened_attr_ids)) == len(attr_map)

    # Index for extracting values from (batched) observation
    # This is different than the other indexes:
    # - self.rel_index[Group.HEX] contains *relative* indexes for 1 hex
    # - self.abs_index[Group.HEX] contains *absolute* indexes for all 165 hexes
    def _build_abs_indices(self):
        t = lambda ary: jnp.array(ary, dtype=jnp.int32)

        self.abs_index = {
            Group.GLOBAL: {
                Group.CONT_ABS: t([]),      # (N_GLOBAL_CONTABS_FEATS)
                Group.CONT_REL: t([]),      # (N_GLOBAL_CONTREL_FEATS)
                Group.CONT_NULLBIT: t([]),  # (N_GLOBAL_EXPLICIT_NULL_CONT_FEATS)
                Group.BINARIES: [],         # [(N_GLOBAL_BIN_FEAT0_BITS), (N_GLOBAL_BIN_FEAT1_BITS), ...]
                Group.CATEGORICALS: [],     # [(N_GLOBAL_CAT_FEAT0_CLASSES), (N_GLOBAL_CAT_FEAT1_CLASSES), ...]
                Group.THRESHOLDS: [],       # [(N_GLOBAL_THR_FEAT0_BINS), (N_GLOBAL_THR_FEAT1_BINS), ...]
            },
            Group.PLAYER: {
                Group.CONT_ABS: t([]),      # (2, N_PLAYER_CABSONT_FEATS)
                Group.CONT_REL: t([]),      # (2, N_PLAYER_CRELONT_FEATS)
                Group.CONT_NULLBIT: t([]),  # (2, N_PLAYER_EXPLICIT_NULL_CONT_FEATS)
                Group.BINARIES: [],         # [(2, N_PLAYER_BIN_FEAT0_BITS), (2, N_PLAYER_BIN_FEAT1_BITS), ...]
                Group.CATEGORICALS: [],     # [(2, N_PLAYER_CAT_FEAT0_CLASSES), (2, N_PLAYER_CAT_FEAT1_CLASSES), ...]
                Group.THRESHOLDS: [],       # [(2, N_PLAYER_THR_FEAT0_BINS), (2, N_PLAYER_THR_FEAT1_BINS), ...]
            },
            Group.HEX: {
                Group.CONT_ABS: t([]),      # (165, N_HEX_COABSNT_FEATS)
                Group.CONT_REL: t([]),      # (165, N_HEX_CORELNT_FEATS)
                Group.CONT_NULLBIT: t([]),  # (165, N_HEX_EXPLICIT_NULL_CONT_FEATS)
                Group.BINARIES: [],         # [(165, N_HEX_BIN_FEAT0_BITS), (165, N_HEX_BIN_FEAT1_BITS), ...]
                Group.CATEGORICALS: [],     # [(165, N_HEX_CAT_FEAT0_CLASSES), (165, N_HEX_CAT_FEAT1_CLASSES), ...]
                Group.THRESHOLDS: [],       # [(165, N_HEX_THR_FEAT0_BINS), (165, N_HEX_THR_FEAT1_BINS), ...]
            },
        }

        # Global

        if self.rel_index[Group.GLOBAL][Group.CONT_ABS].size:
            self.abs_index[Group.GLOBAL][Group.CONT_ABS] = self.rel_index[Group.GLOBAL][Group.CONT_ABS]

        if self.rel_index[Group.GLOBAL][Group.CONT_REL].size:
            self.abs_index[Group.GLOBAL][Group.CONT_REL] = self.rel_index[Group.GLOBAL][Group.CONT_REL]

        if self.rel_index[Group.GLOBAL][Group.CONT_NULLBIT].size:
            self.abs_index[Group.GLOBAL][Group.CONT_NULLBIT] = self.rel_index[Group.GLOBAL][Group.CONT_NULLBIT]

        if self.rel_index[Group.GLOBAL][Group.BINARIES]:
            self.abs_index[Group.GLOBAL][Group.BINARIES] = self.rel_index[Group.GLOBAL][Group.BINARIES]

        if self.rel_index[Group.GLOBAL][Group.CATEGORICALS]:
            self.abs_index[Group.GLOBAL][Group.CATEGORICALS] = self.rel_index[Group.GLOBAL][Group.CATEGORICALS]

        if self.rel_index[Group.GLOBAL][Group.THRESHOLDS]:
            self.abs_index[Group.GLOBAL][Group.THRESHOLDS] = self.rel_index[Group.GLOBAL][Group.THRESHOLDS]

        # Helper function to reduce code duplication
        # Essentially replaces this:
        # if len(model.rel_index[Group.PLAYER]["binary"]):
        #     ind = torch.zeros([2, len(model.rel_index[Group.PLAYER]["binary"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.rel_index[Group.PLAYER]["binary"] + offset
        #     obs_index[Group.PLAYER]["binary"] = ind
        # if len(model.rel_index[Group.PLAYER]["continuous"]):
        #     ind = torch.zeros([2, len(model.rel_index[Group.PLAYER]["continuous"])], dtype=torch.int64)
        #     for i in range(2):
        #         offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #         ind[i, :] = model.rel_index[Group.PLAYER]["continuous"] + offset
        #     obs_index[Group.PLAYER]["continuous"] = ind
        # if len(model.rel_index[Group.PLAYER][Group.CATEGORICALS]):
        #     for cat_ind in model.rel_index[Group.PLAYER][Group.CATEGORICALS]:
        #         ind = torch.zeros([2, len(cat_ind)], dtype=torch.int64)
        #         for i in range(2):
        #             offset = STATE_SIZE_GLOBAL + i*STATE_SIZE_ONE_PLAYER
        #             ind[i, :] = cat_ind + offset
        #         obs_index[Group.PLAYER][Group.CATEGORICALS].append(cat_ind)
        # ...
        # - `indexes` is an array of *relative* indexes for 1 element (e.g. hex)
        def repeating_index(n, base_offset, repeating_offset, indexes):
            if indexes.size == 0:
                return jnp.zeros([n, 0], dtype=jnp.int32)
            ind = jnp.zeros([n, len(indexes)], dtype=jnp.int32)
            for i in range(n):
                offset = base_offset + i*repeating_offset
                ind = ind.at[i, :].set(indexes + offset)

            return ind

        # Players (2)
        repind_players = partial(
            repeating_index,
            2,
            STATE_SIZE_GLOBAL,
            STATE_SIZE_ONE_PLAYER
        )

        self.abs_index[Group.PLAYER][Group.CONT_ABS] = repind_players(self.rel_index[Group.PLAYER][Group.CONT_ABS])
        self.abs_index[Group.PLAYER][Group.CONT_REL] = repind_players(self.rel_index[Group.PLAYER][Group.CONT_REL])
        self.abs_index[Group.PLAYER][Group.CONT_NULLBIT] = repind_players(self.rel_index[Group.PLAYER][Group.CONT_NULLBIT])

        for bin_ind in self.rel_index[Group.PLAYER][Group.BINARIES]:
            self.abs_index[Group.PLAYER][Group.BINARIES].append(repind_players(bin_ind))

        for cat_ind in self.rel_index[Group.PLAYER][Group.CATEGORICALS]:
            self.abs_index[Group.PLAYER][Group.CATEGORICALS].append(repind_players(cat_ind))

        for thr_ind in self.rel_index[Group.PLAYER][Group.THRESHOLDS]:
            self.abs_index[Group.PLAYER][Group.THRESHOLDS].append(repind_players(thr_ind))

        # Hexes (165)
        repind_hexes = partial(
            repeating_index,
            165,
            STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER,
            STATE_SIZE_ONE_HEX
        )

        self.abs_index[Group.HEX][Group.CONT_ABS] = repind_hexes(self.rel_index[Group.HEX][Group.CONT_ABS])
        self.abs_index[Group.HEX][Group.CONT_REL] = repind_hexes(self.rel_index[Group.HEX][Group.CONT_REL])
        self.abs_index[Group.HEX][Group.CONT_NULLBIT] = repind_hexes(self.rel_index[Group.HEX][Group.CONT_NULLBIT])

        for bin_ind in self.rel_index[Group.HEX][Group.BINARIES]:
            self.abs_index[Group.HEX][Group.BINARIES].append(repind_hexes(bin_ind))

        for cat_ind in self.rel_index[Group.HEX][Group.CATEGORICALS]:
            self.abs_index[Group.HEX][Group.CATEGORICALS].append(repind_hexes(cat_ind))

        for thr_ind in self.rel_index[Group.HEX][Group.THRESHOLDS]:
            self.abs_index[Group.HEX][Group.THRESHOLDS].append(repind_hexes(thr_ind))

    def attr_name(self, context, attr_id):
        return self.attr_names[context][attr_id]
