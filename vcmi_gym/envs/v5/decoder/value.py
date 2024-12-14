from .. import pyprocconnector as pyconnector
import numpy as np
import sys

NA = pyconnector.STATE_VALUE_NA


class Value:
    def __repr__(self):
        if self.v is None:
            return "Value(null)"
        elif self.struct:
            return f"Value(struct={self.struct})"
        else:
            return f"Value(v={self.v})"

    def __init__(self, name, enctype, n, vmax, raw, raw0=None, struct_cls=None, struct_mapping=None):
        self._name = name
        self._enctype = enctype
        self._n = n
        self._vmax = vmax
        self._name_mapping = struct_mapping

        self.raw = raw
        self.raw0 = raw0
        self.v = None
        self.struct = None

        PRECISION = 1    # number of digits after "."
        ROUND_FRACS = 5  # 5 = round to nearest 0.2 (3.14 => 3.2)

        if PRECISION > 0:
            self.raw_rounded = np.round(raw, PRECISION)
        elif ROUND_FRACS > 1:
            self.raw_rounded = np.round(raw * ROUND_FRACS) / ROUND_FRACS
        else:
            self.raw_rounded = raw

        if enctype.endswith("EXPLICIT_NULL"):
            if self.raw_rounded[0] == 1:
                return
            raw_nonnull_rounded_ = self.raw_rounded[1:]
            raw_nonnull = self.raw[1:]
        else:
            raw_nonnull_rounded_ = self.raw_rounded
            raw_nonnull = self.raw

        if enctype.endswith("IMPLICIT_NULL") and not any(raw_nonnull_rounded_):
            return

        if enctype.endswith("MASKING_NULL") and raw_nonnull_rounded_[0] == NA:
            return

        if enctype.startswith("ACCUMULATING"):
            self.v = raw_nonnull_rounded_.argmin() - 1
        elif enctype.startswith("BINARY"):
            self.v = raw_nonnull_rounded_.astype(int)
        elif enctype.startswith("CATEGORICAL"):
            self.v = raw_nonnull_rounded_.argmax()
        elif enctype.startswith("NORMALIZED"):
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            self.v = round(raw_nonnull[0] * vmax)

        if self.raw0 is not None:
            reencoded = Encoder.encode(enctype, self.v, n, vmax)

            # XXX: use self.raw here (as local raw is different)
            are_equal = all(self.raw0 == reencoded)
            # are_equal = np.allclose(self.raw, reencoded, atol=10**(-PRECISION))
            # assert are_equal, f"all({raw} == {reencoded}); {name}, v={self.v}, n={n}, vmax={vmax}, enctype={enctype}"

            if are_equal:
                print("Match!")
            else:
                # for printing purposes
                display_rounding = 3
                display_values = np.round(np.stack((self.raw, self.raw_rounded, reencoded, self.raw0)), display_rounding)
                linewidth = 5 + (4 + display_rounding) * self.raw.shape[0]
                # suppress = no scientific notation
                np.set_printoptions(suppress=True, linewidth=linewidth)
                # import ipdb; ipdb.set_trace()  # noqa
                print(f"Reencoding mismatch: {name}, v={self.v}, n={n}, vmax={vmax}, enctype={enctype}")

                lines = str(display_values).split("\n")
                assert len(lines) == 4
                lines[0] = lines[0].ljust(linewidth) + "  # raw"
                lines[1] = lines[1].ljust(linewidth) + "  # raw_rounded"
                lines[2] = lines[2].ljust(linewidth) + "  # reencoded"
                lines[3] = lines[3].ljust(linewidth) + "  # raw0 (a.k.a. orig)"
                print("\n".join(lines))
                sys.exit(1)

        if struct_cls:
            assert struct_mapping
            assert isinstance(self.v, np.ndarray), type(self.v)
            # struct_mapping is a dict {"NAME": index}
            self.struct = struct_cls(**{k: int(self.v[v]) for k, v in struct_mapping.items()})


class Encoder:
    @staticmethod
    def encode(enctype, v, n, vmax):
        nn = n-1 if enctype.endswith("EXPLICIT_NULL") else n
        raw = np.zeros(nn, dtype=float)

        if v is None:
            if enctype.endswith("STRICT_NULL"):
                raise Exception("cannot have None value with STRICT_NULL encoding")
            elif enctype.endswith("ZERO_NULL"):
                raw[0] = 1
            elif enctype.endswith("MASKING_NULL"):
                raw.fill(NA)
            elif enctype.endswith("IMPLICIT_NULL"):
                pass
            elif enctype.endswith("EXPLICIT_NULL"):
                raw = np.insert(raw, 0, 1)
            elif enctype.endswith("IMPLICIT_NULL") and not any(raw):
                return
            else:
                raise Exception(f"Unexpected enctype: {enctype}")
        else:
            if enctype.startswith("NORMALIZED"):
                raw[0] = v / vmax
            elif enctype.startswith("CATEGORICAL"):
                raw[v] = 1
            elif enctype.startswith("BINARY"):
                raw[:] = v  # element-wise assign
            elif enctype.startswith("ACCUMULATING"):
                raw[:v+1] = 1
            else:
                raise Exception(f"Unexpected enctype: {enctype}")

            if enctype.endswith("EXPLICIT_NULL"):
                raw = np.insert(raw, 0, 0)

        return raw
