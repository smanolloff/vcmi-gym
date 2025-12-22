from .. import pyconnector
import numpy as np
import math

NA = pyconnector.STATE_VALUE_NA


class Value:
    def __repr__(self):
        if self.v is None:
            return "Value(null)"
        elif self.struct:
            return f"Value(struct={self.struct})"
        else:
            return f"Value(v={self})"

    def __str__(self):
        if self.v is None:
            return ""
        elif self.struct:
            return str(self.struct)
        elif self.vrange:
            return "%s..%s" % self.vrange
        else:
            return str(self.v)

    def __init__(
        self,
        name,
        enctype,
        n,
        vmax,
        slope,
        raw,
        struct_cls=None,
        struct_mapping=None,
        verbose=False,
    ):
        self.name = name
        self.enctype = enctype
        self.n = n
        self.vmax = vmax
        self.slope = slope
        self.struct_cls = struct_cls
        self.struct_mapping = struct_mapping
        self.verbose = verbose

        self.raw = raw
        self.v = None
        self.vrange = None
        self.struct = None

        if enctype.endswith("EXPLICIT_NULL"):
            if raw.argmax() == 0 and self.raw[0] > 0.5:
                self.v = None
                return

            raw_nonnull = self.raw[1:]
        else:
            raw_nonnull = self.raw

        if enctype.endswith("IMPLICIT_NULL") and not any(raw_nonnull):
            return

        if enctype.endswith("MASKING_NULL") and raw_nonnull[0] == NA:
            return

        if enctype.startswith("ACCUMULATING"):
            self.v = raw_nonnull.argmin() - 1
        elif enctype.startswith("BINARY"):
            self.v = raw_nonnull.astype(int)
        elif enctype.startswith("CATEGORICAL"):
            self.v = raw_nonnull.argmax()
        elif enctype.startswith("EXPNORM"):
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            encoded = raw_nonnull[0]
            exp_slope = math.exp(slope)
            numerator = math.exp(encoded * (slope + 1e-6)) - 1
            denominator = exp_slope - 1
            res = (numerator / denominator) * vmax
            self.v = round(res)

        elif enctype.startswith("LINNORM"):
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            self.v = round(raw_nonnull[0] * vmax)
        elif enctype == "RAW":
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            self.v = raw_nonnull[0]

        if struct_cls:
            assert struct_mapping
            assert isinstance(self.v, np.ndarray), type(self.v)
            # struct_mapping is a dict {"NAME": index}
            self.struct = struct_cls(**{k: int(self.v[v]) for k, v in struct_mapping.items()})

    def log(self, msg):
        if self.verbose:
            print(msg)
