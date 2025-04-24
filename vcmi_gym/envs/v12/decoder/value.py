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
            return f"Value(v={self.v})"

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

        if "EXPBIN" in enctype:
            if "ACCUMULATING" in enctype:
                eb_index = raw_nonnull.argmin() - 1
            else:
                eb_index = raw_nonnull.argmax()

            eb_xlow = float(eb_index) / n
            eb_low = math.ceil((math.exp(slope * eb_xlow) - 1) / (math.exp(slope) - 1) * vmax)
            eb_xhigh = (eb_index + 1) / n
            eb_high = math.ceil((math.exp(slope * eb_xhigh) - 1) / (math.exp(slope) - 1) * vmax) - 1
            self.vrange = (eb_low, eb_high)
            self.v = sum(self.vrange) / 2
        elif "LINBIN" in enctype:
            if "ACCUMULATING" in enctype:
                lb_index = raw_nonnull.argmin() - 1
            else:
                lb_index = raw_nonnull.argmax()

            lb_low = math.ceil(lb_index * slope)
            lb_high = math.ceil((lb_index + 1) * slope) - 1
            self.vrange = (lb_low, lb_high)
            self.v = sum(self.vrange) / 2
        elif enctype.startswith("ACCUMULATING"):
            self.v = raw_nonnull.argmin() - 1
        elif enctype.startswith("BINARY"):
            self.v = raw_nonnull.astype(int)
        elif enctype.startswith("CATEGORICAL"):
            self.v = raw_nonnull.argmax()
        elif enctype.startswith("EXPNORM"):
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            self.v = round(vmax ** raw_nonnull[0]) - 1
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
