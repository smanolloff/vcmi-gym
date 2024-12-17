from .. import pyprocconnector as pyconnector
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
        raw,
        raw0=None,
        precision=None,    # number of digits after "."
        roundfracs=None,  # 5 = round to nearest 0.2 (3.14 => 3.2)
        struct_cls=None,
        struct_mapping=None,
        verbose=False,
    ):
        self.name = name
        self.enctype = enctype
        self.n = n
        self.vmax = vmax
        self.struct_cls = struct_cls
        self.struct_mapping = struct_mapping
        self.verbose = verbose

        self.raw = raw
        self.raw0 = raw0
        self.v = None
        self.struct = None

        if precision is not None:
            self.raw_rounded = np.round(raw, precision)
        elif roundfracs is not None:
            self.raw_rounded = np.round(raw * roundfracs) / roundfracs
        else:  # no rounding
            self.raw_rounded = raw

        if enctype.endswith("EXPLICIT_NULL"):
            # if self.raw_rounded[0] == 1:
            if raw.argmax() == 0 and self.raw[0] > 0.2:
                self.v = None
                self._compare()
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
            self.v = raw_nonnull.argmin() - 1
        elif enctype.startswith("BINARY"):
            # similar to NULL values, these are 0 or 1 => use rounded
            self.v = raw_nonnull_rounded_.astype(int)
        elif enctype.startswith("CATEGORICAL"):
            self.v = raw_nonnull.argmax()
        elif enctype.startswith("EXPNORM"):
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            self.v = round(vmax ** raw_nonnull[0]) - 1
        elif enctype.startswith("LINNORM"):
            assert len(raw_nonnull) == 1, f"internal error: len(raw_nonnull): {len(raw_nonnull)} != 1"
            self.v = round(raw_nonnull[0] * vmax)

        self._compare()

        if struct_cls:
            assert struct_mapping
            assert isinstance(self.v, np.ndarray), type(self.v)
            # struct_mapping is a dict {"NAME": index}
            self.struct = struct_cls(**{k: int(self.v[v]) for k, v in struct_mapping.items()})

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _compare(self):
        if self.raw0 is None:
            return

        self.raw_re = Encoder.encode(self.enctype, self.v, self.n, self.vmax)

        v_kwargs = dict(
            name=self.name,
            enctype=self.enctype,
            n=self.n,
            vmax=self.vmax,
            raw0=None,
            struct_cls=self.struct_cls,
            struct_mapping=self.struct_mapping
        )

        self.v_re = self.__class__(**dict(v_kwargs, raw=self.raw_re)).v
        self.v0 = self.__class__(**dict(v_kwargs, raw=self.raw0)).v
        self.v_rounded = self.__class__(**dict(v_kwargs, raw=self.raw_rounded)).v

        # v0 may be an array or float
        # better compare raw
        equal = np.allclose(self.raw0, self.raw_re, atol=1e-3)

        if equal:
            self.log(f"Match: v={self.v}, v0={self.v0}, name={self.name}, enctype={self.enctype}")
            return

        self.log(f"Mismatch: {self.enctype}: {self.name}")

        # for printing purposes
        display_rounding = 2
        display_values = np.round(np.stack((self.raw, self.raw_rounded, self.raw_re, self.raw0)), display_rounding)
        linewidth = 5 + (4 + display_rounding) * self.raw.shape[0]
        # suppress = no scientific notation
        np.set_printoptions(suppress=True, linewidth=linewidth)
        # import ipdb; ipdb.set_trace()  # noqa

        lines = str(display_values).split("\n")
        assert len(lines) == 4
        lines[0] = lines[0].ljust(linewidth) + ("  # %11s # %s" % ("raw", self.v))
        lines[1] = lines[1].ljust(linewidth) + ("  # %11s # %s" % ("raw_rounded", self.v_rounded))
        lines[2] = lines[2].ljust(linewidth) + ("  # %11s # %s" % ("raw_re", self.v_re))
        lines[3] = lines[3].ljust(linewidth) + ("  # %11s # %s" % ("raw0 (orig)", self.v0))
        self.log("\n".join(lines))

        # If just one of the values is None => problem (will mess up rendering)
        v0v = [self.v0, self.v]
        assert all(x is None for x in v0v) or not any(x is None for x in v0v), "critical mismatch"

        if self.v is not None:
            err_abs = np.sum((self.v0 - self.v))
            err_rel = np.sum(np.abs(self.v0) / np.abs(self.v))
            self.log("Error: abs=%.2f => %.2f%%" % (err_abs, err_rel * 100))

        # treat CATEGORICAL mismatches as critical (?):
        # assert not self.enctype.startswith("CATEGORICAL"), "critical mismatch"


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
            eps = 1e-8
            if enctype.startswith("EXPNORM"):
                raw[0] = math.log((v + 1) or eps, vmax)
            elif enctype.startswith("LINNORM"):
                raw[0] = v / (vmax or eps)
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
