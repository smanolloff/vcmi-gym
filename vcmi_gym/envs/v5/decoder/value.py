from .. import pyprocconnector as pyconnector
import numpy as np

NA = pyconnector.STATE_VALUE_NA


class Value:
    def __repr__(self):
        if self.v is None:
            return "Value(null)"
        elif self.struct:
            return f"Value(struct={self.struct})"
        else:
            return f"Value(v={self.v})"

    def __init__(self, name, enctype, n, vmax, raw, struct_cls=None, struct_mapping=None):
        self._name = name
        self._enctype = enctype
        self._n = n
        self._vmax = vmax
        self._name_mapping = struct_mapping

        self.raw = raw
        self.v = None
        self.struct = None

        if enctype.endswith("EXPLICIT_NULL"):
            if raw[0] == 1:
                return
            raw = raw[1:]

        if enctype.endswith("IMPLICIT_NULL") and not any(raw):
            return

        if enctype.endswith("MASKING_NULL") and raw[0] == NA:
            return

        if enctype.startswith("ACCUMULATING"):
            self.v = raw.argmin() - 1
        elif enctype.startswith("BINARY"):
            self.v = raw.astype(int)
        elif enctype.startswith("CATEGORICAL"):
            self.v = raw.argmax()
        elif enctype.startswith("NORMALIZED"):
            assert len(raw) == 1, f"internal error: len(raw): {len(raw)} != 1"
            self.v = round(raw[0] * vmax)

        reencoded = Encoder.encode(enctype, self.v, n, vmax)
        assert all(np.raw == reencoded), f"all({raw} == {reencoded})"

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
                raw = np.unpackbits(np.array([v], dtype=np.uint8))[-nn:].astype(float)
            elif enctype.startswith("ACCUMULATING"):
                raw[:v+1] = 1
            else:
                raise Exception(f"Unexpected enctype: {enctype}")

            if enctype.endswith("EXPLICIT_NULL"):
                raw = np.insert(raw, 0, 0)

        return raw
