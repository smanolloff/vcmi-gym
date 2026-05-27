from typing import NamedTuple, Any
import numpy as np


class AttributeSpec(NamedTuple):
    name: str
    encoding: str
    size: int
    vmax: int | float | None = None

    @property
    def encoded_size(self) -> int:
        if self.encoding == "CATEGORICAL":
            return self.size

        if self.encoding in {"LINNORM", "RAW"}:
            return 1

        raise ValueError(f"Unknown encoding: {self.encoding}")

    def encode(self, v: int | float) -> list[float]:
        if self.encoding == "CATEGORICAL":
            out = [0.0] * self.size
            out[int(v)] = 1.0
            return out

        if self.encoding == "LINNORM":
            if self.vmax in (None, 0):
                raise ValueError(f"Invalid vmax for LINNORM field {self.name}")
            return [float(v) / float(self.vmax)]

        if self.encoding == "RAW":
            return [float(v)]

        raise ValueError(f"Unknown encoding: {self.encoding}")

    def decode(self, xs: np.ndarray) -> int | float:
        if self.encoding == "CATEGORICAL":
            return int(np.argmax(xs))

        if self.encoding == "LINNORM":
            if self.vmax is None:
                raise ValueError(f"Missing vmax for LINNORM field {self.name}")

            return int(round(float(xs[0]) * float(self.vmax)))

        if self.encoding == "RAW":
            value = float(xs[0])

            # Return int when the value is exactly integer-like.
            if value.is_integer():
                return int(value)

            return value

        raise ValueError(f"Unknown encoding: {self.encoding}")


def make_node_namedtuple_class(class_name: str, node_type_dict: dict[str, Any]):
    specs = [
        AttributeSpec(
            name=attr["name"],
            encoding=attr["encoding"],
            size=attr["size"],
            vmax=attr.get("vmax"),
        )
        for attr in node_type_dict["attributes"]
    ]

    fields = [(spec.name, int | float) for spec in specs]

    cls = NamedTuple(class_name, fields)

    cls.__attribute_specs__ = specs
    cls.__encoded_size__ = sum(spec.encoded_size for spec in specs)

    @classmethod
    def decode_one(cls, row: np.ndarray):
        row = np.asarray(row)

        if row.ndim != 1:
            raise ValueError(f"Expected 1D row, got shape {row.shape}")

        if row.shape[0] != cls.__encoded_size__:
            raise ValueError(
                f"Expected encoded size {cls.__encoded_size__}, "
                f"got {row.shape[0]}"
            )

        values = {}
        offset = 0

        for spec in cls.__attribute_specs__:
            width = spec.encoded_size
            chunk = row[offset : offset + width]
            values[spec.name] = spec.decode(chunk)
            offset += width

        return cls(**values)

    @classmethod
    def decode_many(cls, arr: np.ndarray):
        arr = np.asarray(arr)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (N, D), got {arr.shape}")

        if arr.shape[1] != cls.__encoded_size__:
            raise ValueError(
                f"Expected encoded size {cls.__encoded_size__}, "
                f"got {arr.shape[1]}"
            )

        return [cls.decode_one(row) for row in arr]

    @classmethod
    def decode(cls, row_or_arr: np.ndarray):
        arr = np.asarray(row_or_arr)

        if arr.ndim == 1:
            return cls.decode_one(row_or_arr)
        else:
            return cls.decode_many(row_or_arr)

    cls.decode_one = decode_one
    cls.decode_many = decode_many
    cls.decode = decode

    return cls

def dump(namedtuples):
    if not isinstance(namedtuples, list):
        namedtuples = [namedtuples]

    for nt in namedtuples:
        if not hasattr(nt, "_fields"):
            raise TypeError(f"Expected NamedTuple-like object, got {type(nt).__name__}")

    field_width = max(len(field) for nt in namedtuples for field in nt._fields)

    for i, nt in enumerate(namedtuples):
        title = f"[{i}] {nt.__class__.__name__}"
        header_width = max(1, field_width - len(title) - 1)
        print(f"{'-' * header_width} {title} |-----")
        for field in nt._fields:
            print(f"{field.ljust(field_width)} | {getattr(nt, field)}")
