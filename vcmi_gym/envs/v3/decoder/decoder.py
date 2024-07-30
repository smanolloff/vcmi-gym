# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from .battlefield import Battlefield
from .hex import Hex
from .stack import Stack
from .. import pyconnector

NA = pyconnector.STATE_VALUE_NA


class Decoder:
    # declare as class var to allow overwriting by subclasses
    STATE_SIZE_ONE_HEX = pyconnector.STATE_SIZE_ONE_HEX

    @classmethod
    def decode(cls, obs):
        assert obs.shape == (pyconnector.STATE_SIZE,)

        stacks = obs[:pyconnector.STATE_SIZE_STACKS].reshape(20, pyconnector.STATE_SIZE_ONE_STACK)
        hexes = obs[pyconnector.STATE_SIZE_STACKS:].reshape(11, 15, pyconnector.STATE_SIZE_ONE_HEX)

        res = Battlefield()

        for i in range(20):
            res.stacks.append(cls.decode_stack(stacks[i]))

        for y in range(11):
            row = []
            for x in range(15):
                row.append(cls.decode_hex(hexes[y][x]))
            res.hexes.append(row)

        return res

    @classmethod
    def decode_stack(cls, stackdata):
        res = {}
        for attr, (enctype, offset, n, vmax) in pyconnector.STACKATTRMAP.items():
            attrdata = stackdata[offset:][:n]
            res[attr] = cls.decode_attribute(attrdata, enctype, vmax)
        return Stack(**res)

    @classmethod
    def decode_hex(cls, hexdata):
        res = {}
        for attr, (enctype, offset, n, vmax) in pyconnector.HEXATTRMAP.items():
            attrdata = hexdata[offset:][:n]
            res[attr] = cls.decode_attribute(attrdata, enctype, vmax)
        return Hex(**res)

    @classmethod
    def decode_attribute(cls, data, enctype, vmax):
        if enctype.endswith("EXPLICIT_NULL"):
            if data[0] == 1:
                return None
            data = data[1:]

        if enctype.endswith("IMPLICIT_NULL") and not any(data):
            return None

        if enctype.endswith("MASKING_NULL") and data[0] == NA:
            return None

        if enctype.startswith("ACCUMULATING"):
            return data.argmin() - 1
        elif enctype.startswith("BINARY"):
            return data.astype(int)
        elif enctype.startswith("CATEGORICAL"):
            return data.argmax()
        elif enctype.startswith("NORMALIZED"):
            assert len(data) == 1, f"internal error: len(data): {len(data)} != 1"
            return round(data[0] * vmax)

        raise Exception(f"Unexpected encoding type: {enctype}")
