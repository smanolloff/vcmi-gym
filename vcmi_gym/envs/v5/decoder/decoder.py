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

import numpy as np

from .battlefield import Battlefield
from .value import Value
from .hex import Hex
from .stack import Stack, StackFlags
from .. import pyprocconnector as pyconnector

NA = pyconnector.STATE_VALUE_NA


class Decoder:
    # declare as class var to allow overwriting by subclasses
    STATE_SIZE_ONE_HEX = pyconnector.STATE_SIZE_ONE_HEX

    # XXX: `state0` is used in autoencoder testing scenarios where it
    #      represents the real state, while `state` is the reconstructed one
    @classmethod
    def decode(cls, state, is_battle_over, state0=None):
        obs = state
        obs0 = state0
        assert obs.shape == (pyconnector.STATE_SIZE,), f"{obs.shape} == ({pyconnector.STATE_SIZE},)"

        sizes = [
            pyconnector.STATE_SIZE_MISC,
            pyconnector.STATE_SIZE_ONE_STACK * 20,
            pyconnector.STATE_SIZE_ONE_HEX * 11 * 15
        ]

        assert sum(sizes) == pyconnector.STATE_SIZE, f"{sum(sizes)} == {pyconnector.STATE_SIZE}"

        # calculate the indexes of the delimiters from the sizes
        delimiters = np.cumsum(sizes)[:-1]

        misc, stacks, hexes = np.split(obs, delimiters)
        stacks = stacks.reshape(2, 10, pyconnector.STATE_SIZE_ONE_STACK)
        hexes = hexes.reshape(11, 15, pyconnector.STATE_SIZE_ONE_HEX)

        if obs0 is not None:
            misc0, stacks0, hexes0 = np.split(obs0, delimiters)
            stacks0 = stacks0.reshape(2, 10, pyconnector.STATE_SIZE_ONE_STACK)
            hexes0 = hexes0.reshape(11, 15, pyconnector.STATE_SIZE_ONE_HEX)

        res = Battlefield(is_battle_over, envstate=state)

        for side in range(2):
            for i in range(10):
                stackdata = stacks[side][i]
                stackdata0 = stacks0[side][i] if obs0 is not None else None
                res.stacks[side].append(cls.decode_stack(stackdata, stackdata0))

        for y in range(11):
            row = []
            for x in range(15):
                hexdata = hexes[y][x]
                hexdata0 = hexes0[y][x] if obs0 is not None else None
                row.append(cls.decode_hex(hexdata, hexdata0))
            res.hexes.append(row)

        return res

    @classmethod
    def decode_stack(cls, stackdata, stackdata0=None):
        res = {}
        for attr, (enctype, offset, n, vmax) in pyconnector.STACK_ATTR_MAP.items():
            attrdata = stackdata[offset:][:n]
            attrdata0 = stackdata0[offset:][:n] if stackdata0 is not None else None

            if attr == "FLAGS":
                res[attr] = Value(attr, enctype, n, vmax, attrdata, raw0=attrdata0, struct_cls=StackFlags, struct_mapping=pyconnector.STACK_FLAG_MAP)
            else:
                res[attr] = Value(attr, enctype, n, vmax, attrdata, raw0=attrdata0)

        return Stack(**dict(res, data=stackdata))

    @classmethod
    def decode_hex(cls, hexdata, hexdata0=None):
        res = {}
        for attr, (enctype, offset, n, vmax) in pyconnector.HEX_ATTR_MAP.items():
            attrdata = hexdata[offset:][:n]
            attrdata0 = hexdata0[offset:][:n] if hexdata0 is not None else None
            res[attr] = Value(attr, enctype, n, vmax, attrdata, raw0=attrdata0)
        return Hex(**dict(res, data=hexdata))

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
