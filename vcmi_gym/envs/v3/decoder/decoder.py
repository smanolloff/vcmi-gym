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
from .. import pyconnector


class Decoder:
    # declare as class var to allow overwriting by subclasses
    STATE_SIZE_ONE_HEX = pyconnector.STATE_SIZE_ONE_HEX

    @classmethod
    def decode(cls, obs):
        assert len(obs.shape) == 3
        assert obs.shape[0] == 11
        assert obs.shape[1] == 15
        assert obs.shape[2] == cls.STATE_SIZE_ONE_HEX

        res = Battlefield()
        for y in range(11):
            row = []
            for x in range(15):
                row.append(cls.decode_hex(obs[y][x]))
            res.append(row)

        return res

    @staticmethod
    def decode_hex(hexdata):
        res = {}

        for attr, (enctype, offset, n, vmax) in pyconnector.ATTRMAP.items():
            attrdata = hexdata[offset:][:n]

            if attrdata[0] == pyconnector.STATE_VALUE_NA:
                res[attr] = None
                continue

            match enctype:
                case "NUMERIC":
                    res[attr] = attrdata.argmin()
                case "NUMERIC_SQRT":
                    value_sqrt = attrdata.argmin()
                    value_min = round(value_sqrt ** 2)
                    value_max = round((value_sqrt+1) ** 2)
                    assert value_max <= vmax, f"internal error: {value_max} > {vmax}"
                    res[attr] = value_min, value_max
                case "BINARY":
                    bits = attrdata
                    res[attr] = bits.astype(int)
                case "CATEGORICAL":
                    res[attr] = attrdata.argmax()

                case "FLOATING":
                    assert n == 1, f"internal error: {n} != 1"
                    res[attr] = round(attrdata[0] * vmax)
                case "CATEGORICAL":
                    res[attr] = attrdata.argmax()
                case _:
                    raise Exception(f"Unexpected encoding type: {enctype}")

        return Hex(**res)
