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

from . import pyconnector
from ..v1.decoder.decoder import (
    Decoder as Decoder_v1,
    Hex
)


class Decoder(Decoder_v1):
    STATE_SIZE_ONE_HEX = pyconnector.STATE_SIZE_ONE_HEX

    @staticmethod
    def decode_hex(hexdata):
        res = {}

        for attr, (enctype, offset, n, vmax) in pyconnector.ATTRMAP.items():
            attrdata = hexdata[offset:][:n]

            if attrdata[0] == pyconnector.STATE_VALUE_NA:
                res[attr] = None
                continue

            if enctype != "FLOATING":
                raise Exception(f"Unexpected encoding type: {enctype}")

            assert n == 1, f"internal error: {n} != 1"
            v = round(attrdata[0] * vmax)

            if attr.startswith("HEX_ACTION_MASK_FOR_"):
                # binary data encoded as float
                bitstr = format(v, f'0{pyconnector.N_HEX_ACTIONS}b')
                res[attr] = np.array([int(b) for b in reversed(list(bitstr))])
            else:
                res[attr] = v

        return Hex(**res)
