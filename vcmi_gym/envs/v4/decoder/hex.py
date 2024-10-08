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

from collections import namedtuple
import numpy as np

from .other import (HexAction)
from ..pyprocconnector import (
    HEXATTRMAP,
    HEXACTMAP,
    HEXSTATEMAP,
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
)


class Hex(namedtuple("Hex", ["data"] + list(HEXATTRMAP.keys()))):
    def __repr__(self):
        return f'Hex(y={self.Y_COORD} x={self.X_COORD})'

    def dump(self, compact=True):
        maxlen = 0
        lines = []
        for field in self._fields:
            if field == "data":
                continue

            value = getattr(self, field)
            maxlen = max(maxlen, len(field))

            if value is not None:
                match field:
                    case "STATE_MASK":
                        names = list(HEXSTATEMAP.keys())
                        indexes = np.where(value)[0]
                        value = ", ".join([names[i] for i in indexes])
                    case "ACTION_MASK":
                        names = list(HEXACTMAP.keys())
                        indexes = np.where(value)[0]
                        value = ", ".join([names[i] for i in indexes])
            elif compact:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def action(self, hexaction):
        if isinstance(hexaction, str):
            hexaction = HEXACTMAP.get(hexaction, None)

        if hexaction not in HEXACTMAP.values():
            return None

        if not self.ACTION_MASK[hexaction]:
            print("Action not possible for this hex")
            return None

        n = 15*self.Y_COORD + self.X_COORD
        return N_NONHEX_ACTIONS + n*N_HEX_ACTIONS + hexaction

    def actions(self):
        return [k for k, v in HexAction.__dict__.items() if self.ACTION_MASK[v]]
