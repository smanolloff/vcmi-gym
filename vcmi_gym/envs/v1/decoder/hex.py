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
from ..pyconnector import (
    ATTRMAP,
    HEXACTMAP,
    HEXSTATEMAP,
    MELEEDISTMAP,
    SHOOTDISTMAP,
    DMGMODMAP,
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
)


class Hex(namedtuple("Hex", ATTRMAP.keys())):
    def __repr__(self):
        return f'Hex(y={self.HEX_Y_COORD} x={self.HEX_X_COORD})'

    def dump(self, compact=True):
        maxlen = 0
        lines = []
        for field in self._fields:
            value = getattr(self, field)
            maxlen = max(maxlen, len(field))

            if field.startswith("HEX_ACTION_MASK_FOR_"):
                names = list(HEXACTMAP.keys())
                indexes = np.where(value)[0]
                value = None if not any(indexes) else ", ".join([names[i] for i in indexes])
            elif field.startswith("HEX_STATE"):
                value = None if not value else list(HEXSTATEMAP.keys())[value]
            elif field.startswith("HEX_MELEE_DISTANCE_FROM"):
                value = None if not value else list(MELEEDISTMAP.keys())[value]
            elif field.startswith("HEX_SHOOT_DISTANCE_FROM"):
                value = None if not value else list(SHOOTDISTMAP.keys())[value]
            elif field.startswith("HEX_MELEEABLE_BY"):
                value = None if not value else list(DMGMODMAP.keys())[value]

            if value is None and compact:
                continue
            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def action(self, hexaction):
        if isinstance(hexaction, str):
            hexaction = HEXACTMAP.get(hexaction, None)

        if hexaction not in HEXACTMAP.values():
            return None

        if not self.HEX_ACTION_MASK_FOR_ACT_STACK[hexaction]:
            print("Action not possible for this hex")
            return None

        n = 15*self.HEX_Y_COORD + self.HEX_X_COORD
        return N_NONHEX_ACTIONS + n*N_HEX_ACTIONS + hexaction

    def actions(self):
        return [k for k, v in HexAction.__dict__.items() if self.HEX_ACTION_MASK_FOR_ACT_STACK[v]]


