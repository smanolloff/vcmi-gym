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

from ..pyprocconnector import (
    HEX_ATTR_MAP,
    HEX_STATE_MAP,
    PRIMARY_ACTION_MAP,
    AMOVE_ACTION_MAP,
)


class Hex(namedtuple("Hex", ["data"] + list(HEX_ATTR_MAP.keys()))):
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
                        names = list(HEX_STATE_MAP.keys())
                        indexes = np.where(value)[0]
                        value = ", ".join([names[i] for i in indexes])
                    case "ACTION_MASK":
                        names = list(AMOVE_ACTION_MAP.keys())
                        indexes = np.where(value)[0]
                        value = ", ".join([names[i] for i in indexes])
            elif compact:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def action(self, amoveaction):
        if isinstance(amoveaction, str):
            amoveaction = AMOVE_ACTION_MAP.get(amoveaction, None)

        if amoveaction not in AMOVE_ACTION_MAP.values():
            return None

        if not self.ACTION_MASK[amoveaction]:
            print("Action not possible for this hex")
            return None

        hexid = 15*self.Y_COORD + self.X_COORD
        offset = len(PRIMARY_ACTION_MAP) - len(AMOVE_ACTION_MAP)
        return {"action_1": offset + amoveaction, "action_2": hexid}

    def actions(self):
        return [k for k, v in AMOVE_ACTION_MAP.items() if self.ACTION_MASK[v]]
