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

from ..pyprocconnector import (
    HEX_ATTR_MAP,
    HEX_ACT_MAP,
    HEX_STATE_MAP,
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
)


class HexStateFlags(namedtuple("HexStateFlags", list(HEX_STATE_MAP.keys()))):
    def __repr__(self):
        return "{%s}" % ", ".join([f for f in self._fields if getattr(self, f)])


class HexActionFlags(namedtuple("HexActionFlags", list(HEX_ACT_MAP.keys()))):
    def __repr__(self):
        return "{%s}" % ", ".join([f for f in self._fields if getattr(self, f)])


class Hex(namedtuple("Hex", ["data"] + list(HEX_ATTR_MAP.keys()))):
    def __repr__(self):
        return f'Hex(y={self.Y_COORD.v} x={self.X_COORD.v})'

    def dump(self, compact=True):
        maxlen = 0
        lines = []
        for field in self._fields:
            if field == "data":
                continue

            value = getattr(self, field)
            value = value.struct if value.struct else value.v
            maxlen = max(maxlen, len(field))

            if value is not None:
                pass
            elif compact:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def action(self, hexaction):
        if isinstance(hexaction, str):
            hexaction = HEX_ACT_MAP.get(hexaction, None)

        if hexaction not in HEX_ACT_MAP.values():
            return None

        if not self.ACTION_MASK.v[hexaction]:
            import ipdb; ipdb.set_trace()  # noqa
            print("Action not possible for this hex")
            return None

        hexid = 15*self.Y_COORD.v + self.X_COORD.v
        return N_NONHEX_ACTIONS + hexid*N_HEX_ACTIONS + hexaction

    def actions(self):
        return [k for k, v in HEX_ACT_MAP.items() if self.ACTION_MASK.v[v]]
