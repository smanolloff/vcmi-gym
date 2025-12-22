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

from .. import pyconnector


class StackFlags1(namedtuple("StackFlags1", list(pyconnector.STACK_FLAG1_MAP.keys()))):
    def __repr__(self):
        return "{%s}" % ", ".join([f for f in self._fields if getattr(self, f)])


class StackFlags2(namedtuple("StackFlags2", list(pyconnector.STACK_FLAG2_MAP.keys()))):
    def __repr__(self):
        return "{%s}" % ", ".join([f for f in self._fields if getattr(self, f)])


class Stack(namedtuple("Stack", ["hex"] + list([k.removeprefix("STACK_") for k in pyconnector.HEX_ATTR_MAP.keys() if k.startswith("STACK_")]))):
    def __repr__(self):
        desc = f'side={self._side()}'
        if self.hex:
            desc += f' {self.hex}'
        return f"Stack({desc})"

    def dump(self, compact=True):
        maxlen = 0
        lines = []

        for field in self._fields:
            if field in ["hex"]:
                continue

            value = getattr(self, field)

            if value.struct:
                v = value.struct
            elif value.vrange:
                if value.vrange[0] == value.vrange[1]:
                    v = value.vrange[0]
                else:
                    v = "%s…%s" % value.vrange
            else:
                v = value.v

            maxlen = max(maxlen, len(field))

            if value is not None:
                match field:
                    case "SIDE":
                        v = self._side()
            elif compact:
                continue

            lines.append((field, v))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if v is None else v) for (field, v) in lines]))

    def _side(self):
        if self.SIDE.v is None:
            return None
        return list(pyconnector.SIDE_MAP)[self.SIDE.v]

    def alias(self):
        if self.SLOT.v == 7:
            return "M"
        elif self.SLOT.v == 8:
            return "S"
        else:
            assert self.SLOT.v in range(7), self.SLOT.v
            return str(self.SLOT.v)
