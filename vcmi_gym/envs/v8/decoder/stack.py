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

from .. import pyprocconnector as pyconnector


class StackFlags(namedtuple("StackFlags", list(pyconnector.STACK_FLAG_MAP.keys()))):
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
            value = value.struct if value.struct else value.v
            maxlen = max(maxlen, len(field))

            if value is not None:
                match field:
                    case "SIDE":
                        value = self._side()

            elif compact:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def _side(self):
        if self.SIDE.v is None:
            return None
        return list(pyconnector.SIDE_MAP)[self.SIDE.v]

    def alias(self):
        return "?"
