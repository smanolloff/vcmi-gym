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

from ..pyconnector import STACKATTRMAP, SIDEMAP


class Stack(namedtuple("Stack", ["data"] + list(STACKATTRMAP.keys()))):
    def __repr__(self):
        return f'Stack(id={self.ID} side={self._side()} y={self.Y_COORD} x={self.X_COORD})'

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
                    case "SIDE":
                        value = self._side()
            elif compact:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))

    def _side(self):
        if self.SIDE is None:
            return None
        return list(SIDEMAP)[self.SIDE]
