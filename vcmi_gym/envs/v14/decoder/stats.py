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

from ..pyconnector import (
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP
)


class GlobalStats(namedtuple("GlobalStats", ["data"] + list(GLOBAL_ATTR_MAP.keys()))):
    def __repr__(self):
        return "GlobalStats()"

    def dump(self, compact=True):
        maxlen = 0
        lines = []
        for field in self._fields:
            if field == "data":
                continue

            maxlen = max(maxlen, len(field))
            value = getattr(self, field)

            if compact and value.v is None:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if value is None else value) for (field, value) in lines]))


class PlayerStats(namedtuple("PlayerStats", ["data"] + list(PLAYER_ATTR_MAP.keys()))):
    def __repr__(self):
        return "PlayerStats()"

    def dump(self, compact=True):
        maxlen = 0
        lines = []
        for field in self._fields:
            if field == "data":
                continue

            maxlen = max(maxlen, len(field))
            value = getattr(self, field)

            if compact and value.v is None:
                continue

            lines.append((field, value))
        print("\n".join(["%s | %s" % (field.ljust(maxlen), "" if v is None else v) for (field, v) in lines]))
