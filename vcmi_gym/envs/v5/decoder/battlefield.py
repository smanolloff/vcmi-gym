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

class Battlefield():
    def __repr__(self):
        return "Battlefield(11x15)"

    def __init__(self):
        self.hexes = []
        self.stacks = [[], []]  # left, right stacks

    def get_hex(self, y_or_n, x=None):
        if x is not None:
            y = y_or_n
        else:
            y = y_or_n // 15
            x = y_or_n % 15

        if y >= 0 and y < len(self.hexes) and x >= 0 and x < len(self.hexes[y]):
            return self.hexes[y][x]
        else:
            print("Invalid hex (y=%s x=%s)" % (y, x))

    def get_lstack(self, i):
        return self._get_stack(0, i)

    def get_rstack(self, i):
        return self._get_stack(1, i)

    def _get_stack(self, side, i):
        if i >= 0 and i < len(self.stacks[side]):
            return self.stacks[side][i]
        else:
            print("Invalid stack (ID=%s)" % i)
