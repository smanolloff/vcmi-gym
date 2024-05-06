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

#
# XXX: this file is outdated and will not work
#      until it's updated to reflect the latest changes
#      in VcmiEnv's observation space
#

from .. import HexAction


class TestHelper:
    def __init__(self, env, auto_render=True):
        self.env = env
        self.auto_render = auto_render
        self.env.reset()
        self.battlefield = self.env.decode()
        self.env.render()

    def render(self):
        self.env.render()

    def reset(self):
        self.env.reset()
        self.battlefield = self.env.decode()
        self.env.render()

    def _maybe_render(self, action):
        if action is None:
            return

        self.env.step(action)
        self.battlefield = self.env.decode()
        self.env.render()

    def wait(self):
        self._maybe_render(0)

    # x: 1..15
    # y: 1..11
    def move(self, y, x):
        a = self.battlefield.get(y, x).action(HexAction.MOVE)
        self._maybe_render(a)

    def shoot(self, y, x):
        a = self.battlefield.get(y, x).action(HexAction.SHOOT)
        self._maybe_render(a)

    def melee(self, y, x, direction):
        if isinstance(direction, str):
            amove = getattr(HexAction, f"AMOVE_{direction}")
        else:
            amove = direction

        a = self.battlefield.get(y, x).action(amove)
        self._maybe_render(a)

    def amove(self, *args, **kwargs):
        return self.melee(*args, **kwargs)

    def defend(self):
        for row in self.battlefield:
            for x in row:
                if x.STACK_IS_ACTIVE:
                    if x.STACK_IS_WIDE and x.STACK_SIDE == 0:
                        x = self.battlefield.get(x.HEX_Y_COORD, x.HEX_X_COORD + 1)
                    self._maybe_render(x.action(HexAction.MOVE))
                    return
