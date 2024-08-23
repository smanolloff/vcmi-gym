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

import numpy as np
from .. import HexAction


class TestHelper:
    def __init__(self, env, auto_render=True):
        self.env = env
        self.term = False
        self.auto_render = auto_render
        self.env.reset()
        self.battlefield = self.env.decode()
        self.render()

    def render(self):
        print(self.env.render())

    def reset(self):
        self.env.reset()
        self.battlefield = self.env.decode()
        self.render()

    def _maybe_render(self, action):
        if action is None:
            return

        obs, self.rew, self.term, trunc, self.info = self.env.step(action)
        self.battlefield = self.env.decode()
        self.render()

    def wait(self):
        self._maybe_render(1)

    # x: 1..15
    # y: 1..11
    def move(self, y, x):
        a = self.battlefield.get_hex(y, x).action(HexAction.MOVE)
        self._maybe_render(a)

    def shoot(self, y, x):
        a = self.battlefield.get_hex(y, x).action(HexAction.SHOOT)
        self._maybe_render(a)

    def melee(self, y, x, direction):
        if isinstance(direction, str):
            amove = getattr(HexAction, f"AMOVE_{direction}")
        else:
            amove = direction

        a = self.battlefield.get_hex(y, x).action(amove)
        self._maybe_render(a)

    def amove(self, *args, **kwargs):
        return self.melee(*args, **kwargs)

    def defend(self):
        astack = None
        for stack in self.battlefield.stacks:
            if stack.QUEUE_POS == 0:
                astack = stack
                break

        if not astack:
            raise Exception("Could not find active stack")

        # Moving to self results in a defend action
        h = self.battlefield.get_hex(astack.Y_COORD, astack.X_COORD)
        self._maybe_render(h.action(HexAction.MOVE))

    def random(self):
        actions = np.where(self.env.action_mask())[0]
        if actions.any():
            return self._maybe_render(np.random.choice(actions))
        else:
            assert self.term, "action mask allows 0 actions, but last result was not terminal"
            print("Battle ended, re-starting...")
            self.reset()
