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
from .decoder.other import PrimaryAction


class TestHelper:
    def __init__(self, env, auto_render=True):
        self.env = env
        self.term = False
        self.auto_render = auto_render
        self.obs, self.info = self.env.reset()
        self.battlefield = self.env.decode()
        self.render()

    def render(self):
        print(self.env.render())

    def reset(self):
        self.env.reset()
        self.battlefield = self.env.decode()
        self.render()

    def _maybe_render(self, a1, a2=None):
        if a1 is None:
            return

        if not self.obs["action_mask_1"][a1]:
            print("Action not allowed")
            return

        if a2 is None:
            a2 = 0
        elif not self.obs["action_mask_2"][a1][a2]:
            print("Action not allowed")
            return

        action = {"action_1": a1, "action_2": a2 or 0}
        self.obs, self.rew, self.term, trunc, self.info = self.env.step(action)
        self.battlefield = self.env.decode()
        self.render()

    def wait(self):
        self._maybe_render(PrimaryAction.WAIT)

    # x: 1..15
    # y: 1..11
    def move(self, y, x):
        a1 = PrimaryAction.MOVE
        a2 = y*15 + x
        self._maybe_render(a1, a2)

    def shoot(self, stackid):
        # shooting does not require a hex
        a1 = PrimaryAction.ATTACK_0 + stackid
        self._maybe_render(a1)

    def amove(self, y, x, stackid):
        a1 = PrimaryAction.ATTACK_0 + stackid
        a2 = y*15 + x
        self._maybe_render(a1, a2)

    def defend(self):
        a = self.env._defend_action(self.obs["observation"])
        self._maybe_render(a["action_1"], a["action_2"])

    def random(self):
        actions1 = np.where(self.obs["action_mask_1"])[0]
        if actions1.any():
            a1 = np.random.choice(actions1)
            actions2 = np.where(self.obs["action_mask_2"][a1])[0]
            a2 = np.random.choice(actions2) if np.any(actions2) else 0
            return self._maybe_render(a1, a2)
        else:
            assert self.term, "action mask allows no actions, but last result was not terminal"
            print("Battle ended, re-starting...")
            self.reset()

    def help(self):
        print((
            "Help:"
            "\n\t.defend()"
            "\n\t.wait()"
            "\n\t.move(y, x)        - move to hex (y,x)"
            "\n\t.amove(y, x, e)    - move to hex (y,x) and attack enemy #e"
            "\n\t.shoot(e)          - shoot at enemy #e"
            "\n\t.random()          - take a random action"
        ))
