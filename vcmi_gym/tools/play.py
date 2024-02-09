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

import gymnasium as gym
import re
from .. import TestHelper


def user_action(th):
    hlp = """
Actions:
    r           render
    w           wait
    x y         move to x,y (x=1..15, y=1..11)
    x y 1..6    move to x,y + ATTACK at direction (TL/TR/R/BR/BL/L)
    x y 7..9    move to x,y + BACK-ATTACK at direction (BL/L/TL) (RED only)
    x y 10..12  move to x,y + BACK-ATTACK at direction (TR/R/BR) (BLUE only)
    x y 0       shoot at x,y
    h           show this help
"""

    while True:
        words = input("Choose an action ['h'=help]: ").split()

        if len(words) == 0:
            pass
        elif words[0] == "h":
            print(hlp)
            continue
        elif words[0] == "r" and len(words) == 1:
            th.render()
            continue
        elif words[0] == "w" and len(words) == 1:
            return th.wait()
        elif words[0] == "d" and len(words) == 1:
            return th.defend()
            continue
        elif len(words) == 2 \
                and re.match(r"^([1-9]|(1[0-5]))$", words[0]) \
                and re.match(r"^([1-9]|(1[01]))$", words[1]):
            return th.move(int(words[0]), int(words[1]))
        elif len(words) == 3 \
                and re.match(r"^([1-9]|(1[0-5]))$", words[0]) \
                and re.match(r"^([1-9]|(1[01]))$", words[1]) \
                and re.match(r"^[0-8]$", words[2]):
            if words[2] == "0":
                return th.shoot(int(words[0]), int(words[1]))
            else:
                return th.melee(int(words[0]), int(words[1]), int(words[2]))

        print("Invalid action.")
        print(hlp)


def play():
    env = gym.make("local/VCMI-v0")
    env.reset()
    th = TestHelper(env, auto_render=False)

    while True:
        obs, rew, term, trunc, info = user_action(th)
        print(env.render())
        if term:
            break
