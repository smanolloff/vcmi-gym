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

import time
import gymnasium as gym


def a(x, y=None, stack=0):
    if y is None:
        return x - 1
    return 3 + (y*15 + x)*8 + stack - 1


def test(env_kwargs, actions):
    env = gym.make("local/VCMI-v0")
    env.reset()
    print(env.render())
    actions = iter(actions)

    while True:
        action = next(actions) - 1
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        # logging.debug("======== obs: (hidden)")
        # logging.debug("======== rew: %s" % rew)
        # logging.debug("======== term: %s" % term)
        # logging.debug("======== trunc: %s" % trunc)
        # logging.debug("======== info: %s" % info)
        # action += 1

        if env.unwrapped.last_action_was_valid:
            time.sleep(0.2)
            print(env.render())
        else:
            pass
            # s = "Error summary:\n"
            # for i, name in enumerate(self.errnames):
            #     s += "%25s: %d\n" % (name, self.errcounters[i])
            # print(s)

        if term:
            break

    print(env.render())
