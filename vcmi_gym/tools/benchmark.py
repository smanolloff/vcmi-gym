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
import random

from vcmi_gym import VcmiEnv


def random_valid_action(mask):
    valid_action_indexes = [i for i, v in enumerate(mask) if v]
    return random.choice(valid_action_indexes)


def first_valid_action(mask):
    for (i, m) in enumerate(mask):
        if m:
            return i


def main():
    total_steps = 1000
    env = VcmiEnv("gym/generated/88/88-7stack-01.vmap", random_combat=1, encoding_type="float")
    obs, info = env.reset()
    term = False
    trunc = False
    steps = 0
    resets = 0
    tmpsteps = 0
    tmpresets = 0
    tstart = time.time()
    t0 = time.time()
    benchside = info["side"]
    ew = env.unwrapped
    update_every = total_steps // 100  # 100 updates total (every 1%)
    report_every = 10  # every 10%

    assert total_steps % 10 == 0

    print("* Map: %s" % ew.mapname)
    print("* Attacker: %s %s" % (ew.attacker, ew.attacker_model if ew.attacker == "MMAI_MODEL" else ""))
    print("* Defender: %s %s" % (ew.defender, ew.defender_model if ew.defender == "MMAI_MODEL" else ""))
    print("* Total steps: %d" % total_steps)
    print("")

    was_term = False
    termside = -1
    two_users = ew.attacker == "MMAI_USER" and ew.defender == "MMAI_USER"

    try:
        while steps < total_steps:
            if term or trunc:
                assert not was_term
                was_term = two_users
                if info["side"] == benchside:
                    resets += 1
                    tmpresets += 1

                obs, info = env.reset()
                term = False
                trunc = False
            elif was_term and two_users:
                raise Exception("asdasd")
                # means we just processed 1st env's terminal obs, now 2nd
                was_term = False
                assert termside != info["side"]
                if info["side"] == benchside:
                    resets += 1
                    tmpresets += 1
                obs, info = env.reset()
            else:
                act = random_valid_action(ew.action_mask())
                obs, _, term, trunc, info = env.step(act)

            # reset is also a "step" (aka. a round-trip to VCMI)
            steps += 1
            tmpsteps += 1

            if steps % update_every == 0:
                percentage = (steps / total_steps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

                if steps % (update_every*report_every) == 0:
                    s = time.time() - t0
                    print(" steps/s: %-6.0f resets/s: %-6.2f" % (tmpsteps/s, tmpresets/s))
                    tmpsteps = 0
                    tmpresets = 0
                    t0 = time.time()
                    # avoid hiding the percent
                    if percentage < 100:
                        print("\r%d%%..." % percentage, end="", flush=True)

        s = time.time() - tstart
        print("Average: steps/s: %-6.0f resets/s: %-6.2f" % (steps/s, resets/s))
        print("")
        print("* Total time: %.2f seconds" % s)
        print("* Total steps: %d" % steps)
        print("* Total resets: %d" % resets)
    finally:
        env.close()


if __name__ == "__main__":
    main()
