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
import numpy as np
import torch

from vcmi_gym import VcmiEnv


def get_action(model, obs, mask):
    if model is None:
        return np.random.choice(np.where(mask)[0])

    #if not hasattr(model.args, "envmaps"):
    # old scheme (without PERCENT_CUR_TO_START_TOTAL_VALUE)
    obs = obs[:, :, 1:]

    return model.predict(
        torch.as_tensor(obs).float(),
        torch.as_tensor(np.array(mask))
    )


def main():
    total_steps = 1000
    env = VcmiEnv(
        "gym/generated/4096/4096-6stack-100K-01.vmap",
        encoding_type="float",
        random_heroes=1,
        random_obstacles=1,
        # swap_sides=1,
        # defender="MMAI_USER",
        # defender_model="/Users/simo/Projects/vcmi-gym/data/PBT-mppo-obstacle+sideswap-20240512_230506/ee609_00000/checkpoint_000020/agent.pt"
    )
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

    # No model => get_action() will pick a random valid action
    #model = None

    # Normal torch model
    # model = torch.load("rl/models/Attacker model:v2/agent.pt")

    # JIT torch model (~5% faster)
    model = torch.jit.load("rl/models/Attacker model:v2/jit-agent.pt")

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
                act = get_action(model, obs, ew.action_mask())
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
