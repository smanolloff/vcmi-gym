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

import os
import time
import torch
import json
import vcmi_gym

from torch_geometric.data import Batch
from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel, to_hdata_list


def get_action(env, obs, model):
    if model is None:
        return env.random_action()

    hdata = Batch.from_data_list(to_hdata_list(
        torch.as_tensor(obs["observation"]).unsqueeze(0),
        torch.tensor([False]),
        [obs["links"]]
    ))

    return model.model_policy.get_actdata_eval(hdata).action


def main():
    oldcwd = os.getcwd()
    total_steps = 1000
    env = vcmi_gym.VcmiEnv_v13(
        #"gym/generated/4096/4096-6stack-100K-01.vmap",
        "gym/generated/4096/4x1024.vmap",
        # "gym/A1.vmap",
        random_heroes=0,
        random_obstacles=0,
        warmachine_chance=0,
        town_chance=0,
        opponent="BattleAI",
        max_steps=1000,
        # swap_sides=1,
    )
    print("VCMI ver: %d" % env.ENV_VERSION)
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
    print("* Player: %s" % ew.role)
    print("* Opponent: %s %s" % (ew.opponent, ew.opponent_model if ew.opponent == "MMAI_MODEL" else ""))
    print("* Total steps: %d" % total_steps)
    print("")

    was_term = False
    termside = -1
    # two_users = ew.attacker == "MMAI_USER" and ew.defender == "MMAI_USER"
    two_users = False

    # No model => get_action() will pick a random valid action
    model = None

    # Normal torch model
    # with open(f"{oldcwd}/sfcjqcly-1757757007-config.json", "r") as f:
    #     cfg = json.load(f)
    # weights = torch.load(f"{oldcwd}/sfcjqcly-1757757007-model-dna.pt", weights_only=True, map_location="cpu")
    # model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    # model.load_state_dict(weights, strict=True)

    # model = torch.load("rl/models/Attacker model:v7/agent.pt", weights_only=False)

    # JIT torch model (~5% faster)
    #model = torch.jit.load("rl/models/Attacker model:v7/jit-agent.pt")

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
                act = get_action(env, obs, model)
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
