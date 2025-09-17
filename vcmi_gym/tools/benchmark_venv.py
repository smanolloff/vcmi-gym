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
import torch
import json
from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel
from rl.algos.mppo_dna_gnn.dual_vec_env import DualVecEnv


def main():
    def model_factory():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open("sfcjqcly-1757757007-config.json", "r") as f:
            cfg = json.load(f)
        weights = torch.load("sfcjqcly-1757757007-model-dna.pt", weights_only=True, map_location="cpu")
        model = DNAModel(cfg["model"], device).eval()
        model.load_state_dict(weights, strict=True)
        return model

    NUM_ENVS = 10

    venv_kwargs = dict(
        env_kwargs=dict(mapname="gym/generated/4096/4x1024.vmap"),
        num_envs_stupidai=0,
        num_envs_battleai=0,
        num_envs_model=NUM_ENVS,
        model_factory=model_factory,
        e_max=3300
    )

    print(venv_kwargs)
    print("")

    venv = DualVecEnv(**venv_kwargs)

    total_vsteps = 1000

    vsteps = 0
    resets = 0
    tmpvsteps = 0
    tmpresets = 0
    tstart = time.time()
    t0 = time.time()
    update_every = total_vsteps // 100  # 100 updates total (every 1%)
    report_every = 10  # every 10%

    try:
        while vsteps < total_vsteps:
            venv.step(venv.call("random_action"))

            # reset is also a "step" (aka. a round-trip to VCMI)
            vsteps += 1
            tmpvsteps += 1

            if vsteps % update_every == 0:
                percentage = (vsteps / total_vsteps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

                if vsteps % (update_every*report_every) == 0:
                    s = time.time() - t0
                    print(" steps/s: %-6.0f vsteps/s: %-6.0f resets/s: %-6.2f" % (tmpvsteps*NUM_ENVS/s, tmpvsteps/s, tmpresets/s))
                    tmpvsteps = 0
                    tmpresets = 0
                    t0 = time.time()
                    # avoid hiding the percent
                    if percentage < 100:
                        print("\r%d%%..." % percentage, end="", flush=True)

        s = time.time() - tstart
        print("")
        print("Average: steps/s: %-6.0f vsteps/s: %-6.0f resets/s: %-6.2f" % (vsteps*NUM_ENVS/s, vsteps/s, resets/s))
        print("")
        print("* Total time: %.2f seconds" % s)
        print("* Total steps: %d" % (vsteps*NUM_ENVS))
        print("* Total vsteps: %d" % vsteps)
        print("* Total resets: %d" % resets)
        print("* Total envs: %d" % NUM_ENVS)
    finally:
        venv.close()


if __name__ == "__main__":
    main()
