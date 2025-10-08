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

from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel, eval_model
from rl.algos.mppo_dna_gnn.dual_vec_env import DualVecEnv


class DummyLogger:
    def __init__(self, print_every):
        self.print_every = print_every
        self.n = 0

    def debug(self, obj):
        if self.n % self.print_every == 0:
            print(obj)
        self.n += 1


def load_model(prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        with open(f"{prefix}-config.json", "r") as f:
            cfg = json.load(f)
        weights = torch.load(f"{prefix}-model-dna.pt", weights_only=True, map_location=device)
        model = DNAModel(cfg["model"], device).eval()
        model.load_state_dict(weights, strict=True)
    return model, cfg["train"]["env"]["kwargs"]["role"]


def main():
    #MODEL_PREFIX = "tukbajrv-202509222128"
    MODEL_PREFIX = "tukbajrv-1758611103"
    NUM_ENVS = 5
    NUM_VSTEPS = 500

    print("NUM_VSTEPS: %d" % NUM_VSTEPS)
    with torch.inference_mode():
        model, role = load_model(MODEL_PREFIX)
        tstart = time.time()
        stats = eval_model(
            num_vsteps=NUM_VSTEPS,
            model=model,
            logger=DummyLogger(print_every=NUM_VSTEPS // 10),
            venv=DualVecEnv(
                num_envs_battleai=NUM_ENVS,
                env_kwargs=dict(
                    mapname="gym/generated/4096/4x1024.vmap",
                    role=role,
                    random_heroes=1,
                    random_obstacles=1,
                    warmachine_chance=40,
                    tight_formation_chance=20,
                    random_terrain_chance=100,
                )
            )
        )
        tend = time.time()
        s = tend - tstart

    print("")
    print("Stats:")
    print("")
    print("ep_rew_mean: %.2f" % stats.ep_rew_mean)
    print("ep_len_mean: %.2f" % stats.ep_len_mean)
    print("ep_value_mean: %.2f" % stats.ep_value_mean)
    print("ep_is_success_mean: %.2f" % stats.ep_is_success_mean)
    print("num_episodes: %d" % stats.num_episodes)
    print("")
    print("Average: steps/s: %-6.0f vsteps/s: %-6.0f episodes/s: %-6.2f" % (NUM_VSTEPS*NUM_ENVS/s, NUM_VSTEPS/s, stats.num_episodes/s))
    print("")
    print("* Total time: %.2f seconds" % s)
    print("* Total steps: %d" % (NUM_VSTEPS*NUM_ENVS))
    print("* Total vsteps: %d" % NUM_VSTEPS)
    print("* Total envs: %d" % NUM_ENVS)


if __name__ == "__main__":
    main()
