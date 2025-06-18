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
import os
import numpy as np
from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper
from functools import partial
import gymnasium as gym
from types import SimpleNamespace


def main():
    total_steps = 1000
    env_kwargs = dict(
        # "gym/generated/4096/4096-6stack-100K-01.vmap",
        mapname="gym/generated/4096/4x1024.vmap",
        role="defender",
        random_heroes=1,
        random_obstacles=100,
        warmachine_chance=40,
        town_chance=10,
        opponent="StupidAI",
        max_steps=1000,
        # swap_sides=1,
    )

    pid = os.getpid()
    dummy_env = SimpleNamespace(
        metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
        render_mode='ansi',
        action_space=VcmiEnv.ACTION_SPACE,
        observation_space=VcmiEnv.OBSERVATION_SPACE["observation"],
        close=lambda: None
    )

    def env_creator(vec_cls):
        if os.getpid() == pid and vec_cls == gym.vector.AsyncVectorEnv:
            return dummy_env

        env = VcmiEnv(**env_kwargs)
        env = LegacyObservationSpaceWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    num_envs = 2
    # vec_cls = gym.vector.SyncVectorEnv
    vec_cls = gym.vector.AsyncVectorEnv
    funcs = [partial(env_creator, vec_cls) for _ in range(num_envs)]
    venv = vec_cls(funcs)

    venv.reset()
    steps = 0
    tmpsteps = 0
    resets = 0
    tmpresets = 0
    tstart = time.time()
    t0 = time.time()
    update_every = total_steps // 100  # 100 updates total (every 1%)
    report_every = 10  # every 10%

    assert total_steps % 10 == 0

    print("* N_ENVS: %d" % num_envs)
    print("* Map: %s" % env_kwargs["mapname"])
    print("* Player: %s" % env_kwargs["role"])
    print("* Opponent: %s %s" % (env_kwargs["opponent"], env_kwargs["opponent_model"] if env_kwargs["opponent"] == "MMAI_MODEL" else ""))
    print("* Total steps: %d" % total_steps)
    print("")

    while steps < total_steps:
        vact = venv.call("random_action")
        _obs, _rews, terms, truncs, infos = venv.step(vact)

        # reset is also a "step" (aka. a round-trip to VCMI)
        steps += 1
        tmpsteps += 1

        ndones = len(np.logical_or(terms, truncs).nonzero()[0])
        resets += ndones
        tmpresets += ndones

        if steps % update_every == 0:
            percentage = (steps / total_steps) * 100
            print("\r%d%%..." % percentage, end="", flush=True)

            if steps % (update_every*report_every) == 0:
                s = time.time() - t0
                print(" vsteps/s: %-6.0f steps/s: %-6.0f resets/s: %-6.2f" % (tmpsteps/s, (num_envs*tmpsteps)/s, tmpresets/s))
                tmpsteps = 0
                t0 = time.time()
                # avoid hiding the percent
                if percentage < 100:
                    print("\r%d%%..." % percentage, end="", flush=True)

    s = time.time() - tstart
    print("")
    print("* Total time: %.2f seconds" % s)
    print("* Total steps: %d" % steps)
    print("* Total resets: %d" % resets)
    print("")
    print("Average: vsteps/s: %-6.0f" % (steps/s))
    print("Average: steps/s: %-6.0f resets/s: %-6.2f" % ((num_envs*steps)/s, resets/s))


if __name__ == "__main__":
    main()
