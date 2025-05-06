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
# import numpy as np
import torch
import random
import vcmi_gym
import torch
import logging

from vcmi_gym.envs.v12.pyconnector import HEX_ACT_MAP, N_HEX_ACTIONS
from rl.world.util.structured_logger import StructuredLogger
from rl.world.util.dataset_vcmi import DatasetVCMI, Data, Context


def vcmi_dataloader_functor():
    state = {"reward_carry": 0}

    def mw(data: Data, ctx: Context):
        if ctx.transition_id == ctx.num_transitions - 1:
            state["reward_carry"] = data.reward
            if not data.done:
                return None

        if (data.action - 2) % N_HEX_ACTIONS == HEX_ACT_MAP["MOVE"]:
            # Skip 50% of MOVEs
            if random.random() < 0.5:
                return None

        if ctx.transition_id == 0 and ctx.ep_steps > 0:
            data = data._replace(reward=state["reward_carry"])

        return data

    return mw


    # return model.predict(
    #     torch.as_tensor(obs).float(),
    #     torch.as_tensor(np.array(mask))
    # )


def main():
    logger = StructuredLogger(level=getattr(logging, "DEBUG"))
    env_kwargs = dict(
        mapname="gym/generated/4096/4x1024.vmap",
        max_steps=1000,
        opponent="StupidAI",
        random_heroes=1,
        random_obstacles=1,
        swap_sides=1,
        town_chance=30,
        warmachine_chance=40,
        random_stack_chance=65,
        tight_formation_chance=20,
        random_terrain_chance=100,
        allow_invalid_actions=True,
        user_timeout=3600,
        vcmi_timeout=3600,
        boot_timeout=300,
        vcmi_loglevel_global="error",
        vcmi_loglevel_ai="error",
        # swap_sides=1,
    )

    dataloader = torch.utils.data.DataLoader(
        DatasetVCMI(logger=logger, env_kwargs=env_kwargs, mw_functor=vcmi_dataloader_functor),
        batch_size=1000,
        num_workers=10,
        prefetch_factor=None
    )

    it = iter(dataloader)

    for x in range(10):
        print("========= %d ========" % x)
        batches = [next(it) for _ in range(100)]
        actions = torch.cat([b.action for b in batches])
        dones = torch.cat([b.done for b in batches])

        num_episodes = len(dones[dones > 0])
        num_samples = len(actions)
        action_counters = torch.bincount((actions - 2) % N_HEX_ACTIONS, minlength=N_HEX_ACTIONS)

        print("")
        print("* Total samples: %d" % num_samples)
        print("* Total episodes: %d" % num_episodes)
        print("* ep_len_mean: %d" % (num_samples / num_episodes) if num_episodes else float("nan"))

        assert action_counters.sum() == num_samples

        print("Action distribution:")
        print("")
        print("   action  | prob")
        print("-----------|-------")
        for name, count in zip(HEX_ACT_MAP, action_counters / num_samples):
            print(" %-9s | %.3f" % (name, count))

        print("")


if __name__ == "__main__":
    main()
