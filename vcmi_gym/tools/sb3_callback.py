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

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import wandb
import numpy as np
import time
import os
from collections import deque

from .. import InfoDict


class SB3Callback(BaseCallback):
    def __init__(self, observations_dir=None, success_rate_target=None, wandb_enabled=True):
        super().__init__()
        self.wandb_enabled = wandb_enabled

        if observations_dir:
            print("Will store observations in %s" % observations_dir)
            os.makedirs(observations_dir, exist_ok=True)

        self.observations_dir = observations_dir
        self.success_rate_target = success_rate_target
        self.rollout_episodes = 0
        self.rollouts = 0
        self.ep_rew_mean = 0
        self.success_rates = deque(maxlen=20)
        self.this_env_rollouts = 0

        # wdb tables are logged from OUTSIDE SB3Callback
        # only once, at end of the iteration
        # (ie. once every config["rollouts_per_iteration"])
        self.wdb_tables = {}

    def _on_step(self):
        new_episodes = self.locals["dones"].sum()
        self.rollout_episodes += new_episodes

        for idx, info in enumerate(self.locals["infos"]):
            if not self.locals["dones"][idx]:
                continue

            self.success_rates.append(info.get("is_success"))

        # at least 5 episodes
        if self.success_rate_target and len(self.success_rates) > 10:
            success_rate = safe_mean(self.success_rates)
            if success_rate > self.success_rate_target:
                self.model.logger.record("is_success", success_rate)
                if self.wandb_enabled:
                    wandb.log({"success_rate": success_rate})
                print("Early stopping after %d rollouts due to: success rate > %.2f (%.2f)" % (
                    self.this_env_rollouts,
                    self.success_rate_target,
                    success_rate
                ))
                self.logger.dump(step=self.num_timesteps)  # log dump required for watchdog
                return False  # early stop training

        return True

    def _on_rollout_end(self):
        if self.observations_dir:
            np.save(
                "%s/observations-%d" % (self.observations_dir, time.time() * 1000),
                self.model.rollout_buffer.observations
            )

        self.rollouts += 1
        self.this_env_rollouts += 1
        wdb_log = {
            "num_timesteps": self.num_timesteps,
            "rollout/n_episodes": self.rollout_episodes,
        }

        self.rollout_episodes = 0

        if self.rollouts % self.locals["log_interval"] != 0:
            return

        for k in InfoDict.SCALAR_VALUES:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"{k}", v)
            wdb_log[k] = v

        # From here on it's W&B stuff only
        wdb_log["n_rollouts"] = self.rollouts

        # Also add sb3's Monitor info keys: "r" (reward) and "l" (length)
        # (these are already recorded to TB by sb3, but not in W&B)
        self.ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_rew_mean"] = self.ep_rew_mean

        v = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_len_mean"] = v

        # for (k, columns) in InfoDict.D1_ARRAY_VALUES.items():
        #     action_types_vec_2d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
        #     ary = np.mean(action_types_vec_2d, axis=0)

        #     # In SB3's logger, Tensor objects are logged as a Histogram
        #     # https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/logger.py#L412
        #     # NOT logging this to TB, it's not visualized well there
        #     # tb_data = torch.as_tensor(ary)
        #     # self.model.logger.record(f"user/{k}", tb_data)

        #     # In W&B, we need to unpivot into a name/count table
        #     # NOTE: reserved column names: "id", "name", "_step" and "color"
        #     wk = f"table/{k}"
        #     rotated = [list(row) for row in zip(columns, ary)]
        #     if wk not in self.wdb_tables:
        #         self.wdb_tables[wk] = wandb.Table(columns=["key", "value"])

        #     wb_table = self.wdb_tables[wk]
        #     for row in rotated:
        #         wb_table.add_data(*row)

        if self.wandb_enabled:
            wandb.log(wdb_log, commit=True)
