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

import wandb


def wandb_init(trial_id, trial_name, experiment_name, config):
    # print("[%s] INITWANDB: PID: %s, trial_id: %s" % (time.time(), os.getpid(), trial_id))

    # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
    wandb.init(
        id=trial_id,
        name="T%d" % int(trial_name.split("_")[-1]),
        resume="allow",
        reinit=True,
        allow_val_change=True,
        # To disable System/ stats:
        # settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        group=experiment_name,
        project=config["wandb_project"],
        config=config,
        # NOTE: this takes a lot of time, better to have detailed graphs
        #       tb-only (local) and log only most important info to wandb
        # sync_tensorboard=True,
        sync_tensorboard=False,
    )
    # print("[%s] DONE WITH INITWANDB" % time.time())
