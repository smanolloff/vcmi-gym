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
import copy
import re
import ray.tune
import ray.train
import wandb
from datetime import datetime
from .common import debuglog
from . import common
from ..algos import mppo


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Trainable.html
class TrainableMPPO(ray.tune.Trainable):
    def log(self, msg):
        print("-- %s [%s I=%d] %s" % (datetime.now().isoformat(), self.trial_name, self.iteration, msg))

    @staticmethod
    def default_resource_request(_config):
        return ray.tune.execution.placement_groups.PlacementGroupFactory([{"CPU": 1}])

    # XXX: in case of SIGTERM/SIGINT, ray does not wait for cleanup to finish.
    #      During regular perturbation, though, it does.
    @debuglog
    def cleanup(self):
        wandb.finish(quiet=True)

    @debuglog
    def setup(self, cfg, initargs):
        self.agent = None
        self.experiment_name = initargs["experiment_name"]

        os.environ["NO_SAVE"] = "true"
        common.wandb_init(self.trial_id, self.trial_name, self.experiment_name, initargs["config"])
        wandb.log({"trial/checkpoint_origin": int(self.trial_id.split("_")[1])}, commit=False)  # at init, we are the origin

        self.cfg = copy.deepcopy(initargs["config"]["all_params"])
        self.cfg["wandb_project"] = wandb.run.project
        self.reset_config(cfg)

    @debuglog
    def reset_config(self, cfg):
        self.cfg = common.deepmerge(copy.deepcopy(self.cfg), cfg)

        for k in common.flattened_dict_keys(cfg, "."):
            assert "/" not in k
            self.cfg["logparams"][f"params/{k}"] = k

        return True

    @debuglog
    def save_checkpoint(self, checkpoint_dir):
        assert self.agent, "save_checkpoint called but self.agent is None"
        f = os.path.join(checkpoint_dir, "agent.pt")
        mppo.save(self.agent, f)
        return checkpoint_dir

    @debuglog
    def load_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "agent.pt")

        # Checkpoint tracking: log the trial ID of the checkpoint we are restoring now
        relpath = re.match(fr".+/{self.experiment_name}/(.+)", f).group(1)
        # => "6e59d_00004/checkpoint_000038/model.zip"
        origin = int(re.match(r".+?_(\d+)/.+", relpath).group(1))
        # => int("00004") => 4
        wandb.log({"trial/checkpoint_origin": origin}, commit=False)

        self.log("Load %s (origin: %d)" % (relpath, origin))
        self.cfg["agent_load_file"] = f  # algo will load it

    @debuglog
    def step(self):
        wandb.log({"trial/iteration": self.iteration}, commit=False)
        self.cfg["skip_wandb_log_code"] = (self.iteration > 0)
        args = mppo.Args(wandb.run.id, wandb.run.group, **self.cfg)
        self.agent, rew_mean = mppo.main(args)
        return {"rew_mean": rew_mean}
