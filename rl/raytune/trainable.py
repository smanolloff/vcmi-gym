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
import importlib
from datetime import datetime
from .common import debuglog
from . import common


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Trainable.html
class Trainable(ray.tune.Trainable):
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
        os.environ["NO_SAVE"] = "true"
        self.agent = None
        self.experiment_name = initargs["_raytune"]["experiment_name"]

        if not hasattr(self, "algo"):
            self.algo = importlib.import_module("rl.algos.{a}.{a}".format(a=initargs["_raytune"]["algo"]))

        wandb.init(
            project=initargs["_raytune"]["wandb_project"],
            group=self.experiment_name,
            id=self.trial_id,
            name="T%d" % int(self.trial_name.split("_")[-1]),
            resume="allow",
            reinit=True,
            allow_val_change=True,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
            config=initargs,
            sync_tensorboard=False,
        )

        wandb.log({"trial/checkpoint_origin": int(self.trial_id.split("_")[1])}, commit=False)  # at init, we are the origin

        self.cfg = copy.deepcopy(initargs)
        self.cfg["wandb_project"] = initargs["_raytune"]["wandb_project"]  # needed by algo
        del self.cfg["_raytune"]  # rejected by algo

        # Define param name mapping for algo's wandb logging
        for k in common.flattened_dict_keys(cfg, "."):
            assert "/" not in k
            self.cfg["logparams"][f"params/{k}"] = k

        self.reset_config(cfg)

    @debuglog
    def reset_config(self, cfg):
        self.cfg = common.deepmerge(copy.deepcopy(self.cfg), cfg, allow_new=False, update_existing=True)
        return True

    @debuglog
    def save_checkpoint(self, checkpoint_dir):
        assert self.agent, "save_checkpoint called but self.agent is None"
        f = os.path.join(checkpoint_dir, "agent.pt")
        self.algo.Agent.save(self.agent, f)
        if int(self.trial_id.split("_")[1]) == 0:
            wandb.run.log_model(path=f, name="agent.pt")
        return checkpoint_dir

    @debuglog
    def load_checkpoint(self, checkpoint_dir):
        f = os.path.join(checkpoint_dir, "agent.pt")

        # Checkpoint tracking: log the trial ID of the checkpoint we are restoring now
        relpath = re.match(fr".+/{re.escape(self.experiment_name)}/(.+)", f).group(1)
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
        args = self.algo.Args(wandb.run.id, wandb.run.group, **self.cfg)
        self.agent, rew_mean = self.algo.main(args)
        return {"rew_mean": rew_mean}
