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

import torch.optim
import torch.nn
from vcmi_gym import VcmiPPO
from .ppo_trainer import PPOTrainer

DEBUG = True


class VPPOTrainer(PPOTrainer):
    def _learner_kwargs_for_init(self):
        policy_kwargs = {
            "net_arch": self.cfg["net_arch"],
            "activation_fn": getattr(torch.nn, self.cfg["activation"]),
            "lstm_hidden_size": self.cfg["lstm_hidden_size"],
            "enable_critic_lstm": self.cfg["enable_critic_lstm"],
            "features_extractor_kwargs": self.cfg["features_extractor"]["kwargs"],
            "optimizer_class": getattr(torch.optim, self.cfg["optimizer"]["class_name"]),
            "optimizer_kwargs": self.cfg["optimizer"]["kwargs"]
        }

        return dict(
            self.cfg["learner_kwargs"],
            policy="VcmiPolicy",
            policy_kwargs=policy_kwargs
        )

    def _model_internal_init(self, venv, **learner_kwargs):
        print("VcmiPPO(%s)" % learner_kwargs)
        return VcmiPPO(env=venv, **learner_kwargs)

    def _model_internal_load(self, f, venv, **learner_kwargs):
        return VcmiPPO.load(f, env=venv, **learner_kwargs)
