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

# from sb3_contrib import QRDQN
import torch
# import numpy as np
import connexport
import os


# XXX: maybe import VcmiEnv and load offset from there?
ACTION_OFFSET = 1
OBS_SHAPE = (11, 15, connexport.get_state_size_one_hex())


class Loader:
    class MPPO:
        def __init__(self, file):
            try:
                file = os.path.realpath(file)
                self.model = torch.load(file)
            except Exception as e:
                print("MPPO Load Error: %s" % repr(e))
                raise
            # self.obs = np.ndarray((2310,), dtype=np.float32)
            # self.actmasks = np.ndarray((1652,), dtype=np.bool)

        # Obs is flattened here (shape is (9240))
        def predict(self, obs, mask):
            action = self.model.predict(obs.reshape(OBS_SHAPE), mask[ACTION_OFFSET:])
            action += ACTION_OFFSET
            print("Agent prediction: %s" % action)
            return action
