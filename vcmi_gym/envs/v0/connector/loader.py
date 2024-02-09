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

from sb3_contrib import MaskablePPO
from sb3_contrib import QRDQN
import connexport


# XXX: maybe import VcmiEnv and load offset from there?
ACTION_OFFSET = 1
OBS_SHAPE = (1, 11, 15 * connexport.get_n_hex_attrs())


class Loader:
    class MPPO:
        # def __init__(self, file):
        #     self.model = MaskablePPO.load(file)
        #     # self.obs = np.ndarray((2310,), dtype=np.float32)
        #     # self.actmasks = np.ndarray((1652,), dtype=np.bool)

        # def predict(self, obs, actmasks):
        #     # np.copyto(self.obs, obs)
        #     # np.copyto(self.actmasks, actmasks)
        #     action, _states = self.model.predict(
        #         obs.reshape(OBS_SHAPE),
        #         action_masks=actmasks[ACTION_OFFSET:]
        #     )
        #     return action + ACTION_OFFSET

        #
        # QRDQN
        #
        def __init__(self, file):
            self.model = QRDQN.load(file)

        def predict(self, obs, actmasks):
            action, _states = self.model.predict(obs.reshape(OBS_SHAPE))
            return action + ACTION_OFFSET
