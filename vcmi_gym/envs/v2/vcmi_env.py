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

import numpy as np
import gymnasium as gym

from ..v1.vcmi_env import VcmiEnv as VcmiEnv_v1
from .decoder import Decoder
from .pyconnector import (
    PyConnector,
    N_ACTIONS,
    STATE_VALUE_NA,
    STATE_SIZE_ONE_HEX
)


class VcmiEnv(VcmiEnv_v1):
    CONNECTOR_CLASS = PyConnector
    ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)
    OBSERVATION_SPACE = gym.spaces.Box(
        low=STATE_VALUE_NA,
        high=1,
        shape=(11, 15, STATE_SIZE_ONE_HEX),
        dtype=np.float32
    )

    @staticmethod
    def decode_obs(obs):
        return Decoder.decode(obs)
