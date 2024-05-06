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
# distanceributed under the License is distanceributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import gymnasium
from .envs.v0.vcmi_env import VcmiEnv, Hex, State, Action, ShootDistance, MeleeDistance, DmgMod, Side, InfoDict
from .tools.test_helper import TestHelper

all = [
    VcmiEnv,
    Hex,
    State,
    Action,
    MeleeDistance,
    ShootDistance,
    DmgMod,
    Side,
    InfoDict,
    TestHelper,
]

gymnasium.register(id="VCMI-v0", entry_point="vcmi_gym:VcmiEnv")
