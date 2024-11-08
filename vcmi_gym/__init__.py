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
from .envs.v1.vcmi_env import VcmiEnv as VcmiEnv_v1, InfoDict
from .envs.v1.decoder.other import (
    HexAction,
    HexState,
    MeleeDistance,
    ShootDistance,
    DmgMod,
    Side,
)

from .envs.v2.vcmi_env import VcmiEnv as VcmiEnv_v2
from .envs.v3.vcmi_env import VcmiEnv as VcmiEnv_v3
from .envs.v4.vcmi_env import VcmiEnv as VcmiEnv_v4

from .envs.util.dual_env import DualEnvController, DualEnvClient
from .envs.util.wrappers import LegacyActionSpaceWrapper, LegacyObservationSpaceWrapper
from .tools.test_helper import TestHelper


def register_envs():
    common_opts = dict(disable_env_checker=True, order_enforce=False)
    for v in [1, 2, 3, 4]:
        env_id = f"VCMI-v{v}"
        entry_point = f"vcmi_gym:VcmiEnv_v{v}"
        if env_id not in gymnasium.envs.registration.registry:
            gymnasium.register(id=env_id, entry_point=entry_point, **common_opts)


all = [
    register_envs,
    VcmiEnv_v1,
    VcmiEnv_v2,
    VcmiEnv_v3,
    VcmiEnv_v4,
    DualEnvController,
    DualEnvClient,
    HexAction,
    HexState,
    MeleeDistance,
    ShootDistance,
    DmgMod,
    Side,
    InfoDict,
    TestHelper,
    LegacyActionSpaceWrapper,
    LegacyObservationSpaceWrapper
]
