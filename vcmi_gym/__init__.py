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
from .envs.v3.vcmi_env import VcmiEnv as VcmiEnv_v3
from .envs.v7.vcmi_env import VcmiEnv as VcmiEnv_v7
from .envs.v8.vcmi_env import VcmiEnv as VcmiEnv_v8
from .envs.v9.vcmi_env import VcmiEnv as VcmiEnv_v9
from .envs.v10.vcmi_env import VcmiEnv as VcmiEnv_v10

from .envs.util.dual_env import DualEnvController, DualEnvClient


def register_envs():
    common_opts = dict(disable_env_checker=True, order_enforce=False)
    for v in [3, 7, 8, 9, 10]:
        env_id = f"VCMI-v{v}"
        entry_point = f"vcmi_gym:VcmiEnv_v{v}"
        if env_id not in gymnasium.envs.registration.registry:
            gymnasium.register(id=env_id, entry_point=entry_point, **common_opts)


all = [
    register_envs,
    VcmiEnv_v3,
    VcmiEnv_v7,
    VcmiEnv_v8,
    VcmiEnv_v9,
    VcmiEnv_v10,
    DualEnvController,
    DualEnvClient,
]
