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

from ray.tune.logger import TBXLoggerCallback


class TBXDummyCallback(TBXLoggerCallback):
    """ A dummy class to be passed to ray Tuner at init.

    This will trick ray into believing it has a TBX logger already
    and will not create a new, default one.
    I dont want hundreds of tb files created with useless info in my data dir
    """

    def __init__(self):
        pass

    def log_trial_start(self, *args, **kwargs):
        pass

    def log_trial_result(self, *args, **kwargs):
        pass

    def log_trial_end(self, *args, **kwargs):
        pass
