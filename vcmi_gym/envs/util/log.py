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

import logging
import numpy as np
import threading
import string
import random


class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.reltime = "%.2f" % (record.relativeCreated / 1000)
        record.thread_id = np.base_repr(threading.current_thread().ident, 36).lower()
        return super().format(record)


def get_logger(name, level):
    # Add a random string to logger name to prevent issues when
    # multiple envs open use the same logger name
    chars = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(chars) for _ in range(8))

    logger = logging.getLogger("%s/%s" % (random_string, name.split(".")[-1]))
    logger.setLevel(getattr(logging, level))

    # fmt = "-- %(reltime)ss (%(process)d) [%(name)s] %(levelname)s: %(message)s"
    fmt = f"-- %(asctime)s (%(process)d/%(thread_id)s) [{name}] %(levelname)s: %(message)s"

    formatter = CustomFormatter(fmt)
    formatter.default_time_format = "%H:%M:%S"
    formatter.default_msec_format = "%s.%03d"

    loghandler = logging.StreamHandler()
    loghandler.setLevel(logging.DEBUG)
    loghandler.setFormatter(formatter)
    logger.addHandler(loghandler)
    logger.propagate = False

    return logger


def trunc(string, maxlen):
    return f"{string[0:maxlen]}..." if len(string) > maxlen else string
