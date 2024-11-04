import logging
import numpy as np
import threading
import string
import random

from ray.rllib.utils.annotations import override
from ray.tune.logger import Logger


class MPPO_Logger(Logger):
    # @override(Logger)
    # def _init(self):
    #     super().__init__(self)

    @override(Logger)
    def on_result(self, result: dict):
        # print(f"{self.prefix} " f"result[{ENV_RUNNER_RESULTS}][net_value]: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
        pass

    # @override(Logger)
    # def close(self):
    #     # Releases all resources used by this logger.
    #     pass

    # @override(Logger)
    # def flush(self):
    #     # Flushing all possible disk writes to permanent storage.
    #     pass


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

    return logger
