import logging
import numpy as np
import threading
import string
import random


class CustomFormatter(logging.Formatter):
    def __init__(self, origname, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origname = origname

    def format(self, record):
        record.reltime = "%.2f" % (record.relativeCreated / 1000)
        record.thread_id = np.base_repr(threading.current_thread().ident, 36).lower()
        record.origname = self.origname
        return super().format(record)


def get_logger(name, level):
    # Add a random string to logger name to prevent issues when
    # multiple envs open use the same logger name
    chars = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(chars) for _ in range(8))

    logger = logging.getLogger("%s/%s" % (random_string, name.split(".")[-1]))
    logger.setLevel(getattr(logging, level))

    # fmt = "-- %(reltime)ss (%(process)d) [%(name)s] %(levelname)s: %(message)s"
    fmt = "-- %(asctime)s (%(process)d/%(thread_id)s) [%(origname)s] %(levelname)s: %(message)s"

    formatter = CustomFormatter(fmt)
    formatter.default_time_format = "%H:%M:%S"
    formatter.default_msec_format = "%s.%03d"

    loghandler = logging.StreamHandler()
    loghandler.setLevel(logging.DEBUG)
    loghandler.setFormatter(formatter)
    logger.addHandler(loghandler)

    return logger


def trunc(string, maxlen):
    return f"{string[0:maxlen]}..." if len(string) > maxlen else string
