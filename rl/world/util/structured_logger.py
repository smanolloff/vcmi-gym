import json
import logging
import numpy as np
import threading
from datetime import datetime


class StructuredLogger:
    def __init__(self, level, filename=None, context={}):
        self.level = level
        self.filename = filename
        self.context = context
        self.info(dict(event="logger init", filename=filename))

        assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        self.level = level

    def sublogger(self, context={}):
        return self.__class__(self.level, self.filename, dict(self.context, **context))

    def log(self, obj):
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds')
        thread_id = np.base_repr(threading.current_thread().ident, 36).lower()
        log_obj = dict(timestamp=timestamp, thread_id=thread_id, **dict(self.context, message=obj))
        # print(yaml.dump(log_obj, sort_keys=False))
        print(json.dumps(log_obj, sort_keys=False))

        if self.filename:
            with open(self.filename, "a+") as f:
                f.write(json.dumps(log_obj) + "\n")

    def debug(self, obj):
        self._level_log(obj, logging.DEBUG, "DEBUG")

    def info(self, obj):
        self._level_log(obj, logging.INFO, "INFO")

    def warn(self, obj):
        self._level_log(obj, logging.WARN, "WARN")

    def warning(self, obj):
        self._level_log(obj, logging.WARNING, "WARNING")

    def error(self, obj):
        self._level_log(obj, logging.ERROR, "ERROR")

    def _level_log(self, obj, level, levelname):
        if self.level > level:
            return
        if isinstance(obj, dict):
            self.log(dict(obj))
        else:
            self.log(dict(string=obj))
