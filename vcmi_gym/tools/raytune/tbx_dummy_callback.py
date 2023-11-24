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
