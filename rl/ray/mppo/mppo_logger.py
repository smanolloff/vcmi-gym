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
