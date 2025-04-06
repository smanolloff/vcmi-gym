import time
import torch
import logging
import queue
from torch.utils.data import IterableDataset

from .. import common


class VCMIDatasetAction(IterableDataset):
    def __init__(self, logger, env_kwargs, metric_queue=None, metric_report_interval=5):
        self.logger = logger
        self.env_kwargs = env_kwargs
        self.metric_queue = metric_queue
        self.metric_report_interval = metric_report_interval
        self.metric_reported_at = time.time()
        self.timer_all = common.Timer()
        self.timer_idle = common.Timer()

        print("Env kwargs: %s" % self.env_kwargs)
        self.env = None

    def __del__(self):
        self.env.close() if self.env else None

    def __iter__(self):
        # worker_id = torch.utils.data.get_worker_info().id
        # i = 0

        if self.env is None:
            from vcmi_gym import VcmiEnv_v10 as VcmiEnv
            self.env = VcmiEnv(**self.env_kwargs)
            self.env.reset()

        with self.timer_all:
            while True:
                for obs, action in zip(self.env.result.intstates[1:], self.env.result.intactions[1:]):
                    # if i % 1000 == 0:
                    #     print("[%d] %d" % (worker_id, i))
                    # i += 1

                    with self.timer_idle:
                        yield obs, action

                    if self.metric_queue and time.time() - self.metric_reported_at > self.metric_report_interval:
                        self.metric_reported_at = time.time()
                        try:
                            self.metric_queue.put(self.utilization(), block=False)
                        except queue.Full:
                            logger.warn("Failed to report metric (queue full)")

                if self.env.terminated or self.env.truncated:
                    self.env.reset()
                else:
                    self.env.step(self.env.random_action())

    def utilization(self):
        if self.timer_all.peek() == 0:
            return 0
        return 1 - (self.timer_idle.peek() / self.timer_all.peek())


if __name__ == "__main__":
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    dataset = VCMIDatasetAction(
        logger=logger,
        env_kwargs=dict(
            max_steps=500,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            random_heroes=1,
            random_obstacles=1,
            town_chance=10,
            warmachine_chance=40,
            mana_min=0,
            mana_max=0,
            reward_step_fixed=-1,
            reward_dmg_mult=1,
            reward_term_mult=1,
            swap_sides=0,
            user_timeout=0,
            vcmi_timeout=0,
            boot_timeout=0,
            conntype="thread"
        )
    )

    # XXX: prefetch_factor=N means:
    #      always keep N*num_workers*batch_size records preloaded
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10_000, num_workers=5, prefetch_factor=1)
    for x in dataloader:
        import ipdb; ipdb.set_trace()  # noqa
        print("yee")
