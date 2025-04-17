import time
import torch
import logging
import queue
from torch.utils.data import IterableDataset

from ..util.timer import Timer


class VCMIDatasetAction(IterableDataset):
    def __init__(self, logger, env_kwargs, metric_queue=None, metric_report_interval=5):
        self.logger = logger
        self.env_kwargs = env_kwargs
        self.metric_queue = metric_queue
        self.metric_report_interval = metric_report_interval
        self.metric_reported_at = time.time()
        self.timer_all = Timer()
        self.timer_idle = Timer()

        print("Env kwargs: %s" % self.env_kwargs)
        self.env = None

    def __del__(self):
        self.env.close() if self.env else None

    def __iter__(self):
        # worker_id = torch.utils.data.get_worker_info().id
        # i = 0

        assert self.env is None, "multiple calls to __iter__ not supported"

        from vcmi_gym import VcmiEnv_v10 as VcmiEnv
        self.env = VcmiEnv(**self.env_kwargs)

        obs = self.env.reset()[0]
        term = False
        trunc = False
        ep_steps = 0

        with self.timer_all:
            while True:
                dones = [False] * len(obs["transitions"]["observations"])
                dones[-1] = term or trunc

                zipped = zip(
                    obs["transitions"]["observations"],
                    obs["transitions"]["action_masks"],
                    obs["transitions"]["rewards"],
                    dones,
                    obs["transitions"]["actions"],
                )

                # =============================================================
                # See notes in t10n/util/vcmidataset.py

                final_t = len(obs["transitions"]["observations"]) - 1

                for t, (t_obs, t_mask, t_reward, t_done, t_action) in enumerate(zipped):
                    # First transition is OUR state (the action is random for it)
                    # Last transition is OUR state (the action is always -1 there)
                    # => skip unless state is terminal (action=-1 stands for done=true)
                    if t == 0 or (t == final_t and not t_done):
                        continue

                    with self.timer_idle:
                        yield t_obs, t_mask, t_reward, t_done, t_action

                    if self.metric_queue and time.time() - self.metric_reported_at > self.metric_report_interval:
                        self.metric_reported_at = time.time()
                        try:
                            self.metric_queue.put(self.utilization(), block=False)
                        except queue.Full:
                            logger.warn("Failed to report metric (queue full)")

                if term or trunc:
                    obs = self.env.reset()[0]
                    term = False
                    trunc = False
                    ep_steps = 0
                else:
                    obs, _rew, term, trunc, _info = self.env.step(self.env.random_action())
                    ep_steps += 1

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
            mapname="gym/generated/4096/4x1024.vmap",
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

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=5, prefetch_factor=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, num_workers=0)

    for x in dataloader:
        import ipdb; ipdb.set_trace()  # noqa
        print("yee")
