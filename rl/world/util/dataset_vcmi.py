import time
import torch
import logging
import enum
import queue
import numpy as np
from torch.utils.data import IterableDataset
from typing import NamedTuple

from .timer import Timer


class Data(NamedTuple):
    obs: np.ndarray
    mask: np.ndarray
    reward: float
    done: bool
    action: int


class Context(NamedTuple):
    ep_steps: int
    transition_id: int
    num_transitions: int


class DataInstruction(enum.IntEnum):
    USE = 0
    SKIP = enum.auto()
    # SKIP_ALL = enum.auto()


# Need a functor instead of simple function when using DataLoader
# as each sub-process will need to use a separate copy of this function
def noop_functor():
    return lambda data, ctx: (data, DataInstruction.USE)


class DatasetVCMI(IterableDataset):
    def __init__(
        self,
        logger,
        env_kwargs,
        metric_queue=None,
        metric_report_interval=5,
        mw_functor=None
    ):
        self.logger = logger
        self.env_kwargs = env_kwargs
        self.metric_queue = metric_queue
        self.metric_report_interval = metric_report_interval
        self.metric_reported_at = time.time()
        self.timer_all = Timer()
        self.timer_idle = Timer()
        self.mw_functor = mw_functor or noop_functor

        print("Env kwargs: %s" % self.env_kwargs)
        self.env = None

    def __del__(self):
        self.env.close() if self.env else None

    def __iter__(self):
        assert self.env is None, "multiple calls to __iter__ not supported"

        from vcmi_gym import VcmiEnv_v12 as VcmiEnv
        self.env = VcmiEnv(**self.env_kwargs)

        obs = self.env.reset()[0]
        term = False
        trunc = False
        ep_steps = 0
        middleware = self.mw_functor()

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
                # Example for episode with 4 steps, 3 transitions each:
                #
                # Legend:
                #       O(s0t0) = observation at step0, transition0
                #       R(s0t0) = observation at step1, transition1
                #       A(s0t0) = action in O(s0t0) leading to O(s0t1)
                #                 NOTE: A(*t0) is always our action
                #                 NOTE: A(*t2) is always -1 (assuming 3 transitions)
                #
                #
                # t=0           | t=1          | t=2               |
                # --------------|--------------|-------------------|
                #        A(s0t0)|       A(s0t1)|       A(s0t2)=-1  |
                #       /       \      /       \      /            |  s=0 (ep start)
                # O(s0t0)       |O(s0t1)       |O(s0t2)            |
                # R(s0t0)=NaN   |R(s0t1)       |R(s0t2)            |
                #               |              |                   |
                # --------------|--------------|-------------------|
                #        A(s1t0)|       A(s1t1)|       A(s1t2)=-1  |
                #       /       \      /       \      /            |  s=1
                # O(s1t0)       |O(s1t1)       |O(s1t2)            |
                # R(s1t0)=NaN   |R(s1t1)       |R(s1t2)            |
                #               |              |                   |
                # --------------|--------------|-------------------|
                # ...           |              |                   |  s=2
                # --------------|--------------|-------------------|
                #        A(s3t0)|       A(s3t1)|       A(s3t2)=-1  |
                #       /       \      /       \      /            |  s=3 (ep end)
                # O(s3t0)       |O(s3t1)       |O(s3t2)            |
                # R(s3t0)=NaN   |R(s3t1)       |R(s3t2)            |
                #
                # =============================================================
                # IMPORTANT: duplicate observations at the edges:
                #   O(s0t2) == O(s1t0)
                #   O(s1t2) == O(s2t0)
                #   ... etc
                #

                num_transitions = len(obs["transitions"]["observations"])

                for t, (t_obs, t_mask, t_reward, t_done, t_action) in enumerate(zipped):
                    data, instruction = middleware(
                        Data(obs=t_obs, mask=t_mask, reward=t_reward, done=t_done, action=t_action),
                        Context(ep_steps=0, transition_id=t, num_transitions=num_transitions)
                    )

                    if instruction == DataInstruction.USE:
                        with self.timer_idle:
                            yield data
                    elif instruction == DataInstruction.SKIP:
                        pass
                    else:
                        raise Exception("Invalid DataInstruction: %s" % instruction)

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

    dataset = DatasetVCMI(
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
