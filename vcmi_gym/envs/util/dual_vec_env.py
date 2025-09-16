import os
import enum
import gymnasium as gym
import numpy as np
import multiprocessing
import ctypes
import threading
import torch
from torch_geometric.data import Batch

from types import SimpleNamespace
from functools import partial
from multiprocessing.shared_memory import SharedMemory

from vcmi_gym.envs.v13 import VcmiEnv
from vcmi_gym.envs.v13.pyconnector import STATE_SIZE, LINK_TYPES
from vcmi_gym.envs.util.log import get_logger
from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper
from rl.algos.mppo_dna_gnn.mppo_dna_gnn import to_hdata_list


class EnvState(enum.IntEnum):
    UNSET = 0
    AWAITING_ACTION = enum.auto()
    DONE = enum.auto()


class DualEnvController():
    def __init__(self, num_envs, model_factory):
        self.num_envs = num_envs
        self.model_factory = model_factory
        self.logger = get_logger("controller", "DEBUG")

        self.controller_env_cond = multiprocessing.Condition()
        self.controller_act_cond = multiprocessing.Condition()

        e_max = 3300  # max number of links of a given type

        ary = lambda dummy, shm: np.ndarray(shape=dummy.shape, dtype=dummy.dtype, buffer=shm.buf)

        dummy = np.empty((num_envs,), dtype=np.uint8)
        self.shm_states = SharedMemory(create=True, size=dummy.nbytes)
        self.env_states = ary(dummy, self.shm_states)

        dummy = np.empty((num_envs, STATE_SIZE), dtype=np.float32)
        self.shm_obs = SharedMemory(create=True, size=dummy.nbytes)
        self.env_obs = ary(dummy, self.shm_obs)

        dummy = np.empty((num_envs, len(LINK_TYPES)), dtype=np.int64)
        self.shm_num_links = SharedMemory(create=True, size=dummy.nbytes)
        self.env_num_links = ary(dummy, self.shm_num_links)

        dummy = np.empty((num_envs, len(LINK_TYPES), 2, e_max), dtype=np.int64)
        self.shm_link_inds = SharedMemory(create=True, size=dummy.nbytes)
        self.env_link_inds = ary(dummy, self.shm_link_inds)

        dummy = np.empty((num_envs, len(LINK_TYPES), e_max, 1), dtype=np.float32)
        self.shm_link_attrs = SharedMemory(create=True, size=dummy.nbytes)
        self.env_link_attrs = ary(dummy, self.shm_link_attrs)

        dummy = np.empty((num_envs,), dtype=np.int64)
        self.shm_actions = SharedMemory(create=True, size=dummy.nbytes)
        self.env_actions = ary(dummy, self.shm_actions)

    # XXX: this should be started in a SUB-THREAD
    def run(self):
        self.logger.debug("Creating model")
        model = model_factory()

        self.logger.debug("with self.controller_env_cond")
        with self.controller_env_cond:
            # let the parent thread know we are ready, i.e. it can start envs
            # (prevents race cond where envs start filling env_states before
            #   the controller has had a chance to acquire the env_cond)
            self.logger.debug("self.controller_env_cond.notify()")
            self.controller_env_cond.notify()

            while True:
                self.logger.debug("controller_env_cond.wait_for(no_unsets)")
                self.controller_env_cond.wait_for(lambda: all(self.env_states != EnvState.UNSET))

                ids = np.where(self.env_states == EnvState.AWAITING_ACTION)[0]
                self.logger.debug("AWAITING_ACTION env_ids: %s" % str(ids))

                if len(ids) == 0:
                    # none are awaiting action i.e. all are DONE => reset states
                    self.logger.debug("env_states.fill(EnvState.UNSET)")
                    self.env_states.fill(EnvState.UNSET)
                    continue

                b_obs = torch.as_tensor(self.env_obs[ids])
                b_done = torch.zeros(len(ids))
                b_links = []    # B dicts, each a single obs's "links"

                for env_id in ids:
                    links = {}
                    for i, lt in enumerate(LINK_TYPES.keys()):
                        e = self.env_num_links[env_id, i]
                        links[lt] = dict(
                            index=self.env_link_inds[env_id, i, :, :e],
                            attrs=self.env_link_attrs[env_id, i, :e, :],
                        )
                    b_links.append(links)

                hdata = Batch.from_data_list(to_hdata_list(b_obs, b_done, b_links))

                with self.controller_act_cond:
                    self.logger.debug("model_policy.get_actdata_eval(hdata)")
                    self.env_actions[ids] = model.get_actdata_eval(hdata).action.numpy()
                    self.env_states[ids] = EnvState.UNSET
                    self.logger.debug("controller_act_cond.notify_all()")
                    self.controller_act_cond.notify_all()


def dual_env_bot(
    main_env,
    env_id,
    num_envs,
    e_max,
    controller_env_cond,
    controller_act_cond,
    shm_name_states,
    shm_name_obs,
    shm_name_num_links,
    shm_name_link_inds,
    shm_name_link_attrs,
    shm_name_actions,
):
    logger = get_logger(f"bot.{env_id}", "DEBUG")
    shm_actions = SharedMemory(name=shm_name_actions)
    env_actions = np.ndarray(shape=(num_envs,), dtype=np.int64, buffer=shm_actions.buf)
    shm_states = SharedMemory(name=shm_name_states)
    env_states = np.ndarray(shape=(num_envs,), dtype=np.uint8, buffer=shm_states.buf)
    shm_obs = SharedMemory(name=shm_name_obs)
    env_obs = np.ndarray(shape=(num_envs, STATE_SIZE), dtype=np.float32, buffer=shm_obs.buf)
    shm_num_links = SharedMemory(name=shm_name_num_links)
    env_num_links = np.ndarray(shape=(num_envs, len(LINK_TYPES)), dtype=np.int64, buffer=shm_num_links.buf)

    shm_link_inds = SharedMemory(name=shm_name_link_inds)
    env_link_inds = np.ndarray(shape=(num_envs, len(LINK_TYPES), 2, e_max), dtype=np.int64, buffer=shm_link_inds.buf)

    shm_link_attrs = SharedMemory(name=shm_name_link_attrs)
    env_link_attrs = np.ndarray(shape=(num_envs, len(LINK_TYPES), e_max, 1), dtype=np.float32, buffer=shm_link_attrs.buf)

    logger.debug("Create VcmiEnv")
    env = VcmiEnv(opponent="OTHER_ENV", main_env=main_env)
    logger.debug("env.connect()")
    env.connect()

    obs = env.obs
    done = False

    # Endless loop

    while True:
        if done:
            logger.debug("env.reset()")
            obs, info = env.reset()
            done = False

        # act = env.random_action()
        logger.debug("with controller_act_cond")
        with controller_act_cond:
            logger.debug("with controller_env_cond")
            with controller_env_cond:
                env_obs[env_id] = obs["observation"]
                for i, link_type in enumerate(LINK_TYPES.keys()):
                    links = obs["links"][link_type]
                    num = links["index"].shape[1]
                    env_num_links[env_id, i] = num
                    env_link_inds[env_id, i, :, :num] = links["index"]
                    env_link_attrs[env_id, i, :num, :] = links["attrs"]

                env_states[env_id] = EnvState.AWAITING_ACTION
                logger.debug("controller_env_cond.notify()")
                controller_env_cond.notify()
            logger.debug("controller_act_cond.wait()")
            controller_act_cond.wait()
            act = env_actions[env_id]

        logger.debug("env.step(%d)" % act)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc


class DualEnvMainWrapper(gym.Wrapper):
    def __init__(self, env, env_id, num_envs, controller_env_cond, shm_name_states):
        super().__init__(env)
        self.logger = get_logger(f"env.{env_id}", "DEBUG")
        self.env_id = env_id
        self.num_envs = num_envs
        self.controller_env_cond = controller_env_cond
        self.shm_states = SharedMemory(name=shm_name_states)
        self.env_states = np.ndarray((num_envs,), dtype=np.uint8, buffer=self.shm_states.buf)
        env.connect()

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.logger.debug("[reset] with self.controller_env_cond")
        with self.controller_env_cond:
            self.env_states[self.env_id] = EnvState.DONE
            self.logger.debug("[reset] self.controller_env_cond.notify()")
            self.controller_env_cond.notify()
        return obs, info

    def step(self, *args, **kwargs):
        self.logger.debug("[step] %s %s" % (str(args), str(kwargs)))
        obs, rew, term, trunc, info = self.env.step(*args, **kwargs)
        self.logger.debug("[step] with self.controller_env_cond")
        with self.controller_env_cond:
            self.env_states[self.env_id] = EnvState.DONE
            self.logger.debug("[step] self.controller_env_cond.notify()")
            self.controller_env_cond.notify()
        return obs, rew, term, trunc, info


def create_dual_venv(
    env_kwargs,
    num_envs_model,
    num_envs_stupidai,
    num_envs_battleai,
    model_factory,
    e_max=3300
):
    num_envs_total = num_envs_model + num_envs_stupidai + num_envs_battleai

    # AsyncVectorEnv creates a dummy_env() in the main process just to
    # extract metadata, which causes VCMI init pid error afterwards
    pid = os.getpid()
    dummy_env = SimpleNamespace(
        metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
        render_mode='ansi',
        action_space=VcmiEnv.ACTION_SPACE,
        observation_space=VcmiEnv.OBSERVATION_SPACE["observation"],
        close=lambda: None
    )

    controller = DualEnvController(num_envs_model, model_factory)
    threading.Thread(target=controller.run, daemon=True).start()

    bot_common_kwargs = dict(
        num_envs=num_envs_model,
        e_max=e_max,
        controller_env_cond=controller.controller_env_cond,
        controller_act_cond=controller.controller_act_cond,
        shm_name_states=controller.shm_states.name,
        shm_name_obs=controller.shm_obs.name,
        shm_name_num_links=controller.shm_num_links.name,
        shm_name_link_inds=controller.shm_link_inds.name,
        shm_name_link_attrs=controller.shm_link_attrs.name,
        shm_name_actions=controller.shm_actions.name,
    )

    def env_creator_wrapper(env_creator):
        if os.getpid() == pid:
            return dummy_env

        env = env_creator()
        env = LegacyObservationSpaceWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def env_creator_model(i):
        env = VcmiEnv(**env_kwargs, opponent="OTHER_ENV")
        threading.Thread(
            target=dual_env_bot,
            kwargs=dict(bot_common_kwargs, main_env=env, env_id=i),
            daemon=True
        ).start()
        env = DualEnvMainWrapper(
            env=env,
            env_id=i,
            num_envs=num_envs_total,
            controller_env_cond=controller.controller_env_cond,
            shm_name_states=controller.shm_states.name
        )
        return env

    def env_creator_stupidai():
        return VcmiEnv(**env_kwargs, opponent="StupidAI")

    def env_creator_battleai():
        return VcmiEnv(**env_kwargs, opponent="BattleAI")

    env_creators = []
    env_creators.extend([partial(env_creator_model, i) for i in range(num_envs_model)])
    env_creators.extend([env_creator_stupidai for i in range(num_envs_stupidai)])
    env_creators.extend([env_creator_battleai for i in range(num_envs_battleai)])
    funcs = [partial(env_creator_wrapper, env_creator) for env_creator in env_creators]
    vec_env = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    return vec_env


if __name__ == "__main__":
    import json
    from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel

    def model_factory():
        with open("sfcjqcly-1757757007-config.json", "r") as f:
            cfg = json.load(f)
        weights = torch.load("sfcjqcly-1757757007-model-dna.pt", weights_only=True, map_location="cpu")
        model = DNAModel(cfg["model"], torch.device("cpu")).eval()
        model.load_state_dict(weights, strict=True)
        return model.model_policy

    venv = create_dual_venv(
        env_kwargs=dict(mapname="gym/A1.vmap"),
        num_envs_model=5,
        num_envs_stupidai=2,
        num_envs_battleai=2,
        model_factory=model_factory
    )

    import ipdb; ipdb.set_trace()  # noqa
    pass
