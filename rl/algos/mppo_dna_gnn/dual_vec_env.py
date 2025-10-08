import enum
import os
import gymnasium as gym
import multiprocessing
import numpy as np
import torch
import threading
from functools import partial
from torch_geometric.data import HeteroData, Batch
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace
from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper
from vcmi_gym.envs.util.log import get_logger, trunc
from vcmi_gym.envs.v13.vcmi_env import VcmiEnv
from vcmi_gym.envs.v13.pyconnector import (
    LINK_TYPES,
    STATE_SIZE,
    STATE_SIZE_HEXES,
    STATE_SIZE_ONE_HEX,
)


TRACE = os.getenv("VCMIGYM_DEBUG", "0") == "1"


def tracelog(func, maxlen=80):
    if not TRACE:
        return func

    def wrapper(*args, **kwargs):
        this = args[0]
        this.logger.debug("Begin: %s (args=%s, kwargs=%s)" % (func.__name__, args[1:], trunc(repr(kwargs), maxlen)))
        result = func(*args, **kwargs)
        this.logger.debug("End: %s (return %s)" % (func.__name__, trunc(repr(result), maxlen)))
        return result

    return wrapper


class EnvState(enum.IntEnum):
    UNSET = 0
    AWAITING_ACTION = enum.auto()
    DONE = enum.auto()


class DualEnvController():
    def __init__(self, num_envs, model_factory, logprefix="", loglevel="INFO"):
        self.num_envs = num_envs
        self.model_factory = model_factory
        self.logger = get_logger(f"{logprefix}controller", loglevel)
        self.logger.debug("Initializing...")

        self.controller_env_cond = multiprocessing.Condition()
        self.controller_act_cond = multiprocessing.Condition()

        e_max = 3300  # max number of links of a given type

        ary = lambda dummy, shm: np.ndarray(shape=dummy.shape, dtype=dummy.dtype, buffer=shm.buf)

        self.smm = SharedMemoryManager()
        self.smm.start()

        dummy = np.empty((num_envs,), dtype=np.uint8)
        self.shm_states = self.smm.SharedMemory(size=dummy.nbytes)
        self.env_states = ary(dummy, self.shm_states)

        dummy = np.empty((num_envs, STATE_SIZE), dtype=np.float32)
        self.shm_obs = self.smm.SharedMemory(size=dummy.nbytes)
        self.env_obs = ary(dummy, self.shm_obs)

        dummy = np.empty((num_envs, len(LINK_TYPES)), dtype=np.int64)
        self.shm_num_links = self.smm.SharedMemory(size=dummy.nbytes)
        self.env_num_links = ary(dummy, self.shm_num_links)

        dummy = np.empty((num_envs, len(LINK_TYPES), 2, e_max), dtype=np.int64)
        self.shm_link_inds = self.smm.SharedMemory(size=dummy.nbytes)
        self.env_link_inds = ary(dummy, self.shm_link_inds)

        dummy = np.empty((num_envs, len(LINK_TYPES), e_max, 1), dtype=np.float32)
        self.shm_link_attrs = self.smm.SharedMemory(size=dummy.nbytes)
        self.env_link_attrs = ary(dummy, self.shm_link_attrs)

        dummy = np.empty((num_envs,), dtype=np.int64)
        self.shm_actions = self.smm.SharedMemory(size=dummy.nbytes)
        self.env_actions = ary(dummy, self.shm_actions)

        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.logger.debug("Starting runner thread")
        self.thread.start()

    # XXX: this runs in a thread
    def _run(self):
        self.logger.debug("Creating model")

        with torch.inference_mode():
            model = self.model_factory()

        def no_unsets():
            self.logger.debug(f"[no_unsets]: env_states = {self.env_states}")
            return all(self.env_states != EnvState.UNSET)

        self.logger.debug("with self.controller_env_cond")
        with self.controller_env_cond:
            while True:
                self.logger.debug("controller_env_cond.wait_for(no_unsets)")
                self.controller_env_cond.wait_for(no_unsets)

                ids = np.where(self.env_states == EnvState.AWAITING_ACTION)[0]

                if len(ids) == 0:
                    # none are awaiting action i.e. all are DONE => reset states
                    self.logger.debug("all envs DONE => env_states.fill(EnvState.UNSET)")
                    self.env_states.fill(EnvState.UNSET)
                    continue

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

                with torch.inference_mode():
                    b_obs = torch.as_tensor(self.env_obs[ids])
                    b_done = torch.zeros(len(ids))
                    b_result = torch.zeros(len(ids, dtype=torch.int64))
                    hdata = Batch.from_data_list(to_hdata_list(b_obs, b_done, b_result, b_links)).to(model.device)

                    with self.controller_act_cond:
                        self.logger.debug("model.model_policy.get_actdata_eval(hdata)")
                        self.env_actions[ids] = model.model_policy.get_actdata_eval(hdata).action.cpu().numpy()
                        self.logger.debug(f"self.env_states[{ids}] = {EnvState.UNSET}")
                        self.env_states[ids] = EnvState.UNSET
                        self.logger.debug("controller_act_cond.notify_all()")
                        self.controller_act_cond.notify_all()


class DualEnvWrapper(gym.Wrapper):
    @staticmethod
    def _bot_loop(
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
        logprefix=""
    ):
        env_logtag = f"{main_env.vcmienv_logtag}-other"
        logger = get_logger(f"{logprefix}bot({env_logtag})", main_env.vcmienv_loglevel)
        logger.debug("initializing...")

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

        logger.debug("Create other VcmiEnv")
        env = VcmiEnv(opponent="OTHER_ENV", main_env=main_env, vcmienv_loglevel=main_env.vcmienv_loglevel, vcmienv_logtag=f"{logprefix}{env_logtag}")

        logger.debug("env.connect() -- as %s" % env.role)
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

                    logger.debug(f"env_states[{env_id}] = {EnvState.AWAITING_ACTION}")
                    env_states[env_id] = EnvState.AWAITING_ACTION
                    logger.debug("controller_env_cond.notify()")
                    controller_env_cond.notify()
                logger.debug("controller_act_cond.wait()")
                controller_act_cond.wait()
                act = env_actions[env_id]

            logger.debug("env.step(%d)" % act)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc

    def __init__(
        self,
        env,
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
        logprefix=""
    ):
        super().__init__(env)

        self.logger = get_logger(f"{logprefix}wrapper({env.vcmienv_logtag})", env.vcmienv_loglevel)
        self.logger.debug("Initializing...")
        self.env_id = env_id
        self.num_envs = num_envs
        self.controller_env_cond = controller_env_cond
        self.shm_states = SharedMemory(name=shm_name_states)
        self.env_states = np.ndarray((num_envs,), dtype=np.uint8, buffer=self.shm_states.buf)

        self.bot_thread = threading.Thread(target=self.__class__._bot_loop, daemon=True, kwargs=dict(
            main_env=env,
            env_id=env_id,
            num_envs=num_envs,
            e_max=e_max,
            controller_env_cond=controller_env_cond,
            controller_act_cond=controller_act_cond,
            shm_name_states=shm_name_states,
            shm_name_obs=shm_name_obs,
            shm_name_num_links=shm_name_num_links,
            shm_name_link_inds=shm_name_link_inds,
            shm_name_link_attrs=shm_name_link_attrs,
            shm_name_actions=shm_name_actions,
        ))

        self.logger.debug("starting bot thread")
        self.bot_thread.start()

        self.logger.debug("env.connect() -- as %s" % env.role)
        env.connect()

        # .connect() calls .reset() internally
        # => must explicitly notify controller
        with self.controller_env_cond:
            self.logger.debug(f"env_states[{self.env_id}] = {EnvState.DONE}")
            self.env_states[self.env_id] = EnvState.DONE
            self.logger.debug("[init] self.controller_env_cond.notify()")
            self.controller_env_cond.notify()

    @tracelog
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.logger.debug("[reset] with self.controller_env_cond")
        with self.controller_env_cond:
            self.logger.debug(f"env_states[{self.env_id}] = {EnvState.DONE}")
            self.env_states[self.env_id] = EnvState.DONE
            self.logger.debug("[reset] self.controller_env_cond.notify()")
            self.controller_env_cond.notify()
        return obs, info

    @tracelog
    def step(self, *args, **kwargs):
        self.logger.debug("[step] %s %s" % (str(args), str(kwargs)))
        obs, rew, term, trunc, info = self.env.step(*args, **kwargs)
        self.logger.debug("[step] with self.controller_env_cond")

        if term or trunc:
            # Setting DONE here would cause controller to re-set ALL states to UNSET.
            # However, the Vector env will automatically call .reset()
            # => if the bot is first after reset, this env ends up in AWAITING_ACTION state
            #    but the other envs are are UNSET => controller won't wake up.
            self.logger.debug("terminal step -- not setting DONE flag (expecting reset)")
        else:
            with self.controller_env_cond:
                self.logger.debug(f"env_states[{self.env_id}] = {EnvState.DONE}")
                self.env_states[self.env_id] = EnvState.DONE
                self.logger.debug("[step] self.controller_env_cond.notify()")
                self.controller_env_cond.notify()
        return obs, rew, term, trunc, info


class DualVecEnv(gym.vector.AsyncVectorEnv):
    def __init__(
        self,
        env_kwargs,
        num_envs_stupidai=0,
        num_envs_battleai=0,
        num_envs_model=0,
        model_factory=None,
        e_max=3300,
        logprefix="",
    ):
        num_envs_total = num_envs_model + num_envs_stupidai + num_envs_battleai
        assert num_envs_total > 0, f"{num_envs_total} > 0"

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

        if num_envs_model > 0:
            # test if model can be loaded (avoids errors in sub-processes)
            model_factory()

            self.controller = DualEnvController(
                num_envs_model,
                model_factory,
                loglevel=env_kwargs.get("vcmienv_loglevel", "INFO"),
                logprefix=logprefix,
            )
            self.controller.start()

            dual_kwargs = dict(
                num_envs=num_envs_model,
                e_max=e_max,
                controller_env_cond=self.controller.controller_env_cond,
                controller_act_cond=self.controller.controller_act_cond,
                shm_name_states=self.controller.shm_states.name,
                shm_name_obs=self.controller.shm_obs.name,
                shm_name_num_links=self.controller.shm_num_links.name,
                shm_name_link_inds=self.controller.shm_link_inds.name,
                shm_name_link_attrs=self.controller.shm_link_attrs.name,
                shm_name_actions=self.controller.shm_actions.name,
                logprefix=logprefix
            )

            def env_creator_model(i):
                env = VcmiEnv(**env_kwargs, opponent="OTHER_ENV", vcmienv_logtag=f"{logprefix}env.model.{i}")
                env = DualEnvWrapper(env, env_id=i, **dual_kwargs)
                return env

        def env_creator_stupidai(i):
            return VcmiEnv(**env_kwargs, opponent="StupidAI", vcmienv_logtag=f"{logprefix}env.stupidai.{i}")

        def env_creator_battleai(i):
            return VcmiEnv(**env_kwargs, opponent="BattleAI", vcmienv_logtag=f"{logprefix}env.battleai.{i}")

        def env_creator_wrapper(env_creator):
            if os.getpid() == pid:
                return dummy_env

            env = env_creator()
            env = LegacyObservationSpaceWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        env_creators = []
        env_creators.extend([partial(env_creator_model, i) for i in range(num_envs_model)])
        env_creators.extend([partial(env_creator_stupidai, i) for i in range(num_envs_stupidai)])
        env_creators.extend([partial(env_creator_battleai, i) for i in range(num_envs_battleai)])
        funcs = [partial(env_creator_wrapper, env_creator) for env_creator in env_creators]

        super().__init__(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def to_hdata(obs, done, result, links):
    device = obs.device
    res = HeteroData()
    res.obs = obs.unsqueeze(0)
    res.done = done.unsqueeze(0).float()
    res.value = torch.tensor(0., device=device)
    res.action = torch.tensor(0, device=device)
    res.reward = torch.tensor(0., device=device)
    res.logprob = torch.tensor(0., device=device)
    res.advantage = torch.tensor(0., device=device)
    res.ep_return = torch.tensor(0., device=device)
    res.ep_result = result.unsqueeze(0)

    res["hex"].x = obs[-STATE_SIZE_HEXES:].view(165, STATE_SIZE_ONE_HEX)
    for lt in LINK_TYPES.keys():
        res["hex", lt, "hex"].edge_index = torch.as_tensor(links[lt]["index"], device=device)
        res["hex", lt, "hex"].edge_attr = torch.as_tensor(links[lt]["attrs"], device=device)

    return res


# b_obs: torch.tensor of shape (B, STATE_SIZE)
# tuple_links: tuple of B dicts, where each dict is a single obs's "links"
def to_hdata_list(b_obs, b_done, b_ep_result, tuple_links):
    b_hdatas = []
    for obs, done, ep_result, links in zip(b_obs, b_done, b_ep_result, tuple_links):
        b_hdatas.append(to_hdata(obs, done, ep_result, links))
    # XXX: this concatenates along the first dim
    # i.e. stacking two (165, STATE_SIZE_ONE_HEX)
    #       gives  (330, STATE_SIZE_ONE_HEX)
    #       not sure if that's required for GNN to work?
    #       but it breaks my encode() which uses torch.split()
    return b_hdatas


if __name__ == "__main__":
    import json
    from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel

    def model_factory():
        with open("sfcjqcly-1757757007-config.json", "r") as f:
            cfg = json.load(f)
        weights = torch.load("sfcjqcly-1757757007-model-dna.pt", weights_only=True, map_location="cpu")
        model = DNAModel(cfg["model"], torch.device("cpu")).eval()
        model.load_state_dict(weights, strict=True)
        return model

    venv = DualVecEnv(
        env_kwargs=dict(mapname="gym/A1.vmap"),
        num_envs_stupidai=2,
        num_envs_battleai=2,
        num_envs_model=5,
        model_factory=model_factory,
        logprefix="test-",
        e_max=3300,
    )

    import ipdb; ipdb.set_trace()  # noqa
    pass
