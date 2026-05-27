import enum
import os
import gymnasium as gym
import multiprocessing
import ctypes
import numpy as np
import torch
import threading
from functools import partial
from torch_geometric.data import HeteroData, Batch
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace
from vcmi_gym.envs.util.wrappers import BlankObservationSpaceWrapper
from vcmi_gym.envs.util.log import get_logger, trunc
from vcmi_gym.envs.v15.vcmi_env import VcmiEnv
from vcmi_gym.envs.v15.pyconnector import (
    NODE_TYPES,
    EDGE_TYPES,
)

from abc import ABC, abstractmethod


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


class AbstractModelLoader(ABC):
    @abstractmethod
    def configure(self, config_file: str):
        ...

    @abstractmethod
    def load(self, weights_file: str):
        ...

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        ...


class EnvState(enum.IntEnum):
    UNSET = 0
    AWAITING_ACTION = enum.auto()
    DONE = enum.auto()


class DualEnvController():
    def __init__(self, num_envs, model_loader: AbstractModelLoader, logprefix="", loglevel="INFO"):
        self.num_envs = num_envs
        self.model_loader = model_loader
        self.logger = get_logger(f"{logprefix}controller", loglevel)
        self.logger.debug("Initializing...")

        self.controller_env_cond = multiprocessing.Condition()
        self.controller_act_cond = multiprocessing.Condition()

        self.smm = SharedMemoryManager()
        self.smm.start()

        self.ipc_nodes = {}
        self.ipc_edges = {}

        make_ary = lambda dummy, shm: np.ndarray(shape=dummy.shape, dtype=dummy.dtype, buffer=shm.buf)

        def add_node_ipc(name, traits, nmax):
            # Use a dummy np array to easily calculate needed sizes
            dummy = np.empty((num_envs, nmax, traits["size"]), dtype=np.float32)
            shm = self.smm.SharedMemory(size=dummy.nbytes)
            ary = make_ary(dummy, shm)
            n_max = multiprocessing.Value(ctypes.c_int64)
            self.ipc_nodes[name] = dict(shm=shm, ary=ary, n_max=n_max)

        for name, traits in NODE_TYPES:
            match name:
                case "Global": nmax = 1
                case "Player": nmax = 2
                case "Unit": nmax = 30  # TODO: guard at runtime
                case "Hex": nmax = 165
                case "Action": nmax = 200 * 30  # ~200 max actions per unit
                case _: raise Exception(f"Unknown node type: {name}")

            add_node_ipc(name, traits, nmax)

        def add_edge_ipc(key, traits, emax):
            # Use a dummy np array to easily calculate needed sizes
            dummy_index = np.empty((num_envs, 2, emax), dtype=np.int64)
            dummy_attrs = np.empty((num_envs, emax, traits["size"]), dtype=np.float32)
            shm_index = self.smm.SharedMemory(size=dummy_index.nbytes)
            shm_attrs = self.smm.SharedMemory(size=dummy_attrs.nbytes)
            ary_index = make_shm_array(dummy_index, shm)
            ary_attrs = make_shm_array(dummy_attrs, shm)
            e_max = multiprocessing.Value(ctypes.c_int64)
            self.ipc_edges[key] = dict(
                index=dict(shm=shm_index, ary=ary_index),
                attrs=dict(shm=shm_attrs, ary=ary_attrs),
                emax=emax
            )

        for key, traits in EDGE_TYPES:
            match key:
                case ("Global", "Has", "Player"): emax = 2
                case ("Global", "Has", "Unit"): emax = self.ipc_nodes["Unit"]["nmax"]
                case ("Global", "Has", "Hex"): emax = self.ipc_nodes["Hex"]["nmax"]
                case ("Player", "Owns", "Unit"): emax = self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Hex", "Adjacent", "Hex"): emax = 888
                case ("Unit", "ActsBefore", "Unit"): emax = self.ipc_nodes["Unit"]["nmax"]
                case ("Unit", "MeleeDmg", "Unit"): emax = self.ipc_nodes["Unit"]["nmax"]
                case ("Unit", "ShootDmg", "Unit"): emax = self.ipc_nodes["Unit"]["nmax"]
                case ("Unit", "Blocks", "Unit"): emax = self.ipc_nodes["Unit"]["nmax"]
                case ("Unit", "Occupies", "Hex"): emax = self.ipc_nodes["Unit"]["nmax"] * 2  # wide units occupy 2 hexes
                case ("Action", "By", "Unit"): emax = self.ipc_nodes["Action"]["nmax"]
                case ("Action", "EndsAt", "Hex"): emax = self.ipc_nodes["Action"]["nmax"] * 2
                case ("Action", "Blocks", "Unit"): emax = self.ipc_nodes["Action"]["nmax"]
                case ("Action", "ExposesToMeleeFrom", "Unit"): emax = self.ipc_nodes["Action"]["nmax"] * self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Action", "ExposesToShootFrom", "Unit"): emax = self.ipc_nodes["Action"]["nmax"] * self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Action", "Melees", "Unit"): emax = self.ipc_nodes["Action"]["nmax"] * self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Action", "Shoots", "Unit"): emax = self.ipc_nodes["Action"]["nmax"] * self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Action", "EnablesMeleeAt", "Unit"): emax = self.ipc_nodes["Action"]["nmax"] * self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Action", "EnablesShootAt", "Unit"): emax = self.ipc_nodes["Action"]["nmax"] * self.ipc_nodes["Unit"]["nmax"] / 2
                case ("Action", "EnablesMeleeAt", "Hex"): emax = (self.ipc_nodes["Action"]["nmax"] / self.ipc_nodes["Unit"]["nmax"]) * 165 # these two are for active unit only
                case ("Action", "EnablesShootAt", "Hex"): emax = (self.ipc_nodes["Action"]["nmax"] / self.ipc_nodes["Unit"]["nmax"]) * 165 #
                case _: raise Exception(f"Unknown edge type: {key}")

            add_edge_ipc(key, traits, emax)

        # XXX: this is WIP:
        # DualEnvWrapper needs to be amended to use the above ipc info
        # This is only needed when num_envs_model > 0, so not implementing for now

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.reload_lock = threading.RLock()

    def start(self):
        self.logger.debug("Starting runner thread")
        self.thread.start()

    # XXX: this runs in a thread
    def _run(self):
        self.logger.debug("Creating model")

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
                    hdata = Batch.from_data_list(to_hdata_list(b_done, b_links)).to(self.model_loader.get_model().device)

                    with self.controller_act_cond:
                        self.logger.debug("model.model_policy.get_actdata_eval(hdata)")
                        self.env_actions[ids] = self.model_loader.get_model().model_policy.get_action_logits(hdata).sample().action.cpu().numpy()
                        self.logger.debug(f"self.env_states[{ids}] = {EnvState.UNSET}")
                        self.env_states[ids] = EnvState.UNSET
                        self.logger.debug("controller_act_cond.notify_all()")
                        self.controller_act_cond.notify_all()

    def reload_model(self):
        if not self.model_loader:
            return

        with self.controller_act_cond:
            self.model_loader.load()


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
        self.logger.debug("[init] with self.controller_env_cond")
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

        if term or trunc:
            # Setting DONE here would cause controller to re-set ALL states to UNSET.
            # However, the Vector env will automatically call .reset()
            # => if the bot is first after reset, this env ends up in AWAITING_ACTION state
            #    but the other envs are are UNSET => controller won't wake up.
            self.logger.debug("terminal step -- not setting DONE flag (expecting reset)")
        else:
            self.logger.debug("[step] with self.controller_env_cond")
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
        num_envs_mmai_battleai=0,
        num_envs_model=0,
        model_loader: AbstractModelLoader = None,
        logprefix="",
    ):
        num_envs_total = num_envs_model + num_envs_stupidai + num_envs_battleai + num_envs_mmai_battleai
        assert num_envs_total > 0, f"{num_envs_total} > 0"

        # AsyncVectorEnv creates a dummy_env() in the main process just to
        # extract metadata, which causes VCMI init pid error afterwards
        pid = os.getpid()
        dummy_env = SimpleNamespace(
            metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
            render_mode='ansi',
            action_space=VcmiEnv.ACTION_SPACE,
            observation_space=BlankObservationSpaceWrapper.SPACE,
            close=lambda: None
        )

        if num_envs_model > 0:
            # test model exists (avoids errors in sub-processes)
            model_loader.get_model()

            self.controller = DualEnvController(
                num_envs_model,
                model_loader,
                loglevel=env_kwargs.get("vcmienv_loglevel", "INFO"),
                logprefix=logprefix,
            )
            self.controller.start()

            shm_names_nodes = {name: ipc["shm"].name for name, ipc in self.controller.ipc_nodes}
            shm_names_edges = {
                key: {"index": ipc["shm_index"].name, "attrs": ipc["shm_attrs"].name}
                for key, ipc in self.controller.ipc_edges
            }

            dual_kwargs = dict(
                num_envs=num_envs_model,
                controller_env_cond=self.controller.controller_env_cond,
                controller_act_cond=self.controller.controller_act_cond,
                shm_names_nodes=shm_names_nodes,
                shm_names_edges=shm_names_edges,
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

        def env_creator_mmai_battleai(i):
            return VcmiEnv(**env_kwargs, opponent="MMAI_BATTLEAI", vcmienv_logtag=f"{logprefix}env.mmaibattleai.{i}")

        def env_creator_wrapper(env_creator):
            if os.getpid() == pid:
                return dummy_env

            env = env_creator()

            # XXX: this is needed since my observations now have irregular shapes
            # => AsyncVectorEnv can't batch them
            # => expose a dummy observation and never use it in the RL algo
            #   (use .call("graph") which should return separate dicts with edges and nodes)
            env = BlankObservationSpaceWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            return env

        env_creators = []
        env_creators.extend([partial(env_creator_model, i) for i in range(num_envs_model)])
        env_creators.extend([partial(env_creator_stupidai, i) for i in range(num_envs_stupidai)])
        env_creators.extend([partial(env_creator_battleai, i) for i in range(num_envs_battleai)])
        env_creators.extend([partial(env_creator_mmai_battleai, i) for i in range(num_envs_mmai_battleai)])
        funcs = [partial(env_creator_wrapper, env_creator) for env_creator in env_creators]

        super().__init__(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    def reload_model(self):
        self.controller.reload_model()


def to_hdata(done, obs):
    device = obs.device
    res = HeteroData()
    res.done = done.unsqueeze(0).float()
    res.value = torch.tensor(0., device=device)
    res.action = torch.tensor(0, device=device)
    res.reward = torch.tensor(0., device=device)
    res.logprob = torch.tensor(0., device=device)
    res.advantage = torch.tensor(0., device=device)
    res.ep_return = torch.tensor(0., device=device)

    for node, attrs in obs["nodes"]:
        res[node].x = torch.as_tensor(attrs, device=device)

    for key, edge in obs["edges"]:
        res[key].edge_index = torch.as_tensor(edge["index"], device=device)
        res[key].edge_attr = torch.as_tensor(edge["attrs"], device=device)

    return res


# b_obs: torch.tensor of shape (B, STATE_SIZE)
# tuple_links: tuple of B dicts, where each dict is a single obs's "links"
def to_hdata_list(b_done, obs):
    b_hdatas = []
    for done, obs in zip(b_done, obs):
        b_hdatas.append(to_hdata(done, obs))
    # XXX: this concatenates along the first dim
    # i.e. stacking two (165, STATE_SIZE_ONE_HEX)
    #       gives  (330, STATE_SIZE_ONE_HEX)
    #       (that's how GNN batching works)
    return b_hdatas


if __name__ == "__main__":
    import json
    from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel

    class TestModelLoader(AbstractModelLoader):
        def __init__(self):
            self.model = False

        def configure(self, config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        # This function will be called from within another process
        # (referenced non-local objects must be serializable for IPC).
        def load(self, weights_file):
            print(f"[TestModelLoader] Loading model weights from {weights_file}")
            with torch.inference_mode():
                weights = torch.load(weights_file, weights_only=True, map_location="cpu")
                if not self.model:
                    self.model = DNAModel(self.config["model"], torch.device("cpu")).eval()
                self.model.load_state_dict(weights, strict=True)

        def get_model(self):
            return self.model

    model_loader = TestModelLoader()
    venv = DualVecEnv(
        env_kwargs=dict(mapname="gym/A1.vmap"),
        num_envs_stupidai=0,
        num_envs_battleai=0,
        num_envs_mmai_battleai=0,
        num_envs_model=1,
        model_loader=model_loader,
        logprefix="test-",
        e_max=3300,
    )

    model_loader.configure("export/tukbajrv-202509241418-config.json")
    model_loader.load("export/tukbajrv-202509241418-model-dna.pt")

    import ipdb; ipdb.set_trace()  # noqa
    pass
