import enum
import os
import gymnasium as gym
import multiprocessing
import numpy as np
import torch
import threading
from typing import NamedTuple
from functools import partial
from torch_geometric.data import Batch
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from vcmi_gym.envs.util.wrappers import BlankObservationSpaceWrapper
from vcmi_gym.envs.util.log import get_logger, trunc
from vcmi_gym.envs.v15.vcmi_env import VcmiEnv
from vcmi_gym.envs.v15.pyconnector import (
    NODE_TYPES,
    EDGE_TYPES,
)
from rl.v15.gnn_model import to_hdata_list, add_action_active_local_ids

from abc import ABC, abstractmethod


TRACE = os.getenv("VCMIGYM_DEBUG", "0") == "1"


class ThreadFailure:
    def __init__(self, thread_name: str):
        self.thread_name: str = thread_name
        self.event: threading.Event = threading.Event()
        self.exception: BaseException | None = None

    def raise_if_set(self):
        if self.event.is_set():
            assert self.exception is not None
            raise RuntimeError(f"Background thread {self.thread_name!r} failed") from self.exception


def start_thread(*, target, logger, kwargs=None, daemon=True, notify_conditions=()):
    thread_name = target.__name__
    failure = ThreadFailure(thread_name)

    def run():
        try:
            target(**(kwargs or {}))
        except BaseException as exc:
            failure.exception = exc
            failure.event.set()
            logger.exception("Unhandled exception in background thread %s", thread_name)

            for condition in notify_conditions:
                with condition:
                    condition.notify_all()

    thread = threading.Thread(target=run, name=thread_name, daemon=daemon)
    thread.start()
    return thread, failure


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
    def get_model(self) -> torch.nn.Module | None:
        ...


class EnvState(enum.IntEnum):
    UNSET = 0
    AWAITING_ACTION = enum.auto()
    DONE = enum.auto()


class DualEnvController():
    NODE_MAX = {
        "Global": 1,
        "Player": 2,
        "Unit": 30,
        "Hex": 165,
        "Action": 3000,
    }

    EDGE_MAX = {
        ("Global", "To", "Player"): 2,
        ("Player", "To", "Global"): 2,
        ("Global", "To", "Unit"): 30,
        ("Unit", "To", "Global"): 30,
        ("Global", "To", "Hex"): 165,
        ("Hex", "To", "Global"): 165,
        ("Global", "To", "Action"): 3000,
        ("Player", "Owns", "Unit"): 30,
        ("Unit", "OwnedBy", "Player"): 30,
        ("Unit", "Occupies", "Hex"): 50,
        ("Hex", "OccupiedBy", "Unit"): 50,
        ("Action", "By", "Unit"): 3000,
        ("Unit", "Has", "Action"): 3000,
        ("Hex", "Adjacent", "Hex"): 888,
        ("Unit", "ActsBefore", "Unit"): 300,
        ("Unit", "MeleeDmg", "Unit"): 300,
        ("Unit", "ShootDmg", "Unit"): 300,
        ("Unit", "Blocks", "Unit"): 30,
        ("Action", "EndsAt", "Hex"): 6000,
        ("Hex", "IsEndOf", "Action"): 6000,
        ("Action", "Blocks", "Unit"): 2000,
        ("Unit", "BlockedBy", "Action"): 2000,
        ("Unit", "BecomesMeleeThreatAfter", "Action"): 15000,
        ("Unit", "BecomesShootThreatAfter", "Action"): 15000,
        ("Unit", "IsMeleedBy", "Action"): 700,
        ("Unit", "IsShotBy", "Action"): 500,  # magog or lich vs clumped VIP/guard stacks makes a lot

        # NOTE: these edges are significantly less than the max values if the env ignores them:
        # ignored_edges=[
        #     # These often dominate the graph (e.g. 30k of total 40k edges)
        #     ["Hex", "BecomesMeleeTargetAfter", "Action"],
        #     ["Hex", "BecomesShootTargetAfter", "Action"],
        # ],

        ("Unit", "BecomesMeleeTargetAfter", "Action"): 20000,
        ("Unit", "BecomesShootTargetAfter", "Action"): 20000,
        ("Hex", "BecomesMeleeTargetAfter", "Action"): 60000,
        ("Hex", "BecomesShootTargetAfter", "Action"): 60000,
    }

    for nt in NODE_TYPES.keys():
        assert nt in NODE_MAX, nt

    for et in EDGE_TYPES.keys():
        assert et in EDGE_MAX, et

    def __init__(self, num_envs, model_loader: AbstractModelLoader, logprefix="", loglevel="INFO"):
        self.num_envs = num_envs
        self.model_loader = model_loader
        self.logger = get_logger(f"{logprefix}controller", loglevel)
        self.logger.debug("Initializing...")

        self.controller_env_cond = multiprocessing.Condition()
        self.controller_act_cond = multiprocessing.Condition()
        self.node_names = tuple(NODE_TYPES.keys())
        self.edge_keys = tuple(EDGE_TYPES.keys())

        self.shms = []

        self.ipc_nodes = {}
        self.ipc_edges = {}

        def make_ary(dummy, shm):
            return np.ndarray(shape=dummy.shape, dtype=dummy.dtype, buffer=shm.buf)

        def make_shm_array(shape, dtype):
            dummy = np.empty(shape, dtype=dtype)
            shm = SharedMemory(create=True, size=dummy.nbytes)
            self.shms.append(shm)
            return shm, make_ary(dummy, shm)

        self.shm_states, self.env_states = make_shm_array((num_envs,), np.uint8)
        self.env_states.fill(EnvState.UNSET)

        self.shm_actions, self.env_actions = make_shm_array((num_envs,), np.int64)
        self.shm_num_nodes, self.env_num_nodes = make_shm_array((num_envs, len(self.node_names)), np.int64)
        self.shm_num_edges, self.env_num_edges = make_shm_array((num_envs, len(self.edge_keys)), np.int64)
        self.shm_num_active_action_ids, self.env_num_active_action_ids = make_shm_array((num_envs,), np.int64)
        self.shm_active_action_ids, self.env_active_action_ids = make_shm_array(
            (num_envs, self.NODE_MAX["Action"]),
            np.int64,
        )

        for name, traits in NODE_TYPES.items():
            if name not in self.NODE_MAX:
                raise Exception(f"Unknown node type: {name}")

            nmax = self.NODE_MAX[name]
            size = traits["size"]
            shm, ary = make_shm_array((num_envs, nmax, size), np.float32)
            self.ipc_nodes[name] = dict(shm=shm, ary=ary, nmax=nmax, size=size)

        for key, traits in EDGE_TYPES.items():
            if key not in self.EDGE_MAX:
                raise Exception(f"Unknown edge type: {key}")

            emax = self.EDGE_MAX[key]
            size = traits["size"]
            shm_index, ary_index = make_shm_array((num_envs, 2, emax), np.int64)

            if size > 0:
                shm_attrs, ary_attrs = make_shm_array((num_envs, emax, size), np.float32)
            else:
                shm_attrs = None
                ary_attrs = None

            self.ipc_edges[key] = dict(
                index=dict(shm=shm_index, ary=ary_index),
                attrs=dict(shm=shm_attrs, ary=ary_attrs),
                emax=emax,
                size=size,
            )

        self.thread = None
        self.thread_failure = None
        self.reload_lock = threading.RLock()

    def start(self):
        self.logger.debug("Starting runner thread")
        self.thread, self.thread_failure = start_thread(
            target=self._run,
            logger=self.logger,
            notify_conditions=(self.controller_env_cond, self.controller_act_cond),
        )

    def raise_if_thread_failed(self):
        if self.thread_failure is not None:
            self.thread_failure.raise_if_set()

    def close(self):
        for shm in reversed(self.shms):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _model_device(model):
        if hasattr(model, "device"):
            return model.device

        return next(model.parameters()).device

    @staticmethod
    def _model_policy(model):
        return model.model_policy if hasattr(model, "model_policy") else model

    def _obs_from_shm(self, env_id):
        nodes = {}
        edges = {}

        for i, name in enumerate(self.node_names):
            n = int(self.env_num_nodes[env_id, i])
            nodes[name] = self.ipc_nodes[name]["ary"][env_id, :n, :]

        for i, key in enumerate(self.edge_keys):
            e = int(self.env_num_edges[env_id, i])
            size = self.ipc_edges[key]["size"]
            edge_attrs = (
                self.ipc_edges[key]["attrs"]["ary"][env_id, :e, :]
                if size > 0
                else np.empty((e, 0), dtype=np.float32)
            )
            edges[key] = dict(
                index=self.ipc_edges[key]["index"]["ary"][env_id, :, :e],
                attrs=edge_attrs,
            )

        n_active = int(self.env_num_active_action_ids[env_id])
        active_action_ids = self.env_active_action_ids[env_id, :n_active]

        return dict(
            nodes=nodes,
            edges=edges,
            active_action_ids=active_action_ids,
        )

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

                model = self.model_loader.get_model()
                if model is None or model is False:
                    raise RuntimeError("DualEnvController requires model_loader.get_model() to return a loaded model")

                b_obs = [self._obs_from_shm(env_id) for env_id in ids]

                with torch.inference_mode():
                    b_done = torch.zeros(len(ids), dtype=torch.bool)
                    hdata = getattr(Batch.from_data_list(to_hdata_list(b_obs, b_done)), "to")(self._model_device(model))
                    add_action_active_local_ids(hdata)
                    action, _logprob, _entropy = self._model_policy(model).forward_policy(hdata)

                    with self.controller_act_cond:
                        self.env_actions[ids] = action.cpu().numpy()
                        self.logger.debug(f"self.env_states[{ids}] = {EnvState.UNSET}")
                        self.env_states[ids] = EnvState.UNSET
                        self.logger.debug("controller_act_cond.notify_all()")
                        self.controller_act_cond.notify_all()

    def reload_model(self, *args, **kwargs):
        self.raise_if_thread_failed()

        if not self.model_loader:
            return

        with self.controller_act_cond:
            self.model_loader.load(*args, **kwargs)

        self.raise_if_thread_failed()


class DualEnvWrapper(gym.Wrapper):
    @staticmethod
    def _attach_array(shm_name, shape, dtype):
        shm = SharedMemory(name=shm_name)
        ary = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        return shm, ary

    @staticmethod
    def _copy_obs_to_shm(
        obs,
        env_id,
        node_names,
        edge_keys,
        node_buffers,
        edge_buffers,
        env_num_nodes,
        env_num_edges,
        env_num_active_action_ids,
        env_active_action_ids,
    ):
        for i, name in enumerate(node_names):
            attrs = np.asarray(obs["nodes"][name], dtype=np.float32)
            num = attrs.shape[0]
            nmax = node_buffers[name]["nmax"]
            if num > nmax:
                raise RuntimeError(f"Too many {name} nodes for IPC buffer: {num} > {nmax}")

            env_num_nodes[env_id, i] = num
            node_buffers[name]["ary"][env_id, :num, :] = attrs

        for i, key in enumerate(edge_keys):
            edge = obs["edges"][key]
            index = np.asarray(edge["index"], dtype=np.int64)
            attrs = np.asarray(edge["attrs"], dtype=np.float32)
            num = index.shape[1]
            emax = edge_buffers[key]["emax"]
            if num > emax:
                raise RuntimeError(f"Too many {key} edges for IPC buffer: {num} > {emax}")

            env_num_edges[env_id, i] = num
            edge_buffers[key]["index"]["ary"][env_id, :, :num] = index
            if edge_buffers[key]["size"] > 0:
                edge_buffers[key]["attrs"]["ary"][env_id, :num, :] = attrs

        active_action_ids = np.asarray(obs["active_action_ids"], dtype=np.int64)
        num_active = active_action_ids.shape[0]
        if num_active > env_active_action_ids.shape[1]:
            raise RuntimeError(
                f"Too many active actions for IPC buffer: {num_active} > {env_active_action_ids.shape[1]}"
            )

        env_num_active_action_ids[env_id] = num_active
        env_active_action_ids[env_id, :num_active] = active_action_ids

    @staticmethod
    def _bot_loop(
        main_env,
        env_id,
        num_envs,
        node_names,
        edge_keys,
        controller_env_cond,
        controller_act_cond,
        shm_name_states,
        shm_name_actions,
        shm_name_num_nodes,
        shm_name_num_edges,
        shm_name_num_active_action_ids,
        shm_name_active_action_ids,
        shm_names_nodes,
        shm_names_edges,
        logprefix=""
    ):
        env_logtag = f"{main_env.vcmienv_logtag}-other"
        logger = get_logger(f"{logprefix}bot({env_logtag})", main_env.vcmienv_loglevel)
        logger.debug("initializing...")

        _shm_actions, env_actions = DualEnvWrapper._attach_array(shm_name_actions, (num_envs,), np.int64)
        _shm_states, env_states = DualEnvWrapper._attach_array(shm_name_states, (num_envs,), np.uint8)
        _shm_num_nodes, env_num_nodes = DualEnvWrapper._attach_array(shm_name_num_nodes, (num_envs, len(node_names)), np.int64)
        _shm_num_edges, env_num_edges = DualEnvWrapper._attach_array(shm_name_num_edges, (num_envs, len(edge_keys)), np.int64)
        _shm_num_active_action_ids, env_num_active_action_ids = DualEnvWrapper._attach_array(shm_name_num_active_action_ids, (num_envs,), np.int64)
        _shm_active_action_ids, env_active_action_ids = DualEnvWrapper._attach_array(
            shm_name_active_action_ids,
            (num_envs, DualEnvController.NODE_MAX["Action"]),
            np.int64,
        )

        node_buffers = {}
        edge_buffers = {}

        for name, spec in shm_names_nodes.items():
            shm, ary = DualEnvWrapper._attach_array(
                spec["shm"],
                (num_envs, spec["nmax"], spec["size"]),
                np.float32,
            )
            node_buffers[name] = dict(shm=shm, ary=ary, nmax=spec["nmax"], size=spec["size"])

        for key, spec in shm_names_edges.items():
            shm_index, ary_index = DualEnvWrapper._attach_array(
                spec["index"],
                (num_envs, 2, spec["emax"]),
                np.int64,
            )

            if spec["size"] > 0:
                shm_attrs, ary_attrs = DualEnvWrapper._attach_array(
                    spec["attrs"],
                    (num_envs, spec["emax"], spec["size"]),
                    np.float32,
                )
            else:
                shm_attrs = None
                ary_attrs = None

            edge_buffers[key] = dict(
                index=dict(shm=shm_index, ary=ary_index),
                attrs=dict(shm=shm_attrs, ary=ary_attrs),
                emax=spec["emax"],
                size=spec["size"],
            )

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

            logger.debug("with controller_act_cond")
            with controller_act_cond:
                logger.debug("with controller_env_cond")
                with controller_env_cond:
                    DualEnvWrapper._copy_obs_to_shm(
                        obs=obs,
                        env_id=env_id,
                        node_names=node_names,
                        edge_keys=edge_keys,
                        node_buffers=node_buffers,
                        edge_buffers=edge_buffers,
                        env_num_nodes=env_num_nodes,
                        env_num_edges=env_num_edges,
                        env_num_active_action_ids=env_num_active_action_ids,
                        env_active_action_ids=env_active_action_ids,
                    )

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
        node_names,
        edge_keys,
        controller_env_cond,
        controller_act_cond,
        shm_name_states,
        shm_name_actions,
        shm_name_num_nodes,
        shm_name_num_edges,
        shm_name_num_active_action_ids,
        shm_name_active_action_ids,
        shm_names_nodes,
        shm_names_edges,
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

        self.logger.debug("starting bot thread")
        self.bot_thread, self.bot_thread_failure = start_thread(
            target=self.__class__._bot_loop,
            logger=self.logger,
            kwargs=dict(
                main_env=env,
                env_id=env_id,
                num_envs=num_envs,
                node_names=node_names,
                edge_keys=edge_keys,
                controller_env_cond=controller_env_cond,
                controller_act_cond=controller_act_cond,
                shm_name_states=shm_name_states,
                shm_name_actions=shm_name_actions,
                shm_name_num_nodes=shm_name_num_nodes,
                shm_name_num_edges=shm_name_num_edges,
                shm_name_num_active_action_ids=shm_name_num_active_action_ids,
                shm_name_active_action_ids=shm_name_active_action_ids,
                shm_names_nodes=shm_names_nodes,
                shm_names_edges=shm_names_edges,
            ),
            notify_conditions=(controller_env_cond, controller_act_cond),
        )

        self.logger.debug("env.connect() -- as %s" % env.role)
        env.connect()
        self.bot_thread_failure.raise_if_set()

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
        self.bot_thread_failure.raise_if_set()
        obs, info = self.env.reset(*args, **kwargs)
        self.bot_thread_failure.raise_if_set()
        self.logger.debug("[reset] with self.controller_env_cond")

        with self.controller_env_cond:
            self.logger.debug(f"env_states[{self.env_id}] = {EnvState.DONE}")
            self.env_states[self.env_id] = EnvState.DONE
            self.logger.debug("[reset] self.controller_env_cond.notify()")
            self.controller_env_cond.notify()

        self.bot_thread_failure.raise_if_set()
        return obs, info

    @tracelog
    def step(self, *args, **kwargs):
        self.bot_thread_failure.raise_if_set()
        self.logger.debug("[step] %s %s" % (str(args), str(kwargs)))
        obs, rew, term, trunc, info = self.env.step(*args, **kwargs)
        self.bot_thread_failure.raise_if_set()

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

        self.bot_thread_failure.raise_if_set()
        return obs, rew, term, trunc, info

    @property
    def obs(self):
        self.bot_thread_failure.raise_if_set()
        return self.env.obs


class EnvMeta(NamedTuple):
    type: str = ""  # StupidAI | BattleAI | VIPBot | HARBot | MMAI_MODEL | torch_model
    num: int = 0
    kwargs: dict[str, any] = {}  # env-specific kwargs to be merged with env_kwargs
    model_loader: None = None  # needed for torch_model envs only


class DualVecEnv(gym.vector.AsyncVectorEnv):
    def __init__(
        self,
        seed: int,
        env_metas: list[EnvMeta],
        logprefix=""
    ):
        num_envs_total = sum(em.num for em in env_metas)
        assert num_envs_total > 0, f"{num_envs_total} > 0"
        assert seed >= 0 and seed <= (2**31 - 1 - num_envs_total)

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

        self.controller = None
        have_torch_model_envmeta = False
        env_creators = []

        for ienv, em in enumerate(env_metas):
            if em.num == 0:
                continue

            if em.type in ["StupidAI", "BattleAI", "VIPBot", "HARBot"]:
                # Avoid late binding via a "opponent" arg with default value
                def creator(i, ienv=ienv, em=em):
                    return VcmiEnv(
                        **dict(em.kwargs, seed=seed + i, **em.kwargs),
                        opponent=em.type,
                        vcmienv_logtag=f"{logprefix}env.{ienv}.{i}"
                    )
            elif em.type == "onnx_model":
                assert em.kwargs["opponent_model"] is not None, "opponent_model is required for onnx_model envs"
                def creator(i, ienv=ienv, em=em):
                    return VcmiEnv(
                        **dict(em.kwargs, seed=seed + i, **em.kwargs),
                        opponent="MMAI_MODEL",
                        vcmienv_logtag=f"{logprefix}env.{ienv}.{i}"
                    )
            elif em.type == "torch_model":
                assert not have_torch_model_envmeta, "there can be at most 1 DualVecEnv of type torch_model"
                have_torch_model_envmeta = True
                assert em.model_loader is not None, "model_loader is required for torch_model envs"

                # Test model exists early; otherwise the controller would block the bot side later.
                if em.model_loader.get_model() is None or em.model_loader.get_model() is False:
                    raise ValueError("model_loader must be configured and loaded before creating DualVecEnv with model opponents")

                user_timeout = em.kwargs.get("user_timeout", None)
                vcmi_timeout = em.kwargs.get("vcmi_timeout", None)

                # Model envs use a separate VcmiEnv for both players
                # => user_timeout for one is vcmi_timeout for the other
                # => the two timeouts must match
                assert user_timeout == vcmi_timeout, f"user_timeout and vcmi_timeout must match: {user_timeout} <> {vcmi_timeout}"

                controller = DualEnvController(
                    em.num,
                    em.model_loader,
                    loglevel=em.kwargs.get("vcmienv_loglevel", "INFO"),
                    logprefix=logprefix,
                )
                self.controller = controller
                controller.start()

                shm_names_nodes = {
                    name: {
                        "shm": ipc["shm"].name,
                        "nmax": ipc["nmax"],
                        "size": ipc["size"],
                    }
                    for name, ipc in controller.ipc_nodes.items()
                }
                shm_names_edges = {
                    key: {
                        "index": ipc["index"]["shm"].name,
                        "attrs": ipc["attrs"]["shm"].name if ipc["attrs"]["shm"] is not None else None,
                        "emax": ipc["emax"],
                        "size": ipc["size"],
                    }
                    for key, ipc in controller.ipc_edges.items()
                }

                node_names = controller.node_names
                edge_keys = controller.edge_keys
                controller_env_cond = controller.controller_env_cond
                controller_act_cond = controller.controller_act_cond
                shm_name_states = controller.shm_states.name
                shm_name_actions = controller.shm_actions.name
                shm_name_num_nodes = controller.shm_num_nodes.name
                shm_name_num_edges = controller.shm_num_edges.name
                shm_name_num_active_action_ids = controller.shm_num_active_action_ids.name
                shm_name_active_action_ids = controller.shm_active_action_ids.name

                def creator(i, ienv=ienv, em=em):
                    env = VcmiEnv(
                        **dict(em.kwargs, seed=seed + i),
                        opponent="OTHER_ENV",
                        vcmienv_logtag=f"{logprefix}env.{ienv}.{i}"
                    )

                    env = DualEnvWrapper(
                        env,
                        env_id=i,
                        num_envs=em.num,
                        node_names=node_names,
                        edge_keys=edge_keys,
                        controller_env_cond=controller_env_cond,
                        controller_act_cond=controller_act_cond,
                        shm_name_states=shm_name_states,
                        shm_name_actions=shm_name_actions,
                        shm_name_num_nodes=shm_name_num_nodes,
                        shm_name_num_edges=shm_name_num_edges,
                        shm_name_num_active_action_ids=shm_name_num_active_action_ids,
                        shm_name_active_action_ids=shm_name_active_action_ids,
                        shm_names_nodes=shm_names_nodes,
                        shm_names_edges=shm_names_edges,
                        logprefix=logprefix,
                    )
                    return env
            else:
                raise RuntimeError(f"Unknown em.type: {em.type}")

            env_creators.extend([partial(creator, i) for i in range(em.num)])

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

        funcs = [partial(env_creator_wrapper, env_creator) for env_creator in env_creators]
        super().__init__(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)  # type: ignore[arg-type]

    def _raise_if_controller_thread_failed(self):
        if self.controller:
            self.controller.raise_if_thread_failed()

    def reset(self, *args, **kwargs):
        self._raise_if_controller_thread_failed()
        result = super().reset(*args, **kwargs)
        self._raise_if_controller_thread_failed()
        return result

    def step(self, *args, **kwargs):
        self._raise_if_controller_thread_failed()
        result = super().step(*args, **kwargs)
        self._raise_if_controller_thread_failed()
        return result

    def reload_model(self, *args, **kwargs):
        self._raise_if_controller_thread_failed()
        if self.controller:
            self.controller.reload_model(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.controller and "terminate" not in kwargs:
            kwargs["terminate"] = True

        try:
            super().close(*args, **kwargs)
        finally:
            if self.controller:
                self.controller.close()


def main():
    import json
    from rl.v15.gnn_model import GNNModel

    class TestPPOModel(torch.nn.Module):
        def __init__(self, node_types, edge_types, config, device):
            super().__init__()
            self.model = GNNModel(node_types, edge_types, config)
            self.device = device
            self.to(device)

        def forward_policy(self, *args, **kwargs):
            return self.model.forward_policy(*args, **kwargs)

        def forward_value(self, *args, **kwargs):
            return self.model.forward_value(*args, **kwargs)

    class TestDNAModel(torch.nn.Module):
        def __init__(self, node_types, edge_types, config, device):
            super().__init__()
            self.model_policy = GNNModel(node_types, edge_types, config)
            self.model_value = GNNModel(node_types, edge_types, config)
            self.device = device
            self.to(device)

    class TestModelLoader(AbstractModelLoader):
        def __init__(self):
            self.model = None

        def configure(self, config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        def _ignored_edges(self):
            return self.config.get("train", {}).get("env", {}).get("kwargs", {}).get("ignored_edges", [])

        # This function will be called from within another process
        # (referenced non-local objects must be serializable for IPC).
        def load(self, weights_file):
            print(f"[TestModelLoader] Loading model weights from {weights_file}")
            with torch.inference_mode():
                weights = torch.load(weights_file, weights_only=True, map_location="cpu")
                node_types = VcmiEnv.node_types()
                edge_types = VcmiEnv.filtered_edge_types(self._ignored_edges())
                device = torch.device("cpu")

                if any(k.startswith("model_policy.") for k in weights):
                    self.model = TestDNAModel(node_types, edge_types, self.config["model"], device).eval()
                elif any(k.startswith("model.") for k in weights):
                    self.model = TestPPOModel(node_types, edge_types, self.config["model"], device).eval()
                else:
                    self.model = GNNModel(node_types, edge_types, self.config["model"]).eval()

                self.model.load_state_dict(weights, strict=True)

        def get_model(self):
            return self.model

    model_loader = TestModelLoader()
    model_loader.configure("export/pdpyqkrb-202707290651-config.json")
    model_loader.load("export/pdpyqkrb-202707290651-model-ppo.pt")
    env_kwargs = dict(mapname="gym/ml-mini.vmap")

    venv = DualVecEnv(
        seed=42,
        env_metas=[
            EnvMeta(type="StupidAI", num=0, kwargs={}),
            EnvMeta(type="BattleAI", num=0, kwargs={}),
            EnvMeta(type="VIPBot", num=1, kwargs={}),
            EnvMeta(type="onnx_model", num=0, kwargs=dict(env_kwargs, mapname="gym/ml-mini.vmap", opponent_model="simo.onnx")),
            EnvMeta(type="torch_model", num=0, kwargs=env_kwargs, model_loader=model_loader),
        ],
        logprefix="test-",
    )

    import ipdb; ipdb.set_trace()  # noqa
    pass

if __name__ == "__main__":
    main()
