from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import safe_mean
import os
import math
import stable_baselines3
import sb3_contrib
import numpy as np
import importlib

from . import common
from .. import InfoDict


class LogCallback(BaseCallback):
    """Logs user-defined `info` values into tensorboard"""
    def __init__(self, wandb_run=None):
        super().__init__()
        self.rollout_episodes = 0
        self.rollouts = 0

        # batch table logging to avoid hundreds of files
        self.tablebatch = 0
        self.wdb_tables = {}

        if wandb_run:
            self.wandb = importlib.import_module("wandb")
        else:
            self.wandb = None

    def _on_step(self):
        self.rollout_episodes += self.locals["dones"].sum()

    def _on_rollout_end(self):
        self.rollouts += 1
        wdb_log = {"rollout/n_episodes": self.rollout_episodes}
        self.rollout_episodes = 0

        if self.rollouts % self.locals["log_interval"] != 0:
            return

        for k in InfoDict.SCALAR_VALUES:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"{k}", v)
            wdb_log[k] = v

        # From here on it's W&B stuff only
        if not self.wandb:
            return

        wdb_log["num_timesteps"] = self.num_timesteps
        wdb_log["rollout/count"] = self.rollouts

        # Also add sb3's Monitor info keys: "r" (reward) and "l" (length)
        # (these are already recorded to TB by sb3, but not in W&B)
        v = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_rew_mean"] = v

        v = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_len_mean"] = v

        for (k, columns) in InfoDict.D1_ARRAY_VALUES.items():
            action_types_vec_2d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
            ary = np.mean(action_types_vec_2d, axis=0)

            # In SB3's logger, Tensor objects are logged as a Histogram
            # https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/logger.py#L412
            # NOT logging this to TB, it's not visualized well there
            # tb_data = torch.as_tensor(ary)
            # self.model.logger.record(f"user/{k}", tb_data)

            # In W&B, we need to unpivot into a name/count table
            # NOTE: reserved column names: "id", "name", "_step" and "color"
            wk = f"table/{k}"
            rotated = [list(row) for row in zip(columns, ary)]
            if wk not in self.wdb_tables:
                self.wdb_tables[wk] = self.wandb.Table(columns=["key", "value"])

            wb_table = self.wdb_tables[wk]
            for row in rotated:
                wb_table.add_data(*row)

        for k in InfoDict.D2_ARRAY_VALUES:
            action_types_vec_3d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
            ary_2d = np.mean(action_types_vec_3d, axis=0)

            wk = f"table/{k}"
            if wk not in self.wdb_tables:
                # Also log the "rollout" so that inter-process logs (which are different _step)
                # can be aggregated if needed
                self.wdb_tables[wk] = self.wandb.Table(columns=["x", "y", "value"])

            wb_table = self.wdb_tables[wk]

            for (y, row) in enumerate(ary_2d):
                for (x, cell) in enumerate(row):
                    wb_table.add_data(x, y, cell)

        self.tablebatch += 1
        if self.tablebatch % 100 == 0:
            wdb_log = dict(wdb_log, **self.wdb_tables)
            self.wdb_tables = {}

        # Make sure to log just once to prevent incrementing "step"
        self.wandb.log(wdb_log)


def init_model(
    venv,
    seed,
    model_load_file,
    model_load_update,
    model_load_checkpoint,
    learner_cls,
    learner_kwargs,
    learning_rate,
    log_tensorboard,
    out_dir,
):
    alg = None

    match learner_cls:
        case "PPO":
            alg = stable_baselines3.PPO
        case "QRDQN":
            alg = sb3_contrib.QRDQN
        case _:
            raise Exception("Unexpected learner_cls: %s" % learner_cls)

    model = None

    if model_load_file:
        print("Loading %s model from %s" % (alg.__name__, model_load_file))
        model = alg.load(model_load_file, env=venv)
    elif model_load_checkpoint:
        with model_load_checkpoint.as_directory() as checkpoint_dir:
            f = os.path.join(checkpoint_dir, "model.zip")
            # print("Loading %s model from checkpoint file: %s" % (alg.__name__, f))
            kwargs = dict(learner_kwargs, learning_rate=learning_rate, seed=seed)
            model = alg.load(f, env=venv, **kwargs)
            # print("<train_sb3> Loading model from %s with kwargs: %s. New lr: %s" % (f, kwargs, model.learning_rate))
    else:
        kwargs = dict(learner_kwargs, learning_rate=learning_rate, seed=seed)
        # print("------------------ 2: %s" % kwargs)
        model = alg(env=venv, **kwargs)

    if log_tensorboard:
        os.makedirs(out_dir, exist_ok=True)
        log = logger.configure(folder=out_dir, format_strings=["tensorboard"])
        model.set_logger(log)

    return model


#
# A note about tensorboard logging of user-defined values in `info`:
#
# On each step, if env is done, Monitor wrapper will read `info_keywords`
# from `info` and copy them into `info["episode"]`:
# https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/monitor.py#L103
#
# Then, on each step, SB3 algos (PPO/DQN/...) put all `info["episode"]`
# dicts from the vec_env's step into `ep_info_buffer`:
# https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/base_class.py#L441
#
# Then, on each Nth rollout (and *after* .on_rollout_end() is called),
# SB3 algos compute the rollout/* metrics from this buffer:
# https://github.com/DLR-RM/stable-baselines3/blob/v2.2.1/stable_baselines3/common/on_policy_algorithm.py#L292
# (TODO: off-policy algos may behave differently)
#
# SB3 then dumps the log (writing to TB, if enabled)
#
# This buffer can also be accessed custom in callbacks
# That's how user-defined values in `info` (set by VcmiEnv) can be
# logged into tensorboard (ie. into model's Logger)
#
#
#
# Here is a full timeline for ON_POLICY algos:
#
#     while num_timesteps < cfg["total_timesteps"]:
#         # CALL cb.on_rollout_start()
#         while n_steps < cfg["n_steps"]:  # collect_rollouts
#             infos = []
#
#             for env in vec_env.envs:
#                 info = env.step()     # <--- our env's info!
#                 if term or trunc:
#                     info["episode"] = {...}
#                 infos.append(info)
#
#             # CALL cb.on_step()
#
#             for info in infos:
#                 if info["episode"]:
#                     self.ep_info_buffer.extend([info])
#         # CALL cb.on_rollout_end()
#
#         if iteration % cfg["log_interval"] == 0:
#             self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] ...])
#             self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] ...])
#             self.logger.dump(step=self.num_timesteps)
#
#         self.train()  # update NN

def create_vec_env(seed, n_envs):
    # XXX: not wrapping in TimeLimit, as it gets applied AFTER Monitor
    # => there will be no info["episode"] in case of truncations
    # => metrics won't be correctly calculated
    # => implement truncation in the env itself
    venv = make_vec_env(
        "local/VCMI-v0",
        n_envs=n_envs,
        env_kwargs={"seed": seed},
        monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
    )

    return venv


def train_sb3(
    learner_cls,
    seed,
    run_id,
    model_load_file,
    model_load_update,
    learner_kwargs,
    learning_rate,
    learner_lr_schedule,
    total_timesteps,
    n_checkpoints,
    n_envs,
    out_dir_template,
    log_tensorboard,
    progress_bar,
    reset_num_timesteps,
    extras,
):
    venv = create_vec_env(seed, n_envs)

    try:
        out_dir = common.out_dir_from_template(out_dir_template, seed, run_id, extras is not None)

        if learner_lr_schedule:
            assert learning_rate is None, "both learner_lr_schedule and learning_rate given"
            learning_rate = common.lr_from_schedule(learner_lr_schedule)
        else:
            assert learning_rate is not None, "neither learner_lr_schedule nor learning_rate given"

        model = init_model(
            venv=venv,
            seed=seed,
            model_load_file=model_load_file,
            model_load_update=model_load_update,
            model_load_checkpoint=extras.get("checkpoint", None),
            learner_cls=learner_cls,
            learner_kwargs=learner_kwargs,
            learning_rate=learning_rate,
            log_tensorboard=log_tensorboard,
            out_dir=out_dir,
        )

        callbacks = [LogCallback(wandb_run=extras.get("wandb_run", None))]

        if n_checkpoints > 0:
            callbacks.append(CheckpointCallback(
                save_freq=math.ceil(total_timesteps / n_checkpoints),
                save_path=out_dir,
                name_prefix="model",
            ))

        if "train_sb3.callback" in extras:
            callbacks.append(extras["train_sb3.callback"])

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            callback=callbacks
        )

        return {"out_dir": out_dir}
    finally:
        venv.close()
