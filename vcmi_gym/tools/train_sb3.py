from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import safe_mean
import os
import stable_baselines3
import sb3_contrib
import math

from . import common
from .. import InfoDict, VcmiCNN


class LogCallback(BaseCallback):
    """Logs user-defined `info` values into tensorboard"""
    def __init__(self):
        super().__init__()
        self.rollout_episodes = 0
        self.rollouts = 0

    def _on_step(self):
        self.rollout_episodes += self.locals["dones"].sum()

    def _on_rollout_end(self):
        self.rollouts += 1
        self.rollout_episodes = 0

        if self.rollouts % self.locals["log_interval"] != 0:
            return

        for k in InfoDict.SCALAR_VALUES:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"{k}", v)


def init_model(
    venv,
    seed,
    model_load_file,
    model_load_update,
    learner_cls,
    learner_kwargs,
    learning_rate,
    vcmi_cnn_kwargs,
    log_tensorboard,
    out_dir,
):
    alg = None

    match learner_cls:
        case "PPO":
            alg = stable_baselines3.PPO
        case "MPPO":
            alg = sb3_contrib.MaskablePPO
        case "QRDQN":
            alg = sb3_contrib.QRDQN
        case _:
            raise Exception("Unexpected learner_cls: %s" % learner_cls)

    model = None

    if model_load_file:
        print("Loading %s model from %s" % (alg.__name__, model_load_file))
        model = alg.load(model_load_file, env=venv)
    else:
        kwargs = dict(learner_kwargs, learning_rate=learning_rate, seed=seed)

        if vcmi_cnn_kwargs:
            kwargs["policy"] = "CnnPolicy"
            kwargs["policy_kwargs"] = dict(
                features_extractor_class=VcmiCNN,
                features_extractor_kwargs=vcmi_cnn_kwargs,
            )

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
    vcmi_cnn_kwargs,
    total_timesteps,
    n_checkpoints,
    n_envs,
    out_dir_template,
    log_tensorboard,
    progress_bar,
    reset_num_timesteps,
):
    venv = create_vec_env(seed, n_envs)

    try:
        out_dir = common.out_dir_from_template(out_dir_template, seed, run_id)
        learning_rate = common.lr_from_schedule(learner_lr_schedule)

        model = init_model(
            venv=venv,
            seed=seed,
            model_load_file=model_load_file,
            model_load_update=model_load_update,
            learner_cls=learner_cls,
            learner_kwargs=learner_kwargs,
            learning_rate=learning_rate,
            vcmi_cnn_kwargs=vcmi_cnn_kwargs,
            log_tensorboard=log_tensorboard,
            out_dir=out_dir,
        )

        callbacks = [LogCallback()]

        if n_checkpoints > 0:
            every = math.ceil(total_timesteps / n_checkpoints)
            print(f"Saving every {every} into {out_dir}")

            callbacks.append(CheckpointCallback(
                save_freq=every,
                save_path=out_dir,
                name_prefix="model",
            ))

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            callback=callbacks
        )

        return {"out_dir": out_dir}
    finally:
        venv.close()
