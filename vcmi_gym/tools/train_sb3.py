from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
import re
import gymnasium as gym
import stable_baselines3
import sb3_contrib
import vcmi_gym
import itertools
import wandb
import copy
import torch.optim

from . import common
from . import sb3_callback
from .. import InfoDict


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
        case "MQRDQN":
            alg = vcmi_gym.MaskableQRDQN
        case _:
            raise Exception("Unexpected learner_cls: %s" % learner_cls)

    model = None

    alg_kwargs = copy.deepcopy(learner_kwargs)
    alg_kwargs["learning_rate"] = learning_rate

    policy_kwargs = alg_kwargs.get("policy_kwargs", None)
    if policy_kwargs:
        fecn = policy_kwargs.get("features_extractor_class_name", None)
        if fecn:
            del policy_kwargs["features_extractor_class_name"]
            policy_kwargs["features_extractor_class"] = getattr(vcmi_gym, fecn)

        ocn = policy_kwargs.get("optimizer_class_name", None)
        if ocn:
            del policy_kwargs["optimizer_class_name"]
            policy_kwargs["optimizer_class"] = getattr(torch.optim, ocn)

    print("Learner kwargs: %s" % alg_kwargs)

    if model_load_file:
        print("Loading %s model from %s" % (alg.__name__, model_load_file))
        model = alg.load(model_load_file, env=venv, **alg_kwargs)
    else:
        model = alg(env=venv, **alg_kwargs)

    if log_tensorboard:
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

def create_venv(role, mpool, offset, run_id, rollouts, rollouts_per_map):
    env_id = "local/VCMI-v0"
    env_kwargs = dict(attacker="StupidAI", defender="StupidAI")
    env_kwargs[role] = "MMAI_USER"

    mid = (offset + rollouts // rollouts_per_map) % len(mpool)
    env_kwargs["mapname"] = "ai/generated/%s" % (mpool[mid])
    mapnum = int(re.match(r".+?([0-9]+)\.vmap", env_kwargs["mapname"]).group(1))

    env_kwargs["actions_log_file"] = f"/tmp/{run_id}-actions.log"

    wandb.log(
        # 0=attacker, 1=defender
        {"mapnum": mapnum, "role": ["attacker", "defender"].index(role)},
        commit=False
    )

    # XXX: not wrapping in TimeLimit, as it gets applied AFTER Monitor
    # => there will be no info["episode"] in case of truncations
    # => metrics won't be correctly calculated
    # => implement truncation in the env itself
    return make_vec_env(
        env_id,
        n_envs=1,
        env_kwargs=env_kwargs,
        monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
    )


def train_sb3(
    learner_cls,
    seed,
    run_id,
    group_id,
    model_load_file,
    model_load_update,
    learner_kwargs,
    learning_rate,
    learner_lr_schedule,
    vcmi_cnn_kwargs,
    rollouts_total,
    rollouts_per_map,
    rollouts_per_role,
    rollouts_per_log,
    map_pool,
    map_pool_offset_idx,
    n_envs,
    out_dir,
    log_tensorboard,
    progress_bar,
    reset_num_timesteps,
    config_log,
):
    # Ensure both roles get equal amount of rollouts
    assert rollouts_total % rollouts_per_role == 0
    assert (rollouts_total // rollouts_per_role) % 2 == 0

    # Ensure there's equal amount of logs for each role and map
    assert rollouts_per_role % rollouts_per_log == 0
    assert rollouts_total % rollouts_per_map == 0
    assert rollouts_per_map % rollouts_per_role == 0
    assert rollouts_per_map // rollouts_per_role % 2 == 0

    assert map_pool, "no map pool given"

    # prevent warnings for action_masks method
    gym.logger.set_level(gym.logger.ERROR)

    learning_rate = common.lr_from_schedule(learner_lr_schedule)
    sb3_cb = sb3_callback.SB3Callback()
    roles = itertools.cycle(["attacker", "defender"])
    ep_rew_means = []
    rollouts = 0

    if rollouts_total == 0:
        rollouts_total = 10**9

    model = None

    try:
        model = init_model(
            venv=create_venv(next(roles), map_pool, map_pool_offset_idx, run_id, rollouts, rollouts_per_map),
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

        wandb.log(config_log, commit=False)

        metric_log = dict((v, 0) for v in InfoDict.SCALAR_VALUES)
        metric_log["rollout/ep_rew_mean"] = 0
        metric_log["rollout/ep_len_mean"] = 0
        model.logger.record("config", logger.HParam(config_log, metric_log))

        while rollouts < rollouts_total:
            print(".", end="", flush=True)
            wandb.log({"iterations": rollouts // rollouts_per_role}, commit=False)

            if rollouts > 0:
                common.save_model(out_dir, model)
                model.env.close()
                model.env = create_venv(next(roles), map_pool, map_pool_offset_idx, run_id, rollouts, rollouts_per_map)
                model.env.reset()

            model.learn(
                total_timesteps=rollouts_per_role * model.n_steps,
                log_interval=rollouts_per_log,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
                callback=[sb3_cb]
            )

            diff_rollouts = sb3_cb.rollouts - rollouts
            assert diff_rollouts == rollouts_per_role, f"expected {rollouts_per_role}, got: {diff_rollouts}"

            rollouts += rollouts_per_role
            ep_rew_means.append(sb3_cb.ep_rew_mean)

        return {"out_dir": out_dir}
    finally:
        if model and model.env:
            model.env.close()
        wandb.finish(quiet=True)
