# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# from stable_baselines3.common.env_util import make_vec_env
import random
import glob
import gymnasium as gym
import stable_baselines3
import sb3_contrib
import vcmi_gym
import wandb
import copy
import torch.optim
import threading

from stable_baselines3.common import logger
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import VecFrameStack

from . import common
from . import sb3_callback
from .. import InfoDict

DEBUGGING = False


def init_model(
    venv,
    seed,
    features_extractor_load_file,
    features_extractor_load_file_type,
    features_extractor_freeze,
    model_load_file,
    model_load_update,
    learner_cls,
    learner_kwargs,
    learning_rate,
    net_arch,
    activation,
    n_global_steps_max,
    n_envs,
    features_extractor,
    lstm,
    optimizer,
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
        case "VPPO":
            alg = vcmi_gym.VcmiPPO
        case _:
            raise Exception("Unexpected learner_cls: %s" % learner_cls)

    model = None

    alg_kwargs = copy.deepcopy(learner_kwargs)
    alg_kwargs["learning_rate"] = learning_rate

    if n_global_steps_max:
        alg_kwargs["n_steps"] = n_global_steps_max // n_envs

    if model_load_file:
        print("Learner kwargs: %s" % alg_kwargs)
        print("Loading %s model from %s" % (alg.__name__, model_load_file))
        model = alg.load(model_load_file, env=venv, **alg_kwargs)
        if seed:
            model.set_random_seed(seed)
    else:
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": getattr(torch.nn, activation),
        }

        # Any custom features extractor is assumed to be a VcmiCNN-type policy
        if features_extractor:
            policy_kwargs["features_extractor_class"] = getattr(vcmi_gym, features_extractor["class_name"])
            policy_kwargs["features_extractor_kwargs"] = features_extractor["kwargs"]

        if optimizer:
            policy_kwargs["optimizer_class"] = getattr(torch.optim, optimizer["class_name"])
            policy_kwargs["optimizer_kwargs"] = optimizer["kwargs"]

        if learner_cls == "VPPO":
            alg_kwargs["policy"] = "VcmiPolicy"
            policy_kwargs = dict(policy_kwargs, **lstm)
        else:
            alg_kwargs["policy"] = "MlpPolicy"

        alg_kwargs["policy_kwargs"] = policy_kwargs

        print("Learner kwargs: %s" % alg_kwargs)
        print("Initializing %s model from scratch" % alg.__name__)
        model = alg(env=venv, **alg_kwargs)

    if features_extractor_load_file:
        if features_extractor_load_file_type == "sb3":
            print("Loading features extractor (sb3 model) from %s" % features_extractor_load_file)
            _data, params, _pytorch_variables = load_from_zip_file(features_extractor_load_file)
            prefix = "features_extractor."
            features_extractor_params = dict(
                (k.removeprefix(prefix), v) for (k, v) in params["policy"].items() if k.startswith(prefix)
            )
        elif features_extractor_load_file_type == "params":
            print("Loading features extractor (torch params) from %s" % features_extractor_load_file)
            autoencoder_encoder_params = torch.load(features_extractor_load_file)
            features_extractor_params = dict(
                (f"network.{k}", v) for (k, v) in autoencoder_encoder_params.items()
            )
        elif features_extractor_load_file_type == "model":
            print("Loading features extractor (torch model) from %s" % features_extractor_load_file)
            model = torch.load(features_extractor_load_file)
            features_extractor_params = model.state_dict()
        else:
            raise Exception("Unexpected features_extractor_load_file_type: %s" % features_extractor_load_file_type)

        model.policy.features_extractor.load_state_dict(features_extractor_params, strict=True)

        if features_extractor_freeze:
            print("Freezing features extractor...")
            # Freeze the parameters of the features extractor
            for param in model.policy.features_extractor.parameters():
                param.requires_grad = False

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

def create_venv(n_envs, env_cls, framestack, env_kwargs, mapmask, randomize, run_id, model_file, iteration=0, wrappers=[]):
    mappath = "/Users/simo/Library/Application Support/vcmi/Maps"
    all_maps = glob.glob("%s/%s" % (mappath, mapmask))
    all_maps = [m.replace("%s/" % mappath, "") for m in all_maps]
    all_maps.sort()

    if n_envs == 1:
        n_maps = 1
    else:
        assert n_envs % 2 == 0
        assert n_envs <= len(all_maps) * 2
        n_maps = n_envs // 2

    if randomize:
        maps = random.sample(all_maps, n_maps)
    else:
        i = (n_maps * iteration) % len(all_maps)
        new_i = (i + n_maps) % len(all_maps)
        if not DEBUGGING:
            wandb.log({"map_offset": i}, commit=False)

        if new_i > i:
            maps = all_maps[i:new_i]
        else:
            maps = all_maps[i:] + all_maps[:new_i]

        assert len(maps) == n_maps

    pairs = [[("attacker", m), ("defender", m)] for m in maps]
    pairs = [x for y in pairs for x in y]  # aka. pairs.flatten(1)...
    state = {"n": 0}
    lock = threading.RLock()

    def env_creator(**_env_kwargs):
        with lock:
            assert state["n"] < n_envs
            role, mapname = pairs[state["n"]]
            # logfile = f"/tmp/{run_id}-env{state['n']}-actions.log"
            logfile = None

            env_kwargs2 = dict(
                env_kwargs,
                mapname=mapname,
                attacker="StupidAI",
                defender="StupidAI",
                actions_log_file=logfile
            )

            if model_file:
                env_kwargs2.update({"attacker": "MMAI_MODEL", "attacker_model": model_file, "defender": "MMAI_MODEL", "defender_model": model_file})

            env_kwargs2[role] = "MMAI_USER"
            print("Env kwargs (env.%d): %s" % (state["n"], env_kwargs2))
            state["n"] += 1

        return env_cls(**env_kwargs2)

    venv = common.make_vec_env_parallel(
        min(n_envs, 8),
        env_creator,
        n_envs=n_envs,
        monitor_kwargs={"info_keywords": InfoDict.ALL_KEYS},
        wrappers=wrappers,
    )

    if framestack > 1:
        venv = VecFrameStack(venv, n_stack=framestack, channels_order="first")

    return venv


def train_sb3(
    learner_cls,
    seed,
    run_id,
    group_id,
    features_extractor_load_file,
    features_extractor_load_file_type,
    features_extractor_freeze,
    model_load_file,
    model_load_update,
    iteration,
    learner_kwargs,
    learning_rate,
    learner_lr_schedule,
    net_arch,
    activation,
    features_extractor,
    lstm,
    optimizer,
    env_cls_name,
    env_kwargs,
    mapmask,
    randomize_maps,
    n_global_steps_max,
    rollouts_total,
    rollouts_per_iteration,
    rollouts_per_log,
    n_envs,
    framestack,
    save_every,
    max_saves,
    out_dir,
    observations_dir,
    log_tensorboard,
    progress_bar,
    reset_num_timesteps,
    config_log,
    opponent_model_dir_pattern,
    success_rate_target,
    wrappers=[],
):

    # prevent warnings for action_masks method
    gym.logger.set_level(gym.logger.ERROR)

    learning_rate = common.lr_from_schedule(learner_lr_schedule)
    sb3_cb = sb3_callback.SB3Callback(
        observations_dir,
        success_rate_target=success_rate_target,
        debugging=DEBUGGING
    )

    ep_rew_means = []
    env_cls = getattr(vcmi_gym, env_cls_name)

    if rollouts_total:
        iterations = rollouts_total // rollouts_per_iteration
    else:
        iterations = 10**9

    model = None
    t = None
    start_iteration = iteration
    opponent_model = None

    if opponent_model_dir_pattern:
        # find_models("data/sparse-rewards/mult-sanity2-lr0.00001-target0.7-*")
        models = list(common.find_models(opponent_model_dir_pattern).values())
        # https://stackoverflow.com/a/74033260
        opponent_model, = models[0:] or [model_load_file]

        # XXX: If opponent_model=None, it is most likely a human error
        # Without this assert, training will be done vs. StupidAI
        assert opponent_model is not None, f"bad opponent_model_dir_pattern: {opponent_model_dir_pattern}"

    try:
        model = init_model(
            venv=create_venv(n_envs, env_cls, framestack, env_kwargs, mapmask, randomize_maps, run_id, opponent_model, iteration, wrappers),  # noqa: E501
            seed=seed,
            features_extractor_load_file=features_extractor_load_file,
            features_extractor_load_file_type=features_extractor_load_file_type,
            features_extractor_freeze=features_extractor_freeze,
            model_load_file=model_load_file,
            model_load_update=model_load_update,
            learner_cls=learner_cls,
            learner_kwargs=learner_kwargs,
            learning_rate=learning_rate,
            net_arch=net_arch,
            activation=activation,
            n_global_steps_max=n_global_steps_max,
            n_envs=n_envs,
            features_extractor=features_extractor,
            lstm=lstm,
            optimizer=optimizer,
            log_tensorboard=log_tensorboard,
            out_dir=out_dir,
        )

        if not DEBUGGING:
            wandb.log(config_log, commit=False)
            wandb.watch(model.policy, log="all")

        # metric_log = dict((v, 0) for v in InfoDict.SCALAR_VALUES)
        # metric_log["rollout/ep_rew_mean"] = 0
        # metric_log["rollout/ep_len_mean"] = 0
        # model.logger.record("config", logger.HParam(config_log, metric_log))

        steps_per_rollout = model.train_freq.frequency if learner_cls in ["QRDQN", "MQRDQN"] else model.n_steps
        total_timesteps_per_iteration = rollouts_per_iteration * steps_per_rollout * n_envs

        while iteration < iterations:
            print(".", end="", flush=True)

            if not DEBUGGING:
                wandb.log({"iteration": iteration}, commit=False)

            if (iteration - start_iteration) > 0:
                common.save_model(out_dir, model)
                with open(f"{out_dir}/iteration", "w") as f:
                    f.write(str(iteration))
                model.env.close()

                if opponent_model_dir_pattern:
                    models = list(common.find_models(opponent_model_dir_pattern).values())
                    opponent_model, = models[0:] or [model.env.opponent_model]

                model.env = create_venv(n_envs, env_cls, framestack, env_kwargs, mapmask, randomize_maps, run_id, opponent_model, iteration, wrappers)  # noqa: E501
                model.env.reset()
                sb3_cb.success_rates.clear()
                sb3_cb.this_env_rollouts = 0

            model.learn(
                total_timesteps=total_timesteps_per_iteration,
                log_interval=rollouts_per_log,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
                callback=[sb3_cb]
            )

            diff_rollouts = sb3_cb.rollouts - (iteration - start_iteration)*rollouts_per_iteration
            if not success_rate_target:
                assert diff_rollouts == rollouts_per_iteration, f"expected {rollouts_per_iteration}, got: {diff_rollouts}"  # noqa: E501
            iteration += 1
            ep_rew_means.append(sb3_cb.ep_rew_mean)
            t = common.maybe_save(t, model, out_dir, save_every, max_saves)

        return {"out_dir": out_dir}
    finally:
        if model:
            try:
                common.save_model(out_dir, model)
            except Exception as e:
                print("Failed to save model: %s" % e.str())
                pass

            if model.env:
                model.env.close()


if __name__ == "__main__":
    DEBUGGING = True

    train_sb3(
        learner_cls="MPPO",
        seed=42,
        run_id="debug-sb3",
        group_id="debug-sb3",
        features_extractor_load_file=None,
        features_extractor_load_file_type="params",
        features_extractor_freeze=False,
        model_load_file="debugging/session2/model-sb3.pt",
        # model_load_file=None,
        model_load_update=False,
        iteration=0,
        learner_kwargs=dict(
            batch_size=2,
            n_epochs=2,
            gamma=0.8,
            gae_lambda=0.8,
            clip_range=0.4,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        ),
        learning_rate=None,
        learner_lr_schedule="const_0.001",
        net_arch=[],
        activation="LeakyReLU",
        features_extractor=dict(
            class_name="VcmiFeaturesExtractor",
            kwargs=dict(
                layers=[
                    dict(t="Conv2d", out_channels=32, kernel_size=[1, 56], stride=[1, 56], padding=0),
                    dict(t="BatchNorm2d", num_features=32),
                    dict(t="LeakyReLU"),
                    dict(t="Flatten"),
                    dict(t="Linear", in_features=5280, out_features=1024),
                    dict(t="LeakyReLU"),
                ]
            )
        ),
        lstm={},
        optimizer=dict(class_name="AdamW", kwargs=dict(eps=1e-5, weight_decay=0)),
        env_cls_name="VcmiEnv",
        env_kwargs=dict(
            max_steps=500,
            reward_clip_mod=None,
            reward_dmg_factor=5,
            vcmi_loglevel_global="error",
            vcmi_loglevel_ai="error",
            vcmienv_loglevel="WARN",
            consecutive_error_reward_factor=-1,
            sparse_info=True,
            step_reward_mult=1,
            term_reward_mult=0,
        ),
        mapmask="ai/P1.vmap",
        randomize_maps=False,
        n_global_steps_max=4,
        rollouts_total=1000000,
        rollouts_per_iteration=1000,
        rollouts_per_log=1000,
        n_envs=1,
        framestack=1,
        save_every=2000000000,  # greater than time.time()
        max_saves=3,
        out_dir="data/debug-sb3/debug-sb3",
        observations_dir=None,
        log_tensorboard=False,
        progress_bar=False,
        reset_num_timesteps=True,
        config_log={},
        opponent_model_dir_pattern=None,
        success_rate_target=None,
        wrappers=[dict(module="debugging.defend_wrapper", cls="DefendWrapper")],
    )
