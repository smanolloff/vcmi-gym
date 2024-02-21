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

import os
import yaml
import re
import argparse
import signal
import sys
# import wandb
from copy import deepcopy

from . import common

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


def handle_signal(signum, frame):
    print("*** [main.py] received signal %s ***" % signum)
    sys.exit(0)


def run(action, cfg, group_id, run_id, model_load_file, iteration, rest=[]):
    # print("**** ENV WANDB_RUN_ID: %s" % os.environ["WANDB_RUN_ID"])
    # import wandb
    # print("**** wandb.run.id: %s" % wandb.run.id)

    cwd = os.getcwd()
    env_wrappers = cfg.pop("env_wrappers", {})
    env_kwargs = cfg.pop("env_kwargs", {})

    match action:
        case "train_ppo" | "train_qrdqn" | "train_mppo" | "train_mqrdqn" | "train_vppo":
            from .train_sb3 import train_sb3
            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            # common.register_env(expanded_env_kwargs, env_wrappers)

            learner_cls = action.split("_")[-1].upper()
            default_template = "data/%s-{group_id}/{run_id}" % learner_cls
            out_dir_template = cfg.get("out_dir_template", default_template)
            seed = cfg.get("seed", None) or common.gen_seed()

            if group_id is None:
                group_id = cfg.get("group_id", None) or common.gen_id()

            if run_id is None:
                run_id = cfg.get("run_id", None) or group_id

            if model_load_file is None:
                model_load_file = cfg.get("model_load_file", None)

            if iteration is None:
                iteration = cfg.get("iteration", None) or 0

            features_extractor_load_file = cfg.get("features_extractor_load_file", False)
            features_extractor_load_file_type = cfg.get("features_extractor_load_file_type", "sb3")
            features_extractor_freeze = cfg.get("features_extractor_freeze", False)

            if model_load_file is not None:
                # Possibly a mistake (or watchdog-continued run)
                # assert features_extractor_load_file, "loading features_extractor supported for new models only"
                if features_extractor_load_file:
                    print("\n\n\n****** WARNING ******")
                    print("ignoring `features_extractor_load_file` because model_load_file is given")
                    features_extractor_load_file = None

            assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]+[A-Za-z0-9]$", group_id), "invalid group_id: %s" % group_id

            out_dir = out_dir_template.format(seed=seed, group_id=group_id, run_id=run_id)
            print("Output dir: %s" % out_dir)
            out_dir = common.make_absolute(cwd, out_dir)
            os.makedirs(out_dir, exist_ok=True)

            observations_dir = cfg.get("observations_dir", None)
            if observations_dir:
                os.makedirs(observations_dir, exist_ok=True)

            # learner_cls is not part of the config
            run_config = deepcopy(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "group_id": group_id,
                    "features_extractor_load_file": features_extractor_load_file,
                    "features_extractor_load_file_type": features_extractor_load_file_type,
                    "features_extractor_freeze": features_extractor_freeze,
                    "model_load_file": model_load_file,
                    "model_load_update": cfg.get("model_load_update", False),
                    "iteration": iteration,
                    "out_dir": out_dir,
                    "observations_dir": observations_dir,
                    "log_tensorboard": cfg.get("log_tensorboard", False),
                    "progress_bar": cfg.get("progress_bar", True),
                    "reset_num_timesteps": cfg.get("reset_num_timesteps", False),
                    "learner_kwargs": cfg.get("learner_kwargs", {}),
                    "net_arch": cfg.get("net_arch", []),
                    "activation": cfg.get("activation", "ReLU"),
                    "features_extractor": cfg.get("features_extractor", {}),
                    "lstm": cfg.get("lstm", {}),
                    "optimizer": cfg.get("optimizer", {}),
                    "env_cls_name": cfg.get("env_cls_name", "VcmiEnv"),
                    "env_kwargs": expanded_env_kwargs,
                    "mapmask": cfg.get("mapmask", "ai/generated/A*.vmap"),
                    "randomize_maps": cfg.get("randomize_maps", False),
                    "n_global_steps_max": cfg.get("n_global_steps_max", None),
                    "rollouts_total": cfg.get("rollouts_total", 0),
                    "rollouts_per_iteration": cfg.get("rollouts_per_iteration", 100),
                    "rollouts_per_log": cfg.get("rollouts_per_log", 5),
                    "n_envs": cfg.get("n_envs", 1),
                    "framestack": cfg.get("framestack", 1),
                    "save_every": cfg.get("save_every", 3600),
                    "max_saves": cfg.get("max_saves", 3),
                    "learning_rate": cfg.get("learning_rate", None),
                    "learner_lr_schedule": cfg.get(
                        "learner_lr_schedule", "const_0.003"
                    ),
                    "self_play": cfg.get("self_play", False),
                }
            )

            # env_kwargs should be logged
            all_cfg = dict(run_config, env_kwargs=env_kwargs)

            run_config["config_log"] = {}
            for (k, v) in cfg.get("logparams", {}).items():
                run_config["config_log"][k] = common.extract_dict_value_by_path(all_cfg, v) or "NULL"

            print("Starting run %s with seed %s" % (run_id, seed))

            os.environ["WANDB_SILENT"] = "true"
            notes = cfg.get("notes", None)
            common.wandb_init(run_id, group_id, notes, all_cfg)

            run_duration, run_values = common.measure(
                train_sb3, dict(run_config, learner_cls=learner_cls)
            )

            common.save_run_metadata(
                action=action,
                cfg=dict(run_config, env_kwargs=env_kwargs),
                duration=run_duration,
                values=dict(run_values, env=expanded_env_kwargs),
            )

        case "spectate":
            from .spectate import spectate
            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            common.register_env(expanded_env_kwargs, env_wrappers)

            if len(rest) > 0:
                cfg["mapname"] = rest[0]
            if len(rest) > 1:
                cfg["model_file"] = rest[1]

            spectate(
                fps=cfg.get("fps", 2),
                reset_delay=cfg.get("reset_delay", 5),
                mapname=cfg["mapname"],
                model_file=cfg["model_file"],
                model_mod=cfg.get("model_mod", "stable_baselines3"),
                model_cls=cfg.get("model_cls", "PPO"),
            )

        case "benchmark":
            from .benchmark import benchmark

            if len(rest) > 0:
                env_kwargs["mapname"] = rest[0]

            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            common.register_env(expanded_env_kwargs, env_wrappers)
            steps = cfg.get("steps", 10000)
            benchmark(steps)

        case "play":
            from .play import play

            if len(rest) > 0:
                env_kwargs["mapname"] = rest[0]

            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            common.register_env(expanded_env_kwargs, env_wrappers)
            play()

        case "test":
            from .test import test

            # Example:
            # python vcmi-gym.py test test/M6
            #
            # will open file "test/M6". File structure:
            # - comments: (anything starting with "#")
            # - first non-comment line: MAP location
            # - everything else: integer actions (as seen by BAI, ie. w/o offest)

            actions = []
            with open(os.path.join(rest[0]), "r") as f:
                actions = f.read()

            actions = [a.strip() for a in actions.split("\n") if a.strip()]
            comments = [a for a in actions if a.startswith("#")]
            actions = [a for a in actions if a not in comments]
            env_kwargs["mapname"] = actions[0]
            actions = [int(a) for a in actions[1:]]

            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            common.register_env(expanded_env_kwargs, env_wrappers)

            test(env_kwargs, actions)
            print("\nCommentary:\n%s" % "\n".join(comments))

        case _:
            print("Unknown action: %s" % action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help=argparse.SUPPRESS)
    parser.add_argument("-g", metavar="GROUP_ID", help="group_id")
    parser.add_argument("-r", metavar="RUN_ID", help="run_id")
    parser.add_argument("-l", metavar="LOADFILE", help="zip file to load model from")
    parser.add_argument("-i", metavar="ITERATION", type=int, help="iteration")
    parser.add_argument(
        "-c",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="config file, defaults to config/<action>.yml",
    )
    parser.add_argument('rest', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.usage = "%(prog)s [options] <action> [<value>]"
    parser.epilog = """
action:
  train_ppo         train using Proximal Policy Optimization (PPO)
  train_mppo        train using Maskable Proximal Policy Optimization (MPPO)
  train_qrdqn       train using Quantile Regression DQN (QRDQN)
  train_mqrdqn      train using Maskable Quantile Regression DQN (my impl)
  spectate          watch a trained model play VCMI
  benchmark [map]   evaluate the actions/s achievable with this env
  test [map]        for testing purposes only
  play [map]        play VCMI
  help              print this help message

examples:
  %(prog)s train_qrdqn
  %(prog)s -c path/to/config.yml train_qrdqn
"""

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)

    if args.c is None:
        args.c = open(os.path.join("config", f"{args.action}.yml"), "r")

    print("Loading configuration from %s" % args.c.name)
    cfg = yaml.safe_load(args.c)
    args.c.close()

    run(args.action, cfg, args.g, args.r, args.l, args.i, rest=args.rest)


if __name__ == "__main__":
    main()
