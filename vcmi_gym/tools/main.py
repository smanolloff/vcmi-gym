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


def run(action, cfg, rest=[]):
    # print("**** ENV WANDB_RUN_ID: %s" % os.environ["WANDB_RUN_ID"])
    # import wandb
    # print("**** wandb.run.id: %s" % wandb.run.id)

    cwd = os.getcwd()
    env_wrappers = cfg.pop("env_wrappers", {})
    env_kwargs = cfg.pop("env_kwargs", {})

    match action:
        case "train_ppo" | "train_qrdqn" | "train_mppo" | "train_mqrdqn":
            from .train_sb3 import train_sb3
            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            # common.register_env(expanded_env_kwargs, env_wrappers)

            learner_cls = action.split("_")[-1].upper()
            default_template = "data/%s-{group_id}/{run_id}" % learner_cls
            out_dir_template = cfg.get("out_dir_template", default_template)
            seed = cfg.get("seed", None) or common.gen_seed()
            run_id = cfg.get("run_id", None) or common.gen_id()
            group_id = cfg.get("group_id", None) or run_id
            model_load_file = cfg.get("model_load_file", None)

            if len(rest) > 0:
                group_id = rest[0]

            if len(rest) > 1:
                run_id = rest[1]

            if len(rest) > 2:
                model_load_file = rest[2]

            assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]+[A-Za-z0-9]$", group_id), "invalid group_id: %s" % group_id

            out_dir = out_dir_template.format(seed=seed, group_id=group_id, run_id=run_id)
            print("Output dir: %s" % out_dir)
            out_dir = common.make_absolute(cwd, out_dir)
            os.makedirs(out_dir, exist_ok=True)

            # learner_cls is not part of the config
            run_config = deepcopy(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "group_id": group_id,
                    "model_load_file": model_load_file,
                    "model_load_update": cfg.get("model_load_update", False),
                    "out_dir": out_dir,
                    "log_tensorboard": cfg.get("log_tensorboard", False),
                    "progress_bar": cfg.get("progress_bar", True),
                    "reset_num_timesteps": cfg.get("reset_num_timesteps", False),
                    "learner_kwargs": cfg.get("learner_kwargs", {}),
                    "net_arch": cfg.get("net_arch", []),
                    "activation": cfg.get("activation", "ReLU"),
                    "features_extractor": cfg.get("features_extractor", {}),
                    "optimizer": cfg.get("optimizer", {}),
                    "env_kwargs": expanded_env_kwargs,
                    "mapmask": cfg.get("mapmask", "ai/generated/A*.vmap"),
                    "randomize_maps": cfg.get("randomize_maps", False),
                    "n_global_steps_max": cfg.get("n_global_steps_max", None),
                    "rollouts_total": cfg.get("rollouts_total", 0),
                    "rollouts_per_iteration": cfg.get("rollouts_per_iteration", 100),
                    "rollouts_per_log": cfg.get("rollouts_per_log", 5),
                    "n_envs": cfg.get("n_envs", 1),
                    "save_every": cfg.get("save_every", 3600),
                    "max_saves": cfg.get("max_saves", 3),
                    "learning_rate": cfg.get("learning_rate", None),
                    "learner_lr_schedule": cfg.get(
                        "learner_lr_schedule", "const_0.003"
                    ),
                }
            )

            # env_kwargs should be logged
            all_cfg = dict(run_config, env_kwargs=env_kwargs)

            run_config["config_log"] = {}
            for (k, v) in cfg.get("logparams", {}).items():
                run_config["config_log"][k] = common.extract_dict_value_by_path(all_cfg, v)

            print("Starting run %s with seed %s" % (run_id, seed))

            os.environ["WANDB_SILENT"] = "true"
            common.wandb_init(run_id, group_id, all_cfg)

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
  train_ppo [group] [run] [loadfile]     train using Proximal Policy Optimization (PPO)
  train_mppo [group] [run] [loadfile]    train using Maskable Proximal Policy Optimization (MPPO)
  train_qrdqn [group] [run] [loadfile]   train using Quantile Regression DQN (QRDQN)
  train_mqrdqn [group] [run] [loadfile]  train using Maskable Quantile Regression DQN (my impl)
  spectate                               watch a trained model play VCMI
  benchmark [map]                        evaluate the actions/s achievable with this env
  test [map]                             for testing purposes only
  play [map]                             play VCMI
  help                                   print this help message

examples:
  %(prog)s train_qrdqn
  %(prog)s -c path/to/config.yml train_qrdqn
"""

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)

    if args.c is None:
        print("AAAAAA")
        breakpoint()
        args.c = open(os.path.join("config", f"{args.action}.yml"), "r")

    print("Loading configuration from %s" % args.c.name)
    cfg = yaml.safe_load(args.c)
    args.c.close()

    run(args.action, cfg, rest=args.rest)


if __name__ == "__main__":
    main()
