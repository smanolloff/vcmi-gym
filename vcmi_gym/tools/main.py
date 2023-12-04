import os
import yaml
import argparse
from copy import deepcopy

from . import common

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


def run(action, cfg, rest=[]):
    # print("**** ENV WANDB_RUN_ID: %s" % os.environ["WANDB_RUN_ID"])
    # import wandb
    # print("**** wandb.run.id: %s" % wandb.run.id)

    cwd = os.getcwd()
    env_wrappers = cfg.pop("env_wrappers", {})
    env_kwargs = cfg.pop("env_kwargs", {})

    match action:
        case "train_ppo" | "train_qrdqn" | "train_mppo":
            from .train_sb3 import train_sb3
            expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
            common.register_env(expanded_env_kwargs, env_wrappers)

            learner_cls = action.split("_")[-1].upper()
            default_template = "data/%s-{run_id}" % learner_cls
            out_dir_template = cfg.get("out_dir_template", default_template)
            out_dir_template = common.make_absolute(cwd, out_dir_template)
            seed = cfg.get("seed", None) or common.gen_seed()
            run_id = cfg.get("run_id", None) or common.gen_id()

            # learner_cls is not part of the config
            run_config = deepcopy(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "model_load_file": cfg.get("model_load_file", None),
                    "model_load_update": cfg.get("model_load_update", False),
                    "out_dir_template": out_dir_template,
                    "log_tensorboard": cfg.get("log_tensorboard", False),
                    "progress_bar": cfg.get("progress_bar", True),
                    "reset_num_timesteps": cfg.get("reset_num_timesteps", False),
                    "learner_kwargs": cfg.get("learner_kwargs", {}),
                    "total_timesteps": cfg.get("total_timesteps", 1000000),
                    "n_checkpoints": cfg.get("n_checkpoints", 5),
                    "n_envs": cfg.get("n_envs", 1),
                    "learning_rate": cfg.get("learning_rate", None),
                    "learner_lr_schedule": cfg.get(
                        "learner_lr_schedule", "const_0.003"
                    ),
                }
            )

            print("Starting run %s with seed %s" % (run_id, seed))

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
                cfg["model_file"] = rest[0]

            spectate(
                fps=cfg.get("fps", 2),
                reset_delay=cfg.get("reset_delay", 5),
                model_file=cfg["model_file"],
                model_mod=cfg.get("model_mod", "stable_baselines3"),
                model_cls=cfg.get("model_cls", "PPO"),
            )

        case "benchmark":
            from .benchmark import benchmark

            # See comment in "test"
            # (here we use the actions only)
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

            steps = cfg.get("steps", 10000)
            benchmark(steps, actions)

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
  train_ppo         train using Proximal Policy Optimization (PPO)
  train_qrdqn       train using Quantile Regression DQN (QRDQN)
  spectate          watch a trained model play VCMI
  benchmark         evaluate the actions/s achievable with this env
  test              for testing purposes only
  help              print this help message

examples:
  %(prog)s train_qrdqn
"""

    args = parser.parse_args()

    if args.c is None:
        args.c = open(os.path.join("config", f"{args.action}.yml"), "r")

    print("Loading configuration from %s" % args.c.name)
    cfg = yaml.safe_load(args.c)
    args.c.close()

    run(args.action, cfg, rest=args.rest)


if __name__ == "__main__":
    main()
