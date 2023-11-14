import os
import yaml
import argparse
import time
import logging
import sys
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger

from vcmi_gym import VcmiEnv
from . import common

logging.basicConfig(
    format="[PY][%(filename)s] (%(funcName)s) %(message)s",
    level=logging.DEBUG
)

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


def run(action, cfg, tag=None):
    env_kwargs = cfg.pop("env_kwargs", {})
    expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)

    def env_creator(**kwargs):
        return VcmiEnv(**kwargs)

    gym.envs.register(id="local/VCMI-v0", entry_point=env_creator, kwargs=expanded_env_kwargs)

    match action:
        case "train_qrdqn":
            venv = make_vec_env("local/VCMI-v0")
            learner_kwargs = cfg.get("learner_kwargs")
            learning_rate = common.lr_from_schedule(learner_lr_schedule)
            kwargs = dict(learner_kwargs, learning_rate=learning_rate)
            model = sb3_contrib.QRDQN(env=venv, **kwargs)
            out_dir_template = cfg.get("out_dir_template", "data/QRDQN-{run_id}")
            out_dir = common.out_dir_from_template(out_dir_template, seed, run_id)

            if log_tensorboard:
                os.makedirs(out_dir, exist_ok=True)
                log = logger.configure(folder=out_dir, format_strings=["tensorboard"])
                model.set_logger(log)

            total_timesteps = cfg.get("total_timesteps", 1000000)
            n_checkpoints = cfg.get("n_checkpoints", 5)

            model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                progress_bar=True
            )

            common.save_model(out_dir, model)

        case "benchmark":
            from .benchmark import benchmark
            steps = cfg.get("steps", 10000)
            benchmark(steps)

        case "test":
            from .test import test
            test(env_kwargs)

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
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.usage = "%(prog)s [options] <action>"
    parser.epilog = """
action:
  train_qrdqn       train using Quantile Regression DQN (QRDQN)
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

    run(args.action, cfg)


if __name__ == "__main__":
    main()
