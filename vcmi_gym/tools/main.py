import os
import yaml
import argparse
from copy import deepcopy

from . import common

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


# "extras" is an arbitary dict object which is action-dependent
# It is used when calling "run" from raytune, for example.
def run(action, cfg, extras={}):
    # print("**** ENV WANDB_RUN_ID: %s" % os.environ["WANDB_RUN_ID"])
    # import wandb
    # print("**** wandb.run.id: %s" % wandb.run.id)

    cwd = os.getcwd()
    env_wrappers = cfg.pop("env_wrappers", {})
    env_kwargs = cfg.pop("env_kwargs", {})
    expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
    common.register_env(expanded_env_kwargs, env_wrappers, extras.get("overwrite_env", False))

    match action:
        case "train_ppo" | "train_qrdqn":
            from .train_sb3 import train_sb3

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
                    "extras": extras,
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

            spectate(
                fps=cfg.get("fps", 2),
                reset_delay=cfg.get("reset_delay", 5),
                model_file=cfg["model_file"],
                model_mod=cfg.get("model_mod", "stable_baselines3"),
                model_cls=cfg.get("model_cls", "PPO"),
            )

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

    # TESTING WANDB
    # import wandb
    # wandb_run = wandb.init(project="vcmi")
    # run(args.action, cfg, extras={"wandb_run": wandb_run})

    run(args.action, cfg)


if __name__ == "__main__":
    main()
