import argparse

from .mppo_i2a import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
    # parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    # parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument("--loglevel", metavar="LOGLEVEL", default="INFO", help="DEBUG | INFO | WARN | ERROR")

    parser.add_argument("--num_envs", metavar="INT", type=int, default=1)
    parser.add_argument("--num_vsteps", metavar="INT", type=int, default=256)
    parser.add_argument("--num_minibatches", metavar="INT", type=int, default=4)
    parser.add_argument("--num_trajectories", metavar="INT", type=int, default=5)
    parser.add_argument("--horizon", metavar="INT", type=int, default=3)
    parser.add_argument("--update_epochs", metavar="INT", type=int, default=1)
    parser.add_argument("--mapname", metavar="STR", default="gym/A1.vmap")
    parser.add_argument("--rollouts", metavar="INT", type=int, default=10)

    args = parser.parse_args()

    from .config import config

    config["train"]["num_vsteps"] = args.num_vsteps
    config["train"]["num_minibatches"] = args.num_minibatches
    config["train"]["update_epochs"] = args.update_epochs
    config["train"]["env"]["num_envs"] = args.num_envs
    config["train"]["env"]["kwargs"]["mapname"] = args.mapname
    config["model"]["num_trajectories"] = args.num_trajectories
    config["model"]["horizon"] = args.horizon

    config["eval"]["env"]["num_envs"] = 1
    config["eval"]["interval_s"] = int(1e6)
    config["checkpoint"]["permanent_interval_s"] = int(1e7)
    config["train"]["env"]["kwargs"]["user_timeout"] = int(1e9)

    _, _, cumulative_timer_values = main(
        config=config,
        resume_config=args.f,
        loglevel=args.loglevel,
        dry_run=True,
        no_wandb=True,
        total_rollouts=args.rollouts,
    )

    print("=========================")
    print("Timers:")
    timer_all = cumulative_timer_values.pop("all")
    for k, v in cumulative_timer_values.items():
        print("%-15s %.3f   (%.2fs)" % (k, v/timer_all, v))
    # print("Total vsteps: %d" % (args.rollouts * args.num_vsteps * args.num_envs))
    print("-------------------------")
    print("%-15s %.1fs" % ("Total time", timer_all))
    print("%-15s %.2fs" % ("1 rollout", timer_all / args.rollouts))
