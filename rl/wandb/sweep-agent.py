import wandb
import os
import sys
import signal
import importlib


def handle_signal(signum, frame):
    print("*** [main.py] received signal %s ***" % signum)
    sys.exit(0)


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    signal.signal(signal.SIGTERM, handle_signal)

    wandb.init(
        # sync_tensorboard=True,
        sync_tensorboard=False,  # tb logs are just filling up disk space
        save_code=False,  # code saved manually below
        group="sweeps",  # can't set group after wandb.init was called => hard-code
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )

    config = dict(
        wandb.config,
        wandb_project=os.environ["WANDB_PROJECT"],
        run_id=os.environ["WANDB_RUN_ID"],
        group_id="sweeps",
        skip_wandb_init=True,
    )

    script_cfg = config.pop("script")
    mod = importlib.import_module(script_cfg["module"])
    args = mod.Args(**config)
    mod.main(args)


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    main()
