import wandb
import os

from vcmi_gym.tools.crl.mppo_heads import main, Args

if __name__ == "__main__":
    print(os.environ)

    wandb.init(
        sync_tensorboard=True,
        save_code=False,  # code saved manually below
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )

    config = dict(
        wandb.config["c"],
        run_id=os.environ["WANDB_RUN_ID"],
        skip_wandb_init=True,
    )

    args = Args(**config)
    main(args)
