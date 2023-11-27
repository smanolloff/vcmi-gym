import wandb


def wandb_init(meth, trial_id, trial_name, experiment_name, config):
    # print("[%s] INITWANDB: PID: %s, trial_id: %s" % (time.time(), os.getpid(), trial_id))

    # https://github.com/ray-project/ray/blob/ray-2.8.0/python/ray/air/integrations/wandb.py#L601-L607
    wandb.init(
        id=trial_id,
        name="%s_%s" % (meth, trial_name.split("_")[-1]),
        resume="allow",
        reinit=True,
        allow_val_change=True,
        # To disable System/ stats:
        # settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        group=experiment_name,
        project=config["wandb_project"],
        config=config,
        # NOTE: this takes a lot of time, better to have detailed graphs
        #       tb-only (local) and log only most important info to wandb
        # sync_tensorboard=True,
        sync_tensorboard=False,
    )
    # print("[%s] DONE WITH INITWANDB" % time.time())
