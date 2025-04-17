import pathlib


def setup_wandb(config, model, src_file):
    import wandb

    resumed = config["run"]["resumed_config"] is not None

    wandb.init(
        project="vcmi-gym",
        group="transition-model",
        name=config["run"]["name"],
        id=config["run"]["id"],
        resume="must" if resumed else "never",
        # resume="allow",  # XXX: reuse id for insta-failed runs
        config=config,
        sync_tensorboard=False,
        save_code=False,  # code saved manually below
        allow_val_change=resumed,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable System/ stats
    )

    # https://docs.wandb.ai/ref/python/run#log_code
    # XXX: "path" is relative to `root`
    #      but args.cfg_file is relative to vcmi-gym ROOT dir
    src_file = pathlib.Path(src_file)

    def code_include_fn(path):
        p = pathlib.Path(path).absolute()
        return p.samefile(src_file)

    wandb.run.log_code(root=src_file.parent, include_fn=code_include_fn)
    wandb.watch(model, log="all", log_graph=True, log_freq=1000)
    return wandb
