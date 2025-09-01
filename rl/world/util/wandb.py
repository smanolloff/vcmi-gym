import pygit2
import os
import tempfile
import json
import vastai_sdk
from datetime import datetime


def setup_wandb(config, model, src_file, wandb_kwargs={}):
    import wandb

    resumed = config["run"]["resumed_config"] is not None

    git = pygit2.Repository(os.path.dirname(__file__))
    now = datetime.utcnow()
    patch = git.diff().patch

    start_info = dict(
        git_head=str(git.head.target),
        git_head_message=git[git.head.target].message,
        git_status={k: v.name for k, v in git.status().items()},
        timestamp=now.isoformat(timespec='milliseconds'),
        vastai_instance_id=os.getenv("VASTAI_INSTANCE_ID"),
    )

    start_info.update(git_is_dirty=True, git_diff_artifact=f"startinfo-{now.strftime('%Y%m%d%H%M%S')}")

    kwargs = dict(
        project="vcmi-gym",
        group=config["wandb_group"],
        name=config["run"]["name"],
        id=config["run"]["id"],
        # resume="must" if resumed else "never",
        resume="allow",  # XXX: reuse id for insta-failed runs
        config=config,
        sync_tensorboard=False,
        save_code=False,  # code saved manually below
        allow_val_change=resumed,
        # settings=wandb.Settings(_disable_stats=True),  # disable System/ stats
    )

    kwargs.update(wandb_kwargs)
    wandb.init(**kwargs)

    start_infos = wandb.config.get("_start_infos", [])
    start_infos.append(start_info)
    # Store VastAI instance ID separately (outside of the array) for UI convenience
    wandb.config.update(dict(vastai_instance_id=os.getenv("VASTAI_INSTANCE_ID"), _start_infos=start_infos), allow_val_change=True)

    art = wandb.Artifact(name=f"startinfo-{now.strftime('%Y%m%d%H%M%S')}", type="text")

    # Must be after wandb.init
    art = wandb.Artifact(name=start_info["git_diff_artifact"], type="text")
    art.description = f"Start info for HEAD@{start_info['git_head']} from {start_info['timestamp']}"
    art.metadata["timestsamp"] = start_info["timestamp"]
    art.metadata["head"] = start_info["git_head"]

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as cfg_file:
        json.dump(config, cfg_file)
        art.add_file(cfg_file.name, name="config.json", policy="mutable")

    if os.getenv("VASTAI_INSTANCE_ID") is not None:
        v = vastai_sdk.VastAI()
        instance_id = int(os.environ["VASTAI_INSTANCE_ID"])
        v.label_instance(id=instance_id, label=config["run"]["id"])
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as instance_file:
            json.dump(v.show_instance(id=instance_id), instance_file)
            art.add_file(instance_file.name, name="vastai.json", policy="mutable")

    if start_info["git_is_dirty"]:
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as diff_file:
            diff_file.write(f"# Head: {start_info['git_head']}\n")
            diff_file.write(f"# Timestamp: {start_info['timestamp']}\n")
            diff_file.write(patch)
            diff_file.flush()
            # mutable = wandb creates a copy (safe to delete this file)
            art.add_file(diff_file.name, name="diff.patch", policy="mutable")

    wandb.run.log_artifact(art)

    # XXX: no "Model" will be shown in the W&B UI when .forward()
    #       returns a non-tensor value (e.g. a tuple)
    wandb.watch(model, log="all", log_graph=True, log_freq=1000)
    return wandb
