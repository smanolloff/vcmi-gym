import pygit2
import os
import tempfile
from datetime import datetime


def setup_wandb(config, model, src_file):
    import wandb

    resumed = config["run"]["resumed_config"] is not None

    git = pygit2.Repository(os.path.dirname(__file__))
    now = datetime.utcnow()
    patch = git.diff().patch

    start_info = dict(
        git_head=str(git.head.target),
        git_status={k: v.name for k, v in git.status().items()},
        timestamp=now.isoformat(timespec='milliseconds'),
        vastai_instance_id=os.getenv("VASTAI_INSTANCE_ID"),
    )

    if patch:
        start_info.update(git_is_dirty=True, git_diff_artifact="gitdiff-%d" % now.timestamp())
    else:
        start_info.update(git_is_dirty=False)

    wandb.init(
        project="vcmi-gym",
        group=config["wandb_group"],
        name=config["run"]["name"],
        id=config["run"]["id"],
        resume="must" if resumed else "never",
        # resume="allow",  # XXX: reuse id for insta-failed runs
        config=config,
        sync_tensorboard=False,
        save_code=False,  # code saved manually below
        allow_val_change=resumed,
        # settings=wandb.Settings(_disable_stats=True),  # disable System/ stats
    )

    start_infos = wandb.config.get("_start_infos", [])
    start_infos.append(start_info)
    wandb.config.update(dict(_start_infos=start_infos), allow_val_change=True)

    # Must be after wandb.init
    if start_info["git_is_dirty"]:
        art = wandb.Artifact(name=start_info["git_diff_artifact"], type="text")
        art.description = f"Git diff for HEAD@{start_info['git_head']} from {start_info['timestamp']}"
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
            art.metadata["timestsamp"] = start_info["timestamp"]
            art.metadata["head"] = start_info["git_head"]
            temp_file.write(f"# Head: {start_info['git_head']}\n")
            temp_file.write(f"# Timestamp: {start_info['timestamp']}\n")
            temp_file.write(patch)
            temp_file.flush()
            art.add_file(temp_file.name, name="diff.patch")
            wandb.run.log_artifact(art)

    wandb.watch(model, log="all", log_graph=True, log_freq=1000)
    return wandb
