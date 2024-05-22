import os
import sys
import wandb

api = wandb.Api()

if len(sys.argv) == 2:
    name = sys.argv[1]
    artifact = api.artifact(name)
elif len(sys.argv) == 3:
    os.environ["WANDB_PROJECT"] = "vcmi-gym"
    os.environ["WANDB_ENTITY"] = "s-manolloff"
    run_id = sys.argv[1]
    tag = sys.argv[2]
    run = api.run(run_id)
    artifact = api.artifact(name)
    name = f"model-{run.group}.{run.id}:{tag}"
else:
    print("Usage (1): python -m rl.wandb.scripts.download_model FULL_NAME")
    print("Usage (2): python -m rl.wandb.scripts.download_model <RUN_ID> ALIAS")
    print("Example:")
    print("    (1) python -m rl.wandb.scripts.download_model s-manolloff/vcmi-gym/model-PBT-...b6623_00000:v2")
    print("    (2) python -m rl.wandb.scripts.download_model b6623_00000 v4")

dest = artifact.download(f"rl/models/{artifact.name}")
print(f"Downloaded to {dest}")
