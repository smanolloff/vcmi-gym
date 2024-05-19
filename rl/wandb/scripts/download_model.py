import os
import sys
import wandb

if len(sys.argv) != 3:
    print("Usage: python -m rl.wandb.scripts.download_model <RUN_ID> ALIAS")
    print("Example:")
    print("    python -m rl.wandb.scripts.download_model b6623_00000 v4")
    sys.exit(1)

run_id = sys.argv[1]
tag = sys.argv[2]

os.environ["WANDB_PROJECT"] = "vcmi-gym"
os.environ["WANDB_ENTITY"] = "s-manolloff"

api = wandb.Api()
run = api.run(run_id)
name = f"model-{run.group}.{run.id}:{tag}"
artifact = api.artifact(name)
dest = artifact.download(f"rl/models/{artifact.name}")
print(f"Downloaded to {dest}")
