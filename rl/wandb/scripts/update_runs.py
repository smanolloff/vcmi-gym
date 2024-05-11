import wandb

#
# All-purpose script for mass-updating wandb run attributes.
#

api = wandb.Api()
runs = api.runs(
    path="s-manolloff/vcmi-gym",
    #filters={"group": {"$regex": "PBT-mppo-obsmask-20240425_182811"}}
    filters={"name": {"$regex": "eval-(7cb19_00000|41540_00000)"}}
)

for run in runs:
    print(f"Updating run {run.id}...")
    #run.tags = ["Map-4096-mixstack", "StupidAI"]
    run.tags = [tag for tag in run.tags if tag not in ["MPPO_DNA"]]
    run.save()
