import wandb

#
# All-purpose script for mass-updating wandb run attributes.
#

api = wandb.Api()
eval_runs = api.runs(
    path="s-manolloff/vcmi-gym",
    filters={"group": {"$regex": "4096-mixstack"}}
)

for eval_run in eval_runs:
    print(f"Updating run {eval_run.id}...")
    orig_id = eval_run.id.removeprefix("eval-")
    orig_run = api.run(f"s-manolloff/vcmi-gym/{orig_id}")

    # eval_run.tags = orig_run.tags + ["eval"]
    eval_run.group = "4096-6stack"
    eval_run.tags = ["Map-6stack-01", "StupidAI"]

    eval_run.save()
