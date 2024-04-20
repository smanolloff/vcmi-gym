import wandb

#
# All-purpose script for mass-updating wandb run attributes.
#

api = wandb.Api()
eval_runs = api.runs(
    path="s-manolloff/vcmi-gym",
    filters={"group": {"$regex": "evaluator"}}
)

for eval_run in eval_runs:
    print(f"Updating run {eval_run.id}...")
    orig_id = eval_run.id.removeprefix("eval-")
    orig_run = api.run(f"s-manolloff/vcmi-gym/{orig_id}")

    # eval_run.tags = orig_run.tags + ["eval"]
    eval_run.config = {
        "orig_group": orig_run.group,
        "orig_sweep": getattr(orig_run.sweep, "id", None)
    }

    eval_run.save()
