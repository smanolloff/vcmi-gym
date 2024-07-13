import wandb
import datetime

#
# All-purpose script for mass-updating wandb run attributes.
#

api = wandb.Api()
gt = datetime.datetime.now() - datetime.timedelta(days=21)
runs = wandb.Api().runs(
    path="s-manolloff/vcmi-gym",
    filters={
        "updatedAt": {"$gt": gt.isoformat()},
        "display_name": "T0"
    }
)

for run in runs:
    print("Scanning artifacts of run %s (%s/%s)" % (run.name, run.group, run.id))
    artifacts = [(a, datetime.datetime.fromisoformat(a.created_at)) for a in run.logged_artifacts()]
    print("Found %d artifacts" % len(artifacts))

    now = datetime.datetime.now()

    for artifact, dt in sorted(artifacts, key=lambda x: x[1]):
        if not artifact.ttl:
            continue

        created_at = datetime.datetime.fromisoformat(artifact.created_at)
        if created_at + artifact.ttl < now:
            print("Deleting artifact created at %s with TTL %s" % (artifact.created_at, artifact.ttl))
            artifact.delete()
