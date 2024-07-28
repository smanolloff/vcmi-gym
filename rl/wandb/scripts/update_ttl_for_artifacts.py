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
        "group": {"$regex": "PBT-v3-mac-lstm-20240716_153830"},
        "display_name": "T0"
    }
)

for run in runs:
    print("Scanning artifacts of run %s (%s/%s)" % (run.name, run.group, run.id))
    artifacts = [(a, datetime.datetime.fromisoformat(a.created_at)) for a in run.logged_artifacts()]
    print("Found %d artifacts" % len(artifacts))

    now = datetime.datetime.now()

    for artifact, dt in sorted(artifacts, key=lambda x: x[1]):
        if not artifact.name.startswith("agent.pt:v"):
            continue

        if not artifact.ttl:
            continue

        artifact.ttl = datetime.timedelta(days=30)
        print("Updating artifact created at %s with new TTL: %s" % (artifact.created_at, artifact.ttl))
        artifact.save()
