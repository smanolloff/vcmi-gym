import torch
import os
import sys
import time
import wandb
import json


def main(file, mdfile=None):
    ctime = int(os.path.getmtime(file))

    folder = os.path.dirname(file)
    jitfile = os.path.join(folder, f"jit-{os.path.basename(file)}")

    agent = torch.load(file, map_location=torch.device("cpu"), weights_only=False)

    run = wandb.init(project="vcmi-gym", id=agent.args.run_id, resume="must", reinit=True)
    art = wandb.Artifact(
        name=f"model-{agent.args.group_id}.{agent.args.run_id}",
        type="model",
        description=f"Snapshot of agent.pt from {time.ctime(ctime)}"
    )

    if mdfile:
        with open(mdfile, "r") as jfile:
            art.metadata = json.load(jfile)

    art.metadata["origin"] = {
        "run_id": agent.args.run_id,
        "group_id": agent.args.group_id
    }

    agent.__class__.jsave(agent, jitfile)

    art.add_file(file, name="agent.pt")
    art.add_file(jitfile, name="jit-agent.pt")

    print("Uploading artifact:")
    print(f"    name: {art.name}")
    print(f"    description: {art.description}")
    print(f"    metadata: {art.metadata}")

    run.log_artifact(art)
    print(f"URL: https://wandb.ai/s-manolloff/vcmi-gym/artifacts/model/{art.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m rl.wandb.scripts.upload_model /path/to/agent.pt")
        print("Usage: python -m rl.wandb.scripts.upload_model /path/to/agent.pt /path/to/metadata.json")
    else:
        main(*sys.argv[1:])
