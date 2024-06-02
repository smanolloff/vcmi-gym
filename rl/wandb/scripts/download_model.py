import sys
import wandb
import json
import os


def main(name):
    os.environ["WANDB_ENTITY"] = "s-manolloff"
    os.environ["WANDB_PROJECT"] = "vcmi-gym"

    api = wandb.Api()

    artifact = api.artifact(name)

    dest = artifact.download(f"rl/models/{artifact.name}")
    print(f"Downloaded to {dest}")

    with open(f"{dest}/description.txt", "w") as f:
        f.write(artifact.description)

    with open(f"{dest}/metadata.json", "w") as f:
        f.write(json.dumps(artifact.metadata, indent=4))

    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m rl.wandb.scripts.download_model ARTIFACT_NAME")
        print("Examples:")
        print("    python -m rl.wandb.scripts.download_model s-manolloff/vcmi-gym/model-PBT-...b6623_00000:v2")
        print("    python -m rl.wandb.scripts.download_model agent.pt:v7")
    else:
        main(sys.argv[1])
