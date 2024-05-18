import torch
import os
import sys
import time
import wandb

assert len(sys.argv) == 2, "Usage: python -m rl.wandb.scripts.upload_model /path/to/agent.pt"

file = sys.argv[1]
ctime = int(os.path.getmtime(file))

folder = os.path.dirname(file)
jitfile = os.path.join(folder, f"jit-{os.path.basename(file)}")

agent = torch.load(file)

run = wandb.init(project="vcmi-gym", id=agent.args.run_id, resume="must", reinit=True)
art = wandb.Artifact(
    name=f"model-{agent.args.group_id}.{agent.args.run_id}",
    type="model",
    description=f"Snapshot of agent.pt from {time.ctime(ctime)}"
)

agent.__class__.jsave(agent, jitfile)

art.add_file(file, name="agent.pt")
art.add_file(jitfile, name="jit-agent.pt")

print("Logging artifact:")
print(f"    name: {art.name}")
print(f"    description: {art.description}")

run.log_artifact(art)
