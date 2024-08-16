import os
import sys
import torch


#
# Convert agent.pt to jit-agent.pt
#

def main(file, mdfile=None):
    folder = os.path.dirname(file)
    jitfile = os.path.join(folder, f"jit-{os.path.basename(file)}")
    agent = torch.load(file, map_location=torch.device("cpu"))
    agent.__class__.jsave(agent, jitfile)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m rl.wandb.scripts.convert_model /path/to/agent.pt")
    else:
        main(*sys.argv[1:])
