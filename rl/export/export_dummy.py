import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from vcmi_gym.envs.v12.pyconnector import STATE_SIZE

# DUMMY model to export for testing behaviour in isolation

MODEL_EXPORT_PATH = "/Users/simo/Projects/vcmi-play/Mods/MMAI/models/dummy.pte"


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask_action", torch.zeros([1, 4], dtype=torch.bool), persistent=False)

    def forward(self, obs):
        obs = obs.unsqueeze(dim=0)
        mask_action = torch.zeros_like(self.mask_action)
        mask_action[:, 0] = obs[:, 15].to(torch.bool)
        return (obs[:, 15], mask_action[:, 0])


model = DummyModel().eval().cpu().float()
example_inputs = (torch.randn(STATE_SIZE),)
model.predict(example_inputs[0])
ep = {"predict": export(model, args=example_inputs, dynamic_shapes=None, strict=True)}
edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
exec_prog = edge.to_executorch()
with open(MODEL_EXPORT_PATH, "wb") as f:
    exec_prog.write_to_file(f)

print(f"Wrote: {MODEL_EXPORT_PATH}")
