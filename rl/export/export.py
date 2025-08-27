import json
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from rl.algos.mppo_dna_heads.mppo_dna_heads_new import JitDNAModel
from vcmi_gym.envs.v12.pyconnector import STATE_SIZE

filebase = "agicelt2-1756110154"
model_cfg_path = f"{filebase}-config.json"
model_weights_path = f"{filebase}-model-dna.pt"

MODEL_EXPORT_PATH = f"/Users/simo/Projects/vcmi-play/Mods/MMAI/models/{filebase}.pte"


class ModelWrapper(torch.nn.Module):
    def __init__(self, m, method_name: str):
        super().__init__()
        self.m = m
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.m, self.method_name)(*args, **kwargs)


config = json.load(open(model_cfg_path, "r"))["model"]
model = JitDNAModel(config)
weights = torch.load(model_weights_path, weights_only=True, map_location="cpu")
model.load_state_dict(weights)
for p in model.parameters():
    p.requires_grad_(False)

example_inputs = (torch.randn(STATE_SIZE),)
model.predict(example_inputs[0])
model.get_value(example_inputs[0])
ep_predict = export(ModelWrapper(model.model_policy, "predict").eval(), args=example_inputs, dynamic_shapes=None, strict=True)
ep_get_value = export(ModelWrapper(model.model_value, "get_value").eval(), args=example_inputs, dynamic_shapes=None, strict=True)
ep_get_version = export(ModelWrapper(model, "get_version").eval(), args=())

edge_programs = {
    "predict": ep_predict,
    "get_value": ep_get_value,
    "get_version": ep_get_version,
}

edge_model = to_edge_transform_and_lower(edge_programs, partitioner=[XnnpackPartitioner()]).to_executorch()

with open(MODEL_EXPORT_PATH, "wb") as f:
    f.write(edge_model.buffer)

print(f"Wrote: {MODEL_EXPORT_PATH}")
