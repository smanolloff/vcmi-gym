import os
import json
import torch
import numpy as np
from torch.export import export, export_for_training
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from rl.algos.mppo_dna_heads.mppo_dna_heads_new import ExecuTorchDNAModel


class ModelWrapper(torch.nn.Module):
    def __init__(self, m, method_name: str):
        super().__init__()
        self.m = m
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.m, self.method_name)(*args, **kwargs)


if __name__ == "__main__":
    filebase = "sfcjqcly-1757584171"
    model_cfg_path = f"{filebase}-config.json"
    model_weights_path = f"{filebase}-model-dna.pt"
    obs_path = "%s/obs.np" % os.path.dirname(__file__)

    MODEL_EXPORT_PATH = f"/Users/simo/Projects/vcmi-play/Mods/MMAI/models/{filebase}.pte"

    # Reduces size x4
    QUANTIZE = True

    example_input = torch.as_tensor(np.fromfile(obs_path, dtype=np.float32)).contiguous()
    config = json.load(open(model_cfg_path, "r"))["model"]
    model = ExecuTorchDNAModel(config)
    weights = torch.load(model_weights_path, weights_only=True, map_location="cpu")
    model.load_state_dict(weights)
    model = model.eval().cpu()
    model.predict(example_input)
    model.get_value(example_input)

    # XXX: if model_value is excluded, size will be further reduced x2

    m_predict = ModelWrapper(model.model_policy, "predict").eval().cpu()
    m_get_value = ModelWrapper(model.model_value, "get_value").eval().cpu()
    m_get_ver = ModelWrapper(model, "get_version").eval().cpu()
    m_get_e_max = ModelWrapper(model, "get_e_max").eval().cpu()
    m_get_k_max = ModelWrapper(model, "get_k_max").eval().cpu()

    if QUANTIZE:
        print("Quantizing model...")
        # Quantizer
        # XXX: comparing outputs, is_per_channel=True gives *sligthtly* better results => use it
        q = XNNPACKQuantizer()
        q.set_global(get_symmetric_quantization_config(is_per_channel=True))

        # --- PT2E prepare/convert ---
        # export_for_training -> .module() for PT2E helpers
        pre_predict = export_for_training(m_predict, (example_input,), strict=True).module()
        pre_get_value = export_for_training(m_get_value, (example_input,), strict=True).module()
        pre_get_ver = export_for_training(m_get_ver, (), strict=True).module()
        pre_get_e_max = export_for_training(m_get_e_max, (), strict=True).module()
        pre_get_k_max = export_for_training(m_get_k_max, (), strict=True).module()

        # Insert observers
        prep_predict = prepare_pt2e(pre_predict, q)
        prep_get_value = prepare_pt2e(pre_get_value, q)
        # Skip quant for get_version (optional)
        prep_get_ver = pre_get_ver
        prep_get_e_max = pre_get_e_max
        prep_get_k_max = pre_get_k_max

        # Calibrate
        _ = prep_predict(example_input)
        _ = prep_get_value(example_input)

        # Convert to quantized modules
        m_predict = convert_pt2e(prep_predict)
        m_get_value = convert_pt2e(prep_get_value)
        m_get_ver = prep_get_ver  # left unquantized on purpose
        m_get_e_max = prep_get_e_max  # left unquantized on purpose
        m_get_k_max = prep_get_k_max  # left unquantized on purpose

    ep = {
        "predict": export(m_predict, (example_input,), strict=True),
        "get_value": export(m_get_value, (example_input,), strict=True),
        "get_version": export(m_get_ver, (), strict=True),
        "get_e_max": export(m_get_e_max, (), strict=True),
        "get_k_max": export(m_get_k_max, (), strict=True),
    }

    print("Exporting model...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    with open(MODEL_EXPORT_PATH, "wb") as f:
        edge.to_executorch().write_to_file(f)

    print(f"Wrote: {MODEL_EXPORT_PATH}")
