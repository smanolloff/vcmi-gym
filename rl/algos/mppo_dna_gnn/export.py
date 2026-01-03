import json
import io
import torch
import torch_geometric
from time import perf_counter_ns
import numpy as np

import onnx
import onnxruntime as ort

from .export_common import (
    LINK_TYPES,
    ExportableGENConv,
    ExportableGNNBlock,
    ExportableDNAModel,
    transform_key,
    flatten_edges,
    build_inputs,
    build_hdata,
)

from .mppo_dna_gnn import DNAModel, GNNBlock
from .dual_vec_env import to_hdata_list, DualVecEnv

# https://chatgpt.com/s/t_69582f5c098c8191ab8afa78678b6016
# opset 18 is the first ONNX opset that can represent per-destination amax
# via scatter reduction, because ONNX adds max/min as valid reduction modes
# for ScatterElements (and ScatterND) in opset 18.
#
# Register custom symbolics for:
# aten::scatter_add → ONNX ScatterElements with reduction="add"
# aten::scatter_reduce with reduce="amax" → ONNX ScatterElements with reduction="max"
#   (and similarly amin → min if needed)

import torch.onnx.symbolic_helper as sym_help


def _get_axis_i(dim):
    axis = sym_help._maybe_get_scalar(dim)
    if axis is None:
        raise RuntimeError("scatter: dim must be constant for ONNX export")
    return int(axis)


def scatter_add_to_scatterelements(g, self, dim, index, src):
    axis_i = _get_axis_i(dim)
    return g.op("ScatterElements", self, index, src, axis_i=axis_i, reduction_s="add")


def scatter_reduce_to_scatterelements(g, self, dim, index, src, reduce, include_self):
    axis_i = _get_axis_i(dim)
    red = sym_help._maybe_get_const(reduce, "s")  # e.g. "amax"
    if red == "amax":
        reduction = "max"
    elif red == "amin":
        reduction = "min"
    elif red == "sum":
        reduction = "add"
    elif red == "prod":
        reduction = "mul"
    else:
        raise RuntimeError(f"Unsupported scatter_reduce reduce={red}")

    # Your code uses include_self=True and initializes `self` appropriately (e.g. -inf for max),
    # so include_self does not need special handling here.
    return g.op("ScatterElements", self, index, src, axis_i=axis_i, reduction_s=reduction)


torch.onnx.register_custom_op_symbolic("aten::scatter_add", scatter_add_to_scatterelements, 18)
torch.onnx.register_custom_op_symbolic("aten::scatter_reduce", scatter_reduce_to_scatterelements, 18)


def test_gnn():
    """ Tests GENConv vs ExportableGENConv. """

    import torch_geometric
    gen = torch_geometric.nn.GENConv(5, 12, edge_dim=1).eval()
    mygen = ExportableGENConv(5, 12, 1).eval()
    mygen.load_state_dict(gen.state_dict(), strict=True)

    hd = torch_geometric.data.HeteroData()
    hd['hex'].x = torch.randn(3, 5)
    edge_index_lt1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr_lt1 = torch.tensor([[1.0], [1.0]], dtype=torch.float)
    # edge_index_lt2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    # edge_attr_lt2 = torch.tensor([[0.5], [0.2]], dtype=torch.float)
    hd['hex', 'lt1', 'hex'].edge_index = edge_index_lt1
    hd['hex', 'lt1', 'hex'].edge_attr = edge_attr_lt1
    # hd['hex', 'lt2', 'hex'].edge_index = edge_index_lt2
    # hd['hex', 'lt2', 'hex'].edge_attr = edge_attr_lt2

    inputs = {
        "x": hd["hex"].x,
        "edge_index": hd["hex", "lt1", "hex"].edge_index,
        "edge_attr": hd["hex", "lt1", "hex"].edge_attr
    }

    res = gen(*inputs.values())
    myres = mygen(*inputs.values())
    assert torch.allclose(res, myres, rtol=1e-5, atol=1e-5)

    myres = mygen(*inputs.values())
    sgen = torch.jit.script(mygen)
    sres = sgen(*inputs.values())
    assert torch.allclose(res, sres, rtol=1e-6, atol=1e-6)

    buffer = io.BytesIO()

    torch.onnx.export(
        sgen,
        tuple(inputs.values()),
        buffer,
        input_names=["x", "edge_index", "edge_attr"],
        output_names=["out"],
        opset_version=18,  # onnxruntime 1.14+
        do_constant_folding=True,
        # dynamic_axes={
        #     "ei_flat": {1: "ei_dim"},       # S=[2, 1646], M=[2, 2478], ...
        #     "ea_flat": {0: "ea_dim"},       # S=[1646, 1], M=[2478, 1], ...
        #     "nbr_flat": {1: "nbr_dim"},     # S=[165, 32], M=[165, 52], ...
        # },
        # XXX: dynamo is the *new* torch ONNX exporter and will become the
        #       default in torch-2.9.0, however as of torch 2.8.0 there are
        #       missing operator implementations, and 2.9.0 is not viable
        #       as torch_geometric segfaults (it is still on 2.8.0)
        # dynamo=True
    )
    edge = ort.InferenceSession(buffer.getvalue())
    eres = edge.run(None, {k: v.numpy() for k, v in inputs.items()})[0]

    assert torch.allclose(res, torch.as_tensor(eres), rtol=1e-6, atol=1e-6)

    print("test_gnn: OK")


def test_block():
    """ Tests GNNBlock vs ExportableGNNBlock. """

    N = 3       # num_nodes
    hd = torch_geometric.data.HeteroData()
    hd['baba'].x = torch.randn(N, 5)
    edge_index_lt1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr_lt1 = torch.tensor([[1.0], [1.0]], dtype=torch.float)
    # edge_index_lt2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    # edge_attr_lt2 = torch.tensor([[0.5], [0.2]], dtype=torch.float)
    edge_index_lt2 = torch.tensor([[0, 2, 0], [2, 1, 1]], dtype=torch.long)
    edge_attr_lt2 = torch.tensor([[0.5], [0.2], [0.1]], dtype=torch.float)
    hd['baba', 'lt1', 'baba'].edge_index = edge_index_lt1
    hd['baba', 'lt1', 'baba'].edge_attr = edge_attr_lt1
    hd['baba', 'lt2', 'baba'].edge_index = edge_index_lt2
    hd['baba', 'lt2', 'baba'].edge_attr = edge_attr_lt2
    hd.obs = hd['baba'].x

    num_layers = 3
    in_channels = hd["baba"].x.size(1)
    hidden_channels = 6
    out_channels = 7

    assert hd.node_types == ["baba"]
    node_type = hd.node_types[0]
    link_types = [lt for (_, lt, _) in hd.edge_types]

    block = GNNBlock(
        num_layers=num_layers,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        gnn_kwargs=dict(add_self_loops=True),
        link_types=link_types,
        node_type=node_type,
    ).eval()

    myblock = ExportableGNNBlock(
        num_layers=num_layers,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        link_types=link_types,
    ).eval()

    mydict = {transform_key(k, node_type, link_types): v for k, v in block.state_dict().items()}
    myblock.load_state_dict(mydict, strict=True)

    # Test with two different "sizes"
    ei_flat, ea_flat, lengths = flatten_edges(hd)

    inputs = {
        "x_hex": hd["baba"].x,
        "ei_flat": ei_flat,
        "ea_flat": ea_flat,
        "lengths": lengths
    }

    res = block(hd)["baba"]
    myres0 = myblock(*inputs.values())

    assert torch.equal(res, myres0)

    sblock = torch.jit.script(myblock)
    sres = sblock(*inputs.values())
    assert torch.allclose(res, sres, rtol=1e-6, atol=1e-6)

    buffer = io.BytesIO()

    torch.onnx.export(
        sblock,
        tuple(inputs.values()),
        buffer,
        input_names=["x_hex", "ei_flat", "ea_flat", "lengths"],
        output_names=["z_hex"],
        opset_version=18,  # onnxruntime 1.14+
        do_constant_folding=True,
        # dynamic_axes={
        #     "ei_flat": {1: "ei_dim"},       # S=[2, 1646], M=[2, 2478], ...
        #     "ea_flat": {0: "ea_dim"},       # S=[1646, 1], M=[2478, 1], ...
        #     "nbr_flat": {1: "nbr_dim"},     # S=[165, 32], M=[165, 52], ...
        # },
        # XXX: dynamo is the *new* torch ONNX exporter and will become the
        #       default in torch-2.9.0, however as of torch 2.8.0 there are
        #       missing operator implementations, and 2.9.0 is not viable
        #       as torch_geometric segfaults (it is still on 2.8.0)
        # dynamo=True
    )
    edge = ort.InferenceSession(buffer.getvalue())
    eres = edge.run(None, {k: v.numpy() for k, v in inputs.items()})[0]

    assert torch.allclose(res, torch.as_tensor(eres), rtol=1e-6, atol=1e-6)

    print("test_block: OK")


def test_model(cfg, weights_file):
    """ Tests DNA Model vs ExecuTorchModel. """

    venv = DualVecEnv(dict(mapname="gym/A3.vmap", role="defender"), num_envs_stupidai=1)
    # venv = DualVecEnv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs_stupidai=1)
    # venv = DualVecEnv(dict(mapname="gym/archangels.vmap", role="defender", random_heroes=1), num_envs_stupidai=1)
    # venv = DualVecEnv(dict(
    #     mapname="gym/generated/4096/4x1024.vmap",
    #     role=cfg["train"]["env"]["kwargs"]["role"]
    # ), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExportableDNAModel(cfg["model"], eside).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    torch.set_printoptions(sci_mode=False, precision=6)

    # JIT transform is only useful for identifying export-ralted issues quicker
    print("=== JIT transform ===")
    smodel = torch.jit.script(emodel)
    smodel.eval()

    print("=== ONNX transform ===")
    buffer = io.BytesIO()

    torch.onnx.export(
        emodel,
        tuple(build_inputs(build_hdata(venv)).values()),
        buffer,
        input_names=["obs", "ei_flat", "ea_flat", "lengths"],
        output_names=["probs"],
        opset_version=18,  # onnxruntime 1.14+
        do_constant_folding=True,
        dynamic_axes={
            "ei_flat": {1: "ei_N"},       # [2, 1646], [2, 2478], ...
            "ea_flat": {0: "ea_N"},       # [1646, 1], [2478, 1], ...
        },
        # XXX: dynamo is the *new* torch ONNX exporter and will become the
        #       default in torch-2.9.0, however as of torch 2.8.0 there are
        #       missing operator implementations, and 2.9.0 is not viable
        #       as torch_geometric segfaults (it is still on 2.8.0)
        # dynamo=True
    )

    edge = ort.InferenceSession(buffer.getvalue())

    num_steps = 10
    for n in range(num_steps):
        print(venv.render()[0])
        hdata = build_hdata(venv)
        inputs = build_inputs(hdata)

        # 0. DNAModel baseline

        actsample = model.get_actsample_eval(hdata, deterministic=True)
        action = actsample.action

        # 1. Test ExportableModel

        probs = emodel.forward(*inputs.values())
        myaction = probs.argmax(1)
        assert torch.equal(action, myaction)

        # 2. Test torchscript of ExportableModel

        sprobs = smodel(*inputs.values())
        saction = sprobs.argmax(1)
        assert torch.equal(action, saction)

        # 3. Test ONNX export of ExportableModel

        for k, v in inputs.items():
            print(f"Arg {k}: {v.shape}")

        t0 = perf_counter_ns()
        eprobs = torch.as_tensor(edge.run(None, {k: v.numpy() for k, v in inputs.items()})[0])
        ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
        print("Predict time: %d ms" % ms)

        eaction = eprobs.argmax(1)
        assert torch.equal(action, eaction)

        topk = probs.topk(5)
        for k in range(5):
            action = topk.indices[0, k]
            a0, h1, h2 = model.inverse_table[action]
            p0 = actsample.act0_dist.probs[0, a0] * actsample.hex1_dist.probs[0, h1] * actsample.hex2_dist.probs[0, h2]
            p1 = topk.values[0, k]
            print("\t[k=%d] p0 = %.3f | p1 = %.3f" % (k, p0, p1))

            # These are not always that close (sometimes diff is almost 0.1)
            # if not torch.allclose(p0, p1, atol=0.01):
            #     import ipdb; ipdb.set_trace()  # noqa
            #     pass

        venv.step([action])

    print("test_model: OK")


def test_load(cfg, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExportableDNAModel. """

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExportableDNAModel(cfg["model"], eside).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    hdata = build_hdata(venv)
    inputs = build_inputs(hdata)

    for k, v in inputs.items():
        print(f"Arg {k}: {v.shape}")

    actsample = model.get_actsample_eval(hdata, deterministic=True)
    raise NotImplementedError()


def export_model(cfg, weights_file):
    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel0 = ExportableDNAModel(cfg["model"], eside).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel0.load_state_dict(eweights, strict=True)
    emodel = emodel0.model_policy.eval()

    print("=== ONNX transform ===")
    buffer = io.BytesIO()

    torch.onnx.export(
        emodel,
        tuple(build_inputs(build_hdata(venv)).values()),
        buffer,
        input_names=["obs", "ei_flat", "ea_flat", "lengths"],
        output_names=["probs"],
        opset_version=18,  # onnxruntime 1.14+
        do_constant_folding=True,
        dynamic_axes={
            "ei_flat": {1: "ei_N"},       # [2, 1646], [2, 2478], ...
            "ea_flat": {0: "ea_N"},       # [1646, 1], [2478, 1], ...
        },
        # XXX: dynamo is the *new* torch ONNX exporter and will become the
        #       default in torch-2.9.0, however as of torch 2.8.0 there are
        #       missing operator implementations, and 2.9.0 is not viable
        #       as torch_geometric segfaults (it is still on 2.8.0)
        # dynamo=True
    )

    # Can't set metadata via torch.onnx.export => load, add then save again
    loaded_model = onnx.load_from_string(buffer.getvalue())

    metadata = {
        "version": str(emodel.get_version().item()),
        "side": str(emodel.get_side().item())
    }

    for k, v in metadata.items():
        p = loaded_model.metadata_props.add()
        p.key = k
        p.value = v

    # To access metadata:
    # value = model.get_modelmeta().custom_metadata_map["<key>"]

    newbuffer = io.BytesIO()
    onnx.save(loaded_model, newbuffer)

    exported_model = newbuffer.getvalue()

    return exported_model


def verify_export(cfg, weights_file, loaded_model, num_steps=10):
    weights = torch.load(weights_file, weights_only=True, map_location="cpu")
    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]

    print("Testing metadata methods...")
    md = loaded_model.get_modelmeta().custom_metadata_map
    assert md["version"] == "13"
    assert md["side"] == str(eside)

    print("Testing data methods for %d steps..." % (num_steps))

    venv = DualVecEnv(dict(
        mapname="gym/generated/4096/4x1024.vmap",
        role=cfg["train"]["env"]["kwargs"]["role"]
    ), num_envs_stupidai=1)

    ms_total = 0

    for n in range(num_steps):
        print(venv.render()[0])

        hdata = build_hdata(venv)

        actsample = model.model_policy.get_action_logits(hdata).sample(deterministic=True)
        inputs = build_inputs(hdata)

        t0 = perf_counter_ns()
        probs = torch.as_tensor(loaded_model.run(None, {k: v.numpy() for k, v in inputs.items()})[0])
        ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
        print("Predict time: %d ms" % ms)
        ms_total += ms

        eaction = probs.argmax(1)
        print("Action: %d | %d" % (actsample.action, eaction))
        assert torch.equal(actsample.action, eaction)

        topk = probs.topk(5)
        for k in range(5):
            action = topk.indices[0, k]
            a0, h1, h2 = model.model_policy.inverse_table[action]
            p0 = actsample.act0_dist.probs[0, a0] * actsample.hex1_dist.probs[0, h1] * actsample.hex2_dist.probs[0, h2]
            p1 = topk.values[0, k]
            print("\t[k=%d] p0 = %.3f | p1 = %.3f" % (k, p0, p1))

            # These are not always that close (sometimes diff is almost 0.1)
            # if not torch.allclose(p0, p1, atol=0.01):
            #     import ipdb; ipdb.set_trace()  # noqa
            #     pass

        venv.step([action])

    print("Total execution time: %dms" % ms_total)
    print("Model role: %s" % cfg["train"]["env"]["kwargs"]["role"])
    print("verify_export: OK")


def load_exported_model(m):
    return ort.InferenceSession(m)


def save_exported_model(m, export_dir, basename):
    dst = f"{export_dir}/{basename}.onnx"  # extension is based on exptype
    with open(dst, "wb") as f:
        f.write(m)

    print("Wrote %s" % dst)


def main():
    MODEL_PREFIXES = [
        # "nkjrmrsq-202509231549",
        # "nkjrmrsq-202509252116",
        # "nkjrmrsq-202509291846",
        # "tukbajrv-202509171940",
        # "tukbajrv-202509211112",
        # "tukbajrv-202509222128",
        # "tukbajrv-202509241418",
        # "lcfcwxbc-202510020051",
        # "aspnnqwg-1762370851",
        # "rqqartou-202511050135"  # overfitted
        # "rqqartou-1761858383",   # overfitted
        # "rqqartou-1761771948",
        # "sjigvvma-202511011415"
        # "gophftwt-1766488041"
        "gophftwt-1766837365"
    ]

    with torch.inference_mode():
        for prefix in MODEL_PREFIXES:
            model_cfg_path = f"{prefix}-config.json"
            model_weights_path = f"{prefix}-model-dna.pt"
            export_dir = "/Users/simo/Library/Application Support/vcmi/Mods/mmai/models"

            with open(model_cfg_path, "r") as f:
                cfg = json.load(f)

            export_basename = "%s-%s" % (cfg["train"]["env"]["kwargs"]["role"], prefix)

            #
            # Tests (for debugging):
            #

            # test_gnn()
            # test_block()
            test_model(cfg, model_weights_path)
            assert 0
            # # test_quantized(cfg, model_weights_path)
            # test_load(cfg, model_weights_path, exptype)

            #
            # Actual export
            #

            # exported_model = export_model(cfg, model_weights_path)
            # loaded_model = load_exported_model(exported_model)
            # loaded_model = load_exported_model(exptype, "/Users/simo/Projects/vcmi-play/Mods/MMAI/models/defender-sjigvvma-202511011415.onnx")
            # import ipdb; ipdb.set_trace()  # noqa
            verify_export(cfg, model_weights_path, loaded_model)

            # save_exported_model(exported_model, export_dir, export_basename)


if __name__ == "__main__":
    main()
