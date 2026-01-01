import json
import io
import torch
import torch_geometric
from time import perf_counter_ns
from torch_geometric.data import Batch
import numpy as np

import onnx
import onnxruntime as ort

from .export_common import (
    ALL_MODEL_SIZES,
    LINK_TYPES,
    ModelSizelessWrapper,
    ExportableGENConv,
    ExportableGNNBlock,
    ExportableDNAModel,
    transform_key,
    build_edge_inputs,
)

from .mppo_dna_gnn import DNAModel, GNNBlock, CategoricalMasked
from .dual_vec_env import to_hdata_list, DualVecEnv


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

    # Emax, Kmax for lt1
    model_sizes = torch.tensor([[5, 7]], dtype=ALL_MODEL_SIZES.dtype)

    inputs = (hd["hex"].x, hd["hex", "lt1", "hex"].edge_index, hd["hex", "lt1", "hex"].edge_attr)
    myinputs = (hd["hex"].x, *build_edge_inputs(hd, model_sizes))

    res = gen(*inputs)
    myres = mygen(*myinputs)

    raise NotImplementedError()


def test_block():
    """ Tests GNNBlock vs ExportableGNNBlock. """

    N = 3       # num_nodes
    hd = torch_geometric.data.HeteroData()
    hd['baba'].x = torch.randn(N, 5)
    edge_index_lt1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr_lt1 = torch.tensor([[1.0], [1.0]], dtype=torch.float)
    edge_index_lt2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    edge_attr_lt2 = torch.tensor([[0.5], [0.2]], dtype=torch.float)
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

    all_model_sizes = torch.tensor([
        [
            [10, 5],        # lt1
            [5, 4],         # lt2
        ],
        [
            [20, 10],       # lt1
            [8, 9],         # lt2
        ],
    ])

    block = GNNBlock(
        num_layers=num_layers,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        link_types=link_types,
        node_type=node_type,
    ).eval()

    myblock = ExportableGNNBlock(
        num_layers=num_layers,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        all_sizes=all_model_sizes,
        link_types=link_types,
    ).eval()

    mydict = {transform_key(k, node_type, link_types): v for k, v in block.state_dict().items()}
    myblock.load_state_dict(mydict, strict=True)

    # Test with two different "sizes"
    myinputs0 = (hd["baba"].x, *build_edge_inputs(hd, all_model_sizes[0]), 0)
    myinputs1 = (hd["baba"].x, *build_edge_inputs(hd, all_model_sizes[1]), 1)

    # import ipdb; ipdb.set_trace()  # noqa

    res = block(hd)["baba"]
    myres0 = myblock(*myinputs0)
    myres1 = myblock(*myinputs1)

    assert torch.equal(res, myres0)
    assert torch.equal(res, myres1)
    print("test_block: OK")

    raise NotImplementedError()


def test_model(cfg, weights_file):
    """ Tests DNA Model vs ExecuTorchModel. """
    # venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    # venv = DualVecEnv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs_stupidai=1)
    venv = DualVecEnv(dict(mapname="gym/archangels.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExportableDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    # obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
    # done = torch.tensor(venv.call("terminated"))
    # links = [venv.call("obs")[0]["links"]]
    # hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

    v_obs, v_info = venv.reset()
    v_done = np.zeros([venv.num_envs], dtype=bool)

    torch.set_printoptions(sci_mode=False, precision=6)

    hdata_list = to_hdata_list(torch.as_tensor(v_obs), torch.as_tensor(v_done), venv.call("links"))
    hdata = Batch.from_data_list(hdata_list)

    actsample = model.get_action_logits(hdata).sample(deterministic=True)

    # XXX: limit to first 2 sizes only (XNN export is very slow)
    # all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]
    all_edge_inputs = [
        build_edge_inputs(hdata, ALL_MODEL_SIZES[0]),
        build_edge_inputs(hdata, ALL_MODEL_SIZES[1]),
    ]

    for i, edge_inputs in enumerate(all_edge_inputs):
        print("Testing size %d..." % i)
        einputs = (hdata.obs[0], *edge_inputs, ALL_MODEL_SIZES[i])
        for i1, arg in enumerate(einputs):
            print(f"Arg {i1}: ", end="")
            if isinstance(arg, torch.Tensor):
                print(f"tensor: {arg.shape}")
            else:
                print(f"{arg.__class__.__name__}: {arg}")

        (
            action,
            act0_logits,
            hex1_logits,
            hex2_logits,
            mask_act0,
            mask_hex1,
            mask_hex2,
            act0,
            hex1,
            hex2
        ) = emodel.predict_with_logits(*einputs)

        assert torch.equal(actsample.action, action)
        assert torch.equal(actsample.act0, act0)
        assert torch.equal(actsample.hex1, hex1)
        assert torch.equal(actsample.hex2, hex2)

        # Weird thing: logits may be different, but probs are ultimately the same!
        # err_act0_logits = (actsample.act0_logits - act0_logits) / actsample.act0_logits
        # err_hex1_logits = (actsample.hex1_logits - hex1_logits) / actsample.hex1_logits
        # err_hex2_logits = (actsample.hex2_logits - hex2_logits) / actsample.hex2_logits
        # print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
        # print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
        # print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

        m0 = mask_act0.bool()
        m1 = mask_hex1.bool()
        m2 = mask_hex2.bool()

        act0_dist = CategoricalMasked(logits=act0_logits, mask=m0)
        hex1_dist = CategoricalMasked(logits=hex1_logits, mask=m1[0, act0])
        hex2_dist = CategoricalMasked(logits=hex2_logits, mask=m2[0, act0, hex1])

        err_act0_probs = (actsample.act0_dist.probs - act0_dist.probs) / actsample.act0_dist.probs.clamp_min(1e-8)
        err_hex1_probs = (actsample.hex1_dist.probs - hex1_dist.probs) / actsample.hex1_dist.probs.clamp_min(1e-8)
        err_hex2_probs = (actsample.hex2_dist.probs - hex2_dist.probs) / actsample.hex2_dist.probs.clamp_min(1e-8)

        print("Relative error (probs): act0: mean=%.6f, max=%.6f" % (err_act0_probs.mean(), err_act0_probs.max()))
        print("Relative error (probs): hex1: mean=%.6f, max=%.6f" % (err_hex1_probs.mean(), err_hex1_probs.max()))
        print("Relative error (probs): hex2: mean=%.6f, max=%.6f" % (err_hex2_probs.mean(), err_hex2_probs.max()))

        # 1e-5 = 0.001%
        assert torch.allclose(actsample.act0_dist.probs, act0_dist.probs, rtol=1e-5, atol=1e-5)
        assert torch.allclose(actsample.hex1_dist.probs, hex1_dist.probs, rtol=1e-5, atol=1e-5)
        assert torch.allclose(actsample.hex2_dist.probs, hex2_dist.probs, rtol=1e-5, atol=1e-5)

        print("(size=%d) test_model: OK" % i)

    print("=== JIT transform ===")
    sw = ModelSizelessWrapper(emodel)
    ssw = torch.jit.script(sw)
    ssw.eval()

    print("=== ONNX transform ===")
    buffer = io.BytesIO()

    torch.onnx.export(
        ssw,
        (hdata.obs[0], *all_edge_inputs[0], ALL_MODEL_SIZES[0]),
        buffer,
        input_names=["obs", "ei_flat", "ea_flat", "nbr_flat", "size"],
        output_names=[
            "action",
            "act0_logits",
            "hex1_logits",
            "hex2_logits",
            "mask_act0",
            "mask_hex1",
            "mask_hex2",
            "act0",
            "hex1",
            "hex2",
        ],
        opset_version=16,  # onnxruntime 1.11+
        do_constant_folding=True,
        dynamic_axes={
            "ei_flat": {1: "ei_dim"},       # S=[2, 1646], M=[2, 2478], ...
            "ea_flat": {0: "ea_dim"},       # S=[1646, 1], M=[2478, 1], ...
            "nbr_flat": {1: "nbr_dim"},     # S=[165, 32], M=[165, 52], ...
        },
        # XXX: dynamo is the *new* torch ONNX exporter and will become the
        #       default in torch-2.9.0, however as of torch 2.8.0 there are
        #       missing operator implementations, and 2.9.0 is not viable
        #       as torch_geometric segfaults (it is still on 2.8.0)
        # dynamo=True
    )

    edge = ort.InferenceSession(buffer.getvalue())

    predict_time_total = 0
    predict_count_total = 0

    for i, edge_inputs in enumerate(all_edge_inputs):
        print("Testing size %d..." % i)
        einputs = (hdata.obs[0], *edge_inputs, ALL_MODEL_SIZES[i])

        for i1, arg in enumerate(einputs):
            print(f"Arg {i1}: ", end="")
            if isinstance(arg, torch.Tensor):
                print(f"tensor: {arg.shape}")
            else:
                print(f"{arg.__class__.__name__}: {arg}")

        t0 = perf_counter_ns()
        named_einputs = {n: einputs[i].numpy() for i, n in enumerate(["obs", "ei_flat", "ea_flat", "nbr_flat", "size"])}
        call = lambda: [torch.as_tensor(x) for x in edge.run(None, named_einputs)]

        (
            action,
            act0_logits,
            hex1_logits,
            hex2_logits,
            mask_act0,
            mask_hex1,
            mask_hex2,
            act0,
            hex1,
            hex2
        ) = call()

        ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
        print("Predict time: %d ms" % ms)
        predict_time_total += ms
        predict_count_total += 1

        assert torch.equal(actsample.action, action)
        assert torch.equal(actsample.act0, act0)
        assert torch.equal(actsample.hex1, hex1)
        assert torch.equal(actsample.hex2, hex2)

        err_act0_logits = (actsample.act0_logits - act0_logits) / actsample.act0_logits
        err_hex1_logits = (actsample.hex1_logits - hex1_logits) / actsample.hex1_logits
        err_hex2_logits = (actsample.hex2_logits - hex2_logits) / actsample.hex2_logits

        print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
        print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
        print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

        assert err_act0_logits.max() < 0.01
        assert err_hex1_logits.max() < 0.01
        assert err_hex2_logits.max() < 0.01

        print("(size=%d) test_model: OK" % i)

    print("predict_time_total=%d" % predict_time_total)
    print("predict_count_total=%d" % predict_count_total)
    print("ms_per_perdiction=%d" % (predict_time_total / predict_count_total))


def test_load(cfg, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExportableDNAModel. """

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExportableDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
    done = torch.tensor([False])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

    # XXX: test only with size "S" (faster)
    einputs = (obs[0], *build_edge_inputs(hdata, ALL_MODEL_SIZES[0]))

    for i1, arg in enumerate(einputs):
        print(f"Arg {i1}: ", end="")
        if isinstance(arg, torch.Tensor):
            print(f"tensor: {arg.shape}")
        else:
            print(f"{arg.__class__.__name__}: {arg}")

    actdata = model.get_actdata_eval(hdata, deterministic=True)

    # XXX: test only with size "S" (quantizing is very slow)
    einputs = (obs[0], *build_edge_inputs(hdata, ALL_MODEL_SIZES[0]))

    raise NotImplementedError()


def export_model(cfg, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExportableDNAModel. """

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel0 = ExportableDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel0.load_state_dict(eweights, strict=True)
    emodel = emodel0.model_policy.eval()

    obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
    done = torch.tensor([False])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

    all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]

    # XXX: it should work OK if args_tail == einputs' model size
    #       it returns incorrect result if args_tail < einputs' model size
    #       it fails with RuntimeError: index_select(): ... otherwise

    sw = ModelSizelessWrapper(emodel)
    ssw = torch.jit.script(sw)
    ssw.eval()

    print("=== ONNX transform ===")
    buffer = io.BytesIO()

    torch.onnx.export(
        ssw,
        (obs[0], *all_edge_inputs[0], ALL_MODEL_SIZES[0]),
        buffer,
        input_names=["obs", "ei_flat", "ea_flat", "nbr_flat", "size"],
        output_names=[
            "action",
            "act0_logits",
            "hex1_logits",
            "hex2_logits",
            "mask_act0",
            "mask_hex1",
            "mask_hex2",
            "act0",
            "hex1",
            "hex2",
        ],
        opset_version=20,  # onnxruntime 1.17+
        do_constant_folding=True,
        dynamic_axes={
            "ei_flat": {1: "ei_dim"},       # S=[2, 1646], M=[2, 2478], ...
            "ea_flat": {0: "ea_dim"},       # S=[1646, 1], M=[2478, 1], ...
            "nbr_flat": {1: "nbr_dim"},     # S=[165, 32], M=[165, 52], ...
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
        "all_sizes": json.dumps(emodel.get_all_sizes().tolist()),
        "version": str(emodel.get_version().item()),
        "side": str(emodel.get_side().item()),
        "action_table": json.dumps(emodel.action_table.tolist()),
        "is_dynamic": "1",
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
    assert json.loads(md["all_sizes"]) == ALL_MODEL_SIZES.tolist()
    assert json.loads(md["action_table"]) == model.model_policy.action_table.tolist()

    print("Testing data methods for %d steps..." % (num_steps))

    # XXX: 4x1024.vmap will cause errors in build_edge_inputs for smaller sizes
    #       (which is OK, but it's better to test all sizes)
    # venv = DualVecEnv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs_stupidai=1)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role=cfg["train"]["env"]["kwargs"]["role"]), num_envs_stupidai=1)
    venv.reset()

    benchmarks = torch.zeros(len(ALL_MODEL_SIZES))

    for n in range(num_steps):
        print(venv.render()[0])

        obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
        done = torch.tensor([False])
        links = [venv.call("obs")[0]["links"]]
        hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

        actdata = model.model_policy.get_actdata_eval(hdata, deterministic=True)

        all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]

        for i, edge_inputs in enumerate(all_edge_inputs):
            einputs = (obs[0], *edge_inputs, ALL_MODEL_SIZES[i])
            t0 = perf_counter_ns()
            named_einputs = {n: einputs[i].numpy() for i, n in enumerate(["obs", "ei_flat", "ea_flat", "nbr_flat", "size"])}
            loaded_res = [torch.as_tensor(x) for x in loaded_model.run(None, named_einputs)]

            ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
            benchmarks[i] += ms

            (
                action,
                act0_logits,
                hex1_logits,
                hex2_logits,
                mask_act0,
                mask_hex1,
                mask_hex2,
                act0,
                hex1,
                hex2
            ) = loaded_res

            print("(step=%d, size=%d) TEST ACTION: %d <> %d (%s ms)" % (n, i, actdata.action, action.item(), ms))

            err_act0_logits = (actdata.act0_logits - act0_logits) / actdata.act0_logits
            err_hex1_logits = (actdata.hex1_logits - hex1_logits) / actdata.hex1_logits
            err_hex2_logits = (actdata.hex2_logits - hex2_logits) / actdata.hex2_logits

            print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
            print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
            print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

            assert actdata.action.item() == action.item()
            assert actdata.act0.item() == act0.item()
            assert actdata.hex1.item() == hex1.item()
            assert actdata.hex2.item() == hex2.item()

            assert err_act0_logits.max() < 0.01
            assert err_hex1_logits.max() < 0.01
            assert err_hex2_logits.max() < 0.01

            # Not testing value (value model excluded)
            # value = model.get_value(hdata)[0]
            # myvalue = loaded_get_value.execute(einputs)
            # print("(%d) TEST VALUE: %.3f <> %.3f" % (n, value.item(), myvalue.item()))

        venv.step([actdata.action])

    print("Total execution time:")

    for i, ms in enumerate(benchmarks):
        print("  %d: %d ms" % (i, ms.item()))

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

            # test_gnn(exptype)
            # test_block(exptype)
            test_model(cfg, model_weights_path)
            assert 0
            # # test_quantized(cfg, model_weights_path)
            # test_load(cfg, model_weights_path, exptype)

            #
            # Actual export
            #

            exported_model = export_model(cfg, model_weights_path)
            loaded_model = load_exported_model(exported_model)
            # loaded_model = load_exported_model(exptype, "/Users/simo/Projects/vcmi-play/Mods/MMAI/models/defender-sjigvvma-202511011415.onnx")
            # import ipdb; ipdb.set_trace()  # noqa
            verify_export(cfg, model_weights_path, loaded_model)

            save_exported_model(exported_model, export_dir, export_basename)


if __name__ == "__main__":
    main()
