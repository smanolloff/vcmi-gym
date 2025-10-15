import json
import io
import torch
import enum
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch_geometric
from time import perf_counter_ns
from torch_geometric.data import Batch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.export import export, export_for_training
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import coremltools
from executorch.runtime import Runtime

from .export_common import (
    ALL_MODEL_SIZES,
    LINK_TYPES,
    ModelWrapper,
    ExportableGENConv,
    ExportableGNNBlock,
    ExportableDNAModel,
    HardcodedModelWrapper,
    transform_key,
    pad_edges,
    build_nbr,
    build_edge_inputs,
)

from .mppo_dna_gnn import DNAModel, GNNBlock
from .dual_vec_env import to_hdata_list, DualVecEnv


# coreML requires dummy inputs...
DUMMY_INPUTS = (torch.tensor([0], dtype=torch.int32),)


class ExportType(enum.IntEnum):
    EXECUTORCH = 0
    LIBTORCH = enum.auto()


def test_gnn(exptype):
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

    print("ExportType type: %s" % exptype.name)
    if exptype == ExportType.EXECUTORCH:
        ep = {"forward": export(mygen, myinputs, strict=True)}

        print("=== XNN transform ===")
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

        expres = edge.exported_program("forward").module()(*myinputs)
    else:
        ts = torch.jit.script(mygen, example_inputs=myinputs)
        expres = ts(*myinputs)

    assert torch.equal(res, myres)
    assert torch.equal(res, expres)
    print("test_gnn: OK")


def test_block(exptype):
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

    print("ExportType type: %s" % exptype.name)
    if exptype == ExportType.EXECUTORCH:
        ep = {
            "forward0": export(myblock, myinputs0, strict=True),
            "forward1": export(myblock, myinputs1, strict=True),
        }

        print("=== XNN transform ===")
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

        exported_forward0 = edge.exported_program("forward0").module()
        exported_forward1 = edge.exported_program("forward1").module()
        expres0 = exported_forward0(*myinputs0)
        expres1 = exported_forward1(*myinputs1)
    else:
        ts = torch.jit.script(myblock)
        expres0 = ts(*myinputs0)
        expres1 = ts(*myinputs1)

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    assert torch.equal(res, expres0)
    assert torch.equal(res, expres1)


def test_model(cfg, weights_file, exptype):
    """ Tests DNA Model vs ExecuTorchModel. """
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
    result = torch.tensor([0])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, result, links))

    actdata = model.get_actdata_eval(hdata, deterministic=True)

    # XXX: limit to first 2 sizes only (XNN export is very slow)
    # all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]
    all_edge_inputs = [
        build_edge_inputs(hdata, ALL_MODEL_SIZES[0]),
        build_edge_inputs(hdata, ALL_MODEL_SIZES[1]),
    ]

    for i, edge_inputs in enumerate(all_edge_inputs):
        print("Testing size %d..." % i)
        einputs = (obs[0], *edge_inputs, i)
        for i1, arg in enumerate(einputs):
            print(f"Arg {i1}: ", end="")
            if isinstance(arg, torch.Tensor):
                print(f"tensor: {arg.shape}")
            else:
                print(f"{arg.__class__.__name__}: {arg}")

        action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = emodel._predict_with_logits(*einputs)

        # import ipdb; ipdb.set_trace()  # noqa
        assert torch.equal(actdata.action, action)
        assert torch.equal(actdata.act0_logits, act0_logits)
        assert torch.equal(actdata.act0, act0)
        assert torch.equal(actdata.hex1_logits, hex1_logits)
        assert torch.equal(actdata.hex1, hex1)
        assert torch.equal(actdata.hex2_logits, hex2_logits)
        assert torch.equal(actdata.hex2, hex2)
        print("(size=%d) test_model: OK" % i)

    # XXX: it should work OK if args_tail == einputs' model size
    #       it returns incorrect result if args_tail < einputs' model size
    #       it fails with RuntimeError: index_select(): ... otherwise

    hmw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES[:2])

    print("Exporting (exptype: %s)" % exptype.name)
    if exptype == ExportType.EXECUTORCH:
        for i, edge_inputs in enumerate(all_edge_inputs):
            getattr(hmw, f"predict{i}")(obs[0], *edge_inputs)
            getattr(hmw, f"_predict_with_logits{i}")(obs[0], *edge_inputs)

        ep = {}

        w = ModelWrapper(hmw, "get_all_sizes")
        ep["get_all_sizes"] = export(w, DUMMY_INPUTS, strict=True)

        w = ModelWrapper(hmw, "get_version")
        ep["get_version"] = export(w, DUMMY_INPUTS, strict=True)

        w = ModelWrapper(hmw, "get_side")
        ep["get_side"] = export(w, DUMMY_INPUTS, strict=True)

        for i, edge_inputs in enumerate(all_edge_inputs):
            einputs = (obs[0], *edge_inputs)
            w = ModelWrapper(hmw, f"_predict_with_logits{i}")
            ep[f"_predict_with_logits{i}"] = export(w, einputs, strict=True)

        # # XNNPACK
        # # slow at runtime :(
        # print("=== XNNPACK transform ===")
        # edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

        # # CoreML
        # # Example: force iOS16 to avoid CoreML7 (which requires iOS17/macOS14)
        # print("=== CoreML transform ===")
        # coreml_specs = CoreMLBackend.generate_compile_specs(
        #     minimum_deployment_target=coremltools.target.iOS16,   # choose iOS15/16 as needed
        #     compute_unit=coremltools.ComputeUnit.ALL,             # optional
        #     compute_precision=coremltools.precision.FLOAT16       # optional
        # )
        # edge = to_edge_transform_and_lower(ep, partitioner=[CoreMLPartitioner(
        #     compile_specs=coreml_specs,
        #     # If export still picks a newer opset, remove them from the model or allow partial delegation:
        #     # lower_full_graph=False
        # )])

        # Vulkan
        # Android 7.0+ with Vulkan driver. Performance varies by GPU and driver.
        # No "min API" knob in export; gate at install (minSdkVersion) and at runtime via Vulkan presence.
        print("=== Vulkan transform ===")
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])

        print("=== Test save/load ===")
        exported_model = edge.to_executorch()
        load_exported_model(exptype, exported_model)
        # import ipdb; ipdb.set_trace()  # noqa
        # pass
    else:
        print("=== JIT transform ===")
        mw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES).eval().cpu()
        jw = torch.jit.script(mw)

        for i, edge_inputs in enumerate(all_edge_inputs):
            getattr(jw, f"predict{i}")(obs[0], *edge_inputs)
            getattr(jw, f"_predict_with_logits{i}")(obs[0], *edge_inputs)
        print("Optimizing...")
        method_names = [f"_predict_with_logits{i}" for i in range(len(ALL_MODEL_SIZES))]
        method_names.extend([f"predict{i}" for i in range(len(ALL_MODEL_SIZES))])
        edge = optimize_for_mobile(jw, preserved_methods=method_names)

    predict_time_total = 0
    predict_count_total = 0

    for i, edge_inputs in enumerate(all_edge_inputs):
        print("Testing size %d..." % i)
        einputs = (obs[0], *edge_inputs)

        for i1, arg in enumerate(einputs):
            print(f"Arg {i1}: ", end="")
            if isinstance(arg, torch.Tensor):
                print(f"tensor: {arg.shape}")
            else:
                print(f"{arg.__class__.__name__}: {arg}")

        method_name = f"_predict_with_logits{i}"

        t0 = perf_counter_ns()
        if exptype == ExportType.EXECUTORCH:
            program = edge.exported_program(method_name).module()
            action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = program(*einputs)
        else:
            action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = getattr(edge, method_name)(*einputs)

        ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
        print("Predict time: %d ms" % ms)
        predict_time_total += ms
        predict_count_total += 1

        assert torch.equal(actdata.action, action)
        assert torch.equal(actdata.act0, act0)
        assert torch.equal(actdata.hex1, hex1)
        assert torch.equal(actdata.hex2, hex2)

        err_act0_logits = (actdata.act0_logits - act0_logits) / actdata.act0_logits
        err_hex1_logits = (actdata.hex1_logits - hex1_logits) / actdata.hex1_logits
        err_hex2_logits = (actdata.hex2_logits - hex2_logits) / actdata.hex2_logits

        print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
        print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
        print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

        assert err_act0_logits.max() < 1e-4
        assert err_hex1_logits.max() < 1e-4
        assert err_hex2_logits.max() < 1e-4

        print("(size=%d) test_model: OK" % i)

    print("predict_time_total=%d" % predict_time_total)
    print("predict_count_total=%d" % predict_count_total)
    print("ms_per_perdiction=%d" % (predict_time_total / predict_count_total))


def test_quantized(cfg, weights_file, exptype):
    """ Tests DNAModel vs the XNN-lowered-and-quantized ExportableDNAModel. """

    if exptype != ExportType.EXECUTORCH:
        print("test_quantized is supported only for EXECUTORCH exports")
        return

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
    result = torch.tensor([0])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, result, links))

    # einputs = build_einputs(hdata, E_MAX, K_MAX)
    # for i, arg in enumerate(einputs):
    #     print("Arg %d shape: %s" % (i, arg.shape))

    # XXX: test only with size "S" (quantizing is very slow)
    einputs = (obs[0], *build_edge_inputs(hdata, ALL_MODEL_SIZES[0]))

    hmw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES[:1])
    m__predict_with_logits = ModelWrapper(hmw, "_predict_with_logits0")

    print("Quantizing...")
    # Quantizer
    # XXX: is_per_channel seems to have no effect on model accuracy
    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=False))

    # --- PT2E prepare/convert ---
    # export_for_training -> .module() for PT2E helpers
    trainable__predict_with_logits = export_for_training(m__predict_with_logits, einputs, strict=True).module()

    # Insert observers
    prepared__predict_with_logits = prepare_pt2e(trainable__predict_with_logits, q)

    # Calibrate
    print("Calibrating...")
    _ = prepared__predict_with_logits(*einputs)

    # Convert to quantized modules
    converted__predict_with_logits = convert_pt2e(prepared__predict_with_logits)

    print("Exporting...")
    ep = {
        "_predict_with_logits": export(converted__predict_with_logits, einputs, strict=True),
    }

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

    exported__predict_with_logits = edge.exported_program("_predict_with_logits").module()

    print("Testing...")
    actdata = model.get_actdata_eval(hdata, deterministic=True)
    action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = exported__predict_with_logits(*einputs)

    err_act0_logits = (actdata.act0_logits - act0_logits) / actdata.act0_logits
    err_hex1_logits = (actdata.hex1_logits - hex1_logits) / actdata.hex1_logits
    err_hex2_logits = (actdata.hex2_logits - hex2_logits) / actdata.hex2_logits

    print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
    print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
    print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

    assert torch.equal(actdata.action, action)
    assert torch.equal(actdata.act0, act0)
    assert torch.equal(actdata.hex1, hex1)
    assert torch.equal(actdata.hex2, hex2)

    assert err_act0_logits.max() < 1e-4
    assert err_hex1_logits.max() < 1e-4
    assert err_hex2_logits.max() < 1e-4

    print("test_quantized: OK")


def test_load(cfg, weights_file, exptype):
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
    result = torch.tensor([0])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, result, links))

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

    hmw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES[:1])

    print("ExportType type: %s" % exptype.name)
    if exptype == ExportType.EXECUTORCH:
        mw = ModelWrapper(hmw, "_predict_with_logits0")
        ep = {"_predict_with_logits0": export(mw, einputs, strict=True)}

        print("=== XNN transform ===")
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

        rt = Runtime.get()
        print("Loading...")
        loaded = rt.load_program(edge.to_executorch().buffer)
        loaded_predict_with_logits = loaded.load_method("_predict_with_logits0")
        print("Testing...")
        action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = loaded_predict_with_logits.execute(einputs)
    else:
        print("=== JIT transform ===")
        jw = torch.jit.script(hmw)
        method_names = [f"_predict_with_logits{i}" for i in range(len(ALL_MODEL_SIZES))]
        method_names.extend([f"predict{i}" for i in range(len(ALL_MODEL_SIZES))])
        edge = optimize_for_mobile(jw, preserved_methods=method_names)
        print("Loading...")
        loaded = torch.jit.load(io.BytesIO(edge._save_to_buffer_for_lite_interpreter()))
        action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = loaded._predict_with_logits0(*einputs)

    err_act0_logits = (actdata.act0_logits - act0_logits) / actdata.act0_logits
    err_hex1_logits = (actdata.hex1_logits - hex1_logits) / actdata.hex1_logits
    err_hex2_logits = (actdata.hex2_logits - hex2_logits) / actdata.hex2_logits

    print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
    print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
    print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

    assert torch.equal(actdata.action, action)
    assert torch.equal(actdata.act0, act0)
    assert torch.equal(actdata.hex1, hex1)
    assert torch.equal(actdata.hex2, hex2)

    assert err_act0_logits.max() < 1e-4
    assert err_hex1_logits.max() < 1e-4
    assert err_hex2_logits.max() < 1e-4

    print("test_load: OK")


def export_model(cfg, weights_file, exptype, is_tiny):
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
    result = torch.tensor([0])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, result, links))

    all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]

    # XXX: it should work OK if args_tail == einputs' model size
    #       it returns incorrect result if args_tail < einputs' model size
    #       it fails with RuntimeError: index_select(): ... otherwise

    programs = {}

    print("NOTE: Using get_value from policy model to reduce export size")

    hmw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES, emodel0.model_value)

    print("Exporting (exptype: %s)" % exptype.name)
    if exptype == ExportType.EXECUTORCH:
        for i, edge_inputs in enumerate(all_edge_inputs):
            einputs = (obs[0], *edge_inputs)
            w = ModelWrapper(hmw, f"predict{i}")
            programs[f"predict{i}"] = export(w, einputs, strict=True)
            if not is_tiny:
                w = ModelWrapper(hmw, f"get_value{i}")
                programs[f"get_value{i}"] = export(w, einputs, strict=True)
                w = ModelWrapper(hmw, f"_predict_with_logits{i}")
                programs[f"_predict_with_logits{i}"] = export(w, einputs, strict=True)
        w = ModelWrapper(hmw, "get_version")
        programs["get_version"] = export(w, DUMMY_INPUTS, strict=True)
        w = ModelWrapper(hmw, "get_side")
        programs["get_side"] = export(w, DUMMY_INPUTS, strict=True)
        w = ModelWrapper(hmw, "get_all_sizes")
        programs["get_all_sizes"] = export(w, DUMMY_INPUTS, strict=True)

        # for name, program in programs.items():
        #     print("Program: %s" % name)
        #     for n in program.graph.nodes:
        #         print(str(n.target))
        #         if "aten.slice_scatter" in str(n.target) or "copy_" in str(n.target):
        #             print("FOUND SCATTER: %s" % n.format_node())  # shows producer, dtype, shapes

        # # XNNPACK
        # print("=== XNNPACK transform ===")
        # edge = to_edge_transform_and_lower(programs, partitioner=[XnnpackPartitioner()])

        # # CoreML
        # # Example: force iOS16 to avoid CoreML7 (which requires iOS17/macOS14)
        # print("=== CoreML transform ===")
        # coreml_specs = CoreMLBackend.generate_compile_specs(
        #     minimum_deployment_target=coremltools.target.iOS16,   # choose iOS15/16 as needed
        #     compute_unit=coremltools.ComputeUnit.ALL,             # optional
        #     compute_precision=coremltools.precision.FLOAT16       # optional
        # )
        # edge = to_edge_transform_and_lower(programs, partitioner=[CoreMLPartitioner(
        #     compile_specs=coreml_specs,
        #     # If export still picks a newer opset, remove them from the model or allow partial delegation:
        #     # lower_full_graph=False
        # )])

        # Vulkan
        # Android 7.0+ with Vulkan driver. Performance varies by GPU and driver.
        # No "min API" knob in export; gate at install (minSdkVersion) and at runtime via Vulkan presence.
        print("=== Vulkan transform ===")
        edge = to_edge_transform_and_lower(programs, partitioner=[VulkanPartitioner()])

        print("Exported programs:\n  %s" % "\n  ".join(list(programs.keys())))
        exported_model = edge.to_executorch()
    else:
        print("=== JIT transform ===")
        jw = torch.jit.script(hmw)
        for i, edge_inputs in enumerate(all_edge_inputs):
            getattr(jw, f"predict{i}")(obs[0], *edge_inputs)
            if not is_tiny:
                # getattr(jw, f"get_value{i}")(obs[0], *edge_inputs)
                getattr(jw, f"_predict_with_logits{i}")(obs[0], *edge_inputs)
        method_names = [f"predict{i}" for i in range(len(ALL_MODEL_SIZES))]
        if not is_tiny:
            method_names.extend([f"_predict_with_logits{i}" for i in range(len(ALL_MODEL_SIZES))])
            method_names.extend([f"get_value{i}" for i in range(len(ALL_MODEL_SIZES))])
        method_names.extend(["get_version"])
        method_names.extend(["get_side"])
        method_names.extend(["get_all_sizes"])
        edge = optimize_for_mobile(jw, preserved_methods=method_names)
        print("Exported methods:\n  %s" % "\n  ".join(method_names))
        exported_model = optimize_for_mobile(edge, preserved_methods=method_names)

    return exported_model


def verify_export(cfg, weights_file, exptype, loaded_model, is_tiny, num_steps=10):
    weights = torch.load(weights_file, weights_only=True, map_location="cpu")
    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]

    print("Testing metadata methods (%s)..." % exptype)
    if exptype == ExportType.EXECUTORCH:
        loaded_methods = {name: loaded_model.load_method(name) for name in loaded_model.method_names}
        if is_tiny:
            # 3 metadata methods + 1*sizes methods (predict)
            assert len(loaded_methods) == 3 + 1*len(ALL_MODEL_SIZES), len(loaded_methods)
        else:
            # 3 metadata methods + 3*sizes methods (get_value, predict, _predict_with_logits)
            assert len(loaded_methods) == 3 + 3*len(ALL_MODEL_SIZES), len(loaded_methods)
        assert loaded_methods["get_version"].execute(DUMMY_INPUTS)[0].item() == 13
        assert loaded_methods["get_side"].execute(DUMMY_INPUTS)[0].item() == eside
        assert torch.equal(loaded_methods["get_all_sizes"].execute(DUMMY_INPUTS)[0], ALL_MODEL_SIZES)
    else:
        assert loaded_model.get_version().item() == 13
        assert loaded_model.get_side().item() == eside
        assert torch.equal(loaded_model.get_all_sizes(), ALL_MODEL_SIZES)

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
        result = torch.tensor([0])
        links = [venv.call("obs")[0]["links"]]
        hdata = Batch.from_data_list(to_hdata_list(obs, done, result, links))

        actdata = model.model_policy.get_actdata_eval(hdata, deterministic=True)

        all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]

        for i, edge_inputs in enumerate(all_edge_inputs):
            einputs = (obs[0], *edge_inputs)
            t0 = perf_counter_ns()

            if exptype == ExportType.EXECUTORCH:
                action = loaded_methods[f"predict{i}"].execute(einputs)[0]
            else:
                action = getattr(loaded_model, f"predict{i}")(*einputs)

            ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
            benchmarks[i] += ms

            print("(step=%d, size=%d) TEST ACTION: %d <> %d (%s ms)" % (n, i, actdata.action, action.item(), ms))
            assert actdata.action == action.item()

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


def load_exported_model(exptype, m):
    is_file = isinstance(m, str)

    if exptype == ExportType.EXECUTORCH:
        loadable = m if is_file else m.buffer
        return Runtime.get().load_program(loadable)
    else:
        loadable = m if is_file else io.BytesIO(m._save_to_buffer_for_lite_interpreter())
        return torch.jit.load(loadable)


def save_exported_model(exptype, m, export_dir, basename):
    ext = "pte" if exptype == ExportType.EXECUTORCH else "ptl"
    dst = f"{export_dir}/{basename}.{ext}"
    print("Writing to %s" % dst)

    if exptype == ExportType.EXECUTORCH:
        with open(dst, "wb") as f:
            m.write_to_file(f)
    else:
        m._save_for_lite_interpreter(dst)


def main():
    MODEL_PREFIXES = [
      # "nkjrmrsq-202509231549",
      # "nkjrmrsq-202509252116",
      # "nkjrmrsq-202509291846",
      "tukbajrv-202509171940",
      # "tukbajrv-202509211112",
      # "tukbajrv-202509222128",
      # "tukbajrv-202509241418",
      # "lcfcwxbc-202510020051",
    ]

    with torch.inference_mode():
        for prefix in MODEL_PREFIXES:
            model_cfg_path = f"{prefix}-config.json"
            model_weights_path = f"{prefix}-model-dna.pt"
            export_dir = "/Users/simo/Projects/vcmi-play/Mods/MMAI/models"

            # XXX: NO extension here (added based on exptype)
            export_basename = f"{prefix}-vulkan"

            # For faster ET exports: no get_value, no _predict_with_logis
            tiny = True

            with open(model_cfg_path, "r") as f:
                cfg = json.load(f)

            export_basename = "%s-%s" % (cfg["train"]["env"]["kwargs"]["role"], export_basename)
            export_basename += ("-tiny" if tiny else "")

            exptypes = [ExportType.EXECUTORCH]
            # exptypes = [ExportType.LIBTORCH]
            # exptypes = [ExportType.EXECUTORCH, ExportType.LIBTORCH]

            for exptype in exptypes:
                #
                # Tests (for debugging):
                #

                # test_gnn(exptype)
                # test_block(exptype)
                # test_model(cfg, model_weights_path, exptype)
                # assert 0
                # # test_quantized(cfg, model_weights_path)
                # test_load(cfg, model_weights_path, exptype)

                #
                # Actual export
                #

                exported_model = export_model(cfg, model_weights_path, exptype, tiny)
                # loaded_model = load_exported_model(exptype, exported_model)
                # loaded_model = load_exported_model(exptype, "/Users/simo/Projects/vcmi-play/Mods/MMAI/models/attacker-nkjrmrsq-202509231549-tiny.pte")
                # import ipdb; ipdb.set_trace()  # noqa
                # verify_export(cfg, model_weights_path, exptype, loaded_model, tiny)

                save_exported_model(exptype, exported_model, export_dir, export_basename)


if __name__ == "__main__":
    main()
