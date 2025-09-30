import io
import json
import torch
from torch_geometric.data import Batch
from torch.export import export, export_for_training
from torch.utils.mobile_optimizer import optimize_for_mobile

from .mppo_dna_gnn import DNAModel

from .export_common import (
    ALL_MODEL_SIZES,
    LINK_TYPES,
    ModelWrapper,
    ExportableGENConv,
    ExportableGNNBlock,
    ExportableDNAModel,
    HardcodedModelWrapper,
    build_edge_inputs,
    transform_key,
)

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

    jgen = torch.jit.script(mygen, myinputs)
    jres = jgen(*myinputs)

    assert torch.equal(res, myres)
    assert torch.equal(res, jres)
    print("test_gnn: OK")


def test_block():
    """ Tests GNNBlock vs ExportableGNNBlock. """

    import torch_geometric
    from .mppo_dna_gnn import GNNBlock

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
    ], dtype=ALL_MODEL_SIZES.dtype)

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

    myinputs0 = (hd["baba"].x, *build_edge_inputs(hd, all_model_sizes[0]), 0)
    myres0 = myblock(*myinputs0)

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

    print("Testing...")

    jblock = torch.jit.script(myblock, myinputs0)
    jres0 = jblock(*myinputs0)
    jres1 = jblock(*myinputs1)
    assert torch.equal(res, jres0)
    assert torch.equal(res, jres1)

    print("test_block: OK")


def test_model(cfg_file, weights_file):
    """ Tests DNA Model vs ExportableModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

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
        assert torch.equal(actdata.hex1_logits, hex1_logits)
        assert torch.equal(actdata.hex2_logits, hex2_logits)
        print("(size=%d) test_model: OK" % i)

    print("=== JIT transform ===")
    print("Exporting...")
    mw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES).eval().cpu()
    jw = torch.jit.script(mw)

    for i, edge_inputs in enumerate(all_edge_inputs):
        getattr(jw, f"predict{i}")(obs[0], *edge_inputs)
        getattr(jw, f"_predict_with_logits{i}")(obs[0], *edge_inputs)

    print("Optimizing...")
    method_names = [f"_predict_with_logits{i}" for i in range(len(ALL_MODEL_SIZES))]
    method_names.extend([f"predict{i}" for i in range(len(ALL_MODEL_SIZES))])
    opt = optimize_for_mobile(jw, preserved_methods=method_names)

    for i, edge_inputs in enumerate(all_edge_inputs):
        einputs = (obs[0], *edge_inputs)

        for i1, arg in enumerate(einputs):
            print(f"Arg {i1}: ", end="")
            if isinstance(arg, torch.Tensor):
                print(f"tensor: {arg.shape}")
            else:
                print(f"{arg.__class__.__name__}: {arg}")

        method_name = f"_predict_with_logits{i}"
        action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = getattr(opt, method_name)(*einputs)

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

        print("Optimized (size=%d) test_model: OK" % i)


def test_load(cfg_file, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExportableDNAModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

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

    print("=== JIT transform ===")
    print("Exporting...")
    mw = HardcodedModelWrapper(emodel, eside, ALL_MODEL_SIZES).eval().cpu()
    jw = torch.jit.script(mw)

    getattr(jw, "predict0")(*einputs)
    getattr(jw, "_predict_with_logits0")(*einputs)

    print("Optimizing...")
    method_names = [f"_predict_with_logits{i}" for i in range(len(ALL_MODEL_SIZES))]
    method_names.extend([f"predict{i}" for i in range(len(ALL_MODEL_SIZES))])
    opt = optimize_for_mobile(jw, preserved_methods=method_names)

    loaded = torch.jit.load(io.BytesIO(opt._save_to_buffer_for_lite_interpreter()))

    print("Testing...")
    actdata = model.get_actdata_eval(hdata, deterministic=True)
    action, act0_logits, act0, hex1_logits, hex1, hex2_logits, hex2 = loaded._predict_with_logits0(*einputs)

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

    print("test_load: OK")


def export_model(cfg_file, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExportableDNAModel. """
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExportableDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

    eweights = {
        transform_key(k, "hex", list(LINK_TYPES)): v
        for k, v in weights.items()
    }
    emodel.load_state_dict(eweights, strict=True)

    obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
    done = torch.tensor([False])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

    all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]

    # XXX: it should work OK if args_tail == einputs' model size
    #       it returns incorrect result if args_tail < einputs' model size
    #       it fails with RuntimeError: index_select(): ... otherwise

    print("NOTE: Using get_value from policy model to reduce export size")

    print("=== JIT transform ===")
    print("Exporting...")
    mw = HardcodedModelWrapper(emodel.model_policy, eside, ALL_MODEL_SIZES).eval().cpu()
    jw = torch.jit.script(mw)

    for i, edge_inputs in enumerate(all_edge_inputs):
        getattr(jw, f"get_value{i}")(obs[0], *edge_inputs)
        getattr(jw, f"predict{i}")(obs[0], *edge_inputs)
        getattr(jw, f"_predict_with_logits{i}")(obs[0], *edge_inputs)

    print("Optimizing...")
    method_names = [f"_predict_with_logits{i}" for i in range(len(ALL_MODEL_SIZES))]
    method_names.extend([f"predict{i}" for i in range(len(ALL_MODEL_SIZES))])
    method_names.extend([f"get_value{i}" for i in range(len(ALL_MODEL_SIZES))])
    method_names.extend(["get_version"])
    method_names.extend(["get_side"])
    method_names.extend(["get_all_sizes"])

    return optimize_for_mobile(jw, preserved_methods=method_names)


def verify_export(cfg_file, weights_file, loaded_model, num_steps=10):
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")
    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]

    assert loaded_model.get_version() == 13
    assert loaded_model.get_side() == eside
    assert torch.equal(loaded_model.get_all_sizes(), ALL_MODEL_SIZES)

    # import ipdb; ipdb.set_trace()  # noqa

    print("Testing data methods for %d steps..." % (num_steps))

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    benchmarks = torch.zeros(len(ALL_MODEL_SIZES))

    from time import perf_counter_ns

    for n in range(num_steps):
        print(venv.render()[0])

        obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
        done = torch.tensor([False])
        links = [venv.call("obs")[0]["links"]]
        hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

        actdata = model.model_policy.get_actdata_eval(hdata, deterministic=True)

        all_edge_inputs = [build_edge_inputs(hdata, model_size) for model_size in ALL_MODEL_SIZES]

        for i, edge_inputs in enumerate(all_edge_inputs):
            einputs = (obs[0], *edge_inputs)

            t0 = perf_counter_ns()
            action = getattr(loaded_model, f"predict{i}")(*einputs)
            ms = (perf_counter_ns() - t0) / 1e6  # ns -> ms
            benchmarks[i] += ms

            print("(step=%d, size=%d) TEST ACTION: %d <> %d (%s ms)" % (n, i, actdata.action, action.item(), ms))
            assert actdata.action == action.item()

            # Not testing value (value model excluded)
            # value = model.get_value(hdata)
            # myvalue = loaded_model.get_value(*einputs)
            # print("(%d) TEST VALUE: %.3f <> %.3f" % (n, value.item(), myvalue.item()))

        venv.step([actdata.action])

    print("Total execution time:")

    for i, ms in enumerate(benchmarks):
        print("  %d: %d ms" % (i, ms.item()))

    import ipdb; ipdb.set_trace()  # noqa
    print("Model role: %s" % cfg["train"]["env"]["kwargs"]["role"])
    print("verify_export: OK")


if __name__ == "__main__":
    MODEL_PREFIX = "tukbajrv-202509171940"

    with torch.inference_mode():
        model_cfg_path = f"{MODEL_PREFIX}-config.json"
        model_weights_path = f"{MODEL_PREFIX}-model-dna.pt"
        export_dst = f"/Users/simo/Projects/vcmi-play/Mods/MMAI/models/{MODEL_PREFIX}.pts"

        test_gnn()
        test_block()
        test_model(model_cfg_path, model_weights_path)
        # # test_quantized(model_cfg_path, model_weights_path)
        test_load(model_cfg_path, model_weights_path)
        exported_model = export_model(model_cfg_path, model_weights_path)

        loaded_model = torch.jit.load(io.BytesIO(exported_model._save_to_buffer_for_lite_interpreter()))

        verify_export(model_cfg_path, model_weights_path, loaded_model)

        import ipdb; ipdb.set_trace()  # noqa
        print("Writing to %s" % export_dst)
        exported_model._save_for_lite_interpreter(export_dst)
