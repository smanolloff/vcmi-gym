import argparse
import copy
import io
import json
import os
import sys
import tempfile
from time import perf_counter_ns

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from torch_geometric.data import Batch

from .dual_vec_env import DualVecEnv
from .gnn_model import GNNModel, to_hdata_list, add_action_active_local_ids
from .ppo_gnn import PPOModel
from vcmi_gym.envs.v15.vcmi_env import VcmiEnv


# opset 18 is the first ONNX opset that can represent per-destination amax
# via scatter reduction. GENConv/softmax aggregation needs scatter_add and
# scatter_reduce(amax), so mirror the v14 exporter custom symbolics.
import torch.onnx.symbolic_helper as sym_help


_maybe_get_scalar = getattr(sym_help, "_maybe_get_scalar")
_maybe_get_const = getattr(sym_help, "_maybe_get_const")
_register_custom_op_symbolic = getattr(torch.onnx, "register_custom_op_symbolic")


def _get_axis_i(dim):
    axis = _maybe_get_scalar(dim)
    if axis is None:
        raise RuntimeError("scatter: dim must be constant for ONNX export")
    return int(axis)


def scatter_add_to_scatterelements(g, self, dim, index, src):
    axis_i = _get_axis_i(dim)
    return g.op("ScatterElements", self, index, src, axis_i=axis_i, reduction_s="add")


def scatter_reduce_to_scatterelements(g, self, dim, index, src, reduce, include_self):
    axis_i = _get_axis_i(dim)
    red = _maybe_get_const(reduce, "s")
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

    return g.op("ScatterElements", self, index, src, axis_i=axis_i, reduction_s=reduction)


_register_custom_op_symbolic("aten::scatter_add", scatter_add_to_scatterelements, 18)
_register_custom_op_symbolic("aten::scatter_reduce", scatter_reduce_to_scatterelements, 18)


MIN_FLOAT32 = -((2 - 2**-23) * 2**127)




def _model_edge_types(cfg):
    ignored_edges = cfg["train"]["env"]["kwargs"].get("ignored_edges", [])
    return VcmiEnv.filtered_edge_types(ignored_edges)


def _ensure_supported_cfg(cfg):
    model_cfg = cfg["model"]
    conv_cls = model_cfg.get("gnn_conv_cls", "GENConv")
    if conv_cls != "GENConv":
        raise NotImplementedError(f"rl.v15.export currently supports gnn_conv_cls='GENConv' only, got {conv_cls!r}")

    conv_kwargs = model_cfg.get("gnn_conv_kwargs", {})
    aggr = conv_kwargs.get("aggr", "softmax")
    if aggr != "softmax":
        raise NotImplementedError(f"rl.v15.export currently supports GENConv aggr='softmax' only, got {aggr!r}")

    if conv_kwargs.get("num_layers", 2) != 2:
        raise NotImplementedError("rl.v15.export currently expects GENConv num_layers=2")

    if conv_kwargs.get("norm", None) is not None:
        raise NotImplementedError("rl.v15.export currently expects GENConv norm=None")


def edge_softmax_per_dst(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int, t: torch.Tensor, eps: float = 1e-16):
    """
    ONNX-exportable equivalent of PyG SoftmaxAggregation for GENConv.

    scores: (E, H)
    dst:    (E,) int64 destination node index per edge
    t:      scalar or per-channel softmax temperature
    """

    e, h = scores.shape
    dst = dst.to(torch.int64)

    if t.numel() == 1:
        alpha_scores = scores * t
    else:
        alpha_scores = scores * t.view(1, h)

    neg_inf = torch.tensor(MIN_FLOAT32, device=scores.device, dtype=scores.dtype)
    max_per_node = neg_inf.expand(num_nodes, h).clone()
    max_per_node.scatter_reduce_(
        0,
        dst[:, None].expand(e, h),
        alpha_scores,
        reduce="amax",
        include_self=True,
    )

    exp_scores = (alpha_scores - max_per_node.index_select(0, dst)).exp()

    sum_per_node = torch.zeros((num_nodes, h), device=scores.device, dtype=scores.dtype)
    sum_per_node.scatter_add_(0, dst[:, None].expand(e, h), exp_scores)

    return exp_scores / (sum_per_node.index_select(0, dst) + eps)


def scatter_sum_per_dst(values: torch.Tensor, dst: torch.Tensor, num_nodes: int):
    e, h = values.shape
    dst = dst.to(torch.int64)

    out = torch.zeros((num_nodes, h), device=values.device, dtype=values.dtype)
    out.scatter_add_(0, dst[:, None].expand(e, h), values)
    return out


class ExportableGENConv(nn.Module):
    """Small ONNX-exportable subset of PyG GENConv used by rl.v15."""

    def __init__(self, conv):
        super().__init__()

        self.eps = conv.eps
        self.lin_src = copy.deepcopy(conv.lin_src) if hasattr(conv, "lin_src") else nn.Identity()
        self.lin_dst = copy.deepcopy(conv.lin_dst) if hasattr(conv, "lin_dst") else nn.Identity()
        self.lin_edge = copy.deepcopy(conv.lin_edge) if hasattr(conv, "lin_edge") else None
        self.mlp = copy.deepcopy(conv.mlp)
        self.register_buffer("softmax_t", conv.aggr_module.t.detach().clone(), persistent=True)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        num_dst_nodes = x_dst.size(0)

        src_feat = self.lin_src(x_src)
        dst_feat = self.lin_dst(x_dst)

        src = edge_index[0].to(torch.int64)
        dst = edge_index[1].to(torch.int64)

        x_j = src_feat.index_select(0, src)

        if self.lin_edge is not None:
            msg = x_j + self.lin_edge(edge_attr)
        else:
            msg = x_j

        msg = msg.relu() + self.eps
        softmax_t = self._buffers["softmax_t"]
        assert softmax_t is not None
        alpha = edge_softmax_per_dst(msg, dst, num_dst_nodes, softmax_t)
        out = scatter_sum_per_dst(msg * alpha, dst, num_dst_nodes)
        out = out + dst_feat
        return self.mlp(out)


class ExportableGNNBlock(nn.Module):
    """Fixed-schema wrapper around rl.v15.gnn_model.GNNBlock."""

    def __init__(self, src_block):
        super().__init__()

        self.node_order = list(src_block.node_types.keys())
        self.edge_order = list(src_block.edge_types.keys())
        self.edge_attr_dims = [src_block.edge_types[edge_type]["size"] for edge_type in self.edge_order]
        self.dst_node_indices = [self.node_order.index(edge_type[2]) for edge_type in self.edge_order]
        self.src_node_indices = [self.node_order.index(edge_type[0]) for edge_type in self.edge_order]

        self.node_encoder = copy.deepcopy(src_block.node_encoder)
        self.norms = copy.deepcopy(src_block.norms)
        self.activations = copy.deepcopy(src_block.activations)

        layers = []
        for hetero_conv in src_block.convs:
            layers.append(nn.ModuleList([
                ExportableGENConv(hetero_conv.convs[edge_type])
                for edge_type in self.edge_order
            ]))
        self.convs = nn.ModuleList(layers)

    def _encode_nodes(self, x_by_type):
        return [
            self.node_encoder.modules_by_key[node_type](x)
            for node_type, x in zip(self.node_order, x_by_type)
        ]

    def _apply_norm(self, norm, node_type, x_old, x_new):
        # Matches HeteroResidualNorm.forward for fixed node order.
        if hasattr(norm, "norms"):
            if x_new is None:
                return x_old
            return norm.norms[node_type](x_old + x_new)

        # This mirrors the non-residual path in GNNBlock. In the normal v15
        # config every node type receives at least one relation per layer.
        return x_old if x_new is None else x_new

    def _apply_activation(self, activation, x):
        if hasattr(activation, "activation"):
            return activation.activation(x)
        return activation(x)

    def forward(self, x_by_type, edge_index_flat: torch.Tensor, edge_attr_flat: torch.Tensor, edge_lengths: torch.Tensor):
        x_by_type = self._encode_nodes(x_by_type)
        edge_lengths = edge_lengths.to(torch.int64)

        for convs, norm, activation in zip(self.convs, self.norms, self.activations):
            x_old = x_by_type
            x_new = [None for _ in self.node_order]
            cur_e = 0

            for edge_idx, conv in enumerate(convs):  # type: ignore[reportGeneralTypeIssues]
                edge_len = edge_lengths[edge_idx]
                e0 = cur_e
                e1 = cur_e + edge_len
                cur_e = e1

                edge_index = edge_index_flat[:, e0:e1]
                edge_attr_dim = self.edge_attr_dims[edge_idx]
                edge_attr = edge_attr_flat[e0:e1, :edge_attr_dim]

                src_i = self.src_node_indices[edge_idx]
                dst_i = self.dst_node_indices[edge_idx]
                out = conv(x_old[src_i], x_old[dst_i], edge_index, edge_attr)

                if x_new[dst_i] is None:
                    x_new[dst_i] = out
                else:
                    x_new[dst_i] = x_new[dst_i] + out

            x_by_type = [
                self._apply_activation(
                    activation,
                    self._apply_norm(norm, node_type, old, new),
                )
                for node_type, old, new in zip(self.node_order, x_old, x_new)
            ]

        return x_by_type


class ExportableGNNModel(nn.Module):
    """ONNX-friendly single-graph policy/value model for rl.v15 GNNModel."""

    def __init__(self, src_model: GNNModel, cfg):
        super().__init__()

        self.node_order = list(src_model.gnn.node_types.keys())
        self.edge_order = list(src_model.gnn.edge_types.keys())
        self.gnn = ExportableGNNBlock(src_model.gnn)
        self.policy_head = copy.deepcopy(src_model.policy_head)
        self.value_head = copy.deepcopy(src_model.value_head)

        side = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
        self.register_buffer("version", torch.tensor([15], dtype=torch.int32), persistent=False)
        self.register_buffer("side", torch.tensor([side], dtype=torch.int32), persistent=False)

    def get_version(self):
        version = self._buffers["version"]
        assert version is not None
        return version.clone()

    def get_side(self):
        side = self._buffers["side"]
        assert side is not None
        return side.clone()

    def forward(self, *args):
        num_node_inputs = len(self.node_order)
        x_by_type = list(args[:num_node_inputs])
        edge_index_flat = args[num_node_inputs]
        edge_attr_flat = args[num_node_inputs + 1]
        edge_lengths = args[num_node_inputs + 2]
        active_action_ids = args[num_node_inputs + 3].to(torch.int64)

        gnn_out = self.gnn(x_by_type, edge_index_flat, edge_attr_flat, edge_lengths)
        global_embeddings = gnn_out[self.node_order.index("Global")]
        action_embeddings = gnn_out[self.node_order.index("Action")]

        value = self.value_head(global_embeddings)[0]

        active_embeddings = action_embeddings.index_select(0, active_action_ids)
        active_logits = self.policy_head(active_embeddings).squeeze(-1)
        active_probs = active_logits.softmax(dim=0)
        greedy_pos = active_logits.argmax(dim=0)
        action = active_action_ids[greedy_pos]

        return action, active_probs, value, active_action_ids


def flatten_edges(hdata, edge_order, edge_attr_dims):
    max_attr_dim = max(edge_attr_dims) if edge_attr_dims else 0
    device = hdata["Global"].x.device

    edge_index_flat = torch.zeros((2, 0), dtype=torch.int64, device=device)
    edge_attr_flat = torch.zeros((0, max_attr_dim), dtype=torch.float32, device=device)
    lengths = []

    for edge_type, edge_attr_dim in zip(edge_order, edge_attr_dims):
        reldata = hdata[edge_type]
        edge_index = reldata.edge_index.to(torch.int64)
        edge_attr = getattr(reldata, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1], 0), dtype=torch.float32, device=device)

        assert edge_attr.shape[1] == edge_attr_dim, (edge_type, edge_attr.shape[1], edge_attr_dim)

        lengths.append(edge_index.shape[1])
        edge_index_flat = torch.cat([edge_index_flat, edge_index], dim=1)

        if edge_attr_dim < max_attr_dim:
            padding = torch.zeros(
                (edge_attr.shape[0], max_attr_dim - edge_attr_dim),
                dtype=edge_attr.dtype,
                device=edge_attr.device,
            )
            edge_attr = torch.cat([edge_attr, padding], dim=1)

        edge_attr_flat = torch.cat([edge_attr_flat, edge_attr], dim=0)

    return edge_index_flat, edge_attr_flat, torch.tensor(lengths, dtype=torch.int32, device=device)


def build_hdata(venv):
    b_done = torch.tensor(venv.call("terminated"), dtype=torch.bool)
    hdata = Batch.from_data_list(to_hdata_list(venv.call("graph_obs"), b_done))
    add_action_active_local_ids(hdata)
    return hdata


def build_inputs(hdata, model):
    node_inputs = {
        f"x_{node_type}": hdata[node_type].x
        for node_type in model.node_order
    }
    edge_index_flat, edge_attr_flat, edge_lengths = flatten_edges(
        hdata,
        model.edge_order,
        model.gnn.edge_attr_dims,
    )

    return {
        **node_inputs,
        "edge_index_flat": edge_index_flat,
        "edge_attr_flat": edge_attr_flat,
        "edge_lengths": edge_lengths,
        "active_action_ids": hdata["Action"].active_local_action_ids.to(torch.int64),
    }


def onnx_fwd(edge, inputs):
    return [
        torch.as_tensor(x)
        for x in edge.run(None, {k: v.detach().cpu().numpy() for k, v in inputs.items()})
    ]


def migrate_edge_key_typos(state_dict):
    replacements = {
        "<Global___Has___Action>": "<Global___To___Action>",
        "<Unit___By___Action>": "<Unit___Has___Action>",
    }

    migrated = {}

    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)

        if new_key in migrated:
            raise RuntimeError(f"Checkpoint migration collision: {key} -> {new_key}")

        migrated[new_key] = value

    return migrated


def load_gnn_model(cfg, weights_file):
    _ensure_supported_cfg(cfg)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")
    weights = migrate_edge_key_typos(weights)
    node_types = VcmiEnv.node_types()
    edge_types = _model_edge_types(cfg)


    if any(k.startswith("model.") for k in weights):
        model = PPOModel(node_types, edge_types, cfg["model"], torch.device("cpu")).eval()
        model.load_state_dict(weights, strict=True)
        return model.model.eval()

    if any(k.startswith("model_policy.") for k in weights):
        # Convenience for older v15 DNAModel checkpoints. The requested ppo_gnn
        # model uses the branch above.
        from .dna_gnn import DNAModel

        model = DNAModel(node_types, edge_types, cfg["model"], torch.device("cpu")).eval()
        model.load_state_dict(weights, strict=True)
        return model.model_policy.eval()

    model = GNNModel(node_types, edge_types, cfg["model"]).eval()
    model.load_state_dict(weights, strict=True)
    return model.eval()


def export_model(cfg, weights_file):
    venv = DualVecEnv(dict(cfg["train"]["env"]["kwargs"], mapname="gym/ml-mini.vmap", seed=0), envs_stupidai=dict(num=1, kwargs={}))
    venv.reset()

    src_model = load_gnn_model(cfg, weights_file)
    emodel = ExportableGNNModel(src_model, cfg).eval()
    inputs = build_inputs(build_hdata(venv), emodel)

    print("=== ONNX transform ===")

    input_names = list(inputs.keys())
    output_names = ["action", "active_probs", "value", "active_action_ids_out"]

    dynamic_axes = {
        **{f"x_{node_type}": {0: f"n_{node_type}"} for node_type in emodel.node_order},
        "edge_index_flat": {1: "num_edges"},
        "edge_attr_flat": {0: "num_edges"},
        "active_action_ids": {0: "num_active_actions"},
        "active_probs": {0: "num_active_actions"},
        "active_action_ids_out": {0: "num_active_actions"},
    }

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        torch.onnx.export(
            emodel,
            tuple(inputs.values()),
            tmp.name,
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

        loaded_model = onnx.load(tmp.name)

    metadata = {
        "version": str(emodel.get_version().item()),
        "side": str(emodel.get_side().item()),
        "node_order": json.dumps(emodel.node_order),
        "edge_order": json.dumps([list(edge_type) for edge_type in emodel.edge_order]),
        "output_contract": json.dumps(output_names),
    }

    for k, v in metadata.items():
        p = loaded_model.metadata_props.add()
        p.key = k
        p.value = v

    newbuffer = io.BytesIO()
    onnx.save(loaded_model, newbuffer)
    return newbuffer.getvalue()


def verify_export(cfg, weights_file, onnx_model, num_steps=10):
    src_model = load_gnn_model(cfg, weights_file)
    emodel = ExportableGNNModel(src_model, cfg).eval()

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]

    print("Testing metadata methods...")
    md = onnx_model.get_modelmeta().custom_metadata_map
    assert md["version"] == "15"
    assert md["side"] == str(eside)
    assert json.loads(md["node_order"]) == emodel.node_order
    assert [tuple(edge_type) for edge_type in json.loads(md["edge_order"])] == emodel.edge_order

    venv = DualVecEnv(dict(cfg["train"]["env"]["kwargs"], mapname="gym/ml-mini.vmap", seed=0), envs_stupidai=dict(num=1, kwargs={}))
    venv.reset()

    print("Testing data methods for %d steps..." % num_steps)
    ms_total = 0

    for _ in range(num_steps):
        print(venv.render()[0])  # type: ignore[index]

        hdata = build_hdata(venv)
        inputs = build_inputs(hdata, emodel)

        with torch.inference_mode():
            b_value = src_model.forward_value(hdata)
            b_action, _logprob, _entropy = src_model.forward_policy(hdata, deterministic=True)

            gnn_out = src_model.gnn(hdata)
            active_logits, _active_batch_index, active_local_action_ids, batch_size = src_model._get_active_logits(gnn_out, hdata)
            assert batch_size == 1
            assert torch.equal(active_local_action_ids, inputs["active_action_ids"])
            active_probs = active_logits.softmax(dim=0)

            eaction, eactive_probs, evalue, eactive_action_ids = emodel(*inputs.values())

        assert torch.equal(b_action[0].cpu(), eaction.cpu())
        assert torch.allclose(active_probs.cpu(), eactive_probs.cpu(), atol=1e-5, rtol=0)
        assert torch.allclose(b_value[0].cpu(), evalue.cpu(), atol=1e-5, rtol=0)
        assert torch.equal(inputs["active_action_ids"].cpu(), eactive_action_ids.cpu())

        t0 = perf_counter_ns()
        oaction, oactive_probs, ovalue, oactive_action_ids = onnx_fwd(onnx_model, inputs)
        ms = (perf_counter_ns() - t0) / 1e6
        print("Predict time: %.2f ms" % ms)
        ms_total += ms

        assert torch.equal(b_action[0].cpu(), oaction.cpu())
        assert torch.allclose(active_probs.cpu(), oactive_probs.cpu(), atol=1e-5, rtol=0)
        assert torch.allclose(b_value[0].cpu(), ovalue.cpu(), atol=1e-5, rtol=0)
        assert torch.equal(inputs["active_action_ids"].cpu(), oactive_action_ids.cpu())

        venv.step([int(b_action[0].item())])

    print("Total execution time: %dms (mean %.2f)" % (ms_total, ms_total / num_steps))
    print("Model role: %s" % cfg["train"]["env"]["kwargs"]["role"])
    print("verify_export: OK")


def load_exported_model(m):
    return ort.InferenceSession(m)


def save_exported_model(m, export_dir, symlink_dir, basename):
    os.makedirs(export_dir, exist_ok=True)
    dst = f"{export_dir}/{basename}.onnx"

    with open(dst, "wb") as f:
        f.write(m)

    print("Wrote %s" % dst)

    if symlink_dir:
        os.makedirs(symlink_dir, exist_ok=True)
        symlink = f"{symlink_dir}/{basename}.onnx"
        if os.path.islink(symlink) or os.path.exists(symlink):
            os.unlink(symlink)
        os.symlink(dst, symlink)
        print("Linked %s" % symlink)


def _load_cfg(path):
    with open(path, "r") as f:
        return json.load(f)


def _default_paths(prefix, src_dir):
    return (
        f"{src_dir}/{prefix}-config.json",
        f"{src_dir}/{prefix}-model-ppo.pt",
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Export an rl.v15.ppo_gnn GNNModel policy to ONNX")
    parser.add_argument("prefix", nargs="?", help="model prefix; reads ./export/<prefix>-config.json and ./export/<prefix>-model-ppo.pt")
    parser.add_argument("--config", help="path to config JSON")
    parser.add_argument("--weights", help="path to PPO weights file")
    parser.add_argument("--src-dir", default="./export", help="source directory used with prefix mode")
    parser.add_argument("--dst-dir", default="./export", help="directory where the ONNX file is written")
    parser.add_argument("--symlink-dir", default="/Users/simo/Library/Application Support/vcmi/Mods/mmai/models", help="optional symlink directory")
    parser.add_argument("--suffix", default="", help="optional suffix appended to the export basename")
    parser.add_argument("--verify-steps", type=int, default=10, help="number of env steps used for ONNX verification")
    parser.add_argument("--no-verify", action="store_true", help="skip ONNX verification")
    parser.add_argument("--output", help="explicit output .onnx path; disables basename/symlink handling")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.prefix:
        config_path, weights_path = _default_paths(args.prefix, args.src_dir)
    else:
        config_path, weights_path = args.config, args.weights

    if not config_path or not weights_path:
        raise SystemExit("Provide either <prefix> or both --config and --weights")

    cfg = _load_cfg(config_path)
    export_basename = "%s-%s" % (cfg["train"]["env"]["kwargs"]["role"], args.prefix or os.path.splitext(os.path.basename(weights_path))[0])
    if args.suffix:
        export_basename += f"-{args.suffix}"

    with torch.inference_mode():
        exported_model = export_model(cfg, weights_path)
        loaded_model = load_exported_model(exported_model)

        if not args.no_verify:
            verify_export(cfg, weights_path, loaded_model, num_steps=args.verify_steps)

        print("Model version: %s" % loaded_model.get_modelmeta().custom_metadata_map["version"])

        if args.output:
            with open(args.output, "wb") as f:
                f.write(exported_model)
            print("Wrote %s" % args.output)
        else:
            save_exported_model(exported_model, args.dst_dir, args.symlink_dir, export_basename)


if __name__ == "__main__":
    main()
