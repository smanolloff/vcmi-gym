import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.utils import softmax as gnn_softmax
from torch_geometric.data import HeteroData, Batch
from torch_scatter import scatter_sum, scatter_max

# obs: graph obs with {"nodes": ..., "edges": ...} (VcmiEnv v15 obs)
def to_hdata(obs, done, device=torch.device("cpu")):
    res = HeteroData()
    res.done = done.unsqueeze(0).float()
    res.value = torch.tensor(0., device=device)
    res.action = torch.tensor(0, device=device)
    res.reward = torch.tensor(0., device=device)
    res.logprob = torch.tensor(0., device=device)
    res.advantage = torch.tensor(0., device=device)
    res.ep_return = torch.tensor(0., device=device)
    res.active_action_ids = torch.as_tensor(obs["active_action_ids"])
    res.num_active_actions = obs["active_action_ids"].size

    for node_name, attrs in obs["nodes"].items():
        res[node_name].x = torch.as_tensor(attrs, device=device)

    for edge_key, edge in obs["edges"].items():
        res[edge_key].edge_index = torch.as_tensor(edge["index"], device=device)
        res[edge_key].edge_attr = torch.as_tensor(edge["attrs"], device=device)

    return res


def to_hdata_list(b_obs, b_done):
    hdatas = []
    for obs, done in zip(b_obs, b_done):
        hdatas.append(to_hdata(obs, done))
    # XXX: this concatenates along the first dim
    # i.e. stacking two (165, STATE_SIZE_ONE_HEX)
    #       gives  (330, STATE_SIZE_ONE_HEX)
    #       (that's how GNN batching works)
    return hdatas

#
# Add batch-friendly active action info
#
def add_action_active_local_ids(hdata_batch):
    device = hdata_batch["Action"].batch.device

    active_local_action_ids = hdata_batch.active_action_ids.to(device=device)
    num_active_actions = hdata_batch.num_active_actions.to(device=device)
    batch_size = num_active_actions.numel()
    num_actions_per_graph = torch.bincount(hdata_batch["Action"].batch, minlength=batch_size)

    action_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=device),
        num_actions_per_graph.cumsum(0)[:-1],
    ])

    active_batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_active_actions,
    )

    active_global_ids = active_local_action_ids + action_offsets[active_batch_index]

    hdata_batch["Action"].active_global_ids = active_global_ids
    hdata_batch["Action"].active_local_action_ids = active_local_action_ids
    hdata_batch["Action"].active_batch_index = active_batch_index


class DictModule(nn.Module):
    """
    Applies one module per key in a dictionary.
    Example:
        x_dict["Unit"] = module["Unit"](x_dict["Unit"])
    """

    def __init__(self, modules):
        super().__init__()
        self.modules_by_key = nn.ModuleDict(modules)

    def forward(self, x_dict):
        return {
            key: self.modules_by_key[key](x)
            for key, x in x_dict.items()
        }


class HeteroActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x_dict):
        return {
            key: self.activation(x)
            for key, x in x_dict.items()
        }


class HeteroResidualNorm(nn.Module):
    """
    Applies residual + LayerNorm per node type.

    Assumes input and output channels are the same.
    """

    def __init__(self, node_types, channels):
        super().__init__()
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(channels)
            for node_type in node_types
        })

    def forward(self, x_old, x_new):
        out = {}

        for node_type, x in x_old.items():
            if node_type in x_new:
                out[node_type] = self.norms[node_type](x + x_new[node_type])
            else:
                # If a node type receives no messages in this layer,
                # keep it unchanged.
                out[node_type] = x

        return out


class GNNBlock(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        num_gnn_layers: int,
        out_channels: int,
        hidden_channels: int | None = None,
        conv_cls=gnn.GENConv,
        aggr: str = "sum",
        activation=None,
        use_residual: bool = True,
        conv_kwargs: dict | None = None,
    ):
        """
        Args:
            node_types:
                Dict like:
                {
                    "Global": {"size": 10, "attributes": [...]},
                    "Unit": {"size": 37, "attributes": [...]},
                    ...
                }

            edge_types:
                Dict like:
                {
                    ("Global", "Has", "Player"): {"size": 0, "attributes": []},
                    ("Hex", "Adjacent", "Hex"): {"size": 6, "attributes": [...]},
                    ...
                }

            num_gnn_layers:
                Number of message passing layers.

            out_channels:
                Output embedding size for every node type.

            hidden_channels:
                Internal embedding size. Defaults to out_channels.

            conv_cls:
                Message-passing layer class. Default: GENConv.

            aggr:
                HeteroConv aggregation over multiple relations targeting
                the same node type.

            activation:
                Activation module. Defaults to LeakyReLU.

            use_residual:
                Whether to use residual connections and LayerNorm.

            conv_kwargs:
                Extra kwargs passed to conv_cls.
        """

        super().__init__()

        assert num_gnn_layers >= 1

        if hidden_channels is None:
            hidden_channels = out_channels

        if activation is None:
            activation = nn.LeakyReLU()

        if conv_kwargs is None:
            conv_kwargs = {}

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_gnn_layers = num_gnn_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_residual = use_residual

        # 1. Encode all node types into common hidden dimensionality.
        self.node_encoder = DictModule({
            node_type: nn.Linear(info["size"], hidden_channels)
            for node_type, info in node_types.items()
        })

        # 2. Build hetero GNN layers.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for layer_idx in range(num_gnn_layers):
            is_last = layer_idx == num_gnn_layers - 1
            layer_out_channels = out_channels if is_last else hidden_channels

            conv_dict = {}

            for edge_type, edge_info in edge_types.items():
                edge_dim = edge_info["size"]

                kwargs = dict(conv_kwargs)

                if edge_dim > 0:
                    kwargs["edge_dim"] = edge_dim

                # For heterogeneous/bipartite relations, use tuple input sizes.
                # Since all node embeddings are already projected to hidden_channels,
                # both source and destination use hidden_channels.
                conv_dict[edge_type] = conv_cls(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=layer_out_channels,
                    **kwargs,
                )

            self.convs.append(
                gnn.HeteroConv(conv_dict, aggr=aggr)
            )

            if not is_last:
                self.activations.append(HeteroActivation(activation))
            else:
                self.activations.append(nn.Identity())

            if use_residual and hidden_channels == layer_out_channels:
                self.norms.append(
                    HeteroResidualNorm(self.node_types, hidden_channels)
                )
            else:
                self.norms.append(nn.Identity())

            hidden_channels = layer_out_channels

    def _clean_edge_attr_dict(self, edge_attr_dict):
        """
        PyG usually expects edge_attr only for relations whose conv has edge_dim.

        If an edge type has no attributes, it is better to pass None / omit it.
        This method removes zero-width edge attributes.
        """

        if edge_attr_dict is None:
            return None

        cleaned = {}

        for edge_type, edge_attr in edge_attr_dict.items():
            if edge_attr is None:
                continue

            if edge_attr.numel() == 0:
                continue

            if edge_attr.dim() == 2 and edge_attr.size(-1) == 0:
                continue

            cleaned[edge_type] = edge_attr

        return cleaned

    def forward(self, hdata):
        """
        Args:
            hdata:
                PyG HeteroData with:
                    hdata.x_dict
                    hdata.edge_index_dict
                    hdata.edge_attr_dict

        Returns:
            x_dict:
                Updated node embeddings per node type.
        """

        x_dict = self.node_encoder(hdata.x_dict)

        edge_index_dict = hdata.edge_index_dict
        edge_attr_dict = self._clean_edge_attr_dict(
            getattr(hdata, "edge_attr_dict", None)
        )

        for conv, norm, activation in zip(
            self.convs,
            self.norms,
            self.activations,
        ):
            x_old = x_dict

            if edge_attr_dict is not None:
                x_new = conv(
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict=edge_attr_dict,
                )
            else:
                x_new = conv(
                    x_dict,
                    edge_index_dict,
                )

            if isinstance(norm, HeteroResidualNorm):
                x_dict = norm(x_old, x_new)
            else:
                x_dict = x_new

            x_dict = activation(x_dict)

        return x_dict

class GNNModel(nn.Module):
    def __init__(self, node_types, edge_types, config):
        super().__init__()

        self.gnn = GNNBlock(
            node_types=node_types,
            edge_types=edge_types,
            num_gnn_layers=config["gnn_num_layers"],
            hidden_channels=config["gnn_hidden_channels"],
            out_channels=config["gnn_out_channels"],
            conv_cls=gnn.GENConv,
            conv_kwargs=config["gnn_conv_kwargs"],
        )

        self.policy_head = nn.Sequential(
            nn.LayerNorm(config["gnn_out_channels"]),
            nn.Linear(config["gnn_out_channels"], config["policy_head_hidden_channels"]),
            nn.LeakyReLU(),
            nn.Linear(config["policy_head_hidden_channels"], config["policy_head_hidden_channels"]),
            nn.LeakyReLU(),
            nn.Linear(config["policy_head_hidden_channels"], 1),
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(config["gnn_out_channels"]),
            nn.Linear(config["gnn_out_channels"], config["value_head_hidden_channels"]),
            nn.LeakyReLU(),
            nn.Linear(config["value_head_hidden_channels"], config["value_head_hidden_channels"]),
            nn.LeakyReLU(),
            nn.Linear(config["value_head_hidden_channels"], 1),
        )

    @staticmethod
    def process_flat_logits(
        active_logits,
        active_batch_index,
        active_local_action_ids,
        batch_size,
        b_action,
        deterministic
    ):
        if b_action is not None:
            return GNNModel._process_flat_logits_train(
                active_logits=active_logits,
                active_batch_index=active_batch_index,
                active_local_action_ids=active_local_action_ids,
                batch_size=batch_size,
                b_action=b_action,
            )

        if deterministic:
            return GNNModel._process_flat_logits_deterministic(
                active_logits=active_logits,
                active_batch_index=active_batch_index,
                active_local_action_ids=active_local_action_ids,
                batch_size=batch_size,
            )

        return GNNModel._process_flat_logits_sample(
            active_logits=active_logits,
            active_batch_index=active_batch_index,
            active_local_action_ids=active_local_action_ids,
            batch_size=batch_size,
        )

    @staticmethod
    def _process_flat_logits_deterministic(
        active_logits,
        active_batch_index,
        active_local_action_ids,
        batch_size,
    ):
        probs = gnn_softmax(active_logits, active_batch_index)
        log_probs_all = torch.log(probs.clamp_min(1e-12))

        entropy = scatter_sum(
            -probs * log_probs_all,
            active_batch_index,
            dim=0,
            dim_size=batch_size,
        )

        _, argmax_flat_pos = scatter_max(
            active_logits,
            active_batch_index,
            dim=0,
            dim_size=batch_size,
        )

        b_action = active_local_action_ids[argmax_flat_pos]
        b_logprob = log_probs_all[argmax_flat_pos]

        return b_action, b_logprob, entropy

    @staticmethod
    def _process_flat_logits_sample(
        active_logits,
        active_batch_index,
        active_local_action_ids,
        batch_size,
    ):
        device = active_logits.device

        counts = torch.bincount(active_batch_index, minlength=batch_size)
        max_actions = int(counts.max().item())

        padded_logits = torch.full(
            (batch_size, max_actions),
            -torch.inf,
            device=device,
            dtype=active_logits.dtype,
        )

        padded_action_ids = torch.full(
            (batch_size, max_actions),
            -1,
            device=device,
            dtype=active_local_action_ids.dtype,
        )

        local_pos = torch.arange(active_logits.numel(), device=device) - torch.repeat_interleave(
            torch.cumsum(counts, dim=0) - counts,
            counts,
        )

        padded_logits[active_batch_index, local_pos] = active_logits
        padded_action_ids[active_batch_index, local_pos] = active_local_action_ids

        dist = Categorical(logits=padded_logits)

        sampled_pos = dist.sample()

        b_action = padded_action_ids[
            torch.arange(batch_size, device=device),
            sampled_pos,
        ]

        b_logprob = dist.log_prob(sampled_pos)
        entropy = dist.entropy()

        return b_action, b_logprob, entropy

    @staticmethod
    def _process_flat_logits_train(
        active_logits,
        active_batch_index,
        active_local_action_ids,
        batch_size,
        b_action,
    ):
        probs = gnn_softmax(active_logits, active_batch_index)
        log_probs_all = torch.log(probs.clamp_min(1e-12))

        b_entropy = scatter_sum(
            -probs * log_probs_all,
            active_batch_index,
            dim=0,
            dim_size=batch_size,
        )

        chosen_mask = active_local_action_ids == b_action[active_batch_index]

        if not torch.all(torch.bincount(active_batch_index[chosen_mask], minlength=batch_size) == 1):
            raise RuntimeError("Each batch item must have exactly one matching action.")

        chosen_batch = active_batch_index[chosen_mask]
        b_logprob = torch.empty(batch_size, device=active_logits.device, dtype=log_probs_all.dtype)
        b_logprob[chosen_batch] = log_probs_all[chosen_mask]

        return b_action, b_logprob, b_entropy

    def forward(self, hdata, b_action=None, deterministic=False):
        gnn_out = self.gnn(hdata)
        b_value = self._forward_value(gnn_out)
        b_action, b_logprob, b_entropy = self._forward_policy(gnn_out, hdata, b_action, deterministic)
        return b_value, b_action, b_logprob, b_entropy

    def forward_value(self, hdata, b_action=None, deterministic=False):
        gnn_out = self.gnn(hdata)
        b_value = self._forward_value(gnn_out)
        return b_value

    def forward_policy(self, hdata, b_action=None, deterministic=False):
        gnn_out = self.gnn(hdata)
        b_action, b_logprob, b_entropy = self._forward_policy(gnn_out, hdata, b_action, deterministic)
        return b_action, b_logprob, b_entropy

    def _forward_value(self, gnn_out):
        return self.value_head(gnn_out["Global"])

    def _forward_policy(self, gnn_out, hdata, b_action=None, deterministic=False):
        (
            active_logits,
            active_batch_index,
            active_local_action_ids,
            batch_size,
        ) = self._get_active_logits(gnn_out, hdata)

        b_action, b_logprob, b_entropy = GNNModel.process_flat_logits(
                active_logits=active_logits,
                active_batch_index=active_batch_index,
                active_local_action_ids=active_local_action_ids,
                batch_size=batch_size,
                b_action=b_action,
                deterministic=deterministic,
        )

        return b_action, b_logprob, b_entropy

    def _get_active_logits(self, gnn_out, hdata):
        action_embeddings = gnn_out["Action"]

        active_global_ids = hdata["Action"].active_global_ids
        active_batch_index = hdata["Action"].active_batch_index
        active_local_action_ids = hdata["Action"].active_local_action_ids
        batch_size = hdata.num_graphs

        active_embeddings = action_embeddings[active_global_ids]
        active_logits = self.policy_head(active_embeddings).squeeze(-1)

        return active_logits, active_batch_index, active_local_action_ids, batch_size


if __name__ == "__main__":
    from rl.v15.dual_vec_env import DualVecEnv

    # venv = DualVecEnv(num_envs_stupidai=2, env_kwargs=dict(mapname="gym/generated/4096/4x1024.vmap", random_heroes=1))
    venv = DualVecEnv(num_envs_stupidai=2, env_kwargs=dict(mapname="gym/A1.vmap"))

    model_config = dict(
        gnn_num_layers=2,
        gnn_hidden_channels=16,
        gnn_out_channels=8,
        gnn_conv_kwargs={
            "aggr": "softmax",
            "learn_t": True,
            "num_layers": 2,
            "norm": "layer",
            "add_self_loops": False,  # avoid automatic self-loops in rich graphs
        },
        policy_head_hidden_channels=16,
        policy_head_dropout=0,
        value_head_hidden_channels=16,
        value_head_dropout=0,
    )

    model = GNNModel(model_config)

    # DualVecEnv returns dummy obs due to issues with dynamic obs in VectorEnv
    _dummy, b_rew, b_term, b_trunc, b_info = venv.step(venv.call("random_action"))
    b_obs = venv.call("graph_obs")
    b_done = torch.as_tensor(np.logical_or(b_term, b_trunc))
    hdata = Batch.from_data_list(to_hdata_list(b_obs, b_done))

    import ipdb; ipdb.set_trace()  # noqa

    # Deterministic inference
    b_action1, b_logprob1, b_entropy1 = model.forward_policy(hdata, deterministic=True)
    # Stochastic inference
    b_action2, b_logprob2, b_entropy2 = model.forward_policy(hdata, deterministic=False)
    # PPO/train mode: action already known
    b_action3, b_logprob3, b_entropy3 = model.forward_policy(hdata, b_action=b_action2)

    b_value = model.forward_value(hdata)

    venv.step(b_action1)
    import ipdb; ipdb.set_trace()  # noqa
    print(b_value)
    print("Done.")
