import torch
import torch.nn as nn


def broadcast(src, ref, dim):
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def scatter_sum(src, index, dim, dim_size):
    dim = src.dim() + dim
    size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]
    index = broadcast(index, src, dim)
    return src.new_zeros(size).scatter_add_(dim, index, src)


def scatter_max(src, index, dim, dim_size):
    dim = src.dim() + dim
    size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]
    index = broadcast(index, src, dim)
    return src.new_zeros(size).scatter_reduce_(dim, index, src, reduce='amax', include_self=False)


def softmax(src, index, num_nodes, dim):
    N = num_nodes
    src_max = scatter_max(src.detach(), index, dim, dim_size=N)
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = scatter_sum(out, index, dim, dim_size=N) + 1e-16
    out_sum = out_sum.index_select(dim, index)
    return out / out_sum


class MyGENConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()

        self.node_dim = -2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = 1e-7

        if in_channels != out_channels:
            self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_src = nn.Identity()
            self.lin_dst = nn.Identity()

        self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels*2, bias=False),
            nn.BatchNorm1d(out_channels*2, affine=True),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(out_channels*2, out_channels, bias=False),
        )

    def forward(self, x, edge_index, edge_attr):
        x_src = self.lin_src(x)
        x_dst = self.lin_dst(x)
        x_j = x_src.index_select(self.node_dim, edge_index[0])

        out = self.message(x_j, edge_attr)
        out = self.aggregate(out, edge_index[1], x.size(self.node_dim))
        out = out + x_dst

        return self.mlp(out)

    def message(self, x_j, edge_attr):
        edge_attr = self.lin_edge(edge_attr)
        msg = x_j + edge_attr
        return msg.relu() + self.eps

    def aggregate(self, x, index, dim_size):
        alpha = softmax(x, index, dim_size, self.node_dim)
        res = scatter_sum(x * alpha, index, self.node_dim, dim_size)
        return res


class MyGNNBlock(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, edge_dim, link_types):
        super().__init__()

        layers = []

        # First L-1 layers with activation
        for i in range(num_layers - 1):
            ch_in = in_channels if i == 0 else hidden_channels
            layers.append([MyGENConv(ch_in, hidden_channels, edge_dim) for _ in link_types])

        # Last layer without extra activation beyond the internal MLP
        layers.append([MyGENConv(hidden_channels, out_channels, edge_dim) for _ in link_types])

        self.layers = nn.ModuleList([nn.ModuleList(convs) for convs in layers])
        self.act = nn.LeakyReLU()

    def forward(self, x_hex, *edge_args):
        # edge_args = (ei1, ea1, ei2, ea2, ..., eiR, eaR)
        assert len(edge_args) == 2 * len(self.layers[0]), "Expected 2*num_rels tensors"
        eis = edge_args[0::2]
        eas = edge_args[1::2]

        x = x_hex
        L = len(self.layers)
        for i, convs in enumerate(self.layers):
            y = None
            for r, conv in enumerate(convs):
                yr = conv(x, eis[r], eas[r])
                y = yr if y is None else y + yr
            x = y
            if i < L - 1:
                x = self.act(x)
        return x


if __name__ == "__main__":
    # import torch_geometric
    # gen = torch_geometric.nn.GENConv(5, 12, edge_dim=1).eval()
    # mygen = MyGENConv(5, 12, 1).eval()
    # mygen.load_state_dict(gen.state_dict(), strict=True)

    # hd = torch_geometric.data.HeteroData()
    # hd['hex'].x = torch.randn(3, 5)
    # edge_index_lt1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # edge_attr_lt1 = torch.tensor([[1.0], [1.0]], dtype=torch.float)
    # edge_index_lt2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    # edge_attr_lt2 = torch.tensor([[0.5], [0.2]], dtype=torch.float)
    # hd['hex', 'lt1', 'hex'].edge_index = edge_index_lt1
    # hd['hex', 'lt1', 'hex'].edge_attr = edge_attr_lt1
    # hd['hex', 'lt2', 'hex'].edge_index = edge_index_lt2
    # hd['hex', 'lt2', 'hex'].edge_attr = edge_attr_lt2

    # inputs = (hd["hex"].x, hd["hex", "lt1", "hex"].edge_index, hd["hex", "lt1", "hex"].edge_attr)
    # res = gen(*inputs)
    # myres = mygen(*inputs)

    # assert torch.equal(res, myres)

    # from rl.algos.mppo_dna_gnn.mppo_dna_gnn import GNNBlock

    # link_types = ["lt1", "lt2"]

    # block = GNNBlock(num_layers=2, in_channels=5, hidden_channels=4, out_channels=3, num_heads=1, link_types=link_types).eval()
    # myblock = MyGNNBlock(num_layers=2, in_channels=5, hidden_channels=4, out_channels=3, edge_dim=1, link_types=link_types).eval()

    def gnn_to_nn_key(gnn_key, link_types):
        import re
        pattern = re.compile(r"layers.module_(\d+)\.convs\.<hex___(\w+)___hex>\.(.+)")
        match = pattern.match(gnn_key)
        assert match, f"No match: {gnn_key}"
        module_id, link_type, rest = match.groups()
        assert int(module_id) % 2 == 0
        assert link_type in link_types, f"{link_type} not in {link_types}"
        return "layers.%d.%d.%s" % (int(module_id) / 2, link_types.index(link_type), rest)

    # my_state_dict = {gnn_to_nn_key(k, link_types): v for k, v in block.state_dict().items()}
    # myblock.load_state_dict(my_state_dict, strict=True)

    # myinputs = (
    #     hd["hex"].x,
    #     hd["hex", "lt1", "hex"].edge_index,
    #     hd["hex", "lt1", "hex"].edge_attr,
    #     hd["hex", "lt2", "hex"].edge_index,
    #     hd["hex", "lt2", "hex"].edge_attr,
    # )

    # res = block(hd)
    # myres = myblock(*myinputs)

    # assert torch.equal(res["hex"], myres)

    import json
    with open("sfcjqcly-1757584171-config.json", "r") as f:
        cfg = json.load(f)

    from torch_geometric.data import Batch
    from rl.algos.mppo_dna_gnn.mppo_dna_gnn import DNAModel, to_hdata_list
    from vcmi_gym.envs.v13.vcmi_env import VcmiEnv
    from vcmi_gym.envs.v13.pyconnector import STATE_SIZE, STATE_SIZE_ONE_HEX, LINK_TYPES

    model = DNAModel(cfg["model"], torch.device("cpu"))
    model.load_state_dict(torch.load("sfcjqcly-1757584171-model-dna.pt", weights_only=True, map_location="cpu"), strict=True)
    gnn = model.model_policy.encoder_hexes.eval()

    mygnn = MyGNNBlock(
        num_layers=cfg["model"]["gnn_num_layers"],
        in_channels=STATE_SIZE_ONE_HEX,
        hidden_channels=cfg["model"]["gnn_hidden_channels"],
        out_channels=cfg["model"]["gnn_out_channels"],
        edge_dim=1,
        link_types=list(LINK_TYPES)
    ).eval()
    my_state_dict = {gnn_to_nn_key(k, list(LINK_TYPES)): v for k, v in gnn.state_dict().items()}
    mygnn.load_state_dict(my_state_dict, strict=True)

    obs = torch.randn([1, STATE_SIZE])
    done = torch.zeros(1)
    links = [VcmiEnv.OBSERVATION_SPACE["links"].sample()]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, links))
    res = gnn(hdata)

    myinputs = [hdata["hex"].x]
    for lt in LINK_TYPES:
        myinputs.append(hdata["hex", lt, "hex"].edge_index)
        myinputs.append(hdata["hex", lt, "hex"].edge_attr)

    myinputs = tuple(myinputs)
    myres = mygnn(*myinputs)

    assert torch.equal(res["hex"], myres)

    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    ep = export(mygnn, args=myinputs, dynamic_shapes=None, strict=True)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    exec_prog = edge.to_executorch()

    import ipdb; ipdb.set_trace()  # noqa
    pass

    # with open(MODEL_EXPORT_PATH, "wb") as f:
    #     exec_prog.write_to_file(f)
