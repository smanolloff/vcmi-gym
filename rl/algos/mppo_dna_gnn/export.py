import os
import json
import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch_geometric.data import Batch
from torch.export import export, export_for_training
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.runtime import Runtime

from .mppo_dna_gnn import (
    STATE_SIZE,
    STATE_SIZE_ONE_HEX,
    STATE_SIZE_HEXES,
    GLOBAL_ATTR_MAP,
    GLOBAL_ACT_MAP,
    HEX_ATTR_MAP,
    HEX_ACT_MAP,
    LINK_TYPES,
    DNAModel,
    Model,
    MainAction,
    to_hdata_list,
    create_venv
)


def transform_key(key, link_types):
    import re
    pattern = re.compile(r"(.+)\.module_(\d+)\.convs\.<hex___(\w+)___hex>\.(.+)")
    match = pattern.match(key)
    if not match:
        return key
    assert match, f"No match: {key}"
    pre, module_id, link_type, rest = match.groups()
    assert int(module_id) % 2 == 0
    assert link_type in link_types, f"{link_type} not in {link_types}"
    return "%s.%d.%d.%s" % (pre, int(module_id) / 2, link_types.index(link_type), rest)


def build_einputs(hdata, e_max, k_max):
    eis = []
    eas = []
    nbrs = []

    for lt in LINK_TYPES:
        reldata = hdata["hex", lt, "hex"]
        ei, ea = pad_edges(reldata.edge_index, reldata.edge_attr, e_max)
        nbr = build_nbr(reldata.edge_index[1], 165, k_max)

        eis.append(ei)
        eas.append(ea)
        nbrs.append(nbr)

    return (
        hdata.obs[0],
        torch.stack(eis, dim=0),
        torch.stack(eas, dim=0),
        torch.stack(nbrs, dim=0),
    )


# Usage: build_nbr(edge_index[1], x.size(0), K_MAX)
def build_nbr(dst, num_nodes, k_max):
    """
    Build nbr[v, k] = edge id of k-th incoming edge to node v; -1 if unused.
    Use this OUTSIDE the exported graph (host-side) for the current edge_index.
    """
    nbr = torch.full((num_nodes, k_max), -1, dtype=torch.long)
    fill = torch.zeros(num_nodes, dtype=torch.long)
    for e, v in enumerate(dst.tolist()):
        p = int(fill[v])
        if p >= k_max:
            raise ValueError(f"node {v} exceeds k_max={k_max}")
        nbr[v, p] = e
        fill[v] = p + 1
    return nbr


def pad_edges(edge_index, edge_attr, e_max):
    # edge_index: (2, E), long; edge_attr: (E, 1), float
    E = edge_index.size(1)
    if E > e_max:
        raise ValueError(f"E={E} exceeds e_max={e_max}")
    pad = e_max - E
    if pad:
        edge_index = torch.cat([edge_index, edge_index.new_zeros(2, pad)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr.new_zeros(pad, 1)], dim=0)

    return edge_index, edge_attr


class ExecuTorchModel(nn.Module):
    def __init__(self, config, e_max, k_max):
        super().__init__()
        self.dim_other = STATE_SIZE - STATE_SIZE_HEXES
        self.dim_hexes = STATE_SIZE_HEXES
        self.state_size_hexes = STATE_SIZE_HEXES
        self.state_size_one_hex = STATE_SIZE_ONE_HEX
        self.hex_move_offset = HEX_ATTR_MAP["ACTION_MASK"][1] + HEX_ACT_MAP["MOVE"]
        self.hex_shoot_offset = HEX_ATTR_MAP["ACTION_MASK"][1] + HEX_ACT_MAP["SHOOT"]
        self.global_wait_offset = GLOBAL_ATTR_MAP["ACTION_MASK"][1] + GLOBAL_ACT_MAP["WAIT"]
        self.n_main_actions = len(MainAction)

        action_table, inverse_table, amove_hexes = Model.build_action_tables()

        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0))
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1)
        self.register_buffer("action_table", action_table)
        self.register_buffer("inverse_table", inverse_table)

        self.encoder_hexes = ExportableGNNBlock(
            num_layers=config["gnn_num_layers"],
            in_channels=STATE_SIZE_ONE_HEX,
            hidden_channels=config["gnn_hidden_channels"],
            out_channels=config["gnn_out_channels"],
            edge_dim=1,
            link_types=list(LINK_TYPES),
            num_nodes=165,
            e_max=e_max,
            k_max=k_max
        ).eval()

        d = config["gnn_out_channels"]

        self.encoder_other = nn.Sequential(
            nn.Linear(self.dim_other, d),
            nn.LeakyReLU()
        )

        self.act0_head = nn.Linear(d, len(MainAction))
        self.emb_act0 = nn.Embedding(len(MainAction), d)
        self.Wk_hex1 = nn.Linear(d, d, bias=False)
        self.Wk_hex2 = nn.Linear(d, d, bias=False)
        self.Wq_hex1 = nn.Linear(2*d, d)
        self.Wq_hex2 = nn.Linear(2*d, d)

        self.critic = nn.Sequential(
            # nn.LayerNorm(d), helps?
            nn.Linear(d, config["critic_hidden_features"]),
            nn.LeakyReLU(),
            nn.Linear(config["critic_hidden_features"], 1)
        )

        self.register_buffer("mask_hex1", torch.zeros([1, self.n_main_actions, 165], dtype=torch.bool), persistent=False)
        self.register_buffer("mask_hex2", torch.zeros([1, self.n_main_actions, 165, 165], dtype=torch.bool), persistent=False)
        self.register_buffer("mask_action", torch.zeros([1, self.n_main_actions], dtype=torch.bool), persistent=False)
        self.register_buffer("b_idx", torch.arange(1).view(1, 1, 1).expand_as(self.amove_hexes), persistent=False)
        self.register_buffer("s_idx", torch.arange(165).view(1, 165, 1).expand_as(self.amove_hexes), persistent=False)
        self.register_buffer("mask_value", torch.tensor(torch.finfo(torch.float32).min), persistent=False)
        self.register_buffer("hexactmask_inds", torch.as_tensor(torch.arange(12) + HEX_ATTR_MAP["ACTION_MASK"][1]), persistent=False)

    # edge_triplets is [edge_ind1, edge_attr1, edge_nbr1, edge_ind2, edge_attr2, edge_nbr2, ...]
    # (one triplet for each link type)
    def encode(self, obs, edge_triplets):
        hexes = obs[0, self.dim_other:].view(165, self.state_size_one_hex)
        other = obs[0, :self.dim_other]
        z_hexes = self.encoder_hexes(hexes, *edge_triplets).unsqueeze(0)
        z_other = self.encoder_other(other).unsqueeze(0)
        z_global = z_other + z_hexes.mean(1)
        return z_hexes, z_global

    def get_value(self, obs, *edge_triplets):
        obs = obs.unsqueeze(dim=0)
        _, z_global = self.encode(obs, edge_triplets)
        return self.critic(z_global), z_global

    def predict(self, obs, *edge_triplets):
        return self._predict_with_logits(obs, *edge_triplets)[0]

    def _predict_with_logits(self, obs, *edge_triplets):
        obs = obs.unsqueeze(dim=0)
        z_hexes, z_global = self.encode(obs, edge_triplets)

        act0_logits = self.act0_head(z_global)

        # 1. MASK_HEX1 - ie. allowed hex#1 for each action
        mask_hex1 = torch.zeros_like(self.mask_hex1)
        hexobs = obs[:, -self.state_size_hexes:].view([-1, 165, self.state_size_one_hex])

        # XXX: EXPLICIT casting to torch.bool is required to prevent a
        #      a nasty bug with the mask when the model is loaded in C++

        # 1.1 for 0=WAIT: nothing to do (all zeros)
        # 1.2 for 1=MOVE: Take MOVE bit from obs's action mask
        movemask = hexobs[:, :, self.hex_move_offset].to(torch.bool)
        mask_hex1[:, 1, :] = movemask

        # 1.3 for 2=AMOVE: Take any(AMOVEX) bits from obs's action mask
        amovemask = hexobs[:, :, self.hexactmask_inds].to(torch.bool)
        mask_hex1[:, 2, :] = amovemask.any(dim=-1)

        # 1.4 for 3=SHOOT: Take SHOOT bit from obs's action mask
        shootmask = hexobs[:, :, self.hex_shoot_offset].to(torch.bool)
        mask_hex1[:, 3, :] = shootmask

        # 2. MASK_HEX2 - ie. allowed hex2 for each (action, hex1) combo
        mask_hex2 = torch.zeros_like(self.mask_hex2)

        # 2.1 for 0=WAIT: nothing to do (all zeros)
        # 2.2 for 1=MOVE: nothing to do (all zeros)
        # 2.3 for 2=AMOVE: For each SRC hex, create a DST hex mask of allowed hexes
        valid = amovemask & self.amove_hexes_valid

        # # Select only valid triples and write
        # b_sel = self.b_idx[valid]
        # s_sel = self.s_idx[valid]
        # t_sel = self.amove_hexes[valid]
        # mask_hex2[b_sel, 2, s_sel, t_sel] = True

        # "How do I build mask_hex2[b, 2, s, t] = True when self.amove_hexes
        # contains -1 entries, without boolean indexing and with static shapes?"
        # Shapes:
        # self.mask_hex2: [B, 4, S, T] (bool)  e.g., [B,4,165,165]
        # self.amove_hexes: [B, S, K] (long)    target t-indices, may contain -1
        # valid: [B, S, K] (bool)               which triplets are active

        # 1) Plane we will write: channel 2 of the mask
        plane = torch.zeros_like(self.mask_hex2.select(1, 2))  # [B, S, T], bool

        # 2) Ensure invalid entries are excluded from updates
        idx = self.amove_hexes                                      # [B, S, K], long
        valid = valid & (idx >= 0)                                  # drop t == -1

        # 3) Replace -1 by 0 (any in-range index works) but make src zero there,
        #    so those updates are no-ops on a zero-initialized buffer.
        safe_idx = torch.where(valid, idx, idx.new_zeros(idx.shape))            # [B,S,K]
        src_i32 = valid.to(torch.int32)                                         # [B,S,K]

        # 4) Accumulate along T, then binarize
        accum = torch.zeros((idx.size(0), idx.size(1), plane.size(-1)), dtype=torch.int32,
                            device=idx.device)                                   # [B,S,T]
        accum = accum.scatter_add(-1, safe_idx, src_i32)                         # [B,S,T]
        plane = accum.ne(0)                                                      # bool

        # 5) Write back into the full mask tensor
        mask_hex2 = torch.zeros_like(self.mask_hex2)                             # [B,4,S,T]
        mask_hex2[:, 2] = plane

        # 2.4 for 3=SHOOT: nothing to do (all zeros)

        # 3. MASK_ACTION - ie. allowed main action mask
        mask_action = torch.zeros_like(self.mask_action)

        # 0=WAIT
        mask_action[:, 0] = obs[:, self.global_wait_offset].to(torch.bool)

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_action[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1)

        # Next, we sample:
        #
        # 1. Sample MAIN ACTION
        probs_act0 = self._categorical_masked(logits0=act0_logits, mask=mask_action)
        act0 = torch.argmax(probs_act0, dim=1)

        # 2. Sample HEX1 (with mask corresponding to the main action)
        act0_emb = self.emb_act0(act0)
        d = act0_emb.size(-1)
        q_hex1 = self.Wq_hex1(torch.cat([z_global, act0_emb], -1))              # (B, d)
        k_hex1 = self.Wk_hex1(z_hexes)                                          # (B, 165, d)
        hex1_logits = (k_hex1 @ q_hex1.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        probs_hex1 = self._categorical_masked(logits0=hex1_logits, mask=mask_hex1[0, act0])
        hex1 = torch.argmax(probs_hex1, dim=1)

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        z_hex1 = z_hexes[0, hex1, :]                                       # (B, d)
        q_hex2 = self.Wq_hex2(torch.cat([z_global, z_hex1], -1))                # (B, d)
        k_hex2 = self.Wk_hex2(z_hexes)                                          # (B, 165, d)
        hex2_logits = (k_hex2 @ q_hex2.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        probs_hex2 = self._categorical_masked(logits0=hex2_logits, mask=mask_hex2[0, act0, hex1])
        hex2 = torch.argmax(probs_hex2, dim=1)

        action = self.action_table[act0, hex1, hex2]
        return action, act0_logits, hex1_logits, hex2_logits

    def _categorical_masked(self, logits0, mask):
        logits1 = torch.where(mask, logits0, self.mask_value)
        logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs


class ExecuTorchDNAModel(nn.Module):
    def __init__(self, config, e_max, k_max):
        super().__init__()
        self.model_policy = ExecuTorchModel(config, e_max, k_max)
        self.model_value = ExecuTorchModel(config, e_max, k_max)
        self.register_buffer("version", torch.tensor(13, dtype=torch.long), persistent=False)
        self.register_buffer("e_max", torch.tensor(e_max, dtype=torch.long), persistent=False)
        self.register_buffer("k_max", torch.tensor(k_max, dtype=torch.long), persistent=False)

    def get_e_max(self):
        return self.e_max.clone()

    def get_k_max(self):
        return self.k_max.clone()

    def get_version(self):
        return self.version.clone()

    def get_value(self, obs, *edge_triplets):
        return self.model_value.get_value(obs, *edge_triplets)

    def predict(self, obs, *edge_triplets):
        return self.model_policy.predict(obs, *edge_triplets)


class ExportableGENConv(nn.Module):
    @classmethod
    def scatter_sum(cls, src, index, num_nodes):
        size = (num_nodes, src.size(1))
        index = index.unsqueeze(1).expand_as(src)
        return src.new_zeros(size).scatter_add_(0, index, src)

    @staticmethod
    def scatter_sum_via_nbr(src, nbr):
        N, K = nbr.shape
        H = src.size(1)
        idx = nbr.clamp_min(0).reshape(-1)                 # (N*K,)
        gathered = src.index_select(0, idx).reshape(N, K, H)
        valid = (nbr >= 0).unsqueeze(-1)
        zeros = src.new_zeros((1, 1, H))
        gathered = torch.where(valid, gathered, zeros)
        return gathered.sum(dim=1)                         # (N, H)

    @staticmethod
    def scatter_max(src, index, num_nodes):
        size = (num_nodes, src.size(1))
        index = index.unsqueeze(1).expand_as(src)
        return src.new_zeros(size).scatter_reduce_(0, index, src, reduce='amax', include_self=False)

    #
    # Export-friendly replacement for:
    #
    @staticmethod
    def scatter_max_via_nbr(src, nbr):
        N, K = nbr.shape
        H = src.size(1)
        idx = nbr.clamp_min(0).reshape(-1)
        gathered = src.index_select(0, idx).reshape(N, K, H)
        valid = (nbr >= 0).unsqueeze(-1)
        neg_inf = torch.finfo(src.dtype).min
        gathered = torch.where(valid, gathered, gathered.new_full((1, 1, H), neg_inf))
        return gathered.max(dim=1).values

    @classmethod
    def softmax_via_nbr(cls, src, index, nbr):
        src_max = cls.scatter_max_via_nbr(src, nbr)                    # (N, H)
        out = (src - src_max.index_select(0, index)).exp()             # (E_max, H)
        out_sum = cls.scatter_sum_via_nbr(out, nbr) + 1e-16            # (N, H)
        denom = out_sum.index_select(0, index)                         # (E_max, H)
        return out / denom

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()

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

    # NOTE: all inputs have a fixed length
    # (edge_index and edge_attr are PADDED with zeros up to E_MAX)
    def forward(self, x, edge_index, edge_attr, nbr):
        src = self.lin_src(x)
        dst = self.lin_dst(x)
        x_j = src.index_select(0, edge_index[0])
        out = self.message(x_j, edge_attr)
        out = self.aggregate(out, edge_index[1], nbr)
        out = out + dst
        return self.mlp(out)

    def message(self, x_j, edge_attr):
        edge_attr = self.lin_edge(edge_attr)
        msg = x_j + edge_attr
        return msg.relu() + self.eps

    def aggregate(self, x, index, nbr):
        alpha = self.__class__.softmax_via_nbr(x, index, nbr)
        res = self.__class__.scatter_sum_via_nbr(x * alpha, nbr)
        return res


class ExportableGNNBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim,
        link_types,
        num_nodes,
        e_max,
        k_max,
    ):
        super().__init__()

        layers = []
        self.num_nodes = num_nodes
        self.e_max = e_max
        self.k_max = k_max

        # First L-1 layers with activation
        for i in range(num_layers - 1):
            ch_in = in_channels if i == 0 else hidden_channels
            layers.append([ExportableGENConv(ch_in, hidden_channels, edge_dim) for _ in link_types])

        # Last layer without extra activation beyond the internal MLP
        layers.append([ExportableGENConv(hidden_channels, out_channels, edge_dim) for _ in link_types])

        self.layers = nn.ModuleList([nn.ModuleList(convs) for convs in layers])
        self.act = nn.LeakyReLU()

    def forward(self, x_hex, edge_inds, edge_attrs, nbrs):
        nconvs = len(self.layers[0])
        ei_shape = torch.Size([nconvs, 2, self.e_max])
        ea_shape = torch.Size([nconvs, self.e_max, 1])
        nb_shape = torch.Size([nconvs, self.num_nodes, self.k_max])

        assert edge_inds.shape == ei_shape, f"Expected edge_inds of shape {ei_shape}, got: {edge_inds.shape}"
        assert edge_attrs.shape == ea_shape, f"Expected edge_attrs of shape {ea_shape}, got: {edge_attrs.shape}"
        assert nbrs.shape == nb_shape, f"Expected nbrs of shape {nb_shape}, got: {nbrs.shape}"

        x = x_hex
        L = len(self.layers)
        for i, convs in enumerate(self.layers):
            y = None
            for r, conv in enumerate(convs):
                yr = conv(x, edge_inds[r], edge_attrs[r], nbrs[r])
                y = yr if y is None else y + yr
            x = y
            if i < L - 1:
                x = self.act(x)
        return x


class ModelWrapper(torch.nn.Module):
    def __init__(self, m, method_name: str):
        super().__init__()
        self.m = m
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.m, self.method_name)(*args, **kwargs)


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
    edge_index_lt2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    edge_attr_lt2 = torch.tensor([[0.5], [0.2]], dtype=torch.float)
    hd['hex', 'lt1', 'hex'].edge_index = edge_index_lt1
    hd['hex', 'lt1', 'hex'].edge_attr = edge_attr_lt1
    hd['hex', 'lt2', 'hex'].edge_index = edge_index_lt2
    hd['hex', 'lt2', 'hex'].edge_attr = edge_attr_lt2

    inputs = (hd["hex"].x, hd["hex", "lt1", "hex"].edge_index, hd["hex", "lt1", "hex"].edge_attr)

    # add NBR to inputs
    N = hd["hex"].x.size(0)
    E_max = 400  # max number of edges
    K_max = 300  # max number of incoming edges for 1 hex

    ei, ea = pad_edges(hd["hex", "lt1", "hex"].edge_index, hd["hex", "lt1", "hex"].edge_attr, e_max=E_max)
    nbr = build_nbr(hd["hex", "lt1", "hex"].edge_index[1], N, k_max=K_max)

    myinputs = (hd["hex"].x, ei, ea, nbr)

    res = gen(*inputs)
    myres = mygen(*myinputs)

    # import ipdb; ipdb.set_trace()  # noqa
    assert torch.equal(res, myres)
    print("test_gnn: OK")


def test_block():
    """ Tests GNNBlock vs ExportableGNNBlock. """

    import torch_geometric
    from .mppo_dna_gnn import GNNBlock

    E_max = 4       # max number of edges
    K_max = 5       # max number of incoming edges for 1 hex
    N = 3           # num_nodes

    hd = torch_geometric.data.HeteroData()
    hd['hex'].x = torch.randn(N, 5)
    edge_index_lt1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr_lt1 = torch.tensor([[1.0], [1.0]], dtype=torch.float)
    edge_index_lt2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    edge_attr_lt2 = torch.tensor([[0.5], [0.2]], dtype=torch.float)
    hd['hex', 'lt1', 'hex'].edge_index = edge_index_lt1
    hd['hex', 'lt1', 'hex'].edge_attr = edge_attr_lt1
    hd['hex', 'lt2', 'hex'].edge_index = edge_index_lt2
    hd['hex', 'lt2', 'hex'].edge_attr = edge_attr_lt2

    num_layers = 3
    in_channels = hd["hex"].x.size(1)
    hidden_channels = 6
    out_channels = 7

    block = GNNBlock(
        num_layers=num_layers,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        link_types=["lt1", "lt2"],
    ).eval()

    myblock = ExportableGNNBlock(
        num_layers=num_layers,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=1,
        link_types=["lt1", "lt2"],
        num_nodes=N,
        e_max=E_max,
        k_max=K_max,
    ).eval()

    mydict = {transform_key(k, ["lt1", "lt2"]): v for k, v in block.state_dict().items()}
    myblock.load_state_dict(mydict, strict=True)

    ei1, ea1 = pad_edges(hd["hex", "lt1", "hex"].edge_index, hd["hex", "lt1", "hex"].edge_attr, e_max=E_max)
    ei2, ea2 = pad_edges(hd["hex", "lt2", "hex"].edge_index, hd["hex", "lt2", "hex"].edge_attr, e_max=E_max)

    ei = torch.stack([ei1, ei2], dim=0)
    ea = torch.stack([ea1, ea2], dim=0)
    nbr = torch.stack([
        build_nbr(hd["hex", "lt1", "hex"].edge_index[1], N, k_max=K_max),
        build_nbr(hd["hex", "lt2", "hex"].edge_index[1], N, k_max=K_max),
    ], dim=0)

    res = block(hd)
    myres = myblock(hd["hex"].x, ei, ea, nbr)

    # import ipdb; ipdb.set_trace()  # noqa
    assert torch.equal(res, myres)
    print("test_block: OK")


def test_model(cfg_file, weights_file):
    """ Tests DNA Model vs ExecuTorchModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    E_MAX = 3300    # full REACH for 20 units (165 hexes each) = 3300 rels
    K_MAX = 20      # 20 units with REACH to the same hex seems like a good max

    emodel = ExecuTorchDNAModel(cfg["model"], E_MAX, K_MAX).eval()
    eweights = {transform_key(k, list(LINK_TYPES)): v for k, v in weights.items()}
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    # venv = create_venv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs=1, sync=False)
    venv = create_venv(dict(mapname="gym/A1.vmap", role="defender"), num_envs=1, sync=False)
    venv.reset()

    obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
    done = torch.tensor([False])
    links = [venv.call("obs")[0]["links"]]
    hdata = Batch.from_data_list(to_hdata_list(obs, done, links))

    einputs = build_einputs(hdata, E_MAX, K_MAX)
    for i, arg in enumerate(einputs):
        print("Arg %d shape: %s" % (i, arg.shape))

    actdata = model.get_actdata_eval(hdata, deterministic=True)
    action, act0_logits, hex1_logits, hex2_logits = emodel._predict_with_logits(*einputs)

    # import ipdb; ipdb.set_trace()  # noqa
    assert torch.equal(actdata.action, action)
    assert torch.equal(actdata.act0_logits, act0_logits)
    assert torch.equal(actdata.hex1_logits, hex1_logits)
    assert torch.equal(actdata.hex2_logits, hex2_logits)
    print("test_model: OK")


def test_xnn(cfg_file, weights_file):
    """ Tests DNAModel vs the XNN-lowered ExecuTorchDNAModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    E_MAX = 3300    # full REACH for 20 units (165 hexes each) = 3300 rels
    K_MAX = 20      # 20 units with REACH to the same hex seems like a good max

    emodel = ExecuTorchDNAModel(cfg["model"], E_MAX, K_MAX).eval()
    eweights = {transform_key(k, list(LINK_TYPES)): v for k, v in weights.items()}
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    # venv = create_venv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs=1, sync=False)
    venv = create_venv(dict(mapname="gym/A1.vmap", role="defender"), num_envs=1, sync=False)
    venv.reset()

    hdata = Batch.from_data_list(to_hdata_list(
        torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0),
        torch.tensor([False]),
        [venv.call("obs")[0]["links"]]
    ))

    einputs = build_einputs(hdata, E_MAX, K_MAX)
    for i, arg in enumerate(einputs):
        print("Arg %d shape: %s" % (i, arg.shape))

    m__predict_with_logits = ModelWrapper(emodel, "_predict_with_logits").eval().cpu()

    print("Exporting...")
    ep = {
        "_predict_with_logits": export(m__predict_with_logits, einputs, strict=True),
    }

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    exported__predict_with_logits = edge.exported_program("_predict_with_logits").module()

    print("Testing...")
    actdata = model.get_actdata_eval(hdata, deterministic=True)
    action, act0_logits, hex1_logits, hex2_logits = exported__predict_with_logits(*einputs)

    assert torch.equal(actdata.action, action)
    assert torch.equal(actdata.act0_logits, act0_logits)
    assert torch.equal(actdata.hex1_logits, hex1_logits)
    assert torch.equal(actdata.hex2_logits, hex2_logits)
    print("test_xnn: OK")


def test_xnn_quantized(cfg_file, weights_file):
    """ Tests DNAModel vs the XNN-lowered-and-quantized ExecuTorchDNAModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    E_MAX = 3300    # full REACH for 20 units (165 hexes each) = 3300 rels
    K_MAX = 20      # 20 units with REACH to the same hex seems like a good max

    emodel = ExecuTorchDNAModel(cfg["model"], E_MAX, K_MAX).eval()
    eweights = {transform_key(k, list(LINK_TYPES)): v for k, v in weights.items()}
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    # venv = create_venv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs=1, sync=False)
    venv = create_venv(dict(mapname="gym/A1.vmap", role="defender"), num_envs=1, sync=False)
    venv.reset()

    hdata = Batch.from_data_list(to_hdata_list(
        torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0),
        torch.tensor([False]),
        [venv.call("obs")[0]["links"]]
    ))

    einputs = build_einputs(hdata, E_MAX, K_MAX)
    for i, arg in enumerate(einputs):
        print("Arg %d shape: %s" % (i, arg.shape))

    m__predict_with_logits = ModelWrapper(emodel, "_predict_with_logits").eval().cpu()

    print("Quantizing...")
    # Quantizer
    # XXX: comparing outputs, is_per_channel=True gives *sligthtly* better results => use it
    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=True))

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
    action, act0_logits, hex1_logits, hex2_logits = exported__predict_with_logits(*build_einputs(hdata, E_MAX, K_MAX))

    err_act0_logits = (actdata.act0_logits - act0_logits) / actdata.act0_logits
    err_hex1_logits = (actdata.hex1_logits - hex1_logits) / actdata.hex1_logits
    err_hex2_logits = (actdata.hex2_logits - hex2_logits) / actdata.hex2_logits

    print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
    print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
    print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

    # import ipdb; ipdb.set_trace()  # noqa
    assert err_act0_logits.max() < 1e-4
    assert err_hex1_logits.max() < 1e-4
    assert err_hex2_logits.max() < 1e-4
    print("test_xnn_quantized: OK")


def test_load(cfg_file, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExecuTorchDNAModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    E_MAX = 3300    # full REACH for 20 units (165 hexes each) = 3300 rels
    K_MAX = 20      # 20 units with REACH to the same hex seems like a good max

    emodel = ExecuTorchDNAModel(cfg["model"], E_MAX, K_MAX).eval()
    eweights = {transform_key(k, list(LINK_TYPES)): v for k, v in weights.items()}
    emodel.load_state_dict(eweights, strict=True)
    emodel = emodel.model_policy

    # venv = create_venv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs=1, sync=False)
    venv = create_venv(dict(mapname="gym/A1.vmap", role="defender"), num_envs=1, sync=False)
    venv.reset()

    hdata = Batch.from_data_list(to_hdata_list(
        torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0),
        torch.tensor([False]),
        [venv.call("obs")[0]["links"]]
    ))

    einputs = build_einputs(hdata, E_MAX, K_MAX)
    for i, arg in enumerate(einputs):
        print("Arg %d shape: %s" % (i, arg.shape))

    m__predict_with_logits = ModelWrapper(emodel, "_predict_with_logits").eval().cpu()

    print("Exporting...")
    ep = {
        "_predict_with_logits": export(m__predict_with_logits, einputs, strict=True),
    }

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

    print("Exporting and loading...")
    rt = Runtime.get()
    loaded = rt.load_program(edge.to_executorch().buffer)
    loaded_predict_with_logits = loaded.load_method("_predict_with_logits")

    print("Testing...")
    actdata = model.get_actdata_eval(hdata, deterministic=True)
    action, act0_logits, hex1_logits, hex2_logits = loaded_predict_with_logits.execute(build_einputs(hdata, E_MAX, K_MAX))

    err_act0_logits = (actdata.act0_logits - act0_logits) / actdata.act0_logits
    err_hex1_logits = (actdata.hex1_logits - hex1_logits) / actdata.hex1_logits
    err_hex2_logits = (actdata.hex2_logits - hex2_logits) / actdata.hex2_logits

    print("Relative error: act0: mean=%.6f, max=%.6f" % (err_act0_logits.mean(), err_act0_logits.max()))
    print("Relative error: hex1: mean=%.6f, max=%.6f" % (err_hex1_logits.mean(), err_hex1_logits.max()))
    print("Relative error: hex2: mean=%.6f, max=%.6f" % (err_hex2_logits.mean(), err_hex2_logits.max()))

    # import ipdb; ipdb.set_trace()  # noqa
    assert err_act0_logits.max() < 1e-4
    assert err_hex1_logits.max() < 1e-4
    assert err_hex2_logits.max() < 1e-4

    print("test_load: OK")


def export_model(cfg_file, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExecuTorchDNAModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)

    E_MAX = 3300    # full REACH for 20 units (165 hexes each) = 3300 rels
    K_MAX = 20      # 20 units with REACH to the same hex seems like a good max

    emodel = ExecuTorchDNAModel(cfg["model"], E_MAX, K_MAX).eval()
    eweights = {transform_key(k, list(LINK_TYPES)): v for k, v in weights.items()}
    emodel.load_state_dict(eweights, strict=True)

    # venv = create_venv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs=1, sync=False)
    venv = create_venv(dict(mapname="gym/A1.vmap", role="defender"), num_envs=1, sync=False)
    venv.reset()

    hdata = Batch.from_data_list(to_hdata_list(
        torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0),
        torch.tensor([False]),
        [venv.call("obs")[0]["links"]]
    ))

    einputs = build_einputs(hdata, E_MAX, K_MAX)
    for i, arg in enumerate(einputs):
        print("Arg %d shape: %s" % (i, arg.shape))

    m_predict = ModelWrapper(emodel.model_policy, "predict").eval().cpu()
    # XXX: cut size in half by excluding model_value (not used anyway)
    # m_get_value = ModelWrapper(emodel.model_value, "get_value").eval().cpu()
    m_get_value = ModelWrapper(emodel.model_policy, "get_value").eval().cpu()
    m_get_ver = ModelWrapper(emodel, "get_version").eval().cpu()
    m_get_e_max = ModelWrapper(emodel, "get_e_max").eval().cpu()
    m_get_k_max = ModelWrapper(emodel, "get_k_max").eval().cpu()

    print("Exporting...")
    ep = {
        "predict": export(m_predict, einputs, strict=True),
        "get_value": export(m_get_value, einputs, strict=True),
        "get_version": export(m_get_ver, (), strict=True),
        "get_e_max": export(m_get_e_max, (), strict=True),
        "get_k_max": export(m_get_k_max, (), strict=True),
    }

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

    return edge.to_executorch()


def verify_export(cfg_file, weights_file, exported_model, num_steps=10):
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")
    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)

    print("Loading...")
    rt = Runtime.get()
    loaded_model = rt.load_program(exported_model.buffer)
    loaded_predict = loaded_model.load_method("predict")

    E_MAX = 3300    # full REACH for 20 units (165 hexes each) = 3300 rels
    K_MAX = 20      # 20 units with REACH to the same hex seems like a good max

    venv = create_venv(dict(mapname="gym/generated/4096/4x1024.vmap", role="defender"), num_envs=1, sync=False)
    venv.reset()

    print("Testing for %d steps..." % (num_steps))
    for n in range(num_steps):
        print(venv.render()[0])

        hdata = Batch.from_data_list(to_hdata_list(
            torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0),
            torch.tensor([False]),
            [venv.call("obs")[0]["links"]]
        ))

        einputs = build_einputs(hdata, E_MAX, K_MAX)

        actdata = model.model_policy.get_actdata_eval(hdata, deterministic=True)
        action = loaded_predict.execute(einputs)[0]
        print("(%d) TEST ACTION: %d <> %d" % (n, actdata.action, action.item()))

        # Not testing value (value model excluded)
        # value = model.get_value(hdata)[0]
        # myvalue = loaded_get_value.execute(einputs)
        # print("(%d) TEST VALUE: %.3f <> %.3f" % (n, value.item(), myvalue.item()))

        assert actdata.action == action.item()
        venv.step([actdata.action])

    print("verify_export: OK")


if __name__ == "__main__":
    with torch.inference_mode():
        filebase = "sfcjqcly-1757757007"
        model_cfg_path = f"{filebase}-config.json"
        model_weights_path = f"{filebase}-model-dna.pt"
        export_dst = f"/Users/simo/Projects/vcmi-play/Mods/MMAI/models/{filebase}.pte"

        # test_gnn()
        # test_block()
        # test_model(model_cfg_path, model_weights_path)
        # test_xnn(model_cfg_path, model_weights_path)
        # # test_xnn_quantized(model_cfg_path, model_weights_path)
        # test_load(model_cfg_path, model_weights_path)

        exported_model = export_model(model_cfg_path, model_weights_path)
        verify_export(model_cfg_path, model_weights_path, exported_model)

        print("Writing to %s" % export_dst)
        with open(export_dst, "wb") as f:
            exported_model.write_to_file(f)
