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
)

from .dual_vec_env import (
    to_hdata_list,
    DualVecEnv
)


def transform_key(key, node_type, link_types):
    import re
    pattern = re.compile(rf"(.+)\.module_(\d+)\.convs\.<{node_type}___(\w+)___{node_type}>\.(.+)")
    match = pattern.match(key)
    if not match:
        return key
    assert match, f"No match: {key}"
    pre, module_id, link_type, rest = match.groups()
    assert int(module_id) % 2 == 0
    assert link_type in link_types, f"{link_type} not in {link_types}"
    return "%s.%d.%d.%s" % (pre, int(module_id) / 2, link_types.index(link_type), rest)


# def build_einputs(hdata, e_max, k_max):
#     eis = []
#     eas = []
#     nbrs = []
#     for lt in LINK_TYPES:
#         reldata = hdata["hex", lt, "hex"]
#         ei, ea = pad_edges(reldata.edge_index, reldata.edge_attr, e_max)
#         nbr = build_nbr(reldata.edge_index[1], 165, k_max)
#         eis.append(ei)
#         eas.append(ea)
#         nbrs.append(nbr)
#     return (
#         hdata.obs[0],
#         torch.stack(eis, dim=0),
#         torch.stack(eas, dim=0),
#         torch.stack(nbrs, dim=0),
#     )

#
# Stats (10k steps) - obtained via statscounter.py:
#
#        Num edges (E)   avg   max   p99   p90   p75   p50   p25
# -----------------------------------------------------------------
#             ADJACENT   888   888   888   888   888   888   888
#                REACH   355   988   820   614   478   329   209
#           RANGED_MOD   408   2403  1285  646   483   322   162
#          ACTS_BEFORE   51    268   203   118   75    35    15
#        MELEE_DMG_REL   43    198   160   103   60    31    14
#        RETAL_DMG_REL   27    165   113   67    38    18    8
#       RANGED_DMG_REL   12    133   60    29    18    9     4
#
#    Inbound edges (K)   avg   max   p99   p90   p75   p50   p25
# -----------------------------------------------------------------
#             ADJACENT   5.4   6     6     6     6     6     6
#                REACH   2.2   13    10    8     6     4     3
#           RANGED_MOD   2.5   15    8     4     3     2     1
#          ACTS_BEFORE   0.3   23    19    15    12    8     5
#        MELEE_DMG_REL   0.3   10    9     8     7     5     3
#        RETAL_DMG_REL   0.2   10    9     8     6     5     3
#       RANGED_DMG_REL   0.1   8     6     3     2     2     1


# Sizes are (E, K) tuples
ALL_MODEL_SIZES = torch.tensor([
    [  # 0 (S): p50
        [888, 6],       # ADJACENT
        [330, 4],       # REACH
        [330, 2],       # RANGED_MOD
        [36, 8],        # ACTS_BEFORE
        [32, 5],        # MELEE_DMG_REL
        [20, 5],        # RETAL_DMG_REL
        [10, 2],        # RANGED_DMG_REL
    ],
    [  # 1 (M): p90
        [888, 6],       # ADJACENT
        [620, 8],       # REACH
        [650, 4],       # RANGED_MOD
        [120, 15],      # ACTS_BEFORE
        [100, 8],       # MELEE_DMG_REL
        [70, 8],        # RETAL_DMG_REL
        [30, 3],        # RANGED_DMG_REL
    ],
    [  # 2 (L): p99
        [888, 6],       # ADJACENT
        [820, 10],      # REACH
        [1300, 8],      # RANGED_MOD
        [200, 19],      # ACTS_BEFORE
        [160, 9],       # MELEE_DMG_REL
        [110, 9],       # RETAL_DMG_REL
        [60, 6],        # RANGED_DMG_REL
    ],
    [  # 3 (XL): max+
        [888, 6],       # ADJACENT
        [1000, 14],     # REACH
        [2500, 16],     # RANGED_MOD
        [300, 24],      # ACTS_BEFORE
        [250, 11],      # MELEE_DMG_REL
        [200, 11],      # RETAL_DMG_REL
        [150, 9],       # RANGED_DMG_REL
    ],
    [  # 4 (XXL): fallback (bug?)
        [888, 6],       # ADJACENT        # fixed
        [3000, 30],     # REACH           # 20 arch devils, no obstacles: 20*146 = 2920
        [4000, 30],     # RANGED_MOD      # 24 shooters: 24*165 = 3960
        [1000, 50],     # ACTS_BEFORE     # 30 stacks: 30*29 = 870
        [500, 20],      # MELEE_DMG_REL   # 15 wide stacks per side: 15*30 = 450
        [500, 20],      # RETAL_DMG_REL   # same as MELEE_DMG_REL
        [250, 15],      # RANGED_DMG_REL  # 15 shooters per side: 15*15 = 225
    ]
])

# Inputs:
#   hdata: HeteroData
#   model_sizes: tensor with shape (NUM_LT, 2), e.g. an entry from ALL_MODEL_SIZES
#
# Returns:
#   [ei_flat, ea_flat, nbr_flat]
#
# Shapes:
# ei_flat:  (2, sum_E)
# ea_flat:  (sum_E, edge_dim)
# nbr_flat: (num_nodes, sum_K)
#
# The exported model will contain a method for these model_sizes which knows
# exactly how to decompose the flattened tensors.
#
def build_edge_inputs(hdata, model_sizes):
    assert len(hdata.node_types) == 1, hdata.node_types
    assert model_sizes.ndim == 2
    assert model_sizes.shape[0] == len(hdata.edge_types)
    assert all(n == 1 for n in hdata.num_edge_features.values()), hdata.num_edge_features.values()

    sum_e = model_sizes[:, 0].sum()
    sum_k = model_sizes[:, 1].sum()

    ei_flat = torch.zeros((2, sum_e), dtype=torch.long)
    ea_flat = torch.zeros((sum_e, 1), dtype=torch.float32)
    nbr_flat = torch.zeros((hdata.num_nodes, sum_k), dtype=torch.long)

    e0 = 0
    k0 = 0

    for i, edge_type in enumerate(hdata.edge_types):
        e = model_sizes[i, 0]
        k = model_sizes[i, 1]
        e1 = e0 + e
        k1 = k0 + k

        reldata = hdata[edge_type]
        ei, ea = pad_edges(reldata.edge_index, reldata.edge_attr, 1, e)
        nbr = build_nbr(reldata.edge_index[1], hdata.num_nodes, k)

        ei_flat[:, e0:e1] = ei
        ea_flat[e0:e1, :] = ea
        nbr_flat[:, k0:k1] = nbr

        e0 = e1
        k0 = k1

    assert e0 == sum_e
    assert k0 == sum_k

    return ei_flat, ea_flat, nbr_flat


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


# NOTE: OK to use 0 for padding (padded positions are not present in NBR)
def pad_edges(edge_index, edge_attr, edge_dim, e_max):
    # edge_index: (2, E), long; edge_attr: (E, edge_dim), float
    E = edge_index.size(1)
    if E > e_max:
        raise ValueError(f"E={E} exceeds e_max={e_max}")
    pad = e_max - E
    if pad:
        edge_index = torch.cat([edge_index, edge_index.new_zeros(2, pad)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr.new_zeros(pad, edge_dim)], dim=0)

    return edge_index, edge_attr


class ExecuTorchModel(nn.Module):
    def __init__(self, config, all_sizes):
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

        # XXX: these should have persistent=False, but due to a bug they were
        # saved with the weights => load_state_dict() would fail unless they
        # are persistent here as well.
        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0))
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1)
        self.register_buffer("action_table", action_table)
        self.register_buffer("inverse_table", inverse_table)

        self.encoder_hexes = ExportableGNNBlock(
            num_layers=config["gnn_num_layers"],
            in_channels=STATE_SIZE_ONE_HEX,
            hidden_channels=config["gnn_hidden_channels"],
            out_channels=config["gnn_out_channels"],
            all_sizes=all_sizes,
            link_types=list(LINK_TYPES),
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
    def encode(self, obs, *gnn_block_args):
        hexes = obs[0, self.dim_other:].view(165, self.state_size_one_hex)
        other = obs[0, :self.dim_other]
        z_hexes = self.encoder_hexes(hexes, *gnn_block_args).unsqueeze(0)
        z_other = self.encoder_other(other).unsqueeze(0)
        z_global = z_other + z_hexes.mean(1)
        return z_hexes, z_global

    def get_value(self, obs, *gnn_block_args):
        obs = obs.unsqueeze(dim=0)
        _, z_global = self.encode(obs, *gnn_block_args)
        return self.critic(z_global), z_global

    def predict(self, *args):
        return self._predict_with_logits(*args)[0]

    def _predict_with_logits(self, obs, *gnn_block_args):
        obs = obs.unsqueeze(dim=0)
        z_hexes, z_global = self.encode(obs, *gnn_block_args)

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

        # 4) Accumulate along T, then binarize
        accum = torch.zeros((idx.size(0), idx.size(1), plane.size(-1)), dtype=torch.long)  # [B,S,T]
        accum = accum.scatter_add(-1, safe_idx, valid.to(torch.long))                         # [B,S,T]
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
        act0 = torch.argmax(probs_act0, dim=1).to(torch.long).contiguous()
        act0 = act0.clamp_(0, self.emb_act0.num_embeddings - 1)  # Safety clamp (prevents rare out-of-range writes from numeric noise)

        # 2. Sample HEX1 (with mask corresponding to the main action)
        act0_emb = self.emb_act0(act0)
        d = act0_emb.size(-1)
        q_hex1 = self.Wq_hex1(torch.cat([z_global, act0_emb], -1))              # (B, d)
        k_hex1 = self.Wk_hex1(z_hexes)                                          # (B, 165, d)
        hex1_logits = (k_hex1 @ q_hex1.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        m_hex1 = mask_hex1[0, act0.long().contiguous()]            # [B,S]
        probs_hex1 = self._categorical_masked(logits0=hex1_logits, mask=m_hex1)
        # XXX: ExecuTorch/XNNPACK on Windows can materialize argmax as 32-bit writes
        hex1 = torch.argmax(probs_hex1, dim=1).to(torch.long).contiguous()
        hex1 = hex1.clamp_(0, z_hexes.size(1) - 1)

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        z_hex1 = z_hexes[0, hex1, :]                                       # (B, d)
        q_hex2 = self.Wq_hex2(torch.cat([z_global, z_hex1], -1))                # (B, d)
        k_hex2 = self.Wk_hex2(z_hexes)                                          # (B, 165, d)
        hex2_logits = (k_hex2 @ q_hex2.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        m_hex2 = mask_hex2[0, act0.long().contiguous(), hex1.long().contiguous()]   # [B,T]
        probs_hex2 = self._categorical_masked(logits0=hex2_logits, mask=m_hex2)
        hex2 = torch.argmax(probs_hex2, dim=1).to(torch.long).contiguous()
        hex2 = hex2.clamp_(0, z_hexes.size(1) - 1)

        action = self.action_table[act0, hex1, hex2]
        return (
            action.long().clone(),
            act0_logits,
            act0,
            hex1_logits,
            hex1,
            hex2_logits,
            hex2
        )

    def _categorical_masked(self, logits0, mask):
        logits1 = torch.where(mask, logits0, self.mask_value)
        logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs


class ExecuTorchDNAModel(nn.Module):
    def __init__(self, config, side, all_sizes):
        super().__init__()
        self.model_policy = ExecuTorchModel(config, all_sizes)
        self.model_value = ExecuTorchModel(config, all_sizes)

        self.register_buffer("version", torch.tensor(13, dtype=torch.long), persistent=False)
        self.register_buffer("side", torch.tensor(side, dtype=torch.long), persistent=False)

    def get_version(self):
        return self.version.clone()

    # Models are usually trained as either attackers or defenders
    # (0=attacker, 1=defender, 2=both)
    def get_side(self):
        return self.side.clone()

    def get_all_sizes(self):
        return self.model_value.encoder_hexes.all_sizes.clone()

    def get_value(self, *args):
        return self.model_value.get_value(*args)

    def predict(self, *args):
        return self.model_policy.predict(*args)


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
        all_sizes,
        link_types,
        edge_dim=1,
    ):
        super().__init__()

        layers = []

        # First L-1 layers with activation
        for i in range(num_layers - 1):
            ch_in = in_channels if i == 0 else hidden_channels
            layers.append([ExportableGENConv(ch_in, hidden_channels, edge_dim) for _ in link_types])

        # Last layer without extra activation beyond the internal MLP
        layers.append([ExportableGENConv(hidden_channels, out_channels, edge_dim) for _ in link_types])

        # Each layer is a ModuleList of N convs
        self.layers = nn.ModuleList([nn.ModuleList(convs) for convs in layers])
        self.act = nn.LeakyReLU()

        self.register_buffer("all_sizes", all_sizes.long().clone(), persistent=False)

        # Precompute per-size, per-layer [e0,e1,k0,k1] slice tuples as Python ints.
        # S=num_sizes       (e.g. 4 if S, M, L, XL)
        # L=num_link_types  (e.g. for VCMI obs, this is 7: REACH, ADJACENT, ...)
        S, L, _ = all_sizes.shape

        assert L == len(link_types)

        # list of length S; each item is list of L tuples
        self.all_sizes_segments = []

        for s in range(S):
            e_sizes = all_sizes[s, :, 0].tolist()  # length L
            k_sizes = all_sizes[s, :, 1].tolist()  # length L
            e_off, k_off = [0], [0]
            for l in range(L):
                e_off.append(e_off[-1] + int(e_sizes[l]))
                k_off.append(k_off[-1] + int(k_sizes[l]))
            segs_s = [(e_off[l], e_off[l+1], k_off[l], k_off[l+1]) for l in range(L)]
            self.all_sizes_segments.append(segs_s)

    # ei_flat:  (2, sum_E[size_idx])
    # ea_flat:  (sum_E[size_idx], edge_dim)
    # nbr_flat: (num_nodes, sum_K[size_idx])
    # size_idx: (1) - a single-integer tensor
    def forward(self, x_hex, ei_flat, ea_flat, nbr_flat, size_idx):
        x = x_hex
        L = len(self.layers)
        segments = self.all_sizes_segments[size_idx]  # list[(e0,e1,k0,k1)] of length L

        for i, convs in enumerate(self.layers):
            y = None
            for conv, (e0, e1, k0, k1) in zip(convs, segments):
                edge_inds = ei_flat[:, e0:e1]     # (2, E_i)
                edge_attrs = ea_flat[e0:e1, :]     # (E_i, D_e)
                nbrs = nbr_flat[:, k0:k1]    # (N, K_i)
                yr = conv(x, edge_inds, edge_attrs, nbrs)
                y = yr if y is None else y + yr

            x = self.act(y) if i < L - 1 else y

        return x


class ModelWrapper(torch.nn.Module):
    def __init__(self, m, method_name, args_head=(), args_tail=()):
        super().__init__()
        self.m = m
        self.method_name = method_name
        self.args_head = args_head
        self.args_tail = args_tail

    def forward(self, *args):
        return getattr(self.m, self.method_name)(*self.args_head, *args, *self.args_tail)


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

    print("=== XNN transform ===")
    print("Exporting...")
    ep = {
        "forward0": export(myblock, myinputs0, strict=True),
        "forward1": export(myblock, myinputs1, strict=True),
    }

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    exported_forward0 = edge.exported_program("forward0").module()
    exported_forward1 = edge.exported_program("forward1").module()

    print("Testing...")
    expmyres0 = exported_forward0(*myinputs0)
    expmyres1 = exported_forward1(*myinputs1)
    assert torch.equal(res, expmyres0)
    assert torch.equal(res, expmyres1)


def test_model(cfg_file, weights_file):
    """ Tests DNA Model vs ExecuTorchModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExecuTorchDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

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

        action, act0_logits, hex1_logits, hex2_logits = emodel._predict_with_logits(*einputs)

        # import ipdb; ipdb.set_trace()  # noqa
        assert torch.equal(actdata.action, action)
        assert torch.equal(actdata.act0_logits, act0_logits)
        assert torch.equal(actdata.hex1_logits, hex1_logits)
        assert torch.equal(actdata.hex2_logits, hex2_logits)
        print("(size=%d) test_model: OK" % i)

    print("=== XNN transform ===")

    # XXX: it should work OK if args_tail == einputs' model size
    #       it returns incorrect result if args_tail < einputs' model size
    #       it fails with RuntimeError: index_select(): ... otherwise

    print("Exporting...")
    ep = {}
    for i, edge_inputs in enumerate(all_edge_inputs):
        einputs = (obs[0], *edge_inputs)
        w = ModelWrapper(emodel, "_predict_with_logits", args_tail=(i,)).eval().cpu()
        ep[f"_predict_with_logits{i}"] = export(w, einputs, strict=True)

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

    for i, edge_inputs in enumerate(all_edge_inputs):
        print("XNN Testing size %d..." % i)
        einputs = (obs[0], *edge_inputs)

        for i1, arg in enumerate(einputs):
            print(f"Arg {i1}: ", end="")
            if isinstance(arg, torch.Tensor):
                print(f"tensor: {arg.shape}")
            else:
                print(f"{arg.__class__.__name__}: {arg}")

        program = edge.exported_program(f"_predict_with_logits{i}").module()
        action, act0_logits, hex1_logits, hex2_logits = program(*einputs)

        assert torch.equal(actdata.action, action)
        assert torch.equal(actdata.act0_logits, act0_logits)
        assert torch.equal(actdata.hex1_logits, hex1_logits)
        assert torch.equal(actdata.hex2_logits, hex2_logits)
        print("XNN (size=%d) test_model: OK" % i)


def test_quantized(cfg_file, weights_file):
    """ Tests DNAModel vs the XNN-lowered-and-quantized ExecuTorchDNAModel. """
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExecuTorchDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

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

    # einputs = build_einputs(hdata, E_MAX, K_MAX)
    # for i, arg in enumerate(einputs):
    #     print("Arg %d shape: %s" % (i, arg.shape))

    # XXX: test only with size "S" (quantizing is very slow)
    einputs = (obs[0], *build_edge_inputs(hdata, ALL_MODEL_SIZES[0]), 0)
    m__predict_with_logits = ModelWrapper(emodel, "_predict_with_logits").eval().cpu()

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
    action, act0_logits, hex1_logits, hex2_logits = exported__predict_with_logits(*einputs)

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
    print("test_quantized: OK")


def test_load(cfg_file, weights_file):
    """ Tests DNAModel vs the loaded XNN-lowered ExecuTorchDNAModel. """

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)
    model = model.model_policy

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExecuTorchDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

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
    einputs = (obs[0], *build_edge_inputs(hdata, ALL_MODEL_SIZES[0]), 0)

    for i1, arg in enumerate(einputs):
        print(f"Arg {i1}: ", end="")
        if isinstance(arg, torch.Tensor):
            print(f"tensor: {arg.shape}")
        else:
            print(f"{arg.__class__.__name__}: {arg}")

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
    action, act0_logits, hex1_logits, hex2_logits, action_table = loaded_predict_with_logits.execute(einputs)

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

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]
    emodel = ExecuTorchDNAModel(cfg["model"], eside, ALL_MODEL_SIZES).eval()

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

    programs = {}

    print("NOTE: Using get_value from policy model to reduce export size")

    for i, edge_inputs in enumerate(all_edge_inputs):
        einputs = (obs[0], *edge_inputs)

        print(f"Exporting predict{i}")
        w = ModelWrapper(emodel.model_policy, "predict", args_tail=(i,)).eval().cpu()
        programs[f"predict{i}"] = export(w, einputs, strict=True)

        # w = ModelWrapper(emodel.model_value, "get_value", args_tail=(i,)).eval().cpu()
        w = ModelWrapper(emodel.model_policy, "get_value", args_tail=(i,)).eval().cpu()

        programs[f"get_value{i}"] = export(w, einputs, strict=True)

    einputs3 = (obs[0], *(all_edge_inputs[3]))
    w = ModelWrapper(emodel.model_policy, "_predict_with_logits", args_tail=(3,)).eval().cpu()
    programs["_predict_with_logits3"] = export(w, einputs3, strict=True)

    w = ModelWrapper(emodel, "get_version").eval().cpu()
    programs["get_version"] = export(w, (), strict=True)

    w = ModelWrapper(emodel, "get_side").eval().cpu()
    programs["get_side"] = export(w, (), strict=True)

    w = ModelWrapper(emodel, "get_all_sizes").eval().cpu()
    programs["get_all_sizes"] = export(w, (), strict=True)

    print("Exported programs:\n  %s" % "\n  ".join(list(programs.keys())))

    print("Lowering to XNN...")
    edge = to_edge_transform_and_lower(programs, partitioner=[XnnpackPartitioner()])

    return edge.to_executorch()


def verify_export(cfg_file, weights_file, loaded_model, num_steps=10):
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    weights = torch.load(weights_file, weights_only=True, map_location="cpu")
    model = DNAModel(cfg["model"], torch.device("cpu")).eval()
    model.load_state_dict(weights, strict=True)

    venv = DualVecEnv(dict(mapname="gym/A1.vmap", role="defender"), num_envs_stupidai=1)
    venv.reset()

    loaded_methods = {name: loaded_model.load_method(name) for name in loaded_model.method_names}

    print("Testing metadata methods...")

    # 3 metadata methods + 2*sizes methods (predict & get_value)
    if "_predict_with_logits3" in loaded_methods:
        assert len(loaded_methods) == 1 + 3 + 2*len(ALL_MODEL_SIZES), len(loaded_methods)
    else:
        assert len(loaded_methods) == 3 + 2*len(ALL_MODEL_SIZES), len(loaded_methods)

    eside = dict(attacker=0, defender=1)[cfg["train"]["env"]["kwargs"]["role"]]

    assert loaded_methods["get_version"].execute(())[0].item() == 13
    assert loaded_methods["get_side"].execute(())[0].item() == eside
    assert torch.equal(loaded_methods["get_all_sizes"].execute(())[0], ALL_MODEL_SIZES)

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
            action = loaded_methods[f"predict{i}"].execute(einputs)[0]
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

    import ipdb; ipdb.set_trace()  # noqa
    print("Model role: %s" % cfg["train"]["env"]["kwargs"]["role"])
    print("verify_export: OK")


if __name__ == "__main__":
    MODEL_PREFIX = "nkjrmrsq-202509252116"

    with torch.inference_mode():
        model_cfg_path = f"{MODEL_PREFIX}-config.json"
        model_weights_path = f"{MODEL_PREFIX}-model-dna.pt"
        export_dst = f"/Users/simo/Projects/vcmi-play/Mods/MMAI/models/{MODEL_PREFIX}-logits-fix1.pte"

        # test_gnn()
        # test_block()
        # test_model(model_cfg_path, model_weights_path)
        # # test_quantized(model_cfg_path, model_weights_path)
        # test_load(model_cfg_path, model_weights_path)

        exported_model = export_model(model_cfg_path, model_weights_path)

        rt = Runtime.get()
        loaded_model = rt.load_program(exported_model.buffer)
        # loaded_model = rt.load_program("/Users/simo/Projects/vcmi-play/Mods/MMAI/models/tukbajrv-202509171940-sizes.pte")

        verify_export(model_cfg_path, model_weights_path, loaded_model)

        print("Writing to %s" % export_dst)
        with open(export_dst, "wb") as f:
            exported_model.write_to_file(f)
