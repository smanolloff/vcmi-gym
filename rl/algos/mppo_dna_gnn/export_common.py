import torch
import torch.nn as nn

from .mppo_dna_gnn import (
    STATE_SIZE,
    STATE_SIZE_ONE_HEX,
    STATE_SIZE_HEXES,
    GLOBAL_ATTR_MAP,
    GLOBAL_ACT_MAP,
    HEX_ATTR_MAP,
    HEX_ACT_MAP,
    LINK_TYPES,
    Model,
    MainAction,
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

    ei_flat = torch.zeros((2, sum_e), dtype=torch.int32)
    ea_flat = torch.zeros((sum_e, 1), dtype=torch.float32)
    nbr_flat = torch.zeros((hdata.num_nodes, sum_k), dtype=torch.int32)

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
    nbr = torch.full((num_nodes, k_max), -1, dtype=torch.int32)
    fill = torch.zeros(num_nodes, dtype=torch.int32)
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


class ExportableModel(nn.Module):
    def __init__(self, config, side, all_sizes):
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

        self.register_buffer("version", torch.tensor([13], dtype=torch.int32), persistent=False)
        self.register_buffer("side", torch.tensor([side], dtype=torch.int32), persistent=False)

        # XXX: these should have persistent=False, but due to a bug they were
        # saved with the weights => load_state_dict() would fail unless they
        # are persistent here as well.
        self.register_buffer("amove_hexes", amove_hexes.unsqueeze(0).int())
        self.register_buffer("amove_hexes_valid", self.amove_hexes != -1)
        self.register_buffer("action_table", action_table.int())
        self.register_buffer("inverse_table", inverse_table.int())

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
        self.register_buffer("b_idx", torch.arange(1).view(1, 1, 1).expand_as(self.amove_hexes).int(), persistent=False)
        self.register_buffer("s_idx", torch.arange(165).view(1, 165, 1).expand_as(self.amove_hexes).int(), persistent=False)
        # self.register_buffer("mask_value", torch.tensor(torch.finfo(torch.float32).min), persistent=False)
        self.register_buffer("mask_value", torch.tensor(-((2 - 2**-23) * 2**127), dtype=torch.float32), persistent=False)
        self.register_buffer("hexactmask_inds", torch.as_tensor(torch.arange(12) + HEX_ATTR_MAP["ACTION_MASK"][1]).int(), persistent=False)

    def get_version(self):
        return self.version.clone()

    # Models are usually trained as either attackers or defenders
    # (0=attacker, 1=defender, 2=both)
    def get_side(self):
        return self.side.clone()

    def get_all_sizes(self):
        return self.model_value.encoder_hexes.all_sizes.clone()

    # edge_triplets is [edge_ind1, edge_attr1, edge_nbr1, edge_ind2, edge_attr2, edge_nbr2, ...]
    # (one triplet for each link type)
    @torch.jit.export
    def encode(self, obs, ei_flat, ea_flat, nbr_flat, size_id: int):
        hexes = obs[0, self.dim_other:].view(165, self.state_size_one_hex)
        other = obs[0, :self.dim_other]
        z_hexes = self.encoder_hexes(hexes, ei_flat, ea_flat, nbr_flat, size_id).unsqueeze(0)
        z_other = self.encoder_other(other).unsqueeze(0)
        # XXX: workaround for Vulkan partitioner bug https://github.com/pytorch/executorch/issues/12227?utm_source=chatgpt.com
        # z_global = z_other + z_hexes.mean(1)
        z_global = z_other + z_hexes.mean(dim=1, keepdim=True).squeeze(1)
        return z_hexes, z_global

    @torch.jit.export
    def get_value(self, obs, ei_flat, ea_flat, nbr_flat, size_id: int):
        obs = obs.unsqueeze(dim=0)
        _, z_global = self.encode(obs, ei_flat, ea_flat, nbr_flat, size_id)
        return self.critic(z_global)[0]

    @torch.jit.export
    def predict(self, obs, ei_flat, ea_flat, nbr_flat, size_id: int):
        return self._predict_with_logits(obs, ei_flat, ea_flat, nbr_flat, size_id)[0]

    @torch.jit.export
    def _predict_with_logits(self, obs, ei_flat, ea_flat, nbr_flat, size_id: int):
        obs = obs.unsqueeze(dim=0)
        z_hexes, z_global = self.encode(obs, ei_flat, ea_flat, nbr_flat, size_id)

        act0_logits = self.act0_head(z_global)

        # 1. MASK_HEX1 - ie. allowed hex#1 for each action
        mask_hex1 = torch.zeros((1, 4, 165), dtype=torch.int32)
        hexobs = obs[:, -self.state_size_hexes:].view([-1, 165, self.state_size_one_hex])

        # XXX: EXPLICIT casting to torch.bool is required to prevent a
        #      a nasty bug with the mask when the model is loaded in C++

        # 1.1 for 0=WAIT: nothing to do (all zeros)
        # 1.2 for 1=MOVE: Take MOVE bit from obs's action mask
        movemask = hexobs[:, :, self.hex_move_offset].to(torch.bool)
        mask_hex1[:, 1, :] = movemask.to(torch.int32)

        # 1.3 for 2=AMOVE: Take any(AMOVEX) bits from obs's action mask
        amovemask = hexobs[:, :, self.hexactmask_inds].to(torch.bool)
        mask_hex1[:, 2, :] = amovemask.any(dim=-1).to(torch.int32)

        # 1.4 for 3=SHOOT: Take SHOOT bit from obs's action mask
        shootmask = hexobs[:, :, self.hex_shoot_offset].to(torch.bool)
        mask_hex1[:, 3, :] = shootmask.to(torch.int32)

        # 2. MASK_HEX2 - ie. allowed hex2 for each (action, hex1) combo
        mask_hex2 = torch.zeros((1, 4, 165, 165), dtype=torch.int32)

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
        idx = self.amove_hexes.int()                                      # [B, S, K], long
        valid = valid & (idx >= 0)                                  # drop t == -1

        # 3) Replace -1 by 0 (any in-range index works) but make src zero there,
        #    so those updates are no-ops on a zero-initialized buffer.
        safe_idx = torch.where(valid, idx, idx.new_zeros(idx.shape))            # [B,S,K]

        # 4) Accumulate along T, then binarize

        # XXX: 4) Option A: use scatter_add
        # accum = torch.zeros((idx.size(0), idx.size(1), plane.size(-1)))  # [B,S,T]
        # accum = accum.scatter_add(-1, safe_idx.long(), valid.float())                         # [B,S,T]

        # XXX: 4) Option B: avoid scatter_add
        classes = torch.arange(165, dtype=torch.int32).view(1, 1, 1, 165)  # [1,1,1,T]
        onehot = (safe_idx.unsqueeze(-1) == classes)                      # [B,S,K,T], bool
        src = valid.to(torch.int32).unsqueeze(-1)                       # [B,S,K,1]

        # XXX: workaround for Vulkan partitioner bug https://github.com/pytorch/executorch/issues/12227?utm_source=chatgpt.com
        # accum = (onehot.to(torch.int32) * src).sum(dim=-2)                  # [B,S,T]
        accum = (onehot.to(torch.int32) * src).sum(dim=-2, keepdim=True).squeeze(-2)

        plane = accum.ne(0).to(torch.int32)

        # 5) Write back into the full mask tensor
        mask_hex2 = torch.zeros_like(self.mask_hex2)                             # [B,4,S,T]
        mask_hex2[:, 2] = plane

        # 2.4 for 3=SHOOT: nothing to do (all zeros)

        # 3. MASK_ACTION - ie. allowed main action mask
        mask_action = torch.zeros((1, 4), dtype=torch.int32)

        # 0=WAIT
        mask_action[:, 0] = obs[:, self.global_wait_offset].to(torch.bool).to(torch.int32)

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_action[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1).to(torch.int32)

        # Next, we sample:
        #
        # 1. Sample MAIN ACTION
        probs_act0 = self._categorical_masked(logits0=act0_logits, mask=mask_action.to(torch.bool))
        act0 = torch.argmax(probs_act0, dim=1)

        # 2. Sample HEX1 (with mask corresponding to the main action)
        act0_emb = self.emb_act0(act0)
        d = act0_emb.size(-1)
        q_hex1 = self.Wq_hex1(torch.cat([z_global, act0_emb], -1))              # (B, d)
        k_hex1 = self.Wk_hex1(z_hexes)                                          # (B, 165, d)
        hex1_logits = (k_hex1 @ q_hex1.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        # Use per-batch gather of the proper action channel; avoid boolean writes
        # mask_hex1_sel: [B,S]
        mask_hex1_sel = mask_hex1.gather(1, act0.view(-1, 1, 1).expand(-1, 1, 165)).squeeze(1).to(torch.bool)
        probs_hex1 = self._categorical_masked(logits0=hex1_logits, mask=mask_hex1_sel)
        hex1 = torch.argmax(probs_hex1, dim=1)

        # 3. Sample HEX2 (with mask corresponding to the main action + HEX1)
        z_hex1 = z_hexes[0, hex1, :]                                       # (B, d)
        q_hex2 = self.Wq_hex2(torch.cat([z_global, z_hex1], -1))                # (B, d)
        k_hex2 = self.Wk_hex2(z_hexes)                                          # (B, 165, d)
        hex2_logits = (k_hex2 @ q_hex2.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165)
        mask_hex2_amove = mask_hex2[:, 2]  # [B,S,T]
        mask_hex2_sel = mask_hex2_amove.gather(1, hex1.view(-1, 1, 1).expand(-1, 1, 165)).squeeze(1).to(torch.bool)
        probs_hex2 = self._categorical_masked(logits0=hex2_logits, mask=mask_hex2_sel)
        hex2 = torch.argmax(probs_hex2, dim=1)

        action = self.action_table[act0, hex1, hex2]
        return (
            action.clone(),
            act0_logits,
            act0,
            hex1_logits,
            hex1,
            hex2_logits,
            hex2
        )

    @torch.jit.export
    def _categorical_masked(self, logits0, mask):
        neg_inf = _neg_inf_like(logits0)
        logits1 = torch.where(mask, logits0, neg_inf)
        logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        probs = logits.softmax(dim=-1)
        return probs


# Used in exports only because loading weights from the checkpoint file
# is easier as it expects this model structure (DNA -> policy, value models)
class ExportableDNAModel(nn.Module):
    def __init__(self, config, side, all_sizes):
        super().__init__()
        self.model_policy = ExportableModel(config, side, all_sizes)
        self.model_value = ExportableModel(config, side, all_sizes)


MIN_FLOAT32 = torch.finfo(torch.float32).min
MIN_FLOAT16 = torch.finfo(torch.float16).min
MIN_BFLOAT16 = torch.finfo(torch.bfloat16).min


@torch.jit.export
def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16:
        v = -65504.0
    elif x.dtype == torch.float32:
        v = -((2 - 2**-23) * 2**127)
    else:
        # fallback: cast to fp32, use fp32 -inf, then back
        v = -((2 - 2**-23) * 2**127)
        return x.new_full((1,), v, dtype=torch.float32).to(x.dtype)
    return x.new_full((1,), v, dtype=x.dtype)


@torch.jit.export
def scatter_sum_via_nbr(src, nbr):
    N, K = nbr.shape
    H = src.size(1)
    # idx = nbr.clamp_min(0).reshape(-1)                 # (N*K,)
    idx64 = nbr.clamp_min(0).to(torch.int64).reshape(-1)       # only here
    gathered = src.index_select(0, idx64).reshape(N, K, H)
    valid = (nbr >= 0).unsqueeze(-1)
    zeros = src.new_zeros((1, 1, H))
    gathered = torch.where(valid, gathered, zeros)

    # XXX: workaround for Vulkan partitioner bug https://github.com/pytorch/executorch/issues/12227?utm_source=chatgpt.com
    # return gathered.sum(dim=1)                  # (N, H)
    return gathered.sum(dim=1, keepdim=True).squeeze(1)                  # (N, H)


@torch.jit.export
def scatter_max_via_nbr(src, nbr):
    N, K = nbr.shape
    H = src.size(1)
    # idx = nbr.clamp_min(0).reshape(-1)
    idx64 = nbr.clamp_min(0).to(torch.int64).reshape(-1)
    gathered = src.index_select(0, idx64).reshape(N, K, H)
    valid = (nbr >= 0).unsqueeze(-1)
    # XXX: torchscript does not allow to use torch.finfo(...)
    # neg_inf = -((2 - 2**-23) * 2**127)
    neg_inf = _neg_inf_like(src)[0]
    gathered = torch.where(valid, gathered, neg_inf)
    return gathered.max(dim=1).values


@torch.jit.export
def softmax_via_nbr(src, index, nbr):
    src_max = scatter_max_via_nbr(src, nbr)                    # (N, H)
    # out = (src - src_max.index_select(0, index)).exp()             # (E_max, H)
    out = (src - src_max.index_select(0, index.to(torch.int64))).exp()
    out_sum = scatter_sum_via_nbr(out, nbr) + 1e-16            # (N, H)
    # denom = out_sum.index_select(0, index)                         # (E_max, H)
    denom = out_sum.index_select(0, index.to(torch.int64))
    return out / denom


class ExportableGENConv(nn.Module):
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
    @torch.jit.export
    def forward(self, x, edge_index, edge_attr, nbr):
        src = self.lin_src(x)
        dst = self.lin_dst(x)
        x_j = src.index_select(0, edge_index[0])
        out = self.message(x_j, edge_attr)
        out = self.aggregate(out, edge_index[1], nbr)
        out = out + dst
        return self.mlp(out)

    @torch.jit.export
    def message(self, x_j, edge_attr):
        edge_attr = self.lin_edge(edge_attr)
        msg = x_j + edge_attr
        return msg.relu() + self.eps

    @torch.jit.export
    def aggregate(self, x, index, nbr):
        alpha = softmax_via_nbr(x, index, nbr)
        res = scatter_sum_via_nbr(x * alpha, nbr)
        return res


class ExportableGNNBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        in_channels,
        hidden_channels,
        out_channels,
        all_sizes,     # Tensor[S, L, 2] with ints; constant at model build time
        link_types,
        edge_dim: int = 1,
    ):
        super().__init__()

        # ---- layers ----
        layers = []
        for i in range(num_layers - 1):
            ch_in = in_channels if i == 0 else hidden_channels
            layers.append([ExportableGENConv(ch_in, hidden_channels, edge_dim) for _ in link_types])
        layers.append([ExportableGENConv(hidden_channels, out_channels, edge_dim) for _ in link_types])

        self.layers = nn.ModuleList([nn.ModuleList(convs) for convs in layers])
        self.act = nn.LeakyReLU()

        # Keep original sizes as a buffer (useful for verification/debug)
        self.register_buffer("all_sizes", all_sizes.int().clone(), persistent=False)

        # ---- precompute constant offsets on CPU as Python lists (Dynamo-friendly) ----
        # e_offsets[s][l] gives the starting column for link l; last entry is total sum_E[s]
        # k_offsets[s][l] gives the starting column for link l; last entry is total sum_K[s]
        sizes_list = all_sizes.detach().to("cpu").tolist()  # S x L x 2
        S = len(sizes_list)
        self.e_offsets = []
        self.k_offsets = []
        for s in range(S):
            e_off = [0]
            k_off = [0]
            for l in range(len(sizes_list[s])):
                e_off.append(e_off[-1] + int(sizes_list[s][l][0]))
                k_off.append(k_off[-1] + int(sizes_list[s][l][1]))
            self.e_offsets.append(e_off)
            self.k_offsets.append(k_off)

    def forward(
        self,
        x_hex: torch.Tensor,
        ei_flat: torch.Tensor,   # (2, sum_E[size])
        ea_flat: torch.Tensor,   # (sum_E[size], edge_dim)
        nbr_flat: torch.Tensor,  # (N, sum_K[size])
        size_idx: int
    ) -> torch.Tensor:
        x = x_hex
        num_layers = len(self.layers)
        e_off = self.e_offsets[size_idx]
        k_off = self.k_offsets[size_idx]

        for i, convs in enumerate(self.layers):
            y_init = False
            y = torch.empty(0, device=x.device)

            for l, conv in enumerate(convs):
                e0, e1 = e_off[l], e_off[l + 1]
                k0, k1 = k_off[l], k_off[l + 1]

                edge_inds = ei_flat[:, e0:e1]
                edge_attrs = ea_flat[e0:e1, :]
                nbrs = nbr_flat[:, k0:k1]

                out = conv(x, edge_inds, edge_attrs, nbrs)
                if not y_init:
                    y = out
                    y_init = True
                else:
                    y = y + out

            x = self.act(y) if i < num_layers - 1 else y

        return x


class HardcodedModelWrapper(torch.nn.Module):
    def __init__(self, m, side, all_sizes, m_value=None):
        super().__init__()
        self.m = m.eval().cpu()
        self.m_value = (m_value or m).eval().cpu()
        self.all_sizes = all_sizes
        assert len(ALL_MODEL_SIZES) == 5

    def forward(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.predict0(obs, ei_flat, ea_flat, nbr_flat)

    @torch.jit.export
    def get_version(self, dummy_input):
        # XXX: workaround for Vulkan partitioner bug https://github.com/pytorch/executorch/issues/12227?utm_source=chatgpt.com
        # return (dummy_input.sum() * 0) + self.m.version.clone()
        return dummy_input.sum(dim=0, keepdim=True).squeeze(0) + self.m.version.clone()

    # Models are usually trained as either attackers or defenders
    # (0=attacker, 1=defender, 2=both)
    @torch.jit.export
    def get_side(self, dummy_input):
        # return (dummy_input.sum() * 0) + self.m.side.clone()
        return dummy_input.sum(dim=0, keepdim=True).squeeze(0) + self.m.side.clone()

    @torch.jit.export
    def get_all_sizes(self, dummy_input):
        # return (dummy_input.sum() * 0) + self.m.encoder_hexes.all_sizes.clone()
        return dummy_input.sum(dim=0, keepdim=True).squeeze(0) + self.m.encoder_hexes.all_sizes.clone()

    # .get_valueN

    @torch.jit.export
    def get_value0(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m_value.get_value(obs, ei_flat, ea_flat, nbr_flat, 0)

    @torch.jit.export
    def get_value1(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m_value.get_value(obs, ei_flat, ea_flat, nbr_flat, 1)

    @torch.jit.export
    def get_value2(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m_value.get_value(obs, ei_flat, ea_flat, nbr_flat, 2)

    @torch.jit.export
    def get_value3(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m_value.get_value(obs, ei_flat, ea_flat, nbr_flat, 3)

    @torch.jit.export
    def get_value4(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m_value.get_value(obs, ei_flat, ea_flat, nbr_flat, 4)

    # .predictN

    @torch.jit.export
    def predict0(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m.predict(obs, ei_flat, ea_flat, nbr_flat, 0)

    @torch.jit.export
    def predict1(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m.predict(obs, ei_flat, ea_flat, nbr_flat, 1)

    @torch.jit.export
    def predict2(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m.predict(obs, ei_flat, ea_flat, nbr_flat, 2)

    @torch.jit.export
    def predict3(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m.predict(obs, ei_flat, ea_flat, nbr_flat, 3)

    @torch.jit.export
    def predict4(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m.predict(obs, ei_flat, ea_flat, nbr_flat, 4)

    # ._predict_with_logitsN

    @torch.jit.export
    def _predict_with_logits0(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m._predict_with_logits(obs, ei_flat, ea_flat, nbr_flat, 0)

    @torch.jit.export
    def _predict_with_logits1(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m._predict_with_logits(obs, ei_flat, ea_flat, nbr_flat, 1)

    @torch.jit.export
    def _predict_with_logits2(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m._predict_with_logits(obs, ei_flat, ea_flat, nbr_flat, 2)

    @torch.jit.export
    def _predict_with_logits3(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m._predict_with_logits(obs, ei_flat, ea_flat, nbr_flat, 3)

    @torch.jit.export
    def _predict_with_logits4(self, obs, ei_flat, ea_flat, nbr_flat):
        return self.m._predict_with_logits(obs, ei_flat, ea_flat, nbr_flat, 4)


class ModelWrapper(torch.nn.Module):
    def __init__(self, m, method_name, args_head=(), args_tail=()):
        super().__init__()
        self.m = m
        self.method_name = method_name
        self.args_head = args_head
        self.args_tail = args_tail

    def forward(self, *args):
        return getattr(self.m, self.method_name)(*self.args_head, *args, *self.args_tail)
