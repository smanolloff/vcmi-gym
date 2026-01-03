import torch
import torch.nn as nn

from torch_geometric.data import Batch
from .dual_vec_env import to_hdata_list

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


def build_hdata(venv):
    obs = torch.as_tensor(venv.call("obs")[0]["observation"]).unsqueeze(0)
    done = torch.tensor(venv.call("terminated"))
    links = [venv.call("obs")[0]["links"]]
    return Batch.from_data_list(to_hdata_list(obs, done, links))


def build_inputs(hdata):
    ei_flat, ea_flat, lengths = flatten_edges(hdata)
    return {
        "obs": hdata.obs[0],
        "ei_flat": ei_flat,
        "ea_flat": ea_flat,
        "lengths": lengths
    }


def flatten_edges(hdata):
    assert len(hdata.node_types) == 1, hdata.node_types
    assert all(n == 1 for n in hdata.num_edge_features.values()), hdata.num_edge_features.values()

    ei_flat = torch.zeros((2, 0), dtype=torch.int32)
    ea_flat = torch.zeros((0, 1), dtype=torch.float32)
    lengths = []

    for i, edge_type in enumerate(hdata.edge_types):
        reldata = hdata[edge_type]
        lengths.append(reldata.edge_index.shape[1])
        ei_flat = torch.cat([ei_flat, reldata.edge_index], dim=1)
        ea_flat = torch.cat([ea_flat, reldata.edge_attr], dim=0)

    sum_e = sum(hdata[et].edge_index.shape[1] for et in hdata.edge_types)
    assert ei_flat.shape[1] == sum_e
    assert ea_flat.shape[0] == sum_e

    return ei_flat, ea_flat, torch.tensor(lengths)


def onnx_fwd(edge, inputs):
    return [
        torch.as_tensor(x)
        for x in edge.run(None, {k: v.numpy() for k, v in inputs.items()})
    ]


def build_action_probs(
    act0_probs: torch.Tensor,      # (B, 4)
    hex1_probs: torch.Tensor,      # (B, 4, 165)   = P(hex1 | act0)
    hex2_probs: torch.Tensor,      # (B, 165, 165) = P(hex2 | hex1)
    mask: torch.Tensor,            # (B, num_actions) boolean/0-1
    action_table: torch.Tensor,    # (4, 165, 165) int64 action ids
) -> torch.Tensor:
    B = act0_probs.size(0)
    H = hex1_probs.size(-1)
    device = act0_probs.device
    dtype = act0_probs.dtype

    out = torch.zeros_like(mask, dtype=torch.float32)

    # 0) WAIT: all triplets map to 0 when act0=0
    out[:, 1] += act0_probs[:, 0]

    # 1) MOVE: act0=1, depends on hex1 only (hex2 irrelevant)
    move_ids = action_table[1, :, 0].to(device)                 # (H,)
    move_p = act0_probs[:, 1, None] * hex1_probs[:, 1, :]       # (B, H)
    out.scatter_add_(1, move_ids[None, :].expand(B, H), move_p)

    # 2) AMOVE: act0=2, depends on (hex1, hex2); invalid pairs map to 0
    amove_ids = action_table[2].reshape(-1).to(device)          # (H*H,)
    amove_p = (
        act0_probs[:, 2, None, None]
        * hex1_probs[:, 2, :, None]
        * hex2_probs
    )                                                           # (B, H, H)
    out.scatter_add_(1, amove_ids[None, :].expand(B, H*H), amove_p.reshape(B, H*H))

    # 3) SHOOT: act0=3, depends on hex1 only (hex2 irrelevant)
    shoot_ids = action_table[3, :, 0].to(device)                # (H,)
    shoot_p = act0_probs[:, 3, None] * hex1_probs[:, 3, :]      # (B, H)
    out.scatter_add_(1, shoot_ids[None, :].expand(B, H), shoot_p)

    # Optional: apply mask over final action ids and renormalize
    out = out * mask.to(dtype)
    z = out.sum(dim=-1, keepdim=True)
    # fallback to action 0 if everything masked out
    out = torch.where(z > 0, out / z, torch.nn.functional.one_hot(
        torch.zeros(B, device=device, dtype=torch.long), out.shape[1]
    ).to(dtype))

    return out


class ExportableModel(nn.Module):
    def __init__(self, config, side):
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
        self.register_buffer("mask_act0", torch.zeros([1, self.n_main_actions], dtype=torch.bool), persistent=False)
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

    @torch.jit.export
    def encode(self, obs, ei_flat, ea_flat, lengths):
        hexes = obs[0, self.dim_other:].view(165, self.state_size_one_hex)
        other = obs[0, :self.dim_other]
        z_hexes = self.encoder_hexes(hexes, ei_flat, ea_flat, lengths).unsqueeze(0)
        z_other = self.encoder_other(other).unsqueeze(0)
        # XXX: workaround for Vulkan partitioner bug https://github.com/pytorch/executorch/issues/12227?utm_source=chatgpt.com
        # z_global = z_other + z_hexes.mean(1)
        z_global = z_other + z_hexes.mean(dim=1, keepdim=True).squeeze(1)
        return z_hexes, z_global

    @torch.jit.export
    def get_value(self, obs, ei_flat, ea_flat, lengths):
        obs = obs.unsqueeze(dim=0)
        _, z_global = self.encode(obs, ei_flat, ea_flat, lengths)
        return self.critic(z_global)[0]

    @torch.jit.export
    def forward(self, obs, ei_flat, ea_flat, lengths):
        B = 1
        obs = obs.unsqueeze(dim=0)
        z_hexes, z_global = self.encode(obs, ei_flat, ea_flat, lengths)

        act0_logits = self.act0_head(z_global)

        # 1. MASK_HEX1 - ie. allowed hex#1 for each action
        mask_hex1 = torch.zeros((B, 4, 165), dtype=torch.int32)
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
        # (it's only for AMOVE action, => shape is (B, 165, 165) instead of (B, 4, 165, 165))

        # 2.1 for 0=WAIT: nothing to do (all zeros)
        # 2.2 for 1=MOVE: nothing to do (all zeros)
        # 2.3 for 2=AMOVE: For each SRC hex, create a DST hex mask of allowed hexes
        valid = amovemask & self.amove_hexes_valid

        # Ensure invalid entries are excluded from updates
        idx = self.amove_hexes.int()                                      # [B, S, K], long
        valid = valid & (idx >= 0)                                  # drop t == -1

        # Replace -1 by 0 (any in-range index works) but make src zero there,
        # so those updates are no-ops on a zero-initialized buffer.
        safe_idx = torch.where(valid, idx, idx.new_zeros(idx.shape))            # [B,S,K]

        # Accumulate along T, then binarize
        accum = torch.zeros((B, 165, 165))  # [B,S,T]
        accum = accum.scatter_add(-1, safe_idx.long(), valid.float())                         # [B,S,T]
        mask_hex2 = accum.ne(0).to(torch.int32)

        # 2.4 for 3=SHOOT: nothing to do (all zeros)

        # 3. MASK_ACTION - ie. allowed main action mask
        mask_act0 = torch.zeros((1, 4), dtype=torch.int32)

        # 0=WAIT
        mask_act0[:, 0] = obs[:, self.global_wait_offset].to(torch.bool).to(torch.int32)

        # 1=MOVE, 2=AMOVE, 3=SHOOT: if at least 1 target hex
        mask_act0[:, 1:] = mask_hex1[:, 1:, :].any(dim=-1).to(torch.int32)

        # MAIN ACTION
        act0_probs = categorical_masked(logits0=act0_logits, mask=mask_act0.to(torch.bool))

        B = z_hexes.size(0)

        # HEX1
        act0_emb = self.emb_act0(torch.arange(4))                               # (4, d)
        d = act0_emb.size(-1)
        q_hex1 = self.Wq_hex1(
            torch.cat([
                z_global.unsqueeze(1).expand(B, 4, d),                          # (B, 4, d)
                act0_emb.unsqueeze(0).expand(B, 4, d)                           # (B, 4, d)
            ], dim=-1)                                                          # (B, 4, 2d)
        )                                                                       # (B, 4, d)
        k_hex1 = self.Wk_hex1(z_hexes).unsqueeze(1).expand(B, 4, 165, d)        # (B, 4, 165, d)
        hex1_logits = (k_hex1 @ q_hex1.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 4, 165)
        hex1_probs = categorical_masked(logits0=hex1_logits, mask=mask_hex1.to(torch.bool))

        # HEX2
        q_hex2 = self.Wq_hex2(
            torch.cat([
                z_global.unsqueeze(1).expand(B, 165, d),                        # (B, 165, d)
                z_hexes                                                         # (B, 165, d)
            ], dim=-1)                                                          # (B, 165, 2d)
        )                                                                       # (B, 165, d)
        k_hex2 = self.Wk_hex2(z_hexes).unsqueeze(1).expand(B, 165, 165, d)      # (B, 165, 165, d)
        hex2_logits = (k_hex2 @ q_hex2.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)  # (B, 165, 165)
        hex2_probs = categorical_masked(logits0=hex2_logits, mask=mask_hex2.to(torch.bool))

        mask_hexes = torch.cat([
            amovemask,
            movemask.unsqueeze(-1),
            shootmask.unsqueeze(-1)
        ], -1).flatten(1)                                                       # (B, 2310)

        mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool),                                # RETREAT (never allowed)
            obs[:, self.global_wait_offset].unsqueeze(-1).to(torch.bool),       # WAIT
            mask_hexes
        ], -1)                                                                  # (B, 2312)

        # Final probs (B, N_ACTIONS)
        probs = build_action_probs(act0_probs, hex1_probs, hex2_probs, mask, self.action_table)

        return probs, mask


# Used in exports only because loading weights from the checkpoint file
# is easier as it expects this model structure (DNA -> policy, value models)
class ExportableDNAModel(nn.Module):
    def __init__(self, config, side):
        super().__init__()
        self.model_policy = ExportableModel(config, side)
        self.model_value = ExportableModel(config, side)


MIN_FLOAT32 = torch.finfo(torch.float32).min
MIN_FLOAT16 = torch.finfo(torch.float16).min
MIN_BFLOAT16 = torch.finfo(torch.bfloat16).min


@torch.jit.export
def neg_inf_like(x: torch.Tensor) -> torch.Tensor:
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
def categorical_masked(logits0, mask):
    neg_inf = neg_inf_like(logits0)
    logits1 = torch.where(mask, logits0, neg_inf)
    logits = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
    probs = logits.softmax(dim=-1)
    return probs


def edge_softmax_per_dst(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int, eps: float = 1e-16):
    """
    scores: (E, H)
    dst:    (E,) int64 destination node index per edge
    returns alpha: (E, H), softmax over edges grouped by dst (per feature dim)
    """
    E, H = scores.shape
    dst = dst.to(torch.int64)

    # Per-node max: (N, H)
    # IMPORTANT for export: include_self=True is commonly required/assumed by PyTorch ONNX exporter for scatter_reduce.
    # (Many people hit an export error when include_self=False.)
    neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
    max_per_node = neg_inf.expand(num_nodes, H).clone()
    max_per_node.scatter_reduce_(
        0,
        dst[:, None].expand(E, H),
        scores,
        reduce="amax",
        include_self=True,
    )

    exp_scores = (scores - max_per_node.index_select(0, dst)).exp()

    # Per-node sum: (N, H)
    sum_per_node = torch.zeros((num_nodes, H), device=scores.device, dtype=scores.dtype)
    sum_per_node.scatter_add_(0, dst[:, None].expand(E, H), exp_scores)

    return exp_scores / (sum_per_node.index_select(0, dst) + eps)


def scatter_sum_per_dst(values: torch.Tensor, dst: torch.Tensor, num_nodes: int):
    """
    values: (E, H)
    dst:    (E,) int64 destination node index per edge
    returns: (N, H)
    """
    E, H = values.shape
    dst = dst.to(torch.int64)

    out = torch.zeros((num_nodes, H), device=values.device, dtype=values.dtype)
    out.scatter_add_(0, dst[:, None].expand(E, H), values)
    return out


class ExportableGENConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        self.eps = 1e-7

        if in_channels != out_channels:
            self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_src = nn.Identity()
            self.lin_dst = nn.Identity()

        self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2, bias=False),
            nn.BatchNorm1d(out_channels * 2, affine=True),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(out_channels * 2, out_channels, bias=False),
        )

    def message(self, x_j, edge_attr):
        edge_attr = self.lin_edge(edge_attr)
        msg = x_j + edge_attr
        return msg.relu() + self.eps

    def forward(self, x, edge_index, edge_attr):
        """
        x:         (N, Fin)
        edge_index:(2, E)  (src, dst)
        edge_attr: (E, edge_dim)
        """
        N = x.size(0)

        src_feat = self.lin_src(x)   # (N, Fout)
        dst_feat = self.lin_dst(x)   # (N, Fout)

        src = edge_index[0].to(torch.int64)  # (E,)
        dst = edge_index[1].to(torch.int64)  # (E,)

        x_j = src_feat.index_select(0, src)      # (E, Fout)
        msg = self.message(x_j, edge_attr)       # (E, Fout)

        alpha = edge_softmax_per_dst(msg, dst, num_nodes=N)         # (E, Fout)
        agg = scatter_sum_per_dst(msg * alpha, dst, num_nodes=N)    # (N, Fout)

        out = agg + dst_feat
        return self.mlp(out)


class ExportableGNNBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        in_channels,
        hidden_channels,
        out_channels,
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

    def forward(
        self,
        x_hex: torch.Tensor,
        ei_flat: torch.Tensor,   # (2, sum_E)
        ea_flat: torch.Tensor,   # (sum_E, edge_dim)
        lengths: torch.Tensor    # (L) where L = len(self.layers)
    ) -> torch.Tensor:
        x = x_hex
        num_layers = len(self.layers)

        for i, convs in enumerate(self.layers):
            y_init = False
            y = torch.empty(0, device=x.device)

            cur_e = 0

            for l, conv in enumerate(convs):
                length = lengths[l]
                e0, e1 = cur_e, cur_e + length
                cur_e = e1

                edge_inds = ei_flat[:, e0:e1]
                edge_attrs = ea_flat[e0:e1, :]

                out = conv(x, edge_inds, edge_attrs)

                if not y_init:
                    y = out
                    y_init = True
                else:
                    y = y + out

            x = self.act(y) if i < num_layers - 1 else y

        return x
